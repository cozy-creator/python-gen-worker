from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import struct
import tarfile
import tempfile
import zlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, BinaryIO, cast

from .declaration import DeclarationError, GraphClassDeclaration, _update_literal_digest
from .host_isa import HostISAError, _validate_host_facts
from .identity import (
    ARTIFACT_KIND,
    GRAPH_CLASS_BLOCK,
    from_artifact_metadata,
    is_compiled_graph_key,
)
from .introspection import (
    PackageIntrospectionError,
    _package_entry_names,
    code_only_violations,
    declared_constants,
)

COMPILED_GRAPH_FORMAT = 1
_COMPILED_GRAPH_FORMAT_AXIS = "compiled_graph_format"
METADATA_NAME = "metadata.json"
PACKAGE_NAME = "model.pt2"
LITERALS_NAME = "constants.safetensors"
_REQUIRED_MEMBERS = frozenset((METADATA_NAME, PACKAGE_NAME))
_MEMBERS = _REQUIRED_MEMBERS | {LITERALS_NAME}
_MAX_METADATA_BYTES = 8 << 20
_MAX_SAFETENSORS_HEADER_BYTES = 8 << 20
# A code-only AOTI package must not carry model weights. Sixteen GiB is four
# times ZIP32's 4 GiB boundary: enough for unusually large generated host/CUDA
# code while still making remote decompression a finite admission operation.
_MAX_PACKAGE_BYTES = 16 << 30
# Literal constants are the only tensor bytes this envelope permits. Four GiB
# is the ZIP32 boundary; larger payloads are model-state in practice and belong
# in the separately bound repository snapshot, not one compiled graph.
_MAX_LITERALS_BYTES = 4 << 30
# The sum, rather than a second convention, is the exact v1 envelope budget.
_MAX_MEMBER_BYTES = _MAX_METADATA_BYTES + _MAX_PACKAGE_BYTES + _MAX_LITERALS_BYTES
# Deflate's documented worst-case expansion is well below one percent. That
# margin plus one canonical tar record bounds the fetched/compressed input too,
# including concatenated empty gzip members that add CPU without member bytes.
_MAX_ARCHIVE_BYTES = _MAX_MEMBER_BYTES + _MAX_MEMBER_BYTES // 100 + tarfile.RECORDSIZE
# Three member headers/padding plus the USTAR terminator fit well within two
# records. This is the gzip decoder's output ceiling before tar semantics run.
_MAX_TAR_BYTES = _MAX_MEMBER_BYTES + 2 * tarfile.RECORDSIZE
_CONSTANT_FIELDS = frozenset(("fqn", "source", "dtype", "shape"))
_CONSTANT_SOURCES = frozenset(("state_dict", "computed", "literal"))
_TOP_LEVEL_FIELDS = frozenset(
    (
        _COMPILED_GRAPH_FORMAT_AXIS,
        "kind",
        "compiled_graph_key",
        GRAPH_CLASS_BLOCK,
        "sm",
        "toolchain",
        "host_isa",
        "package_constants_in_so",
        "constant_folding_fenced",
    )
)

_SAFETENSORS_DTYPES: dict[str, tuple[str, int]] = {
    "BOOL": ("bool", 1),
    "U8": ("uint8", 1),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "U16": ("uint16", 2),
    "I32": ("int32", 4),
    "U32": ("uint32", 4),
    "I64": ("int64", 8),
    "U64": ("uint64", 8),
    "F16": ("float16", 2),
    "BF16": ("bfloat16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "C64": ("complex64", 8),
    "C128": ("complex128", 16),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E4M3FNUZ": ("float8_e4m3fnuz", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "F8_E5M2FNUZ": ("float8_e5m2fnuz", 1),
    "F8_E8M0": ("float8_e8m0fnu", 1),
}
_GRAPH_CLASS_FIELDS = frozenset(
    (
        "name",
        "target",
        "class_hash",
        "graph",
        "graph_witness",
        "range_digest",
        "fork",
        "class_dims",
        "strict",
        "lora_bucket",
        "literal_values",
        "literal_payload_values",
        "placement",
        "constants",
    )
)


class ArtifactError(ValueError):
    """A compiled-graph envelope is malformed or internally inconsistent."""


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validated_constants(graph_class: Mapping[str, Any]) -> list[dict[str, Any]]:
    constants = graph_class.get("constants")
    if not isinstance(constants, list):
        raise ArtifactError("graph_class constants must be an array")
    checked: list[dict[str, Any]] = []
    fqns: set[str] = set()
    for index, row in enumerate(constants):
        if not isinstance(row, Mapping):
            raise ArtifactError("graph_class constant must be an object")
        if set(row) != _CONSTANT_FIELDS:
            raise ArtifactError(
                "graph_class constant "
                f"{index} fields must be exactly {sorted(_CONSTANT_FIELDS)!r}"
            )
        fqn = row.get("fqn")
        source = row.get("source")
        dtype = row.get("dtype")
        shape = row.get("shape")
        if not isinstance(fqn, str) or not fqn or fqn != fqn.strip():
            raise ArtifactError(f"graph_class constant {index} fqn must be a non-empty string")
        if fqn in fqns:
            raise ArtifactError(f"graph_class constants contain duplicate fqn {fqn!r}")
        if source not in _CONSTANT_SOURCES:
            raise ArtifactError(
                "graph_class constant "
                f"{index} source must be one of {sorted(_CONSTANT_SOURCES)!r}"
            )
        if not isinstance(dtype, str) or not dtype or dtype != dtype.strip():
            raise ArtifactError(
                f"graph_class constant {index} dtype must be a non-empty string"
            )
        if not isinstance(shape, list) or not all(
            type(dimension) is int and dimension >= 0 for dimension in shape
        ):
            raise ArtifactError(
                "graph_class constant "
                f"{index} shape must be an array of non-negative integers"
            )
        fqns.add(fqn)
        checked.append({"fqn": fqn, "source": source, "dtype": dtype, "shape": list(shape)})
    return checked


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ArtifactError(f"JSON object declares duplicate key {name!r}")
        result[name] = value
    return result


def _bounded_chunks(source: BinaryIO, offset: int, size: int) -> Iterator[bytes]:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(1 << 20, remaining))
        if not chunk:
            raise ArtifactError("safetensors tensor data is truncated")
        remaining -= len(chunk)
        yield chunk


def _verify_literals(path: Path | None, metadata: Mapping[str, Any]) -> None:
    graph_class = cast(Mapping[str, Any], metadata[GRAPH_CLASS_BLOCK])
    literal_rows = {
        cast(str, row["fqn"]): row
        for row in cast(list[dict[str, Any]], graph_class["constants"])
        if row["source"] == "literal"
    }
    if not literal_rows:
        if path is not None:
            raise ArtifactError("artifact carries a literal payload but declares no literals")
        return
    if path is None:
        raise ArtifactError("graph_class declares literal constants but carries no literal payload")

    try:
        total_size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ArtifactError("safetensors payload has no complete header length")
            header_size = struct.unpack("<Q", prefix)[0]
            if (
                header_size == 0
                or header_size % 8 != 0
                or header_size > _MAX_SAFETENSORS_HEADER_BYTES
                or header_size > total_size - 8
            ):
                raise ArtifactError("safetensors payload has an invalid header length")
            encoded = source.read(header_size)
            if len(encoded) != header_size:
                raise ArtifactError("safetensors payload has a truncated header")
            try:
                decoded = json.loads(encoded, object_pairs_hook=_unique_object)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactError(f"safetensors payload has invalid JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ArtifactError("safetensors header must be an object")
            metadata_row = decoded.pop("__metadata__", None)
            if metadata_row is not None and (
                not isinstance(metadata_row, dict)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in metadata_row.items()
                )
            ):
                raise ArtifactError("safetensors __metadata__ must contain only strings")
            if set(decoded) != set(literal_rows):
                missing = sorted(set(literal_rows) - set(decoded))
                extra = sorted(set(decoded) - set(literal_rows))
                raise ArtifactError(
                    "safetensors names do not match declared literals "
                    f"(missing {missing}, extra {extra})"
                )

            data_start = 8 + header_size
            data_size = total_size - data_start
            tensors: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
            ranges: list[tuple[int, int, str]] = []
            for name, raw in decoded.items():
                if not isinstance(raw, dict) or set(raw) != {"dtype", "shape", "data_offsets"}:
                    raise ArtifactError(f"safetensors tensor {name!r} has invalid fields")
                raw_dtype = raw["dtype"]
                raw_shape = raw["shape"]
                raw_offsets = raw["data_offsets"]
                dtype_info = (
                    _SAFETENSORS_DTYPES.get(raw_dtype) if isinstance(raw_dtype, str) else None
                )
                if dtype_info is None:
                    raise ArtifactError(f"safetensors tensor {name!r} has unsupported dtype")
                if not isinstance(raw_shape, list) or not all(
                    type(dimension) is int and dimension >= 0 for dimension in raw_shape
                ):
                    raise ArtifactError(f"safetensors tensor {name!r} has invalid shape")
                if (
                    not isinstance(raw_offsets, list)
                    or len(raw_offsets) != 2
                    or not all(type(offset) is int and offset >= 0 for offset in raw_offsets)
                ):
                    raise ArtifactError(f"safetensors tensor {name!r} has invalid data offsets")
                start, end = raw_offsets
                shape = tuple(raw_shape)
                elements = 1
                for dimension in shape:
                    elements *= dimension
                if end < start or end - start != elements * dtype_info[1] or end > data_size:
                    raise ArtifactError(f"safetensors tensor {name!r} has an invalid byte range")
                declared = literal_rows[name]
                if declared["dtype"] != dtype_info[0]:
                    raise ArtifactError(
                        f"safetensors tensor {name!r} dtype does not match declaration"
                    )
                if tuple(declared["shape"]) != shape:
                    raise ArtifactError(
                        f"safetensors tensor {name!r} shape does not match declaration"
                    )
                tensors[name] = (dtype_info[0], shape, start, end)
                ranges.append((start, end, name))

            expected_offset = 0
            for start, end, name in sorted(ranges):
                if start != expected_offset:
                    raise ArtifactError(
                        f"safetensors tensor {name!r} has overlapping or non-contiguous data"
                    )
                expected_offset = end
            if expected_offset != data_size:
                raise ArtifactError("safetensors payload has unclaimed trailing data")

            digest = hashlib.sha256()
            for name in sorted(tensors):
                canonical_dtype, shape, start, end = tensors[name]
                _update_literal_digest(
                    digest,
                    name=name,
                    dtype=f"torch.{canonical_dtype}",
                    shape=shape,
                    chunks=_bounded_chunks(source, data_start + start, end - start),
                )
            if digest.hexdigest()[:32] != graph_class["literal_payload_values"]:
                raise ArtifactError(
                    "safetensors values do not match graph_class literal_payload_values"
                )
    except OSError:
        raise


def validate_metadata(
    raw: Mapping[str, Any], *, has_literals: bool | None = None
) -> dict[str, Any]:
    """Validate, detach, and return the sole v1 metadata representation."""

    try:
        decoded: object = json.loads(json.dumps(dict(raw), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"metadata is not finite JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ArtifactError("metadata root must be an object")
    metadata = cast(dict[str, Any], decoded)
    if set(metadata) != _TOP_LEVEL_FIELDS:
        raise ArtifactError(f"metadata fields must be exactly {sorted(_TOP_LEVEL_FIELDS)!r}")
    if (
        type(metadata.get(_COMPILED_GRAPH_FORMAT_AXIS)) is not int
        or metadata.get(_COMPILED_GRAPH_FORMAT_AXIS) != COMPILED_GRAPH_FORMAT
    ):
        raise ArtifactError(
            f"{_COMPILED_GRAPH_FORMAT_AXIS} must be {COMPILED_GRAPH_FORMAT}"
        )
    if metadata.get("kind") != ARTIFACT_KIND:
        raise ArtifactError(f"kind must be {ARTIFACT_KIND!r}")
    if metadata.get("package_constants_in_so") is not False:
        raise ArtifactError("package_constants_in_so must be false")
    if metadata.get("constant_folding_fenced") is not True:
        raise ArtifactError("constant_folding_fenced must be true")
    sm = metadata.get("sm")
    if not isinstance(sm, str) or not sm or sm != sm.strip():
        raise ArtifactError("sm must be a non-empty string")
    toolchain = metadata.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or not toolchain
        or any(
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(value, str)
            or not value
            or value != value.strip()
            for name, value in toolchain.items()
        )
    ):
        raise ArtifactError("toolchain must contain non-empty string facts")
    host_isa = metadata.get("host_isa")
    if not isinstance(host_isa, dict):
        raise ArtifactError("host_isa must be an object")
    try:
        _validate_host_facts(cast(dict[str, str], host_isa))
    except HostISAError as exc:
        raise ArtifactError(f"host_isa facts are invalid: {exc}") from exc
    graph_class = metadata.get(GRAPH_CLASS_BLOCK)
    if not isinstance(graph_class, dict):
        raise ArtifactError("metadata must contain one graph_class object")
    if set(graph_class) != _GRAPH_CLASS_FIELDS:
        raise ArtifactError(
            f"graph_class fields must be exactly {sorted(_GRAPH_CLASS_FIELDS)!r}"
        )
    for field in ("name", "target", "class_hash", "graph_witness", "range_digest"):
        if not isinstance(graph_class.get(field), str) or not graph_class[field].strip():
            raise ArtifactError(f"graph_class {field!r} must be a non-empty string")
    graph = graph_class.get("graph")
    if not isinstance(graph, dict) or not graph:
        raise ArtifactError("graph_class graph must be a non-empty object")
    literal_values = graph_class.get("literal_values")
    literal_payload_values = graph_class.get("literal_payload_values")
    placement = graph_class.get("placement")
    fork = graph_class.get("fork")
    class_dims = graph_class.get("class_dims")
    strict = graph_class.get("strict")
    lora_bucket = graph_class.get("lora_bucket")
    if not isinstance(literal_values, str):
        raise ArtifactError("graph_class literal_values must be a string")
    if not isinstance(literal_payload_values, str) or (
        literal_payload_values
        and (
            len(literal_payload_values) != 32
            or any(character not in "0123456789abcdef" for character in literal_payload_values)
        )
    ):
        raise ArtifactError(
            "graph_class literal_payload_values must be empty or 32 lowercase "
            "hexadecimal characters"
        )
    if not isinstance(placement, list) or not all(
        isinstance(device, str) and device and device == device.strip() for device in placement
    ):
        raise ArtifactError("graph_class placement must be an array of non-empty strings")
    if not isinstance(fork, list) or not all(
        isinstance(row, list) and len(row) == 2 and isinstance(row[0], str) for row in fork
    ):
        raise ArtifactError("graph_class fork must be an array of name/value pairs")
    if not isinstance(class_dims, list) or not all(
        isinstance(row, list)
        and len(row) == 2
        and isinstance(row[0], str)
        and type(row[1]) is int
        for row in class_dims
    ):
        raise ArtifactError("graph_class class_dims must be an array of name/integer pairs")
    if type(strict) is not bool:
        raise ArtifactError("graph_class strict must be a boolean")
    if type(lora_bucket) is not int:
        raise ArtifactError("graph_class lora_bucket must be an integer")
    try:
        declaration = GraphClassDeclaration(
            graph_class=graph_class["name"],
            target=graph_class["target"],
            graph=graph,
            graph_witness=graph_class["graph_witness"],
            range_digest=graph_class["range_digest"],
            fork=tuple((row[0], row[1]) for row in fork),
            class_dims=tuple((row[0], row[1]) for row in class_dims),
            strict=strict,
            lora_bucket=lora_bucket,
            literal_values=literal_values,
            placement=tuple(placement),
        )
    except DeclarationError as exc:
        raise ArtifactError(f"graph_class declaration is invalid: {exc}") from exc
    if (
        graph_class["name"] != declaration.graph_class
        or graph_class["target"] != declaration.target
    ):
        raise ArtifactError("graph_class name and target must be canonical")
    if placement != list(declaration.placement):
        raise ArtifactError("graph_class placement must be sorted and unique")
    if fork != [[name, value] for name, value in declaration.fork]:
        raise ArtifactError("graph_class fork must be canonical")
    if class_dims != [[name, value] for name, value in declaration.class_dims]:
        raise ArtifactError("graph_class class_dims must be canonical")
    if graph_class["class_hash"] != declaration.class_hash:
        raise ArtifactError("graph_class class_hash does not restate its declaration facts")
    expected = from_artifact_metadata(metadata).value
    stamped = metadata.get("compiled_graph_key")
    if not isinstance(stamped, str) or not is_compiled_graph_key(stamped):
        raise ArtifactError("compiled_graph_key is missing or malformed")
    if stamped != expected:
        raise ArtifactError(
            f"compiled_graph_key {stamped!r} does not restate artifact facts ({expected})"
        )
    constants = _validated_constants(graph_class)
    graph_class["constants"] = constants
    needs_literals = any(row["source"] == "literal" for row in constants)
    if needs_literals and not literal_values:
        raise ArtifactError("graph_class literal constants require a literal_values digest")
    if needs_literals and not literal_payload_values:
        raise ArtifactError(
            "graph_class literal constants require a literal_payload_values digest"
        )
    if not needs_literals and literal_payload_values:
        raise ArtifactError(
            "graph_class literal_payload_values requires declared literal constants"
        )
    if has_literals is not None and has_literals != needs_literals:
        if needs_literals:
            raise ArtifactError(
                "graph_class declares literal constants but carries no literal payload"
            )
        raise ArtifactError("artifact carries a literal payload but declares no literals")
    return metadata


def build_metadata(
    *,
    graph_class: Mapping[str, Any],
    sm: str,
    toolchain: Mapping[str, str],
    host_isa: Mapping[str, str],
) -> dict[str, Any]:
    """Build metadata from recorded facts and stamp the key derived from them."""

    metadata: dict[str, Any] = {
        _COMPILED_GRAPH_FORMAT_AXIS: COMPILED_GRAPH_FORMAT,
        "kind": ARTIFACT_KIND,
        GRAPH_CLASS_BLOCK: dict(graph_class),
        "sm": str(sm),
        "toolchain": dict(toolchain),
        "host_isa": dict(host_isa),
        "package_constants_in_so": False,
        "constant_folding_fenced": True,
    }
    metadata["compiled_graph_key"] = from_artifact_metadata(metadata).value
    return validate_metadata(metadata)


def _metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    checked = validate_metadata(metadata)
    return json.dumps(
        checked, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def _member_limit(name: str) -> int:
    limits = {
        METADATA_NAME: _MAX_METADATA_BYTES,
        PACKAGE_NAME: _MAX_PACKAGE_BYTES,
        LITERALS_NAME: _MAX_LITERALS_BYTES,
    }
    return limits[name]


def _validate_member_sizes(sizes: Mapping[str, int]) -> None:
    total = 0
    for name, size in sizes.items():
        if type(size) is not int or size < 0:
            raise ArtifactError(f"artifact member {name!r} has an invalid size")
        limit = _member_limit(name)
        if size > limit:
            raise ArtifactError(f"{name} exceeds the {limit}-byte v1 budget")
        total += size
        if total > _MAX_MEMBER_BYTES:
            raise ArtifactError(
                f"artifact total uncompressed bytes exceed the {_MAX_MEMBER_BYTES}-byte v1 budget"
            )


def pack_artifact(
    package: str | Path,
    output: str | Path,
    metadata: Mapping[str, Any],
    *,
    literals: str | Path | None = None,
) -> Path:
    """Create the deterministic v1 tar+gzip envelope."""

    package_path = Path(package)
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    literal_path = Path(literals) if literals is not None else None
    if literal_path is not None and not literal_path.is_file():
        raise FileNotFoundError(literal_path)
    checked = validate_metadata(metadata, has_literals=literal_path is not None)
    encoded = _metadata_bytes(checked)
    sizes = {
        METADATA_NAME: len(encoded),
        PACKAGE_NAME: package_path.stat().st_size,
    }
    if literal_path is not None:
        sizes[LITERALS_NAME] = literal_path.stat().st_size
    _validate_member_sizes(sizes)
    _verify_literals(literal_path, checked)
    verify_package(package_path, checked)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive:
                    archive.addfile(_tarinfo(METADATA_NAME, len(encoded)), io.BytesIO(encoded))
                    for name, source in (
                        (PACKAGE_NAME, package_path),
                        (LITERALS_NAME, literal_path),
                    ):
                        if source is None:
                            continue
                        with source.open("rb") as handle:
                            archive.addfile(_tarinfo(name, source.stat().st_size), handle)
            raw.flush()
            os.fsync(raw.fileno())
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.verify-", dir=target.parent
        ) as raw_verified:
            unpack_artifact(temporary_name, Path(raw_verified) / "artifact")
        os.replace(temporary_name, target)
        _fsync_dir(target.parent)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def verify_package(package: str | Path, metadata: Mapping[str, Any]) -> None:
    """Cross-check metadata against one code-only AOTInductor package."""

    package_path = Path(package)
    checked = validate_metadata(metadata)
    graph_class = cast(dict[str, Any], checked[GRAPH_CLASS_BLOCK])
    name = cast(str, graph_class["name"])
    try:
        names = _package_entry_names(package_path)
        if names != (name,):
            raise ArtifactError(
                f"package entries {names!r} do not match graph_class {name!r}"
            )
        package_constants = [
            constant.as_manifest_row() for constant in declared_constants(package_path, name)
        ]
        if package_constants != graph_class["constants"]:
            raise ArtifactError("metadata constants do not match the package constant table")
        violations = code_only_violations(package_path, name)
    except PackageIntrospectionError as exc:
        raise ArtifactError(f"cannot verify AOTInductor package: {exc}") from exc
    if violations:
        raise ArtifactError("AOTInductor package is not code-only: " + "; ".join(violations))


def _validate_single_gzip_member(artifact: Path) -> None:
    decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    uncompressed = 0
    try:
        with artifact.open("rb") as source:
            while compressed := source.read(1 << 16):
                pending = compressed
                while pending:
                    output = decoder.decompress(pending, 1 << 20)
                    uncompressed += len(output)
                    if uncompressed > _MAX_TAR_BYTES:
                        raise ArtifactError(
                            "artifact decompressed bytes exceed the "
                            f"{_MAX_TAR_BYTES}-byte v1 budget"
                        )
                    pending = decoder.unconsumed_tail
                    if decoder.eof:
                        if decoder.unused_data or pending or source.read(1):
                            raise ArtifactError("artifact must contain exactly one gzip member")
                        return
            if not decoder.eof:
                raise ArtifactError("artifact has a truncated gzip member")
    except ArtifactError:
        raise
    except (OSError, zlib.error) as exc:
        raise ArtifactError(f"cannot decompress artifact {artifact}: {exc}") from exc


def _members(artifact: Path) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
    archive: tarfile.TarFile | None = None
    try:
        artifact_size = artifact.stat().st_size
        if artifact_size > _MAX_ARCHIVE_BYTES:
            raise ArtifactError(
                f"artifact compressed bytes exceed the {_MAX_ARCHIVE_BYTES}-byte v1 budget"
            )
        _validate_single_gzip_member(artifact)
        archive = tarfile.open(artifact, mode="r:*")
        members: dict[str, tarfile.TarInfo] = {}
        sizes: dict[str, int] = {}
        while (row := archive.next()) is not None:
            if (
                row.name not in _MEMBERS
                or not row.isfile()
                or row.pax_headers
                or row.name in members
            ):
                raise ArtifactError(f"unexpected or duplicate artifact member {row.name!r}")
            sizes[row.name] = row.size
            _validate_member_sizes(sizes)
            members[row.name] = row
        # The v1 writer closes USTAR with two zero blocks and pads to the next
        # 20-block record. Consume exactly those canonical bytes, then force one
        # EOF read so gzip validates its CRC and end marker.
        expected_end = (
            (
                archive.offset
                + 2 * tarfile.BLOCKSIZE
                + tarfile.RECORDSIZE
                - 1
            )
            // tarfile.RECORDSIZE
        ) * tarfile.RECORDSIZE
        remaining_padding = expected_end - archive.fileobj.tell()
        if not 0 <= remaining_padding <= tarfile.RECORDSIZE:
            raise ArtifactError("artifact has a non-canonical USTAR end marker")
        while remaining_padding:
            trailer = archive.fileobj.read(min(1 << 20, remaining_padding))
            if not trailer:
                raise ArtifactError("artifact has truncated USTAR padding")
            if trailer.strip(b"\0"):
                raise ArtifactError("artifact has nonzero bytes after the USTAR end marker")
            remaining_padding -= len(trailer)
        if archive.fileobj.read(1):
            raise ArtifactError("artifact has bytes after the canonical USTAR record")
        missing = _REQUIRED_MEMBERS - set(members)
        if missing:
            raise ArtifactError(f"artifact is missing {sorted(missing)!r}")
        return archive, members
    except ArtifactError:
        if archive is not None:
            archive.close()
        raise
    except (EOFError, OSError, tarfile.TarError) as exc:
        if archive is not None:
            archive.close()
        raise ArtifactError(f"cannot read artifact {artifact}: {exc}") from exc


def read_metadata(artifact: str | Path) -> dict[str, Any]:
    path = Path(artifact)
    archive, members = _members(path)
    try:
        info = members[METADATA_NAME]
        handle = archive.extractfile(info)
        if handle is None:
            raise ArtifactError("metadata member is unreadable")
        try:
            raw = json.load(handle, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"metadata is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ArtifactError("metadata root must be an object")
        return validate_metadata(raw, has_literals=LITERALS_NAME in members)
    except ArtifactError:
        raise
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise ArtifactError(f"cannot read artifact {path}: {exc}") from exc
    finally:
        archive.close()


def _verify_materialized(directory: Path) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise ArtifactError(f"materialized artifact is not a regular directory: {directory}")
    try:
        rows = list(directory.iterdir())
    except OSError:
        raise
    names: set[str] = set()
    for row in rows:
        if row.name not in _MEMBERS or row.is_symlink() or not row.is_file() or row.name in names:
            raise ArtifactError(f"unexpected materialized artifact member {row.name!r}")
        names.add(row.name)
    missing = _REQUIRED_MEMBERS - names
    if missing:
        raise ArtifactError(f"materialized artifact is missing {sorted(missing)!r}")
    sizes = {name: (directory / name).stat().st_size for name in names}
    _validate_member_sizes(sizes)
    metadata_path = directory / METADATA_NAME
    try:
        raw = json.loads(metadata_path.read_bytes(), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"metadata is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ArtifactError("metadata root must be an object")
    checked = validate_metadata(raw, has_literals=LITERALS_NAME in names)
    verify_package(directory / PACKAGE_NAME, checked)
    literal_path = directory / LITERALS_NAME if LITERALS_NAME in names else None
    _verify_literals(literal_path, checked)
    return checked


def _copy_member(
    archive: tarfile.TarFile,
    info: tarfile.TarInfo,
    destination: Path,
) -> None:
    extracted = archive.extractfile(info)
    if extracted is None:
        raise ArtifactError(f"artifact member {info.name!r} is unreadable")
    remaining = info.size
    try:
        with extracted as source:
            with destination.open("wb") as output:
                while remaining:
                    chunk = source.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ArtifactError(f"artifact member {info.name!r} is truncated")
                    output.write(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
    except ArtifactError:
        raise
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise ArtifactError(f"cannot read artifact member {info.name!r}: {exc}") from exc


def unpack_artifact(artifact: str | Path, destination: str | Path) -> dict[str, Any]:
    """Validate and atomically materialize an artifact into a new directory."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    archive, members = _members(Path(artifact))
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name, info in members.items():
            _copy_member(archive, info, stage / name)
        metadata = _verify_materialized(stage)
        _fsync_dir(stage)
        os.replace(stage, target)
        _fsync_dir(target.parent)
        return metadata
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        archive.close()
