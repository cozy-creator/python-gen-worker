"""Read publishability facts from a PyTorch AOTInductor ``.pt2`` package."""

from __future__ import annotations

import io
import re
import struct
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast


class PackageIntrospectionError(RuntimeError):
    """A compiled package is malformed or does not declare an expected fact."""


@dataclass(frozen=True)
class DeclaredConstant:
    """One row in the constant table emitted into an AOTInductor wrapper."""

    index: int
    name: str
    fqn: str
    data_size: int
    kind: str
    dtype: str
    shape: tuple[int, ...]
    from_folded: bool

    @property
    def source(self) -> str:
        if self.kind in _STATE_DICT_TYPES:
            return "state_dict"
        if self.kind in _COMPUTED_TYPES:
            return "computed"
        return "literal"

    def as_manifest_row(self) -> dict[str, object]:
        return {
            "fqn": self.fqn,
            "source": self.source,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


_CONSTANT_FIELD = re.compile(
    r"constants_info_\[(\d+)\]\.(name|data_size|from_folded|original_fqn)\s*=\s*"
    r'(?:"([^"]*)"|(\d+)|(true|false))\s*;'
)
_CONSTANT_TYPE = re.compile(
    r"constants_info_\[(\d+)\]\.type\s*=\s*static_cast<int32_t>\s*\(\s*"
    r"torch::aot_inductor::ConstantType::(\w+)\s*\)\s*;"
)
_CONSTANT_DTYPE = re.compile(r"constants_info_\[(\d+)\]\.dtype\s*=\s*cached_torch_dtype_(\w+)\s*;")
_CONSTANT_SHAPE = re.compile(r"constants_info_\[(\d+)\]\.shape\s*=\s*\{([^}]*)\}\s*;")
_MODEL_BASE = "AOTInductorModelBase("
_AOTINDUCTOR_ROOT = "data/aotinductor/"
_ELF_MAGIC = b"\x7fELF"
_ELF64_SECTION_HEADER_SIZE = 64
_SHT_NULL = 0
_SHT_PROGBITS = 1
_SHT_STRTAB = 3
_SHT_NOBITS = 8
_MAX_SECTION_NAME_TABLE_BYTES = 16 << 20
_LRODATA = ".lrodata"
_STATE_DICT_TYPES = frozenset(("Parameter", "Buffer"))
_COMPUTED_TYPES = frozenset(("FoldedConstant",))


def _valid_entry(entry: str) -> bool:
    return (
        bool(entry)
        and "\\" not in entry
        and "\0" not in entry
        and all(component not in {"", ".", ".."} for component in entry.split("/"))
    )


def _member_layout(name: str) -> tuple[str, str, str] | None:
    if name.startswith(_AOTINDUCTOR_ROOT):
        root = ""
        rest = name[len(_AOTINDUCTOR_ROOT) :]
    else:
        root, separator, rest = name.partition(f"/{_AOTINDUCTOR_ROOT}")
        if not separator or "/" in root or not _valid_entry(root):
            return None
    entry, separator, filename = rest.rpartition("/")
    if not separator or not filename or not _valid_entry(entry):
        return None
    return root, entry, filename


def _member_entry(name: str, suffix: str) -> str | None:
    layout = _member_layout(name)
    if layout is None or not layout[2].endswith(suffix):
        return None
    return layout[1]


def _members(package: Path, suffix: str, entry: str = "") -> list[str]:
    if entry and not _valid_entry(entry):
        raise PackageIntrospectionError(f"{package}: invalid package entry name {entry!r}")
    try:
        with zipfile.ZipFile(package) as archive:
            layouts = [layout for name in archive.namelist() if (layout := _member_layout(name))]
            roots = {root for root, _, _ in layouts}
            if len(roots) > 1:
                raise PackageIntrospectionError(
                    f"{package}: package members use multiple AOTInductor roots {sorted(roots)!r}"
                )
            names = [
                name
                for name in archive.namelist()
                if (member_entry := _member_entry(name, suffix)) is not None
                and (not entry or member_entry == entry)
            ]
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PackageIntrospectionError(f"{package}: invalid .pt2 package: {exc}") from exc
    return names


def _one_member(package: Path, suffix: str, entry: str) -> str:
    names = _members(package, suffix, entry)
    if len(names) != 1:
        scope = f" for entry {entry!r}" if entry else ""
        raise PackageIntrospectionError(
            f"{package}: expected exactly one *{suffix} in the package{scope}, found {len(names)}"
        )
    return names[0]


def _read_member(package: Path, name: str) -> bytes:
    try:
        with zipfile.ZipFile(package) as archive:
            return archive.read(name)
    except (
        KeyError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise PackageIntrospectionError(
            f"{package}: cannot read package member {name!r}: {exc}"
        ) from exc


@contextmanager
def _open_member(package: Path, name: str) -> Iterator[tuple[BinaryIO, int]]:
    try:
        with zipfile.ZipFile(package) as archive:
            info = archive.getinfo(name)
            with archive.open(info) as source:
                yield cast(BinaryIO, source), info.file_size
    except PackageIntrospectionError:
        raise
    except (
        KeyError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise PackageIntrospectionError(
            f"{package}: cannot read package member {name!r}: {exc}"
        ) from exc


def _wrapper_source(package: Path, entry: str = "") -> str:
    name = _one_member(package, ".wrapper.cpp", entry)
    try:
        return _read_member(package, name).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackageIntrospectionError(
            f"{package}: package member {name!r} is not UTF-8: {exc}"
        ) from exc


def _package_entry_names(package: Path) -> tuple[str, ...]:
    """Return named graph entries discovered from wrapper paths."""

    return tuple(
        sorted(
            {
                entry
                for member in _members(Path(package), ".wrapper.cpp")
                if (entry := _member_entry(member, ".wrapper.cpp")) is not None
            }
        )
    )


def _call_arguments(source: str, opening: str) -> tuple[str, ...]:
    start = source.find(opening)
    if start < 0:
        raise PackageIntrospectionError(f"generated wrapper has no {opening!r} call")
    index = start + len(opening)
    depth = 1
    argument_start = index
    arguments: list[str] = []
    while index < len(source):
        char = source[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                arguments.append(source[argument_start:index].strip())
                return tuple(arguments)
        elif char == "," and depth == 1:
            arguments.append(source[argument_start:index].strip())
            argument_start = index + 1
        index += 1
    raise PackageIntrospectionError(f"unbalanced {opening!r} argument list")


def _model_base_arguments(source: str) -> tuple[str, ...]:
    arguments = _call_arguments(source, _MODEL_BASE)
    if len(arguments) < 4:
        raise PackageIntrospectionError(
            f"generated wrapper has an unsupported {_MODEL_BASE[:-1]} signature"
        )
    return arguments


def _declared_constant_count(source: str) -> int:
    argument = _model_base_arguments(source)[2]
    if re.fullmatch(r"\d+", argument) is None:
        raise PackageIntrospectionError(
            "generated wrapper has no literal constant count in "
            f"{_MODEL_BASE[:-1]} argument 3: {argument!r}"
        )
    return int(argument)


def _constants_in_so(package: Path, entry: str = "") -> bool:
    """Return the package's declared ``load_constants_from_blob`` value."""

    package = Path(package)
    argument = _model_base_arguments(_wrapper_source(package, entry))[-1]
    if argument not in {"true", "false"}:
        raise PackageIntrospectionError(
            f"{package}: expected a boolean load_constants_from_blob argument, got {argument!r}"
        )
    return argument == "true"


def declared_constants(package: Path, entry: str = "") -> tuple[DeclaredConstant, ...]:
    """Parse the package's generated constant table in declaration order."""

    package = Path(package)
    source = _wrapper_source(package, entry)
    expected_count = _declared_constant_count(source)
    fields: dict[int, dict[str, object]] = {}

    def assign(index: int, field_name: str, value: object) -> None:
        row = fields.setdefault(index, {})
        if field_name in row:
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}] declares {field_name!r} twice"
            )
        row[field_name] = value

    for raw_index, field_name, text, number, boolean in _CONSTANT_FIELD.findall(source):
        index = int(raw_index)
        if field_name == "data_size":
            if not number:
                raise PackageIntrospectionError(
                    f"{package}: constants_info_[{index}].data_size is not an integer"
                )
            assign(index, field_name, int(number))
        elif field_name == "from_folded":
            if not boolean:
                raise PackageIntrospectionError(
                    f"{package}: constants_info_[{index}].from_folded is not a boolean"
                )
            assign(index, field_name, boolean == "true")
        else:
            if number or boolean:
                raise PackageIntrospectionError(
                    f"{package}: constants_info_[{index}].{field_name} is not a string"
                )
            assign(index, field_name, text)
    for raw_index, kind in _CONSTANT_TYPE.findall(source):
        assign(int(raw_index), "kind", kind)
    for raw_index, dtype in _CONSTANT_DTYPE.findall(source):
        assign(int(raw_index), "dtype", dtype)
    for raw_index, dimensions in _CONSTANT_SHAPE.findall(source):
        index = int(raw_index)
        try:
            shape = tuple(
                int(dimension.strip()) for dimension in dimensions.split(",") if dimension.strip()
            )
        except ValueError as exc:
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}].shape is not a literal integer shape"
            ) from exc
        if any(dimension < 0 for dimension in shape):
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}].shape has a negative dimension"
            )
        assign(index, "shape", shape)

    expected_indices = set(range(expected_count))
    if set(fields) != expected_indices:
        missing_indices = sorted(expected_indices - set(fields))
        unexpected = sorted(set(fields) - expected_indices)
        raise PackageIntrospectionError(
            f"{package}: constant table does not match declared count {expected_count} "
            f"(missing indices {missing_indices}, unexpected indices {unexpected})"
        )

    constants: list[DeclaredConstant] = []
    for index in sorted(fields):
        row = fields[index]
        required = {"name", "data_size", "from_folded", "kind", "dtype", "shape"}
        missing_fields = sorted(required - set(row))
        if missing_fields:
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}] is missing required fields {missing_fields}"
            )
        name = row["name"]
        data_size = row["data_size"]
        kind = row["kind"]
        dtype = row["dtype"]
        raw_shape = row["shape"]
        from_folded = row["from_folded"]
        if not isinstance(name, str) or not name:
            raise PackageIntrospectionError(f"{package}: constants_info_[{index}] has no name")
        if not isinstance(data_size, int) or isinstance(data_size, bool) or data_size < 0:
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}] has an invalid data_size"
            )
        if not isinstance(kind, str) or not kind:
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}] has no constant type"
            )
        if not isinstance(dtype, str) or not dtype:
            raise PackageIntrospectionError(f"{package}: constants_info_[{index}] has no dtype")
        if not isinstance(raw_shape, tuple):
            raise PackageIntrospectionError(f"{package}: constants_info_[{index}] has no shape")
        if not isinstance(from_folded, bool):
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}] has no from_folded flag"
            )
        fqn = row.get("original_fqn") or name
        if not isinstance(fqn, str):
            raise PackageIntrospectionError(
                f"{package}: constants_info_[{index}] has an invalid original_fqn"
            )
        constants.append(
            DeclaredConstant(
                index=index,
                name=name,
                fqn=fqn,
                data_size=data_size,
                kind=kind,
                dtype=dtype,
                shape=raw_shape,
                from_folded=from_folded,
            )
        )
    return tuple(constants)


def _read_at(source: BinaryIO, offset: int, size: int, total_size: int, what: str) -> bytes:
    if offset < 0 or size < 0 or offset + size > total_size:
        raise PackageIntrospectionError(f"packaged .so has an invalid {what} range")
    source.seek(offset)
    data = source.read(size)
    if len(data) != size:
        raise PackageIntrospectionError(f"packaged .so has a truncated {what}")
    return data


def _elf_section_sizes(source: BinaryIO, total_size: int) -> dict[str, int]:
    if total_size < 64:
        raise PackageIntrospectionError("packaged .so is not an ELF image")
    header = _read_at(source, 0, 64, total_size, "ELF header")
    if not header.startswith(_ELF_MAGIC):
        raise PackageIntrospectionError("packaged .so is not an ELF image")
    if header[4] != 2 or header[5] != 1:
        raise PackageIntrospectionError(
            "packaged .so is not 64-bit little-endian ELF "
            f"(EI_CLASS={header[4]}, EI_DATA={header[5]})"
        )
    if header[6] != 1:
        raise PackageIntrospectionError("packaged .so has an unsupported ELF version")

    section_offset = struct.unpack_from("<Q", header, 0x28)[0]
    section_size, section_count, string_table_index = struct.unpack_from("<HHH", header, 0x3A)
    table_size = section_size * section_count
    if (
        section_offset < len(header)
        or not section_count
        or section_size < _ELF64_SECTION_HEADER_SIZE
        or string_table_index >= section_count
        or section_offset + table_size > total_size
    ):
        raise PackageIntrospectionError("packaged .so has an invalid ELF section table")

    def section_header(index: int) -> bytes:
        return _read_at(
            source,
            section_offset + index * section_size,
            _ELF64_SECTION_HEADER_SIZE,
            total_size,
            "ELF section header",
        )

    string_header = section_header(string_table_index)
    string_type = struct.unpack_from("<I", string_header, 0x04)[0]
    string_offset, string_size = struct.unpack_from("<QQ", string_header, 0x18)
    if string_type != _SHT_STRTAB or not string_size:
        raise PackageIntrospectionError("packaged .so has an invalid ELF section-name table")
    if string_size > _MAX_SECTION_NAME_TABLE_BYTES:
        raise PackageIntrospectionError("packaged .so has an oversized ELF section-name table")
    string_table = _read_at(
        source, string_offset, string_size, total_size, "ELF section-name table"
    )

    sizes: dict[str, int] = {}
    for index in range(section_count):
        section = section_header(index)
        name_offset, section_type = struct.unpack_from("<II", section)
        file_offset, section_bytes = struct.unpack_from("<QQ", section, 0x18)
        if (
            section_type not in {_SHT_NULL, _SHT_NOBITS}
            and file_offset + section_bytes > total_size
        ):
            raise PackageIntrospectionError(
                f"packaged .so has an invalid file range for ELF section {index}"
            )
        if name_offset >= len(string_table):
            raise PackageIntrospectionError("packaged .so has an invalid ELF section name")
        end = string_table.find(b"\0", name_offset)
        if end < 0:
            raise PackageIntrospectionError("packaged .so has an unterminated ELF section name")
        try:
            name = string_table[name_offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PackageIntrospectionError(
                "packaged .so has a non-UTF-8 ELF section name"
            ) from exc
        if name == _LRODATA and section_type != _SHT_PROGBITS:
            raise PackageIntrospectionError(
                f"packaged .so has {_LRODATA} with unexpected ELF section type {section_type}"
            )
        sizes[name] = sizes.get(name, 0) + section_bytes
    return sizes


def _elf_section_sizes_from_bytes(blob: bytes) -> dict[str, int]:
    """Read section sizes from a 64-bit little-endian ELF image."""

    return _elf_section_sizes(io.BytesIO(blob), len(blob))


def code_only_violations(package: Path, entry: str = "") -> list[str]:
    """Return reasons one package entry has constants embedded in its shared object."""

    package = Path(package)
    constants = declared_constants(package, entry)
    declared_bytes = sum(constant.data_size for constant in constants)
    reasons: list[str] = []

    if _constants_in_so(package, entry):
        largest = sorted(constants, key=lambda constant: -constant.data_size)[:5]
        details = ", ".join(f"{constant.fqn} ({constant.data_size}B)" for constant in largest)
        suffix = f"; largest: {details}" if details else ""
        reasons.append(
            "package declares load_constants_from_blob=true: "
            f"{len(constants)} constants totalling {declared_bytes}B are embedded in the .so"
            f"{suffix}"
        )

    shared_object = _one_member(package, ".so", entry)
    with _open_member(package, shared_object) as (source, size):
        readonly_data = _elf_section_sizes(source, size).get(_LRODATA, 0)
    if declared_bytes and readonly_data >= declared_bytes:
        reasons.append(
            f"{shared_object}: {_LRODATA} is {readonly_data}B, at least the {declared_bytes}B "
            f"declared across {len(constants)} constants"
        )
    return reasons
