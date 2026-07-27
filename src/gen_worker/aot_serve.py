"""AOTInductor ``.pt2`` compiled artifacts (pgw#721) — the THIRD producer/
consumer on the compile-cache rails (gw#384 / th#569 / #390).

``compile_cache`` serves the dynamo lane (a JIT warmed from seeded FX
entries); ``trt_engine`` serves per-SKU TensorRT plans; this module serves
``torch.export`` -> ``aoti_compile_and_package`` artifacts. Same trust
model, storage, delivery, and arming seam as both — cells live as flavors
of ``root/family-<family>``::

    root/family-<f>#aot-<sku>-torch<maj.min>-<precision>

Artifact = deterministic ``.tar.gz`` (the receipts gate reads
``metadata.json`` straight out of the digested bytes, so the envelope is
identical in shape to a TRT engine's)::

    metadata.json           kind/format, runtime key (sku, torch, cuda),
                            family, module, cell_key, the INPUT CONTRACT
                            (per-input dtype + symbolic shape), the SYMBOL
                            RANGES, and the declared CONSTANT manifest
    model.pt2               the AOTI package — CODE ONLY
    constants.safetensors   optional: the non-weight lifted constants

Why the artifact is code-only, and what that costs
--------------------------------------------------
``aot_inductor.package_constants_in_so`` defaults ``True``, which BAKES the
weights into the ``.so`` — measured at 4.79 GiB (plain) / 2.73 GiB (w8a8)
per cell. That would duplicate model weights into every cell and destroy
the CAS / th#883 distribution model the whole cell system rests on, so
cells are minted with the flag ``False`` and weights bind at load from the
resident (CAS-provisioned) module. This is a correctness requirement for
the fleet, not an optimization, and the same decision the LoRA hot-swap
needs.

The price is pgw#704's B1: invoking a code-only artifact BEFORE
``load_constants`` **SEGFAULTS inside** ``AOTICompiledModel.__call__`` —
killing the worker, not the request. A segfault cannot be caught, so it
must be made unreachable: :class:`ArtifactRunner` refuses every call until
its own binding proof has passed (:func:`bind_constants` -> exact declared-
vs-bound FQN set), and the module swap is not installed until then.

And pgw#704's B2: an exported graph carries ZERO symbolic-range assertions
(``ep.range_constraints`` is metadata only), so out-of-declared-range input
is SILENTLY ACCEPTED — measured, 2048x2048 through an artifact declaring
``max=160`` latent units. That is a silent-failure path, which the
no-silent-failure rule forbids, so the range is asserted at OUR ingress
(:func:`assert_ingress`) where the refusal is NAMED and composes with the
cell contract, instead of depending on an upstream opt-in export pass.

Torch is imported inside functions (never at module scope) — the whole
compile stack keeps ``import gen_worker`` off the torch/pb import graph.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import activity as activity_mod
from .compile_cache import (
    AdoptError,
    CompiledLaneUnavailableError,
    _clean_tarinfo,
    parse_cell_ref,
    sku_slug,
)
from .models import lora_lifted

logger = logging.getLogger(__name__)

METADATA_NAME = "metadata.json"
PACKAGE_NAME = "model.pt2"
LITERALS_NAME = "constants.safetensors"
ARTIFACT_KIND = "aot-inductor"
ARTIFACT_FORMAT = 1
_MARKER_ATTR = "_cozy_aot"
_REQUIRED_MEMBERS = (METADATA_NAME, PACKAGE_NAME)
_OPTIONAL_MEMBERS = (LITERALS_NAME,)
_MEMBERS = _REQUIRED_MEMBERS + _OPTIONAL_MEMBERS

#: A constant whose value comes from the resident module's ``state_dict``
#: (every model weight — the bytes the CAS already delivered).
SOURCE_STATE_DICT = "state_dict"
#: A constant lifted by export that is NOT a module weight (a traced tensor
#: literal). Tiny, so it ships inside the artifact rather than being
#: reconstructed — nothing outside the artifact knows its value.
SOURCE_LITERAL = "literal"


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------


class ConstantsUnboundError(RuntimeError):
    """A code-only artifact was asked to run before its constants were
    proven bound (pgw#704 B1).

    This exception EXISTS so the segfault does not: reaching
    ``AOTICompiledModel.__call__`` with unbound constants takes down the
    whole worker process, which no ``except`` can recover. Raised by name,
    before any call crosses into the compiled extension.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


class IngressContractError(RuntimeError):
    """A call's shapes/dtypes fall outside the artifact's DECLARED contract
    (pgw#704 B2).

    The exported graph will happily run it and return numerically
    unvalidated output. Named refusal instead; the caller serves the request
    eagerly, so the request succeeds and the deviation is observable.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(detail or reason)


# ---------------------------------------------------------------------------
# Key
# ---------------------------------------------------------------------------


def torch_version() -> str:
    """Full torch version (``2.13.0+cu130``) — the artifact is ABI-locked to
    it, exactly like a TRT plan is locked to its library build."""
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return ""


def torch_maj_min(version: str) -> str:
    parts = str(version or "").split(".")
    if len(parts) < 2 or not parts[0]:
        return ""
    return f"{parts[0]}.{parts[1]}"


def runtime_key() -> Dict[str, str]:
    """Consumer-side half of the artifact key, probed from this process."""
    key = {"sku": "", "torch": torch_version(), "cuda": ""}
    try:
        import torch

        key["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return key


def flavor_label(sku: str, version: str, precision: str) -> str:
    """``aot-l4-torch2.13-w8a8``. The FULL torch version lives in metadata;
    the label carries maj.min for humans and selection."""
    mm = torch_maj_min(version)
    if not sku or not mm or not precision:
        return ""
    return f"aot-{sku}-torch{mm}-{precision}"


def is_aot_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names an AOTI cell (optionally of one family)."""
    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
    return flavor.startswith("aot-")


# ---------------------------------------------------------------------------
# The declared contract — shapes, symbols, constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputContract:
    """One declared graph input.

    ``shape`` entries are either an ``int`` (statically specialized — the
    call must match EXACTLY) or a ``str`` naming a symbol whose range lives
    in :attr:`ArtifactContract.symbols`. ``position`` is the positional
    index in the exported call convention; ``name`` is the keyword the
    pipeline's own forward uses, so a call can be matched either way.
    """

    name: str
    position: int
    dtype: str
    shape: Tuple[Any, ...]
    optional: bool = False


@dataclass(frozen=True)
class ArtifactContract:
    """The complete ingress contract of one artifact."""

    inputs: Tuple[InputContract, ...]
    #: symbol -> (min, max), inclusive. Mint packs this from
    #: ``ep.range_constraints``.
    symbols: Mapping[str, Tuple[int, int]]


@dataclass(frozen=True)
class ConstantSpec:
    """One declared constant of a code-only artifact."""

    fqn: str
    source: str
    dtype: str
    shape: Tuple[int, ...]


def contract_from_meta(meta: Mapping[str, Any]) -> ArtifactContract:
    """Parse the packed ingress contract. Raises :class:`ValueError` on a
    malformed declaration — an unparseable contract must never degrade into
    an unasserted one (that is B2 all over again)."""
    inputs: List[InputContract] = []
    raw_inputs = meta.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("metadata declares no inputs")
    for idx, row in enumerate(raw_inputs):
        if not isinstance(row, dict):
            raise ValueError(f"input {idx} is not an object")
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError(f"input {idx} has no name")
        dtype = str(row.get("dtype") or "").strip()
        if not dtype:
            raise ValueError(f"input {name!r} has no dtype")
        raw_shape = row.get("shape")
        if not isinstance(raw_shape, list):
            raise ValueError(f"input {name!r} has no shape list")
        shape: List[Any] = []
        for dim in raw_shape:
            if isinstance(dim, bool):
                raise ValueError(f"input {name!r} has a bool dim")
            if isinstance(dim, int):
                shape.append(int(dim))
            elif isinstance(dim, str) and dim.strip():
                shape.append(dim.strip())
            else:
                raise ValueError(f"input {name!r} has a malformed dim {dim!r}")
        inputs.append(InputContract(
            name=name,
            position=int(row.get("position", idx)),
            dtype=dtype,
            shape=tuple(shape),
            optional=bool(row.get("optional", False)),
        ))

    symbols: Dict[str, Tuple[int, int]] = {}
    raw_symbols = meta.get("symbols") or {}
    if not isinstance(raw_symbols, dict):
        raise ValueError("symbols is not an object")
    for sym, bounds in raw_symbols.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"symbol {sym!r} range is not a [min, max] pair")
        lo, hi = int(bounds[0]), int(bounds[1])
        if hi < lo:
            raise ValueError(f"symbol {sym!r} range {lo}..{hi} is inverted")
        symbols[str(sym)] = (lo, hi)

    # Every symbol a shape references must be bounded. An unbounded symbol
    # is the B2 hole with extra steps: it would admit any value.
    for spec in inputs:
        for pos, dim in enumerate(spec.shape):
            if isinstance(dim, str) and dim not in symbols:
                raise ValueError(
                    f"input {spec.name!r} dim {pos} references symbol "
                    f"{dim!r} with no declared range")
    return ArtifactContract(inputs=tuple(inputs), symbols=symbols)


def constants_from_meta(meta: Mapping[str, Any]) -> Tuple[ConstantSpec, ...]:
    """Parse the declared constant manifest."""
    raw = meta.get("constants")
    if not isinstance(raw, list):
        raise ValueError("metadata declares no constants list")
    out: List[ConstantSpec] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"constant {idx} is not an object")
        fqn = str(row.get("fqn") or "").strip()
        if not fqn:
            raise ValueError(f"constant {idx} has no fqn")
        source = str(row.get("source") or "").strip()
        if source not in (SOURCE_STATE_DICT, SOURCE_LITERAL):
            raise ValueError(
                f"constant {fqn!r} has unknown source {source!r} "
                f"(expected {SOURCE_STATE_DICT!r} or {SOURCE_LITERAL!r})")
        shape = row.get("shape")
        if not isinstance(shape, list):
            raise ValueError(f"constant {fqn!r} has no shape list")
        out.append(ConstantSpec(
            fqn=fqn,
            source=source,
            dtype=str(row.get("dtype") or "").strip(),
            shape=tuple(int(d) for d in shape),
        ))
    seen: Dict[str, int] = {}
    for spec in out:
        seen[spec.fqn] = seen.get(spec.fqn, 0) + 1
    dupes = sorted(f for f, n in seen.items() if n > 1)
    if dupes:
        raise ValueError(f"constant manifest repeats {dupes!r}")
    return tuple(out)


def range_digest(meta: Mapping[str, Any]) -> str:
    """Canonical digest of the DECLARED admissible traffic of one artifact.

    Owed to the ck6 identity lane (pgw#716/#717): declared dim ranges live
    in ``ep.range_constraints``, NOT in the graph nodes — three exports
    differing only in declared range produced the identical node-only
    digest. A node-only ``graph_hashes`` therefore collides artifacts that
    admit different traffic. Folding THIS digest into the per-class hash
    closes it. Exposed here (not in ``cell_key``) because this module owns
    the contract's canonical form.
    """
    contract = contract_from_meta(meta)
    canon = {
        "inputs": [
            {
                "name": s.name,
                "position": s.position,
                "dtype": s.dtype,
                "shape": list(s.shape),
                "optional": s.optional,
            }
            for s in sorted(contract.inputs, key=lambda s: (s.position, s.name))
        ],
        "symbols": {k: list(v) for k, v in sorted(contract.symbols.items())},
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def artifact_metadata(
    *,
    family: str,
    module: str,
    precision: str,
    cell_key: str,
    inputs: Sequence[Mapping[str, Any]],
    symbols: Mapping[str, Sequence[int]],
    constants: Sequence[Mapping[str, Any]],
    source_ref: str = "",
    source_digest: str = "",
) -> Dict[str, Any]:
    """Build one artifact's ``metadata.json``.

    THE single source of truth for the envelope: the mint lane calls this
    rather than hand-rolling a dict, so producer and consumer cannot drift
    into two interpretations of the same bytes. Validates the contract it
    is handed (a malformed contract must fail at MINT, on the pod, not at
    serve time on a paying request).
    """
    meta: Dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "kind": ARTIFACT_KIND,
        **runtime_key(),
        "family": str(family or ""),
        "module": str(module or ""),
        "precision": str(precision or ""),
        "cell_key": str(cell_key or ""),
        "inputs": [dict(row) for row in inputs],
        "symbols": {str(k): [int(v[0]), int(v[1])] for k, v in symbols.items()},
        "constants": [dict(row) for row in constants],
        "package_constants_in_so": False,
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
    }
    contract_from_meta(meta)
    constants_from_meta(meta)
    meta["range_digest"] = range_digest(meta)
    return meta


def verify(meta: Dict[str, Any], *, family: str = "") -> str:
    """'' when the artifact matches this runtime, else the mismatch reason.

    An AOTI ``.pt2`` is a ``dlopen``-ed ELF built against one exact torch
    C++ ABI on one compute capability — the FULL torch version must match,
    not maj.min, or the load either fails obscurely or is undefined.
    """
    if int(meta.get("format") or 0) != ARTIFACT_FORMAT:
        return f"format {meta.get('format')!r} != {ARTIFACT_FORMAT}"
    if str(meta.get("kind") or "") != ARTIFACT_KIND:
        return f"kind {meta.get('kind')!r} != {ARTIFACT_KIND}"
    # A weights-baked artifact is refused OUTRIGHT, never merely warned
    # about: it would duplicate multi-GiB weights per cell and break the CAS
    # distribution model (pgw#704 B1). Absent flag = a pre-contract mint.
    if meta.get("package_constants_in_so") is not False:
        return (
            "artifact was minted with package_constants_in_so != False "
            "(weights baked into the .so; breaks the CAS cell model)")
    here = runtime_key()
    if not here["torch"]:
        return "torch not importable"
    for field_name in ("sku", "torch", "cuda"):
        want, have = str(meta.get(field_name) or ""), here[field_name]
        if want != have:
            return f"{field_name} {want!r} != runtime {have!r}"
    want_fam = str(meta.get("family") or "")
    if family and want_fam and want_fam != family:
        return f"family {want_fam!r} != {family!r}"
    try:
        contract_from_meta(meta)
        constants_from_meta(meta)
    except ValueError as exc:
        return f"malformed declared contract: {exc}"
    stamped = str(meta.get("range_digest") or "")
    if stamped and stamped != range_digest(meta):
        return "range_digest does not match the declared contract"
    return ""


# ---------------------------------------------------------------------------
# Pack / unpack
# ---------------------------------------------------------------------------


def pack(content_dir: Path, out_path: Path, metadata: Dict[str, Any]) -> Path:
    """Deterministic artifact from ``content_dir`` holding ``model.pt2``
    (and optionally ``constants.safetensors``)."""
    content_dir = Path(content_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                meta_bytes = json.dumps(metadata, sort_keys=True, indent=1).encode()
                ti = _clean_tarinfo(tarfile.TarInfo(METADATA_NAME))
                ti.size = len(meta_bytes)
                tar.addfile(ti, io.BytesIO(meta_bytes))
                for name in (PACKAGE_NAME,) + _OPTIONAL_MEMBERS:
                    p = content_dir / name
                    if not p.exists():
                        if name in _REQUIRED_MEMBERS:
                            raise ValueError(f"{name} missing from {content_dir}")
                        continue
                    ti = _clean_tarinfo(tarfile.TarInfo(name))
                    ti.size = p.stat().st_size
                    with open(p, "rb") as f:
                        tar.addfile(ti, f)
    return out_path


def unpack(artifact: Path, dest_root: Path) -> Dict[str, Any]:
    """Extract the fixed member set into ``dest_root``; returns metadata."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {}
    seen: set[str] = set()
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            name = member.name
            if name not in _MEMBERS or not member.isfile() or name in seen:
                raise ValueError(
                    f"unexpected member in {ARTIFACT_KIND} artifact: {member.name!r}")
            seen.add(name)
            src = tar.extractfile(member)
            assert src is not None
            data = src.read()
            if name == METADATA_NAME:
                meta = json.loads(data.decode())
                continue
            (dest_root / name).write_bytes(data)
    missing = set(_REQUIRED_MEMBERS) - seen
    if missing:
        raise ValueError(
            f"{ARTIFACT_KIND} artifact {artifact} is incomplete; "
            f"missing {sorted(missing)!r}")
    if not meta:
        raise ValueError(f"{ARTIFACT_KIND} artifact {artifact} has no {METADATA_NAME}")
    # A literal-sourced constant with no payload member would only be
    # discovered at bind time, mid-arm. Name it here.
    literals = [
        s.fqn for s in constants_from_meta(meta) if s.source == SOURCE_LITERAL]
    if literals and LITERALS_NAME not in seen:
        raise ValueError(
            f"{ARTIFACT_KIND} artifact {artifact} declares literal constants "
            f"{sorted(literals)[:4]!r} but carries no {LITERALS_NAME}")
    return meta


def unpack_metadata(artifact: Path) -> Dict[str, Any]:
    """Read ONLY metadata.json from an artifact (kind sniffing — cheap)."""
    with tarfile.open(artifact, mode="r:*") as tar:
        for member in tar:
            if member.name == METADATA_NAME and member.isfile():
                src = tar.extractfile(member)
                assert src is not None
                return json.loads(src.read().decode())
    raise ValueError(f"artifact {artifact} has no {METADATA_NAME}")


def is_aot_artifact(artifact: Path) -> bool:
    """Kind sniff for the ``provision.enable_compiled`` dispatch. Never
    raises: an unreadable/foreign artifact is simply not ours."""
    try:
        meta = unpack_metadata(Path(artifact))
    except Exception:
        return False
    return str(meta.get("kind") or "") == ARTIFACT_KIND


@dataclass
class _StagedAotArtifact:
    metadata: Dict[str, Any]
    root: Path
    temporary: "tempfile.TemporaryDirectory[str]"

    def close(self) -> None:
        self.temporary.cleanup()


def stage_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None,
) -> _StagedAotArtifact:
    """Extract and runtime-verify a complete artifact in an isolated tree.

    The live/shared cache and pipeline remain untouched on every rejection.
    Concurrent attempts use distinct trees; a process crash can leave only
    an unreferenced staging directory, never a partially published ``.pt2``.
    """
    base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker"
    base.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="aot-stage-", dir=base)
    root = Path(temporary.name)
    try:
        meta = unpack(Path(artifact), root)
        reason = verify(meta, family=family)
        if reason:
            raise AdoptError("key_mismatch", reason)
        return _StagedAotArtifact(meta, root, temporary)
    except AdoptError:
        temporary.cleanup()
        raise
    except Exception as exc:
        temporary.cleanup()
        raise AdoptError("artifact_invalid", str(exc)) from exc


def find_artifact(root: Path) -> Optional[Path]:
    """The artifact tarball inside a downloaded snapshot dir (or the file)."""
    root = Path(root)
    if root.is_file():
        return root
    return next(iter(sorted(root.rglob("*.tar.gz"))), None)


# ---------------------------------------------------------------------------
# B2 — the ingress range assertion
# ---------------------------------------------------------------------------


def _dtype_name(value: Any) -> str:
    """``torch.bfloat16`` -> ``bfloat16``; anything else -> ''."""
    raw = str(getattr(value, "dtype", "") or "")
    if not raw:
        return ""
    return raw.split(".")[-1]


def bind_call_inputs(
    contract: ArtifactContract, args: Sequence[Any], kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Match one call's actual arguments to the declared inputs by keyword
    first, then by position. Missing non-optional input => named refusal."""
    out: Dict[str, Any] = {}
    for spec in contract.inputs:
        if spec.name in kwargs:
            out[spec.name] = kwargs[spec.name]
        elif 0 <= spec.position < len(args):
            out[spec.name] = args[spec.position]
        elif spec.optional:
            continue
        else:
            raise IngressContractError(
                "input_missing",
                f"declared input {spec.name!r} (position {spec.position}) is "
                f"absent from the call ({len(args)} positional, "
                f"kwargs {sorted(kwargs)[:8]!r})")
    return out


def marshal_positional(
    contract: ArtifactContract,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> List[Any]:
    """Flatten one call into the package's POSITIONAL signature.

    MEASURED on the first-light pod (pgw#721 S8 / #723): an AOTI package
    takes **positional inputs only** — splatting the pipeline's kwargs at it
    dies with ``Ran into a kwarg keyword mismatch: Got [...] but expected
    []`` before any compiled code runs. Diffusers, meanwhile, calls a
    denoiser with a mix of positional and (nested) keyword arguments, so the
    serve path cannot pass the call through untouched: it has to flatten to
    the exact order export recorded, or it feeds the right tensor to the
    wrong graph input.

    ``position`` in the declared contract is that order. Every declared
    input must be present — a package has a FIXED flat arity, so a missing
    one cannot be skipped without silently shifting every later argument
    into the wrong slot. That is a named refusal, not a best effort.
    """
    bound = bind_call_inputs(contract, args, kwargs)
    feeds: List[Any] = []
    for spec in sorted(contract.inputs, key=lambda s: s.position):
        if spec.name not in bound:
            raise IngressContractError(
                "input_missing",
                f"declared input {spec.name!r} (position {spec.position}) is "
                "absent; an AOTI package has a fixed flat arity and takes "
                "positional inputs only, so a missing input would shift "
                "every later argument into the wrong graph slot")
        feeds.append(bound[spec.name])
    return feeds


def assert_ingress(
    contract: ArtifactContract,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Dict[str, int]:
    """Assert one call against the artifact's DECLARED contract (pgw#704 B2).

    Returns the resolved symbol bindings on success; raises
    :class:`IngressContractError`, naming the input, the dim, the symbol,
    the value and the bound, on any violation. Checks, per input:

    * present (unless declared optional);
    * dtype EXACT — an exported graph is specialized on dtype;
    * rank EXACT;
    * static dims EXACT — a specialized dim is not a range;
    * symbolic dims inside the declared inclusive range;
    * **symbol CONSISTENCY** — one symbol appearing in two shapes must take
      the same value. ``range_constraints`` cannot express this, but the
      graph requires it, so a mismatch is out-of-contract even when both
      values are individually in range.
    """
    bound = bind_call_inputs(contract, args, kwargs)
    symbols: Dict[str, int] = {}
    owner: Dict[str, str] = {}
    for spec in contract.inputs:
        if spec.name not in bound:
            continue
        value = bound[spec.name]
        shape = getattr(value, "shape", None)
        if shape is None:
            raise IngressContractError(
                "input_not_tensor",
                f"declared input {spec.name!r} is a "
                f"{type(value).__name__} with no shape")
        got_dtype = _dtype_name(value)
        if got_dtype != spec.dtype:
            raise IngressContractError(
                "dtype_mismatch",
                f"input {spec.name!r} dtype {got_dtype or '<unknown>'} != "
                f"declared {spec.dtype}")
        actual = tuple(int(d) for d in shape)
        if len(actual) != len(spec.shape):
            raise IngressContractError(
                "rank_mismatch",
                f"input {spec.name!r} rank {len(actual)} != declared "
                f"{len(spec.shape)} (declared shape {list(spec.shape)!r})")
        for pos, (declared, got) in enumerate(zip(spec.shape, actual)):
            if isinstance(declared, int):
                if got != declared:
                    raise IngressContractError(
                        "static_dim_mismatch",
                        f"input {spec.name!r} dim {pos} = {got} != "
                        f"statically specialized {declared}")
                continue
            lo, hi = contract.symbols[declared]
            if not (lo <= got <= hi):
                raise IngressContractError(
                    "range_violation",
                    f"input {spec.name!r} dim {pos} (symbol {declared!r}) = "
                    f"{got} outside declared range [{lo}, {hi}]")
            prior = symbols.get(declared)
            if prior is not None and prior != got:
                raise IngressContractError(
                    "symbol_inconsistent",
                    f"symbol {declared!r} = {got} on input {spec.name!r} dim "
                    f"{pos} but {prior} on input {owner[declared]!r}")
            symbols[declared] = got
            owner.setdefault(declared, spec.name)
    return symbols


# ---------------------------------------------------------------------------
# B1 — the constants-bound gate
# ---------------------------------------------------------------------------


def _load_package(path: Path) -> Any:
    """Load one ``.pt2`` (the sole torch entry point for the load path —
    tests substitute this)."""
    import torch

    return torch._inductor.aoti_load_package(str(path))


def _load_literals(path: Path, device: str) -> Dict[str, Any]:
    from safetensors.torch import load_file

    return dict(load_file(str(path), device=device))


def resolve_constants(
    specs: Sequence[ConstantSpec],
    state_dict: Mapping[str, Any],
    literals: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble the full constant mapping from the resident weights plus the
    artifact's literal payload.

    Export preserves module FQNs, so a ``state_dict``-sourced constant is a
    direct name lookup — no value-identity matching (``trt_engine`` needs
    that only because ONNX renames). An unresolvable FQN is a named refusal:
    binding a partial set is exactly the state that segfaults.
    """
    out: Dict[str, Any] = {}
    missing: List[str] = []
    for spec in specs:
        table = state_dict if spec.source == SOURCE_STATE_DICT else literals
        if spec.fqn not in table:
            missing.append(f"{spec.fqn} (source={spec.source})")
            continue
        out[spec.fqn] = table[spec.fqn]
    if missing:
        raise ConstantsUnboundError(
            "constant_unresolved",
            f"{len(missing)} declared constant(s) have no value: "
            f"{sorted(missing)[:6]!r}")
    return out


def assert_bindable(
    specs: Sequence[ConstantSpec], runner_fqns: Iterable[str],
) -> None:
    """The artifact's OWN constant table must equal the declared manifest.

    Both directions matter. A declared FQN the artifact does not want means
    the manifest describes different bytes than we loaded. An FQN the
    artifact wants that is undeclared would be left UNBOUND — the segfault
    precondition — and, when it is a LoRA branch, a missing FQN is also the
    constant-folded-adapter bug in a different hat (pgw#704 G1).
    """
    declared = {s.fqn for s in specs}
    actual = set(runner_fqns)
    only_declared = sorted(declared - actual)
    only_actual = sorted(actual - declared)
    if only_declared or only_actual:
        raise ConstantsUnboundError(
            "constant_set_mismatch",
            f"artifact constant table != declared manifest; "
            f"declared-only={only_declared[:6]!r} "
            f"artifact-only={only_actual[:6]!r}")


@dataclass
class ArtifactRunner:
    """A loaded code-only ``.pt2`` behind the two pgw#704 gates.

    Every call is refused until :meth:`bind` has proven the constant set
    complete (B1 — the alternative is a process-killing segfault) and is
    checked against the declared contract (B2). The underlying compiled
    model is reached ONLY from :meth:`__call__`, after both gates.
    """

    package: Any
    contract: ArtifactContract
    constants: Tuple[ConstantSpec, ...]
    module_name: str = ""
    bound: bool = False
    bound_fqns: Tuple[str, ...] = ()
    calls: int = 0
    refusals: Dict[str, int] = field(default_factory=dict)

    def declared_fqns(self) -> Tuple[str, ...]:
        return tuple(s.fqn for s in self.constants)

    def bind(
        self, state_dict: Mapping[str, Any], literals: Mapping[str, Any],
    ) -> None:
        """Bind constants from the resident weights + literal payload, then
        PROVE the artifact reports them all bound.

        Order is load-bearing: nothing may call the package before this
        returns, and this must not mark itself bound on a partial update.
        """
        try:
            table = self.package.get_constant_fqns()
        except Exception as exc:
            raise ConstantsUnboundError(
                "constant_table_unreadable",
                f"artifact will not report its constant FQNs: "
                f"{type(exc).__name__}: {exc}") from exc
        assert_bindable(self.constants, table)
        values = resolve_constants(self.constants, state_dict, literals)
        # check_full_update=True is the artifact's own assertion that the
        # update covers its ENTIRE table. We already proved set equality
        # above; this makes torch refuse rather than leave a hole.
        #
        # MEASURED (pgw#721 S8 first light, L4/torch 2.9.1+cu128): this HOLDS
        # on a real sdxl w8a8 cell — 2,422 constants, all state_dict-sourced,
        # declared and package sets identical, strict update accepted.
        # CAVEAT, still open: that artifact folded NOTHING (zero literals,
        # zero from_folded), so strictness is UNTESTED against a folding
        # artifact. If a folding cell ever refuses here, the choice is to
        # relax to check_full_update=False (keeping our own set-equality
        # proof above as the real gate) or to have the mint ship the folded
        # bytes — decide on evidence, not by loosening the gate first.
        self.package.load_constants(values, check_full_update=True)
        self.bound_fqns = tuple(sorted(values))
        self.bound = True

    def assert_ready(self) -> None:
        """The gate that keeps the segfault unreachable."""
        if not self.bound:
            raise ConstantsUnboundError(
                "constants_unbound",
                f"refusing to invoke code-only artifact "
                f"({self.module_name or 'unknown module'}) with "
                f"{len(self.constants)} unbound constant(s): calling before "
                f"load_constants segfaults the worker process")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.assert_ready()
        try:
            assert_ingress(self.contract, args, kwargs)
            feeds = marshal_positional(self.contract, args, kwargs)
        except IngressContractError as exc:
            self.refusals[exc.reason] = self.refusals.get(exc.reason, 0) + 1
            raise
        out = self.package(*feeds)
        self.calls += 1
        return out


# ---------------------------------------------------------------------------
# Serve — module swap behind a fail-soft guard
# ---------------------------------------------------------------------------


def lifted_call_kwargs(module: Any) -> Dict[str, Any]:
    """The lifted-adapter call kwargs for one denoiser, or ``{}`` (pgw#725).

    Under input-lifting the rank bucket arrives as two flat tensors in the
    CALL rather than as module state, so nothing can be baked and the
    no-baked-adapter gate degenerates to a signature check. Both kwargs are
    MANDATORY when a binding exists: ``bind_views`` refuses a ``None`` half
    by name, so the EAGER fallback needs them exactly as much as the
    artifact does — which is why the wrapper merges them once, up front, for
    both paths.

    The tensors are returned by reference and never copied. The adapter
    machinery mutates them IN PLACE through views, so a swap needs no
    artifact interaction at all and the call arguments stay pointer-stable
    (what cudagraph static inputs require).

    ``bucket=0`` is its own branchless graph class with no lifted signature,
    so it has no binding and gets no kwargs.
    """
    binding = lora_lifted.lifted_binding(module)
    return binding.call_kwargs() if binding is not None else {}


def assert_lifted_contract(module: Any, contract: ArtifactContract) -> None:
    """The module's lifted state and the artifact's signature must AGREE.

    Either mismatch is the pgw#704 S9 defect in one direction or the other:
    an artifact that declares the adapter inputs but a module with no
    binding has no way to supply a mandatory input, while a lifted module
    served by an artifact that does NOT declare them means the branch was
    traced away — every request silently gets the base model. Caught at arm
    time, by name, instead of at the first call.
    """
    declared = {spec.name for spec in contract.inputs}
    wanted = set(lora_lifted.LIFTED_INPUT_NAMES)
    lifted = lora_lifted.lifted_binding(module) is not None
    if lifted and not wanted <= declared:
        raise AdoptError(
            "lifted_inputs_undeclared",
            f"module carries a lifted LoRA adapter but the artifact declares "
            f"no {sorted(wanted - declared)!r} input(s); the branch was traced "
            "away, so every request would silently serve the base model")
    if not lifted and wanted & declared:
        raise AdoptError(
            "lifted_inputs_unbindable",
            f"artifact declares lifted adapter input(s) "
            f"{sorted(wanted & declared)!r} but the module has no lifted "
            "binding to supply them (bucket=0 is a different graph class)")


def wrap_module(
    module: Any,
    runner: ArtifactRunner,
    meta: Dict[str, Any],
    *,
    eager_forward: Optional[Callable[..., Any]] = None,
) -> None:
    """Swap ``module.forward`` for the artifact behind a fail-soft guard.

    Mirrors ``trt_engine.wrap_module``: the first artifact ERROR
    synchronously revokes scheduler-visible compiled proof and permanently
    routes to eager; the module object (config, dtype, device, weights)
    stays untouched, and its weights remain the constant-binding source.

    An :class:`IngressContractError` is NOT such an error. It is a named,
    counted, per-request contract refusal — the request serves eagerly and
    the artifact stays armed for in-contract traffic, because one
    out-of-range request says nothing about the artifact's health.

    pgw#725 LoRA seam: when the denoiser carries a lifted adapter, both
    ``lora_a``/``lora_b`` are MANDATORY call kwargs and this wrapper is the
    call site that supplies them — see :func:`lifted_call_kwargs`.
    """
    original = eager_forward or module.forward
    state: Dict[str, Any] = {
        "failed": False,
        "successful_calls": 0,
        "ingress_refusals": 0,
        "last_refusal": "",
        "original": original,
        "failure_callback": None,
        "revocation_error": "",
        "runner": runner,
    }

    def aot_forward(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledLaneUnavailableError(state["revocation_error"])
        # The lifted pair is resolved PER CALL, not captured at wrap time:
        # the LoRA lane may install/remove lifting independently of arming,
        # and a stale capture would either starve the graph of a mandatory
        # input or feed one to a forward that no longer takes it.
        kwargs = {**kwargs, **lifted_call_kwargs(module)}
        if state["failed"]:
            return original(*args, **kwargs)
        try:
            return runner(*args, **kwargs)
        except IngressContractError as exc:
            # B2: the exported graph would have run this and returned
            # unvalidated output. Named refusal + eager service.
            state["ingress_refusals"] = int(state["ingress_refusals"]) + 1
            state["last_refusal"] = f"{exc.reason}: {exc}"
            logger.warning(
                "aot-serve: %s REFUSED out-of-contract input (%s: %s); "
                "serving this request eager, artifact stays armed",
                meta.get("module"), exc.reason, exc)
            activity_mod.emit_event(
                "aot_ingress_refused",
                f"family={meta.get('family')} module={meta.get('module')}: {exc}",
                phase=exc.reason,
            )
            return original(*args, **kwargs)
        except ConstantsUnboundError as exc:
            # Reaching here means the arm order was violated. The gate did
            # its job (no segfault); the lane is structurally unusable.
            state["failed"] = True
            logger.error(
                "aot-serve: %s invoked with unbound constants (%s); eager for "
                "the rest of this process", meta.get("module"), exc)
            activity_mod.emit_event(
                "aot_constants_unbound",
                f"family={meta.get('family')} module={meta.get('module')}: {exc}",
                phase=exc.reason,
            )
            _revoke(state, f"constants unbound: {exc}")
            return original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — ANY artifact problem => eager
            state["failed"] = True
            detail = (
                f"AOTI artifact {meta.get('module')} failed: "
                f"{type(exc).__name__}: {exc}")
            _revoke(state, detail)
            logger.warning(
                "aot-serve: %s failed (%s: %s); eager for the rest of this "
                "process", meta.get("module"), type(exc).__name__, exc)
            return original(*args, **kwargs)

    def _counted_forward(*args: Any, **kwargs: Any) -> Any:
        out = aot_forward(*args, **kwargs)
        state["successful_calls"] = int(runner.calls)
        return out

    module.forward = _counted_forward
    setattr(module, _MARKER_ATTR, {
        "meta": {k: meta.get(k) for k in ("sku", "torch", "precision", "symbols")},
        "state": state,
    })


def _revoke(state: Dict[str, Any], detail: str) -> None:
    """Run the scheduler-state revocation callback for a failed artifact."""
    callback = state.get("failure_callback")
    if not callable(callback):
        return
    try:
        callback(detail)
    except Exception as callback_exc:
        state["revocation_error"] = (
            f"compiled-state revocation failed: "
            f"{type(callback_exc).__name__}: {callback_exc}")
        logger.exception("aot-serve: %s", state["revocation_error"])
        raise CompiledLaneUnavailableError(state["revocation_error"]) from callback_exc


def load_and_wrap(
    pipeline: Any, cfg: Any, artifact: Path, cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stage + verify + load + BIND CONSTANTS, then perform the sole live wrap.

    Raises :class:`AdoptError` with a classified reason on any failure, and
    never publishes extracted files into a shared live cache. The wrap is
    the last step: an artifact that cannot bind its constants never becomes
    reachable from ``module.forward``.
    """
    family = str(getattr(cfg, "family", "") or "")
    staged = stage_artifact(Path(artifact), family, cache_dir=cache_dir)
    try:
        meta = staged.metadata
        module_name = str(meta.get("module") or "unet")
        module = getattr(pipeline, module_name, None)
        if module is None:
            raise AdoptError("no_target", f"pipeline has no module {module_name!r}")

        eager_forward: Optional[Callable[..., Any]] = None
        old_marker = getattr(pipeline, _MARKER_ATTR, None)
        if old_marker is not None:
            old_module = old_marker.get("module")
            old_state = old_marker.get("state") or {}
            eager_forward = old_state.get("original")
            if old_module is not module or not callable(eager_forward):
                raise AdoptError(
                    "old_marker_invalid",
                    "existing AOT marker does not retain this module's eager "
                    "callable")

        t0 = time.monotonic()
        try:
            contract = contract_from_meta(meta)
            constants = constants_from_meta(meta)
            # pgw#725: the lifted-adapter signature must match the module's
            # actual lifted state before anything is armed.
            assert_lifted_contract(module, contract)
        except ValueError as exc:
            raise AdoptError("contract_invalid", str(exc)) from exc

        package = _load_package(staged.root / PACKAGE_NAME)
        runner = ArtifactRunner(
            package=package, contract=contract, constants=constants,
            module_name=module_name)

        literals: Dict[str, Any] = {}
        literals_path = staged.root / LITERALS_NAME
        if literals_path.exists():
            device = str(getattr(module, "device", "cuda"))
            literals = _load_literals(literals_path, device)
        try:
            runner.bind(dict(module.state_dict()), literals)
        except ConstantsUnboundError as exc:
            raise AdoptError(f"constants_{exc.reason}", str(exc)) from exc

        # First live mutation. Everything above is proven: complete artifact,
        # matching runtime key, resolved target, loaded package, and a
        # constant table proven bound against the declared manifest.
        wrap_module(module, runner, meta, eager_forward=eager_forward)
        module_marker = getattr(module, _MARKER_ATTR, {})
        setattr(pipeline, _MARKER_ATTR, {
            "meta": meta,
            "state": module_marker.get("state", {}),
            "module": module,
        })
        logger.info(
            "aot-serve: loaded+bound %s in %.1fs (%d constants, symbols %s)",
            module_name, time.monotonic() - t0, len(constants),
            dict(contract.symbols))
        return meta
    finally:
        staged.close()


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> bool:
    """Consumer entry point: verify + load + bind + swap an AOTI artifact.

    Returns False (staying eager) on ANY miss — the caller's ordinary miss
    policy (fleet self-mint / eager / typed refusal) takes over, exactly as
    for a TRT engine. Returning True IS the HIT: ``fleet_cells`` treats it
    as a genuine match and skips the self-mint.
    """
    if artifact is None:
        return False
    try:
        meta = load_and_wrap(pipeline, cfg, Path(artifact), cache_dir=cache_dir)
    except Exception as exc:
        logger.warning("aot-serve: artifact unusable (%s); staying eager", exc)
        return False
    logger.info(
        "aot-serve: armed %s (sku=%s torch=%s precision=%s, constants bound "
        "from resident weights)",
        meta.get("module"), meta.get("sku"), meta.get("torch"),
        meta.get("precision"))
    return True


def execution_count(pipeline: Any) -> int:
    """Successful artifact calls observed on this exact wrapped pipeline."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    state = marker.get("state") or {}
    runner = state.get("runner")
    if runner is not None:
        return int(getattr(runner, "calls", 0))
    return int(state.get("successful_calls", 0))


def ingress_refusals(pipeline: Any) -> int:
    """Out-of-contract calls refused by name on this pipeline (B2)."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    return int((marker.get("state") or {}).get("ingress_refusals", 0))


def proven_since(pipeline: Any, before: int) -> bool:
    """The exported lane's ADOPTION PROOF (pgw#735).

    An exported artifact performs no FX cache lookup, so the dynamo lane's
    ``cache_hit_count > 0`` proof can never pass for it — scoring a ``.pt2``
    that way disproves every honest adoption. Its own evidence is: it EXECUTED
    since ``before``, and it is STILL armed (an artifact that ran and then
    revoked on a B1/B2 refusal has not proven anything). Both halves are
    load-bearing, and neither is a synthesized counter: this is the one path
    whose entire job is to detect a lie about serving compiled.
    """
    return execution_count(pipeline) > int(before) and is_armed(pipeline)


def is_armed(pipeline: Any) -> bool:
    """Whether an AOTI artifact is currently serving this pipeline."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    state = marker.get("state") or {}
    return bool(state) and not state.get("failed", False)


def set_guard_failure_callback(pipeline: Any, callback: Any) -> bool:
    """Bind scheduler-state revocation to an armed artifact's guard."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    state = marker.get("state")
    if not isinstance(state, dict):
        return False
    state["failure_callback"] = callback
    return True


def unwrap(pipeline: Any) -> bool:
    """Restore eager forward — rotation/eviction and the unproven-adoption
    rollback both go through here, same as the TRT lane."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    module = marker.get("module")
    state = marker.get("state") or {}
    original = state.get("original")
    if module is None or not callable(original):
        return False
    module.forward = original
    for holder in (module, pipeline):
        try:
            delattr(holder, _MARKER_ATTR)
        except AttributeError:
            pass
    return True


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_KIND",
    "ArtifactContract",
    "ArtifactRunner",
    "ConstantSpec",
    "ConstantsUnboundError",
    "IngressContractError",
    "InputContract",
    "LITERALS_NAME",
    "METADATA_NAME",
    "PACKAGE_NAME",
    "SOURCE_LITERAL",
    "SOURCE_STATE_DICT",
    "artifact_metadata",
    "assert_bindable",
    "assert_lifted_contract",
    "assert_ingress",
    "bind_call_inputs",
    "constants_from_meta",
    "contract_from_meta",
    "enable",
    "execution_count",
    "proven_since",
    "find_artifact",
    "flavor_label",
    "ingress_refusals",
    "is_aot_artifact",
    "is_aot_ref",
    "is_armed",
    "lifted_call_kwargs",
    "load_and_wrap",
    "marshal_positional",
    "pack",
    "range_digest",
    "resolve_constants",
    "runtime_key",
    "set_guard_failure_callback",
    "torch_maj_min",
    "torch_version",
    "unpack",
    "unpack_metadata",
    "unwrap",
    "verify",
    "wrap_module",
]
