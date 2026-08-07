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

    metadata.json           kind/format, runtime key (sm, torch, cuda + sku),
                            family, cell_key, and the ENTRIES map — one
                            block per NAMED GRAPH CLASS carrying its
                            target, fork/class-dim coordinate, INPUT
                            CONTRACT, SYMBOL RANGES, declared CONSTANT
                            manifest, and per-class hash
    model.pt2               ONE AOTI package holding every entry as a
                            named model (``data/aotinductor/<entry>/``) —
                            CODE ONLY
    constants.safetensors   optional: non-weight lifted constants, keys
                            namespaced ``<entry>::<fqn>``

Format 2 — multi-graph cells (pgw#758, Paul's ruling)
-----------------------------------------------------
"generate and generate_turbo are separate functions, they have separate
graphs, but they are COMBINED TOGETHER INTO ONE FILE." One cell per
(family x lane x contract) carries EVERY declared graph class as a named
entry; one resident artifact serves them all. Serve-side dispatch selects
the entry whose DECLARED ingress contract admits the call — zero admitting
entries is a named refusal (eager service), and more than one is
``entry_ambiguous``: a declaration defect made visible, never a guess.
Every pgw#704 gate (B1 constants-bound, B2 ingress) holds PER ENTRY —
the unbound-entry segfault was re-measured per named model on the pin.

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

import copy
import gzip
import hashlib
import io
import json
import logging
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
    Union,
)

from . import activity as activity_mod
from . import aot_flatten
from . import host_isa
from .cell_adopt import AdoptOutcome
from . import shape_growth
from .compile_cache import (
    AdoptError,
    CompiledExecutionLaneUnavailableError,
    _clean_tarinfo,
    _resolve_target,
    parse_cell_ref,
    sku_slug,
)
from .models import lora_lifted

logger = logging.getLogger(__name__)

METADATA_NAME = "metadata.json"
PACKAGE_NAME = "model.pt2"
LITERALS_NAME = "constants.safetensors"
ARTIFACT_KIND = "aot-inductor"
#: pgw#791: an input the artifact was compiled for as 16-byte aligned arrived
#: unaligned (or non-contiguous) and this ingress realigned it. Typed and
#: hub-visible because the ALTERNATIVE is what shipped: AOTInductor's own
#: ``run_impl`` copies the input on EVERY call and says so on the worker's
#: stderr, which is unreachable on hub-spawned pods — a fleet paying the tax
#: was indistinguishable from one that was not. Coalesced: once per
#: (entry, input, reason).
REALIGN_EVENT = "aot_input_realigned"
#: AOTInductor generates its aligned-input fast path at 16 bytes
#: (``torch._inductor.codegen.aoti_runtime``'s ALIGNMENT). An input whose
#: ``data_ptr()`` is not a multiple of this — diffusers hands the denoiser
#: ``timesteps[i]``, a scalar VIEW at an odd element offset — makes the
#: runner clone it per call. Not a knob: it is the compiler's constant.
AOTI_ALIGNMENT = 16
#: Format 2 = multi-graph cells (pgw#758): the envelope carries an
#: ``entries`` map instead of one flat contract. Format-1 cells are RETIRED
#: (exact identity: a recipe change strands old cells, which is fine).
ARTIFACT_FORMAT = 2
#: Separator between the entry name and the constant FQN in
#: ``constants.safetensors`` keys. Entry names never contain it (targets are
#: dotted identifiers; coordinate values are ints/bools/identifiers).
LITERAL_SEP = "::"
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

#: The hardware/toolchain axes an ``.pt2`` is genuinely pinned to (pgw#765).
#: ``sm`` is the GPU identity: AOTInductor itself keys on
#: ``AOTI_COMPUTE_CAPABILITY`` — capability, never the marketing name
#: (``codecache.get_device_information``). ``sku`` is deliberately ABSENT:
#: Paul's ruling ("AOT cells are locked into the sm_x version, not the actual
#: GPU"), the pgw#691 collapse that removed it from cell identity on
#: byte-identical evidence, and the pgw#754 ISA clamp that made the host half
#: portable. It stays in metadata for observability and as the discovery
#: SELECTION PREFERENCE (``aot_cells._candidates``) — never as a refusal.
IDENTITY_AXES: Tuple[str, ...] = ("sm", "torch", "cuda")


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
    """Consumer-side half of the artifact key, probed from this process.

    ``sm`` is the compiled-code identity (:data:`IDENTITY_AXES`); ``sku`` is
    the GPU's marketing slug, recorded for observability and selection.
    """
    key = {"sku": "", "sm": "", "torch": torch_version(), "cuda": ""}
    try:
        import torch

        key["cuda"] = str(torch.version.cuda or "")
        if torch.cuda.is_available():
            key["sku"] = sku_slug(torch.cuda.get_device_name(0))
            major, minor = torch.cuda.get_device_capability(0)
            key["sm"] = f"sm_{major}{minor}"
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


# Stamped cell keys this process LEARNED name aot-inductor artifacts
# (pgw#722 F1 discovery). Published AOT cells ride the same key space as
# their store flavor — indistinguishable from a dynamo cell's flavor by
# string shape alone — so discovery registers each learned key here and
# :func:`is_aot_ref` consults the set. Without this the executor's kind
# dispatch (#734/#735) would score an armed ``.pt2`` by FX cache hits and
# disprove every honest adoption.
_KNOWN_AOT_KEYS: set[str] = set()
_KNOWN_AOT_KEYS_LOCK = threading.Lock()


def note_aot_key(cell_key: str) -> None:
    """Record that ``cell_key`` (a stamped cell-key digest) is an AOT cell."""
    key = str(cell_key or "").strip()
    if not key:
        return
    with _KNOWN_AOT_KEYS_LOCK:
        _KNOWN_AOT_KEYS.add(key)


def is_aot_ref(ref: str, family: str = "") -> bool:
    """True when ``ref`` names an AOTI cell (optionally of one family).

    Recognizes both the label form (``#aot-<sku>-...``) and any stamped
    cell key this process learned via :func:`note_aot_key`.
    """
    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
    if flavor.startswith("aot-"):
        return True
    with _KNOWN_AOT_KEYS_LOCK:
        return flavor in _KNOWN_AOT_KEYS


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

    :attr:`param` / :attr:`param_position` / :attr:`path` are the input's
    IDENTITY IN THE CALL (pgw#994): which argument it lives in, where that
    argument sits, and the path into it. ``position`` counts FLATTENED graph
    inputs and a container argument occupies one caller slot while producing
    N of them, so position alone binds the wrong value the moment any input is
    a list or a dict. Defaults are the trivial identity — an input that IS its
    argument — which is what every row published before pgw#994 declares.
    """

    name: str
    position: int
    dtype: str
    shape: Tuple[Any, ...]
    optional: bool = False
    param: str = ""
    param_position: int = -1
    path: Tuple[Union[int, str], ...] = ()

    @property
    def call_param(self) -> str:
        """The argument this input lives in (itself, when trivial)."""
        return self.param or self.name

    @property
    def call_position(self) -> int:
        """The argument's own position (this input's, when trivial)."""
        return self.position if self.param_position < 0 else self.param_position

    @property
    def trivial_identity(self) -> bool:
        """True when the identity says exactly what its absence would say.

        Defined on the RESOLVED identity, not on which fields were written, so
        a row that spells its trivial identity out and one that omits it are
        the same contract — which is what ``range_digest`` needs in order to
        key the field without re-keying anything already published.
        """
        return (not self.path and self.call_param == self.name
                and self.call_position == self.position)


@dataclass(frozen=True)
class ArtifactContract:
    """The complete ingress contract of one artifact."""

    inputs: Tuple[InputContract, ...]
    #: symbol -> (min, max), inclusive. Mint packs this from
    #: ``ep.range_constraints``.
    symbols: Mapping[str, Tuple[int, int]]
    #: Inputs this graph class REFUSES to be given (pgw#790). The positive
    #: contract says what a graph takes; a multi-class cell also needs the
    #: negative one, because "input absent" is what discriminates a
    #: BRANCHLESS class from a branch-bearing one whose extra inputs a
    #: name-keyed bind would simply ignore. Without it both classes admit an
    #: adapter-bearing call and :class:`EntryDispatch` refuses
    #: ``entry_ambiguous`` — a declaration that cannot discriminate two graph
    #: classes by ingress, which its own contract calls a defect.
    excluded: Tuple[str, ...] = ()


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
        raw_path = row.get("path", [])
        if not isinstance(raw_path, list):
            raise ValueError(f"input {name!r} has a malformed path {raw_path!r}")
        path: List[Union[int, str]] = []
        for step in raw_path:
            if isinstance(step, bool) or not isinstance(step, (int, str)):
                raise ValueError(
                    f"input {name!r} has a malformed path step {step!r}")
            path.append(int(step) if isinstance(step, int) else str(step))
        inputs.append(InputContract(
            name=name,
            position=int(row.get("position", idx)),
            dtype=dtype,
            shape=tuple(shape),
            optional=bool(row.get("optional", False)),
            param=str(row.get("param") or ""),
            param_position=int(row.get("param_position", -1)),
            path=tuple(path),
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

    raw_excluded = meta.get("excluded_inputs") or []
    if not isinstance(raw_excluded, (list, tuple)):
        raise ValueError("excluded_inputs is not a list")
    excluded: List[str] = []
    declared_names = {spec.name for spec in inputs}
    for value in raw_excluded:
        name = str(value or "").strip()
        if not name:
            raise ValueError("excluded_inputs carries an empty name")
        if name in declared_names:
            raise ValueError(
                f"input {name!r} is declared AND excluded — a graph class "
                f"cannot both take an input and refuse it")
        excluded.append(name)
    return ArtifactContract(
        inputs=tuple(inputs), symbols=symbols,
        excluded=tuple(sorted(set(excluded))))


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

    Owed to the exact-identity lane (pgw#716/#717): declared dim ranges live
    in ``ep.range_constraints``, NOT in the graph nodes — three exports
    differing only in declared range produced the identical node-only
    digest. A node-only ``graph_hashes`` therefore collides artifacts that
    admit different traffic. Folding THIS digest into the per-class hash
    closes it. Exposed here (not in ``cell_key``) because this module owns
    the contract's canonical form.
    """
    contract = contract_from_meta(meta)

    def _row(s: InputContract) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "name": s.name,
            "position": s.position,
            "dtype": s.dtype,
            "shape": list(s.shape),
            "optional": s.optional,
        }
        if not s.trivial_identity:
            # pgw#994, on the `excluded` precedent below: the call identity is
            # part of the declared traffic (two classes that take the same
            # tensors in different argument structures are different graphs),
            # but it is keyed only when it is not the trivial identity. Every
            # row published before pgw#994 is trivial, so no live cell is
            # re-keyed by this field existing.
            row["param"] = s.call_param
            row["param_position"] = s.call_position
            row["path"] = list(s.path)
        return row

    canon = {
        "inputs": [
            _row(s)
            for s in sorted(contract.inputs, key=lambda s: (s.position, s.name))
        ],
        "symbols": {k: list(v) for k, v in sorted(contract.symbols.items())},
    }
    # pgw#790: the NEGATIVE half of the declared admissible traffic. Two
    # classes that differ only in what they REFUSE admit different traffic,
    # so the digest must see it or the collision this function exists to
    # close reopens for adapter forks. Keyed only when non-empty: a contract
    # that excludes nothing is the contract every already-published cell
    # declares, and re-keying the fleet's 144 live checkpoints to add a field
    # that says "unchanged" would strand every one of them.
    if contract.excluded:
        canon["excluded"] = list(contract.excluded)
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def class_hash(
    entry: Mapping[str, Any], *, strict: bool, lora_bucket: int,
) -> str:
    """The per-class graph hash of one packaged entry (pgw#716/#758).

    Folds the entry's coordinate (target, fork, class dims), its
    ``range_digest`` (the MEASURED node-only-collision fix: three exports
    differing only in declared range hashed identically), its graph
    interface block, and the trace-mode/lora facts. 16-hex, recomputable
    from the entry block alone — so a consumer can prove the stamp and a
    mismatch NAMES the class (the receipts principle).
    """
    facts = {
        "v": 2,
        "target": str(entry.get("target") or ""),
        "fork": [[str(n), v] for n, v in (entry.get("fork") or [])],
        "class_dims": [
            [str(n), int(v)] for n, v in (entry.get("class_dims") or [])],
        "range_digest": str(entry.get("range_digest") or ""),
        "graph": dict(entry.get("graph") or {}),
        "strict": bool(strict),
        "lora_bucket": int(lora_bucket or 0),
    }
    blob = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def combined_graph_hash(hashes: Iterable[str]) -> str:
    """The combined hash, VERBATIM per pgw#716: the first 16 hex chars
    of the sha256 over the newline-joined SORTED per-class hash values
    (sorted by the hash string itself, single ``\\n`` joins, no trailing
    newline, UTF-8 bytes)."""
    joined = "\n".join(sorted(str(h) for h in hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def entries_from_meta(meta: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The validated ``entries`` map of a format-2 artifact.

    Every entry block must parse as a full contract (inputs, symbols,
    constants) and carry a target — an entry the dispatch cannot route or
    assert is B2 with extra steps. Raises :class:`ValueError` naming the
    entry."""
    raw = meta.get("entries")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("metadata declares no entries map")
    out: Dict[str, Dict[str, Any]] = {}
    for name, block in raw.items():
        label = str(name or "").strip()
        if not label:
            raise ValueError("entries map carries an unnamed entry")
        if not isinstance(block, Mapping):
            raise ValueError(f"entry {label!r} is not an object")
        if LITERAL_SEP in label:
            raise ValueError(
                f"entry name {label!r} contains {LITERAL_SEP!r}, which the "
                f"literal namespace reserves")
        if not str(block.get("target") or "").strip():
            raise ValueError(f"entry {label!r} declares no target")
        try:
            contract_from_meta(block)
            constants_from_meta(block)
        except ValueError as exc:
            raise ValueError(f"entry {label!r}: {exc}") from exc
        out[label] = dict(block)
    return out


def artifact_metadata(
    *,
    family: str,
    precision: str,
    cell_key: str,
    entries: Mapping[str, Mapping[str, Any]],
    strict_export: bool = True,
    lora_bucket: int = 0,
    source_ref: str = "",
    source_digest: str = "",
) -> Dict[str, Any]:
    """Build one multi-graph artifact's ``metadata.json`` (format 2).

    THE single source of truth for the envelope: the mint lane calls this
    rather than hand-rolling a dict, so producer and consumer cannot drift
    into two interpretations of the same bytes. Each entry block carries
    ``target``/``fork``/``class_dims``/``inputs``/``symbols``/``constants``
    (+ ``graph``); this function validates every one, stamps its
    ``range_digest`` and ``class_hash``, and stamps the
    ``combined_graph_hash`` over the sorted per-class hashes. A malformed
    contract must fail at MINT, on the pod, not at serve time on a paying
    request.
    """
    stamped: Dict[str, Dict[str, Any]] = {}
    for name, block in entries.items():
        # Deep copy: the stamped envelope must not alias the caller's nested
        # containers (a later caller-side mutation would silently rewrite the
        # recorded contract).
        row = copy.deepcopy(dict(block))
        row["range_digest"] = range_digest(row)
        row["class_hash"] = class_hash(
            row, strict=bool(strict_export), lora_bucket=int(lora_bucket or 0))
        stamped[str(name)] = row
    meta: Dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "kind": ARTIFACT_KIND,
        **runtime_key(),
        "family": str(family or ""),
        "precision": str(precision or ""),
        "cell_key": str(cell_key or ""),
        "entries": stamped,
        "strict_export": bool(strict_export),
        "lora_bucket": int(lora_bucket or 0),
        "package_constants_in_so": False,
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
        # pgw#754: the host-CPU execution requirement of the packaged host
        # code (wrapper .so + cpu kernels). Consumers refuse by name when
        # this host cannot execute it — the .so must never be dlopen'd
        # first and SIGILL second.
        "host_isa": host_isa.stamp(),
    }
    entries_from_meta(meta)
    meta["combined_graph_hash"] = combined_graph_hash(
        row["class_hash"] for row in stamped.values())
    return meta


#: The metadata keys :func:`verify_declared` rules on — every axis a cell's
#: publish DECLARE carries, and therefore everything discovery may refuse a
#: cell for before it has downloaded a byte.
#:
#: pgw#988: this set and ``fleet_cells.control_plane_metadata`` are two halves
#: of ONE contract, and they used to be two independent computations of it.
#: th#1645 moved ``entries`` out of the declare (correctly — it is unbounded in
#: the model and the declare is control-plane) while the pre-download filter
#: still demanded it, so every AOT cell published for the next day was rejected
#: as ``malformed declared contract`` by every pod, and a pod that finds no cell
#: mints its own — the fleet paid a full compile per cold boot and the symptom
#: presented as cost, not as an error. ``fleet_cells`` now asserts at import
#: that nothing it strips appears here.
DECLARED_AXES: Tuple[str, ...] = (
    "format", "kind", "package_constants_in_so", *IDENTITY_AXES,
    "host_isa", "family",
)


def verify_declared(meta: Dict[str, Any], *, family: str = "") -> str:
    """'' when a cell's DECLARE matches this runtime, else the reason.

    The pre-download half of :func:`verify`: exactly the axes a bounded
    control-plane declare carries (:data:`DECLARED_AXES`). Discovery rules on
    this against the hub listing row, so an unloadable cell costs no bytes.

    An AOTI ``.pt2`` is a ``dlopen``-ed ELF built against one exact torch
    C++ ABI on one compute capability — the FULL torch version must match,
    not maj.min, or the load either fails obscurely or is undefined.

    Fail-closed on every REAL axis (:data:`IDENTITY_AXES` + host ISA), each of
    them STRICTLY — an axis a cell is silent on is refused by name, never
    skipped. Never on ``sku`` (pgw#765): a cell minted on an l4
    and a cell minted on an rtx-4090 are the same sm_89 compiled code, and
    refusing the cross-SKU adoption discards the whole point of the pgw#691
    collapse, the FX inner-key shim, and the pgw#754 ISA clamp. The JIT lane
    (``compile_cache.verify``) carried the identical hard sku pin and shed it
    in the ck3 wave; this is the same defect on the exported lane.
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
    for field_name in IDENTITY_AXES:
        want, have = str(meta.get(field_name) or ""), here[field_name]
        if want != have:
            return f"{field_name} {want!r} != runtime {have!r}"
    isa_reason = host_isa_reason(meta)
    if isa_reason:
        return isa_reason
    want_fam = str(meta.get("family") or "")
    if family and want_fam and want_fam != family:
        return f"family {want_fam!r} != {family!r}"
    return ""


def verify_contract(meta: Dict[str, Any]) -> str:
    """'' when the artifact's ``entries`` contract is self-consistent, else
    the reason.

    The post-download half of :func:`verify`. ``entries`` is unbounded in the
    size of the model and rides INSIDE the artifact — :func:`unpack` reads it
    off ``metadata.json``, which is where ``aot_serve`` has always served it
    from. It is verified HERE, on the staged bytes, and never against a
    control-plane declare that is not required to carry it (pgw#988).
    """
    try:
        entries = entries_from_meta(meta)
    except ValueError as exc:
        return f"malformed declared contract: {exc}"
    strict = bool(meta.get("strict_export", True))
    bucket = int(meta.get("lora_bucket") or 0)
    hashes: List[str] = []
    for name, block in entries.items():
        stamped = str(block.get("range_digest") or "")
        if stamped and stamped != range_digest(block):
            return f"entry {name!r}: range_digest does not match its contract"
        stamped_hash = str(block.get("class_hash") or "")
        if not stamped_hash:
            return f"entry {name!r}: no class_hash stamped"
        if stamped_hash != class_hash(block, strict=strict, lora_bucket=bucket):
            # The receipts principle (pgw#716): a hash mismatch NAMES the class.
            return f"entry {name!r}: class_hash does not match its recorded facts"
        hashes.append(stamped_hash)
    stamped_combined = str(meta.get("combined_graph_hash") or "")
    if stamped_combined and stamped_combined != combined_graph_hash(hashes):
        return "combined_graph_hash does not match the per-entry class hashes"
    return ""


def verify(meta: Dict[str, Any], *, family: str = "") -> str:
    """'' when a cell's FULL metadata matches this runtime, else the reason.

    Both halves, for callers holding an artifact's own ``metadata.json``
    (:func:`stage_artifact`). Discovery, which holds only a declare, calls
    :func:`verify_declared` and reaches this one after the fetch.
    """
    return verify_declared(meta, family=family) or verify_contract(meta)


#: :func:`host_isa_reason`'s refusal for a cell that stamped no requirement.
#: Also the ``aot_cells`` discovery reject class, so the same fact has one name
#: whether it is ruled on before download or after staging.
NO_HOST_ISA_STAMP = "no_host_isa_stamp"


def host_isa_reason(meta: Mapping[str, Any]) -> str:
    """'' when this host's CPU can execute the artifact's packaged host
    code, else the refusal reason (pgw#754).

    Reads the mint's ``host_isa`` requirement stamp — metadata-only, so
    discovery (``aot_cells._candidates``) filters unexecutable cells BEFORE
    downloading them. An artifact carrying NO stamp is refused here: its true
    ISA need is undiscoverable from metadata, an AVX-512-built ``.pt2``
    SIGILLs (exit 132 inside ``aoti_load_package``) on a host without it, and
    the miss policy for a refused cell is a self-mint that stamps one.
    """
    block = meta.get("host_isa")
    if not isinstance(block, Mapping):
        return f"artifact records no host_isa stamp ({NO_HOST_ISA_STAMP})"
    level, machine = host_isa.requirement_of_meta(block)
    return host_isa.unsupported_reason(level, machine)


def _package_torch_stamps(package: Path) -> List[Dict[str, Any]]:
    """Torch's own ``*_metadata.json`` rows inside a ``.pt2`` — the mint
    environment as TORCH recorded it (``AOTI_CPU_ISA`` / ``AOTI_MACHINE`` /
    ``AOTI_COMPUTE_CAPABILITY``, written by
    ``codecache.get_device_information``). Empty list when unreadable: an
    unreadable package fails later, loudly, in the load path with its own
    named error, and a gate only rules on what it can read.
    """
    rows: List[Dict[str, Any]] = []
    try:
        import zipfile

        with zipfile.ZipFile(package) as zf:
            for name in zf.namelist():
                if not name.endswith("_metadata.json"):
                    continue
                try:
                    row = json.loads(zf.read(name).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except (OSError, zipfile.BadZipFile) as exc:
        logger.debug("aot-serve: package stamp read failed: %s", exc)
    return rows


def verify_package_compute_capability(package: Path) -> str:
    """The staged-bytes GPU-architecture gate: torch's own
    ``AOTI_COMPUTE_CAPABILITY`` inside the ``.pt2`` against this device's
    capability. '' = same architecture (or unknowable).

    The second tier that lets :func:`verify` stop refusing on ``sku``
    without loosening the axis that actually matters (pgw#765). It rules on
    the one case metadata cannot: a cell whose ``sm`` stamp disagrees with
    its own bytes. A
    cubin built for another arch has no PTX fallback (pgw#698 packs cubins
    only), so without this the refusal would be a raw CUDA load error
    instead of a named ``adopt_failed:sm_mismatch``.

    The METADATA ``sm`` axis stays :func:`verify`'s (reported as
    ``key_mismatch`` like every other stamped axis, and rulable before the
    bytes are ever fetched).
    """
    here = runtime_key()["sm"]
    if not here:
        return ""  # no CUDA device to rule against; the load path will say so
    # torch writes the capability as ``major*10+minor`` ("89"); digits-only
    # comparison also admits the dotted/tuple spellings other device
    # interfaces use, so a shape surprise is never read as a mismatch.
    here_digits = "".join(c for c in here if c.isdigit())
    for row in _package_torch_stamps(package):
        raw = str(row.get("AOTI_COMPUTE_CAPABILITY") or "").strip()
        digits = "".join(c for c in raw if c.isdigit())
        if not digits:
            continue
        if digits != here_digits:
            return (f"sm 'sm_{digits}' != runtime {here!r} "
                    f"(torch package stamp)")
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
    # discovered at bind time, mid-arm. Name it (and its entry) here.
    literals = [
        f"{name}{LITERAL_SEP}{s.fqn}"
        for name, block in entries_from_meta(meta).items()
        for s in constants_from_meta(block) if s.source == SOURCE_LITERAL]
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
        # pgw#754: rule on host-CPU executability FIRST and by name — the
        # one failure mode that must never reach dlopen.
        isa_reason = host_isa_reason(meta)
        if isa_reason:
            raise AdoptError("host_isa_unsupported", isa_reason)
        reason = verify(meta, family=family)
        if reason:
            raise AdoptError("key_mismatch", reason)
        # pgw#765: the GPU-architecture axis as the BYTES declare it, ruled
        # on by name before dlopen — the tier that keeps cross-SKU adoption
        # honest now that ``sku`` no longer stands in for the arch. Runs
        # after `verify` so a stamped axis mismatch keeps its own name.
        sm_reason = verify_package_compute_capability(root / PACKAGE_NAME)
        if sm_reason:
            raise AdoptError("sm_mismatch", sm_reason)
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
    """Match one call's actual arguments to the declared inputs by REPLAYING
    each input's recorded identity in the call. Missing non-optional input =>
    named refusal.

    THE RULE (pgw#994): an input is found at ``kwargs[param]`` — or at
    ``args[param_position]`` — followed by its ``path`` into that argument.
    The mint recorded that identity with ``aot_flatten.flatten_call``; this
    replays it with ``aot_flatten.resolve_leaf``, so the two sides of the
    flattening are one rule and not two spellings that agree by luck.

    WHAT THIS REPLACES, because both halves were luck. The old resolution
    tried ``kwargs[name]``, then ``args[position]``, then a SEARCH inside any
    mapping-valued kwarg for ``name``.

    * ``position`` counts FLATTENED graph inputs, but ``args`` is the call
      BEFORE flattening. sdxl escaped because diffusers passes its dict by
      keyword and the positional branch never fired. z-image, whose ``x`` is
      a ``list[Tensor]``, could not escape: ``x.0`` bound the whole list,
      ``x.1`` bound the NEXT argument, and the last input then refused
      ``input_missing`` — measured off-GPU, and it is what pgw#994 filed.
    * the nested search found ``text_embeds`` in ``added_cond_kwargs``
      because no other kwarg happened to carry that key. It is deleted here:
      the contract now SAYS where the value is, so nothing needs to hunt for
      it, and a dict that carries a colliding key can no longer decide a bind.
    """
    out: Dict[str, Any] = {}
    for spec in contract.inputs:
        found, value = aot_flatten.resolve_leaf(
            spec.call_param, spec.call_position, spec.path, args, kwargs)
        if found:
            out[spec.name] = value
        elif spec.optional:
            continue
        else:
            where = f"argument {spec.call_param!r}"
            if spec.path:
                where += " path " + "".join(
                    f"[{step!r}]" for step in spec.path)
            raise IngressContractError(
                "input_missing",
                f"declared input {spec.name!r} ({where}, position "
                f"{spec.call_position}) is absent from the call "
                f"({len(args)} positional, kwargs {sorted(kwargs)[:8]!r})")
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

    THE COMPLEMENT, measured on the pgw#723 residuals pod (2.13.0+cu130):
    the package's call convention mirrors the traced EXAMPLE's args/kwargs
    split. An input fed as a KWARG at export bakes a kwarg-demanding
    ``in_spec`` — the package then refuses a positional call with the same
    error in reverse (``Got [] but expected ['lora_a', 'lora_b']``), the
    wrap treats it as an artifact failure, and the lane silently revokes to
    eager on the FIRST call. So "positional inputs only" is a MINT
    obligation, not a torch invariant: example inputs must be fed
    all-positionally (lifted adapter pair included) so this positional
    marshal matches the package.

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


def excluded_inputs_present(
    contract: ArtifactContract, kwargs: Mapping[str, Any],
) -> Tuple[str, ...]:
    """The contract's EXCLUDED inputs this call actually carries (pgw#790).

    Keyword and nested-mapping only: an excluded input has no position in this
    graph's signature, so a positional index would name a different argument
    entirely. A ``None`` value does not count as carrying it — that is how a
    diffusers pipeline says "absent".

    This still SEARCHES, and deliberately (pgw#994, which deleted the search
    in :func:`bind_call_inputs`). The two questions are not the same one: a
    declared input has a contract row, so its location is recorded and can be
    replayed; an EXCLUDED input has no row by definition, and the question
    asked of it is "does this call carry such a value anywhere" — for which
    there is nothing to replay. A diffusers pipeline can hand an adapter down
    nested (``cross_attention_kwargs``), and a branchless class that missed it
    would silently serve the base model.
    """
    if not contract.excluded:
        return ()
    found: List[str] = []
    for name in contract.excluded:
        value = kwargs.get(name, None)
        if value is None:
            value = next(
                (v[name] for v in kwargs.values()
                 if isinstance(v, Mapping) and v.get(name) is not None),
                None)
        if value is not None:
            found.append(name)
    return tuple(found)


def alignment_gap(value: Any) -> str:
    """'' when a feed satisfies the artifact's aligned-input contract, else
    the named reason (pgw#791).

    AOTInductor compiles the fast path for 16-byte-aligned, contiguous
    inputs. When the run-time pointer is not aligned its ``run_impl`` copies
    the tensor to an aligned buffer ON EVERY CALL and reports it with a C++
    ``TORCH_WARN`` — i.e. on the worker's stderr, which hub-spawned pods do
    not expose. Measured on an RTX 4090 (WARM-INFERENCE-MATRIX §2c): the
    request residual over 28x(per-forward) is 196 ms for the armed AOT
    artifact against 77 ms for the equivalent dynamo cell, and diffusers'
    ``timesteps[i]`` scalar view is the offending input. So the check the
    serve path never performed cost more than the artifact kind was worth.
    """
    ptr = getattr(value, "data_ptr", None)
    if not callable(ptr):
        return ""
    is_contig = getattr(value, "is_contiguous", None)
    if callable(is_contig) and not is_contig():
        return "non_contiguous"
    try:
        if int(ptr()) % AOTI_ALIGNMENT:
            return "unaligned_16b"
    except (RuntimeError, ValueError):  # pragma: no cover — meta/fake tensors
        return ""
    return ""


class FeedAligner:
    """Owned aligned staging buffers for one runner's declared inputs.

    The fix pgw#791 asks for is "align once at ingress rather than copying
    per call", and for a value that CHANGES every call (the timestep does)
    the honest reading is: allocate once, copy the value. The allocation —
    which is what the runner repeats, along with a stderr write and an ATen
    dispatch — happens exactly once per (input, shape, dtype, device); after
    that the feed is a pointer-stable, correctly aligned buffer the call
    writes into. Pointer stability is also what cudagraph static inputs want,
    so this cannot cost the lane a capture later.
    """

    __slots__ = ("_buffers",)

    def __init__(self) -> None:
        self._buffers: Dict[str, Any] = {}

    def staged(self, name: str, value: Any) -> Any:
        """A 16-byte-aligned contiguous copy of ``value`` in an owned buffer."""
        import torch

        buf = self._buffers.get(name)
        if (buf is None or buf.dtype is not value.dtype
                or buf.device != value.device
                or tuple(buf.shape) != tuple(value.shape)):
            buf = torch.empty(
                tuple(value.shape), dtype=value.dtype, device=value.device)
            if int(buf.data_ptr()) % AOTI_ALIGNMENT:
                # torch's caching allocator hands out 512-byte-aligned blocks
                # (CPU: 64). If that ever stops being true, realigning by
                # allocation is not a fix and this must fail loudly rather
                # than quietly hand the runner another unaligned pointer.
                raise IngressContractError(
                    "realign_unavailable",
                    f"a freshly allocated buffer for input {name!r} is itself "
                    f"not {AOTI_ALIGNMENT}-byte aligned "
                    f"(ptr%{AOTI_ALIGNMENT}="
                    f"{int(buf.data_ptr()) % AOTI_ALIGNMENT}); the artifact's "
                    f"aligned-input contract cannot be satisfied here")
            self._buffers[name] = buf
        buf.copy_(value)
        return buf

    def buffered(self) -> Tuple[str, ...]:
        return tuple(sorted(self._buffers))


def aligned_feeds(
    contract: ArtifactContract,
    feeds: Sequence[Any],
    aligner: FeedAligner,
    report: Optional[Callable[[str, str], None]] = None,
) -> List[Any]:
    """``feeds`` with every out-of-contract input realigned in place (pgw#791).

    ``feeds`` is :func:`marshal_positional`'s output, so it is one value per
    declared input in ``position`` order — the same order this walks, which is
    what lets a reason NAME the input instead of an index.
    """
    specs = sorted(contract.inputs, key=lambda s: s.position)
    if len(specs) != len(feeds):
        raise IngressContractError(
            "feed_arity_mismatch",
            f"{len(feeds)} marshalled feed(s) for {len(specs)} declared "
            f"input(s); an aligned-ingress pass that cannot name its inputs "
            f"would realign the wrong slot")
    out = list(feeds)
    for idx, (spec, value) in enumerate(zip(specs, feeds)):
        reason = alignment_gap(value)
        if not reason:
            continue
        out[idx] = aligner.staged(spec.name, value)
        if report is not None:
            report(spec.name, reason)
    return out


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
    present = excluded_inputs_present(contract, kwargs)
    if present:
        raise IngressContractError(
            "input_excluded",
            f"this graph class REFUSES input(s) {list(present)!r}: the call "
            f"carries them, so it must be served by the class that declares "
            f"them (pgw#790 — a branchless class fed an adapter would return "
            f"the base model and look correct)")
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


def _load_package(path: Path, entry: str = "model") -> Any:
    """Load one NAMED model out of a ``.pt2`` (the sole torch entry point
    for the load path — tests substitute this). ``"model"`` is torch's own
    default name for a single-model package."""
    from torch._inductor.package import load_package

    return load_package(str(path), model_name=str(entry or "model"))


def _load_literals(path: Path, device: str) -> Dict[str, Any]:
    from safetensors.torch import load_file

    return dict(load_file(str(path), device=device))


def split_literals(literals: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split a namespaced literal payload into per-entry tables.

    Keys are ``<entry>::<fqn>`` (:data:`LITERAL_SEP`); a key with no
    namespace is refused by name — a literal the dispatch cannot attribute
    to an entry could bind to the wrong graph."""
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in literals.items():
        entry, sep, fqn = str(key).partition(LITERAL_SEP)
        if not sep or not entry or not fqn:
            raise ValueError(
                f"literal key {key!r} is not namespaced "
                f"'<entry>{LITERAL_SEP}<fqn>'")
        out.setdefault(entry, {})[fqn] = value
    return out


def resident_constants(module: Any) -> Dict[str, Any]:
    """Every resident tensor a ``state_dict``-sourced constant can bind to.

    ``module.state_dict()`` is NOT that set, and the difference is pgw#825:
    it omits NON-PERSISTENT buffers. The canonical LoRA branch pair is
    exactly that — ``w8a8_lora.alloc_branch_buffers`` registers
    ``lora_a``/``lora_b`` with ``persistent=False`` so a checkpoint never
    carries a zeroed adapter — yet ``torch.export`` lifts them as BUFFER
    inputs and AOTInductor declares them ``ConstantType::Buffer`` under their
    real FQN, i.e. ``source=state_dict``. A bind table built from
    ``state_dict()`` alone therefore declares 20 constants per sdxl
    ``BasicTransformerBlock`` that no lookup could ever resolve.

    ONE definition, used by the mint's bindability gate and by both arm
    sites, because a gate that models the arm differently from the arm is
    the pgw#816/#822 class: it either refuses a cell that would have served
    or admits one that cannot.

    Binding is by FQN and :func:`resolve_constants` reads only the DECLARED
    names, so the extra keys are inert for any artifact that does not want
    them.
    """
    out: Dict[str, Any] = dict(module.state_dict())
    try:
        buffers = list(module.named_buffers())
    except Exception:  # pragma: no cover — duck-typed owners in unit rigs
        return out
    for name, buf in buffers:
        if buf is not None:
            out.setdefault(str(name), buf)
    return out


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

    **MEASURED CONTRACT FACT (pgw#723 final pod, torch 2.13.0+cu130):**
    ``load_constants`` keys by the ORIGINAL FQN (``lin.bias``), NOT the
    mangled C++ identifier (``lin_bias``) that the package's own table
    answers with. Keying a hand-rolled binder by ``DeclaredConstant.name``
    fails with ``RuntimeError: Constant not found: lin_bias`` — and only at
    bind time, on a pod. This function keys by ``spec.fqn`` (the
    ``original_fqn`` the package records), which is the correct side of that
    split; do not "simplify" it to the C++ name, and do not hand-roll a
    binder that does.
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
    entry: str = ""
    bound: bool = False
    #: pgw#817/D3: True when :meth:`bind` bound BY REFERENCE. Recorded rather
    #: than inferred so an arm report can state whether N instances cost N
    #: weight copies or none.
    user_managed: bool = False
    bound_fqns: Tuple[str, ...] = ()
    calls: int = 0
    refusals: Dict[str, int] = field(default_factory=dict)
    #: pgw#791. ``"<input>/<reason>" -> count``; the typed event fires on the
    #: first of each, the count keeps the whole tax countable afterwards.
    realigned: Dict[str, int] = field(default_factory=dict)
    aligner: FeedAligner = field(default_factory=FeedAligner)
    #: Set by :func:`load_and_wrap` so the typed realignment event can name
    #: the cell it belongs to.
    family: str = ""

    def declared_fqns(self) -> Tuple[str, ...]:
        return tuple(s.fqn for s in self.constants)

    def excludes(self, names: Sequence[str]) -> bool:
        """True when this class refuses every one of ``names`` (pgw#790)."""
        wanted = set(str(n) for n in names)
        return bool(wanted) and wanted <= set(self.contract.excluded)

    def bind(
        self, state_dict: Mapping[str, Any], literals: Mapping[str, Any],
        *, user_managed: bool = False,
    ) -> None:
        """Bind constants from the resident weights + literal payload, then
        PROVE the artifact reports them all bound.

        Order is load-bearing: nothing may call the package before this
        returns, and this must not mark itself bound on a partial update.

        ``user_managed=True`` binds BY REFERENCE (pgw#812 D3): the artifact
        keeps pointers to the caller's tensors instead of copying them into
        its own constant buffer. The default False is right for a whole-graph
        cell — one duplicate of the weights, and the artifact owns its own
        lifetime. It is FATAL for a regional cell: N block instances each
        load their own runner, so a copying bind means N copies of that
        block's weights in VRAM (flux2: a second whole model). The caller
        that passes True is asserting that the tensors outlive this runner —
        which for a regional arm holds by construction, because the values
        come from the resident pipeline's own ``state_dict`` and the shim
        that calls the runner is installed ON that module.
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
        #
        # FOLDING CAVEAT CLOSED (pgw#723 residuals, torch 2.13.0+cu130 — the
        # prod floor): strictness also HOLDS against a genuinely FOLDING
        # artifact. AOTInductor folds ONLY under
        # ``aot_inductor.use_runtime_constant_folding=True`` (default never
        # folds — measured on 3 constant-expression module shapes AND the
        # real sdxl cell), and when it does, the ``_FOLDED_CONST_*`` entries
        # (``from_folded=True``) are EXEMPT from the full-update check and
        # RECOMPUTED from the freshly bound originals (mutate-arm proven:
        # binding values 3x off compile time tracks eager bit-exactly). So
        # strict is safe for both artifact classes. NOTE our own mint cannot
        # ship a folding artifact today: folded entries classify as literals
        # with no ``ep.constants`` value, so ``_write_literals`` refuses the
        # mint by name — and if runtime folding is ever deliberately enabled,
        # the change is to EXCLUDE from_folded constants from the manifest
        # and binding (they are derived), not to loosen this gate.
        #
        # pgw#817/D3: `user_managed` is passed only when asked for, so the
        # whole-graph path's call shape is byte-identical to what pgw#721/#723
        # measured on a pod. A torch whose `load_constants` has no such
        # parameter is a NAMED refusal rather than a silent copy — a regional
        # arm that silently copied would OOM the card N blocks later, which is
        # a far worse way to learn the same fact.
        if user_managed:
            try:
                self.package.load_constants(
                    values, check_full_update=True, user_managed=True)
            except TypeError as exc:
                if "user_managed" not in str(exc):
                    raise
                raise ConstantsUnboundError(
                    "user_managed_unsupported",
                    f"this torch's load_constants has no user_managed "
                    f"parameter, so every constant would be COPIED — for a "
                    f"regional cell that is one copy of the block weights per "
                    f"instance ({type(exc).__name__}: {exc})") from exc
        else:
            self.package.load_constants(values, check_full_update=True)
        self.user_managed = bool(user_managed)
        self.bound_fqns = tuple(sorted(values))
        self.bound = True

    def assert_ready(self) -> None:
        """The gate that keeps the segfault unreachable — per ENTRY: the
        unbound-call segfault was re-measured per named model (pgw#758)."""
        if not self.bound:
            raise ConstantsUnboundError(
                "constants_unbound",
                f"refusing to invoke code-only artifact entry "
                f"({self.entry or self.module_name or 'unknown'}) with "
                f"{len(self.constants)} unbound constant(s): calling before "
                f"load_constants segfaults the worker process")

    def _report_realigned(self, name: str, reason: str) -> None:
        """First occurrence of an (input, reason) is a typed hub-visible
        event; every occurrence is counted (pgw#791).

        Coalesced deliberately: the defect fires 28+ times per request, and a
        per-call event would be the stderr spam it replaces, on a wire that
        costs money. One event names the input; the counter carries the rest.
        """
        key = f"{name}/{reason}"
        seen = self.realigned.get(key, 0)
        self.realigned[key] = seen + 1
        if seen:
            return
        logger.warning(
            "aot-serve: input %r arrived %s for entry %r; realigning into an "
            "owned aligned buffer at ingress (the artifact would otherwise "
            "copy it on every call and report only on stderr)",
            name, reason, self.entry or self.module_name)
        activity_mod.emit_event(
            REALIGN_EVENT,
            f"family={self.family} entry={self.entry or self.module_name} "
            f"target={self.module_name} input={name}: {reason} — realigned at "
            f"ingress into an owned {AOTI_ALIGNMENT}-byte aligned buffer "
            f"(AOTInductor would otherwise copy it on every call)",
            phase=reason,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.assert_ready()
        try:
            assert_ingress(self.contract, args, kwargs)
            feeds = marshal_positional(self.contract, args, kwargs)
            # pgw#791: satisfy the artifact's ALIGNED-input contract here,
            # once, instead of letting the runner discover it per call.
            feeds = aligned_feeds(
                self.contract, feeds, self.aligner, self._report_realigned)
        except IngressContractError as exc:
            self.refusals[exc.reason] = self.refusals.get(exc.reason, 0) + 1
            raise
        out = self.package(*feeds)
        self.calls += 1
        return out


@dataclass
class EntryDispatch:
    """Every named entry of one cell that serves ONE target, behind one
    call site (pgw#758).

    Dispatch is the declared contract itself: the call routes to the entry
    whose ingress contract ADMITS it. No admitting entry is a named
    per-request refusal (the caller serves eagerly); more than one is
    ``entry_ambiguous`` — the declaration failed to discriminate two graph
    classes by ingress, which is a defect to surface, never a coin to flip.
    """

    runners: Tuple[Tuple[str, ArtifactRunner], ...]

    def bind(
        self, state_dict: Mapping[str, Any],
        literals: Mapping[str, Mapping[str, Any]],
        *, user_managed: bool = False,
    ) -> None:
        """Bind EVERY entry of this dispatch from one resident table.

        ``literals`` is keyed by ENTRY NAME (the shape
        :func:`split_literals` produces), because the literal payload is a
        property of a graph class, not of the module the class serves.

        This exists so the regional arm binds through the same code the
        whole-graph arm does: pgw#827 was one bind table built at the wrong
        SCOPE, and the fix must not introduce a second bind implementation
        that can drift from this one.
        """
        for name, runner in self.runners:
            runner.bind(
                state_dict, dict(literals.get(name, {}) or {}),
                user_managed=user_managed)

    def assert_ready(self) -> None:
        """B1 for every entry — an unbound one is a segfault, not a miss."""
        for _name, runner in self.runners:
            runner.assert_ready()

    @property
    def bound(self) -> bool:
        return bool(self.runners) and all(
            runner.bound for _n, runner in self.runners)

    @property
    def user_managed(self) -> bool:
        """True only when EVERY entry bound by reference — one copying entry
        is one copy of the block's weights per instance."""
        return bool(self.runners) and all(
            runner.user_managed for _n, runner in self.runners)

    def declared_fqns(self) -> Tuple[str, ...]:
        names: List[str] = []
        for _n, runner in self.runners:
            names.extend(runner.declared_fqns())
        return tuple(sorted(set(names)))

    def select(
        self, args: Sequence[Any], kwargs: Mapping[str, Any],
    ) -> Tuple[str, ArtifactRunner]:
        admitted: List[Tuple[str, ArtifactRunner]] = []
        reasons: List[str] = []
        for name, runner in self.runners:
            try:
                assert_ingress(runner.contract, args, kwargs)
            except IngressContractError as exc:
                reasons.append(f"{name}: {exc.reason}")
                continue
            admitted.append((name, runner))
        if not admitted:
            raise IngressContractError(
                "no_entry_admits",
                f"no packaged entry admits this call "
                f"({len(self.runners)} tried): {'; '.join(reasons[:6])}")
        if len(admitted) > 1:
            names = sorted(name for name, _ in admitted)
            raise IngressContractError(
                "entry_ambiguous",
                f"{len(admitted)} entries admit this call ({names[:6]!r}) — "
                f"the declaration does not discriminate these graph classes "
                f"by ingress contract")
        return admitted[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        _name, runner = self.select(args, kwargs)
        return runner(*args, **kwargs)

    @property
    def calls(self) -> int:
        return sum(runner.calls for _n, runner in self.runners)

    def refusal_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _n, runner in self.runners:
            for reason, count in runner.refusals.items():
                out[reason] = out.get(reason, 0) + count
        return out

    def realignment_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _n, runner in self.runners:
            for key, count in runner.realigned.items():
                out[key] = out.get(key, 0) + count
        return out

    def entry_calls(self) -> Dict[str, int]:
        """Per-entry served-call counts — which graph class actually served."""
        return {name: runner.calls for name, runner in self.runners}

    def excludes(self, names: Sequence[str]) -> bool:
        """True when some packaged entry REFUSES every one of ``names``.

        The adapter-free routing question, asked of the declaration rather
        than of a flag: is there a class in this cell that serves calls
        WITHOUT these inputs? (pgw#790)
        """
        wanted = set(str(n) for n in names)
        if not wanted:
            return False
        return any(wanted <= set(runner.contract.excluded)
                   for _n, runner in self.runners)


# ---------------------------------------------------------------------------
# Serve — module swap behind a fail-soft guard
# ---------------------------------------------------------------------------


def ingress_class_name(
    target: str, args: Sequence[Any], kwargs: Mapping[str, Any],
) -> str:
    """The DECLARED CLASS one refused call names (pgw#916).

    A shape alone is not a growable unit of work — a mint is asked for a
    class, and a class is the whole ingress coordinate of one target: every
    declared input, in position order, with its dtype and extents.  Rendering
    it here (rather than logging "a shape was refused") is what lets the
    growth path submit the exact thing that is missing, and what lets two
    pods agree they are missing the SAME thing.

    Deliberately independent of any artifact: the whole point is that the
    call arrived at a class no packaged entry declares, so there is no entry
    block to read it off.
    """
    parts: List[str] = []

    def render(name: str, value: Any) -> None:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is None:
            parts.append(f"{name}={value!r}"[:64])
            return
        extents = ",".join(str(int(d)) for d in tuple(shape))
        token = str(dtype or "").replace("torch.", "")
        parts.append(f"{name}={token}[{extents}]")

    for index, value in enumerate(args):
        render(f"#{index}", value)
    for name in sorted(kwargs):
        value = kwargs[name]
        if value is None or isinstance(value, (bool, int, float, str)):
            parts.append(f"{name}={value!r}")
            continue
        render(name, value)
    return f"{target}/{','.join(parts)}"


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


def adapter_call_kwargs(
    module: Any, runner: "ArtifactRunner | EntryDispatch",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(artifact kwargs, eager kwargs)`` for one call's adapter inputs.

    pgw#790's routing rule, in one place. The two differ in exactly one case:
    the denoiser carries a lifted adapter, NO adapter is currently active, and
    the cell packages a class that REFUSES the adapter inputs. Then the
    artifact call omits the pair so the BRANCHLESS class admits it, while the
    eager fallback still receives it — ``bind_views`` refuses a ``None`` half
    by name, so the eager path needs the pair exactly as much as a
    branch-bearing artifact does.

    Nothing here is a new piece of state. "Is an adapter active" is
    ``w8a8_lora``'s own ``_cozy_lora_active``, written by the per-request
    attach/clear path (``utils.lora``) since gw#627; this only reads it. And
    it decides only WHICH pre-compiled class serves the call — never the
    shape of a program, never a baked constant. Both classes return the same
    tensor for adapter-free traffic (a zeroed B adds exactly 0), so the
    routing is a cost decision whose correctness does not depend on it.
    """
    lifted = lifted_call_kwargs(module)
    if not lifted:
        return {}, {}
    if lora_lifted.adapter_active(module):
        return lifted, lifted
    if not runner.excludes(lora_lifted.LIFTED_INPUT_NAMES):
        # One branch-bearing class only: it declares the pair as MANDATORY,
        # so withholding it would refuse every request by name.
        return lifted, lifted
    return {}, lifted


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
    if lifted and wanted <= set(contract.excluded):
        # pgw#790: the BRANCHLESS arm of an adapter-forked cell. It says so
        # explicitly — the adapter inputs are refused, not forgotten — so the
        # dispatch can never route adapter-bearing traffic to it and "the
        # branch was traced away" is impossible to reach silently.
        return
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
    runner: "ArtifactRunner | EntryDispatch",
    meta: Dict[str, Any],
    *,
    attr: str = "forward",
    target: str = "",
    eager_forward: Optional[Callable[..., Any]] = None,
) -> None:
    """Swap ``module.<attr>`` for the cell's dispatch behind a fail-soft
    guard.

    Mirrors ``trt_engine.wrap_module``: the first artifact ERROR
    synchronously revokes scheduler-visible compiled proof and permanently
    routes to eager; the module object (config, dtype, device, weights)
    stays untouched, and its weights remain the constant-binding source.
    ``runner`` is one target's :class:`EntryDispatch` (or a bare
    :class:`ArtifactRunner` in focused tests — both are callable and count
    calls); ``attr`` generalizes the swap beyond ``forward`` for dotted
    targets like ``vae.decode`` (pgw#758).

    An :class:`IngressContractError` is NOT such an error. It is a named,
    counted, per-request contract refusal — the request serves eagerly and
    the artifact stays armed for in-contract traffic, because one
    out-of-range request (or an entry-dispatch miss/ambiguity) says nothing
    about the artifact's health.

    pgw#725 LoRA seam: when the denoiser carries a lifted adapter, both
    ``lora_a``/``lora_b`` are MANDATORY call kwargs and this wrapper is the
    call site that supplies them — see :func:`lifted_call_kwargs`.
    """
    attr = str(attr or "forward")
    label = target or f"{meta.get('family')}.{attr}"
    original = eager_forward or getattr(module, attr)
    state: Dict[str, Any] = {
        "failed": False,
        "successful_calls": 0,
        "ingress_refusals": 0,
        "last_refusal": "",
        "original": original,
        "attr": attr,
        "target": label,
        "failure_callback": None,
        "refusal_callback": None,
        "revocation_error": "",
        "runner": runner,
    }

    def aot_forward(*args: Any, **kwargs: Any) -> Any:
        if state["revocation_error"]:
            raise CompiledExecutionLaneUnavailableError(state["revocation_error"])
        # The lifted pair is resolved PER CALL, not captured at wrap time:
        # the LoRA lane may install/remove lifting independently of arming,
        # and a stale capture would either starve the graph of a mandatory
        # input or feed one to a forward that no longer takes it.
        # pgw#790: an adapter-FREE call omits the pair when the cell packages
        # a branchless class, so the dispatch routes it to the graph that does
        # not spend 32-45% of its compiled forward on arithmetic over zeros.
        # The eager fallback always receives it.
        artifact_lora, eager_lora = adapter_call_kwargs(module, runner)
        eager_kwargs = {**kwargs, **eager_lora}
        kwargs = {**kwargs, **artifact_lora}
        if state["failed"]:
            return original(*args, **eager_kwargs)
        try:
            out = runner(*args, **kwargs)
            # Envelope parity (live-named on the 0.76.x rerun line,
            # 2026-07-29): the exported graph returns the RAW tensor, but a
            # diffusers pipeline calls `unet(..., return_dict=False)[0]` —
            # indexing a bare tensor silently slices the batch dim and the
            # crash surfaces downstream as a broadcast error. Restore the
            # caller's declared envelope: return_dict=False means a 1-tuple.
            if kwargs.get("return_dict") is False and not isinstance(
                    out, tuple):
                return (out,)
            return out
        except IngressContractError as exc:
            # B2: the exported graph would have run this and returned
            # unvalidated output. Named refusal + eager service.
            state["ingress_refusals"] = int(state["ingress_refusals"]) + 1
            state["last_refusal"] = f"{exc.reason}: {exc}"
            logger.warning(
                "aot-serve: %s REFUSED out-of-contract input (%s: %s); "
                "serving this request eager, artifact stays armed",
                label, exc.reason, exc)
            activity_mod.emit_event(
                "aot_ingress_refused",
                f"family={meta.get('family')} target={label}: {exc}",
                phase=exc.reason,
            )
            report_ingress_refusal(state, exc.reason, str(exc))
            # pgw#916: the refusal is also a SHAPE GAP — the armed cell does
            # not cover this declared class, and on the AOT arm nothing was
            # ever going to grow it (hot_swap.enable returns False without a
            # dynamo router, so the executor's three growth call sites are
            # no-ops on every AOT arm). Named, counted, and submitted through
            # the one arm-agnostic growth module.
            shape_growth.report_and_submit(shape_growth.ShapeGap(
                arm=shape_growth.ARM_AOT,
                family=str(meta.get("family") or ""),
                target=label,
                declared_class=ingress_class_name(label, args, kwargs),
                reason=exc.reason,
                detail=str(exc)[:400],
                cell_key=str(meta.get("cell_key") or ""),
            ))
            return original(*args, **eager_kwargs)
        except ConstantsUnboundError as exc:
            # Reaching here means the arm order was violated. The gate did
            # its job (no segfault); the lane is structurally unusable.
            state["failed"] = True
            logger.error(
                "aot-serve: %s invoked with unbound constants (%s); eager for "
                "the rest of this process", label, exc)
            activity_mod.emit_event(
                "aot_constants_unbound",
                f"family={meta.get('family')} target={label}: {exc}",
                phase=exc.reason,
            )
            _revoke(state, f"constants unbound: {exc}")
            return original(*args, **eager_kwargs)
        except Exception as exc:  # noqa: BLE001 — ANY artifact problem => eager
            state["failed"] = True
            detail = (
                f"AOTI artifact {label} failed: "
                f"{type(exc).__name__}: {exc}")
            _revoke(state, detail)
            logger.warning(
                "aot-serve: %s failed (%s: %s); eager for the rest of this "
                "process", label, type(exc).__name__, exc)
            return original(*args, **eager_kwargs)

    def _counted_forward(*args: Any, **kwargs: Any) -> Any:
        out = aot_forward(*args, **kwargs)
        state["successful_calls"] = int(runner.calls)
        return out

    setattr(module, attr, _counted_forward)
    setattr(module, _MARKER_ATTR, {
        "meta": {k: meta.get(k) for k in ("sku", "torch", "precision")},
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
        raise CompiledExecutionLaneUnavailableError(state["revocation_error"]) from callback_exc


def report_ingress_refusal(state: Dict[str, Any], reason: str, detail: str) -> None:
    """Hand ONE per-request ingress refusal to the arming brain (pgw#844).

    The refusal already rides ``aot_ingress_refused`` as a countable typed
    event, but that event names no REQUEST — so the request row that was
    served eager by an armed compiled lane still reported
    ``serving_mode=aot_cell, fallback_reason=""``, i.e. an eager latency
    sample counted as compiled. That is the exact contamination
    :mod:`serving_mode` exists to prevent, and arming a partially
    dispatchable cell (which pgw#844 now does) is what makes it common.

    Never raises: telemetry must not be able to un-serve a request that the
    eager fallback is about to answer correctly.
    """
    callback = state.get("refusal_callback")
    if not callable(callback):
        return
    try:
        callback(str(reason or ""), str(detail or ""))
    except Exception:  # noqa: BLE001 — a reporting failure is not a serve failure
        logger.debug("aot-serve: ingress-refusal report failed", exc_info=True)


def set_ingress_refusal_callback(pipeline: Any, callback: Any) -> bool:
    """Bind per-request eager-fallback accounting to every wrapped target."""
    states = _marker_states(pipeline)
    if not states:
        return False
    for state in states:
        state["refusal_callback"] = callback
    return True


def _target_owner(pipeline: Any, target: str) -> Tuple[Any, str]:
    """``(owner module, attribute)`` one entry target names on a pipeline:
    ``"unet"`` -> ``(pipeline.unet, "forward")``; ``"vae.decode"`` ->
    ``(pipeline.vae, "decode")``; ``"vae.decoder"`` ->
    ``(pipeline.vae.decoder, "forward")``.

    pgw#967: this delegates to ``compile_cache._resolve_target`` — ONE
    resolver, because the two disagreed on a dotted target whose leaf is a
    MODULE. This function partitioned on the FIRST dot and treated the rest
    as a plain attribute, so ``vae.decoder`` resolved to
    ``(pipeline.vae, "decoder")`` and the arm would have replaced a
    SUBMODULE with a wrapped function, while the mint resolved the same
    string to the decoder's ``forward`` and exported that. A cell whose
    entries are traced from one callable and armed onto another is the
    silent-wrongness class every gate in this stack exists to prevent, and
    nothing would have caught it: ``vae.decode`` (a bound method) is the
    only dotted target any family has ever named, and for it the two agreed.
    """
    resolved = _resolve_target(pipeline, str(target))
    if resolved is None:
        raise AdoptError(
            "no_target", f"pipeline has no callable target {target!r}")
    module, attr, _fn = resolved
    return module, attr


def load_and_wrap(
    pipeline: Any, cfg: Any, artifact: Path, cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stage + verify + load EVERY named entry + BIND all constants, then
    perform the live wraps (pgw#758: one resident artifact serves every
    declared class, across every declared target).

    Raises :class:`AdoptError` with a classified reason on any failure, and
    never publishes extracted files into a shared live cache. ALL entries
    bind before ANY wrap: a cell that cannot arm one of its graph classes
    arms none of them — a partially served contract would be a silent
    subset of what the cell key advertises.
    """
    family = str(getattr(cfg, "family", "") or "")
    staged = stage_artifact(Path(artifact), family, cache_dir=cache_dir)
    try:
        meta = staged.metadata
        try:
            entries = entries_from_meta(meta)
        except ValueError as exc:
            raise AdoptError("contract_invalid", str(exc)) from exc

        # Group the entries by target and resolve every owner module first —
        # an unresolvable target must refuse before any package is dlopen'd.
        groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for name in sorted(entries):
            block = entries[name]
            groups.setdefault(str(block.get("target") or ""), []).append(
                (name, block))
        owners: Dict[str, Tuple[Any, str]] = {
            target: _target_owner(pipeline, target) for target in groups}

        # Re-arm: a previous wrap's eager originals must be preserved per
        # target, or unwrap after a re-arm restores a wrapped callable.
        eager_originals: Dict[str, Callable[..., Any]] = {}
        old_marker = getattr(pipeline, _MARKER_ATTR, None)
        if old_marker is not None:
            old_targets = old_marker.get("targets") or {}
            for target, row in old_targets.items():
                old_state = row.get("state") or {}
                original = old_state.get("original")
                if row.get("module") is not owners.get(target, (None,))[0] \
                        or not callable(original):
                    raise AdoptError(
                        "old_marker_invalid",
                        f"existing AOT marker does not retain target "
                        f"{target!r}'s eager callable")
                eager_originals[target] = original

        t0 = time.monotonic()
        literals_by_entry: Dict[str, Dict[str, Any]] = {}
        literals_path = staged.root / LITERALS_NAME
        if literals_path.exists():
            first_module = next(iter(owners.values()))[0]
            device = str(getattr(first_module, "device", "cuda"))
            try:
                literals_by_entry = split_literals(
                    _load_literals(literals_path, device))
            except ValueError as exc:
                raise AdoptError("contract_invalid", str(exc)) from exc

        # Load + bind EVERY entry before the first live mutation.
        dispatches: Dict[str, EntryDispatch] = {}
        total_constants = 0
        for target, rows in groups.items():
            module, _attr = owners[target]
            state_dict = resident_constants(module)
            runners: List[Tuple[str, ArtifactRunner]] = []
            for name, block in rows:
                try:
                    contract = contract_from_meta(block)
                    constants = constants_from_meta(block)
                    # pgw#725: the lifted-adapter signature must match the
                    # module's actual lifted state, per entry.
                    assert_lifted_contract(module, contract)
                except ValueError as exc:
                    raise AdoptError(
                        "contract_invalid", f"entry {name!r}: {exc}") from exc
                package = _load_package(staged.root / PACKAGE_NAME, name)
                runner = ArtifactRunner(
                    package=package, contract=contract, constants=constants,
                    module_name=target, entry=name, family=family)
                try:
                    runner.bind(state_dict, literals_by_entry.get(name, {}))
                except ConstantsUnboundError as exc:
                    raise AdoptError(
                        f"constants_{exc.reason}",
                        f"entry {name!r}: {exc}") from exc
                total_constants += len(constants)
                runners.append((name, runner))
            dispatches[target] = EntryDispatch(tuple(runners))

        # First live mutation. Everything above is proven for EVERY entry:
        # complete artifact, matching runtime key, resolved targets, loaded
        # named models, constant tables proven bound against the manifests.
        target_rows: Dict[str, Dict[str, Any]] = {}
        for target, dispatch in dispatches.items():
            module, attr = owners[target]
            wrap_module(
                module, dispatch, meta, attr=attr, target=target,
                eager_forward=eager_originals.get(target))
            module_marker = getattr(module, _MARKER_ATTR, {})
            target_rows[target] = {
                "module": module,
                "attr": attr,
                "state": module_marker.get("state", {}),
            }
        setattr(pipeline, _MARKER_ATTR, {"meta": meta, "targets": target_rows})
        logger.info(
            "aot-serve: loaded+bound %d entr%s across %d target(s) in %.1fs "
            "(%d declared constants, combined_graph_hash=%s)",
            len(entries), "y" if len(entries) == 1 else "ies", len(groups),
            time.monotonic() - t0, total_constants,
            meta.get("combined_graph_hash"))
        return meta
    finally:
        staged.close()


def _adopt_identity(artifact: Path) -> str:
    """Best-effort ``family=… key=…`` from the artifact's own metadata for
    the typed adopt event — a refusal must name the candidate cell even when
    the refusal itself is a metadata problem."""
    try:
        with tarfile.open(artifact, mode="r:*") as tar:
            for member in tar:
                if member.name == METADATA_NAME and member.isfile():
                    src = tar.extractfile(member)
                    assert src is not None
                    meta = json.loads(src.read().decode())
                    return (f"family={meta.get('family')} "
                            f"key={meta.get('cell_key')}")
    except Exception:  # noqa: BLE001 — identity is best-effort by contract
        pass
    return f"artifact={artifact.name}"


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
) -> AdoptOutcome:
    """Consumer entry point: verify + load + bind + swap an AOTI artifact.

    Falsy (staying eager) on ANY miss — the caller's ordinary miss policy
    (fleet self-mint / eager / typed refusal) takes over, exactly as for a TRT
    engine. Truthy IS the HIT: ``fleet_cells`` treats it as a genuine match and
    skips the self-mint.

    pgw#923: the outcome is RETURNED rather than narrated. The classified
    refusal reason used to leave this function only as the ``phase`` of a
    free-text ``aot_adopt`` event, which is why the adoption that actually
    happens on every boot had no measured row anywhere — the caller could not
    see what it had just been told.
    """
    if artifact is None:
        return AdoptOutcome.miss("no_artifact")
    try:
        meta = load_and_wrap(pipeline, cfg, Path(artifact), cache_dir=cache_dir)
    except Exception as exc:
        reason = str(getattr(exc, "reason", "") or "") or type(exc).__name__
        identity = _adopt_identity(Path(artifact))
        logger.warning(
            "aot-serve: artifact unusable (%s: %s); staying eager",
            reason, exc)
        return AdoptOutcome.miss(
            reason, f"{identity}: {type(exc).__name__}: {exc}", identity)
    logger.info(
        "aot-serve: armed %s [%d entr%s] (sku=%s torch=%s precision=%s, "
        "constants bound from resident weights)",
        meta.get("family"), len(meta.get("entries") or {}),
        "y" if len(meta.get("entries") or {}) == 1 else "ies",
        meta.get("sku"), meta.get("torch"), meta.get("precision"))
    return AdoptOutcome.hit(
        f"family={meta.get('family')} key={meta.get('cell_key')} "
        f"entries={len(meta.get('entries') or {})} sku={meta.get('sku')} "
        f"torch={meta.get('torch')} precision={meta.get('precision')}")


def _marker_states(pipeline: Any) -> List[Dict[str, Any]]:
    """Every wrapped target's state dict on a pipeline marker (format-2
    multi-target markers plus the legacy single-``state`` shape tests use)."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    rows = marker.get("targets")
    if isinstance(rows, dict) and rows:
        return [row.get("state") or {} for row in rows.values()]
    state = marker.get("state")
    return [state] if isinstance(state, dict) and state else []


def execution_count(pipeline: Any) -> int:
    """Successful artifact calls observed on this exact wrapped pipeline,
    summed over every wrapped target."""
    total = 0
    for state in _marker_states(pipeline):
        runner = state.get("runner")
        if runner is not None:
            total += int(getattr(runner, "calls", 0))
        else:
            total += int(state.get("successful_calls", 0))
    return total


def ingress_refusals(pipeline: Any) -> int:
    """Out-of-contract calls refused by name on this pipeline (B2),
    summed over every wrapped target."""
    return sum(
        int(state.get("ingress_refusals", 0))
        for state in _marker_states(pipeline))


def realigned_inputs(pipeline: Any) -> Dict[str, int]:
    """``"<input>/<reason>" -> count`` of ingress realignments (pgw#791).

    Zero is the contract holding. Non-zero is the tax MEASURED rather than
    inferred from a stderr line nobody can read — and it is paid at ingress
    into an owned buffer, not by the runner per call.
    """
    out: Dict[str, int] = {}
    for state in _marker_states(pipeline):
        runner = state.get("runner")
        counts = getattr(runner, "realignment_counts", None)
        rows = counts() if callable(counts) else getattr(runner, "realigned", {})
        for key, count in dict(rows or {}).items():
            out[key] = out.get(key, 0) + int(count)
    return out


def served_entry_calls(pipeline: Any) -> Dict[str, int]:
    """``entry -> served calls`` across every wrapped target (pgw#790).

    Which GRAPH CLASS served, not just that something did: an adapter-forked
    cell is only doing its job when adapter-free traffic lands on the
    branchless entry.
    """
    out: Dict[str, int] = {}
    for state in _marker_states(pipeline):
        runner = state.get("runner")
        getter = getattr(runner, "entry_calls", None)
        rows = getter() if callable(getter) else {}
        for name, count in dict(rows or {}).items():
            out[name] = out.get(name, 0) + int(count)
    return out


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


def armed_targets(pipeline: Any) -> Dict[str, Dict[str, Any]]:
    """Every wrapped target of an armed pipeline, keyed by target name.

    Each row carries ``module``, ``attr`` and the wrap ``state`` — whose
    ``original`` is the EAGER callable the cell replaced and whose ``runner``
    is that target's :class:`EntryDispatch`. The pgw#868 numerics probe needs
    exactly those two to run the cell and its own reference on one feed, and a
    private marker read from another module would be a second interpretation
    of this one's format. ``{}`` when nothing is armed.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    rows = marker.get("targets")
    if not isinstance(rows, dict):
        return {}
    return {str(name): dict(row) for name, row in rows.items()}


def armed_metadata(pipeline: Any) -> Dict[str, Any]:
    """The metadata the ARM itself used, off the live marker.

    The authority for anything asked about an armed cell: a caller that
    re-unpacked the artifact could be reading a different file than the one
    :func:`load_and_wrap` staged, verified and bound.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    meta = marker.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def is_armed(pipeline: Any) -> bool:
    """Whether the AOTI cell is currently serving this pipeline — EVERY
    wrapped target must still be live: one revoked target means the cell no
    longer serves the contract its key advertises."""
    states = _marker_states(pipeline)
    return bool(states) and not any(s.get("failed", False) for s in states)


def set_guard_failure_callback(pipeline: Any, callback: Any) -> bool:
    """Bind scheduler-state revocation to every wrapped target's guard."""
    states = _marker_states(pipeline)
    if not states:
        return False
    for state in states:
        state["failure_callback"] = callback
    return True


def unwrap(pipeline: Any) -> bool:
    """Restore every wrapped target's eager callable — rotation/eviction
    and the unproven-adoption rollback both go through here, same as the
    TRT lane."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    rows: List[Dict[str, Any]] = []
    targets = marker.get("targets")
    if isinstance(targets, dict) and targets:
        rows = list(targets.values())
    elif marker.get("module") is not None:
        rows = [{
            "module": marker.get("module"),
            "attr": (marker.get("state") or {}).get("attr", "forward"),
            "state": marker.get("state") or {},
        }]
    restored = False
    for row in rows:
        module = row.get("module")
        state = row.get("state") or {}
        original = state.get("original")
        if module is None or not callable(original):
            continue
        setattr(module, str(row.get("attr") or "forward"), original)
        try:
            delattr(module, _MARKER_ATTR)
        except AttributeError:
            pass
        restored = True
    if restored:
        try:
            delattr(pipeline, _MARKER_ATTR)
        except AttributeError:
            pass
    return restored


__all__ = [
    "AdoptOutcome",
    "ARTIFACT_FORMAT",
    "ARTIFACT_KIND",
    "ArtifactContract",
    "ArtifactRunner",
    "ConstantSpec",
    "ConstantsUnboundError",
    "DECLARED_AXES",
    "EntryDispatch",
    "IDENTITY_AXES",
    "IngressContractError",
    "InputContract",
    "LITERAL_SEP",
    "LITERALS_NAME",
    "METADATA_NAME",
    "PACKAGE_NAME",
    "SOURCE_LITERAL",
    "SOURCE_STATE_DICT",
    "armed_metadata",
    "armed_targets",
    "artifact_metadata",
    "assert_bindable",
    "assert_lifted_contract",
    "assert_ingress",
    "bind_call_inputs",
    "class_hash",
    "combined_graph_hash",
    "constants_from_meta",
    "contract_from_meta",
    "enable",
    "entries_from_meta",
    "execution_count",
    "proven_since",
    "find_artifact",
    "flavor_label",
    "host_isa_reason",
    "NO_HOST_ISA_STAMP",
    "ingress_class_name",
    "ingress_refusals",
    "is_aot_artifact",
    "is_aot_ref",
    "is_armed",
    "lifted_call_kwargs",
    "load_and_wrap",
    "marshal_positional",
    "note_aot_key",
    "pack",
    "range_digest",
    "resident_constants",
    "resolve_constants",
    "runtime_key",
    "set_guard_failure_callback",
    "set_ingress_refusal_callback",
    "report_ingress_refusal",
    "split_literals",
    "torch_maj_min",
    "torch_version",
    "unpack",
    "unpack_metadata",
    "unwrap",
    "verify",
    "verify_contract",
    "verify_declared",
    "verify_package_compute_capability",
    "wrap_module",
]
