"""AOTInductor ``.pt2`` compiled artifacts (pgw#721) — the THIRD producer/
consumer on the compile-cache rails (gw#384 / th#569 / #390).

``compile_cache`` serves the dynamo lane (a JIT warmed from seeded FX
entries); this module serves ``torch.export`` ->
``aoti_compile_and_package`` artifacts. Same trust model, storage,
delivery, and arming seam — cells live as flavors
of ``root/family-<family>``::

    root/family-<f>#aot-<sku>-torch<maj.min>-<precision>

Artifact = deterministic ``.tar.gz`` (the receipts gate reads
``metadata.json`` straight out of the digested bytes)::

    metadata.json           kind/format, runtime key (sm, torch, cuda + sku),
                            family, cell_key, and the ENTRY block — the ONE
                            NAMED GRAPH CLASS this artifact carries, with
                            its target, fork/class-dim coordinate, INPUT
                            CONTRACT, SYMBOL RANGES, declared CONSTANT
                            manifest, and class hash
    model.pt2               ONE AOTI package holding that entry as its named
                            model (``data/aotinductor/<entry>/``) — CODE ONLY
    constants.safetensors   optional: non-weight lifted constants, keys
                            namespaced ``<entry>::<fqn>``

Format 3 — the atom is ONE GRAPH CLASS (pgw#1176, Paul-directed)
----------------------------------------------------------------
Format 2 packed EVERY declared class into one artifact under one key, and
made identity, adoption, durability, verification, arming and advertisement
the same 36-entry unit. That unit is what forbade the incremental
compile-and-adopt Paul asked for, forced ~32 GiB of all-runners-resident
arming, and destroyed a 1 h 37 m mint when the 36th entry segfaulted.

Format 3 is one entry per artifact. What used to be "a cell" is a derived
CONTRACT MANIFEST (``cell_key.manifest_digest``) — a view, never a thing you
download, verify or arm. Entries accrete: each arms whole or not at all, and
an entry IS one graph, so that is atomic by nature. Serve-side dispatch is
unchanged in kind — it was already built at the right granularity: the call
routes to the entry whose DECLARED ingress contract admits it, zero
admitting entries is a named refusal (eager service), and more than one is
``entry_ambiguous``. What changed is that :class:`EntryDispatch` is a
REGISTRY entries join as they arm, not a frozen tuple built from a complete
cell. Every pgw#704 gate (B1 constants-bound, B2 ingress) holds PER ENTRY —
the unbound-entry segfault was re-measured per named model on the pin.

**Truthfulness is structural.** The pod never claims "cell X armed"; it
reports per-entry serve state. There is no unit left that CAN advertise more
than it serves, so the old all-or-nothing invariant ("a cell that cannot arm
one of its graph classes arms none of them") is not weakened — it is
vacuous.

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

import contextlib
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
    Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional,
    Sequence, Tuple, Union,
)

from . import activity as activity_mod
from . import aot_flatten
from . import aot_identity
from . import artifact_meta
from . import boot_phases
from . import cell_key as cell_key_mod
from . import host_isa
from .cell_adopt import AdoptOutcome
from . import serve_posture
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
from .models.memory import flush_memory, is_cuda_oom

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
#: pgw#1074: a rank-0 input arrived in an INTEGER dtype where the graph is
#: specialized on float32/float64, and this ingress recast it. Same shape and
#: same doctrine as :data:`REALIGN_EVENT` — the ingress normalizes the feed to
#: the artifact's contract and SAYS so, once per (entry, input, dtype pair).
RECAST_EVENT = "aot_input_recast"
#: AOTInductor generates its aligned-input fast path at 16 bytes
#: (``torch._inductor.codegen.aoti_runtime``'s ALIGNMENT). An input whose
#: ``data_ptr()`` is not a multiple of this — diffusers hands the denoiser
#: ``timesteps[i]``, a scalar VIEW at an odd element offset — makes the
#: runner clone it per call. Not a knob: it is the compiler's constant.
AOTI_ALIGNMENT = 16
#: THE compiled-graph artifact metadata/package version. v1 = ONE graph class
#: per artifact: the metadata carries one ``entry`` block, never an ``entries``
#: map.
#:
#: DESIGN-RULINGS §1.38b (Paul, 2026-08-13): *"we are pre-launch, so we should
#: be on v1 for everything, including our compiled-graph format."* The version
#: records the FIRST CONTRACT WE SHIP, not the number of abandoned internal
#: prototypes — carrying the old `ARTIFACT_FORMAT = 3` forward would turn
#: pre-launch history into a permanent public constraint for no benefit. The
#: surviving compiled-graph boundaries each begin at their own v1 and are
#: INDEPENDENT: `cg-key-v1` (key grammar), this (artifact metadata/package),
#: the manifest schema, the local CAS record schema, and the hub's
#: resolve/publish route. A future artifact v2 does not make the key grammar
#: v2.
#:
#: HARD CUT, not compatibility: formats 1/2/3 of the deleted cell/bundle
#: implementations have no reader here. There is no 3->1 mapping and no
#: accepted set — an artifact of a retired format is REFUSED BY NAME
#: (`verify_declared`), which is what makes a v1 reader unable to consume one.
#: The window is real and was measured before this shipped: `cell_store` held
#: 0 rows and no `cg-key-v1` object existed anywhere durable.
#:
#: The name is QUALIFIED on purpose. §1.38b: *"use qualified names such as
#: `compiled_graph_format`, never the generic `format` that let pgw#1230
#: compare the AOT package schema with an unrelated torch-inductor
#: semantic-cache epoch."* That collision cost a fully compiled 4-class mint.
COMPILED_GRAPH_FORMAT = 1

#: The metadata KEY the value above is stamped under, and the arm axis name.
#: One symbol so the stamp and the comparison cannot drift into two spellings
#: — the pgw#1230 failure mode, one level up.
COMPILED_GRAPH_FORMAT_KEY = "compiled_graph_format"
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
#: pgw#1080: a constant AOTInductor COMPUTES for itself at load, from the
#: constants that were bound (`_FOLDED_CONST_*`, produced by the runtime
#: constant-folding pass). Nothing binds it and nothing ships its bytes — it
#: is neither a weight nor a literal, and treating it as either is a refusal
#: for a value that is not missing. Weightless mints (pgw#1080) defer folding
#: to load precisely so a rebindable weight's VALUE is never compiled in, so
#: this class exists wherever that fence is armed.
SOURCE_COMPUTED = "computed"

#: The hardware/toolchain axes an ``.pt2`` is genuinely pinned to (pgw#765).
#: ``sm`` is the GPU identity: AOTInductor itself keys on
#: ``AOTI_COMPUTE_CAPABILITY`` — capability, never the marketing name
#: (``codecache.get_device_information``). ``sku`` is deliberately ABSENT:
#: Paul's ruling ("AOT cells are locked into the sm_x version, not the actual
#: GPU"), the pgw#691 collapse that removed it from cell identity on
#: byte-identical evidence, and the pgw#754 ISA clamp that made the host half
#: portable. It stays in metadata for observability — never as a refusal.
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
    """Full torch version (``2.13.0+cu130``) — the artifact is ABI-locked
    to it."""
    try:
        import torch

        return str(torch.__version__)
    except Exception:
        return ""


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


# Stamped cell keys this process LEARNED name aot-inductor artifacts
# (pgw#722 F1 discovery). Published AOT cells ride the same key space as
# their store flavor — indistinguishable from a dynamo cell's flavor by
# string shape alone — so every reader of a stamped envelope registers the
# key it learned here and :func:`is_aot_ref` consults the set. Without this
# the executor's kind dispatch (#734/#735) would score an armed ``.pt2`` by
# FX cache hits and disprove every honest adoption.
#
# THE RULE (pgw#1033): whoever reads a ``cell_key`` off an ``aot-inductor``
# envelope registers it. There are two such readers on the serving path —
# the delivered/named-cell arm and
# ``fleet_cells.adopt_delegated_mint`` (this pod's OWN mint). Only the first
# registered, so a self-minted cell — the one artifact this process is
# certain is exported — was the one ref ``is_aot_ref`` did not recognize.
#
# pgw#1141b: that rule was a CONVENTION, and the ORDERED arm route — the one
# §4.27 boot-adopt and every hub Plan take — never kept it. `arm_ordered`
# verifies the receipt and calls `provision.arm_aot` directly, so a
# boot-adopted cell wrapped itself onto a live pipeline while `is_aot_ref`
# still answered False for its ref, and every reader asking "is this the
# exported lane?" scored it on the DYNAMO lane's cache-hit ledger, which no
# AOTI artifact can move. Measured on a real pod (0.111.0, POD PROOF #4):
# `functions=()` -> `target_applicability_incomplete` ->
# `armed_target_unresolved` -> eager for life. The registration is now made
# by :func:`load_and_wrap` at the wrap itself — the one place every route
# passes and the moment the fact becomes true — so no future arm route can
# forget it.
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

    ONE recognizer: the stamped cell keys this process learned via
    :func:`note_aot_key`. pgw#1035 deleted the second, a
    ``flavor.startswith("aot-")`` label sniff — the only producer of that label
    form was ``aot_serve.flavor_label``, which had no caller and is gone. A cell
    ref carries a stamped KEY, so a label branch could only ever have matched a
    string this codebase no longer writes.
    """
    fam, flavor = parse_cell_ref(ref)
    if not fam or (family and fam != family):
        return False
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
        if source not in (SOURCE_STATE_DICT, SOURCE_LITERAL, SOURCE_COMPUTED):
            raise ValueError(
                f"constant {fqn!r} has unknown source {source!r} "
                f"(expected {SOURCE_STATE_DICT!r}, {SOURCE_LITERAL!r} or "
                f"{SOURCE_COMPUTED!r})")
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
    """Canonical digest of one entry's DECLARED ENVELOPE slice — the input
    ranges the entry admits.

    Owed to the exact-identity lane (pgw#716/#717): declared dim ranges live
    in ``ep.range_constraints``, NOT in the graph nodes — three exports
    differing only in declared range produced the identical node-only
    digest. A node-only ``graph_hashes`` therefore collides artifacts whose
    declared envelopes differ. Folding THIS digest into the per-class hash
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
            # part of the declared envelope (two classes that take the same
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
    # pgw#790: the NEGATIVE half of the declared envelope. Two
    # classes that differ only in what they REFUSE declare different envelopes,
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
    interface block, the node-level ``graph_witness`` body digest, and the
    trace-mode/lora facts. 16-hex, recomputable from the entry block alone —
    so a consumer can prove the stamp and a mismatch NAMES the class (the
    receipts principle).

    ``graph_witness`` (v3, pgw#1031): the node-level digest of the traced
    program (``graph_hash.graph_hash``, recorded on every keying block by
    ``aot_mint.keying_block``). Before v3 this axis folded only the graph
    INTERFACE (``graph``) — the traced ingress identity — so two endpoints
    whose declarations agreed while their bodies differed shared a key
    (measured 2026-08-10: ``micro-pad32`` 112 nodes vs ``micro-pad32-branchy``
    102 nodes, byte-identical keying block, one key, two artifacts). Folding
    the witness here makes the key sound BY CONSTRUCTION: two different bodies
    key apart, a collision becomes a MISS (eager + mint), which is the cheap
    outcome. The witness stays recorded as a top-level sibling for the adopt
    backstop (``aot_identity.verify_graph_witness``) — defense-in-depth. The
    fold is tolerant of a missing witness (folds ``""``) so a pre-witness
    entry is body-blind rather than unhashable; production entries always
    carry it (``keying_block``), and such stale cells are refused by the
    envelope/structure gates and the witness backstop regardless.

    ``placement`` (pgw#1113) folds in only when the entry states MORE THAN ONE
    distinct device — see the comment at the fold.
    """
    facts = {
        "v": 3,
        "target": str(entry.get("target") or ""),
        "fork": [[str(n), v] for n, v in (entry.get("fork") or [])],
        "class_dims": [
            [str(n), int(v)] for n, v in (entry.get("class_dims") or [])],
        "range_digest": str(entry.get("range_digest") or ""),
        "graph": dict(entry.get("graph") or {}),
        "graph_witness": str(entry.get("graph_witness") or ""),
        "strict": bool(strict),
        "lora_bucket": int(lora_bucket or 0),
    }
    placement = sorted({str(d) for d in (entry.get("placement") or ()) if d})
    if len(placement) > 1:
        # pgw#1113, closing pgw#819 at the key: a program whose own device map
        # spans several cards has that placement baked into its kernels, and
        # the canonical graph form scrubs the device INDEX by deliberate
        # design (`graph_hash._render_scalar`) so it cannot ride `graph`.
        # Keyed only when non-trivial — the `excluded` / `param` / `overlay`
        # precedent — because a single-device placement is what every cell the
        # fleet has published states, and a field that says "unchanged" would
        # strand all of them. No `v` bump for the same reason: the fact is
        # absent from every existing entry and its absence must stay the
        # canonical form.
        facts["placement"] = placement
    blob = json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
    # 64 bits, DERIVED (pgw#1232, §1.38b review). THIS is the `graph` axis of a
    # `cg-key-v1` key — the second of its two 64-bit chokepoints, the other
    # being `graph_hash._DIGEST_HEX`, which produces one of the facts above.
    # The axis has the MINIMUM of the two, so a widening moves both or neither;
    # `graph_hash` carries the birthday derivation (P ~= N^2/2^65: ~3e-12 at
    # 10^4 classes, ~3e-8 at 10^6). Kept at v1 deliberately rather than
    # inherited.
    return hashlib.sha256(blob).hexdigest()[:16]


def stamp_entry(
    name: str, block: Mapping[str, Any], *, strict: bool, lora_bucket: int,
) -> Dict[str, Any]:
    """One validated + stamped ENTRY block: ``name`` folded in, contract and
    constants parsed, ``range_digest`` and ``class_hash`` stamped.

    THE one place a class hash is stamped, so the mint's stamp, the boot
    key's fold and the admission recomputation are the same computation.
    Raises :class:`ValueError` naming the entry — a malformed contract must
    fail at MINT, on the pod, not at serve time on a paying request.
    """
    label = str(name or "").strip()
    if not label:
        raise ValueError("entry block carries no name")
    if LITERAL_SEP in label:
        raise ValueError(
            f"entry name {label!r} contains {LITERAL_SEP!r}, which the "
            f"literal namespace reserves")
    if not isinstance(block, Mapping):
        raise ValueError(f"entry {label!r} is not an object")
    if not str(block.get("target") or "").strip():
        raise ValueError(f"entry {label!r} declares no target")
    # Deep copy: the stamped block must not alias the caller's nested
    # containers (a later caller-side mutation would silently rewrite the
    # recorded contract).
    row = copy.deepcopy(dict(block))
    row["name"] = label
    try:
        contract_from_meta(row)
        constants_from_meta(row)
    except ValueError as exc:
        raise ValueError(f"entry {label!r}: {exc}") from exc
    row["range_digest"] = range_digest(row)
    row["class_hash"] = class_hash(
        row, strict=bool(strict), lora_bucket=int(lora_bucket or 0))
    return row


def entry_from_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """The validated ``entry`` block of a format-3 artifact.

    The block must parse as a full contract (inputs, symbols, constants),
    carry a target and carry a name — an entry the dispatch cannot route or
    assert is B2 with extra steps. Raises :class:`ValueError` naming the
    entry.

    pgw#1176: the plural ``entries_from_meta`` is GONE with the multi-entry
    artifact. A caller that wants several entries holds several artifacts.
    """
    raw = meta.get(cell_key_mod.ENTRY_BLOCK_KEY)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("metadata declares no entry block")
    label = str(raw.get("name") or "").strip()
    if not label:
        raise ValueError("entry block carries no name")
    if LITERAL_SEP in label:
        raise ValueError(
            f"entry name {label!r} contains {LITERAL_SEP!r}, which the "
            f"literal namespace reserves")
    if not str(raw.get("target") or "").strip():
        raise ValueError(f"entry {label!r} declares no target")
    try:
        contract_from_meta(raw)
        constants_from_meta(raw)
    except ValueError as exc:
        raise ValueError(f"entry {label!r}: {exc}") from exc
    return dict(raw)


def entry_metadata(
    *,
    family: str,
    precision: str,
    cell_key: str,
    name: str,
    entry: Mapping[str, Any],
    strict_export: bool = True,
    lora_bucket: int = 0,
    source_ref: str = "",
    source_digest: str = "",
    manifest_digest: str = "",
) -> Dict[str, Any]:
    """Build ONE entry artifact's ``metadata.json`` (format 3).

    THE single source of truth for the artifact-metadata envelope: the mint
    lane calls this rather than hand-rolling a dict, so producer and consumer
    cannot drift into two interpretations of the same bytes.

    ``manifest_digest`` is the declaration-wide coverage LABEL
    (``cell_key.manifest_digest``) — telemetry only, never identity. It is
    optional precisely because it is not identity: an entry minted by a pod
    that has not folded its whole declaration is still a complete, keyable,
    armable artifact.
    """
    stamped = stamp_entry(
        name, entry, strict=bool(strict_export),
        lora_bucket=int(lora_bucket or 0))
    meta: Dict[str, Any] = {
        COMPILED_GRAPH_FORMAT_KEY: COMPILED_GRAPH_FORMAT,
        "kind": ARTIFACT_KIND,
        **runtime_key(),
        "family": str(family or ""),
        "precision": str(precision or ""),
        "cell_key": str(cell_key or ""),
        cell_key_mod.ENTRY_BLOCK_KEY: stamped,
        "manifest_digest": str(manifest_digest or ""),
        "strict_export": bool(strict_export),
        "lora_bucket": int(lora_bucket or 0),
        "package_constants_in_so": False,
        # pgw#1097: the folding fence, DECLARED. `package_constants_in_so`
        # says no weight BYTES ship inside the cell; this says no weight
        # VALUES were compiled into its kernels either. Both are what make
        # one cell legally serve every fine-tune of a family, and both are
        # refused pre-download when absent — a cell minted before the fence
        # may carry its minting checkpoint's copy of any 0-dim or <=8-element
        # weight, which is exactly the tensor a fine-tune changes.
        "constant_folding_fenced": True,
        "source_ref": str(source_ref or ""),
        "source_digest": str(source_digest or ""),
        # pgw#754: the host-CPU execution requirement of the packaged host
        # code (wrapper .so + cpu kernels). Consumers refuse by name when
        # this host cannot execute it — the .so must never be dlopen'd
        # first and SIGILL second.
        "host_isa": host_isa.stamp(),
    }
    entry_from_meta(meta)
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
    COMPILED_GRAPH_FORMAT_KEY, "kind", "package_constants_in_so",
    "constant_folding_fenced", *IDENTITY_AXES,
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
    stated = meta.get(COMPILED_GRAPH_FORMAT_KEY)
    if int(stated or 0) != COMPILED_GRAPH_FORMAT:
        return (f"{COMPILED_GRAPH_FORMAT_KEY} {stated!r} != "
                f"{COMPILED_GRAPH_FORMAT}")
    if str(meta.get("kind") or "") != ARTIFACT_KIND:
        return f"kind {meta.get('kind')!r} != {ARTIFACT_KIND}"
    # A weights-baked artifact is refused OUTRIGHT, never merely warned
    # about: it would duplicate multi-GiB weights per cell and break the CAS
    # distribution model (pgw#704 B1). Absent flag = a pre-contract mint.
    if meta.get("package_constants_in_so") is not False:
        return (
            "artifact was minted with package_constants_in_so != False "
            "(weights baked into the .so; breaks the CAS cell model)")
    # pgw#1097: the same shape of refusal, one layer in. A cell minted before
    # the folding fence carries the minting checkpoint's values for any 0-dim
    # or <=8-element weight inductor inlined, so it is sound for exactly one
    # fine-tune and silently wrong for the rest. Absent flag = a pre-fence
    # mint; refused, not warned about, and re-minting is the remedy.
    if meta.get("constant_folding_fenced") is not True:
        return (
            "artifact was minted without the folding fence "
            "(constant_folding_fenced != True; its weights may carry the "
            "minting checkpoint's values — pgw#1097). Re-mint")
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
    # pgw#939: STRICTLY, like every axis above it and like this function's own
    # docstring already promised. `want_fam and ...` meant an UNSTAMPED cell
    # matched every family it was ever offered to — a wrong cache HIT, not a
    # miss, on the axis that decides which pipeline the .so is dlopen'd into.
    # The mint stamps this from one place (`artifact_metadata`), so a silent
    # cell is a malformed one; when no caller names a family nothing changes.
    if family and want_fam != family:
        return f"family {want_fam!r} != {family!r}"
    return ""


def verify_contract(
    meta: Dict[str, Any],
    *,
    entry: Optional[Mapping[str, Any]] = None,
) -> str:
    """'' when the artifact's ``entry`` contract is self-consistent, else
    the reason.

    The post-download half of :func:`verify`. The contract rides INSIDE the
    artifact — :func:`unpack` reads it off ``metadata.json``, which is where
    ``aot_serve`` has always served it from. It is verified HERE, on the
    staged bytes, and never against a control-plane declare that is not
    required to carry it (pgw#988).

    ``entry`` is the already-validated block from :func:`_unpack` when the
    arm path has one (pgw#1040 — same pure parse, threaded rather than
    repeated); ``None`` parses it here, which is what a caller holding only a
    metadata dict does.
    """
    if entry is None:
        try:
            entry = entry_from_meta(meta)
        except ValueError as exc:
            return f"malformed declared contract: {exc}"
    strict = bool(meta.get("strict_export", True))
    bucket = int(meta.get("lora_bucket") or 0)
    name = str(entry.get("name") or "")
    # pgw#939: absence is a verdict, not a skipped check. `class_hash` below
    # was already written this way and is the model the other axis is brought
    # to — `compile_cache.verify` is strict on every IDENTITY_AXES field for
    # the same reason, and `compile_cache.py` names this exact
    # `if want and want != have` shape as JAX PR #27814's one documented
    # wrong-cache-hit.
    stamped = str(entry.get("range_digest") or "")
    if not stamped:
        return f"entry {name!r}: no range_digest stamped"
    if stamped != range_digest(entry):
        return f"entry {name!r}: range_digest does not match its contract"
    stamped_hash = str(entry.get("class_hash") or "")
    if not stamped_hash:
        return f"entry {name!r}: no class_hash stamped"
    if stamped_hash != class_hash(entry, strict=strict, lora_bucket=bucket):
        # The receipts principle (pgw#716): a hash mismatch NAMES the class.
        return f"entry {name!r}: class_hash does not match its recorded facts"
    # pgw#1059/pgw#1176: the stamped key must be exactly the key the
    # artifact's OWN recorded facts describe — the same recomputation the
    # mint stamped and the publish path corroborated, now proven at ADMISSION
    # on the staged bytes. Two consequences, both deliberate: a forged /
    # hand-edited stamp is refused by name, and a PRE-ATOM cell is refused
    # STRUCTURALLY (its metadata records an `entries` MAP and a
    # `combined_graph_hash`, no per-entry identity, so the recomputation
    # raises rather than matching) — which is what makes the ck1 corpus purge
    # hygiene rather than a correctness precondition.
    # Gated on key SHAPE: an ek-shaped stamp is an identity claim and must
    # restate; a non-key stamp (focused fixtures, torn metadata) is not a
    # claim — and it can never match a hub row either (`IsCellKey` gates the
    # store flavor), so nothing downstream can mistake it for identity.
    stamped_key = str(meta.get("cell_key") or "")
    if stamped_key and cell_key_mod.is_key(stamped_key):
        try:
            recomputed = cell_key_mod.from_entry_metadata(meta)
        except cell_key_mod.CellKeyError as exc:
            return (
                f"stamped cell_key {stamped_key} is not restatable from the "
                f"artifact's own recorded facts ({exc})")
        if recomputed.digest != stamped_key:
            return (
                f"stamped cell_key {stamped_key} != the key the artifact's "
                f"recorded facts describe ({recomputed.digest})")
    return ""


def verify(
    meta: Dict[str, Any],
    *,
    family: str = "",
    entry: Optional[Mapping[str, Any]] = None,
) -> str:
    """'' when an entry's FULL metadata matches this runtime, else the reason.

    Both halves, for callers holding an artifact's own ``metadata.json``
    (:func:`stage_artifact`). Discovery, which holds only a declare, calls
    :func:`verify_declared` and reaches this one after the fetch.

    ``entry`` threads the already-validated block through to
    :func:`verify_contract` (pgw#1040).
    """
    return (verify_declared(meta, family=family)
            or verify_contract(meta, entry=entry))


#: :func:`host_isa_reason`'s refusal for a cell that stamped no requirement.
NO_HOST_ISA_STAMP = "no_host_isa_stamp"


def host_isa_reason(meta: Mapping[str, Any]) -> str:
    """'' when this host's CPU can execute the artifact's packaged host
    code, else the refusal reason (pgw#754).

    Reads the mint's ``host_isa`` requirement stamp — metadata-only.
    An artifact carrying NO stamp is refused here: its true
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
    return _unpack(artifact, dest_root)[0]


def _unpack(
    artifact: Path, dest_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """:func:`unpack`, also handing back the ``entry`` block it had to parse.

    pgw#1040: one arm used to run the entry parse three times over the same
    bytes — here for the literal-payload check, again in
    :func:`verify_contract`, and a third time in the arm — and each pass
    re-parses the entry's full contract and constant table. The parse is the
    same pure function of the same dict every time, so the arm path threads
    ONE result from here instead.
    """
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
    entry = entry_from_meta(meta)
    name = str(entry.get("name") or "")
    literals = [
        f"{name}{LITERAL_SEP}{s.fqn}"
        for s in constants_from_meta(entry) if s.source == SOURCE_LITERAL]
    if literals and LITERALS_NAME not in seen:
        raise ValueError(
            f"{ARTIFACT_KIND} artifact {artifact} declares literal constants "
            f"{sorted(literals)[:4]!r} but carries no {LITERALS_NAME}")
    return meta, entry


#: Read the packed envelope without unpacking the cell.
#:
#: pgw#1035: no serving-path caller since ``is_aot_artifact`` (its only one)
#: was deleted, and DELIBERATELY KEPT — this is the AOT lane's own reader of its
#: own envelope, and the pgw#699 double-mint byte-compare proof drives it over
#: real minted tarballs. It once had a byte-identical twin in
#: ``trt_engine.unpack_metadata``, and the dedup was deferred to a "TRT
#: ratification" that never came — TensorRT was deleted outright in pgw#1187,
#: so this is now the AOT lane's sole reader of its own envelope.
#:
#: pgw#1040 collapsed the OTHER seven envelope readers into
#: :func:`artifact_meta.read_metadata` and left this one alone ON PURPOSE,
#: reasoning that "it costs nothing to wait".
#:
#: pgw#1098 PRICED THE WAIT: $1.584 and 92 minutes. pgw#1013 then bounded the
#: collapsed reader and not this one, so two readers of one member disagreed
#: about row 7's sdxl envelope — and the disagreement was silent in the exact
#: direction that loses work. Delegated now. A second reader of a member is
#: not duplication to be tidied later; it is a divergence waiting for the
#: first caller who bounds one of them.
def unpack_metadata(artifact: Path) -> Dict[str, Any]:
    """Read ONLY metadata.json from an artifact (kind sniffing — cheap).

    pgw#1098: DELEGATES to ``artifact_meta``, which calls itself "the ONE
    reader" and was not. This function kept its own unbounded scan, so the
    two disagreed about the same bytes: on row 7's sdxl cell the bounded
    reader refused the envelope and this one read it fine, which is what made
    the failure asymmetric and invisible — ``arm_aot`` got ``meta=None`` and
    silently skipped the lifted-binding install, then ``enable`` (reaching
    the envelope through here) refused the artifact by a downstream name.
    One reader, one bound, or the next divergence costs another mint.
    """
    return artifact_meta.read_metadata(artifact)


@dataclass
class _StagedAotArtifact:
    metadata: Dict[str, Any]
    #: The validated ``entry`` block of :attr:`metadata`, parsed ONCE while
    #: staging (pgw#1040) and threaded to every consumer in the arm.
    entry: Dict[str, Any]
    root: Path
    temporary: "tempfile.TemporaryDirectory[str]"

    def close(self) -> None:
        self.temporary.cleanup()


def stage_artifact(
    artifact: Path, family: str, cache_dir: Optional[Path] = None,
    *, expected: "Optional[aot_identity.ExpectedIdentity]" = None,
) -> _StagedAotArtifact:
    """Extract and runtime-verify a complete artifact in an isolated tree.

    The live/shared cache and pipeline remain untouched on every rejection.
    Concurrent attempts use distinct trees; a process crash can leave only
    an unreferenced staging directory, never a partially published ``.pt2``.

    ``expected`` (pgw#903) is the identity the current ``ExecutionSpec`` named.
    When supplied, this artifact must BE that one — a declared-identity
    comparison, never a byte comparison (§4.25/§4.26: two mints of one key
    legitimately differ, pgw#1006 measured it). ``None`` is the pre-cutover
    RunJob path, where no immutable spec exists to compare against; it leaves
    behaviour byte-identical rather than inventing an expectation.
    """
    base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker"
    base.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="aot-stage-", dir=base)
    root = Path(temporary.name)
    try:
        meta, entry = _unpack(Path(artifact), root)
        # pgw#754: rule on host-CPU executability FIRST and by name — the
        # one failure mode that must never reach dlopen.
        isa_reason = host_isa_reason(meta)
        if isa_reason:
            raise AdoptError("host_isa_unsupported", isa_reason)
        reason = verify(meta, family=family, entry=entry)
        if reason:
            raise AdoptError("key_mismatch", reason)
        # pgw#903: "can this runtime execute it" is answered above; this
        # answers "is it the artifact the spec named", which nothing asked
        # before there was an immutable spec to ask against. Its own reason
        # class: a runnable cell that is the WRONG cell is a different bug,
        # a different owner and a different fix from an unrunnable one.
        if expected is not None:
            mismatch = aot_identity.verify_declared_identity(meta, expected)
            if mismatch:
                raise AdoptError("expected_identity_mismatch", mismatch)
        # pgw#765: the GPU-architecture axis as the BYTES declare it, ruled
        # on by name before dlopen — the tier that keeps cross-SKU adoption
        # honest now that ``sku`` no longer stands in for the arch. Runs
        # after `verify` so a stamped axis mismatch keeps its own name.
        sm_reason = verify_package_compute_capability(root / PACKAGE_NAME)
        if sm_reason:
            raise AdoptError("sm_mismatch", sm_reason)
        return _StagedAotArtifact(meta, entry, root, temporary)
    except AdoptError:
        temporary.cleanup()
        raise
    except Exception as exc:
        temporary.cleanup()
        raise AdoptError("artifact_invalid", str(exc)) from exc


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


#: pgw#1074: the ONLY dtype normalization this ingress performs. Integer ->
#: float32/float64 on a rank-0 input, and nothing else.
#:
#: float32 represents every integer up to 2**24 exactly, so the recast is
#: value-preserving over any diffusion timestep domain, and it is EXACTLY the
#: op the traced graph's own first node performs on this input (diffusers
#: ``get_timestep_embedding``: ``timesteps[:, None].float()``) — so a recast
#: feed produces bit-identical output to the eager call that presented the
#: integer. bfloat16 and float16 are deliberately NOT targets: bf16 has 8
#: mantissa bits and would round timestep 999 to 1000, which is a numeric
#: change, not a normalization.
RECAST_TARGETS = ("float32", "float64")
_INTEGER_DTYPES = ("int8", "int16", "int32", "int64", "uint8")


def recast_gap(spec: InputContract, value: Any) -> str:
    """The named dtype normalization this feed needs, or ``''`` (pgw#1074).

    ``''`` means either "already in contract" or "must be REFUSED" — this
    function never widens the contract, it only names the one normalization
    the ingress is allowed to perform, and :func:`ingress_report` refuses
    every other dtype disagreement exactly as before.

    **Why this exists.** The dtype a diffusers denoiser is handed for its
    scalar timestep is a per-request SAMPLER fact, not a family fact. Measured
    over ``gen_worker.view.SAMPLERS`` on the fleet's own diffusers (pgw#1074):
    ``euler``/``euler_a``/``euler_trailing``/``heun``/``flow_euler`` present
    **float32**; ``ddim``/``ddim_trailing``/``ddpm``/``deis``/``dpmpp_2m*``/
    ``lcm``/``unipc`` present **int64** (``set_timesteps`` ends in
    ``.to(dtype=torch.int64)``). sdxl's cell was minted float32 (ie#627, and
    correct — it is what the graph is specialized on), so its turbo arm
    (``euler_trailing``) served from the cell and its base arm, on an int64
    sampler, was refused ``no_entry_admits`` by an entry that covered it in
    every other respect. No single declared dtype can be right for a family
    whose sampler is per-request VIEW state, and the sampler is deliberately
    not a compile axis — so the normalization belongs at the boundary that
    knows the contract, once, named and counted.
    """
    if spec.shape:  # rank-0 only: the timestep class, not a tensor of values
        return ""
    if spec.dtype not in RECAST_TARGETS:
        return ""
    got = _dtype_name(value)
    if got not in _INTEGER_DTYPES:
        return ""
    if getattr(value, "shape", None) is None or tuple(value.shape):
        return ""
    return f"{got}_to_{spec.dtype}"


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

    def staged(self, name: str, value: Any, dtype: str = "") -> Any:
        """A 16-byte-aligned contiguous copy of ``value`` in an owned buffer.

        ``dtype`` (pgw#1074) stages into the artifact's DECLARED dtype instead
        of the feed's own. The conversion is ``Tensor.copy_``'s — one device
        kernel into the buffer this ingress already had to write, so the
        recast costs nothing beyond the staging copy and never synchronises.
        """
        import torch

        want = getattr(torch, dtype) if dtype else value.dtype
        buf = self._buffers.get(name)
        if (buf is None or buf.dtype is not want
                or buf.device != value.device
                or tuple(buf.shape) != tuple(value.shape)):
            buf = torch.empty(
                tuple(value.shape), dtype=want, device=value.device)
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
    report: Optional[Callable[[str, str, str], None]] = None,
) -> List[Any]:
    """``feeds`` normalized to the artifact's contract (pgw#791 + pgw#1074).

    Two normalizations, one staging buffer: the declared dtype
    (:func:`recast_gap`) and the declared 16-byte alignment
    (:func:`alignment_gap`). A recast implies a staged copy, so it subsumes
    the alignment pass rather than running after it.

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
        recast = recast_gap(spec, value)
        reason = recast or alignment_gap(value)
        if not reason:
            continue
        out[idx] = aligner.staged(spec.name, value, spec.dtype if recast else "")
        if report is not None:
            report(spec.name, reason, RECAST_EVENT if recast else REALIGN_EVENT)
    return out


#: pgw#1074: how far each refusal reason puts an entry from the call. Dims
#: LAST, so an entry that matches every declared dimension sorts to the front
#: of a refusal listing whatever its remaining complaint is. The rungs are
#: ordinal only — nothing reads their absolute values.
MISS_RUNGS: Mapping[str, int] = {
    # The call fits this graph's shape and disagrees about one scalar fact.
    "dtype_mismatch": 1,
    "input_not_tensor": 2,
    # A branch/adapter routing disagreement: same shape family, wrong class.
    "input_excluded": 3,
    # Shape disagreements — the call does not fit this graph at all.
    "static_dim_mismatch": 4,
    "range_violation": 4,
    "symbol_inconsistent": 4,
    "rank_mismatch": 5,
    # The call does not even carry the input; nothing else was measurable.
    "input_missing": 6,
}
_MISS_RUNG_DEFAULT = 9
#: How many non-closest entries a refusal names individually before it
#: switches to a per-reason count. The closest entry is ALWAYS named in full
#: and always first: this detail is truncated by the hub at ~573 chars, and
#: pgw#1074 is what happens when the one informative entry falls past that.
_MISS_SAMPLE = 3


@dataclass(frozen=True)
class IngressMiss:
    """ONE reason one entry refuses one call (pgw#1074).

    :attr:`rung` is how FAR that reason puts the entry from the call, and it
    exists because a refusal listing has to be ORDERED by something. The
    ordering is dims-first: an entry the call matches in every declared
    dimension and misses only on dtype is the entry a reader is looking for,
    and listing entries in iteration order buried exactly that one (36 tried,
    6 listed, the dims-matching one not among them — the pgw#1074 filing).
    """

    reason: str
    detail: str
    input: str = ""
    rung: int = _MISS_RUNG_DEFAULT


def _rung(reason: str) -> int:
    return MISS_RUNGS.get(reason, _MISS_RUNG_DEFAULT)


def miss_distance(misses: Sequence[IngressMiss]) -> Tuple[int, ...]:
    """Sort key over one entry's misses — lower is CLOSER to the call.

    The sorted rung tuple, so that (a) an entry whose only complaint is a
    shallow one beats an entry with a deep one, and (b) among entries with
    the same shallowest complaint, the one with FEWER complaints wins (a
    prefix sorts before its extension). Deterministic and total.
    """
    return tuple(sorted(_rung(m.reason) for m in misses))


def ingress_report(
    contract: ArtifactContract,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    first_only: bool = False,
) -> Tuple[Tuple[IngressMiss, ...], Dict[str, int]]:
    """EVERY way this call misses this contract, plus the symbol bindings.

    The one implementation of the pgw#704 B2 check. :func:`assert_ingress`
    raises this function's FIRST miss and :meth:`EntryDispatch.select` ranks
    whole entries by all of them, so an admission decision and the sentence
    that explains it can never be computed by two different rules.

    ``first_only`` returns as soon as one miss is found. It is an early EXIT
    from this same walk, never a second rule — every ADMISSION decision takes
    it (an admitted call has no misses, so the two are identical there), and
    the exhaustive walk is paid only on the refusal path, which is already
    falling back to eager. A 36-entry cell is asked this per denoise step.

    Misses are collected in declaration order, per input in
    dtype -> rank -> dims order, which is the order the raising check used
    before this became a collecting one.
    """
    present = excluded_inputs_present(contract, kwargs)
    if present:
        return ((IngressMiss(
            "input_excluded",
            f"this graph class REFUSES input(s) {list(present)!r}: the call "
            f"carries them, so it must be served by the class that declares "
            f"them (pgw#790 — a branchless class fed an adapter would return "
            f"the base model and look correct)",
            str(present[0])),), {})
    try:
        bound = bind_call_inputs(contract, args, kwargs)
    except IngressContractError as exc:
        # An input the call does not carry at all: nothing further about this
        # entry can be measured, so it is one miss and the deepest rung.
        return ((IngressMiss(exc.reason, str(exc)),), {})
    misses: List[IngressMiss] = []
    symbols: Dict[str, int] = {}
    owner: Dict[str, str] = {}
    for spec in contract.inputs:
        if first_only and misses:
            break
        if spec.name not in bound:
            continue
        value = bound[spec.name]
        shape = getattr(value, "shape", None)
        if shape is None:
            misses.append(IngressMiss(
                "input_not_tensor",
                f"declared input {spec.name!r} is a "
                f"{type(value).__name__} with no shape", spec.name))
            continue
        got_dtype = _dtype_name(value)
        if got_dtype != spec.dtype and not recast_gap(spec, value):
            misses.append(IngressMiss(
                "dtype_mismatch",
                f"input {spec.name!r} dtype {got_dtype or '<unknown>'} != "
                f"declared {spec.dtype}", spec.name))
        actual = tuple(int(d) for d in shape)
        if len(actual) != len(spec.shape):
            misses.append(IngressMiss(
                "rank_mismatch",
                f"input {spec.name!r} rank {len(actual)} != declared "
                f"{len(spec.shape)} (declared shape {list(spec.shape)!r})",
                spec.name))
            continue
        for pos, (declared, got) in enumerate(zip(spec.shape, actual)):
            if isinstance(declared, int):
                if got != declared:
                    misses.append(IngressMiss(
                        "static_dim_mismatch",
                        f"input {spec.name!r} dim {pos} = {got} != "
                        f"statically specialized {declared}", spec.name))
                continue
            lo, hi = contract.symbols[declared]
            if not (lo <= got <= hi):
                misses.append(IngressMiss(
                    "range_violation",
                    f"input {spec.name!r} dim {pos} (symbol {declared!r}) = "
                    f"{got} outside declared range [{lo}, {hi}]", spec.name))
                continue
            prior = symbols.get(declared)
            if prior is not None and prior != got:
                misses.append(IngressMiss(
                    "symbol_inconsistent",
                    f"symbol {declared!r} = {got} on input {spec.name!r} dim "
                    f"{pos} but {prior} on input {owner[declared]!r}",
                    spec.name))
                continue
            symbols[declared] = got
            owner.setdefault(declared, spec.name)
    return (tuple(misses[:1]) if first_only else tuple(misses)), symbols


def assert_ingress(
    contract: ArtifactContract,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Dict[str, int]:
    """Assert one call against the artifact's DECLARED contract (pgw#704 B2).

    Returns the resolved symbol bindings on success; raises
    :class:`IngressContractError`, naming the input, the dim, the symbol,
    the value and the bound, on the FIRST violation :func:`ingress_report`
    finds. Checks, per input:

    * present (unless declared optional);
    * dtype EXACT — an exported graph is specialized on dtype — except for
      the one named normalization :func:`recast_gap` performs;
    * rank EXACT;
    * static dims EXACT — a specialized dim is not a range;
    * symbolic dims inside the declared inclusive range;
    * **symbol CONSISTENCY** — one symbol appearing in two shapes must take
      the same value. ``range_constraints`` cannot express this, but the
      graph requires it, so a mismatch is outside the declared envelope even
      when both values are individually in range.
    """
    misses, symbols = ingress_report(contract, args, kwargs, first_only=True)
    if misses:
        raise IngressContractError(misses[0].reason, misses[0].detail)
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
    direct name lookup — no value-identity matching is needed. An
    unresolvable FQN is a named refusal:
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
        if spec.source == SOURCE_COMPUTED:
            # AOTInductor's own const-fold pass produces this one AFTER the
            # bound constants land; handing it a value would be handing it a
            # value it is about to overwrite, and demanding one would refuse a
            # cell that is complete (pgw#1080).
            continue
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


def _tensor_bytes(values: Iterable[Any]) -> int:
    total = 0
    for v in values:
        try:
            total += int(v.numel()) * int(v.element_size())
        except Exception:  # noqa: BLE001 — sizing is context, never a gate
            continue
    return total


def _device_memory_line() -> str:
    """One human line of live device memory for a typed refusal."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "device=cpu"
        free, total = torch.cuda.mem_get_info()
        return (f"device free {free / (1 << 20):.0f} MiB of "
                f"{total / (1 << 20):.0f} MiB")
    except Exception:  # noqa: BLE001 — context, never a gate
        return "device=unknown"


#: :func:`target_constant_pool`'s refusal for a resident tensor an AOTI
#: container cannot take a raw pointer to.
NONCONTIGUOUS_CONSTANT = "constant_noncontiguous"


def target_constant_pool(
    entry_constants: Iterable[Sequence[ConstantSpec]],
    state_dict: Mapping[str, Any],
    into: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The target's state_dict-sourced constants, BY REFERENCE, accreted into
    ``into`` (pgw#1042, pgw#1176, pgw#1177).

    ``into`` is the marker-owned pool an earlier entry already built for this
    target. Entries JOIN a target's pool as they arm — the pool is not a
    frozen product of a complete cell — and an FQN already present is left
    alone, so N entries of one target cost ONE pool whatever their number and
    whatever ORDER they arrive in.

    **THE CLONE IS GONE (pgw#1177, measured).** This function used to hand
    back ``value.detach().clone()`` per FQN. ``update_constant_buffer(...,
    user_managed=True)`` makes no copy of its own, so that clone was **the
    only copy in the system**: one full duplicate of the target's weights,
    held for the life of the arm — ~5.1 GiB on sdxl's single ``unet`` target
    — in direct contradiction of §4.33 step 4 ("the compiled entries bind
    constants BY REFERENCE against the resident weights; there is no second
    copy of the model"). Its stated justification, "a post-arm resident
    mutation cannot silently change an armed cell", is BACKWARDS: eager sees
    such a mutation immediately, so it is the un-mutated compiled entry that
    would silently diverge from the pipeline it serves.

    What the clone ALSO did, and what therefore survives explicitly: it
    normalised CONTIGUITY. An AOTI container takes a raw pointer, so a
    non-contiguous resident tensor cannot be bound by reference. Those are
    cloned individually (the exception, priced per tensor) rather than the
    whole pool being cloned for their sake.
    """
    out: Dict[str, Any] = {} if into is None else into
    for specs in entry_constants:
        for spec in specs:
            if spec.source != SOURCE_STATE_DICT or spec.fqn in out:
                continue
            value = state_dict.get(spec.fqn)
            if value is None:
                continue  # resolve_constants names the miss, typed, per entry
            try:
                contiguous = bool(value.is_contiguous())
            except Exception:  # noqa: BLE001 — duck-typed rigs hand non-tensors
                out[spec.fqn] = value
                continue
            # The exception, and only the exception, is copied.
            out[spec.fqn] = value if contiguous else value.detach().contiguous()
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
    #: pgw#791 + pgw#1074. ``"<input>/<reason>" -> count`` over every ingress
    #: NORMALIZATION — realignment (``unaligned_16b``) and dtype recast
    #: (``int64_to_float32``) alike; the typed event fires on the first of
    #: each, the count keeps the whole tax countable afterwards.
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
        its own constant buffer. A copying bind is one duplicate of the
        weights PER RUNNER — and pgw#758's multi-graph cells bind every
        entry up front, so an N-entry cell paid N duplicates and OOM'd the
        sdxl arm (pgw#1042). The whole-graph arm therefore binds by
        reference against ONE marker-owned pool per target
        (:func:`target_constant_pool`). The caller that passes True is
        asserting that the tensors outlive this runner — the pool rides the
        arm marker for exactly that reason.
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
        # copying path's call shape is byte-identical to what pgw#721/#723
        # measured on a pod. A torch whose `load_constants` has no such
        # parameter is a NAMED refusal rather than a silent copy — a caller
        # that asked for by-reference and silently copied would OOM the card
        # N binds later, which is a far worse way to learn the same fact.
        #
        # pgw#1042: the residual C++ failure is CLASSIFIED. The pod's 36/36
        # sdxl mint died at publish as an anonymous `RuntimeError:
        # update_constant_buffer_func_(...) API call failed at
        # model_container_runner.cpp:289` — the real message (a cudaMalloc
        # OOM from per-entry constant copies) went to a stderr no pod
        # exposes. Every failure inside the AOTI update is now a typed
        # ConstantsUnboundError carrying the entry, the constants' size and
        # the card's live free/total, so the hub row names the failure class.
        try:
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
                        f"parameter, so every constant would be COPIED — one "
                        f"copy of the target weights per bound entry "
                        f"({type(exc).__name__}: {exc})") from exc
            else:
                self.package.load_constants(values, check_full_update=True)
        except (ConstantsUnboundError, TypeError):
            raise
        except RuntimeError as exc:
            raise ConstantsUnboundError(
                "injection_failed",
                f"entry {self.entry or self.module_name or 'unknown'}: the "
                f"artifact refused the constant update inside AOTI "
                f"({type(exc).__name__}: {exc}); {len(values)} constants, "
                f"{_tensor_bytes(values.values()) / (1 << 20):.0f} MiB, "
                f"{_device_memory_line()}") from exc
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

    def _report_normalized(self, name: str, reason: str, event: str) -> None:
        """First occurrence of an (input, reason) is a typed hub-visible
        event; every occurrence is counted (pgw#791, pgw#1074).

        Coalesced deliberately: the defect fires 28+ times per request, and a
        per-call event would be the stderr spam it replaces, on a wire that
        costs money. One event names the input; the counter carries the rest.
        """
        key = f"{name}/{reason}"
        seen = self.realigned.get(key, 0)
        self.realigned[key] = seen + 1
        if seen:
            return
        entry = self.entry or self.module_name
        if event == RECAST_EVENT:
            logger.warning(
                "aot-serve: input %r arrived %s for entry %r; recasting to "
                "the declared dtype at ingress (the sampler, not the family, "
                "decides this dtype)", name, reason, entry)
            detail = (
                f"family={self.family} entry={entry} "
                f"target={self.module_name} input={name}: {reason} — recast "
                f"at ingress to the dtype this graph is specialized on "
                f"(pgw#1074: a scalar timestep's dtype is a per-request "
                f"SAMPLER fact, not a family one)")
        else:
            logger.warning(
                "aot-serve: input %r arrived %s for entry %r; realigning into "
                "an owned aligned buffer at ingress (the artifact would "
                "otherwise copy it on every call and report only on stderr)",
                name, reason, entry)
            detail = (
                f"family={self.family} entry={entry} "
                f"target={self.module_name} input={name}: {reason} — "
                f"realigned at ingress into an owned {AOTI_ALIGNMENT}-byte "
                f"aligned buffer (AOTInductor would otherwise copy it on "
                f"every call)")
        activity_mod.emit_event(event, detail, phase=reason)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.assert_ready()
        try:
            assert_ingress(self.contract, args, kwargs)
            feeds = marshal_positional(self.contract, args, kwargs)
            # pgw#791: satisfy the artifact's ALIGNED-input contract here,
            # once, instead of letting the runner discover it per call.
            feeds = aligned_feeds(
                self.contract, feeds, self.aligner, self._report_normalized)
        except IngressContractError as exc:
            self.refusals[exc.reason] = self.refusals.get(exc.reason, 0) + 1
            raise
        out = self.package(*feeds)
        self.calls += 1
        return out


def no_entry_detail(
    tried: int,
    missed: Sequence[Tuple[Tuple[int, ...], str, Tuple[IngressMiss, ...]]],
) -> str:
    """The ``no_entry_admits`` sentence, CLOSEST ENTRY FIRST (pgw#1074).

    The refusal this replaces said "36 tried" and then listed six in iteration
    order — and the one entry whose dims matched the call was not among them,
    so diagnosing a live refusal meant pulling the published cell apart off-pod
    to find out what the covering entry actually objected to. A refusal that
    hides the one relevant miss is the pgw#1058 lesson repeating one layer up,
    in the diagnostics.

    So: rank by :func:`miss_distance`, name the closest entry and its FULL
    reason first (it survives any downstream truncation), then account for
    every other entry by reason COUNT rather than by naming an arbitrary few.
    Nothing is silently dropped — ``tried`` and the counts always add up.
    """
    if not missed:
        return (
            f"request out of declared envelope: no packaged entry admits "
            f"this call ({tried} tried), so the request is served EAGER "
            f"and named at ingress")
    ranked = sorted(missed, key=lambda row: (row[0], row[1]))
    _distance, closest, misses = ranked[0]
    dims_ok = all(_rung(m.reason) < MISS_RUNGS["static_dim_mismatch"]
                  for m in misses)
    head = (
        f"request out of declared envelope — no packaged entry admits this "
        f"call, served EAGER ({tried} tried); CLOSEST entry "
        f"{closest!r}"
        f"{' — every declared dim MATCHES' if dims_ok else ''}: "
        + "; ".join(f"{m.reason} ({m.detail})" for m in misses[:2]))
    rest = ranked[1:]
    if not rest:
        return head
    # ONE count per entry, under its own closest reason, so the counts sum to
    # exactly the number of entries tried and "36 tried" can be checked
    # against the sentence that follows it.
    tally: Dict[str, int] = {}
    for _d, _name, other in rest:
        primary = min(other, key=lambda m: _rung(m.reason)).reason
        tally[primary] = tally.get(primary, 0) + 1
    named = "; ".join(
        f"{name}: {min(misses_, key=lambda m: _rung(m.reason)).reason}"
        for _d, name, misses_ in rest[:_MISS_SAMPLE])
    counted = ", ".join(
        f"{reason} x{count}" for reason, count in
        sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))
    return f"{head}. Other {len(rest)} entries [{counted}] — next: {named}"


@dataclass
class EntryDispatch:
    """The REGISTRY of armed entries serving ONE target, behind one call site
    (pgw#758, re-based per entry by pgw#1176).

    Dispatch is the declared contract itself: the call routes to the entry
    whose ingress contract ADMITS it. No admitting entry is a named
    per-request refusal (the caller serves eagerly); more than one is
    ``entry_ambiguous`` — the declaration failed to discriminate two graph
    classes by ingress, which is a defect to surface, never a coin to flip.

    pgw#1176: entries JOIN as they arm and LEAVE when they de-arm. The tuple
    that used to be built once, from a complete cell, was the arming half of
    the wrong atom. A registry has a further property the tuple could not:
    a subset is a legitimate STEADY STATE, not merely a stage on the way to
    coverage. pgw#1177 measured why that matters — ~0.75 GiB of device memory
    per resident AOTI container — so a pod that arms its hot classes and
    leaves cold ones eager holds a handful of containers rather than 36, on
    purpose and permanently.

    ``declared`` is every class name this pod's DECLARATION traces to,
    whether armed yet or not. It is what lets a miss distinguish "declared,
    pending compile" (silent eager, count the hotness) from "undeclared
    shape" (a real shape-growth report).
    """

    runners: Tuple[Tuple[str, ArtifactRunner], ...] = ()
    declared: Tuple[str, ...] = ()
    #: entry name -> the reason it left. A de-armed entry is REMEMBERED, not
    #: forgotten: §4.31's de-arm is sticky for the boot, and a re-arm of a
    #: class that failed for cause would be the thing that rule forbids.
    de_armed: Dict[str, str] = field(default_factory=dict)
    #: The entry :meth:`select` last routed to — the only way the fail-soft
    #: wrapper one frame up can name the graph class that raised.
    last_selected: str = ""
    #: Calls served by entries that have since DE-ARMED. Banked rather than
    #: discarded: those executions happened, and `execution_count` is the
    #: adoption proof's evidence — dropping them when a sibling de-arms would
    #: make a pod that served 5,000 compiled requests and then lost one class
    #: read as a pod that never served compiled at all.
    retired_calls: int = 0

    def add(self, name: str, runner: ArtifactRunner) -> None:
        """Register one armed entry. Replaces an entry of the same name (a
        re-arm), and refuses one this dispatch de-armed for cause."""
        label = str(name)
        if label in self.de_armed:
            raise AdoptError(
                "entry_de_armed",
                f"entry {label!r} was de-armed this boot "
                f"({self.de_armed[label]}); §4.31's de-arm is sticky, so it "
                f"must not be re-armed without a new process")
        rows = [(n, r) for n, r in self.runners if n != label]
        rows.append((label, runner))
        self.runners = tuple(sorted(rows, key=lambda row: row[0]))

    def remove(self, name: str, reason: str) -> bool:
        """De-arm ONE entry, sticky for the boot. True when it was armed.

        This is the whole replacement for the old cell-wide revoke: a
        cell-attributable failure in one graph class costs that class, and
        every sibling keeps serving compiled.
        """
        label = str(name)
        before = len(self.runners)
        self.retired_calls += sum(
            int(r.calls) for n, r in self.runners if n == label)
        self.runners = tuple(
            (n, r) for n, r in self.runners if n != label)
        self.de_armed[label] = str(reason or "unstated")
        return len(self.runners) != before

    @property
    def pending(self) -> Tuple[str, ...]:
        """Declared classes that are neither armed nor de-armed — the ones a
        background compile has not reached yet. Serving them eager is
        correct, expected, and not a defect."""
        armed = {n for n, _r in self.runners}
        return tuple(
            n for n in self.declared
            if n not in armed and n not in self.de_armed)

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
        missed: List[Tuple[str, ArtifactRunner]] = []
        for name, runner in self.runners:
            misses, _symbols = ingress_report(
                runner.contract, args, kwargs, first_only=True)
            if misses:
                missed.append((name, runner))
                continue
            admitted.append((name, runner))
        if not admitted:
            # Only now is the exhaustive walk worth its cost: the call is
            # already headed for the eager fallback, and the sentence it
            # leaves behind is the whole diagnosis anyone will ever get.
            ranked = [
                (miss_distance(rep), name, rep) for name, rep in (
                    (name, ingress_report(runner.contract, args, kwargs)[0])
                    for name, runner in missed)]
            pending = self.pending
            if pending:
                # pgw#1176: under accretion the commonest reason nothing
                # admits a call is that its class has not been compiled YET.
                # That is not a shape gap and must not be reported as one —
                # the growth path would submit a class the declaration
                # already contains.
                raise IngressContractError(
                    "entry_pending_compile",
                    f"no ARMED entry admits this call ({len(self.runners)} "
                    f"armed, {len(pending)} declared classes still pending "
                    f"compile) — served EAGER while the background compile "
                    f"reaches them: {list(pending)[:4]!r}. "
                    + no_entry_detail(len(self.runners), ranked))
            raise IngressContractError(
                "no_entry_admits", no_entry_detail(len(self.runners), ranked))
        if len(admitted) > 1:
            names = sorted(name for name, _ in admitted)
            raise IngressContractError(
                "entry_ambiguous",
                f"{len(admitted)} entries admit this call ({names[:6]!r}) — "
                f"the declaration does not discriminate these graph classes "
                f"by ingress contract")
        return admitted[0]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        name, runner = self.select(args, kwargs)
        # Recorded BEFORE the call so a raising entry can be de-armed BY NAME
        # (§4.31 per entry): the wrapper catches the exception a frame up and
        # has no other way to know which of N graph classes produced it.
        self.last_selected = name
        return runner(*args, **kwargs)

    @property
    def calls(self) -> int:
        return int(self.retired_calls) + sum(
            runner.calls for _n, runner in self.runners)

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

    The first artifact ERROR
    synchronously revokes scheduler-visible compiled proof and permanently
    routes to eager; the module object (config, dtype, device, weights)
    stays untouched, and its weights remain the constant-binding source.
    ``runner`` is one target's :class:`EntryDispatch` (or a bare
    :class:`ArtifactRunner` in focused tests — both are callable and count
    calls); ``attr`` generalizes the swap beyond ``forward`` for dotted
    targets like ``vae.decode`` (pgw#758).

    An :class:`IngressContractError` is NOT such an error. It is a named,
    counted, per-request contract refusal — the request serves eagerly and
    the artifact stays armed for traffic inside the declared envelope,
    because one
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

    def _de_arm(reason: str, detail: str) -> None:
        """§4.31 PER ENTRY (pgw#1176): a cell-attributable failure de-arms the
        GRAPH CLASS that produced it, sticky for the boot, and every sibling
        keeps serving compiled. The target's compiled lane is revoked only
        when the registry empties — that is the moment, and the only moment,
        at which this target genuinely stopped serving compiled.
        """
        name = str(getattr(runner, "last_selected", "") or "")
        remove = getattr(runner, "remove", None)
        if name and callable(remove):
            remove(name, reason)
            activity_mod.emit_event(
                "aot_entry_de_armed",
                f"target={label}: {detail}",
                phase=reason,
                family=str(meta.get("family") or ""),
                cell_key=str(meta.get("cell_key") or ""),
                graph_class=name,
            )
            siblings = tuple(getattr(runner, "runners", ()) or ())
            if siblings:  # siblings still serve
                logger.warning(
                    "aot-serve: %s de-armed entry %s (%s); %d sibling "
                    "entr%s still armed", label, name, reason, len(siblings),
                    "y" if len(siblings) == 1 else "ies")
                return
        state["failed"] = True
        _revoke(state, detail)

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
        if serve_posture.eager_only():
            # pgw#1142 / §4.32 item 4: an operator ordered eager. This is THE
            # reversibility seam — the artifact is not unwrapped and `state`
            # is not touched, so releasing the order resumes compiled serving
            # on the very next call with no re-arm, no re-materialize and no
            # re-mint. Unwrapping here would have been the same posture for
            # one boot and a lie afterwards.
            #
            # Ordered BEFORE the `failed` check only for cost; the two are
            # independent and stay independent — releasing the order never
            # resurrects a cell de-armed for cause (§4.31), because that
            # de-arm is evidence and this is policy.
            return original(*args, **eager_kwargs)
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
                "aot-serve: %s REFUSED input outside the declared envelope (%s: %s); "
                "serving this request eager, artifact stays armed",
                label, exc.reason, exc)
            activity_mod.emit_event(
                "aot_ingress_refused",
                f"family={meta.get('family')} target={label}: {exc}",
                phase=exc.reason,
            )
            report_ingress_refusal(state, exc.reason, str(exc))
            if exc.reason == "entry_pending_compile":
                # pgw#1176: the class IS declared; the background compile has
                # not reached it. Submitting a shape gap here would ask the
                # growth path to add a class the declaration already carries.
                return original(*args, **eager_kwargs)
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
            # its job (no segfault); THIS graph class is structurally
            # unusable.
            logger.error(
                "aot-serve: %s invoked with unbound constants (%s); eager for "
                "the rest of this process", label, exc)
            activity_mod.emit_event(
                "aot_constants_unbound",
                f"family={meta.get('family')} target={label}: {exc}",
                phase=exc.reason,
            )
            _de_arm(exc.reason, f"constants unbound: {exc}")
            return original(*args, **eager_kwargs)
        except Exception as exc:  # noqa: BLE001 — ANY artifact problem => eager
            if is_cuda_oom(exc):
                # pgw#1141: ATTRIBUTION. The serve-first doctrine makes the
                # first real request the proof, so what that request blames
                # decides whether a good cell survives — and allocator
                # exhaustion is a fact about the CARD at this instant (a
                # sibling load, a concurrent rotation), not about the artifact.
                # Condemning the cell for it would retire a correct one on the
                # first busy moment and re-mint it on the replacement pod.
                # Serve THIS request eager, stay armed, say so.
                logger.warning(
                    "aot-serve: %s hit CUDA OOM (%s); serving this request "
                    "eager, artifact stays armed — allocator pressure is not "
                    "the cell's fault", label, exc)
                activity_mod.emit_event(
                    "aot_serve_oom",
                    f"family={meta.get('family')} target={label}: "
                    f"{type(exc).__name__}: {exc}",
                    phase="cuda_oom",
                )
                return original(*args, **eager_kwargs)
            entry_name = str(getattr(runner, "last_selected", "") or "")
            detail = (
                f"AOTI artifact {label}"
                + (f" entry {entry_name}" if entry_name else "")
                + f" failed: {type(exc).__name__}: {exc}")
            _de_arm("artifact_failed", detail)
            logger.warning(
                "aot-serve: %s failed (%s: %s)", label,
                type(exc).__name__, exc)
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


def _entry_admission_drift(
    package_path: Path, entry: str, inputs_rows: Sequence[Mapping[str, Any]],
) -> None:
    """The pgw#1058 arm-side identity check: one entry's declared manifest
    rows verified against the artifact's OWN generated input guards, through
    the SAME ``aot_package.admission_drift`` the mint's package gate ran.

    A module attribute (like :func:`_load_package`) so unit rigs serving
    fake package bytes can substitute it; the import is function-local
    because ``aot_package`` imports this module's SOURCE_* vocabulary at
    its top."""
    from . import aot_package

    try:
        drift = aot_package.admission_drift(
            Path(package_path), entry, inputs_rows)
    except aot_package.PackageIntrospectionError as exc:
        raise AdoptError("admission_drift", f"entry {entry!r}: {exc}") from exc
    if drift:
        raise AdoptError(
            "admission_drift", f"entry {entry!r}: " + "; ".join(drift[:6]))


#: The classified refusal a bind that ran out of device memory produces. Same
#: token the two deleted `mint_budget.adopt_headroom` gates used, so every
#: downstream reader (the `cell_adopt_declined` event, `fleet_cells`' abort
#: classification, the hub's phase column) keeps its vocabulary — what changed
#: is that it is now emitted on EVIDENCE rather than on an estimate.
ADOPT_OOM_REASON = "insufficient_adopt_vram"


@contextlib.contextmanager
def _bind_headroom(what: str, armed: int, declared: int) -> Iterator[None]:
    """Turn a CUDA OOM inside one bind into a typed adopt refusal (pgw#1175).

    §4.33 step 4 loads the cell into the LIVE pipeline and keeps it or rejects
    it. Rejection has to be survivable, and the ONLY honest evidence that a
    card cannot hold the runners is a card that says so. This is the whole
    replacement for the deleted headroom estimate: no number is computed, the
    bind is attempted, and the attempt that fails names itself.

    It is deliberately narrow. A CUDA OOM and nothing else — every other
    exception keeps its own classification, because "the artifact is broken"
    and "this pod is full" are different verdicts and only one of them says
    anything about the cell. It also runs strictly BEFORE the first live
    mutation (:func:`wrap_module` is called after every entry binds), so a
    refusal leaves the pipeline exactly as eager as it found it.

    pgw#1176 CHANGED WHAT THE POSITION MEANS, and the change is the whole
    forensic point. It used to be ``(index+1 of total)`` — where in a
    bind-all-then-wrap sequence the OOM landed — which measured "how far
    through the cell did we get" and was the only handle th#1825 had. Under
    the atom that sequence does not exist: each class binds alone, so the
    index is always 1 of 1 and says nothing. What distinguishes "one class too
    big" from "wholly unadoptable" NOW is **how many siblings are already
    armed and still serving**, so that is what the refusal carries. The
    question the old number answered is answered better, because the armed
    count is a fact about what the pod is currently doing rather than about a
    loop it happened to be in.

    THE CHAIN IS WALKED, not just the top frame. ``ArtifactRunner.bind`` wraps
    every ``RuntimeError`` out of ``load_constants`` as
    ``ConstantsUnboundError("injection_failed")`` — a CONTRACT verdict — and
    ``torch.OutOfMemoryError`` is a ``RuntimeError``. So the exact failure this
    guard exists for arrives already re-labelled as the cell's fault, which is
    how "this pod is full" would have retired a correct cell. The cause is what
    decides.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 — re-raised unless it is an OOM
        oom = _oom_in_chain(exc)
        if oom is None:
            raise
        flush_memory()
        raise AdoptError(
            ADOPT_OOM_REASON,
            f"{what} ({armed} of {declared} already armed) ran out of device "
            f"memory while binding ({type(oom).__name__}: {oom}) — this pod "
            f"cannot hold THIS CLASS beside what it is already serving. It "
            f"serves eager; the {armed} class(es) already armed keep serving "
            f"COMPILED. Neither the class nor the declaration is condemned: "
            f"another pod, or this one with less resident, may bind it fine",
        ) from exc


def _oom_in_chain(exc: BaseException) -> Optional[BaseException]:
    """The CUDA OOM in this exception's cause chain, or ``None``."""
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        if is_cuda_oom(cur):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


def _marker(pipeline: Any) -> Dict[str, Any]:
    """The pipeline's arm marker, created empty on first use.

    pgw#1176: the marker is the pod's per-entry serve-state record, not a
    snapshot of one cell. It carries the wrapped targets, the by-reference
    constant pools that must outlive their runners, the literal tables, and
    one row per entry ever armed on this object.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if not isinstance(marker, dict):
        marker = {
            "meta": {}, "targets": {},
            "bound_constants": {"pools": {}, "literals": {}},
            "entries": {},
        }
        setattr(pipeline, _MARKER_ATTR, marker)
    return marker


def _dispatch_for(marker: Dict[str, Any], target: str) -> Optional[EntryDispatch]:
    row = (marker.get("targets") or {}).get(target) or {}
    runner = (row.get("state") or {}).get("runner")
    return runner if isinstance(runner, EntryDispatch) else None


def arm_entry(
    pipeline: Any, cfg: Any, artifact: Path, cache_dir: Optional[Path] = None,
    *, expected: "Optional[aot_identity.ExpectedIdentity]" = None,
    declared: Sequence[str] = (),
) -> Dict[str, Any]:
    """Stage + verify + load + BIND + register ONE compiled graph class.

    THE ATOM (pgw#1176). What this replaces — ``load_and_wrap``'s
    stage-verify-load-EVERY-entry-then-bind-ALL-then-wrap shape — carried the
    invariant "a cell that cannot arm one of its graph classes arms none of
    them, because a partially served contract would be a silent subset of
    what the cell key advertises". That invariant was locally correct and
    globally the disease: it was a faithful guard on the premise that the
    advertising unit is a 36-class set. Shrink the unit to one graph and
    nothing is left that CAN lie — an entry arms whole or not at all, and an
    entry is one graph, so that is atomic by nature.

    Consequences that fall out rather than being engineered:

    * an entry that cannot arm costs THAT class. Its siblings keep serving
      compiled, and the pod reports per-entry serve state rather than a
      cell-level claim;
    * coverage ACCRETES. A second call arms a second class into the same
      registry, the same target pool and the same live wrap. There is no
      "complete" state and nothing waits for one;
    * a deliberate, permanent SUBSET is a legitimate steady state — which
      pgw#1177 measured the reason for: ~0.75 GiB of device memory per
      resident AOTI container, so a pod that arms its hot classes and leaves
      the cold ones eager holds a handful of containers instead of 36;
    * verification costs ONE runner beside the already-resident weights,
      which is §4.33's ~8 GiB achieved by construction rather than by budget.

    ``declared`` is every class name this pod's declaration traces to. It
    feeds :attr:`EntryDispatch.declared`, so a call that no ARMED entry
    admits can say "pending compile" instead of reporting a shape gap for a
    class the declaration already contains.

    ``expected`` (pgw#903) is checked inside :func:`stage_artifact`, i.e.
    strictly before ``_load_package`` — the identity question must be settled
    while the artifact is still inert bytes.

    §4.33 / pgw#1175, CARRIED THROUGH THE ATOM AND IMPROVED BY IT: THE DEVICE
    COST OF THIS FUNCTION IS ATTEMPTED, NEVER PREDICTED. Two estimates used to
    stand in front of it — both ``mint_budget.adopt_headroom``, both pricing
    the arm at twice an "activation" that was a quarter of the RESIDENT SET
    whenever no forward had run, and both compared against a free figure those
    weights were already outside of. The function's own docstring conceded it
    could not refuse the failure it was written for, while stickily refusing
    cards that were fine (it pinned two real mints at 11.09 GiB on cards with
    21.48 GiB free). What refuses now is the bind: ``_load_package`` + ``bind``
    run inside a CUDA-OOM guard, so a card that genuinely cannot hold the
    runner returns a typed ``insufficient_adopt_vram`` refusal NAMING the
    entry — before the first live mutation, so the pipeline is untouched and
    the pod serves eager. The attempt is the measurement.

    **The atom makes that refusal cheaper and more honest than it could be
    under the cell.** A bind OOM here costs exactly ONE graph class: its
    siblings stay armed and keep serving compiled, and the refused class is
    retried by nobody and condemned by nothing — another pod, or this one with
    less resident, may bind it fine. Under the 36-entry cell the same OOM
    discarded every class.

    Raises :class:`AdoptError` with a classified reason on any failure, and
    never publishes extracted files into a shared live cache.
    """
    family = str(getattr(cfg, "family", "") or "")
    # pgw#1087: admission splits in two and the halves have different owners.
    # `cell_verify` is unpack + identity + contract verification on inert
    # bytes (disk + hashing); `entry_admit` below is dlopen, constant bind and
    # ingress-assertion arming (device).
    with boot_phases.span(
        boot_phases.PHASE_CELL_VERIFY, ref=family,
        artifact_kind="aot-inductor",
    ) if boot_phases.in_boot() else contextlib.nullcontext():
        staged = stage_artifact(
            Path(artifact), family, cache_dir=cache_dir, expected=expected)
    try:
        meta = staged.metadata
        block = staged.entry  # parsed and validated once, while staging
        name = str(block.get("name") or "")
        target = str(block.get("target") or "")
        module, attr = _target_owner(pipeline, target)

        marker = _marker(pipeline)
        dispatch = _dispatch_for(marker, target)
        first_for_target = dispatch is None
        if dispatch is None:
            dispatch = EntryDispatch(declared=tuple(str(d) for d in declared))
        elif declared:
            dispatch.declared = tuple(str(d) for d in declared)

        t0 = time.monotonic()
        literals: Dict[str, Any] = {}
        literals_path = staged.root / LITERALS_NAME
        if literals_path.exists():
            device = str(getattr(module, "device", "cuda"))
            try:
                literals = split_literals(
                    _load_literals(literals_path, device)).get(name, {})
            except ValueError as exc:
                raise AdoptError("contract_invalid", str(exc)) from exc

        try:
            contract = contract_from_meta(block)
            constants = constants_from_meta(block)
            # pgw#725: the lifted-adapter signature must match the module's
            # actual lifted state.
            assert_lifted_contract(module, contract)
        except ValueError as exc:
            raise AdoptError(
                "contract_invalid", f"entry {name!r}: {exc}") from exc
        # pgw#1058: the admission contract this dispatch will enforce must BE
        # the one the artifact's own generated guards enforce — the same
        # derivation the mint's package gate ran, re-run where the bytes
        # arrive, so a label that drifted between publish and adopt is a named
        # refusal here and never an opaque per-call admission miss.
        _entry_admission_drift(
            staged.root / PACKAGE_NAME, name, list(block.get("inputs") or []))

        # The target's by-reference pool, ACCRETED (pgw#1176/pgw#1177): this
        # entry's state-dict constants join whatever earlier entries already
        # registered. The pool rides the marker because user_managed binds
        # hold raw pointers, so the bound values must outlive the runners.
        pools = marker["bound_constants"]["pools"]
        pool = pools.setdefault(target, {})
        # pgw#1175, carried: the pool grows by reference, so an OOM here is a
        # CARD fact and gets the card's own verdict rather than the cell's.
        _armed_here = len(dispatch.runners)
        _declared_here = len(dispatch.declared) or (_armed_here + 1)
        with _bind_headroom(
                f"target {target!r} pool", _armed_here, _declared_here):
            target_constant_pool(
                [constants], resident_constants(module), into=pool)

        with boot_phases.span(
            boot_phases.PHASE_ENTRY_ADMIT, ref=family, function=name,
            artifact_kind="aot-inductor",
        ) if boot_phases.in_boot() else contextlib.nullcontext() as sp:
            # pgw#1175 + pgw#1176: THE ATTEMPT IS THE MEASUREMENT, and under
            # the atom it costs exactly one graph class. `_bind_headroom`
            # walks the CAUSE CHAIN, which is load-bearing: `bind` re-labels
            # every RuntimeError out of `load_constants` as a CONTRACT
            # verdict, and `torch.OutOfMemoryError` IS a RuntimeError — so a
            # full card would otherwise condemn a correct cell, and a contract
            # verdict is the kind that quarantines a key (th#1819). Capacity
            # is never a contract verdict.
            with _bind_headroom(
                    f"entry {name!r}", _armed_here, _declared_here):
                package = _load_package(staged.root / PACKAGE_NAME, name)
                runner = ArtifactRunner(
                    package=package, contract=contract, constants=constants,
                    module_name=target, entry=name, family=family)
                try:
                    runner.bind(pool, literals, user_managed=True)
                except ConstantsUnboundError as exc:
                    raise AdoptError(
                        f"constants_{exc.reason}",
                        f"entry {name!r}: {exc}") from exc
            if sp is not None:
                sp.note(f"target={target} constants={len(constants)}")

        # FIRST LIVE MUTATION, and it is exactly one entry wide. Everything
        # above is proven for THIS entry: complete artifact, matching runtime
        # key, restated key, resolved target, loaded named model, constant
        # table proven bound against its manifest.
        dispatch.add(name, runner)
        marker["bound_constants"]["literals"][name] = literals
        if first_for_target:
            wrap_module(
                module, dispatch, meta, attr=attr, target=target,
                eager_forward=None)
            module_marker = getattr(module, _MARKER_ATTR, {})
            marker["targets"][target] = {
                "module": module,
                "attr": attr,
                "state": module_marker.get("state", {}),
            }
        # `meta` on the marker is the DECLARE half every entry of this
        # pipeline shares by construction (`verify_declared` refuses any
        # artifact whose sm/torch/cuda/family disagree), kept so callers that
        # ask the live object what it is armed with get an answer without
        # re-reading a tarball. Per-entry facts live in `entries`.
        marker["meta"] = meta
        marker["entries"][name] = {
            "key": str(meta.get("cell_key") or ""),
            "target": target,
            "class_hash": str(block.get("class_hash") or ""),
            "manifest_digest": str(meta.get("manifest_digest") or ""),
        }
        # pgw#1141b: THE registration, at the one seam every arm route passes.
        note_aot_key(str(meta.get("cell_key") or ""))
        logger.info(
            "aot-serve: armed entry %s on target %s in %.1fs (%d declared "
            "constants, key=%s); %d entr%s now armed on this pipeline",
            name, target, time.monotonic() - t0, len(constants),
            meta.get("cell_key"), len(marker["entries"]),
            "y" if len(marker["entries"]) == 1 else "ies")
        return meta
    finally:
        staged.close()


def disarm_entry(pipeline: Any, name: str, reason: str) -> bool:
    """De-arm ONE graph class, sticky for the boot. True when it was armed.

    The per-entry half of §4.31, reachable from outside a serve call: the
    mint's parity gate refuses an entry here rather than un-arming a cell,
    and an LRU eviction of a cold container is the same operation.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None)
    if not isinstance(marker, dict):
        return False
    dropped = False
    for target, row in dict(marker.get("targets") or {}).items():
        dispatch = _dispatch_for(marker, target)
        if dispatch is None or str(name) not in dict(dispatch.runners):
            continue
        dropped = dispatch.remove(str(name), str(reason)) or dropped
        if not dispatch.runners:
            # The target no longer serves anything compiled. Restore its eager
            # callable rather than leaving a wrapper that only ever falls back.
            state = row.get("state") or {}
            state["failed"] = True
    marker.get("entries", {}).pop(str(name), None)
    marker["bound_constants"]["literals"].pop(str(name), None)
    return dropped


def entry_states(pipeline: Any) -> Dict[str, Dict[str, Any]]:
    """Per-entry SERVE STATE — what this pod actually serves, per graph class.

    pgw#1176 §1.4: the pod never claims "cell X armed". It reports, per entry,
    ``armed`` / ``de_armed(reason)`` / ``pending`` — so there is no unit left
    that can advertise more than it serves. This is the record the per-entry
    hub events (§1.7) are built from.
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for target in (marker.get("targets") or {}):
        dispatch = _dispatch_for(marker, target)
        if dispatch is None:
            continue
        for name, runner in dispatch.runners:
            out[name] = {
                "state": "armed", "target": target, "calls": int(runner.calls)}
        for name, why in dispatch.de_armed.items():
            out[name] = {"state": "de_armed", "target": target, "reason": why}
        for name in dispatch.pending:
            out[name] = {"state": "pending", "target": target}
    return out


def _adopt_identity(artifact: Path) -> str:
    """Best-effort ``family=… key=…`` from the artifact's own metadata for
    the typed adopt event — a refusal must name the candidate cell even when
    the refusal itself is a metadata problem."""
    meta = artifact_meta.try_read_metadata(artifact)
    if meta is None:
        return f"artifact={artifact.name}"
    return f"family={meta.get('family')} key={meta.get('cell_key')}"


def enable(
    pipeline: Any,
    cfg: Any,
    cache_dir: Optional[Path] = None,
    artifact: Optional[Path] = None,
    *,
    expected: "Optional[aot_identity.ExpectedIdentity]" = None,
    declared: Sequence[str] = (),
) -> AdoptOutcome:
    """Consumer entry point: verify + load + bind + register ONE entry.

    Falsy (staying eager) on ANY miss — the caller's ordinary miss policy
    (fleet self-mint / eager / typed refusal) takes over. Truthy IS the HIT:
    ``fleet_cells`` treats it as a genuine match and
    skips the self-mint.

    pgw#1176: a miss here is a miss for ONE graph class. Nothing about it
    un-arms a sibling, and the caller's retry/mint policy is per class too.

    pgw#923: the outcome is RETURNED rather than narrated. The classified
    refusal reason used to leave this function only as the ``phase`` of a
    free-text ``aot_adopt`` event, which is why the adoption that actually
    happens on every boot had no measured row anywhere — the caller could not
    see what it had just been told.
    """
    if artifact is None:
        return AdoptOutcome.miss("no_artifact")
    try:
        meta = arm_entry(
            pipeline, cfg, Path(artifact), cache_dir=cache_dir,
            expected=expected, declared=declared)
    except Exception as exc:
        reason = str(getattr(exc, "reason", "") or "") or type(exc).__name__
        identity = _adopt_identity(Path(artifact))
        logger.warning(
            "aot-serve: entry unusable (%s: %s); this class serves eager",
            reason, exc)
        return AdoptOutcome.miss(
            reason, f"{identity}: {type(exc).__name__}: {exc}", identity)
    entry = dict(meta.get(cell_key_mod.ENTRY_BLOCK_KEY) or {})
    armed = len(armed_entries(pipeline))
    logger.info(
        "aot-serve: armed %s entry %s (sku=%s torch=%s precision=%s, "
        "constants bound BY REFERENCE from resident weights); %d armed",
        meta.get("family"), entry.get("name"),
        meta.get("sku"), meta.get("torch"), meta.get("precision"), armed)
    return AdoptOutcome.hit(
        f"family={meta.get('family')} key={meta.get('cell_key')} "
        f"entry={entry.get('name')} armed={armed} sku={meta.get('sku')} "
        f"torch={meta.get('torch')} precision={meta.get('precision')}")


def _marker_states(subject: Any) -> List[Dict[str, Any]]:
    """Every wrapped target's state dict on a marker — PIPELINE or MODULE.

    pgw#1176, CORRECTED. I first deleted the single-``state`` branch here as
    "a legacy shape only tests build". That was half right and the wrong
    half: a bare ``state`` on a PIPELINE is indeed a shape nothing produces
    (and no fixture builds one any more), but this function is also called
    with a MODULE, and a bare ``state`` is exactly what :func:`wrap_module`
    writes there — on every arm, in production. Deleting it made
    `execution_count(module)` answer 0 for a module that had served.

    So the branch is not legacy and stays; what it reads is named. The two
    markers are different objects and always were: :func:`wrap_module` writes
    ``state`` on the MODULE it swapped, :func:`arm_entry` writes ``targets``
    on the PIPELINE that owns it.
    """
    marker = getattr(subject, _MARKER_ATTR, None) or {}
    rows = marker.get("targets")
    if isinstance(rows, dict):
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

    pgw#1035 audited this for deletion and KEPT it. It has no fleet reader —
    but neither do its two siblings :func:`ingress_refusals` and
    :func:`served_entry_calls` (the mint rig drives the latter), and deleting
    one of three sibling measurements would make the #791 tax unobservable
    while leaving the machinery that computes it. This is the built-but-UNWIRED
    class, whose fix is a caller — a JobMetrics or activity field carrying all
    three — not a diff that hides the gap.
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


def armed_entries(pipeline: Any) -> Dict[str, str]:
    """``entry name -> entry key`` for every graph class ARMED right now.

    pgw#1176: this — not a cell-level boolean — is what the pod may claim. A
    subset is a legitimate steady state, so "how many, and which" is the only
    honest answer to "what does this pod serve compiled".
    """
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    rows = dict(marker.get("entries") or {})
    out: Dict[str, str] = {}
    for target in (marker.get("targets") or {}):
        dispatch = _dispatch_for(marker, target)
        if dispatch is None:
            continue
        for name, _runner in dispatch.runners:
            out[name] = str((rows.get(name) or {}).get("key") or "")
    return out


def is_armed(pipeline: Any) -> bool:
    """Whether this pipeline is serving ANY compiled graph class right now.

    pgw#1176 DELETED the every-target rule ("one revoked target means the
    cell no longer serves the contract its key advertises"). That rule was
    the arming half of the wrong atom: a key that advertised 36 classes made
    partial service a lie, so the guard was locally correct — and it is what
    forbade the incremental compile-and-adopt this design exists to deliver.
    A key now advertises ONE class, an entry arms whole or not at all, and a
    de-armed entry costs itself. Mixed compiled/eager service is not a
    degraded state; it is the design's normal one, and it is numerically as
    proven as full coverage because every armed entry passed the same mint
    parity gate against the same eager reference.

    Ask :func:`armed_entries` / :func:`entry_states` when you need to know
    WHAT is served rather than WHETHER anything is.
    """
    return bool(armed_entries(pipeline))


def holds_exported_cell(pipeline: Any) -> bool:
    """Whether an AOTI cell is WRAPPED onto this object — armed or revoked.

    pgw#1141b: the LANE question, asked of the object instead of a ref string.
    "Is this the exported lane?" decides which failure detector applies — the
    dynamo lane's per-class cache-hit ledger (which an AOTI artifact can never
    move, so it reads every honest adoption as a disproof) or §4.31's
    serve-first rule. Answering it through :func:`is_aot_ref` made the answer
    depend on whether some earlier caller had announced the key to this
    process; the wrap is the fact itself.

    Distinct from :func:`is_armed` on purpose: a cell whose guard revoked it
    still HOLDS this pipeline's eager originals, and the install path has to
    tell "revoked exported cell" (never advertise it) apart from "no cell here
    at all" (an ordinary dynamo/eager object).
    """
    return bool(_marker_states(pipeline))


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
    and the unproven-adoption rollback both go through here."""
    marker = getattr(pipeline, _MARKER_ATTR, None) or {}
    # pgw#1176: one shape, the one `arm_entry` writes. The `module`/`state`
    # fallback that stood here read a pipeline marker no production path
    # produces — see :func:`_marker_states`.
    targets = marker.get("targets")
    rows: List[Dict[str, Any]] = (
        list(targets.values()) if isinstance(targets, dict) else [])
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
    "COMPILED_GRAPH_FORMAT",
    "COMPILED_GRAPH_FORMAT_KEY",
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
    "SOURCE_COMPUTED",
    "SOURCE_LITERAL",
    "SOURCE_STATE_DICT",
    "armed_metadata",
    "armed_targets",
    "arm_entry",
    "armed_entries",
    "assert_bindable",
    "assert_lifted_contract",
    "assert_ingress",
    "bind_call_inputs",
    "class_hash",
    "constants_from_meta",
    "contract_from_meta",
    "enable",
    "entry_from_meta",
    "entry_metadata",
    "entry_states",
    "disarm_entry",
    "execution_count",
    "proven_since",
    "holds_exported_cell",
    "host_isa_reason",
    "NO_HOST_ISA_STAMP",
    "ingress_class_name",
    "ingress_refusals",
    "is_aot_ref",
    "is_armed",
    "lifted_call_kwargs",
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
    "target_constant_pool",
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
