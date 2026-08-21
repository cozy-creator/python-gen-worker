"""The layout converter registry — the CONVERTIBLE rung."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..toolchain import closure_file_digest as _closure_file_digest
from ..models.tensor_layout_contract import (
    AXIS_QUANT,
    AXIS_TOPOLOGY,
    LAYOUT_AXES,
    LAYOUT_AXIS_ANY,
    LayoutId,
    validate_layout_handle,
)
from .repack_spec import DeclarationError, RenameRule
from .writer import (
    read_safetensors_header,
    rewrite_safetensors_keys,
    shard_content_digest,
    shard_payload_digests,
    shard_tensor_entries,
    write_safetensors_shard,
)

MAX_CORPUS_PAYLOAD_BYTES = 1 << 20

MAX_PLAN_HOPS_PER_AXIS = 2

RepackFn = Callable[["ConversionIO"], None]
EquivalenceFn = Callable[[Path, Path], None]

_DTYPE_ITEMSIZE: Dict[str, int] = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


class ConversionProofError(DeclarationError):
    """A converter's bit-exactness obligation did not hold over its corpus."""


class LayoutRung(str, Enum):
    """The four-valued ladder."""

    COMPATIBLE = "compatible"
    CONVERTIBLE = "convertible"
    PRODUCIBLE = "producible"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class CorpusTensor:
    """One tensor as a real artifact's header declares it."""

    dtype: str
    shape: Tuple[int, ...]

    def nbytes(self) -> int:
        itemsize = _DTYPE_ITEMSIZE.get(self.dtype)
        if itemsize is None:
            raise DeclarationError(
                f"unknown safetensors dtype {self.dtype!r}; known: "
                f"{', '.join(sorted(_DTYPE_ITEMSIZE))}")
        count = 1
        for dim in self.shape:
            if int(dim) < 0:
                raise DeclarationError(f"negative dim in shape {self.shape!r}")
            count *= int(dim)
        return count * itemsize


@dataclass(frozen=True)
class ConversionCase:
    """One corpus case: the KEYS, dtypes and shapes of a real artifact."""

    name: str
    tensors: Mapping[str, CorpusTensor]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise DeclarationError("ConversionCase.name is empty")
        if not self.tensors:
            raise DeclarationError(
                f"corpus case {self.name!r} declares no tensors; a case that "
                "acts on nothing proves nothing")
        total = sum(t.nbytes() for t in self.tensors.values())
        if total > MAX_CORPUS_PAYLOAD_BYTES:
            raise DeclarationError(
                f"corpus case {self.name!r} declares {total} payload bytes, "
                f"over the {MAX_CORPUS_PAYLOAD_BYTES} budget — the admission "
                "proof runs at registration, so a case must stay cheap. Use a "
                "real header with representative KEYS and small shapes.")


def _synthetic_payload(case: str, key: str, nbytes: int) -> bytes:
    out = bytearray()
    seed = f"{case}\x00{key}".encode()
    counter = 0
    while len(out) < nbytes:
        out.extend(hashlib.sha256(seed + str(counter).encode()).digest())
        counter += 1
    return bytes(out[:nbytes])


def materialize_case(case: ConversionCase, path: Path) -> Path:
    """Write one corpus case as a real safetensors shard."""
    tensors = {
        key: (spec.dtype, tuple(spec.shape),
              _synthetic_payload(case.name, key, spec.nbytes()))
        for key, spec in case.tensors.items()
    }
    write_safetensors_shard(path, tensors, metadata=dict(case.metadata))
    return path


class ConversionIO:
    """One component shard, in and out, for a QUANT-axis repack."""

    def __init__(self, source: Path, target: Path) -> None:
        self._target = Path(target)
        self._fd = os.open(str(source), os.O_RDONLY)
        try:
            self._header, self._base = read_safetensors_header(self._fd)
        except BaseException:
            os.close(self._fd)
            raise
        self._spool_path = self._target.parent / f".{self._target.name}.spool"
        self._spool = open(self._spool_path, "wb")
        self._emitted: List[Tuple[str, str, Tuple[int, ...], int, int]] = []
        self._metadata: Dict[str, str] = {
            str(k): str(v)
            for k, v in (self._header.get("__metadata__") or {}).items()
        }
        self._cursor = 0

    def keys(self) -> Tuple[str, ...]:
        return tuple(k for k in self._header if k != "__metadata__")

    def spec(self, key: str) -> CorpusTensor:
        meta = self._entry(key)
        shape = meta["shape"]
        assert isinstance(shape, (list, tuple))
        return CorpusTensor(
            dtype=str(meta["dtype"]), shape=tuple(int(d) for d in shape))

    def read(self, key: str) -> bytes:
        meta = self._entry(key)
        offsets = meta["data_offsets"]
        assert isinstance(offsets, (list, tuple))
        start, end = int(offsets[0]), int(offsets[1])
        out = bytearray()
        offset = self._base + start
        remaining = end - start
        while remaining > 0:
            chunk = os.pread(self._fd, min(remaining, 8 << 20), offset)
            if not chunk:
                raise IOError(f"safetensors: short read on {key!r}")
            out.extend(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        return bytes(out)

    def source_metadata(self) -> Mapping[str, str]:
        return dict(self._metadata)

    def emit(self, key: str, *, dtype: str, shape: Sequence[int],
             payload: bytes) -> None:
        if not key or key == "__metadata__":
            raise DeclarationError(f"cannot emit tensor named {key!r}")
        self._spool.write(payload)
        self._emitted.append(
            (key, dtype, tuple(int(d) for d in shape), self._cursor,
             self._cursor + len(payload)))
        self._cursor += len(payload)

    def set_metadata(self, key: str, value: str) -> None:
        self._metadata[str(key)] = str(value)

    def _entry(self, key: str) -> Mapping[str, object]:
        meta = self._header.get(key)
        if not isinstance(meta, dict):
            raise KeyError(f"no tensor {key!r} in this shard")
        return meta

    def close(self) -> Path:
        """Finalize the target shard: header, then the spooled payloads."""
        try:
            self._spool.close()
            header: Dict[str, object] = {}
            if self._metadata:
                header["__metadata__"] = dict(self._metadata)
            for key, dtype, shape, start, end in self._emitted:
                header[key] = {
                    "dtype": dtype, "shape": list(shape),
                    "data_offsets": [start, end],
                }
            blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
            tmp = self._target.parent / f".{self._target.name}.writing"
            with open(tmp, "wb") as out, open(self._spool_path, "rb") as spool:
                out.write(struct.pack("<Q", len(blob)))
                out.write(blob)
                while True:
                    chunk = spool.read(8 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.replace(self._target)
            return self._target
        finally:
            os.close(self._fd)
            self._spool_path.unlink(missing_ok=True)


def apply_rename_rules(
    keys: Sequence[str], rules: Sequence[RenameRule],
) -> Dict[str, str]:
    """Run the declared rename passes over a key set: ``{old: new}``, total."""
    out: Dict[str, str] = {}
    for key in keys:
        renamed = key
        for rule in rules:
            renamed = rule.apply(renamed)
        out[key] = renamed
    return out


@dataclass(frozen=True)
class TopologyConversion:
    """A TOPOLOGY-axis mapping: declared rename passes, and their inverse."""

    from_id: str
    to_id: str
    version: int
    rules: Tuple[RenameRule, ...]
    inverse_rules: Tuple[RenameRule, ...]
    corpus: Tuple[ConversionCase, ...]
    why: str = ""

    axis: str = field(default=AXIS_TOPOLOGY, init=False)
    proof: str = field(default="key_bijection", init=False)
    needs_torch: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not self.rules or not self.inverse_rules:
            raise DeclarationError(
                f"topology {self.from_id} -> {self.to_id}: both rules= and "
                "inverse_rules= are required. A lossless mapping is invertible "
                "by definition, and the inverse is what the admission proof "
                "runs — an edge with no stated inverse cannot be proven "
                "lossless, so it cannot be a CONVERTIBLE edge.")


@dataclass(frozen=True)
class QuantRepack:
    """A QUANT-axis SAME-NUMERICS repack: different packing, identical values."""

    from_id: str
    to_id: str
    version: int
    forward: RepackFn
    inverse: RepackFn
    equivalence: EquivalenceFn
    corpus: Tuple[ConversionCase, ...]
    needs_torch: bool = False
    why: str = ""

    axis: str = field(default=AXIS_QUANT, init=False)
    proof: str = field(default="dequant_equivalence", init=False)


LayoutConversion = Union[TopologyConversion, QuantRepack]


@dataclass(frozen=True)
class LayoutProduction:
    """The PRODUCIBLE rung: re-quantization from a named higher-precision source."""

    axis: str
    from_id: str
    to_id: str
    recipe: str
    quality_gate: str
    why: str = ""


@dataclass(frozen=True)
class ConversionHop:
    axis: str
    from_id: str
    to_id: str
    version: int
    digest: str


@dataclass(frozen=True)
class ConversionPlan:
    """How to reach ONE accepted LayoutId, per axis."""

    target: LayoutId
    topology: Tuple[ConversionHop, ...] = ()
    quant: Tuple[ConversionHop, ...] = ()

    @property
    def hops(self) -> Tuple[ConversionHop, ...]:
        return self.topology + self.quant

    @property
    def digests(self) -> Tuple[str, ...]:
        return tuple(hop.digest for hop in self.hops)


@dataclass(frozen=True)
class LayoutVerdict:
    """One rung, plus everything the caller needs to act or to refuse by name."""

    rung: LayoutRung
    source: LayoutId
    accepts: Tuple[LayoutId, ...]
    plans: Tuple[ConversionPlan, ...] = ()
    productions: Tuple[LayoutProduction, ...] = ()
    unevaluated_axes: Tuple[str, ...] = ()
    reason: str = ""


_lock = RLock()
_edges: Dict[Tuple[str, str, str], ConversionHop] = {}
_specs: Dict[Tuple[str, str, str], LayoutConversion] = {}
_directions: Dict[Tuple[str, str, str], bool] = {}
_productions: Dict[Tuple[str, str, str], LayoutProduction] = {}


def _module_content_digest(fn: Callable[..., object]) -> str:
    module = sys.modules.get(getattr(fn, "__module__", ""), None)
    path = getattr(module, "__file__", None)
    if not path:
        return "unlocatable"
    stat = Path(path).stat()
    return _closure_file_digest(str(path), stat.st_mtime_ns, stat.st_size)


def _body_digest(spec: LayoutConversion, *, forward: bool) -> str:
    if isinstance(spec, TopologyConversion):
        rules = spec.rules if forward else spec.inverse_rules
        return hashlib.sha256(json.dumps(
            [[r.kind, [list(p) for p in r.pairs]] for r in rules],
            separators=(",", ":"),
        ).encode()).hexdigest()[:16]
    return _module_content_digest(spec.forward if forward else spec.inverse)


def converter_digest(
    axis: str, from_id: str, to_id: str, version: int, body: str,
) -> str:
    """`sha256(axis ‖ from ‖ to ‖ version ‖ body digest)`."""
    payload = "\x00".join([axis, from_id, to_id, str(int(version)), body])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def derived_artifact_identity(
    source_digest: str, chain_digests: Sequence[str], target: LayoutId,
) -> str:
    """`sha256(source ‖ ordered converter digests ‖ target LayoutId)`."""
    payload = "\x00".join(
        [str(source_digest), *[str(d) for d in chain_digests], target.render()])
    return hashlib.sha256(payload.encode()).hexdigest()


def register_layout_conversion(
    spec: LayoutConversion, *, replace: bool = False,
) -> LayoutConversion:
    """Admit ONE lossless mapping — both directions — after proving it."""
    _validate_declaration(spec)
    prove_layout_conversion(spec)

    forward_key = (spec.axis, spec.from_id, spec.to_id)
    inverse_key = (spec.axis, spec.to_id, spec.from_id)
    with _lock:
        for key in (forward_key, inverse_key):
            existing = _specs.get(key)
            if existing is not None and existing != spec and not replace:
                raise DeclarationError(
                    f"layout conversion {key[0]} {key[1]} -> {key[2]} is "
                    "already registered with a different declaration; pass "
                    "replace=True only if you own both")
        for key, is_forward in ((forward_key, True), (inverse_key, False)):
            _specs[key] = spec
            _directions[key] = is_forward
            _edges[key] = ConversionHop(
                axis=key[0], from_id=key[1], to_id=key[2],
                version=spec.version,
                digest=converter_digest(
                    key[0], key[1], key[2], spec.version,
                    _body_digest(spec, forward=is_forward)),
            )
    return spec


def register_layout_production(
    spec: LayoutProduction, *, replace: bool = False,
) -> LayoutProduction:
    """Declare a PRODUCIBLE edge: new numerics from a named source, by recipe."""
    if spec.axis not in LAYOUT_AXES:
        raise DeclarationError(
            f"unknown layout axis {spec.axis!r}; the axes are {list(LAYOUT_AXES)}")
    where = f"LayoutProduction({spec.from_id} -> {spec.to_id})"
    validate_layout_handle(spec.from_id, where=where, axis=spec.axis)
    validate_layout_handle(spec.to_id, where=where, axis=spec.axis)
    if spec.from_id == spec.to_id:
        raise DeclarationError(f"{where}: a production to itself is not an edge")
    if not str(spec.recipe or "").strip():
        raise DeclarationError(f"{where}: recipe is empty")
    if not str(spec.quality_gate or "").strip():
        raise DeclarationError(
            f"{where}: quality_gate is empty — producing new numerics without "
            "naming the gate that will judge them is the decision §4.32 puts "
            "on the author")
    key = (spec.axis, spec.from_id, spec.to_id)
    with _lock:
        existing = _productions.get(key)
        if existing is not None and existing != spec and not replace:
            raise DeclarationError(
                f"layout production {key} is already registered with a "
                "different declaration")
        _productions[key] = spec
    return spec


def registered_layout_conversions() -> Tuple[ConversionHop, ...]:
    """Every registered EDGE, both directions, in a deterministic order."""
    with _lock:
        return tuple(sorted(
            _edges.values(), key=lambda e: (e.axis, e.from_id, e.to_id)))


def registered_layout_productions() -> Tuple[LayoutProduction, ...]:
    with _lock:
        return tuple(sorted(
            _productions.values(), key=lambda p: (p.axis, p.from_id, p.to_id)))


def reset_layout_conversions() -> None:
    """Drop every registration."""
    with _lock:
        _edges.clear()
        _specs.clear()
        _directions.clear()
        _productions.clear()


def _validate_declaration(spec: LayoutConversion) -> None:
    where = f"{spec.axis} conversion {spec.from_id} -> {spec.to_id}"
    validate_layout_handle(spec.from_id, where=where, axis=spec.axis)
    validate_layout_handle(spec.to_id, where=where, axis=spec.axis)
    if spec.from_id == spec.to_id:
        raise DeclarationError(f"{where}: a conversion to itself is not an edge")
    if int(spec.version) < 1:
        raise DeclarationError(f"{where}: version must be >= 1")
    if not spec.corpus:
        raise DeclarationError(
            f"{where}: declares no corpus. A mapping ships with at least one "
            "REAL artifact header (§4.25 / th#1580's corpus method) — that is "
            "what the bit-exactness obligation runs against.")


def prove_layout_conversion(spec: LayoutConversion) -> Tuple[str, ...]:
    """Run the mapping's bit-exactness obligation over its own corpus."""
    passed: List[str] = []
    with tempfile.TemporaryDirectory(prefix="cozy-layout-proof-") as tmp:
        root = Path(tmp)
        for case in spec.corpus:
            source = materialize_case(case, root / f"{case.name}.src.safetensors")
            forward = root / f"{case.name}.fwd.safetensors"
            back = root / f"{case.name}.rt.safetensors"
            _run_hop_or_raise(spec, source, forward, forward=True, case=case)
            _prove_keys(spec, case, source, forward)
            passed.append(f"{case.name}:key_bijection")
            if isinstance(spec, TopologyConversion):
                _prove_payload_invariance(spec, case, source, forward)
                passed.append(f"{case.name}:payload_invariance")
            _run_hop_or_raise(spec, forward, back, forward=False, case=case)
            before, after = shard_content_digest(source), shard_content_digest(back)
            if before != after:
                raise ConversionProofError(
                    f"{spec.axis} {spec.from_id} -> {spec.to_id}: the round "
                    f"trip did not recover corpus case {case.name!r} "
                    f"({before} -> {after}). A CONVERTIBLE mapping is lossless, "
                    "and lossless means invertible: information the forward "
                    "mapping dropped cannot come back. A mapping that produces "
                    "NEW NUMERICS is the PRODUCIBLE rung — declare it with "
                    "register_layout_production(), which prices it and names "
                    "its quality gate.")
            passed.append(f"{case.name}:round_trip")
            if isinstance(spec, QuantRepack):
                try:
                    spec.equivalence(source, forward)
                except Exception as exc:  # noqa: BLE001 - re-raised typed
                    raise ConversionProofError(
                        f"quant {spec.from_id} -> {spec.to_id}: reference "
                        f"dequant disagreed on corpus case {case.name!r}: {exc}"
                    ) from exc
                passed.append(f"{case.name}:dequant_equivalence")
    return tuple(passed)


def _run_hop_or_raise(
    spec: LayoutConversion, source: Path, target: Path, *,
    forward: bool, case: ConversionCase,
) -> None:
    try:
        _apply_hop(spec, source, target, forward=forward)
    except Exception as exc:  # noqa: BLE001 - re-raised typed
        direction = "forward" if forward else "inverse"
        raise ConversionProofError(
            f"{spec.axis} {spec.from_id} -> {spec.to_id}: {direction} mapping "
            f"failed on corpus case {case.name!r}: {exc}") from exc


def _apply_hop(
    spec: LayoutConversion, source: Path, target: Path, *, forward: bool,
) -> None:
    if isinstance(spec, TopologyConversion):
        rules = spec.rules if forward else spec.inverse_rules
        keys = tuple(name for name, _ in shard_tensor_entries(source))
        rewrite_safetensors_keys(
            source, target, apply_rename_rules(keys, rules))
        return
    repack: RepackFn = spec.forward if forward else spec.inverse
    io = ConversionIO(source, target)
    try:
        repack(io)
    finally:
        io.close()


def _prove_keys(
    spec: LayoutConversion, case: ConversionCase, source: Path, target: Path,
) -> None:
    src_keys = [name for name, _ in shard_tensor_entries(source)]
    dst_keys = [name for name, _ in shard_tensor_entries(target)]
    if len(dst_keys) != len(set(dst_keys)):
        raise ConversionProofError(
            f"{spec.axis} {spec.from_id} -> {spec.to_id}: corpus case "
            f"{case.name!r} produced a duplicate key — the map is not injective")
    if len(dst_keys) != len(set(src_keys)):
        dropped = len(set(src_keys)) - len(dst_keys)
        raise ConversionProofError(
            f"{spec.axis} {spec.from_id} -> {spec.to_id}: corpus case "
            f"{case.name!r} has {len(set(src_keys))} source keys and "
            f"{len(dst_keys)} output keys ({dropped:+d}). An unmapped key is a "
            "REFUSAL, never a silent skip, and an invented key is worse.")


def _prove_payload_invariance(
    spec: TopologyConversion, case: ConversionCase, source: Path, target: Path,
) -> None:
    before = sorted(shard_payload_digests(source).values())
    after = sorted(shard_payload_digests(target).values())
    if before != after:
        raise ConversionProofError(
            f"topology {spec.from_id} -> {spec.to_id}: corpus case "
            f"{case.name!r} changed payload bytes. The topology axis is keys "
            "only — 'what the weights ARE' is the quant axis.")


def _axis_satisfied(demand: Optional[str], supply: Optional[str]) -> bool:
    if demand is None or demand == LAYOUT_AXIS_ANY:
        return True
    return demand == supply


def _unevaluated(source: LayoutId, accepts: Sequence[LayoutId]) -> Tuple[str, ...]:
    return tuple(
        axis for axis in LAYOUT_AXES
        if source.axis(axis) is None
        or all(a.axis(axis) is None for a in accepts)
    )


def _shortest_chain(
    axis: str, source: Optional[str], target: Optional[str],
) -> Optional[Tuple[ConversionHop, ...]]:
    if _axis_satisfied(target, source):
        return ()
    if source is None or source == LAYOUT_AXIS_ANY or target is None:
        return None
    with _lock:
        edges = [e for e in _edges.values() if e.axis == axis]
    queue: deque[Tuple[str, Tuple[ConversionHop, ...]]] = deque([(source, ())])
    seen = {source}
    while queue:
        node, chain = queue.popleft()
        if len(chain) >= MAX_PLAN_HOPS_PER_AXIS:
            continue
        for edge in sorted(edges, key=lambda e: (e.from_id, e.to_id)):
            if edge.from_id != node or edge.to_id in seen:
                continue
            extended = chain + (edge,)
            if edge.to_id == target:
                return extended
            seen.add(edge.to_id)
            queue.append((edge.to_id, extended))
    return None


def plan_layout_conversions(
    source: LayoutId, accepts: Sequence[LayoutId],
) -> Tuple[ConversionPlan, ...]:
    """Every accepted LayoutId reachable from ``source`` over LOSSLESS edges."""
    plans: List[ConversionPlan] = []
    for target in accepts:
        topology = _shortest_chain(
            AXIS_TOPOLOGY, source.topology, target.topology)
        if topology is None:
            continue
        quant = _shortest_chain(AXIS_QUANT, source.quant, target.quant)
        if quant is None:
            continue
        if not topology and not quant:
            continue
        plans.append(ConversionPlan(
            target=target, topology=topology, quant=quant))
    return tuple(sorted(
        plans, key=lambda p: (len(p.hops), p.target.render())))


def _reachable_productions(
    source: LayoutId, accepts: Sequence[LayoutId],
) -> Tuple[LayoutProduction, ...]:
    with _lock:
        productions = list(_productions.values())
    out: List[LayoutProduction] = []
    for production in productions:
        if source.axis(production.axis) != production.from_id:
            continue
        candidate = source.with_axis(production.axis, production.to_id)
        for target in accepts:
            if all(_axis_satisfied(target.axis(a), candidate.axis(a))
                   for a in LAYOUT_AXES):
                out.append(production)
                break
    return tuple(sorted(out, key=lambda p: (p.axis, p.from_id, p.to_id)))


def classify_layout(
    source: LayoutId, accepts: Sequence[LayoutId],
) -> LayoutVerdict:
    """The ladder for ONE (component, supply, demand): the whole verdict."""
    accepted = tuple(accepts)
    unevaluated = _unevaluated(source, accepted)
    if not accepted:
        return LayoutVerdict(
            rung=LayoutRung.INCOMPATIBLE, source=source, accepts=(),
            unevaluated_axes=unevaluated,
            reason="the demand is UNDECLARED: no accepted layout to match "
                   "against. An empty demand is not 'accepts everything'.")
    for target in accepted:
        if all(_axis_satisfied(target.axis(axis), source.axis(axis))
               for axis in LAYOUT_AXES):
            return LayoutVerdict(
                rung=LayoutRung.COMPATIBLE, source=source, accepts=accepted,
                unevaluated_axes=unevaluated,
                reason=f"{source.render()} is in the accepted set")
    plans = plan_layout_conversions(source, accepted)
    if plans:
        return LayoutVerdict(
            rung=LayoutRung.CONVERTIBLE, source=source, accepts=accepted,
            plans=plans, unevaluated_axes=unevaluated,
            reason=f"{len(plans)} lossless plan(s) reach an accepted layout")
    productions = _reachable_productions(source, accepted)
    if productions:
        return LayoutVerdict(
            rung=LayoutRung.PRODUCIBLE, source=source, accepts=accepted,
            productions=productions, unevaluated_axes=unevaluated,
            reason="no lossless mapping; a registered production recipe can "
                   "make an accepted layout from this source — priced, offered, "
                   "never automatic")
    return LayoutVerdict(
        rung=LayoutRung.INCOMPATIBLE, source=source, accepts=accepted,
        unevaluated_axes=unevaluated,
        reason=f"{source.render()} is not accepted, no registered lossless "
               f"mapping reaches "
               f"{', '.join(t.render() for t in accepted)}, and no production "
               f"recipe produces one")


CONVERSION_PROVENANCE_KEY = "cozy.conversion"

PRODUCED_BY_LOCAL = "local_conversion"


@dataclass(frozen=True)
class ConversionResult:
    path: Path
    target: LayoutId
    identity: str
    chain: Tuple[ConversionHop, ...]


def run_layout_conversion(
    plan: ConversionPlan, source: Path, target: Path, *,
    source_digest: str, produced_by: str,
) -> ConversionResult:
    """Execute a plan, hop by hop, and stamp the chain onto the output."""
    if not str(produced_by or "").strip():
        raise DeclarationError(
            "run_layout_conversion(produced_by=) is required: an artifact that "
            "cannot say where it was converted cannot be judged by the publish "
            "fence (§4.28)")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    hops = plan.hops
    if not hops:
        raise DeclarationError("run_layout_conversion: the plan has no hops")

    current = Path(source)
    with tempfile.TemporaryDirectory(
            prefix="cozy-layout-", dir=str(target.parent)) as tmp:
        for index, hop in enumerate(hops):
            key = (hop.axis, hop.from_id, hop.to_id)
            with _lock:
                spec = _specs.get(key)
                forward = _directions.get(key)
            if spec is None or forward is None:
                raise DeclarationError(
                    f"no registered converter for {hop.axis} {hop.from_id} -> "
                    f"{hop.to_id}; the plan names an edge this process does "
                    "not hold")
            last = index == len(hops) - 1
            out = target if last else Path(tmp) / f"hop{index}.safetensors"
            _apply_hop(spec, current, out, forward=forward)
            current = out

    identity = derived_artifact_identity(source_digest, plan.digests, plan.target)
    _stamp_provenance(target, plan, source_digest, identity, produced_by)
    return ConversionResult(
        path=target, target=plan.target, identity=identity, chain=hops)


def _stamp_provenance(
    path: Path, plan: ConversionPlan, source_digest: str, identity: str,
    produced_by: str,
) -> None:
    provenance = json.dumps({
        "v": 1,
        "identity": identity,
        "source_digest": source_digest,
        "target_layout": plan.target.render(),
        "produced_by": produced_by,
        "chain": [
            {"axis": h.axis, "from": h.from_id, "to": h.to_id,
             "version": h.version, "converter_digest": h.digest}
            for h in plan.hops
        ],
    }, separators=(",", ":"), sort_keys=True)
    rewrite_safetensors_keys(
        path, path, {}, extra_metadata={CONVERSION_PROVENANCE_KEY: provenance})


def conversion_provenance(path: Path) -> Optional[Mapping[str, object]]:
    """The conversion chain an artifact carries, or ``None``."""
    with open(path, "rb") as fh:
        header, _ = read_safetensors_header(fh.fileno())
    raw = (header.get("__metadata__") or {}).get(CONVERSION_PROVENANCE_KEY)
    if not isinstance(raw, str):
        return None
    decoded = json.loads(raw)
    return decoded if isinstance(decoded, dict) else None


__all__ = [
    "CONVERSION_PROVENANCE_KEY",
    "ConversionCase",
    "ConversionHop",
    "ConversionIO",
    "ConversionPlan",
    "ConversionProofError",
    "ConversionResult",
    "CorpusTensor",
    "LayoutConversion",
    "LayoutProduction",
    "LayoutRung",
    "LayoutVerdict",
    "PRODUCED_BY_LOCAL",
    "QuantRepack",
    "TopologyConversion",
    "apply_rename_rules",
    "classify_layout",
    "conversion_provenance",
    "converter_digest",
    "derived_artifact_identity",
    "materialize_case",
    "plan_layout_conversions",
    "prove_layout_conversion",
    "register_layout_conversion",
    "register_layout_production",
    "registered_layout_conversions",
    "registered_layout_productions",
    "reset_layout_conversions",
    "run_layout_conversion",
]
