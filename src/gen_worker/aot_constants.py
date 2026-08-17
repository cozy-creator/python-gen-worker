"""The constant table a compiled graph binds, sourced from the STORE.

``aot_serve.resident_constants`` reads a LIVE ``nn.Module``: serving one
compiled graph therefore required the whole eager pipeline resident purely as
a constant SOURCE, which is why an adopt-only serve host (pgw#1328) could not
shed diffusers. The artifact already carries the identity that makes the
module unnecessary — every constant is declared by its fully-qualified name,
its torch dtype and its exact shape — so the same by-reference mapping can be
built from safetensors bytes the store already holds.

Two phases, in this order and never merged:

1. :func:`plan_store_constants` — headers only. Every declared
   ``state_dict`` FQN must be present, with the DECLARED dtype and the
   DECLARED shape. A miss is a typed refusal naming every FQN at fault, and
   it happens before one device byte is allocated and before any caller has
   mutated dispatch state.
2. :func:`realize_store_constants` — reads the planned FQNs onto the device.

The split is the point. A store that is one tensor short, or one dtype off,
must cost a header read, not a half-populated 20 GiB constant table and a
runner that can never be un-bound (``CompiledGraphRunner.bind`` is
once-only and marks itself failed).

The bytes come through pgw#1330's one reader: ``read_header`` for the index
and ``open_tensor_source`` for the values, so a component whose weights are
CAS objects behind projection stubs indexes and binds without materializing
anything. Nothing here imports ``torch.nn``, diffusers, or any model class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NewType,
    Protocol,
    Sequence,
    Tuple,
)

from .models.safetensors_header import read_header
from .models.tensor_source import TORCH_DTYPES, open_tensor_source

logger = logging.getLogger(__name__)

#: THE constant-manifest schema version, and it is deliberately not a new
#: number. The constants block lives inside the compiled-graph artifact
#: metadata envelope, so its version IS that envelope's version
#: (``aot_serve.COMPILED_GRAPH_FORMAT``). §1.38b's rule is that the compiled
#: boundaries each begin at their own v1 and are INDEPENDENT — it is not a
#: licence to mint a second version number for one envelope, which is exactly
#: how a reader ends up comparing the wrong two integers (pgw#1230).
CONSTANT_MANIFEST_FORMAT = 1

#: The exact field set a manifest constant row carries. TCG's artifact
#: validator already enforces this on the WRITE side
#: (``artifact._validated_constants``); repeating it on the read side is not
#: redundancy — an artifact this worker did not mint reaches this parser, and
#: a row with an extra field is an artifact from a format this reader does not
#: understand, not a row to read three keys out of and hope.
_ROW_FIELDS = frozenset(("fqn", "source", "dtype", "shape"))

# ---------------------------------------------------------------------------
# Validated identifiers — parse, never a bare ``str``
# ---------------------------------------------------------------------------

#: A constant's fully-qualified name inside the exported target, e.g.
#: ``transformer_blocks.0.attn.to_q.weight``. Parsed, because it is the ONLY
#: thing that ties an artifact slot to a store tensor: a stray space or an
#: empty segment is a name that can never resolve, and it must say so at the
#: boundary rather than as a ``KeyError`` under a bind.
ConstantFQN = NewType("ConstantFQN", str)

#: A torch dtype attribute name (``bfloat16``, not ``torch.bfloat16`` and not
#: ``BF16``). Validated against ``torch``'s own attribute table at read time.
TorchDtype = NewType("TorchDtype", str)

#: The graph class the constant table belongs to.
GraphClassName = NewType("GraphClassName", str)

#: WHICH weight-set a store supplies — the checkpoint ref, digest or snapshot
#: id the constants were read from.
#:
#: This is the one INSTANCE-level fact on this path, and it is carried
#: separately from the graph-class identity on purpose (§4.27: class identity
#: is checkpoint-free by construction). A graph class is the family level; a
#: graph class bound to one weight-set is an instance of it, and the same
#: ``.so`` serving sixteen fine-tunes is sixteen instances of one class. A
#: store-sourced arm is therefore keyed by (compiled-graph key, weight-set
#: ref) — the module-sourced arm could only ever say "whatever that module
#: happened to hold", which is precisely the fact an instance needs and a
#: resident module cannot state.
WeightSetRef = NewType("WeightSetRef", str)


class ConstantManifestError(ValueError):
    """The artifact's declared constant table cannot be read as v1."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(detail)


class ConstantResolutionError(ValueError):
    """The store cannot supply the declared constant table, by name.

    ``fqns`` names every constant at fault, not the first one: a wrong
    component directory misses hundreds and the operator needs the count and
    a sample, while a genuine drift misses one and the operator needs which.
    """

    def __init__(self, reason: str, detail: str, fqns: Sequence[str] = ()) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        self.fqns: Tuple[str, ...] = tuple(str(name) for name in fqns)
        super().__init__(detail)


class ConstantStoreError(ValueError):
    """The store itself is unreadable — malformed shard, unknown dtype."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(detail)


def parse_fqn(raw: object) -> ConstantFQN:
    """Parse one constant FQN, or refuse it."""

    if not isinstance(raw, str):
        raise ConstantManifestError(
            "fqn_not_a_string", f"constant fqn must be a string, got {type(raw).__name__}"
        )
    if not raw or raw != raw.strip():
        raise ConstantManifestError(
            "fqn_malformed", f"constant fqn must be non-empty and unpadded: {raw!r}"
        )
    if any(segment == "" for segment in raw.split(".")):
        raise ConstantManifestError(
            "fqn_malformed", f"constant fqn has an empty dotted segment: {raw!r}"
        )
    return ConstantFQN(raw)


def parse_torch_dtype(raw: object) -> TorchDtype:
    """Parse one torch dtype attribute name, or refuse it."""

    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ConstantManifestError(
            "dtype_malformed", f"constant dtype must be a non-empty bare name: {raw!r}"
        )
    if raw.startswith("torch."):
        raise ConstantManifestError(
            "dtype_malformed",
            f"constant dtype is an attribute NAME, not a repr: {raw!r}",
        )
    return TorchDtype(raw)


def parse_weight_set_ref(raw: object) -> WeightSetRef:
    """Parse the ref naming WHICH weight-set a store supplies, or refuse it.

    A store that cannot say which checkpoint it holds cannot be the source of
    an instance-level binding, and an unnamed binding is exactly what makes
    "which weights is this pod actually serving?" unanswerable today.
    """

    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ConstantManifestError(
            "weight_set_ref_malformed",
            f"a constant store must name the weight-set it supplies: {raw!r}",
        )
    return WeightSetRef(raw)


def parse_graph_class_name(raw: object) -> GraphClassName:
    if not isinstance(raw, str) or not raw or raw != raw.strip():
        raise ConstantManifestError(
            "graph_class_malformed", f"graph class name must be non-empty: {raw!r}"
        )
    return GraphClassName(raw)


def _parse_shape(raw: object, fqn: ConstantFQN) -> Tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ConstantManifestError(
            "shape_malformed", f"constant {fqn!r} shape must be an array"
        )
    dimensions: List[int] = []
    for dimension in raw:
        # ``bool`` is an ``int`` subclass and a shape of ``[True]`` is a
        # malformed artifact, not a shape of ``[1]``.
        if type(dimension) is not int or dimension < 0:
            raise ConstantManifestError(
                "shape_malformed",
                f"constant {fqn!r} shape must be non-negative integers, got {dimension!r}",
            )
        dimensions.append(dimension)
    return tuple(dimensions)


class ConstantSourceKind(str, Enum):
    """Where one declared constant's value comes from.

    Mirrors ``torchcg.introspection.DeclaredConstant.source``, as an enum
    rather than a string tag because this module BRANCHES on it and a typo in
    a comparison is otherwise a silently skipped constant.
    """

    STATE_DICT = "state_dict"
    LITERAL = "literal"
    COMPUTED = "computed"


@dataclass(frozen=True, slots=True)
class TensorFacts:
    """The dtype and exact shape of one tensor, from either side."""

    dtype: TorchDtype
    shape: Tuple[int, ...]

    @property
    def elements(self) -> int:
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total


@dataclass(frozen=True, slots=True)
class ConstantSpec:
    """One validated row of the artifact's declared constant table."""

    fqn: ConstantFQN
    source: ConstantSourceKind
    facts: TensorFacts


@dataclass(frozen=True, slots=True)
class ConstantManifest:
    """The artifact's whole declared constant table, v1, validated."""

    graph_class: GraphClassName
    target: str
    constants: Tuple[ConstantSpec, ...]

    @property
    def store_sourced(self) -> Tuple[ConstantSpec, ...]:
        """The constants a STORE must supply.

        ``literal`` rows ride inside the artifact and ``computed`` rows are
        constant-folded into it; TCG's runner resolves both without this
        module. Only ``state_dict`` rows ever needed the eager module.
        """

        return tuple(
            spec for spec in self.constants if spec.source is ConstantSourceKind.STATE_DICT
        )


def parse_constant_manifest(
    graph_class: Mapping[str, Any], *, compiled_graph_format: object
) -> ConstantManifest:
    """Read one artifact's ``graph_class`` block as a v1 constant manifest.

    ``compiled_graph_format`` is the artifact envelope's own version stamp.
    An envelope this reader does not know is refused BEFORE its rows are
    interpreted — reading v2 rows with a v1 reader is how a field that
    changed meaning becomes a wrong tensor rather than an error.
    """

    if compiled_graph_format != CONSTANT_MANIFEST_FORMAT:
        raise ConstantManifestError(
            "manifest_version_unsupported",
            f"constant manifest reader is v{CONSTANT_MANIFEST_FORMAT}; artifact "
            f"declares compiled_graph_format={compiled_graph_format!r}",
        )
    name = parse_graph_class_name(graph_class.get("name"))
    raw_target = graph_class.get("target")
    if not isinstance(raw_target, str) or not raw_target.strip():
        raise ConstantManifestError(
            "target_malformed", f"graph class {name!r} names no target"
        )
    rows = graph_class.get("constants")
    if not isinstance(rows, (list, tuple)):
        raise ConstantManifestError(
            "constants_malformed", f"graph class {name!r} constants must be an array"
        )
    specs: List[ConstantSpec] = []
    seen: Dict[ConstantFQN, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ConstantManifestError(
                "constant_row_malformed", f"constant {index} must be an object"
            )
        if set(row) != _ROW_FIELDS:
            raise ConstantManifestError(
                "constant_row_malformed",
                f"constant {index} fields must be exactly {sorted(_ROW_FIELDS)!r}, "
                f"got {sorted(str(field) for field in row)!r}",
            )
        fqn = parse_fqn(row.get("fqn"))
        if fqn in seen:
            raise ConstantManifestError(
                "constant_duplicate",
                f"constant {fqn!r} declared twice (rows {seen[fqn]} and {index})",
            )
        seen[fqn] = index
        raw_source = row.get("source")
        try:
            source = ConstantSourceKind(raw_source)
        except ValueError as exc:
            raise ConstantManifestError(
                "constant_source_unknown",
                f"constant {fqn!r} declares source {raw_source!r}, which is not one of "
                f"{sorted(kind.value for kind in ConstantSourceKind)!r}",
            ) from exc
        specs.append(
            ConstantSpec(
                fqn=fqn,
                source=source,
                facts=TensorFacts(
                    dtype=parse_torch_dtype(row.get("dtype")),
                    shape=_parse_shape(row.get("shape"), fqn),
                ),
            )
        )
    return ConstantManifest(
        graph_class=name, target=raw_target.strip(), constants=tuple(specs)
    )


# ---------------------------------------------------------------------------
# The store side
# ---------------------------------------------------------------------------


class ConstantStore(Protocol):
    """Everything the arm needs from a weight source, and nothing more.

    Deliberately NOT ``safe_open``'s shape: an arm must be able to check the
    whole declared table against headers before it reads a byte, which
    ``get_tensor``-only sources cannot express.
    """

    @property
    def weight_set(self) -> WeightSetRef:
        """WHICH weight-set this store supplies — the instance-level fact."""

    def describe(self) -> Mapping[ConstantFQN, TensorFacts]:
        """Every tensor this store holds, by FQN, from headers alone."""

    def read(self, fqn: ConstantFQN, *, device: str) -> Any:
        """One tensor, on ``device``. Only ever called for planned FQNs."""


@dataclass(frozen=True, slots=True)
class StoreConstantPlan:
    """A checked promise that this store can supply this table.

    Holding it is the proof the two-phase split exists: a caller cannot get
    one without every declared ``state_dict`` FQN having matched the store on
    name, dtype and shape, and cannot read a constant that is not in it.
    """

    graph_class: GraphClassName
    weight_set: WeightSetRef
    fqns: Tuple[ConstantFQN, ...]
    elements: int

    def __len__(self) -> int:
        return len(self.fqns)


def plan_store_constants(
    manifest: ConstantManifest, store: ConstantStore
) -> StoreConstantPlan:
    """Check the whole declared table against the store's HEADERS.

    Refuses, naming every FQN at fault, on:

    ``constant_absent``
        a declared ``state_dict`` FQN the store does not hold. This is the
        wrong-component-directory and the checkpoint-drift case, and it is
        the one that must never reach a bind: ``CompiledGraphRunner.bind`` is
        once-only, so a partial table costs the runner as well as the memory.
    ``constant_dtype_mismatch``
        present but a different dtype. The graph is SPECIALIZED on dtype;
        binding an fp32 tensor where the artifact wants bf16 is a wrong
        answer at full speed, not a crash.
    ``constant_shape_mismatch``
        present but a different shape, i.e. a different checkpoint.

    No device memory is allocated on any path through this function.
    """

    index = store.describe()
    absent: List[str] = []
    dtype_gap: List[str] = []
    shape_gap: List[str] = []
    planned: List[ConstantFQN] = []
    elements = 0
    for spec in manifest.store_sourced:
        held = index.get(spec.fqn)
        if held is None:
            absent.append(str(spec.fqn))
            continue
        if held.dtype != spec.facts.dtype:
            dtype_gap.append(f"{spec.fqn} (declared {spec.facts.dtype}, store {held.dtype})")
            continue
        if held.shape != spec.facts.shape:
            shape_gap.append(
                f"{spec.fqn} (declared {list(spec.facts.shape)}, store {list(held.shape)})"
            )
            continue
        planned.append(spec.fqn)
        elements += spec.facts.elements
    for reason, faults in (
        ("constant_absent", absent),
        ("constant_dtype_mismatch", dtype_gap),
        ("constant_shape_mismatch", shape_gap),
    ):
        if faults:
            raise ConstantResolutionError(
                reason,
                f"graph class {manifest.graph_class!r}: {len(faults)} declared "
                f"constant(s) {reason.removeprefix('constant_')} in the store: "
                f"{sorted(faults)[:6]!r}",
                faults,
            )
    return StoreConstantPlan(
        graph_class=manifest.graph_class,
        weight_set=parse_weight_set_ref(store.weight_set),
        fqns=tuple(planned),
        elements=elements,
    )


def realize_store_constants(
    plan: StoreConstantPlan, store: ConstantStore, *, device: str
) -> Dict[str, Any]:
    """Read the planned constants onto ``device``, by reference.

    The mapping is handed straight to ``CompiledGraphRunner.bind``, which
    installs the pointers ``user_managed`` — so these tensors must outlive
    the runner, and the caller owns that lifetime exactly as the
    module-sourced arm does.
    """

    target = str(device).strip()
    if not target:
        raise ConstantResolutionError(
            "device_missing", "store-sourced constants require an explicit device"
        )
    values: Dict[str, Any] = {}
    for fqn in plan.fqns:
        values[str(fqn)] = store.read(fqn, device=target)
    return values


# ---------------------------------------------------------------------------
# The safetensors implementation
# ---------------------------------------------------------------------------


def _read_header(path: Path, why: str) -> Mapping[str, Any]:
    """One safetensors header, projection-aware and fail-CLOSED.

    ``safetensors_header.read_header`` is the ONE reader (pgw#1330) and it
    serves a projected tree's stub from the manifest, which is what lets this
    store index a component whose bytes are CAS objects rather than files.
    Its ``{}`` means "no readable header here" — the honest answer for a
    truncated or non-safetensors file, and a fail-open its own callers need.

    It is NOT an answer this store may keep. A component shard that will not
    report a header contributes no FQNs, and a caller that reads that as
    "these constants are absent" refuses the wrong thing (or, worse, plans
    around a table it never saw). Here the empty header is a refusal.
    """

    header = read_header(path, why=why)
    if not header:
        raise ConstantStoreError(
            "shard_unreadable",
            f"{path}: no readable safetensors header. A shard that will not "
            f"report its tensors is unreadable, not empty ({why})",
        )
    return header


def _facts_from_header_row(path: Path, name: str, row: object) -> TensorFacts:
    if not isinstance(row, Mapping):
        raise ConstantStoreError(
            "shard_unreadable", f"{path}: header entry {name!r} is not an object"
        )
    raw_dtype = row.get("dtype")
    if not isinstance(raw_dtype, str) or raw_dtype not in TORCH_DTYPES:
        raise ConstantStoreError(
            "dtype_unknown",
            f"{path}: tensor {name!r} has safetensors dtype {raw_dtype!r}, whose "
            f"element width is unknown here. Refusing to guess.",
        )
    raw_shape = row.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or not all(
        type(dimension) is int and dimension >= 0 for dimension in raw_shape
    ):
        raise ConstantStoreError(
            "shard_unreadable", f"{path}: tensor {name!r} has no integer shape"
        )
    return TensorFacts(
        dtype=TorchDtype(TORCH_DTYPES[raw_dtype]),
        shape=tuple(int(dimension) for dimension in raw_shape),
    )


class SafetensorsConstantStore:
    """A component's safetensors shards, indexed by FQN.

    The index is built from HEADERS — the whole point of the two-phase plan is
    that a table can be refused for the cost of a few kilobytes per shard,
    whatever the component weighs.

    Tensor names in a diffusers component's shards are already the FQNs the
    exported target declares; this store therefore does no renaming at all.
    A name that does not line up is reported by :func:`plan_store_constants`
    as absent, by name, rather than repaired by a prefix heuristic — a
    heuristic here would bind the wrong tensor under the right name, which is
    the one failure this whole path exists to make impossible.
    """

    def __init__(
        self, shards: Iterable[Path], *, weight_set: str, why: str
    ) -> None:
        self._why = str(why)
        self._weight_set = parse_weight_set_ref(weight_set)
        self._shards: Tuple[Path, ...] = tuple(sorted(Path(shard) for shard in shards))
        if not self._shards:
            raise ConstantStoreError(
                "store_empty",
                f"no safetensors shard to source constants from ({self._why})",
            )
        self._index: Dict[ConstantFQN, TensorFacts] = {}
        self._holder: Dict[ConstantFQN, Path] = {}
        for shard in self._shards:
            for name, row in _read_header(shard, self._why).items():
                if name == "__metadata__":
                    continue
                fqn = ConstantFQN(str(name))
                if fqn in self._holder:
                    raise ConstantStoreError(
                        "tensor_duplicated",
                        f"tensor {fqn!r} is in both {self._holder[fqn]} and {shard}; "
                        f"a sharded component that names one tensor twice has no "
                        f"single answer and must not be guessed at ({self._why})",
                    )
                self._holder[fqn] = shard
                self._index[fqn] = _facts_from_header_row(shard, str(name), row)

    @classmethod
    def for_component(
        cls, directory: Path, *, weight_set: str, why: str
    ) -> "SafetensorsConstantStore":
        """Every ``*.safetensors`` shard directly inside one component dir."""

        root = Path(directory)
        return cls(
            sorted(root.glob("*.safetensors")), weight_set=weight_set, why=why
        )

    @property
    def weight_set(self) -> WeightSetRef:
        return self._weight_set

    @property
    def shards(self) -> Tuple[Path, ...]:
        return self._shards

    def describe(self) -> Mapping[ConstantFQN, TensorFacts]:
        return dict(self._index)

    def read(self, fqn: ConstantFQN, *, device: str) -> Any:
        shard = self._holder.get(fqn)
        if shard is None:
            raise ConstantResolutionError(
                "constant_absent",
                f"{fqn!r} is not in this store; only planned constants may be read",
                (str(fqn),),
            )
        with open_tensor_source(shard, device=str(device), why=self._why) as source:
            return source.get_tensor(str(fqn))


__all__ = [
    "CONSTANT_MANIFEST_FORMAT",
    "ConstantFQN",
    "ConstantManifest",
    "ConstantManifestError",
    "ConstantResolutionError",
    "ConstantSourceKind",
    "ConstantSpec",
    "ConstantStore",
    "ConstantStoreError",
    "GraphClassName",
    "SafetensorsConstantStore",
    "StoreConstantPlan",
    "TensorFacts",
    "TorchDtype",
    "WeightSetRef",
    "parse_constant_manifest",
    "parse_fqn",
    "parse_graph_class_name",
    "parse_torch_dtype",
    "parse_weight_set_ref",
    "plan_store_constants",
    "realize_store_constants",
]
