"""Versioned ingress selection: which declared graph class serves one call.

`CallIngress` states what ONE graph class accepts. This module states what a
serve host does with SEVERAL of them: which classes a call disqualifies, how the
refused ones are ORDERED, the two feed normalizations a host may perform, and
what a total miss is. Those semantics lived only in gen-worker's `aot_serve`
(tcg#37), where a second serve host would have had to reverse-engineer them from
Python source — the thing `torchcg.contracts` exists to prevent.

The corpus `contracts/ingress_selection_v1.json` is the cross-language artifact;
everything here is its reference implementation, and `tests/test_selection.py`
replays every vector through it.

**Scope.** Selection only. The candidate set is the host's — this module never
asks why a class is absent, unarmed, or pending a compile. What a host DOES with
an outcome (eager fallback, sticky de-arm, shape-growth reporting, event
emission, refusal wording) is host policy and stays there.

**Torch-free by construction.** Selection reads four facts per feed — dtype
name, shape, contiguity, pointer alignment — so the contract is expressible as
JSON and executable without an accelerator. :func:`describe_call` is the adapter
that reads them off a real call by duck typing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from .ingress import CallIngress, CallInput

#: The corpus schema this module reads and writes. A corpus stamping any other
#: value is refused, never best-effort decoded: a host that silently accepted a
#: newer selection vocabulary would disagree about admission with the fleet that
#: wrote it, and admission disagreements are invisible until they serve wrong.
SELECTION_CONTRACT_VERSION: Final = 1

#: The packaged corpus name, in `torchcg.contracts`.
SELECTION_CONTRACT_FILE: Final = "ingress_selection_v1.json"

#: AOTInductor generates its aligned-input fast path at 16 bytes
#: (`torch._inductor.codegen.aoti_runtime`'s ALIGNMENT). Not a knob: it is the
#: compiler's constant, and a feed that misses it makes the runner clone the
#: tensor on every call.
AOTI_ALIGNMENT: Final = 16

#: The only dtype normalization a host may perform: integer -> float32/float64
#: on a RANK-0 feed. float32 represents every integer below 2**24 exactly, so
#: the recast is value-preserving over any diffusion timestep domain, and it is
#: exactly the op the traced graph's own first node performs on that input — a
#: recast feed produces bit-identical output to the eager call that presented
#: the integer. bfloat16 and float16 are deliberately NOT targets: bf16 has 8
#: mantissa bits and would round timestep 999 to 1000, which is a numeric
#: change, not a normalization.
RECAST_TARGETS: Final = ("float32", "float64")
RECAST_SOURCES: Final = ("int8", "int16", "int32", "int64", "uint8")


class SelectionError(ValueError):
    """A selection input or corpus is incomplete or noncanonical."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = str(reason)
        super().__init__(detail if detail is not None else reason)


class MissReason(StrEnum):
    """Every way one call can miss one graph class's declared ingress.

    Closed on purpose. A host that needs a reason this enum does not carry is
    describing something other than ingress selection, and inventing a member
    locally would give it a rung nobody else agrees on.
    """

    DTYPE_MISMATCH = "dtype_mismatch"
    INPUT_NOT_TENSOR = "input_not_tensor"
    INPUT_EXCLUDED = "input_excluded"
    STATIC_DIM_MISMATCH = "static_dim_mismatch"
    RANGE_VIOLATION = "range_violation"
    SYMBOL_INCONSISTENT = "symbol_inconsistent"
    RANK_MISMATCH = "rank_mismatch"
    INPUT_MISSING = "input_missing"


#: How FAR each reason puts a class from the call. Dims LAST, so a class that
#: matches every declared dimension and disagrees about one scalar fact sorts to
#: the front of a refusal listing whatever its remaining complaint is — that
#: class is the one a reader is looking for, and iteration order buried it.
#:
#: The rungs are ORDINAL ONLY. Nothing may read their absolute values, compute
#: with them, or expose them as a score; only their comparisons are contract.
#: The table is TOTAL over :class:`MissReason` — a closed enum needs no default
#: rung, and a default is how two hosts silently disagree about an unknown one.
MISS_RUNGS: Final[Mapping[MissReason, int]] = {
    # The call fits this graph's shape and disagrees about one scalar fact.
    MissReason.DTYPE_MISMATCH: 1,
    MissReason.INPUT_NOT_TENSOR: 2,
    # A branch/adapter routing disagreement: same shape family, wrong class.
    MissReason.INPUT_EXCLUDED: 3,
    # Shape disagreements — the call does not fit this graph at all.
    MissReason.STATIC_DIM_MISMATCH: 4,
    MissReason.RANGE_VIOLATION: 4,
    MissReason.SYMBOL_INCONSISTENT: 4,
    MissReason.RANK_MISMATCH: 5,
    # The call does not even carry the input; nothing else was measurable.
    MissReason.INPUT_MISSING: 6,
}
if set(MISS_RUNGS) != set(MissReason):  # pragma: no cover - import-time guard
    raise SelectionError("rungs_incomplete", "every miss reason must declare a rung")


class NormalizationKind(StrEnum):
    """The two normalizations a host may perform on an admitted feed."""

    RECAST = "recast"
    REALIGN = "realign"


class RealignReason(StrEnum):
    """Why a feed misses the artifact's aligned-input contract."""

    NON_CONTIGUOUS = "non_contiguous"
    UNALIGNED_16B = "unaligned_16b"


class SelectionOutcome(StrEnum):
    """What selection concluded over one candidate set."""

    ADMITTED = "admitted"
    #: No class admits the call. What the host does about it — serve eager,
    #: report shape growth, refuse — is host policy, not this contract.
    NO_CLASS_ADMITS = "no_class_admits"
    #: Two or more classes admit the call. A DEFECT to surface, never a coin to
    #: flip: the declaration failed to discriminate them by ingress.
    CLASS_AMBIGUOUS = "class_ambiguous"


def _canonical(value: object, what: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SelectionError("selection_invalid", f"{what} must be a canonical string")
    return value


@dataclass(frozen=True, slots=True)
class PresentedValue:
    """The four facts selection reads off one actual feed.

    `shape is None` means "the call presented something with no shape" — a
    Python float, a list, `None` itself. It is the only way to say non-tensor,
    and it ranks (`input_not_tensor`) rather than crashing the walk.

    `alignment_offset` is the feed's address modulo :data:`AOTI_ALIGNMENT`. A
    host that cannot address the feed (meta and fake tensors) reports 0: an
    unmeasurable pointer is not evidence of misalignment, and a normalization
    invented for a tensor with no storage would stage a buffer nothing reads.
    """

    dtype: str = ""
    shape: tuple[int, ...] | None = None
    contiguous: bool = True
    alignment_offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str) or self.dtype != self.dtype.strip():
            raise SelectionError("selection_invalid", "presented dtype must be canonical")
        if self.shape is not None and (
            not isinstance(self.shape, tuple)
            or any(type(dim) is not int or dim < 0 for dim in self.shape)
        ):
            raise SelectionError("selection_invalid", "presented shape is not canonical")
        if type(self.contiguous) is not bool:
            raise SelectionError("selection_invalid", "presented contiguity must be a bool")
        if type(self.alignment_offset) is not int or not (
            0 <= self.alignment_offset < AOTI_ALIGNMENT
        ):
            raise SelectionError(
                "selection_invalid",
                f"alignment_offset must be in [0, {AOTI_ALIGNMENT})",
            )

    @property
    def is_tensor(self) -> bool:
        return self.shape is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": None if self.shape is None else list(self.shape),
            "contiguous": self.contiguous,
            "alignment_offset": self.alignment_offset,
        }

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> PresentedValue:
        if not isinstance(raw, Mapping) or set(raw) != {
            "dtype",
            "shape",
            "contiguous",
            "alignment_offset",
        }:
            raise SelectionError(
                "selection_invalid",
                "presented value fields must be exactly dtype/shape/contiguous/"
                "alignment_offset",
            )
        shape = raw["shape"]
        if shape is not None and not isinstance(shape, list):
            raise SelectionError("selection_invalid", "presented shape must be an array or null")
        return cls(
            dtype=raw["dtype"],
            shape=None if shape is None else tuple(shape),
            contiguous=raw["contiguous"],
            alignment_offset=raw["alignment_offset"],
        )


@dataclass(frozen=True, slots=True)
class PresentedCall:
    """One call, described once, ruled on by every candidate class.

    `feeds` is keyed by the DECLARED INPUT NAME each candidate spells, already
    resolved through that input's recorded param/path coordinate. A declared
    name absent from `feeds` is `input_missing`.

    `excluded_present` names the excluded inputs the call actually carries. A
    class that excludes any of them refuses the call: it would otherwise return
    the base model and look correct.
    """

    feeds: tuple[tuple[str, PresentedValue], ...] = ()
    excluded_present: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.feeds)
        for name in names:
            _canonical(name, "feed name")
        if names != tuple(sorted(set(names))):
            raise SelectionError("selection_invalid", "feeds must be sorted and unique by name")
        for name in self.excluded_present:
            _canonical(name, "excluded input name")
        if self.excluded_present != tuple(sorted(set(self.excluded_present))):
            raise SelectionError(
                "selection_invalid", "excluded_present must be sorted and unique"
            )

    def feed(self, name: str) -> PresentedValue | None:
        for candidate, value in self.feeds:
            if candidate == name:
                return value
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feeds": {name: value.as_dict() for name, value in self.feeds},
            "excluded_present": list(self.excluded_present),
        }

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> PresentedCall:
        if not isinstance(raw, Mapping) or set(raw) != {"feeds", "excluded_present"}:
            raise SelectionError(
                "selection_invalid", "presented call fields must be exactly feeds/excluded_present"
            )
        feeds = raw["feeds"]
        excluded = raw["excluded_present"]
        if not isinstance(feeds, Mapping) or not isinstance(excluded, list):
            raise SelectionError("selection_invalid", "presented call arrays are malformed")
        return cls(
            feeds=tuple(
                (name, PresentedValue.decode(value)) for name, value in sorted(feeds.items())
            ),
            excluded_present=tuple(excluded),
        )


@dataclass(frozen=True, slots=True)
class GraphClassCandidate:
    """One graph class a host offers to selection, by name and ingress."""

    name: str
    ingress: CallIngress

    def __post_init__(self) -> None:
        _canonical(self.name, "graph class name")
        if not isinstance(self.ingress, CallIngress):
            raise SelectionError("selection_invalid", "candidate ingress must be a CallIngress")


@dataclass(frozen=True, slots=True)
class IngressMiss:
    """ONE reason ONE class refuses ONE call.

    `detail` is a human sentence and is deliberately NOT part of equality or of
    the corpus: a second host renders its own refusals in its own language. The
    contract is the reason, the input it names, and the rung.
    """

    reason: MissReason
    input: str = ""
    detail: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reason, MissReason):
            raise SelectionError("selection_invalid", "miss reason must be a MissReason")
        if not isinstance(self.input, str) or self.input != self.input.strip():
            raise SelectionError("selection_invalid", "miss input must be canonical")

    @property
    def rung(self) -> int:
        return MISS_RUNGS[self.reason]

    def as_dict(self) -> dict[str, Any]:
        return {"reason": str(self.reason), "input": self.input}

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> IngressMiss:
        if not isinstance(raw, Mapping) or set(raw) != {"reason", "input"}:
            raise SelectionError(
                "selection_invalid", "miss fields must be exactly reason/input"
            )
        return cls(reason=MissReason(raw["reason"]), input=raw["input"])


def miss_distance(misses: Sequence[IngressMiss]) -> tuple[int, ...]:
    """Sort key over one class's misses — LOWER is closer to the call.

    The sorted rung tuple, so that (a) a class whose only complaint is a shallow
    one beats a class with a deep one, and (b) among classes sharing a
    shallowest complaint, the one with FEWER complaints wins — a prefix sorts
    before its extension. Deterministic and total.
    """

    return tuple(sorted(miss.rung for miss in misses))


@dataclass(frozen=True, slots=True)
class ClassReport:
    """Every way one call misses one class, plus the symbols it did bind."""

    name: str
    misses: tuple[IngressMiss, ...] = ()
    symbols: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _canonical(self.name, "graph class name")
        if not isinstance(self.misses, tuple) or any(
            not isinstance(miss, IngressMiss) for miss in self.misses
        ):
            raise SelectionError("selection_invalid", "class report misses must be IngressMiss")
        names = tuple(name for name, _ in self.symbols)
        if names != tuple(sorted(set(names))):
            raise SelectionError("selection_invalid", "report symbols must be sorted and unique")

    @property
    def admits(self) -> bool:
        return not self.misses

    @property
    def distance(self) -> tuple[int, ...]:
        return miss_distance(self.misses)

    def as_dict(self) -> dict[str, Any]:
        return {
            "class": self.name,
            "distance": list(self.distance),
            "misses": [miss.as_dict() for miss in self.misses],
            "symbols": {name: value for name, value in self.symbols},
        }

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> ClassReport:
        if not isinstance(raw, Mapping) or set(raw) != {
            "class",
            "distance",
            "misses",
            "symbols",
        }:
            raise SelectionError(
                "selection_invalid",
                "class report fields must be exactly class/distance/misses/symbols",
            )
        rows = raw["misses"]
        symbols = raw["symbols"]
        if not isinstance(rows, list) or not isinstance(symbols, Mapping):
            raise SelectionError("selection_invalid", "class report arrays are malformed")
        report = cls(
            name=raw["class"],
            misses=tuple(IngressMiss.decode(row) for row in rows),
            symbols=tuple(sorted((str(k), int(v)) for k, v in symbols.items())),
        )
        if list(report.distance) != raw["distance"]:
            raise SelectionError(
                "selection_invalid",
                f"class {report.name!r} states a distance its misses do not produce",
            )
        return report


@dataclass(frozen=True, slots=True)
class FeedNormalization:
    """One permitted normalization of one admitted feed.

    Producing the plan is this contract's; PERFORMING it is the host's — staging
    buffers, device copies and their lifetime are mechanics no vocabulary can
    state. What the host may not do is deviate from the plan.

    **Invariants a normalization may never break.** A `REALIGN` changes only the
    feed's address and layout: never its dtype, shape, rank, device or values. A
    `RECAST` changes only its dtype, to `dtype` and to nothing else, on a rank-0
    feed whose value survives exactly. Neither reorders feeds, adds one, drops
    one, or is permitted to alter which class was selected — the plan is
    computed AFTER admission, over the selected class alone.
    """

    input: str
    position: int
    kind: NormalizationKind
    reason: str
    dtype: str = ""

    def __post_init__(self) -> None:
        _canonical(self.input, "normalization input")
        if type(self.position) is not int or self.position < 0:
            raise SelectionError("selection_invalid", "normalization position must be >= 0")
        if not isinstance(self.kind, NormalizationKind):
            raise SelectionError("selection_invalid", "normalization kind is not canonical")
        _canonical(self.reason, "normalization reason")
        if self.kind is NormalizationKind.RECAST:
            if self.dtype not in RECAST_TARGETS:
                raise SelectionError(
                    "selection_invalid", f"a recast may only target {list(RECAST_TARGETS)!r}"
                )
        elif self.dtype:
            raise SelectionError("selection_invalid", "a realign may not change dtype")

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "position": self.position,
            "kind": str(self.kind),
            "reason": self.reason,
            "dtype": self.dtype,
        }

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> FeedNormalization:
        if not isinstance(raw, Mapping) or set(raw) != {
            "input",
            "position",
            "kind",
            "reason",
            "dtype",
        }:
            raise SelectionError(
                "selection_invalid",
                "normalization fields must be exactly input/position/kind/reason/dtype",
            )
        return cls(
            input=raw["input"],
            position=raw["position"],
            kind=NormalizationKind(raw["kind"]),
            reason=raw["reason"],
            dtype=raw["dtype"],
        )


@dataclass(frozen=True, slots=True)
class Selection:
    """The whole outcome of selecting among one candidate set."""

    outcome: SelectionOutcome
    selected: str = ""
    symbols: tuple[tuple[str, int], ...] = ()
    normalizations: tuple[FeedNormalization, ...] = ()
    ranked: tuple[ClassReport, ...] = ()
    ambiguous: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, SelectionOutcome):
            raise SelectionError("selection_invalid", "outcome must be a SelectionOutcome")
        admitted = self.outcome is SelectionOutcome.ADMITTED
        if admitted:
            _canonical(self.selected, "selected graph class")
        elif self.selected or self.symbols or self.normalizations:
            raise SelectionError(
                "selection_invalid", "only an admitted selection carries a class, symbols or a plan"
            )
        if self.ranked and self.outcome is not SelectionOutcome.NO_CLASS_ADMITS:
            raise SelectionError(
                "selection_invalid", "only a total miss carries the exhaustive ranking"
            )
        if self.ambiguous and self.outcome is not SelectionOutcome.CLASS_AMBIGUOUS:
            raise SelectionError(
                "selection_invalid", "only an ambiguous selection names competing classes"
            )
        if self.outcome is SelectionOutcome.CLASS_AMBIGUOUS and len(self.ambiguous) < 2:
            raise SelectionError(
                "selection_invalid", "an ambiguous selection names at least two classes"
            )
        names = tuple(name for name, _ in self.symbols)
        if names != tuple(sorted(set(names))):
            raise SelectionError("selection_invalid", "symbols must be sorted and unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": str(self.outcome),
            "selected": self.selected,
            "symbols": {name: value for name, value in self.symbols},
            "normalizations": [row.as_dict() for row in self.normalizations],
            "ranked": [row.as_dict() for row in self.ranked],
            "ambiguous": list(self.ambiguous),
        }

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> Selection:
        if not isinstance(raw, Mapping) or set(raw) != {
            "outcome",
            "selected",
            "symbols",
            "normalizations",
            "ranked",
            "ambiguous",
        }:
            raise SelectionError(
                "selection_invalid",
                "selection fields must be exactly outcome/selected/symbols/"
                "normalizations/ranked/ambiguous",
            )
        symbols = raw["symbols"]
        if (
            not isinstance(symbols, Mapping)
            or not isinstance(raw["normalizations"], list)
            or not isinstance(raw["ranked"], list)
            or not isinstance(raw["ambiguous"], list)
        ):
            raise SelectionError("selection_invalid", "selection arrays are malformed")
        return cls(
            outcome=SelectionOutcome(raw["outcome"]),
            selected=raw["selected"],
            symbols=tuple(sorted((str(k), int(v)) for k, v in symbols.items())),
            normalizations=tuple(
                FeedNormalization.decode(row) for row in raw["normalizations"]
            ),
            ranked=tuple(ClassReport.decode(row) for row in raw["ranked"]),
            ambiguous=tuple(raw["ambiguous"]),
        )


# ---------------------------------------------------------------------------
# The permitted normalizations
# ---------------------------------------------------------------------------


def recast_gap(spec: CallInput, value: PresentedValue) -> str:
    """The named dtype normalization this feed needs, or `''`.

    `''` means either "already in contract" or "must be REFUSED": this never
    widens a contract, it names the ONE normalization a host may perform, and
    :func:`class_report` refuses every other dtype disagreement exactly as it
    would without it.

    The dtype a denoiser is handed for its scalar timestep is a per-request
    SAMPLER fact, not a family fact — euler-family samplers present float32,
    ddim/dpmpp/unipc-family samplers end `set_timesteps` in `.to(int64)`. No
    single declared dtype can be right for a family whose sampler is per-request
    state, and the sampler is deliberately not a compile axis, so the
    normalization belongs at the boundary that knows the contract: once, named,
    countable.

    **This is the only normalization that moves ADMISSION.** A realign never
    does — see :func:`realign_gap`.
    """

    if spec.shape:  # rank-0 only: the timestep class, not a tensor of values
        return ""
    if spec.dtype not in RECAST_TARGETS:
        return ""
    if value.dtype not in RECAST_SOURCES:
        return ""
    if value.shape != ():
        return ""
    return f"{value.dtype}_to_{spec.dtype}"


def realign_gap(value: PresentedValue) -> str:
    """`''` when a feed satisfies the aligned-input contract, else the reason.

    AOTInductor compiles its fast path for 16-byte-aligned, contiguous inputs.
    When the runtime pointer is not aligned its `run_impl` copies the tensor to
    an aligned buffer ON EVERY CALL and reports it on stderr, which hub-spawned
    pods do not expose — a fleet paying the tax was indistinguishable from one
    that was not.

    Alignment is a PERFORMANCE fact, never an admission fact: the compiled graph
    runs correctly either way, so a misaligned feed must not disqualify a class
    that otherwise admits the call.
    """

    if not value.is_tensor:
        return ""
    if not value.contiguous:
        return str(RealignReason.NON_CONTIGUOUS)
    if value.alignment_offset:
        return str(RealignReason.UNALIGNED_16B)
    return ""


def normalization_plan(
    ingress: CallIngress, call: PresentedCall
) -> tuple[FeedNormalization, ...]:
    """The normalizations one admitted class's feeds need, in position order.

    At most ONE per feed: a recast implies a staged copy, so it SUBSUMES the
    realignment rather than running after it. A host that staged twice would
    pay the copy twice for a feed that ends up in the same buffer either way.
    """

    plan: list[FeedNormalization] = []
    for spec in ingress.inputs:
        value = call.feed(spec.name)
        if value is None:
            continue
        recast = recast_gap(spec, value)
        reason = recast or realign_gap(value)
        if not reason:
            continue
        plan.append(
            FeedNormalization(
                input=spec.name,
                position=spec.position,
                kind=NormalizationKind.RECAST if recast else NormalizationKind.REALIGN,
                reason=reason,
                dtype=spec.dtype if recast else "",
            )
        )
    return tuple(plan)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def class_report(
    name: str, ingress: CallIngress, call: PresentedCall, *, first_only: bool = False
) -> ClassReport:
    """Every way `call` misses `ingress`, plus the symbols it did bind.

    A class ADMITS a call exactly when this report carries no miss. Every miss
    disqualifies; none merely ranks. Ranking is what the refusal path does with
    classes that are already out, and the two can never be computed by different
    rules because they are computed by this one walk.

    `first_only` is an early EXIT from that same walk, never a second rule —
    every admission decision takes it (an admitted call has no misses, so the
    two are identical there), and the exhaustive walk is paid only on the
    refusal path, which is already headed somewhere slow. A 36-class family is
    asked this once per denoise step.

    The order is fixed because a truncated report must be the same everywhere:
    excluded inputs first, then a missing input, then per declared input in
    POSITION order, dtype -> rank -> dims, dims left to right.
    """

    present = tuple(row for row in ingress.excluded_inputs if row in call.excluded_present)
    if present:
        return ClassReport(
            name=name,
            misses=(
                IngressMiss(
                    MissReason.INPUT_EXCLUDED,
                    present[0],
                    f"this graph class REFUSES input(s) {list(present)!r}: the call carries "
                    f"them, so it must be served by the class that declares them",
                ),
            ),
        )
    for spec in ingress.inputs:
        if call.feed(spec.name) is not None:
            continue
        # An input the call does not carry at all: nothing further about this
        # class is measurable, so it is one miss and the deepest rung.
        return ClassReport(
            name=name,
            misses=(
                IngressMiss(
                    MissReason.INPUT_MISSING,
                    spec.name,
                    f"declared input {spec.name!r} at argument {spec.param!r} is absent",
                ),
            ),
        )

    misses: list[IngressMiss] = []
    symbols: dict[str, int] = {}
    owner: dict[str, str] = {}
    bounds = ingress.symbol_bounds
    for spec in ingress.inputs:
        if first_only and misses:
            break
        value = call.feed(spec.name)
        if value is None:  # unreachable: every declared name was proven present above
            continue
        if not value.is_tensor:
            misses.append(
                IngressMiss(
                    MissReason.INPUT_NOT_TENSOR,
                    spec.name,
                    f"declared input {spec.name!r} was presented with no shape",
                )
            )
            continue
        if value.dtype != spec.dtype and not recast_gap(spec, value):
            misses.append(
                IngressMiss(
                    MissReason.DTYPE_MISMATCH,
                    spec.name,
                    f"input {spec.name!r} dtype {value.dtype or '<unknown>'} != declared "
                    f"{spec.dtype}",
                )
            )
        actual = value.shape or ()
        if len(actual) != len(spec.shape):
            misses.append(
                IngressMiss(
                    MissReason.RANK_MISMATCH,
                    spec.name,
                    f"input {spec.name!r} rank {len(actual)} != declared {len(spec.shape)} "
                    f"(declared shape {list(spec.shape)!r})",
                )
            )
            continue
        for position, (declared, got) in enumerate(zip(spec.shape, actual, strict=True)):
            if isinstance(declared, int):
                if got != declared:
                    misses.append(
                        IngressMiss(
                            MissReason.STATIC_DIM_MISMATCH,
                            spec.name,
                            f"input {spec.name!r} dim {position} = {got} != statically "
                            f"specialized {declared}",
                        )
                    )
                continue
            low, high = bounds[declared]
            if not low <= got <= high:
                misses.append(
                    IngressMiss(
                        MissReason.RANGE_VIOLATION,
                        spec.name,
                        f"input {spec.name!r} dim {position} (symbol {declared!r}) = {got} "
                        f"outside declared range [{low}, {high}]",
                    )
                )
                continue
            prior = symbols.get(declared)
            if prior is not None and prior != got:
                misses.append(
                    IngressMiss(
                        MissReason.SYMBOL_INCONSISTENT,
                        spec.name,
                        f"symbol {declared!r} = {got} on input {spec.name!r} dim {position} "
                        f"but {prior} on input {owner[declared]!r}",
                    )
                )
                continue
            symbols[declared] = got
            owner.setdefault(declared, spec.name)
    return ClassReport(
        name=name,
        misses=tuple(misses[:1]) if first_only else tuple(misses),
        symbols=tuple(sorted(symbols.items())),
    )


def _candidates(candidates: Sequence[GraphClassCandidate]) -> tuple[GraphClassCandidate, ...]:
    rows = tuple(sorted(candidates, key=lambda row: row.name))
    names = tuple(row.name for row in rows)
    if len(set(names)) != len(names):
        raise SelectionError(
            "candidate_duplicate", "two candidates share one graph class name"
        )
    return rows


def select(candidates: Sequence[GraphClassCandidate], call: PresentedCall) -> Selection:
    """Choose the graph class that serves `call`, or say why none does.

    Candidates are evaluated in ascending NAME order so a host that offers the
    same set in a different order reaches the same outcome and the same ranking.

    On a total miss the exhaustive report is computed — and ONLY then. The call
    is already headed for whatever the host does about it, and the ranking it
    leaves behind is the whole diagnosis anyone will ever get, so it is worth
    the second walk exactly once and never on the hot path.
    """

    rows = _candidates(candidates)
    admitted: list[ClassReport] = []
    missed: list[GraphClassCandidate] = []
    for row in rows:
        report = class_report(row.name, row.ingress, call, first_only=True)
        if report.admits:
            admitted.append(report)
        else:
            missed.append(row)

    if not admitted:
        ranked = sorted(
            (class_report(row.name, row.ingress, call) for row in missed),
            key=lambda report: (report.distance, report.name),
        )
        return Selection(outcome=SelectionOutcome.NO_CLASS_ADMITS, ranked=tuple(ranked))
    if len(admitted) > 1:
        return Selection(
            outcome=SelectionOutcome.CLASS_AMBIGUOUS,
            ambiguous=tuple(report.name for report in admitted),
        )
    chosen = admitted[0]
    ingress = next(row.ingress for row in rows if row.name == chosen.name)
    return Selection(
        outcome=SelectionOutcome.ADMITTED,
        selected=chosen.name,
        symbols=chosen.symbols,
        normalizations=normalization_plan(ingress, call),
    )


# ---------------------------------------------------------------------------
# The adapter from a real call
# ---------------------------------------------------------------------------


def _presented(value: object) -> PresentedValue:
    shape = getattr(value, "shape", None)
    if shape is None:
        return PresentedValue()
    dtype = str(getattr(value, "dtype", "") or "").removeprefix("torch.")
    contiguous = True
    is_contiguous = getattr(value, "is_contiguous", None)
    if callable(is_contiguous):
        contiguous = bool(is_contiguous())
    offset = 0
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        try:
            offset = int(data_ptr()) % AOTI_ALIGNMENT
        except (RuntimeError, ValueError):  # meta and fake tensors have no address
            offset = 0
    return PresentedValue(
        dtype=dtype,
        shape=tuple(int(dim) for dim in shape),
        contiguous=contiguous,
        alignment_offset=offset,
    )


def describe_call(
    candidates: Sequence[GraphClassCandidate],
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> PresentedCall:
    """Describe one real call as the facts every candidate rules on.

    Each declared input is resolved by REPLAYING its recorded param/path
    coordinate — `CallIngress` built and identity-hashed that mapping from the
    exported call, and the same value resolves it here. Nothing searches for a
    declared input: a dict carrying a colliding key must not be able to decide
    a bind.

    Excluded inputs are the one place a search is correct, and deliberately: an
    excluded input has no contract row by definition, so there is nothing to
    replay, and the question asked of it is "does the call carry such a value
    anywhere" — a pipeline can hand an adapter down nested. Keyword and
    nested-mapping only, since an excluded input has no position in this graph's
    signature; `None` does not count as carrying it.

    Two candidates that spell one input NAME from different call coordinates
    cannot be described once, so this refuses rather than resolving one of them
    against the other's coordinate.
    """

    rows = _candidates(candidates)
    coordinate: dict[str, tuple[str, int, tuple[object, ...]]] = {}
    feeds: dict[str, PresentedValue] = {}
    excluded: set[str] = set()
    for row in rows:
        excluded.update(row.ingress.excluded_inputs)
        for spec in row.ingress.inputs:
            here = (spec.param, spec.param_position, tuple(spec.path))
            there = coordinate.setdefault(spec.name, here)
            if there != here:
                raise SelectionError(
                    "input_name_collision",
                    f"graph class {row.name!r} spells input {spec.name!r} at a different call "
                    f"coordinate than a sibling candidate; one call cannot describe both",
                )
            if spec.name in feeds:
                continue
            found, value = spec.resolve(args, kwargs)
            if found:
                feeds[spec.name] = _presented(value)

    carried: list[str] = []
    for name in sorted(excluded):
        value = kwargs.get(name)
        if value is None:
            value = next(
                (
                    nested[name]
                    for nested in kwargs.values()
                    if isinstance(nested, Mapping) and nested.get(name) is not None
                ),
                None,
            )
        if value is not None:
            carried.append(name)
    return PresentedCall(
        feeds=tuple(sorted(feeds.items())), excluded_present=tuple(carried)
    )


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionVector:
    """One corpus row: a candidate set, a described call, and the outcome."""

    name: str
    note: str
    candidates: tuple[GraphClassCandidate, ...]
    call: PresentedCall
    expect: Selection

    def __post_init__(self) -> None:
        _canonical(self.name, "vector name")
        _canonical(self.note, "vector note")

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> SelectionVector:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "note",
            "candidates",
            "call",
            "expect",
        }:
            raise SelectionError(
                "selection_invalid",
                "vector fields must be exactly name/note/candidates/call/expect",
            )
        rows = raw["candidates"]
        if not isinstance(rows, list):
            raise SelectionError("selection_invalid", "vector candidates must be an array")
        candidates: list[GraphClassCandidate] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"class", "ingress"}:
                raise SelectionError(
                    "selection_invalid", "vector candidate fields must be exactly class/ingress"
                )
            candidates.append(
                GraphClassCandidate(name=row["class"], ingress=CallIngress.decode(row["ingress"]))
            )
        return cls(
            name=raw["name"],
            note=raw["note"],
            candidates=tuple(candidates),
            call=PresentedCall.decode(raw["call"]),
            expect=Selection.decode(raw["expect"]),
        )


def decode_selection_corpus(raw: Mapping[str, Any]) -> tuple[SelectionVector, ...]:
    """Decode the `ingress_selection_v1` corpus, refusing any other version.

    Every declared constant is re-checked against this implementation, not
    merely read: a corpus that states a different rung table, alignment or
    recast domain than the reference implementation applies is a contract the
    two consumers would silently disagree about, which is the whole failure this
    file exists to make impossible.
    """

    if not isinstance(raw, Mapping):
        raise SelectionError("selection_invalid", "selection corpus must be an object")
    required = {"contract", "v", "authority", "note", "rungs", "normalizations", "vectors"}
    if set(raw) != required:
        raise SelectionError(
            "selection_invalid", f"selection corpus fields must be exactly {sorted(required)!r}"
        )
    if raw["contract"] != "ingress_selection_v1":
        raise SelectionError(
            "contract_unknown", f"selection corpus names contract {raw['contract']!r}"
        )
    if type(raw["v"]) is not int or raw["v"] != SELECTION_CONTRACT_VERSION:
        raise SelectionError(
            "version_unknown",
            f"selection corpus v={raw['v']!r} is not v{SELECTION_CONTRACT_VERSION}",
        )
    rungs = raw["rungs"]
    if not isinstance(rungs, Mapping) or {
        MissReason(name): value for name, value in rungs.items()
    } != dict(MISS_RUNGS):
        raise SelectionError(
            "rungs_divergent", "selection corpus states a rung table this implementation is not"
        )
    normalizations = raw["normalizations"]
    if not isinstance(normalizations, Mapping) or set(normalizations) != {"recast", "realign"}:
        raise SelectionError(
            "selection_invalid", "corpus normalizations must be exactly recast/realign"
        )
    recast = normalizations["recast"]
    realign = normalizations["realign"]
    if (
        not isinstance(recast, Mapping)
        or tuple(recast.get("targets", ())) != RECAST_TARGETS
        or tuple(recast.get("sources", ())) != RECAST_SOURCES
        or recast.get("rank") != 0
    ):
        raise SelectionError(
            "normalizations_divergent", "corpus states a recast domain this implementation is not"
        )
    if (
        not isinstance(realign, Mapping)
        or realign.get("alignment") != AOTI_ALIGNMENT
        or tuple(realign.get("reasons", ())) != tuple(str(row) for row in RealignReason)
    ):
        raise SelectionError(
            "normalizations_divergent", "corpus states a realign domain this implementation is not"
        )
    vectors = raw["vectors"]
    if not isinstance(vectors, list) or not vectors:
        raise SelectionError("selection_invalid", "selection corpus declares no vectors")
    decoded = tuple(SelectionVector.decode(row) for row in vectors)
    names = tuple(row.name for row in decoded)
    if len(set(names)) != len(names):
        raise SelectionError("selection_invalid", "selection vector names must be unique")
    return decoded


def selection_vectors() -> tuple[SelectionVector, ...]:
    """The packaged `ingress_selection_v1` vectors, decoded and version-checked."""

    from .contracts import read_contract

    return decode_selection_corpus(json.loads(read_contract(SELECTION_CONTRACT_FILE)))


__all__ = [
    "AOTI_ALIGNMENT",
    "MISS_RUNGS",
    "RECAST_SOURCES",
    "RECAST_TARGETS",
    "SELECTION_CONTRACT_FILE",
    "SELECTION_CONTRACT_VERSION",
    "ClassReport",
    "FeedNormalization",
    "GraphClassCandidate",
    "IngressMiss",
    "MissReason",
    "NormalizationKind",
    "PresentedCall",
    "PresentedValue",
    "RealignReason",
    "Selection",
    "SelectionError",
    "SelectionOutcome",
    "SelectionVector",
    "class_report",
    "decode_selection_corpus",
    "describe_call",
    "miss_distance",
    "normalization_plan",
    "realign_gap",
    "recast_gap",
    "select",
    "selection_vectors",
]
