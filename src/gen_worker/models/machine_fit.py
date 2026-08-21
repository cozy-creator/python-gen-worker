from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import msgspec

from .. import hostfacts
from .tensor_layout_contract import (
    KNOWN_REQUIREMENT_TERMS,
    LayoutRequirements,
    RequirementTerms,
    term_meets as term_meets,
)

FACT_OF_TERM: Dict[str, str] = {
    term: term.removeprefix("min_") for term in KNOWN_REQUIREMENT_TERMS
}

LEVEL_MINIMUM = "minimum"
LEVEL_RECOMMENDED = "recommended"


class MachineFacts(msgspec.Struct, frozen=True, kw_only=True):
    """What this machine IS, spelled identically to the requirement terms."""

    sm: int = 0
    vram_gb: float = 0.0
    host_ram_gb: float = 0.0
    cuda: str = ""
    torch: str = ""

    def stated(self, fact: str) -> bool:
        return bool(getattr(self, fact, None))

    def render(self) -> str:
        parts = [f"{f}={getattr(self, f)}"
                 for f in self.__struct_fields__ if self.stated(f)]
        return ", ".join(parts) or "nothing measured"


def measure_machine_facts(
    caps: Any = None,
    *,
    vram_gb: Optional[float] = None,
    host_ram_gb: Optional[float] = None,
) -> MachineFacts:
    """This machine, as facts."""
    sm = int(getattr(caps, "gpu_sm", 0) or 0)
    cuda = str(getattr(caps, "cuda_version", "") or "")
    torch_version = str(getattr(caps, "torch_version", "") or "")
    if vram_gb is None:
        total = hostfacts.total_vram_bytes()
        vram_gb = (total / (1 << 30)) if total else 0.0
    if host_ram_gb is None:
        from .memory import get_total_ram_gb

        host_ram_gb = get_total_ram_gb()
    return MachineFacts(
        sm=sm,
        vram_gb=float(vram_gb or 0.0),
        host_ram_gb=float(host_ram_gb or 0.0),
        cuda=_dotted(cuda),
        torch=_dotted(torch_version),
    )


def _dotted(value: str) -> str:
    head = str(value or "").strip().split("+")[0]
    parts: list[str] = []
    for part in head.split("."):
        if not part.isdigit():
            break
        parts.append(part)
    return ".".join(parts)


@dataclass(frozen=True)
class Shortfall:
    """One declared term this machine does not meet."""

    term: str
    level: str
    declared: Any
    measured: Any
    lane: str = ""

    @property
    def fact(self) -> str:
        return FACT_OF_TERM[self.term]

    def render(self) -> str:
        where = f"{self.lane}: " if self.lane else ""
        return (f"{where}{self.term} declares {self.declared} "
                f"({self.level}); this machine measures "
                f"{self.fact}={self.measured}")


@dataclass(frozen=True)
class LevelVerdict:
    """One LEVEL of one requirement, evaluated."""

    shortfalls: Tuple[Shortfall, ...] = ()
    unevaluated: Tuple[str, ...] = ()

    @property
    def met(self) -> bool:
        return not self.shortfalls


def evaluate_terms(
    terms: RequirementTerms,
    facts: MachineFacts,
    *,
    level: str,
    lane: str = "",
) -> LevelVerdict:
    """Every DECLARED term of one level against the facts, by name lookup."""
    shortfalls: list[Shortfall] = []
    unevaluated: list[str] = []
    for term, declared in terms.declared_terms().items():
        fact = FACT_OF_TERM[term]
        if not facts.stated(fact):
            unevaluated.append(term)
            continue
        measured = getattr(facts, fact)
        if not term_meets(term, measured, declared):
            shortfalls.append(Shortfall(
                term=term, level=level, declared=declared,
                measured=measured, lane=lane))
    return LevelVerdict(
        shortfalls=tuple(shortfalls), unevaluated=tuple(unevaluated))


def under_minimum(
    requirement: Optional[LayoutRequirements],
    facts: MachineFacts,
    *,
    lane: str = "",
) -> LevelVerdict:
    """The MINIMUM level's verdict."""
    if requirement is None:
        return LevelVerdict()
    return evaluate_terms(
        requirement.min_terms(), facts, level=LEVEL_MINIMUM, lane=lane)


def under_recommended(
    requirement: Optional[LayoutRequirements],
    facts: MachineFacts,
    *,
    lane: str = "",
) -> LevelVerdict:
    """The RECOMMENDED level's verdict."""
    if requirement is None:
        return LevelVerdict()
    return evaluate_terms(
        requirement.recommended_terms(), facts, level=LEVEL_RECOMMENDED,
        lane=lane)


@dataclass(frozen=True)
class LaneCandidate:
    """One lane the release binds, with what its author declared about it."""

    lane: str
    requirement: Optional[LayoutRequirements] = None


@dataclass(frozen=True)
class RankedLane:
    lane: str
    minimum: LevelVerdict
    recommended: LevelVerdict
    order: int

    @property
    def key(self) -> Tuple[int, int, int]:
        """MINIMUM shortfalls first, then RECOMMENDED, then the caller's order."""
        return (len(self.minimum.shortfalls),
                len(self.recommended.shortfalls),
                self.order)


@dataclass(frozen=True)
class LaneChoice:
    """The pick, and every rejected lane's verdict beside it."""

    lane: str
    ranked: Tuple[RankedLane, ...] = ()

    @property
    def picked(self) -> Optional[RankedLane]:
        for row in self.ranked:
            if row.lane == self.lane:
                return row
        return None

    @property
    def under_minimum(self) -> Tuple[Shortfall, ...]:
        row = self.picked
        return () if row is None else row.minimum.shortfalls

    @property
    def forced(self) -> bool:
        """True when EVERY candidate is under its own minimum, so the pick is the least-bad rather than a satisfied one."""
        return bool(self.ranked) and all(
            row.minimum.shortfalls for row in self.ranked)


def select_lane(
    candidates: Sequence[LaneCandidate], facts: MachineFacts,
) -> LaneChoice:
    """The best lane these facts afford — and there is ALWAYS one."""
    ranked = tuple(sorted(
        (
            RankedLane(
                lane=c.lane,
                minimum=under_minimum(c.requirement, facts, lane=c.lane),
                recommended=under_recommended(
                    c.requirement, facts, lane=c.lane),
                order=i,
            )
            for i, c in enumerate(candidates)
        ),
        key=lambda r: r.key,
    ))
    return LaneChoice(lane=ranked[0].lane if ranked else "", ranked=ranked)


def lane_candidates(slot: Any) -> Tuple[LaneCandidate, ...]:
    """A model slot's bound lanes, in the slot's canonical handle order."""
    layouts = getattr(slot, "layouts", None) or {}
    requirements = getattr(slot, "layout_requirements", None) or {}
    if not requirements:
        return ()
    seen: list[str] = []
    for handles in layouts.values():
        for handle in handles:
            if handle not in seen:
                seen.append(handle)
    return tuple(
        LaneCandidate(lane=handle, requirement=requirements.get(handle))
        for handle in seen
    )


def summarize(shortfalls: Iterable[Shortfall]) -> str:
    """The shortfalls as one line, each naming term, declared floor and measured fact."""
    return "; ".join(s.render() for s in shortfalls)
