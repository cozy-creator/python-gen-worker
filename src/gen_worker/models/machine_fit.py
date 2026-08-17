"""Q2's evaluator: this machine's measured FACTS against declared requirement
TERMS (pgw#1315 deliverables 3 and 4).

`research/machine-compatibility-design.md` splits compatibility into two
questions. Q1 — *"what is a good GPU for this workload?"* — is the endpoint
AUTHOR's, answered as configuration, and a declared **minimum** gates exactly
one thing hub-side: a config-WRITE. Q2 — *"given a fixed GPU, how well does
this endpoint run on this machine?"* — is this worker's, and **the answer is
always "it runs"**. So nothing in this module refuses anything. It answers two
questions and nothing else:

1. **which lane** — among the lanes the release binds, the one whose declared
   requirements this machine best satisfies (:func:`select_lane`);
2. **how far under** — the terms the machine falls short of, named, so the
   worker can serve and CONFESS (:func:`under_minimum`, reported through
   `memory.report_under_minimum`).

**ONE VOCABULARY, ONE EVALUATOR.** A requirement term and the machine fact it
is compared against carry the same name minus the `min_` prefix, so evaluating
a requirement is a NAME LOOKUP and never a bespoke comparator per term:

    min_sm -> sm    min_vram_gb -> vram_gb    min_host_ram_gb -> host_ram_gb
    min_cuda -> cuda                          min_torch -> torch

The comparison itself is `tensor_layout_contract.term_meets`, the same function
that checks `recommended >= minimum` at declaration. Growing
`KNOWN_REQUIREMENT_TERMS` grows this module for free; nothing here names a
term literal.

**AN UNMEASURED FACT IS UNEVALUATED, NEVER A SHORTFALL.** A `kernels` floor
with no runtime probe behind it is a requirement that silently does not hold,
which is why the SDK refuses that term outright. The same rule applies to a
term whose fact this machine could not measure (no card, so no `sm`): it is
reported as unevaluated, not scored as a pass and not scored as a failure.
Reading "0" as "meets nothing" would turn a cardless pod's every lane into a
shortfall report about the wrong thing — the pod already knows it has no card,
and the CPU rung already says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import msgspec

from .. import hostfacts
from .tensor_layout_contract import (
    KNOWN_REQUIREMENT_TERMS,
    LayoutRequirements,
    RequirementTerms,
    # Re-exported deliberately: this module is where a reader looks for the
    # runtime comparison, and there must be exactly one to find.
    term_meets as term_meets,
)

#: term -> the machine fact it is compared against. Derived, not written: the
#: whole point of the shared vocabulary is that this table cannot drift.
FACT_OF_TERM: Dict[str, str] = {
    term: term.removeprefix("min_") for term in KNOWN_REQUIREMENT_TERMS
}

LEVEL_MINIMUM = "minimum"
LEVEL_RECOMMENDED = "recommended"


class MachineFacts(msgspec.Struct, frozen=True, kw_only=True):
    """What this machine IS, spelled identically to the requirement terms.

    Every field's zero/empty means NOT MEASURED — the same tri-state the
    declaration side uses for NOT DECLARED. ``vram_gb`` is the card's TOTAL,
    never its free bytes: free VRAM is a MOMENT, and the rung walk is what
    reads moments (`memory.select_auto_mode`). A requirement compared against
    a moment would pass or fail depending on who else was loading.
    """

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
    """This machine, as facts. ``caps`` is the already-measured
    :class:`~gen_worker.models.hub_policy.TensorhubWorkerCapabilities` the
    executor holds; anything it does not carry is measured here.

    There is no second probe: SM/CUDA/torch come from the capabilities the
    worker already reported to the hub, VRAM from `hostfacts`, and host RAM
    from `memory.probe_host_ram`, which pgw#973 made the ONE owner of that
    number (imported lazily because `memory` imports this module for the
    confession).
    """
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
    """A version the way the vocabulary spells it: leading numeric components
    only. torch reports `2.9.0+cu128`; the declared floor is `2.9`."""
    head = str(value or "").strip().split("+")[0]
    parts: list[str] = []
    for part in head.split("."):
        if not part.isdigit():
            break
        parts.append(part)
    return ".".join(parts)


@dataclass(frozen=True)
class Shortfall:
    """One declared term this machine does not meet.

    ``term`` is the vocabulary's name for the axis, ``declared`` the floor the
    author wrote, ``measured`` what this machine actually has. That triple IS
    the warning §1.36 asks for — *what was applied, why quantified* — and it
    carries NO suggested card: th#1867 deleted `FnDegraded.recommended_vram_gb`
    because the worker's suggestion was the author's own guess handed back.
    Only the hub knows the catalog (th#2075 clause 3).
    """

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
    """One LEVEL of one requirement, evaluated. Never a pass/fail bool: the
    unevaluated axes are the third state and dropping them is how a floor
    silently stops holding."""

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
    """The MINIMUM level's verdict. A shortfall here is a WARNING and never a
    refusal — a machine below a declared minimum is the normal input to a
    degraded run, which is what the minimum being advisory-at-execution
    means."""
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
    """The RECOMMENDED level's verdict. It gates NOTHING, ever — it only ranks
    lanes below, and even there it can never turn into a floor because
    :func:`select_lane` always returns a lane."""
    if requirement is None:
        return LevelVerdict()
    return evaluate_terms(
        requirement.recommended_terms(), facts, level=LEVEL_RECOMMENDED,
        lane=lane)


# ---------------------------------------------------------------------------
# Deliverable 4 — lane selection by machine facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaneCandidate:
    """One lane the release binds, with what its author declared about it.

    ``requirement`` is None for a lane with no declaration: NO DEFAULTS, so an
    undeclared lane is not evaluated rather than scored as satisfying nothing.
    """

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
        """MINIMUM shortfalls first, then RECOMMENDED, then the caller's order.

        One lexicographic key expresses the design's *"recommended first, then
        minimum"* because the declaration side already refuses a `recommended`
        below its own `minimum`: a lane whose recommended holds has its minimum
        holding too, so "fewest minimum shortfalls, then fewest recommended"
        cannot rank an under-minimum lane above a satisfied one.

        NAMED, NOT HIDDEN: an UNDECLARED level contributes no shortfall, so a
        lane that declares nothing can outrank one that declares a
        recommendation this machine misses. That is NO DEFAULTS applied
        consistently — an undeclared axis is not evaluated, and scoring
        silence as a failure would be inventing the author's declaration — and
        it is harmless because ranking is not gating: every lane here runs.
        An author who wants a lane preferred moves a rung on the ladder, which
        is preference's one authority.
        """
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
        """True when EVERY candidate is under its own minimum, so the pick is
        the least-bad rather than a satisfied one. Still a pick: the worker
        does not get to answer "none"."""
        return bool(self.ranked) and all(
            row.minimum.shortfalls for row in self.ranked)


def select_lane(
    candidates: Sequence[LaneCandidate], facts: MachineFacts,
) -> LaneChoice:
    """The best lane these facts afford — and there is ALWAYS one.

    Candidates arrive in the order their holder ranks them, and the sort is
    stable, so ties keep that order. A caller holding the author's (GPU, lane)
    ladder passes ladder order and gets the author's tie-break; a caller
    holding only a slot's accepted SET passes canonical order, which is
    determinism and explicitly NOT a preference (§1.33 point 2: the set is a
    compatibility filter whose order carries no preference, and preference has
    exactly one authority — the ladder).

    Under-minimum on every candidate picks the LEAST-BAD one and the caller
    warns. Never a refusal: refusing here would re-answer question 1, which
    this design gives to the author, and would break always-runs.
    """
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
    """A model slot's bound lanes, in the slot's canonical handle order.

    The lanes a release binds for a slot ARE its accepted contract handles
    (`Slot(layouts=...)`), and their declared requirements are
    `Slot(layout_requirements=...)`, keyed by the handle they guard. A slot
    that declares no requirement at all yields no candidates: there is nothing
    to rank by, and inventing a ranking over undeclared lanes is exactly the
    guess this program deletes.
    """
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
    """The shortfalls as one line, each naming term, declared floor and
    measured fact. Empty for no shortfall — a warning with nothing in it is
    worse than no warning."""
    return "; ".join(s.render() for s in shortfalls)
