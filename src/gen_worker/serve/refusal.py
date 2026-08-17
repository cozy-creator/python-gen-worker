"""pgw#1328: what an adopt-only pod says INSTEAD of serving eager and minting.

An eager-capable pod answers every adopt miss the same way — serve eager, mint
in the background (§4.28) — so the miss never had to be stated precisely. An
adopt-only pod has no such answer, and the thing it produces instead is this:
one typed value carrying **which key, which selection outcome, which candidate
classes missed and why**.

TWO DISPOSITIONS, AND THE DIFFERENCE IS WHOSE PROBLEM IT IS
-----------------------------------------------------------
:attr:`Disposition.ROUTE` — *this pod cannot serve this, another can.* The
artifact for this pod's key exists nowhere it can reach, or this card has no
room for it. Placement is the fix, so the wire answer is RETRYABLE and the hub
places the work elsewhere. §4.29's pull-by-key already makes this a fleet-level
fact: the hub knows which keysets resolve.

:attr:`Disposition.REFUSE` — *nobody can serve this as asked.* A malformed
keyset document, a call outside every declared class's ingress, an ambiguous
declaration. Re-placing it would buy a second identical failure on somebody
else's card, so the wire answer is terminal.

Getting this backwards is expensive in both directions and the enum exists so
the choice is made once per miss kind, at the site that knows, rather than
inferred from a string at the wire.

THE EVIDENCE IS tcg#37's VOCABULARY
------------------------------------
``torchcg.selection`` is the versioned contract for ranking graph classes
against a call. Its :class:`~torchcg.selection.ClassReport` — class name, the
misses, the ordinal rung tuple — is what a refusal carries, converted here into
frozen worker-side rows so the refusal is a value that can cross a wire and be
counted. torchcg deliberately excludes refusal WORDING from its contract (*"a
second host renders its own refusals in its own language"*), which is exactly
the seam this module sits on: the reasons and the rungs come from the contract,
the sentence is ours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Sequence, Tuple

from .._vendor.torchcg.selection import (
    ClassReport, MissReason, SelectionOutcome)
from ..aot_constants import GraphClassName

logger = logging.getLogger(__name__)

#: How many candidate classes a refusal names before it truncates. The whole
#: refusal rides one activity-event detail column, and a family with a dozen
#: buckets would otherwise spend the budget on the ninth-closest class instead
#: of on the key and the closest one.
CANDIDATE_SAMPLE = 3

#: How many misses are named per candidate. tcg#37 ranks by the SORTED rung
#: tuple, so the first entries are the shallowest complaints — the ones that
#: say what this class actually wanted.
MISS_SAMPLE = 3


class Disposition(StrEnum):
    """Who can fix this — the fleet, or nobody."""

    #: Another pod can serve it. RETRYABLE on the wire.
    ROUTE = "route"
    #: Nobody can serve it as asked. Terminal on the wire.
    REFUSE = "refuse"


class MissKind(StrEnum):
    """WHY the adopt-only role produced no compiled service. Closed.

    The values are the ``phase`` column of :data:`~gen_worker.activity.
    KIND_ADOPT_REFUSED`, so the fleet groups on them; renaming one orphans that
    history. Every member states its disposition in :data:`DISPOSITIONS`, which
    is total over this enum and checked at import — a new member without a
    disposition is a refusal nobody decided the routing of.
    """

    #: No shipped ``cg-keyset-v1`` document holds this pod's closure and no
    #: mint-lane deriver was injected (pgw#1327's ``keyset_absent``). The pod
    #: cannot even STATE a key, so there is nothing to look up.
    NO_KEYSET = "no_keyset"

    #: A keyset document was found and is malformed or a version this worker
    #: does not read (pgw#1327's ``keyset_invalid``). A mint-lane defect.
    KEYSET_INVALID = "keyset_invalid"

    #: This RELEASE declares nothing to adopt — no export declaration, or one
    #: that will not enumerate. An image fact, identical on every pod that runs
    #: it, which is why it does not route.
    NOT_ADOPTABLE = "not_adoptable"

    #: This POD may not arm at all: a topology that forbids it (pgw#775), an
    #: operator's eager-only order (§4.32 item 4), or no readable compute
    #: capability. Another pod without that posture can serve the same work.
    ARM_FORBIDDEN = "arm_forbidden"

    #: §4.29's pull-by-key was refused fail-closed by the resolve path — an
    #: ambiguous same-key answer, a short or out-of-order batch, a receipt this
    #: pod will not share. The hub's own typed codes ride through verbatim in
    #: ``detail``; the disposition is terminal because a second pod asking the
    #: same question gets the same answer.
    RESOLVE_REFUSED = "resolve_refused"

    #: The key is stated and NOBODY holds the artifact — not this machine's own
    #: store, not the hub (§4.29: one artifact or MISS, never a listing).
    ARTIFACT_MISS = "artifact_miss"

    #: The artifact resolved and the arm refused it — a receipt gate, a
    #: publisher check, an identity this pod cannot establish (pgw#1122).
    ARM_REFUSED = "arm_refused"

    #: §4.33/pgw#1175: this card has no room to hold the constant table. The
    #: same token the module and store arms already raise, so the fleet's
    #: existing headroom accounting sees an adopt-only pod exactly as it sees
    #: every other one.
    INSUFFICIENT_ADOPT_VRAM = "insufficient_adopt_vram"

    #: tcg#37 ``no_class_admits``: the call is outside every armed class's
    #: declared ingress. Carries the ranking.
    NO_CLASS_ADMITS = "no_class_admits"

    #: tcg#37 ``class_ambiguous``: more than one class admits. A DECLARATION
    #: defect to surface, never a coin to flip.
    CLASS_AMBIGUOUS = "class_ambiguous"

    #: The class this call needs is declared but not armed. On an eager-capable
    #: pod this is the honest "the background compile has not reached it yet";
    #: on an adopt-only pod no compile is coming, so it is a placement fact.
    CLASS_UNARMED = "class_unarmed"

    #: Something asked this role to mint. It cannot, by construction — the
    #: refusal exists so the attempt is a named event rather than an
    #: ``ImportError`` from :mod:`gen_worker.serve.guard` in somebody's
    #: ``except Exception``.
    MINT_FORBIDDEN = "mint_forbidden"


#: Total over :class:`MissKind`, no default. A default member is how a new
#: refusal silently inherits somebody else's routing decision.
DISPOSITIONS: dict[MissKind, Disposition] = {
    MissKind.NO_KEYSET: Disposition.ROUTE,
    MissKind.KEYSET_INVALID: Disposition.REFUSE,
    MissKind.NOT_ADOPTABLE: Disposition.REFUSE,
    MissKind.ARM_FORBIDDEN: Disposition.ROUTE,
    MissKind.RESOLVE_REFUSED: Disposition.REFUSE,
    MissKind.ARTIFACT_MISS: Disposition.ROUTE,
    MissKind.ARM_REFUSED: Disposition.ROUTE,
    MissKind.INSUFFICIENT_ADOPT_VRAM: Disposition.ROUTE,
    MissKind.NO_CLASS_ADMITS: Disposition.REFUSE,
    MissKind.CLASS_AMBIGUOUS: Disposition.REFUSE,
    MissKind.CLASS_UNARMED: Disposition.ROUTE,
    MissKind.MINT_FORBIDDEN: Disposition.REFUSE,
}

_missing = sorted(k.value for k in MissKind if k not in DISPOSITIONS)
if _missing:  # pragma: no cover — an import-time contradiction
    raise RuntimeError(
        f"MissKind members with no disposition: {_missing}. Every refusal must "
        f"declare whether the fleet can route around it.")
del _missing


@dataclass(frozen=True, slots=True)
class MissNote:
    """ONE contract-level complaint about one input, as tcg#37 states it."""

    reason: MissReason
    input: str = ""
    rung: int = 0

    @classmethod
    def of(cls, miss: object) -> "MissNote":
        reason = getattr(miss, "reason")
        return cls(
            reason=MissReason(reason),
            input=str(getattr(miss, "input", "") or ""),
            rung=int(getattr(miss, "rung", 0)),
        )

    def render(self) -> str:
        return f"{self.input or '?'}:{self.reason.value}" if self.input else (
            self.reason.value)


@dataclass(frozen=True, slots=True)
class CandidateMiss:
    """One graph class that did NOT admit the call, and how far off it was.

    ``distance`` is tcg#37's :meth:`~torchcg.selection.ClassReport.distance` —
    the SORTED rung tuple. It is ORDINAL ONLY: the contract says so, and a
    reader that treats it as a score is reading a rank as a magnitude.
    """

    graph_class: GraphClassName
    distance: Tuple[int, ...] = ()
    misses: Tuple[MissNote, ...] = ()

    @classmethod
    def of(cls, report: ClassReport) -> "CandidateMiss":
        return cls(
            graph_class=GraphClassName(report.name),
            distance=tuple(report.distance),
            misses=tuple(MissNote.of(m) for m in report.misses),
        )

    def render(self) -> str:
        notes = ", ".join(n.render() for n in self.misses[:MISS_SAMPLE])
        more = len(self.misses) - MISS_SAMPLE
        if more > 0:
            notes = f"{notes} (+{more})"
        return f"{self.graph_class}[{'.'.join(str(r) for r in self.distance)}] {notes}"


@dataclass(frozen=True, slots=True)
class AdoptOnlyRefusal:
    """The complete answer an adopt-only pod gives instead of eager + mint.

    Complete in the pgw#1116 sense: a reader never has to join it back to
    anything. The key it asked for, the class it was about, the selection
    outcome the contract returned, and the ranked candidates are all here.
    """

    kind: MissKind
    function: str = ""
    family: str = ""
    compiled_graph_key: str = ""
    graph_class: Optional[GraphClassName] = None
    selection: Optional[SelectionOutcome] = None
    candidates: Tuple[CandidateMiss, ...] = ()
    unarmed: Tuple[GraphClassName, ...] = ()
    detail: str = ""
    #: Set only by :func:`report`, so a refusal that reached the wire is
    #: distinguishable from one that was built and dropped.
    reported: bool = field(default=False, compare=False)

    @property
    def disposition(self) -> Disposition:
        return DISPOSITIONS[self.kind]

    @property
    def routable(self) -> bool:
        return self.disposition is Disposition.ROUTE

    @property
    def phase(self) -> str:
        """The countable token. Never a summary — pgw#1116's whole lesson."""
        return self.kind.value

    def wire_detail(self) -> str:
        """One line naming every identifier this decision was about."""
        parts = [
            f"function={self.function or '?'}",
            f"family={self.family or '?'}",
            f"key={self.compiled_graph_key or '-'}",
            f"disposition={self.disposition.value}",
        ]
        if self.graph_class:
            parts.append(f"graph_class={self.graph_class}")
        if self.selection is not None:
            parts.append(f"selection={self.selection.value}")
        if self.unarmed:
            parts.append(f"unarmed={','.join(self.unarmed[:CANDIDATE_SAMPLE])}")
        if self.candidates:
            shown = "; ".join(
                c.render() for c in self.candidates[:CANDIDATE_SAMPLE])
            more = len(self.candidates) - CANDIDATE_SAMPLE
            parts.append(
                f"closest={shown}" + (f" (+{more} further)" if more > 0 else ""))
        line = " ".join(parts)
        return f"{line} — {self.detail}" if self.detail else line

    def error(self) -> "AdoptOnlyRefused":
        return AdoptOnlyRefused(self)


class AdoptOnlyRefused(RuntimeError):
    """Raised where an eager-capable pod would have degraded to eager."""

    def __init__(self, refusal: AdoptOnlyRefusal) -> None:
        super().__init__(f"{refusal.phase}: {refusal.wire_detail()}")
        self.refusal = refusal

    @property
    def reason(self) -> str:
        """The countable token, spelled the way every other worker error is."""
        return self.refusal.phase


def from_selection(
    outcome: SelectionOutcome,
    ranked: Sequence[ClassReport],
    ambiguous: Sequence[str],
    *,
    function: str = "",
    family: str = "",
    compiled_graph_key: str = "",
    unarmed: Sequence[str] = (),
    detail: str = "",
) -> AdoptOnlyRefusal:
    """Build the refusal for a tcg#37 selection that did not admit.

    ``unarmed`` names declared classes this pod has NOT armed. It changes the
    KIND rather than decorating the detail: "no class admits" and "the class
    that would admit is not armed here" are a terminal defect and a placement
    fact respectively, and reading them as one is how an adopt-only pod would
    fail a request the fleet could have served.
    """
    if outcome is SelectionOutcome.ADMITTED:
        raise ValueError("an admitted selection is not a refusal")
    if outcome is SelectionOutcome.CLASS_AMBIGUOUS:
        return AdoptOnlyRefusal(
            kind=MissKind.CLASS_AMBIGUOUS, function=function, family=family,
            compiled_graph_key=compiled_graph_key, selection=outcome,
            unarmed=tuple(GraphClassName(n) for n in ambiguous),
            detail=detail or (
                f"{len(ambiguous)} classes admit this call — the declaration "
                f"does not discriminate them by ingress contract"))
    kind = MissKind.CLASS_UNARMED if unarmed else MissKind.NO_CLASS_ADMITS
    return AdoptOnlyRefusal(
        kind=kind, function=function, family=family,
        compiled_graph_key=compiled_graph_key, selection=outcome,
        candidates=tuple(CandidateMiss.of(r) for r in ranked),
        unarmed=tuple(GraphClassName(n) for n in unarmed),
        detail=detail)


def report(refusal: AdoptOnlyRefusal) -> AdoptOnlyRefusal:
    """Emit this decision as ONE typed event, and return it with ``reported``.

    Same shape as :func:`gen_worker.boot_adopt.report` and for the same reason:
    a hub-spawned pod exposes no stdout (pgw#760), so a refusal that is only
    logged is a refusal nobody can read — and an adopt-only pod's refusals are
    the ONLY thing it produces on a miss.
    """
    from .. import activity

    activity.emit_event(
        activity.KIND_ADOPT_REFUSED, refusal.wire_detail(),
        phase=refusal.phase)
    logger.warning("adopt-only[%s]: %s", refusal.phase, refusal.wire_detail())
    return AdoptOnlyRefusal(
        kind=refusal.kind, function=refusal.function, family=refusal.family,
        compiled_graph_key=refusal.compiled_graph_key,
        graph_class=refusal.graph_class, selection=refusal.selection,
        candidates=refusal.candidates, unarmed=refusal.unarmed,
        detail=refusal.detail, reported=True)


__all__ = [
    "AdoptOnlyRefusal",
    "AdoptOnlyRefused",
    "CANDIDATE_SAMPLE",
    "CandidateMiss",
    "DISPOSITIONS",
    "Disposition",
    "MISS_SAMPLE",
    "MissKind",
    "MissNote",
    "from_selection",
    "report",
]
