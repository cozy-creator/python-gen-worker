"""pgw#1328 x tcg#37: ingress selection is READ FROM THE CONTRACT, not re-derived.

Ranking several armed graph classes against one live call — which admits, which
is closest, which normalizations a feed needs — used to live only in
``aot_serve``'s ``EntryDispatch.select`` / ``ingress_report`` /
``miss_distance`` / ``recast_gap`` / ``alignment_gap``. tcg#37 published exactly
that as ``ingress_selection_v1``: a schema, a rung table, two normalization
domains and 20 vectors, with ``torchcg.selection`` as the torch-free reference
implementation. This module is gen-worker ADOPTING it, which tcg#37 recorded as
its own follow-on and pgw#1328 owns.

WHAT MOVES AND WHAT STAYS
-------------------------
Moved (now the contract's): admission is all-or-nothing; ``>1`` admitting class
is a DEFECT and never a coin flip; the miss rung table and its ordering; which
normalizations exist (``recast`` alone may move admission, ``realign`` may
not); the plan's order.

Stayed (worker policy, and tcg#37's scope guard says so outright): the
``EntryDispatch`` registry and its sticky de-arm, the FEED BUFFERS that perform
a normalization, event emission, shape-growth submission, eager fallback, and
the WORDING of a refusal — *"a second host renders its own refusals in its own
language"*.

WHY THIS IS A SEPARATE MODULE FROM ``aot_serve``
------------------------------------------------
Because the adopt-only role needs the selection VERDICT as a value it can put
in a typed refusal (:mod:`gen_worker.serve.refusal`), and ``aot_serve``'s
answer is an exception whose only failure mode has always been "serve this
request eager". A pod with no eager tier cannot read that answer. So the
verdict is computed once, here, as data; ``aot_serve`` renders it as the
exception its eager fallback still expects, and the adopt-only path renders it
as a refusal. One walk, two languages — never two walks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generic, Optional, Sequence, Tuple, TypeVar

from .._vendor.torchcg.ingress import CallIngress
from .._vendor.torchcg.selection import (
    FeedNormalization, GraphClassCandidate, Selection, SelectionError,
    SelectionOutcome, describe_call, select)

logger = logging.getLogger(__name__)

R = TypeVar("R")


@dataclass(frozen=True)
class Candidate(Generic[R]):
    """One ARMED graph class, as this pod's dispatch holds it.

    ``runner`` is the host's own object; the contract never sees it. Keeping it
    on the candidate is what lets one walk answer both "which class" and "call
    what" without a second lookup by name.
    """

    name: str
    ingress: CallIngress
    runner: R


@dataclass(frozen=True)
class EntryChoice(Generic[R]):
    """The contract's verdict, plus the host object it resolved to.

    ``selection`` is tcg#37's value verbatim — outcome, symbols, the
    normalization plan, the ranked reports on a total miss, the ambiguous names
    — and is what a refusal quotes. ``runner`` is set exactly when the outcome
    is ``ADMITTED``.
    """

    selection: Selection
    runner: Optional[R] = None

    @property
    def outcome(self) -> SelectionOutcome:
        return self.selection.outcome

    @property
    def admitted(self) -> bool:
        return self.selection.outcome is SelectionOutcome.ADMITTED

    @property
    def name(self) -> str:
        return self.selection.selected

    @property
    def normalizations(self) -> Tuple[FeedNormalization, ...]:
        return tuple(self.selection.normalizations)


class CallUndescribable(ValueError):
    """The candidate set cannot describe this call at all (tcg#37).

    The only member today is ``input_name_collision``: two candidate classes
    spell one input name from different coordinates, so no single presented
    call can stand for both. That is a DECLARATION defect — the same class as
    ``class_ambiguous`` and reported like one — not a property of the call.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def choose(
    candidates: Sequence[Candidate[R]],
    args: Sequence[object],
    kwargs: dict[str, object],
) -> EntryChoice[R]:
    """Rank ``candidates`` against one live call through the tcg#37 contract.

    Never raises for a miss — a miss is an OUTCOME, which is the whole reason
    this can serve both an eager-capable host (which renders it as an exception
    and falls back) and an adopt-only one (which renders it as a refusal).
    """
    rows = tuple(
        GraphClassCandidate(name=c.name, ingress=c.ingress) for c in candidates)
    try:
        call = describe_call(rows, args, kwargs)
    except SelectionError as exc:
        raise CallUndescribable(exc.reason, str(exc)) from exc
    selection = select(rows, call)
    if selection.outcome is not SelectionOutcome.ADMITTED:
        return EntryChoice(selection=selection)
    by_name = {c.name: c.runner for c in candidates}
    return EntryChoice(selection=selection, runner=by_name[selection.selected])


__all__ = [
    "CallUndescribable",
    "Candidate",
    "EntryChoice",
    "choose",
]
