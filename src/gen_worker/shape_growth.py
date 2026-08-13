"""Serve-window shape growth — ONE module, reached by both execution arms.

§1.12's contract has two halves: *serve eager* and *compile in the background
and adopt*.  The second half only ever existed on the dynamo arm:
``hot_swap.enable`` returns False unless the pipeline carries a dynamo router
(``compile_cache._MARKER_ATTR`` -> ``failure_signal.router``), and an AOT-armed
pipeline never has one — ``models.provision.enable_compiled`` returns as soon
as ``arm_aot`` succeeds and stamps ``aot_serve._MARKER_ATTR`` instead.  There
is no republish backend: a grown JIT cache is this pod's, and its cell has no
consumer.

What that costs: a declared class outside the armed cell's envelope stays
eager for the life of the pod, every pod, forever.  The ratified reuse strategy
is *AOT cells only, JIT demotes to intake mode*, so "it heals on the JIT path"
is not an answer.

What this module owns
---------------------
* the **vocabulary**: a shape gap is a named DECLARED CLASS outside the armed
  cell's ENVELOPE — the declared serving region, not a dynamo input signature —
  so it is arm-agnostic by construction;
* the **countable fact**: :data:`activity.KIND_SHAPE_GAP`, the AOT counterpart
  of dynamo's ``guard_miss``, so the hub can count coverage holes on either
  arm with one grouped query.  It is reported per request at the ingress that
  refuses the class BY NAME — measured on the traffic that actually asks for
  it, not on whatever a warm run happened to dispatch;
* the **admission primitives** lifted out of ``hot_swap``:
  :class:`TurnGateBusy` / :class:`TurnGateClosed` (the background GPU turn).
  They live here and ``hot_swap`` imports them, so there is exactly one
  implementation.

What this module deliberately does NOT own
------------------------------------------
A task queue, a GPU-turn scheduler, or a device executor.  Growth submits
through the sole owners (the Reconciler and the DeviceExecutor); a second
scheduler here to close the AOT arm sooner is forbidden.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Tuple

from . import activity as activity_mod

logger = logging.getLogger(__name__)

#: The execution arms a cell can be served on.
#: The arm selects the COMPILER, never whether growth happens at all.
ARM_AOT = "aot"
ARM_DYNAMO = "dynamo"

#: Ingress reasons that mean "this class is not covered". ``no_entry_admits``
#: is a genuine coverage hole; ``entry_ambiguous`` is a DECLARATION defect
#: that growth must never try to compile its way out of — it is
#: recorded and reported, never submitted.
REASON_UNCOVERED = "no_entry_admits"
REASON_AMBIGUOUS = "entry_ambiguous"


class TurnGateClosed(Exception):
    """The executor's background-turn gate is gone (shutdown/drain); the
    growth job is dropped and its class stays eager."""


class TurnGateBusy(Exception):
    """No background turn within the bounded admission window (live tenant
    demand). The job re-queues instead of blocking the growth thread — a
    blocked compile for one instance must never head-of-line delay every
    other instance's jobs."""


@dataclass(frozen=True)
class ShapeGap:
    """One request that arrived at a graph class the armed cell does not
    serve.

    ``declared_class`` is the point of the type: dynamo healing keys on an
    input SIGNATURE, which is a concept the AOT arm does not have.  A
    declared class is what both arms have and what a mint can be asked to add,
    so it is the only key a growth path can be arm-agnostic on.
    """

    arm: str
    family: str
    target: str
    #: The missing class, NAMED — never just a shape.
    declared_class: str
    reason: str
    detail: str = ""
    cell_key: str = ""

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.family, self.target, self.declared_class)

    @property
    def growable(self) -> bool:
        """An ambiguous dispatch is a declaration defect, not a coverage hole:
        compiling another entry cannot fix it, and the mint already refuses
        it."""
        return self.reason != REASON_AMBIGUOUS



@dataclass
class GrowthLedger:
    """Every class this process has seen refused, and how often.

    Deduplicated by ``(family, target, declared_class)`` so one uncovered
    class submits ONE growth job however many requests hit it, and so
    ``compiled_shape_coverage`` can report convergence rather than only the
    initial gap.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counts: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    _gaps: Dict[Tuple[str, str, str], ShapeGap] = field(default_factory=dict)

    def record(self, gap: ShapeGap) -> bool:
        """Record one sighting; True when this class had not been seen."""
        with self._lock:
            first = gap.key not in self._counts
            self._counts[gap.key] = self._counts.get(gap.key, 0) + 1
            self._gaps.setdefault(gap.key, gap)
            return first

    def seen(self) -> Tuple[ShapeGap, ...]:
        with self._lock:
            return tuple(self._gaps[key] for key in sorted(self._gaps))

    def counts(self) -> Dict[Tuple[str, str, str], int]:
        with self._lock:
            return dict(self._counts)

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
            self._gaps.clear()


LEDGER = GrowthLedger()


def report(gap: ShapeGap) -> bool:
    """Record the gap and emit the countable fact. True on first sighting.

    Never raises: an armed cell serving one request eager must not be turned
    into a failed request by its own telemetry.
    """
    try:
        first = LEDGER.record(gap)
    except Exception:  # noqa: BLE001 — accounting never un-serves a request
        logger.debug("shape-growth: ledger record failed", exc_info=True)
        return False
    if not first:
        return False
    try:
        activity_mod.emit_event(
            activity_mod.KIND_SHAPE_GAP,
            f"arm={gap.arm} family={gap.family} target={gap.target} "
            f"cell={gap.cell_key or '<none>'} class={gap.declared_class}: "
            f"request out of declared envelope: the armed cell does not cover "
            f"this graph class, so the request is served EAGER and named at "
            f"ingress"
            + (f" — {gap.detail}" if gap.detail else ""),
            phase=gap.reason,
        )
    except Exception:  # noqa: BLE001
        logger.debug("shape-growth: gap event failed", exc_info=True)
    return True



def report_and_submit(gap: ShapeGap) -> bool:
    """The one call a serving path makes: count the gap once.

    There is no growth-backend seam: a gap is recorded and the class stays
    eager.
    """
    return report(gap)



__all__ = [
    "ARM_AOT",
    "ARM_DYNAMO",
    "GrowthLedger",
    "LEDGER",
    "REASON_AMBIGUOUS",
    "REASON_UNCOVERED",
    "ShapeGap",
    "TurnGateBusy",
    "TurnGateClosed",
    "report",
    "report_and_submit",
]
