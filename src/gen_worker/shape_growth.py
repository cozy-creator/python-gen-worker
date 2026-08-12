"""Serve-window shape growth — ONE module, reached by both execution arms
(pgw#916).

§1.12's contract has two halves: *serve eager* and *compile in the background
and adopt*.  The second half only ever existed on the dynamo arm.
``hot_swap.enable`` returns False unless the pipeline carries a dynamo router
(``compile_cache._MARKER_ATTR`` -> ``failure_signal.router``), and an AOT-armed
pipeline never has one — ``models.provision.enable_compiled`` returns as soon
as ``arm_aot`` succeeds and stamps ``aot_serve._MARKER_ATTR`` instead.  So on
every AOT arm the executor's three growth call sites are no-ops, the
``_shape_warm_republisher`` closure is constructed and discarded, and
and pgw#1010 deleted the republish backend outright (a grown JIT cache is
this pod's, and its cell had no consumer).

What that costs: a declared class outside the armed cell's envelope stays
eager for the life of the pod, every pod, forever.  Under dynamo it would be
routed eager once, warmed in one background thread and republished so the fleet
never pays it again — but the ratified reuse strategy is *AOT cells only, JIT
demotes to intake mode*, with pgw#731 deleting the very apparatus the growth
system lives inside.  "It heals on the JIT path" is therefore not an answer.

pgw#1184 DELETED THE BOOT-TIME CENSUS this paragraph used to quote (a
``compiled_shape_coverage`` row reading "2 of 18 declared graph classes served
compiled at boot").  It was an OBSERVATION: it could only name a class a warm
run happened to dispatch, and on sdxl the warm plan it depended on cost 18 full
generates per handler where the eager plan needs 2.  :data:`activity.
KIND_SHAPE_GAP` — reported per request at the ingress that refuses the class BY
NAME — is the same fact, measured on the traffic that actually asks for it.

What this module owns
---------------------
* the **vocabulary**: a shape gap is a named DECLARED CLASS outside the armed
  cell's ENVELOPE — the declared serving region, not a dynamo input signature —
  so it is arm-agnostic by construction;
* the **countable fact**: :data:`activity.KIND_SHAPE_GAP`, the AOT counterpart
  of pgw#680's ``guard_miss``, so the hub can count coverage holes on either
  arm with one grouped query;
* the **admission primitives** pgw#916 names as the parts of ``hot_swap`` to
  keep and lift: :class:`TurnGateBusy` / :class:`TurnGateClosed` (the pgw#677
  background GPU turn) and :class:`Debounce` (the coalesced republish).  They
  live here and ``hot_swap`` imports them, so there is exactly one
  implementation and the dynamo arm reaches the shared module rather than
  owning it;
* the **submission seam**: :func:`submit` hands a gap to the backend
  registered for its arm.

What this module deliberately does NOT own
------------------------------------------
A task queue, a GPU-turn scheduler, or a device executor.  Growth submits
through the sole owners (pgw#910 Reconciler, pgw#911 DeviceExecutor); building
a second scheduler here to close the AOT arm sooner would violate this issue's
own acceptance, which forbids exactly that.  ``tests/test_shape_growth_pgw916``
holds that line with a dependency test.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from . import activity as activity_mod

logger = logging.getLogger(__name__)

#: The execution arms a cell can be served on (pgw#891/th#1408 vocabulary).
#: The arm selects the COMPILER, never whether growth happens at all.
ARM_AOT = "aot"
ARM_DYNAMO = "dynamo"

#: Ingress reasons that mean "this class is not covered". ``no_entry_admits``
#: is a genuine coverage hole; ``entry_ambiguous`` is a DECLARATION defect
#: (pgw#917) that growth must never try to compile its way out of — it is
#: recorded and reported, never submitted.
REASON_UNCOVERED = "no_entry_admits"
REASON_AMBIGUOUS = "entry_ambiguous"


class TurnGateClosed(Exception):
    """The executor's background-turn gate is gone (shutdown/drain); the
    growth job is dropped and its class stays eager (pgw#677)."""


class TurnGateBusy(Exception):
    """No background turn within the bounded admission window (live tenant
    demand). The job re-queues instead of blocking the growth thread — a
    blocked compile for one instance must never head-of-line delay every
    other instance's jobs (pgw#677)."""


@dataclass(frozen=True)
class ShapeGap:
    """One request that arrived at a graph class the armed cell does not
    serve.

    ``declared_class`` is the point of the type: pgw#622's healing keyed on a
    dynamo input SIGNATURE, which is a concept the AOT arm does not have.  A
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
        compiling another entry cannot fix it and pgw#917 already refuses it
        at mint."""
        return self.reason != REASON_AMBIGUOUS


class GrowthBackend(Protocol):
    """The per-arm compiler. Selecting one is what ``arm`` is FOR."""

    def grow(self, gap: ShapeGap) -> bool:
        """Compile the named class in the background and adopt the grown
        cell. True when the work was accepted (not when it completed)."""


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

_BACKENDS: Dict[str, GrowthBackend] = {}
_BACKEND_LOCK = threading.Lock()


def register_backend(arm: str, backend: Optional[GrowthBackend]) -> None:
    """Bind (or unbind, with ``None``) the compiler for one arm."""
    with _BACKEND_LOCK:
        if backend is None:
            _BACKENDS.pop(str(arm), None)
        else:
            _BACKENDS[str(arm)] = backend


def backend_for(arm: str) -> Optional[GrowthBackend]:
    with _BACKEND_LOCK:
        return _BACKENDS.get(str(arm))


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


def submit(gap: ShapeGap) -> bool:
    """Hand one first-seen, growable gap to its arm's backend.

    False — with a named reason on the log — when the arm has no backend
    bound.  A silent no-op is the defect this issue exists to kill: the AOT
    arm's growth call sites returned False for months and the success log line
    that would have exposed it simply never printed.
    """
    if not gap.growable:
        logger.info(
            "shape-growth: %s is an ambiguous dispatch, not a coverage hole; "
            "growth cannot fix a declaration defect (pgw#917 refuses it at "
            "mint) — recorded, not submitted", gap.declared_class)
        return False
    backend = backend_for(gap.arm)
    if backend is None:
        logger.warning(
            "shape-growth: no growth backend registered for arm=%r, so the "
            "declared class %r stays EAGER for the life of this pod (pgw#916)",
            gap.arm, gap.declared_class)
        return False
    try:
        return bool(backend.grow(gap))
    except Exception:  # noqa: BLE001 — growth never fails a served request
        logger.warning(
            "shape-growth: backend for arm=%r refused %r",
            gap.arm, gap.declared_class, exc_info=True)
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"arm={gap.arm} class={gap.declared_class}: serve-window growth "
            f"backend raised; the class stays eager on this pod",
            phase="growth_failed",
        )
        return False


def report_and_submit(gap: ShapeGap) -> bool:
    """The one call a serving path makes: count it, and grow it once."""
    if not report(gap):
        return False
    return submit(gap)


@dataclass
class Debounce:
    """Coalesce republish bursts into serialized runs of ``fn`` on a
    background thread: at most one in flight, one queued.

    Lifted out of ``hot_swap`` verbatim (pgw#916 names it as one of the two
    parts to keep): republishing a grown cell is arm-agnostic — the artifact
    kind differs, the "at most one in flight, one queued" rule does not.
    """

    fn: Callable[[], None]
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _running: bool = False
    _dirty: bool = False

    def __call__(self) -> None:
        with self._lock:
            if self._running:
                self._dirty = True
                return
            self._running = True
        threading.Thread(
            target=self._run, name="cell-republish", daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                self.fn()
            except Exception:
                logger.warning(
                    "shape-growth: debounced callback failed", exc_info=True)
                # pgw#760: the debounced fn is the cell republish path — a
                # swallowed failure means the fleet re-compiles this class
                # forever.
                activity_mod.emit_event(
                    activity_mod.KIND_SERVE_DEGRADE,
                    "debounced cell-republish callback failed",
                    phase="republish_failed",
                )
            with self._lock:
                if not self._dirty:
                    self._running = False
                    return
                self._dirty = False


__all__ = [
    "ARM_AOT",
    "ARM_DYNAMO",
    "Debounce",
    "GrowthBackend",
    "GrowthLedger",
    "LEDGER",
    "REASON_AMBIGUOUS",
    "REASON_UNCOVERED",
    "ShapeGap",
    "TurnGateBusy",
    "TurnGateClosed",
    "backend_for",
    "register_backend",
    "report",
    "report_and_submit",
    "submit",
]
