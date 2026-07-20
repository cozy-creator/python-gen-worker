"""Generic worker-activity progress (gw#601 / th#929).

The worker reports whatever internal job it is doing — self-mint compile,
warmup — as ActivityUpdate envelopes on the existing worker->hub stream.
Liveness contract: while an activity runs, seq keeps advancing (phase
transitions, step advances, or watchdog heartbeats around long silent calls);
the hub enforces ONE generic stall rule on silence. A silent death is a bug
by contract: terminal FAILED carries the exception.

Kind/phase strings are wire-shared with tensorhub
(internal/orchestrator/grpc/worker_activity.go) — keep them identical.

Without a bound transport sink (cozy-local, tests) reports land on the
logger, which IS the local progress UI.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import TracebackType
from typing import Any, Callable, Coroutine, Optional

from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

KIND_SELF_MINT_COMPILE = "self_mint_compile"
KIND_WARMUP = "warmup"

PHASE_LOAD = "load"
PHASE_TRACE_GRAPH = "trace_graph"
PHASE_INDUCTOR_COMPILE = "inductor_compile"
PHASE_WARMUP_FORWARD = "warmup_forward"
PHASE_SEAL_PUBLISH = "seal_publish"

# Default watchdog cadence; the hub's stall rule (~10 min) tolerates many
# missed beats.
HEARTBEAT_INTERVAL_S = 60.0
# Minimum evidence advance (process CPU seconds) per interval for a heartbeat:
# a hung (blocked/deadlocked) call stops accruing CPU and the beat stops with
# it, which is exactly the silence the hub enforces on.
_EVIDENCE_EPS = 0.05

_lock = threading.Lock()
_seq = 0
_sink: Optional[Callable[[pb.ActivityUpdate], None]] = None
_current: Optional["Activity"] = None


def bind_sink(
    emit: Callable[["pb.WorkerMessage"], Coroutine[Any, Any, None]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Route reports onto the worker->hub stream: emit is the async
    WorkerMessage sender, loop the transport loop. Thread-safe emission."""
    def sink(update: pb.ActivityUpdate) -> None:
        coro = emit(pb.WorkerMessage(activity_update=update))
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
        elif not loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            coro.close()
    global _sink
    with _lock:
        _sink = sink


def _next_seq() -> int:
    global _seq
    with _lock:
        _seq += 1
        return _seq


def _emit(update: pb.ActivityUpdate) -> None:
    with _lock:
        sink = _sink
    try:
        if sink is not None:
            sink(update)
        else:
            state = pb.ActivityState.Name(update.state)
            logger.info(
                "[activity] %s %s %s/%s %s %s", update.kind, update.phase,
                update.step, update.total_steps, state, update.error or update.detail,
            )
    except Exception:  # reporting must never break the work it reports on
        logger.debug("activity report dropped", exc_info=True)


class Activity:
    """One running activity. Use begin() / the context manager `running()`."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._phase = ""
        self._step = 0
        self._total = 0
        self._done = False

    def _report(self, state: "pb.ActivityState", error: str = "", detail: str = "") -> None:
        _emit(pb.ActivityUpdate(
            kind=self.kind, phase=self._phase, step=self._step,
            total_steps=self._total, seq=_next_seq(), state=state,
            error=error, detail=detail,
            updated_at_unix_ms=int(time.time() * 1000),
        ))

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self._phase, self._step, self._total = phase, step, total
        self._report(pb.ActivityState.ACTIVITY_STATE_RUNNING)

    def heartbeat(self) -> None:
        """Re-report the current phase with a fresh seq (liveness proof)."""
        self._report(pb.ActivityState.ACTIVITY_STATE_RUNNING)

    def completed(self) -> None:
        if not self._done:
            self._done = True
            self._report(pb.ActivityState.ACTIVITY_STATE_COMPLETED)
        _end(self)

    def failed(self, exc: BaseException) -> None:
        """The typed activity_failed terminal — a silent death is a bug."""
        if not self._done:
            self._done = True
            self._report(
                pb.ActivityState.ACTIVITY_STATE_FAILED,
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
        _end(self)


def begin(kind: str, phase: str = "") -> Activity:
    global _current
    act = Activity(kind)
    with _lock:
        _current = act
    act.phase(phase) if phase else act.heartbeat()
    return act


def current_phase(phase: str, step: int = 0, total: int = 0) -> None:
    """Report a phase on the current activity; no-op when none is running.
    Setups serialize under the executor load lock, so one current is enough."""
    with _lock:
        act = _current
    if act is not None and not act._done:
        act.phase(phase, step, total)


def _end(act: Activity) -> None:
    global _current
    with _lock:
        if _current is act:
            _current = None


class running:
    """Context manager: begin() on enter; COMPLETED on clean exit, FAILED
    (carrying the exception) on raise."""

    def __init__(self, kind: str, phase: str = "") -> None:
        self._kind, self._phase = kind, phase
        self.activity: Optional[Activity] = None

    def __enter__(self) -> Activity:
        self.activity = begin(self._kind, self._phase)
        return self.activity

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        assert self.activity is not None
        if exc is not None:
            self.activity.failed(exc)
        else:
            self.activity.completed()
        _end(self.activity)


def _process_cpu_evidence() -> float:
    return time.process_time()


class watchdog:
    """Bracket for a long call that may stay wire-silent (inductor compile,
    large fuse): a background thread samples an evidence counter every
    interval and heartbeats the activity ONLY while evidence advances. A hung
    call stops accruing evidence, the beat stops within one interval, and the
    hub's stall rule takes it from there.

    Default evidence is process CPU seconds; pass a monotonic callable
    (e.g. compile-wall-seconds) for calls with better signals."""

    def __init__(
        self,
        act: Activity,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        evidence: Optional[Callable[[], float]] = None,
    ) -> None:
        self._act = act
        self._interval = interval_s
        self._evidence = evidence or _process_cpu_evidence
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="activity-watchdog", daemon=True,
        )

    def _run(self) -> None:
        try:
            last = self._evidence()
        except Exception:
            last = 0.0
        while not self._stop.wait(self._interval):
            try:
                now = self._evidence()
            except Exception:
                continue
            if now - last >= _EVIDENCE_EPS:
                last = now
                self._act.heartbeat()

    def __enter__(self) -> "watchdog":
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
