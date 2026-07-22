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

ie#522 (2026-07-21): the watchdog's default liveness evidence was process
CPU seconds ONLY. Two real activities are honestly alive while burning
near-zero CPU on the reporting process: (1) an I/O-bound network model
fill (large composite checkpoints, tens of GB, minutes over a real link —
CPU-light by design); (2) inductor compile phases that fork subprocess
compile workers (torch's async_compile) — their CPU burn never showed up
in this process's own `time.process_time()`. Both read as "stalled" to the
hub's th#965 layer-3 rule after 10 minutes of no heartbeat, even mid
genuine progress. Reproduced live twice (ie#522 wan-2.2 animegen boot
warmup, two different RunPod hosts, identical ~9.5min self_mint_compile
CancelledError at phase=load). Fixed two ways below: `_process_cpu_evidence`
now sums live (not just reaped) child-process CPU via psutil, and
`note_progress()` lets an I/O callback (model-download byte ticks) heartbeat
the current activity directly — proof-of-life from genuine external
progress, independent of the CPU-sampling watchdog thread entirely.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import TracebackType
from typing import Awaitable, Callable, Optional

import psutil

from . import progress as progress_mod
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

KIND_SELF_MINT_COMPILE = "self_mint_compile"
KIND_WARMUP = "warmup"

PHASE_LOAD = "load"
PHASE_TRACE_GRAPH = "trace_graph"
PHASE_INDUCTOR_COMPILE = "inductor_compile"
PHASE_WARMUP_FORWARD = "warmup_forward"
PHASE_SEAL_PUBLISH = "seal_publish"
# gw#612: post-proof tail — sibling-lane resolution, publish decision,
# residency/target bookkeeping through readiness. Phases are logged
# verbatim hub-side (only kinds are enumerated in worker_activity.go).
PHASE_FINALIZE = "finalize"

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
    emit: Callable[["pb.WorkerMessage"], Awaitable[None]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Route reports onto the worker->hub stream: emit is the async
    WorkerMessage sender, loop the transport loop. Thread-safe emission."""
    def sink(update: pb.ActivityUpdate) -> None:
        async def _ship() -> None:
            await emit(pb.WorkerMessage(activity_update=update))
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(_ship())
        elif not loop.is_closed():
            asyncio.run_coroutine_threadsafe(_ship(), loop)
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
        self._counters: list[progress_mod.Counter] = []

    def counter(
        self, name: str, unit: str, total: float = 0.0,
    ) -> progress_mod.Counter:
        """Register-or-get a progress counter owned by this activity —
        finished automatically when the activity ends, so a reused name
        never carries a stale age into the next activity (gw#621)."""
        c = progress_mod.counter(name, unit, total)
        if c not in self._counters:
            self._counters.append(c)
        return c

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

    def progress_beat(
        self, snap: "progress_mod.Snapshot", self_stalled: bool = False,
    ) -> None:
        """One counter-carrying RUNNING update (gw#621), emitted from the
        10s app beat. The hub judges liveness by counter advancement; a
        self_stalled=True beat is the typed confession it kills on."""
        if self._done:  # racing completion: never re-open a terminal activity
            return
        _emit(pb.ActivityUpdate(
            kind=self.kind, phase=self._phase, step=self._step,
            total_steps=self._total, seq=_next_seq(),
            state=pb.ActivityState.ACTIVITY_STATE_RUNNING,
            counter=snap.name, counter_unit=snap.unit,
            counter_done=snap.done, counter_total=snap.total,
            rate_per_s=snap.rate_per_s, self_stalled=self_stalled,
            stalled_for_ms=int(snap.age_s * 1000) if self_stalled else 0,
            updated_at_unix_ms=int(time.time() * 1000),
        ))

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
    for c in act._counters:
        c.finish()
    act._counters = []


def current() -> Optional[Activity]:
    with _lock:
        act = _current
    return act if act is not None and not act._done else None


def on_beat() -> None:
    """Ride the 10s app heartbeat (lifecycle._heartbeat_loop, gw#621): while
    an activity is open and the progress registry has counters, emit one
    counter-carrying update per beat — frozen counters included (the hub's
    stall clock runs on non-advancement) — plus the typed self-diagnosis
    when even the freshest counter is stale past its per-phase window.
    Never raises; without a sink reports land on the logger."""
    try:
        act = current()
        if act is None:
            return
        snap = progress_mod.freshest()
        if snap is None:
            return
        act.progress_beat(
            snap, self_stalled=progress_mod.self_diagnosis() is not None)
    except Exception:
        logger.debug("progress beat dropped", exc_info=True)


# Floor between note_progress()-driven heartbeats: an active download can
# tick every few seconds (executor.py's _PROGRESS_EVENT_MIN_INTERVAL_S=5s);
# no need to re-report that often, just comfortably inside the hub's 10min
# stall window.
_PROGRESS_HEARTBEAT_MIN_INTERVAL_S = 5.0
_last_progress_heartbeat = 0.0


def note_progress() -> None:
    """Proof-of-life for the CURRENT activity from an external progress
    signal (model-download byte ticks, etc.) — an I/O-bound fill is
    CPU-light by design, so the watchdog's CPU-time evidence alone would
    read a healthy, slow-but-progressing network download as a stalled
    activity. Safe to call unconditionally from generic download code: a
    no-op when no activity is running, rate-limited otherwise."""
    global _last_progress_heartbeat
    now = time.monotonic()
    with _lock:
        if now - _last_progress_heartbeat < _PROGRESS_HEARTBEAT_MIN_INTERVAL_S:
            return
        _last_progress_heartbeat = now
        act = _current
    if act is not None and not act._done:
        act.heartbeat()


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


_this_process = psutil.Process()


def _process_cpu_evidence() -> float:
    """Process CPU seconds, including LIVE (not just reaped) child
    processes. torch's async_compile forks subprocess compile workers for
    inductor phases — resource.getrusage(RUSAGE_CHILDREN) only accounts for
    children that have already exited, which is useless for an in-flight
    compile burning real CPU in a still-running child. psutil reads /proc
    directly, so a live child's usage counts immediately. Best-effort: a
    child that can't be inspected (raced exit, permissions) just doesn't
    contribute — never fatal to the heartbeat."""
    total = time.process_time()
    try:
        children = _this_process.children(recursive=True)
    except psutil.Error:
        children = []
    for child in children:
        try:
            ct = child.cpu_times()
        except psutil.Error:
            continue
        total += ct.user + ct.system
    return total


def _process_io_evidence() -> float:
    """Process disk I/O bytes (read+write), MB granularity. Covers the
    on-disk model-LOAD phase (safetensors mmap/read + tensor
    materialization) — reproduced live: a worker was killed at
    `phase=load` after its network download had already finished, i.e.
    during the local-disk-to-GPU step, which is diffusers/torch/safetensors
    code gen-worker has no progress hook into. Process-level I/O accounting
    is a universal, non-invasive signal instead: real bytes move through
    the kernel's counters for this process regardless of which library
    triggered the read, so it needs no app-level instrumentation of
    third-party loading code at all. Unsupported platforms (no
    io_counters(), e.g. macOS) contribute 0 — never fatal."""
    try:
        io = _this_process.io_counters()
    except (psutil.Error, AttributeError, NotImplementedError):
        return 0.0
    return (io.read_bytes + io.write_bytes) / (1 << 20)


def _default_evidence() -> float:
    """Combined default watchdog evidence: process+live-children CPU
    seconds PLUS process disk I/O megabytes. Either source alone advancing
    proves genuine life: a network download followed by on-disk model load
    is CPU-light throughout but moves real bytes the whole way (covers
    both halves of "load"); an inductor compile subprocess burns real
    (child) CPU while GPU sits idle and I/O sits flat; a true hang
    advances neither."""
    return _process_cpu_evidence() + _process_io_evidence()


class watchdog:
    """Bracket for a long call that may stay wire-silent (inductor compile,
    large fuse): a background thread samples an evidence counter every
    interval and heartbeats the activity ONLY while evidence advances. A hung
    call stops accruing evidence, the beat stops within one interval, and the
    hub's stall rule takes it from there.

    Default evidence is process+children CPU seconds plus process disk I/O
    (see _default_evidence); pass a monotonic callable (e.g.
    compile-wall-seconds) for calls with a better signal."""

    def __init__(
        self,
        act: Activity,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        evidence: Optional[Callable[[], float]] = None,
    ) -> None:
        self._act = act
        self._interval = interval_s
        self._evidence = evidence or _default_evidence
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="activity-watchdog", daemon=True,
        )

    def _run(self) -> None:
        try:
            base = last = self._evidence()
        except Exception:
            base = last = 0.0
        while not self._stop.wait(self._interval):
            try:
                now = self._evidence()
            except Exception:
                continue
            if now - last >= _EVIDENCE_EPS:
                last = now
                # gw#621: evidence advance is ALSO a registry counter, so the
                # 10s beat reports it and the hub sees a moving number
                # instead of inferring health.
                self._counter.set_done(now - base)
                self._act.heartbeat()

    def __enter__(self) -> "watchdog":
        self._counter = progress_mod.counter(
            f"evidence:{self._act.kind}", progress_mod.UNIT_EVIDENCE)
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
        self._counter.finish()
