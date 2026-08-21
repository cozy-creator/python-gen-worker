"""Worker-activity progress on the worker->hub stream. Kind/phase strings are wire-shared with tensorhub (internal/orchestrator/grpc/worker_activity.go) — keep them identical. While an activity runs, seq must keep advancing (the hub enforces one stall rule on silence); without a bound sink, reports land on the logger."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from types import TracebackType
from typing import Awaitable, Callable, Optional

import psutil

from . import progress as progress_mod
from . import warm_spans
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

KIND_SELF_MINT_COMPILE = "self_mint_compile"
KIND_WARMUP = "warmup"
KIND_WARMUP_SUMMARY = "warmup_summary"
KIND_GUARD_MISS = "guard_miss"
KIND_GUARD_LEAK = "guard_leak"
KIND_SHAPE_GAP = "shape_gap"
KIND_SERVE_DEGRADE = "serve_degrade"
KIND_SERVE_POSTURE = "serve_posture"
KIND_LORA_HYGIENE = "lora_hygiene"
KIND_LORA_FIDELITY = "lora_fidelity"
KIND_COMPILED_GRAPH_NUMERICS = "compiled_graph_numerics"
KIND_MODULAR_HYDRATION = "modular_hydration"
KIND_COMPONENT_MISS = "component_miss"
KIND_OUTPUT_INTEGRITY = "output_integrity"
KIND_ROTATION_PRELOAD = "rotation_preload"
KIND_CAPABILITY_RENEWAL = "capability_renewal"
KIND_RESIDENCY_FAULT = "residency_fault"
# THE LANE THE PLATFORM RESOLVED, emitted once per (model class, checkpoint)
# by `serving.serve_loop._resolve_for` — the one moment both facts are in hand,
# the card and what the deploy staged.
#
# pgw#1620 REWROTE THIS PRODUCER, and the correction matters more than the
# field: pgw#1104's emitter was `gen_worker.report_applied_lane()`, an ENDPOINT
# calling in after `quantize_()` to say what it had done. pgw#1599 retired that
# call ("the lane is the Model's declared contract") and gave the job to
# nothing, so every migrated endpoint went silent — no `applied_lane` row, and
# `metrics.lane` absent, on real completed production requests. The successor
# is strictly better than what v1 had: the report now comes from the ladder
# that DECIDED the lane, not from an endpoint remembering to say so.
#
# `phase` is the ranked lane BODY (`bf16-w16a16`), a closed fleet vocabulary
# and therefore countable hub-side; `family` carries the LAYOUT STAMP, which
# under tensor-layout v2 is the pair rendering `sdxl.diffusers@1+plain.bf16@1`
# (pgw#1621 — it used to be the v1 lane handle `sdxl.diffusers-bf16@1`, which
# survives only as a display name). The pair is the spelling the hub already
# stores in `checkpoints.layout_topology`/`layout_quant` and renders back with
# the same `+`, so the two sides join without anything parsing prose; `detail`
# is the ladder's whole confession — card,
# reason, and every rejected rung with its numbers. That last part is the point:
# a ladder that reports only its winner cannot be audited, and its confession
# was a `logger.info` on a pod with no logs API, which is to say nowhere.
KIND_APPLIED_LANE = "applied_lane"
KIND_APPLIED_ATTENTION = "applied_attention"
KIND_BOOT_ADOPT = "boot_adopt"
KIND_BOOT_ADOPT_SUMMARY = "boot_adopt_summary"
KIND_ADOPT_REFUSED = "adopt_refused"
KIND_BOOT_MEMO = "boot_memo_honesty"
KIND_JIT_COMPILE = "jit_compile"
KIND_AOT_MINT = "aot_mint_phases"
KIND_MEASURE_ONLY = "measure_only"
KIND_COMPILE_CHILD = "compile_child"
KIND_PROCESS_ROLE = "process_role"
KIND_SNAPSHOT_PULL = "snapshot_pull"
KIND_SNAPSHOT_CENSUS = "snapshot_census"
# Cold-boot per-stage spans plus one terminal roll-up. The stage vocabulary is CLOSED (boot_stages.Stage): a renderer in another repo binds to these tokens.
KIND_BOOT_STAGES = "boot_stages"
KIND_ENGINE_BOOT = "engine_boot"
KIND_WEIGHT_FETCH = "weight_fetch"
# pgw#1630: the boot materialization span is TELEMETRY. The watchdog verdict is
# kernel evidence only (procsplit/liveness.py), so a fill that declares nothing
# loses a LABEL, not a process. Its OWN kind, deliberately NOT `weight_fetch`:
# that kind's rows are the closed-vocabulary BYTE POSITIONS
# (`weight_position._emit`), and a span with an empty phase mixed into them
# would corrupt the one signal th#2191 reads.
KIND_BOOT_MATERIALIZE = "boot_materialize"
PHASE_MINTED = "minted"

PHASE_LOAD = "load"
PHASE_TRACE_GRAPH = "trace_graph"
PHASE_INDUCTOR_COMPILE = "inductor_compile"
PHASE_ROUTER_DRAIN = warm_spans.PHASE_ROUTER_DRAIN
PHASE_WARMUP_FORWARD = "warmup_forward"
PHASE_SEAL_PUBLISH = "seal_publish"
PHASE_FINALIZE = "finalize"

HEARTBEAT_INTERVAL_S = 60.0
EVIDENCE_EPS = 0.05

_lock = threading.Lock()
_seq = 0
_sink: Optional[Callable[[pb.ActivityUpdate], None]] = None
_current: Optional["Activity"] = None


def bind_sink(
    emit: Callable[["pb.WorkerMessage"], Awaitable[None]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Route reports onto the worker->hub stream: emit is the async WorkerMessage sender, loop the transport loop."""
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


def reset_for_tests() -> None:
    """Drop the bound sink and any in-flight activity."""
    global _sink, _current
    with _lock:
        _sink = None
        _current = None


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
                "[activity] %s %s %s/%s %s%s %s", update.kind, update.phase,
                update.step, update.total_steps, state,
                f" {update.duration_ms / 1000:.1f}s" if update.duration_ms else "",
                update.error or update.detail,
            )
    except Exception:
        logger.debug("activity report dropped", exc_info=True)


class Activity:
    """One running activity."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.id = f"{kind}:{_next_seq()}"
        self._phase = ""
        self._step = 0
        self._total = 0
        self._done = False
        self._counters: dict[str, tuple[str, progress_mod.Counter]] = {}

    def counter(
        self, name: str, unit: str, total: float = 0.0,
    ) -> progress_mod.Counter:
        """Register-or-get a progress counter owned by this activity's CURRENT PHASE — finished when the phase changes, and again when the activity ends (gw#621)."""
        c = progress_mod.counter(name, unit, total, owner=self.id)
        self._counters[name] = (self._phase, c)
        return c

    def _finish_counters(self, keep_phase: Optional[str] = None) -> None:
        for name, (phase, c) in list(self._counters.items()):
            if keep_phase is not None and phase == keep_phase:
                continue
            c.finish()
            del self._counters[name]

    def _report(self, state: "pb.ActivityState", error: str = "", detail: str = "") -> None:
        _emit(pb.ActivityUpdate(
            kind=self.kind, phase=self._phase, step=self._step,
            total_steps=self._total, seq=_next_seq(), state=state,
            error=error, detail=detail,
            updated_at_unix_ms=int(time.time() * 1000),
        ))

    @property
    def phase_name(self) -> str:
        return self._phase

    def phase(self, phase: str, step: int = 0, total: int = 0) -> None:
        self._phase, self._step, self._total = phase, step, total
        self._finish_counters(keep_phase=phase)
        self._report(pb.ActivityState.ACTIVITY_STATE_RUNNING)

    def heartbeat(self) -> None:
        """Re-report the current phase with a fresh seq (liveness proof)."""
        self._report(pb.ActivityState.ACTIVITY_STATE_RUNNING)

    def progress_beat(
        self, snap: "progress_mod.Snapshot", self_stalled: bool = False,
    ) -> None:
        """One counter-carrying RUNNING update (gw#621), emitted from the 10s app beat."""
        if self._done:
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

    def note(self, detail: str) -> None:
        if self._done:
            return
        self._report(
            pb.ActivityState.ACTIVITY_STATE_RUNNING, detail=detail[:2000])

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

    def retrying(self, exc: BaseException, attempt: int, max_attempts: int) -> None:
        """gw#661: this attempt lost, but the work is contractually re-attempted — so the activity is still RUNNING, not FAILED."""
        if self._done:
            _end(self)
            return
        self._done = True
        self._report(
            pb.ActivityState.ACTIVITY_STATE_RUNNING,
            detail=(
                f"retrying (attempt {attempt}/{max_attempts}): "
                f"{type(exc).__name__}: {exc}"
            )[:2000],
        )
        _end(self)


# Compiled-graph identity travels as typed wire fields (proto fields 18-20). proto/worker_scheduler.proto is a byte-for-byte vendored copy of tensorhub's canonical one, gated on PROTO_DIGEST — never rename a field here unilaterally; the one vocabulary translation lives in emit_event.


def emit_event(
    kind: str, detail: str, phase: str = "", duration_ms: int = 0,
    *, family: str = "", compiled_graph_key: str = "", graph_specialization: str = "",
    step: int = 0, total_steps: int = 0,
) -> None:
    """One self-contained COMPLETED ActivityUpdate — a countable typed event, not a running activity (bypasses begin() so it cannot strand a concurrently open activity). duration_ms=0 and step/total_steps=0 mean "not measured / not a count" — hub readers filter on > 0."""
    _emit(pb.ActivityUpdate(
        kind=kind, phase=phase[:300], seq=_next_seq(),
        step=max(0, int(step)), total_steps=max(0, int(total_steps)),
        state=pb.ActivityState.ACTIVITY_STATE_COMPLETED,
        detail=detail[:2000],
        family=str(family or "")[:200],
        compiled_graph_key=str(compiled_graph_key or "")[:200],
        graph_specialization=str(graph_specialization or "")[:300],
        duration_ms=max(0, int(duration_ms)),
        updated_at_unix_ms=int(time.time() * 1000),
    ))


def begin(kind: str, phase: str = "") -> Activity:
    global _current
    act = Activity(kind)
    with _lock:
        _current = act
    act.phase(phase) if phase else act.heartbeat()
    return act


def current_phase(phase: str, step: int = 0, total: int = 0) -> None:
    """Report a phase on the current activity; no-op when none is running."""
    with _lock:
        act = _current
    if act is not None and not act._done:
        act.phase(phase, step, total)


def current_note(detail: str) -> None:
    with _lock:
        act = _current
    if act is not None and not act._done:
        act.note(detail)


def _end(act: Activity) -> None:
    global _current
    with _lock:
        if _current is act:
            _current = None
    act._finish_counters()


def current() -> Optional[Activity]:
    with _lock:
        act = _current
    return act if act is not None and not act._done else None


def scoped_counter(
    name: str, unit: str, total: float = 0.0,
) -> "progress_mod.Counter":
    act = current()
    if act is not None:
        return act.counter(name, unit, total)
    return progress_mod.counter(name, unit, total)


def on_beat() -> None:
    """Ride the 10s app heartbeat (lifecycle._heartbeat_loop, gw#621): while an activity is open and the progress registry has counters, emit one counter-carrying update per beat — frozen counters include..."""
    try:
        act = current()
        if act is None:
            return
        snap = progress_mod.freshest(act.id)
        if snap is None:
            return
        act.progress_beat(
            snap,
            self_stalled=progress_mod.self_diagnosis(act.id) is not None)
    except Exception:
        logger.debug("progress beat dropped", exc_info=True)


_PROGRESS_HEARTBEAT_MIN_INTERVAL_S = 5.0
_last_progress_heartbeat = 0.0


def note_progress() -> None:
    """Proof-of-life for the CURRENT activity from an external progress signal (model-download byte ticks, etc.) — an I/O-bound fill is CPU-light by design, so the watchdog's CPU-time evidence alone would..."""
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
    """Context manager: begin() on enter; COMPLETED on clean exit, FAILED (carrying the exception) on raise."""

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
    """Sum live child CPU (psutil, /proc) AND reaped-child CPU (rusage): a child's CPU moves from its own counters to the parent's cutime/cstime the instant it is reaped, so either source alone is non-monotonic. Best-effort; never fatal to the heartbeat."""
    try:
        times = _this_process.cpu_times()
        total = float(times.user) + float(times.system)
        total += float(getattr(times, "children_user", 0.0) or 0.0)
        total += float(getattr(times, "children_system", 0.0) or 0.0)
    except psutil.Error:
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
        total += float(ct.user) + float(ct.system)
        total += float(getattr(ct, "children_user", 0.0) or 0.0)
        total += float(getattr(ct, "children_system", 0.0) or 0.0)
    return total


def _process_io_evidence() -> float:
    try:
        io = _this_process.io_counters()
    except (psutil.Error, AttributeError, NotImplementedError):
        return 0.0
    return (io.read_bytes + io.write_bytes) / (1 << 20)


def default_evidence() -> float:
    """Combined default watchdog evidence: process+live-children CPU seconds PLUS process disk I/O megabytes."""
    return _process_cpu_evidence() + _process_io_evidence()


class watchdog:
    """Bracket for a long call that may stay wire-silent (inductor compile, large fuse): a background thread samples an evidence counter every interval and heartbeats the activity ONLY while evidence adva..."""

    def __init__(
        self,
        act: Activity,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
        evidence: Optional[Callable[[], float]] = None,
    ) -> None:
        self._act = act
        self._interval = interval_s
        self._evidence = evidence or default_evidence
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
            if now - last >= EVIDENCE_EPS:
                last = now
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
