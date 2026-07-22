"""gw#601 generic worker-activity progress: monotonic phase envelopes over
the real executor setup/warmup path, the evidence-gated watchdog heartbeat
(an induced hang stops the beat within one interval), and the typed
activity_failed terminal (an induced crash never dies silently)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import msgspec
import pytest

from gen_worker import activity
from gen_worker.api import Resources, endpoint
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


@pytest.fixture(autouse=True)
def _reset_activity_sink():
    yield
    with activity._lock:
        activity._sink = None
        activity._current = None
        activity._last_progress_heartbeat = 0.0


def _updates(sent: List[pb.WorkerMessage]) -> List[pb.ActivityUpdate]:
    return [m.activity_update for m in sent if m.WhichOneof("msg") == "activity_update"]


# ---------------------------------------------------------------------------
# real executor code path: scripted setup+warmup emits monotonic phases
# ---------------------------------------------------------------------------


def test_executor_setup_emits_monotonic_activity_phases():
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

        def generate_turbo(self, ctx, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    async def _go() -> None:
        await ex.ensure_setup(specs[0])
        # the sink schedules sends onto this loop; let them flush
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(_go())

    ups = _updates(sent)
    assert ups, "no activity envelopes emitted"
    assert all(u.kind == activity.KIND_WARMUP for u in ups)
    seqs = [u.seq for u in ups]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), seqs
    phases = [(u.phase, u.step, u.total_steps, u.state) for u in ups]
    running = pb.ActivityState.ACTIVITY_STATE_RUNNING
    assert phases[0] == (activity.PHASE_LOAD, 0, 0, running)
    assert (activity.PHASE_WARMUP_FORWARD, 1, 2, running) in phases
    assert (activity.PHASE_WARMUP_FORWARD, 2, 2, running) in phases
    assert ups[-1].state == pb.ActivityState.ACTIVITY_STATE_COMPLETED


def test_executor_setup_crash_emits_typed_activity_failed():
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(vram_gb=8))
    class Ep:
        def setup(self) -> None:
            raise RuntimeError("induced setup crash")

        def generate(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y="ok")

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    async def _go() -> None:
        with pytest.raises(RuntimeError, match="induced setup crash"):
            await ex.ensure_setup(specs[0])
        for _ in range(10):
            await asyncio.sleep(0)

    asyncio.run(_go())

    ups = _updates(sent)
    assert ups and ups[-1].state == pb.ActivityState.ACTIVITY_STATE_FAILED
    assert "RuntimeError: induced setup crash" in ups[-1].error


# ---------------------------------------------------------------------------
# watchdog: heartbeats only while evidence advances
# ---------------------------------------------------------------------------


def test_watchdog_heartbeats_while_evidence_advances_then_stops_on_hang():
    reports: List[pb.ActivityUpdate] = []
    with activity._lock:
        activity._sink = reports.append

    evidence_val = [0.0]
    advancing = threading.Event()
    advancing.set()

    def evidence() -> float:
        if advancing.is_set():
            evidence_val[0] += 1.0
        return evidence_val[0]

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_INDUCTOR_COMPILE)
    baseline = len(reports)
    interval = 0.05
    with activity.watchdog(act, interval_s=interval, evidence=evidence):
        time.sleep(6 * interval)
        beats_while_advancing = len(reports) - baseline
        # induced hang: the wrapped call stops accruing evidence
        advancing.clear()
        time.sleep(2 * interval)  # one interval of grace, then silence
        beats_at_hang = len(reports)
        time.sleep(6 * interval)
        beats_after_hang = len(reports)
    act.completed()

    assert beats_while_advancing >= 2, "no heartbeats while evidence advanced"
    assert beats_after_hang == beats_at_hang, "heartbeat outlived the hang"
    assert reports[-1].state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
    seqs = [u.seq for u in reports]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_running_context_manager_reports_failed_with_exception():
    reports: List[pb.ActivityUpdate] = []
    with activity._lock:
        activity._sink = reports.append

    with pytest.raises(ValueError):
        with activity.running(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD):
            raise ValueError("mint exploded")

    assert reports[-1].state == pb.ActivityState.ACTIVITY_STATE_FAILED
    assert "ValueError: mint exploded" in reports[-1].error
    # terminal cleared the current activity: later phase reports are no-ops
    activity.current_phase(activity.PHASE_SEAL_PUBLISH)
    assert reports[-1].state == pb.ActivityState.ACTIVITY_STATE_FAILED


# ---------------------------------------------------------------------------
# ie#522: default evidence must not blind-spot two honest liveness sources —
# live subprocess CPU (inductor's async_compile workers) and I/O-bound
# download-byte progress (large model fills, CPU-light by design). Both
# were reproduced live killing a healthy worker at ~9.5-10min (th#965 layer
# 3), twice, on two different hosts.
# ---------------------------------------------------------------------------


def test_process_cpu_evidence_counts_live_child_process_cpu():
    """A LIVE (still-running) child process burning real CPU must move the
    default evidence function even though this process's own
    time.process_time() barely moves — resource.RUSAGE_CHILDREN only
    accounts for already-reaped children, which is exactly wrong for an
    in-flight inductor compile subprocess."""
    before_self = time.process_time()
    before_combined = activity._process_cpu_evidence()

    # A real child process that busy-loops burning CPU for ~1s wall time —
    # never reaped until after we've sampled evidence while it's alive.
    proc = subprocess.Popen([
        sys.executable, "-c",
        "import time\nt=time.process_time()\n"
        "while time.process_time() - t < 1.0: pass",
    ])
    try:
        time.sleep(0.5)  # let the child accrue real CPU while still running
        mid_self = time.process_time()
        mid_combined = activity._process_cpu_evidence()
    finally:
        proc.wait(timeout=5)

    self_delta = mid_self - before_self
    combined_delta = mid_combined - before_combined
    assert combined_delta > self_delta + 0.1, (
        f"child CPU not counted: self_delta={self_delta}, "
        f"combined_delta={combined_delta}"
    )


def test_note_progress_heartbeats_current_activity_rate_limited():
    reports: List[pb.ActivityUpdate] = []
    with activity._lock:
        activity._sink = reports.append

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    baseline = len(reports)

    activity.note_progress()
    after_first = len(reports)
    activity.note_progress()  # inside the rate-limit floor: must not double-report
    after_second = len(reports)

    assert after_first == baseline + 1, "note_progress() did not heartbeat"
    assert after_second == after_first, "note_progress() ignored the rate limit"

    act.completed()
    # no current activity: must be a safe no-op, not an error
    activity.note_progress()


def test_watchdog_survives_zero_cpu_download_via_note_progress(monkeypatch):
    """REVERT-TURNS-RED (failure mode a): a fake download that advances
    ONLY via note_progress() ticks — zero CPU burn on this thread or any
    child, exactly a network-bound fill — must keep the activity alive
    while the watchdog's evidence sees nothing at all. A pre-fix worker
    (CPU-evidence watchdog only, no note_progress channel) goes silent
    here and the hub kills it at 10min; this activity must never go
    silent while download ticks keep arriving.

    De-flaked (ie#522): the original real-clock version (0.3s window,
    real default evidence) let note_progress() land exactly ONE beat (the
    5s rate-limit floor swallowed the rest) and silently depended on
    incidental watchdog CPU/IO noise for the second — present on a
    32-core dev box, absent on 2-core CI runners. Determinism now comes
    from three changes that keep the intent intact: the rate-limit floor
    is dropped for this test (the floor itself is covered by
    test_note_progress_heartbeats_current_activity_rate_limited), the
    watchdog evidence is pinned to a constant zero (a perfectly silent
    download — and every counted beat is attributable to note_progress()
    alone, so the revert-turns-red property is airtight), and the wait is
    event-driven on beat count with a generous deadline instead of a
    fixed wall-clock window."""
    reports: List[pb.ActivityUpdate] = []
    with activity._lock:
        activity._sink = reports.append
        activity._last_progress_heartbeat = 0.0
    monkeypatch.setattr(activity, "_PROGRESS_HEARTBEAT_MIN_INTERVAL_S", 0.0)

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    baseline = len(reports)
    stop = threading.Event()

    def _fake_download_ticker() -> None:
        # No CPU work at all — just sleeps and calls note_progress(), the
        # exact shape of cozy_snapshot.py's _on_bytes hook during a real
        # network fetch.
        while not stop.is_set():
            activity.note_progress()
            time.sleep(0.01)

    ticker = threading.Thread(target=_fake_download_ticker, daemon=True)
    ticker.start()
    try:
        # Constant evidence: the watchdog thread can never heartbeat, so
        # every beat below is proof-of-life from note_progress() ticks.
        with activity.watchdog(act, interval_s=0.05, evidence=lambda: 0.0):
            deadline = time.monotonic() + 10.0
            while len(reports) - baseline < 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            beats_during = len(reports) - baseline
    finally:
        stop.set()
        ticker.join(timeout=2)
    act.completed()

    assert beats_during >= 3, (
        f"activity went silent during a zero-CPU download tick stream "
        f"(beats={beats_during}) — note_progress() must keep it alive "
        f"independent of the CPU watchdog"
    )


def test_watchdog_survives_zero_cpu_disk_load_via_io_evidence():
    """REVERT-TURNS-RED (failure mode a, part 2): the on-disk model-LOAD
    step (safetensors mmap + tensor materialization, AFTER any network
    download has already finished) is third-party diffusers/torch/
    safetensors code gen-worker has no progress hook into — no
    note_progress() call reaches it. Reproduced live: the kill landed at
    `phase=load` on a retry whose download evidence should have been
    fine, i.e. the SAME phase covers a second CPU-light sub-step this test
    isolates. Proves the DEFAULT watchdog evidence (_default_evidence, not
    a synthetic override) stays alive from REAL disk I/O bytes alone —
    zero CPU burn, zero note_progress() calls — via psutil.io_counters()."""
    reports: List[pb.ActivityUpdate] = []
    with activity._lock:
        activity._sink = reports.append

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    baseline = len(reports)
    stop = threading.Event()

    def _fake_disk_load() -> None:
        # Real disk writes (page cache I/O the kernel actually counts),
        # deliberately with no CPU-heavy work and no activity/note_progress
        # calls at all — exactly the shape of a third-party mmap-based
        # loader gen-worker cannot instrument.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fake_shard.bin"
            chunk = b"\0" * (1 << 20)  # 1 MiB
            with open(path, "wb") as f:
                while not stop.is_set():
                    f.write(chunk)
                    f.flush()
                    os_fsync_safe(f)
                    time.sleep(0.02)

    ticker = threading.Thread(target=_fake_disk_load, daemon=True)
    ticker.start()
    try:
        with activity.watchdog(act, interval_s=0.05):
            time.sleep(0.3)
            beats_during = len(reports) - baseline
    finally:
        stop.set()
        ticker.join(timeout=2)
    act.completed()

    assert beats_during >= 1, (
        f"activity went silent during real (zero-CPU) disk I/O "
        f"(beats={beats_during}) — the default evidence must pick up "
        f"process io_counters(), not just CPU time"
    )


def os_fsync_safe(f) -> None:
    """fsync forces the write to actually land (bumping io_counters
    reliably) instead of possibly sitting in a page-cache buffer the OS
    hasn't accounted as I/O yet; best-effort on filesystems that reject it."""
    import os

    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def test_watchdog_still_detects_a_genuinely_dead_activity():
    """REVERT-TURNS-RED (failure mode b): with NEITHER CPU work NOR
    note_progress() ticks, the activity must still go silent and stay
    silent — the fix must not make the stall detector toothless."""
    reports: List[pb.ActivityUpdate] = []
    with activity._lock:
        activity._sink = reports.append
        activity._last_progress_heartbeat = 0.0

    act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
    baseline = len(reports)
    with activity.watchdog(act, interval_s=0.05):
        time.sleep(0.3)  # truly idle: no CPU burn, no note_progress() calls
        beats = len(reports) - baseline
    act.completed()

    # ~6 watchdog ticks fit in this window; a healthy stop-on-hang detector
    # heartbeats on effectively none of them. Allow <=1 for incidental
    # background CPU noise in the test process itself (interpreter/pytest
    # housekeeping can tick time.process_time() a hair over _EVIDENCE_EPS
    # once) — the real bug this guards is "heartbeats every tick regardless
    # of evidence", which test_watchdog_survives_zero_cpu_download's
    # beats_during>=2-out-of-similar-ticks would NOT distinguish from this
    # if this threshold were toothless.
    assert beats <= 1, f"a dead activity heartbeated repeatedly (beats={beats})"
