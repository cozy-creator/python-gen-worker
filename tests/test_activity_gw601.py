"""gw#601 generic worker-activity progress: monotonic phase envelopes over
the real executor setup/warmup path, the evidence-gated watchdog heartbeat
(an induced hang stops the beat within one interval), and the typed
activity_failed terminal (an induced crash never dies silently)."""

from __future__ import annotations

import asyncio
import threading
import time
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
