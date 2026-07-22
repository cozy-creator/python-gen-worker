"""gw#621 progress registry + beat: named monotonic counters, per-family
self-diagnosis windows, counter-carrying beats on the app heartbeat, and
the end-to-end confession path over the hub double.

Revert-turns-red guards for the ie#522 false-positive class: a CPU-quiet
but byte-advancing download is visibly ALIVE on the wire (counter beats,
never self_stalled); a genuinely frozen counter still confesses."""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from gen_worker import activity, progress
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double


@pytest.fixture(autouse=True)
def _reset():
    progress.reset()
    yield
    progress.reset()
    with activity._lock:
        activity._sink = None
        activity._current = None


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _updates(sent: List[pb.WorkerMessage]) -> List[pb.ActivityUpdate]:
    return [m.activity_update for m in sent
            if m.WhichOneof("msg") == "activity_update"]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_counter_monotonic_freshest_and_windows(monkeypatch):
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)
    c = progress.counter("download:a", progress.UNIT_BYTES, total=100.0)
    c.set_done(10.0)
    clk.t += 5
    c.set_done(5.0)  # backwards: ignored — monotonic, age keeps growing
    s = progress.freshest()
    assert s is not None
    assert s.done == 10.0 and s.total == 100.0 and s.age_s == 5.0
    assert s.window_s == progress.STALL_WINDOW_S["download"]
    assert progress.self_diagnosis() is None

    clk.t += progress.STALL_WINDOW_S["download"]
    diag = progress.self_diagnosis()
    assert diag is not None and diag.name == "download:a"

    # ANY other advancing counter proves the process alive again.
    progress.counter("upload:bytes", progress.UNIT_BYTES).add(1.0)
    assert progress.self_diagnosis() is None


def test_unknown_family_gets_default_window(monkeypatch):
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)
    progress.counter("mystery:x", progress.UNIT_STEPS).add(1)
    clk.t += progress.DEFAULT_STALL_WINDOW_S + 1
    diag = progress.self_diagnosis()
    assert diag is not None and diag.window_s == progress.DEFAULT_STALL_WINDOW_S


def test_rate_computation(monkeypatch):
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)
    c = progress.counter("download:r", progress.UNIT_BYTES)
    progress.snapshot()  # anchor
    c.add(100.0)
    clk.t += 10
    (s,) = progress.snapshot()
    assert s.rate_per_s == pytest.approx(10.0)


def test_activity_owned_counters_finish_at_end():
    act = activity.begin(activity.KIND_WARMUP, activity.PHASE_LOAD)
    act.counter("warmup:jobs", progress.UNIT_STEPS, total=3).set_done(1)
    assert progress.freshest() is not None
    act.completed()
    # ended activity finishes its counters: a reused name never carries a
    # stale age into the next activity
    assert progress.freshest() is None


def test_tracking_context_manager():
    with progress.tracking("upload:bytes", progress.UNIT_BYTES, total=9.0) as c:
        c.add(4.0)
        assert progress.freshest() is not None
    assert progress.freshest() is None


# ---------------------------------------------------------------------------
# beat emission
# ---------------------------------------------------------------------------


def test_on_beat_emits_counter_then_confession(monkeypatch):
    clk = _Clock()
    monkeypatch.setattr(progress, "_now", clk)
    sent: List[pb.WorkerMessage] = []

    async def _go() -> None:
        loop = asyncio.get_running_loop()

        async def emit(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        activity.bind_sink(emit, loop)
        act = activity.begin(activity.KIND_SELF_MINT_COMPILE, activity.PHASE_LOAD)
        ctr = act.counter("download:m", progress.UNIT_BYTES, total=50.0)
        ctr.set_done(20.0)
        activity.on_beat()  # advancing counter -> plain progress beat
        for _ in range(5):
            await asyncio.sleep(0)
        clk.t += progress.STALL_WINDOW_S["download"] + 1
        activity.on_beat()  # frozen past its window -> typed confession
        for _ in range(5):
            await asyncio.sleep(0)
        act.completed()
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(_go())
    beats = [u for u in _updates(sent) if u.counter == "download:m"]
    assert len(beats) == 2
    assert beats[0].counter_done == 20.0 and beats[0].counter_total == 50.0
    assert beats[0].counter_unit == progress.UNIT_BYTES
    assert not beats[0].self_stalled and beats[0].stalled_for_ms == 0
    assert beats[1].self_stalled
    assert beats[1].stalled_for_ms >= int(
        progress.STALL_WINDOW_S["download"] * 1000)
    # seq stays monotonic across phase reports and beats
    seqs = [u.seq for u in _updates(sent)]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_on_beat_noop_without_activity_or_counters():
    activity.on_beat()  # no activity: nothing to do, never raises
    act = activity.begin(activity.KIND_WARMUP, activity.PHASE_LOAD)
    activity.on_beat()  # activity but no counters: no beat
    act.completed()
    progress.counter("download:orphan", progress.UNIT_BYTES).add(1)
    activity.on_beat()  # counters but no activity: no beat, no crash


# ---------------------------------------------------------------------------
# end-to-end over the hub double (real Worker, real gRPC socket)
# ---------------------------------------------------------------------------


def test_hub_sees_cpu_quiet_download_then_confession(monkeypatch):
    from gen_worker import lifecycle
    from harness import progress_endpoints as pe

    monkeypatch.setattr(lifecycle, "HEARTBEAT_INTERVAL_MS", 250)
    monkeypatch.setitem(progress.STALL_WINDOW_S, "download", 0.8)

    with hub_double(modules=("harness.progress_endpoints",)) as (sched, _):
        conn = sched.wait_connection(0)

        # CPU-quiet but byte-advancing download is visibly alive: counter
        # beats with advancing done and no confession (the ie#522
        # false-positive class — reverting the beat wiring turns this red).
        adv = conn.wait_for(
            lambda m: m.WhichOneof("msg") == "activity_update"
            and m.activity_update.counter == "download:toy/model"
            and m.activity_update.counter_done > 0
            and not m.activity_update.self_stalled,
            timeout=10,
        ).activity_update
        assert adv.counter_total == pe.TOTAL_BYTES
        assert adv.counter_unit == progress.UNIT_BYTES

        # Frozen counter past its window: typed confession on the wire.
        conf = conn.wait_for(
            lambda m: m.WhichOneof("msg") == "activity_update"
            and m.activity_update.self_stalled,
            timeout=15,
        ).activity_update
        assert conf.counter == "download:toy/model"
        assert conf.stalled_for_ms >= 800

        # Local behavior unchanged (the hub owns the kill): setup finishes
        # after the freeze and the activity completes normally.
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "activity_update"
            and m.activity_update.state
            == pb.ActivityState.ACTIVITY_STATE_COMPLETED,
            timeout=20,
        )
        assert pe.SETUP_DONE.is_set()
