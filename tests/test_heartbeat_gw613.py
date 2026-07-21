"""gw#613/th#965 layer 2: universal app-level heartbeat.

Real Lifecycle + Executor over a fake transport: the beat is a force-sent,
byte-unchanged StateDelta emitted from the asyncio event loop in every state
— boot, idle, a stalled startup coroutine, and drain. ie#501 run 26: a
worker answered transport keepalive through 2.5h of app-level silence,
indistinguishable from a hung one; the beat is the app-level signal that
makes the distinction (and the reap) possible.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import gen_worker.lifecycle as lifecycle_mod
from gen_worker.executor import Executor, ModelStore
from gen_worker.lifecycle import HEARTBEAT_INTERVAL_MS, Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


class _FakeTransport:
    def __init__(self) -> None:
        self.connected = True
        self.sent: list[pb.WorkerMessage] = []
        self.queue = SimpleNamespace(pending_result_keys=set())

    async def send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)

    async def close_after_flush(self, timeout=None) -> None:
        self.connected = False

    def deltas(self) -> list[pb.StateDelta]:
        return [m.state_delta for m in self.sent
                if m.WhichOneof("msg") == "state_delta"]


def _lifecycle(tmp_path: Path) -> tuple[Lifecycle, _FakeTransport]:
    store = ModelStore(_noop_send, cache_dir=tmp_path)
    ex = Executor([], _noop_send, store=store)
    lc = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="w-beat",
                        runpod_pod_id="", worker_image_digest=""),
        ex,
    )
    transport = _FakeTransport()
    lc.transport = transport
    return lc, transport


def test_hello_declares_heartbeat_cadence(tmp_path: Path) -> None:
    lc, _ = _lifecycle(tmp_path)
    hello = lc.build_hello()
    assert hello.heartbeat_interval_ms == HEARTBEAT_INTERVAL_MS == 10_000


def test_disk_report_measured_on_ttl_not_per_beat(
    tmp_path: Path, monkeypatch
) -> None:
    """10s beats must not turn the statvfs/ref-index scan into a hot loop:
    the report rides every delta but is recomputed only past the TTL.

    boothang fix: the measurement (_measure_disk_usage_report, run off-loop
    via refresh_disk_usage_report) is gated by maybe_send_state_delta's
    _kick_disk_usage_refresh_if_stale, NOT by _state_delta() itself —
    _state_delta() only ever reads the cache (disk_usage_report()), so it
    is safe to call any number of times without measuring."""

    async def _go() -> None:
        lc, _ = _lifecycle(tmp_path)
        calls = 0
        real = lc.executor.store._measure_disk_usage_report

        def counting():
            nonlocal calls
            calls += 1
            return real()

        monkeypatch.setattr(
            lc.executor.store, "_measure_disk_usage_report", counting)
        await lc.maybe_send_state_delta()
        await lc._disk_report_refresh_task  # fire-and-forget: let it land
        first = lc._state_delta()
        await lc.maybe_send_state_delta()  # still within the TTL: no kick
        second = lc._state_delta()
        assert calls == 1
        assert (first.disk_usage.SerializeToString()
                == second.disk_usage.SerializeToString())
        # _state_delta() alone (no maybe_send_state_delta) never measures.
        lc._state_delta()
        lc._state_delta()
        assert calls == 1
        lc._disk_report_at = 0.0  # age past the TTL
        await lc.maybe_send_state_delta()
        await lc._disk_report_refresh_task
        assert calls == 2

    asyncio.run(_go())


def test_beat_force_sends_unchanged_delta(tmp_path: Path) -> None:
    async def _go() -> None:
        lc, transport = _lifecycle(tmp_path)
        await lc.maybe_send_state_delta()
        await lc.maybe_send_state_delta()  # unchanged -> edge-suppressed
        assert len(transport.deltas()) == 1
        await lc.maybe_send_state_delta(force=True)  # the beat tick
        assert len(transport.deltas()) == 2
        assert (transport.deltas()[0].SerializeToString(deterministic=True)
                == transport.deltas()[1].SerializeToString(deterministic=True))

    asyncio.run(_go())


def test_beats_flow_while_disk_measurement_is_stalled(
    tmp_path: Path, monkeypatch,
) -> None:
    """boothang: the real 0.40.7 LTX shape. A provider VOLUME mount stalled
    under statvfs() must never stop the heartbeat that is SUPPOSED to
    detect exactly this class of trouble — before the fix, the disk
    measurement ran synchronously and INLINE inside _state_delta(), so a
    stalled mount froze the beat task itself (th#965's own liveness signal
    silenced by the thing it was meant to catch)."""

    async def _go() -> None:
        import threading
        import time as _time

        monkeypatch.setattr(lifecycle_mod, "HEARTBEAT_INTERVAL_MS", 10)
        lc, transport = _lifecycle(tmp_path)
        stalled = threading.Event()  # set from the to_thread worker thread

        def _stalled_measure():
            # A synchronous sleep on the CALLING thread — if this ever ran
            # on the event loop thread it would freeze every beat, exactly
            # the pre-fix defect. asyncio.to_thread runs it off-loop.
            _time.sleep(0.3)
            stalled.set()
            return pb.DiskUsageReport()

        monkeypatch.setattr(
            lc.executor.store, "_measure_disk_usage_report", _stalled_measure)
        lc._heartbeat_task = asyncio.create_task(lc._heartbeat_loop())
        await asyncio.sleep(0.15)
        assert not stalled.is_set(), "measurement should still be running"
        beats = len(transport.deltas())
        assert beats >= 3, (
            f"only {beats} beats while disk measurement was stalled — the "
            "heartbeat must never wait on the statvfs() thread")
        for _ in range(100):
            if stalled.is_set():
                break
            await asyncio.sleep(0.05)
        assert stalled.is_set(), "the stalled measurement never completed"
        lc._heartbeat_task.cancel()
        await asyncio.gather(lc._heartbeat_task, return_exceptions=True)
        refresh_task = lc._disk_report_refresh_task
        if refresh_task is not None:
            await refresh_task

    asyncio.run(_go())


def test_beats_flow_while_a_startup_coroutine_is_stuck(
    tmp_path: Path, monkeypatch
) -> None:
    """The gw#612 shape: a boot coroutine parks forever on an await while the
    event loop stays healthy. The beat (same loop, separate task) MUST keep
    flowing — the hub's layer 3, not layer 2, owns this death mode."""

    async def _go() -> None:
        monkeypatch.setattr(lifecycle_mod, "HEARTBEAT_INTERVAL_MS", 10)
        lc, transport = _lifecycle(tmp_path)

        stuck = asyncio.Event()
        real_set_phase = lc.set_phase

        async def set_phase(phase):
            if phase == pb.WORKER_PHASE_LOADING_PIPELINES:
                await stuck.wait()  # never set: startup hangs here
            await real_set_phase(phase)

        monkeypatch.setattr(lc, "set_phase", set_phase)
        startup = asyncio.create_task(lc.startup())
        await asyncio.sleep(0.1)
        assert not startup.done()
        assert lc._heartbeat_task is not None
        beats = len(transport.deltas())
        assert beats >= 3, f"only {beats} beats while startup was stuck"
        startup.cancel()
        lc._heartbeat_task.cancel()
        await asyncio.gather(startup, lc._heartbeat_task,
                             return_exceptions=True)

    asyncio.run(_go())


def test_drain_keeps_beating_until_stream_closes(
    tmp_path: Path, monkeypatch
) -> None:
    async def _go() -> None:
        monkeypatch.setattr(lifecycle_mod, "HEARTBEAT_INTERVAL_MS", 10)
        lc, transport = _lifecycle(tmp_path)
        lc._heartbeat_task = asyncio.create_task(lc._heartbeat_loop())

        async def slow_idle(timeout=None) -> bool:
            await asyncio.sleep(0.08)
            return True

        monkeypatch.setattr(lc.executor, "wait_idle", slow_idle)
        before = len(transport.deltas())
        await lc.drain(0)
        during = len(transport.deltas())
        assert during > before, "heartbeat stopped during drain"
        assert lc.drained.is_set()
        assert lc._heartbeat_task is None  # cancelled after close
        await asyncio.sleep(0.05)
        assert len(transport.deltas()) == during, "beat outlived the drain"

    asyncio.run(_go())
