"""gw#613/th#965 layer 2: universal app-level heartbeat.

Real Lifecycle + Executor over a fake transport: the beat is a force-sent,
byte-unchanged StateDelta emitted from the asyncio event loop in every state
— boot, idle, a stalled startup coroutine, and drain. ie#501 run 26: a hung
worker answered transport keepalive for 2.5h; the beat is the app-level
signal the hub reaps on instead.
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
    assert hello.heartbeat_interval_ms == HEARTBEAT_INTERVAL_MS == 30_000


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
