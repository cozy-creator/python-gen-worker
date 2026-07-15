from __future__ import annotations

import asyncio

from gen_worker.config import Settings
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transport import Transport


class _IdleExecutor(Executor):
    def __init__(self, idle_delay: float = 0.0) -> None:
        self.draining = False
        self.idle_delay = idle_delay
        self.wait_timeout: float | None = None
        self.shutdown_started = asyncio.Event()

    async def wait_idle(self, timeout: float | None = None) -> bool:
        self.wait_timeout = timeout
        await asyncio.sleep(self.idle_delay)
        return True

    async def abort_all(self, safe_message: str = "worker draining") -> None:
        raise AssertionError("idle executor must not be aborted")

    async def shutdown_instances(self) -> None:
        self.shutdown_started.set()


class _DrainLifecycle(Lifecycle):
    def __init__(self, executor: _IdleExecutor, transport: Transport) -> None:
        self.executor = executor
        self.transport = transport
        self.draining = False
        self.drained = asyncio.Event()

    async def maybe_send_state_delta(self) -> None:
        pass


class _RecordingTransport(Transport):
    def __init__(self) -> None:
        super().__init__(Settings(orchestrator_public_addr="localhost:1"), object())
        self.flush_timeout: float | None = None

    async def close_after_flush(self, timeout: float | None = None) -> bool:
        self.flush_timeout = timeout
        return await super().close_after_flush(timeout)


class _DisconnectHandlers:
    async def on_disconnect(self) -> None:
        pass


class _ReconnectTransport(Transport):
    def __init__(self) -> None:
        super().__init__(
            Settings(orchestrator_public_addr="localhost:1"),
            _DisconnectHandlers(),
            backoff_base_s=0.0,
            backoff_cap_s=0.0,
        )
        self.connect_attempts = 0

    async def _connect_once(self, target: str, use_tls: bool) -> None:
        self.connect_attempts += 1
        if self.connect_attempts == 1:
            raise ConnectionError("orchestrator unavailable")
        kind, result = await self.queue.get()
        assert kind == "result"
        await self.queue.mark_result_shipped(result)
        await asyncio.sleep(0)


def _result() -> pb.WorkerMessage:
    return pb.WorkerMessage(job_result=pb.JobResult(
        request_id="request-1", attempt=1, status=pb.JOB_STATUS_OK,
    ))


def test_unbounded_drain_waits_for_blocked_result_queue() -> None:
    async def _run() -> None:
        transport = Transport(Settings(orchestrator_public_addr="localhost:1"), object())
        executor = _IdleExecutor()
        lifecycle = _DrainLifecycle(executor, transport)
        await transport.send(_result())

        drain = asyncio.create_task(lifecycle.drain(0))
        await executor.shutdown_started.wait()
        await asyncio.sleep(0)

        assert not drain.done()
        assert not transport._stopping.is_set()
        kind, result = await transport.queue.get()
        assert kind == "result"
        await transport.queue.mark_result_shipped(result)

        await asyncio.wait_for(drain, 1.0)
        assert lifecycle.drained.is_set()
        assert transport._stopping.is_set()
        assert transport._clean_close

    asyncio.run(_run())


def test_explicit_drain_deadline_still_bounds_blocked_result_queue() -> None:
    async def _run() -> None:
        transport = _RecordingTransport()
        executor = _IdleExecutor(idle_delay=0.02)
        lifecycle = _DrainLifecycle(executor, transport)
        await transport.send(_result())

        await asyncio.wait_for(lifecycle.drain(100), 1.0)

        assert lifecycle.drained.is_set()
        assert transport._stopping.is_set()
        assert not transport._clean_close
        assert transport.queue.pending_result_keys == [("request-1", 1)]
        assert executor.wait_timeout is not None
        assert transport.flush_timeout is not None
        assert transport.flush_timeout < executor.wait_timeout - 0.01

    asyncio.run(_run())


def test_start_drain_stops_admission_and_anchors_deadline_synchronously() -> None:
    async def _run() -> None:
        transport = _RecordingTransport()
        executor = _IdleExecutor()
        lifecycle = _DrainLifecycle(executor, transport)
        loop = asyncio.get_running_loop()

        before = loop.time()
        lifecycle.start_drain(100)

        assert lifecycle.draining
        assert executor.draining
        assert lifecycle._drain_deadline_at is not None
        received_at = lifecycle._drain_deadline_at - 0.1
        assert before <= received_at <= loop.time()
        assert lifecycle._drain_task is not None
        await asyncio.wait_for(lifecycle._drain_task, 1.0)

    asyncio.run(_run())


def test_unbounded_drain_reconnects_until_result_ships() -> None:
    async def _run() -> None:
        transport = _ReconnectTransport()
        executor = _IdleExecutor()
        lifecycle = _DrainLifecycle(executor, transport)
        await transport.send(_result())

        drain = asyncio.create_task(lifecycle.drain(0))
        await executor.shutdown_started.wait()
        run = asyncio.create_task(transport.run())

        await asyncio.wait_for(asyncio.gather(drain, run), 3.0)
        assert transport.connect_attempts == 2
        assert lifecycle.drained.is_set()
        assert transport._clean_close
        assert transport.queue.pending_result_keys == []

    asyncio.run(_run())
