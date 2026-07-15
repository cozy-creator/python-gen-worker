"""#561: worker JWT rotation over the stream (TokenRefresh).

Transport-level contract: a hub-pushed TokenRefresh swaps the stored
credential in place — the live connection keeps running — and every
subsequent reconnect dials with the newest pushed token.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent import futures
from typing import Any, List, Optional

import grpc
import pytest

from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.transport import Transport

_TIMEOUT = 10.0


class _Conn:
    def __init__(self, authorization: str) -> None:
        self.authorization = authorization
        self.out: "queue.Queue[Optional[pb.SchedulerMessage]]" = queue.Queue()
        self.received: List[pb.WorkerMessage] = []

    def send(self, **oneof: Any) -> None:
        self.out.put(pb.SchedulerMessage(**oneof))

    def close(self) -> None:
        self.out.put(None)


class RotatingScheduler(pb_grpc.WorkerSchedulerServicer):
    """Records the Bearer token of every Connect; replies HelloAck."""

    def __init__(self) -> None:
        self.connections: List[_Conn] = []
        self._cond = threading.Condition()

    def Connect(self, request_iterator: Any, context: grpc.ServicerContext) -> Any:
        authz = dict(context.invocation_metadata()).get("authorization", "")
        first = next(request_iterator)
        assert first.WhichOneof("msg") == "hello"
        conn = _Conn(authz)
        conn.send(hello_ack=pb.HelloAck(protocol_version=pb.PROTOCOL_VERSION_CURRENT))
        with self._cond:
            self.connections.append(conn)
            self._cond.notify_all()

        def _reader() -> None:
            try:
                for msg in request_iterator:
                    conn.received.append(msg)
            except Exception:
                pass
            finally:
                conn.out.put(None)

        threading.Thread(target=_reader, daemon=True).start()
        while True:
            item = conn.out.get()
            if item is None:
                return
            yield item

    def wait_connection(self, index: int, timeout: float = _TIMEOUT) -> _Conn:
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self.connections) <= index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"connection #{index} never arrived")
                self._cond.wait(remaining)
            return self.connections[index]


class _Handlers:
    def __init__(self) -> None:
        self.hello_acks = 0
        self.messages: List[pb.SchedulerMessage] = []

    def build_hello(self) -> pb.Hello:
        return pb.Hello(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            worker_id="rotation-worker",
        )

    async def on_hello_ack(self, ack: pb.HelloAck) -> None:
        self.hello_acks += 1

    async def on_message(self, msg: pb.SchedulerMessage) -> None:
        self.messages.append(msg)

    async def on_disconnect(self) -> None:
        pass


@pytest.fixture
def rotating_scheduler():
    scheduler = RotatingScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield scheduler, port
    finally:
        server.stop(grace=0)


def test_token_refresh_swaps_credential_midstream_and_on_reconnect(rotating_scheduler) -> None:
    scheduler, port = rotating_scheduler

    async def scenario() -> None:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="rotation-worker",
            worker_jwt="boot-token",
        )
        handlers = _Handlers()
        transport = Transport(settings, handlers, backoff_base_s=0.05, backoff_cap_s=0.2)
        run_task = asyncio.create_task(transport.run())
        loop = asyncio.get_running_loop()
        try:
            conn1 = await loop.run_in_executor(None, scheduler.wait_connection, 0)
            assert conn1.authorization == "Bearer boot-token"
            assert await transport.wait_connected(_TIMEOUT)

            # Hub pushes a rotated token mid-stream.
            conn1.send(token_refresh=pb.TokenRefresh(token="rotated-token", expires_at_unix=4242))
            deadline = time.monotonic() + _TIMEOUT
            while transport._worker_jwt != "rotated-token":
                assert time.monotonic() < deadline, "TokenRefresh never applied"
                await asyncio.sleep(0.01)

            # Live connection unaffected: still connected, sends still flow,
            # and the refresh was consumed by the transport (not the handlers).
            assert transport.connected
            await transport.send(pb.WorkerMessage(state_delta=pb.StateDelta()))
            deadline = time.monotonic() + _TIMEOUT
            while not conn1.received:
                assert time.monotonic() < deadline, "send after refresh never arrived"
                await asyncio.sleep(0.01)
            assert not handlers.messages, "token_refresh leaked to on_message"

            # Kill the stream: the reconnect must dial with the rotated token.
            conn1.close()
            conn2 = await loop.run_in_executor(None, scheduler.wait_connection, 1)
            assert conn2.authorization == "Bearer rotated-token"

            # A second rotation supersedes the first on the next reconnect.
            conn2.send(token_refresh=pb.TokenRefresh(token="rotated-token-2", expires_at_unix=9999))
            deadline = time.monotonic() + _TIMEOUT
            while transport._worker_jwt != "rotated-token-2":
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            conn2.close()
            conn3 = await loop.run_in_executor(None, scheduler.wait_connection, 2)
            assert conn3.authorization == "Bearer rotated-token-2"
        finally:
            transport.stop()
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), _TIMEOUT)

    asyncio.run(asyncio.wait_for(scenario(), 4 * _TIMEOUT))


def test_empty_token_refresh_is_ignored(rotating_scheduler) -> None:
    scheduler, port = rotating_scheduler

    async def scenario() -> None:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="rotation-worker",
            worker_jwt="boot-token",
        )
        transport = Transport(settings, _Handlers(), backoff_base_s=0.05, backoff_cap_s=0.2)
        run_task = asyncio.create_task(transport.run())
        loop = asyncio.get_running_loop()
        try:
            conn1 = await loop.run_in_executor(None, scheduler.wait_connection, 0)
            assert await transport.wait_connected(_TIMEOUT)
            conn1.send(token_refresh=pb.TokenRefresh(token="", expires_at_unix=1))
            await asyncio.sleep(0.2)
            assert transport._worker_jwt is None
            conn1.close()
            conn2 = await loop.run_in_executor(None, scheduler.wait_connection, 1)
            assert conn2.authorization == "Bearer boot-token"
        finally:
            transport.stop()
            await asyncio.wait_for(asyncio.gather(run_task, return_exceptions=True), _TIMEOUT)

    asyncio.run(asyncio.wait_for(scenario(), 4 * _TIMEOUT))
