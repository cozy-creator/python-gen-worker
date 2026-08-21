from __future__ import annotations

import threading
from concurrent import futures
from contextlib import contextmanager
from typing import Iterator, List, Tuple, Optional

import grpc

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc

DEFAULT_TIMEOUT_S = 15.0


class RecordingHardwareReportServicer(pb_grpc.WorkerSchedulerServicer):

    def __init__(self) -> None:
        self.received: List[pb.WorkerMessage] = []
        self._cond = threading.Condition()

    def Connect(self, request_iterator, context: grpc.ServicerContext):  # noqa: N802
        first = next(request_iterator)
        with self._cond:
            self.received.append(first)
            self._cond.notify_all()
        for _ in request_iterator:
            pass
        return iter(())

    def wait_for_message(self, timeout: Optional[float] = None) -> pb.WorkerMessage:
        """Wait for the worker's first message."""
        import time

        from .progress_wait import Cadence, StalledError

        cadence = Cadence()
        deadline = None if timeout is None else time.monotonic() + timeout
        started = time.monotonic()
        with self._cond:
            while not self.received:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"no WorkerMessage received within {timeout}s"
                        )
                    self._cond.wait(remaining)
                    continue
                silent = time.monotonic() - started
                if silent >= cadence.window_s:
                    raise StalledError(
                        f"no WorkerMessage in {silent:.1f}s of silence "
                        f"(staleness window {cadence.describe()})"
                    )
                self._cond.wait(min(0.25, cadence.window_s - silent))
            return self.received[0]


class OldHubServicer(pb_grpc.WorkerSchedulerServicer):

    def Connect(self, request_iterator, context: grpc.ServicerContext):  # noqa: N802
        first = next(request_iterator)
        if first.WhichOneof("msg") != "hello":
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "protocol_violation: first message must be Hello")
        yield pb.SchedulerMessage(hello_ack=pb.HelloAck(protocol_version=pb.PROTOCOL_VERSION_CURRENT))


@contextmanager
def recording_hub() -> Iterator[Tuple[RecordingHardwareReportServicer, str]]:
    """A real (accepting) hub-double: ``(servicer, "127.0.0.1:<port>")``."""
    servicer = RecordingHardwareReportServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield servicer, f"127.0.0.1:{port}"
    finally:
        server.stop(grace=0)


@contextmanager
def old_hub() -> Iterator[str]:
    """A pre-gw#619 hub that rejects anything but Hello: ``"127.0.0.1:<port>"``."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(OldHubServicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(grace=0)


def closed_port_addr() -> str:
    """A ``127.0.0.1:<port>`` guaranteed nothing is listening on (bind, then close) — a fast, deterministic ECONNREFUSED, distinct from an unroutable blackhole address's slower timeout-shaped failure."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"127.0.0.1:{port}"
