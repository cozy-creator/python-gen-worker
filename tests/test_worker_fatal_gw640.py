"""gw#640/th#1077: a worker fatal must reach the HUB, not just pod stdout.

RunPod exposes no container-logs API, so a fatal written only to stdout is
unreachable — six live th#1085 runs died on a crash whose traceback existed
and could not be read. These tests pin the two halves of the fix:

1. the fatal report is dialed over a real gRPC Connect stream and carries the
   exception class, message and traceback;
2. the run loop ending WITHOUT a Drain or a signal is itself a fatal (it used
   to be a clean, silent exit 0 — the exact gw#640 signature).
"""

from __future__ import annotations

import threading
from concurrent import futures

import grpc
import pytest

from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.worker_fatal import REASON_CLASS, build_fatal_detail, report_worker_fatal


class _FatalCatcher(pb_grpc.WorkerSchedulerServicer):
    def __init__(self) -> None:
        self.reports: list = []
        self.got = threading.Event()

    def Connect(self, request_iterator, context):
        for msg in request_iterator:
            if msg.WhichOneof("msg") == "hardware_unsuitable":
                self.reports.append(msg.hardware_unsuitable)
                self.got.set()
        return
        yield  # pragma: no cover - generator marker


def _server():
    catcher = _FatalCatcher()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(catcher, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return catcher, server, port


def test_fatal_detail_carries_class_message_and_traceback() -> None:
    try:
        raise ValueError("pipeline exploded")
    except ValueError as exc:
        detail = build_fatal_detail("runtime", exc, exit_code=1)
    assert "phase=runtime" in detail
    assert "exit_code=1" in detail
    assert "ValueError: pipeline exploded" in detail
    assert "Traceback (most recent call last)" in detail
    assert "test_worker_fatal_gw640.py" in detail


def test_fatal_detail_is_clipped_but_keeps_both_ends() -> None:
    exc = RuntimeError("H" * 200 + "M" * 40_000 + "T" * 200)
    detail = build_fatal_detail("runtime", exc, exit_code=1)
    assert len(detail) < 12_000
    assert "HHH" in detail and "TTT" in detail
    assert "[clipped]" in detail


def test_worker_fatal_reaches_the_hub_over_the_wire() -> None:
    catcher, server, port = _server()
    try:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="worker-gw640",
            worker_jwt="",
        )
        try:
            raise RuntimeError("boom in reconcile")
        except RuntimeError as exc:
            delivered = report_worker_fatal(settings, "runtime", exc, exit_code=1)
        assert catcher.got.wait(10), "hub never received a fatal report"
        report = catcher.reports[0]
        assert report.reason_class == REASON_CLASS
        assert "RuntimeError: boom in reconcile" in report.detail
        assert "Traceback" in report.detail
        assert report.worker_id == "worker-gw640"
        assert delivered is True
    finally:
        server.stop(grace=0)


def test_report_is_a_noop_without_an_orchestrator_address() -> None:
    settings = load_settings(orchestrator_public_addr="", worker_id="w")
    assert report_worker_fatal(settings, "runtime", RuntimeError("x"), exit_code=1) is False


def test_unexplained_loop_exit_is_a_fatal_not_a_silent_zero() -> None:
    """gw#640's signature: transport.run() returning with no Drain and no
    signal used to be `return 0` — a silent restart the hub could only see as
    a young-worker death."""
    import asyncio

    from gen_worker.worker import UnexpectedWorkerExit, Worker

    worker = Worker.__new__(Worker)  # no real wiring needed for this contract
    worker._stop_requested = False

    class _Lifecycle:
        drained = threading.Event()
        draining = False

        def start_drain(self, deadline_ms):
            pass

        async def startup(self):
            await asyncio.sleep(3600)

    class _Transport:
        connected = False

        async def run(self):
            return  # loop ends on its own — the gw#640 shape

    worker.lifecycle = _Lifecycle()
    worker.transport = _Transport()

    with pytest.raises(UnexpectedWorkerExit) as caught:
        asyncio.run(worker.arun())
    assert "without a Drain command" in str(caught.value)


def test_requested_stop_still_exits_zero() -> None:
    import asyncio

    from gen_worker.worker import Worker

    worker = Worker.__new__(Worker)
    worker._stop_requested = True

    class _Lifecycle:
        drained = threading.Event()
        draining = False

        def start_drain(self, deadline_ms):
            pass

        async def startup(self):
            await asyncio.sleep(3600)

    class _Transport:
        connected = False

        async def run(self):
            return

    worker.lifecycle = _Lifecycle()
    worker.transport = _Transport()
    assert asyncio.run(worker.arun()) == 0
