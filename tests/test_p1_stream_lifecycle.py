"""P1 (th#960/pgw#609 design table): hello/hello_ack/run_job/accepted/result
over a real TCP gRPC socket; exactly one result per (request_id, attempt)
across reconnects; drain = finish in-flight + close; protocol mismatch =
FAILED_PRECONDITION exits.

Absorbed from ``tests/test_worker_grpc_e2e.py`` (#365), consolidated onto
``tests/harness/hub_double.py`` per the design's extraction plan. The
pgw#605 idle-heartbeat row is a documented skip: the proto fields
(``Hello.idle_heartbeat_interval_ms`` / ``WorkerMessage.heartbeat``) do not
exist in this tree yet (pgw#605 is unimplemented) — nothing real to assert
against until they land.
"""

from __future__ import annotations

import time

import grpc
import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import hub_double, is_accept_for, is_ready, is_result_for
from harness.toy_endpoints import EchoIn, EchoOut


def _msgpack(text: str) -> bytes:
    return msgspec.msgpack.encode(EchoIn(text=text))


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


def test_hello_carries_protocol_and_worker_identity() -> None:
    with hub_double(worker_id="p1-worker") as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        assert conn.hello is not None
        assert conn.hello.protocol_version == pb.PROTOCOL_VERSION_CURRENT
        assert conn.hello.worker_id == "p1-worker"


@pytest.mark.parametrize(
    "text,expect_status",
    [
        ("marco", pb.JOB_STATUS_OK),
        ("not-marco", pb.JOB_STATUS_INVALID),
    ],
)
def test_dispatch_accepted_then_result(text: str, expect_status: "pb.JobStatus") -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r1", attempt=1, function_name="echo",
            input_payload=_msgpack(text)))
        conn.wait_for(is_accept_for("r1"))
        res = conn.wait_for(is_result_for("r1")).job_result
        assert res.status == expect_status
        assert res.attempt == 1
        if expect_status == pb.JOB_STATUS_OK:
            assert _decode(res.inline).response == "polo"


def test_unknown_function_is_invalid_no_accept_race() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-unknown", attempt=1, function_name="nope",
            input_payload=_msgpack("x")))
        res = conn.wait_for(is_result_for("r-unknown")).job_result
        assert res.status == pb.JOB_STATUS_INVALID


def test_retransmitted_live_attempt_re_acks_never_re_executes() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-echo", attempt=1, function_name="echo",
            input_payload=_msgpack("marco")))
        conn.wait_for(is_result_for("r-echo"))
        conn.send(run_job=pb.RunJob(
            request_id="r-echo", attempt=1, function_name="echo",
            input_payload=_msgpack("marco")))
        time.sleep(0.3)
        assert conn.count(is_result_for("r-echo")) == 1


def test_kill_mid_job_reconcile_ships_result_exactly_once() -> None:
    """Stream-kill mid-job: the reconnected Hello carries the in-flight
    attempt, and the buffered result ships exactly once across the whole
    connection history — never zero, never twice."""
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-mid", attempt=2, function_name="sleepy",
            input_payload=_msgpack("marco")))
        conn.wait_for(is_accept_for("r-mid"))
        conn.kill()
        conn2 = scheduler.wait_connection(1)
        assert conn2.hello is not None
        in_flight = {(j.request_id, j.attempt) for j in conn2.hello.in_flight}
        assert ("r-mid", 2) in in_flight
        res = conn2.wait_for(is_result_for("r-mid")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert res.attempt == 2
        time.sleep(0.3)
        total = sum(c.count(is_result_for("r-mid")) for c in scheduler.connections)
        assert total == 1, "buffered result must ship exactly once"


def test_drain_finishes_in_flight_then_closes_and_rejects_new_work() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-last", attempt=1, function_name="sleepy",
            input_payload=_msgpack("marco")))
        conn.wait_for(is_accept_for("r-last"))
        conn.send(drain=pb.Drain(deadline_ms=5000))
        conn.send(run_job=pb.RunJob(
            request_id="r-after-drain", attempt=1, function_name="echo",
            input_payload=_msgpack("marco")))
        rejected = conn.wait_for(is_result_for("r-after-drain")).job_result
        assert rejected.status == pb.JOB_STATUS_RETRYABLE
        assert "draining" in rejected.safe_message
        assert conn.count(is_accept_for("r-after-drain")) == 0
        finished = conn.wait_for(is_result_for("r-last")).job_result
        assert finished.status == pb.JOB_STATUS_OK
        assert conn.client_done.wait(15.0), "worker must close the stream after drain"


def test_cancel_mid_job_is_cooperative_abort() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-cancel", attempt=1, function_name="slow",
            input_payload=_msgpack("marco")))
        conn.wait_for(is_accept_for("r-cancel"))
        conn.send(cancel_job=pb.CancelJob(request_id="r-cancel", attempt=1))
        res = conn.wait_for(is_result_for("r-cancel")).job_result
        assert res.status == pb.JOB_STATUS_CANCELED


def test_deadline_marks_fatal_and_frees_the_worker() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-deadline", attempt=1, function_name="slow",
            input_payload=_msgpack("marco"), timeout_ms=300))
        res = conn.wait_for(is_result_for("r-deadline")).job_result
        assert res.status == pb.JOB_STATUS_FATAL
        assert res.safe_message == "deadline exceeded"
        conn.send(run_job=pb.RunJob(
            request_id="r-next", attempt=1, function_name="echo",
            input_payload=_msgpack("marco")))
        res = conn.wait_for(is_result_for("r-next")).job_result
        assert res.status == pb.JOB_STATUS_OK


def test_protocol_mismatch_is_failed_precondition() -> None:
    """A hub speaking an incompatible protocol version aborts the handshake
    with FAILED_PRECONDITION rather than a generic error — worker treats it
    as a permanent-precondition class and exits fast for reap."""

    class _Mismatch:
        def Connect(self, request_iterator, context):  # noqa: N802
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "protocol_version_mismatch")

    from concurrent import futures

    import grpc as grpc_mod

    from gen_worker.config import load_settings
    from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
    from gen_worker.worker import Worker

    server = grpc_mod.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(_Mismatch(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}", worker_id="p1-mismatch", worker_jwt="",
        )
        worker = Worker(settings, ["harness.toy_endpoints"], backoff_base_s=0.05)
        exit_code = worker.run()
        assert exit_code == 1
    finally:
        server.stop(grace=0)


@pytest.mark.skip(
    reason="pgw#605 open: Hello.idle_heartbeat_interval_ms / WorkerMessage.heartbeat "
           "proto fields are not generated in this tree yet (verified: hasattr checks "
           "fail on worker_scheduler_pb2). Add the proto fields first (mirrored "
           "byte-identical from tensorhub per the pgw#605 tracker task list), then "
           "flip this to a real outbound-silence-produces-heartbeats assertion."
)
def test_outbound_silence_produces_periodic_heartbeats() -> None:
    raise AssertionError("unreachable: proto fields for pgw#605 do not exist yet")
