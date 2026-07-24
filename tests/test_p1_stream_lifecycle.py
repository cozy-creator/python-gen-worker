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

import threading
import time

import grpc
import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.hub_double import (
    FakeScheduler,
    custom_scheduler_server,
    hub_double,
    is_accept_for,
    is_ready,
    is_result_for,
)
from harness.toy_endpoints import EchoIn, EchoOut

_TIMEOUT = 15.0


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
        assert conn.hello.worker_session_id
        assert conn.hello.lifecycle_snapshot.full_replace
        assert conn.hello.lifecycle_snapshot.worker_session_id == conn.hello.worker_session_id


def test_unknown_mandatory_intent_rejects_without_legacy_fallback() -> None:
    with hub_double(worker_id="p1-shadow-reject") as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        assert conn.hello is not None
        session_id = conn.hello.worker_session_id
        conn.send(
            hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                file_base_url=scheduler.file_base_url,
                desired_residency=pb.DesiredResidency(
                    generation=77,
                    release_id="release-shadow",
                ),
                desired_state_command=pb.DesiredStateCommand(
                    worker_session_id=session_id,
                    command_seq=77,
                    goal_id="goal-unknown",
                    release_id="release-shadow",
                    mandatory=True,
                    intents=[
                        pb.DesiredIntent(
                            intent_id="intent-unknown",
                            kind=999,
                            cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION,
                            mandatory=True,
                        )
                    ],
                ),
            )
        )
        receipt = conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "goal_receipt" and m.goal_receipt.goal_id == "goal-unknown"
            )
        ).goal_receipt
        assert receipt.status == pb.GOAL_RECEIPT_STATUS_REJECTED
        assert receipt.error_code == pb.LIFECYCLE_ERROR_CODE_UNSUPPORTED_INTENT
        assert receipt.rejections[0].intent_id == "intent-unknown"

        conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "lifecycle_snapshot"
                and any(
                    item.goal_id == "goal-unknown"
                    and item.status == pb.GOAL_RECEIPT_STATUS_REJECTED
                    for item in m.lifecycle_snapshot.goal_receipts
                )
            )
        )
        conn.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "state_delta"
                and m.state_delta.phase == pb.WORKER_PHASE_ERROR
            )
        )

        conn.send(
            run_job=pb.RunJob(
                request_id="r-after-protocol-reject",
                attempt=1,
                function_name="echo",
                input_payload=_msgpack("marco"),
            )
        )
        result = conn.wait_for(is_result_for("r-after-protocol-reject")).job_result
        assert result.status == pb.JOB_STATUS_RETRYABLE
        assert conn.count(is_accept_for("r-after-protocol-reject")) == 0


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


# ---------------------------------------------------------------------------
# th#960/pgw#609 Phase 2b: auth-rejection/precondition/redirect family,
# absorbed from test_worker_grpc_e2e.py (#372) before that file's deletion —
# the worker's handshake-failure taxonomy (transient vs permanent, redirect
# vs exit) is squarely P1's boundary.
# ---------------------------------------------------------------------------


def test_auth_rejection_exits_instead_of_spinning(monkeypatch) -> None:
    # UNAUTHENTICATED can be transient hub-side; the fatal exit is gated on
    # BOTH a failure count and an elapsed window (#372). Shrink the window
    # so the test observes the exit quickly.
    import gen_worker.transport as transport_mod

    monkeypatch.setattr(transport_mod, "_AUTH_FAILURE_EXIT_WINDOW_S", 0.3)
    with custom_scheduler_server(
        lambda: FakeScheduler(reject_unauthenticated=True),
    ) as (_scheduler, harness, _port):
        assert harness.join(timeout=_TIMEOUT) == 1


def test_auth_rejection_within_window_keeps_retrying() -> None:
    """3 quick UNAUTHENTICATED strikes must NOT kill the worker while the
    exit window has not elapsed — a hub pg blip is survivable (#372)."""
    with custom_scheduler_server(
        lambda: FakeScheduler(reject_unauthenticated=True),
    ) as (_scheduler, harness, _port):
        deadline = time.monotonic() + 3.0
        while len(harness.worker.transport.reconnect_delays) < 4:
            assert harness._thread.is_alive(), "worker exited inside the auth window"
            assert time.monotonic() < deadline, "worker never retried"
            time.sleep(0.02)
        assert harness.exit_code is None
        harness.stop()


def test_permanent_precondition_exits_fast() -> None:
    """worker_id_mismatch cannot heal by retrying: exit for reap immediately
    instead of retrying forever (#372)."""

    class _Mismatch(FakeScheduler):
        def Connect(self, request_iterator, context):  # noqa: N802
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "worker_id_mismatch: hello=w1 jwt_sub=w2")

    with custom_scheduler_server(_Mismatch) as (_scheduler, harness, _port):
        assert harness.join(timeout=_TIMEOUT) == 1


def test_hello_ack_deadline_reconnects_instead_of_hanging(monkeypatch) -> None:
    """A hub that accepts the stream but never sends HelloAck must not hang
    the worker forever (#372): the handshake deadline fires and the worker
    reconnects with backoff."""
    import gen_worker.transport as transport_mod

    monkeypatch.setattr(transport_mod, "_HELLO_ACK_TIMEOUT_S", 0.3)
    stalls = {"n": 0}

    class _Stall(FakeScheduler):
        def Connect(self, request_iterator, context):  # noqa: N802
            stalls["n"] += 1
            if stalls["n"] == 1:
                next(request_iterator)  # read Hello, then say nothing
                time.sleep(5.0)         # stall well past the deadline
                return iter(())
            return super().Connect(request_iterator, context)

    with custom_scheduler_server(_Stall) as (scheduler, _harness, _port):
        conn = scheduler.wait_connection(0)
        assert conn.hello is not None
        assert stalls["n"] >= 2


def test_not_leader_redirect_is_followed() -> None:
    """FAILED_PRECONDITION not_leader:<addr> redirects the worker to the
    leader immediately; schemeless targets keep the dialing TLS mode (#372,
    plaintext here on both ends)."""
    with custom_scheduler_server(FakeScheduler) as (real, _real_harness, real_port):
        class _NotLeader(FakeScheduler):
            def Connect(self, request_iterator, context):  # noqa: N802
                context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                              f"not_leader:127.0.0.1:{real_port}")

        with custom_scheduler_server(_NotLeader) as (_stale, _stale_harness, _port):
            conn = real.wait_connection(0)
            assert conn.hello is not None and conn.hello.worker_id == "hub-double-worker"


def test_worker_survives_hub_restart_and_reconnects(tmp_path, monkeypatch) -> None:
    """The hub process dying mid-connection must not kill the worker: it
    keeps reconnecting with backoff, and picks up a REPLACEMENT server bound
    to the same address once one appears (#372)."""
    from concurrent import futures

    from gen_worker.config import load_settings
    from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
    from gen_worker.worker import Worker

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "cache"))
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    settings = load_settings(
        orchestrator_public_addr=f"127.0.0.1:{port}", worker_id="p1-restart", worker_jwt="",
    )
    worker = Worker(settings, ["harness.toy_endpoints"], backoff_base_s=0.05, backoff_cap_s=0.2)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    replacement = None
    try:
        scheduler.wait_connection(0)
        before = len(worker.transport.reconnect_delays)
        assert server.stop(grace=0).wait(_TIMEOUT)

        deadline = time.monotonic() + _TIMEOUT
        while len(worker.transport.reconnect_delays) < before + 4:
            assert thread.is_alive(), "worker exited while hub was down"
            assert time.monotonic() < deadline, "worker did not keep reconnecting"
            time.sleep(0.02)

        replacement_scheduler = FakeScheduler()
        replacement = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_WorkerSchedulerServicer_to_server(replacement_scheduler, replacement)
        assert replacement.add_insecure_port(f"127.0.0.1:{port}") == port
        replacement.start()

        assert replacement_scheduler.wait_connection(0).hello is not None
    finally:
        worker.stop()
        thread.join(_TIMEOUT)
        server.stop(grace=0)
        if replacement is not None:
            replacement.stop(grace=0)


@pytest.mark.skip(
    reason="pgw#605 open: Hello.idle_heartbeat_interval_ms / WorkerMessage.heartbeat "
           "proto fields are not generated in this tree yet (verified: hasattr checks "
           "fail on worker_scheduler_pb2). Add the proto fields first (mirrored "
           "byte-identical from tensorhub per the pgw#605 tracker task list), then "
           "flip this to a real outbound-silence-produces-heartbeats assertion."
)
def test_outbound_silence_produces_periodic_heartbeats() -> None:
    raise AssertionError("unreachable: proto fields for pgw#605 do not exist yet")
