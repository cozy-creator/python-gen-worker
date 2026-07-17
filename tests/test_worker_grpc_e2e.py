"""Worker <-> scheduler e2e over a REAL gRPC socket (#365).

An in-process ``grpc.server`` plays the orchestrator (`FakeScheduler`) and
drives the REAL Worker (transport + lifecycle + executor + registry) through
the full contract:

  connect -> Hello/HelloAck -> dispatch -> progress deltas -> result ->
  cancel -> stream-kill -> reconnect-with-backoff -> kill mid-job ->
  in_flight reconcile ships the buffered result exactly once -> drain.

Plus: deadline enforcement, GPU-semaphore serialization, auth-failure exit,
and the send-queue results-never-dropped policy.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent import futures
from typing import Any, Callable, List, Optional

import grpc
import msgspec
import pytest

from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.transport import SendQueue
from gen_worker.worker import Worker

from e2e_endpoints import EchoIn, EchoOut

_TIMEOUT = 15.0


def _msgpack(text: str) -> bytes:
    return msgspec.msgpack.encode(EchoIn(text=text))


def _decode_out(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


class _Conn:
    """One live worker connection as seen by the fake scheduler."""

    def __init__(self) -> None:
        self.hello: Optional[pb.Hello] = None
        self.received: List[pb.WorkerMessage] = []
        self._recv_cond = threading.Condition()
        self._out: "queue.Queue[Any]" = queue.Queue()
        self.client_done = threading.Event()

    def send(self, **oneof: Any) -> None:
        self._out.put(pb.SchedulerMessage(**oneof))

    def kill(self) -> None:
        """Abruptly fail the stream (server-side error)."""
        self._out.put(RuntimeError("killed"))

    def close(self) -> None:
        """End the response stream cleanly."""
        self._out.put(None)

    def _record(self, msg: pb.WorkerMessage) -> None:
        with self._recv_cond:
            self.received.append(msg)
            self._recv_cond.notify_all()

    def wait_for(
        self, pred: Callable[[pb.WorkerMessage], bool], timeout: float = _TIMEOUT
    ) -> pb.WorkerMessage:
        deadline = time.monotonic() + timeout
        with self._recv_cond:
            checked = 0
            while True:
                for msg in self.received[checked:]:
                    checked += 1
                    if pred(msg):
                        return msg
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    def _label(m: pb.WorkerMessage) -> str:
                        which = m.WhichOneof("msg")
                        if which == "job_result":
                            return f"job_result({m.job_result.request_id})"
                        if which == "job_accepted":
                            return f"job_accepted({m.job_accepted.request_id})"
                        return str(which)
                    raise TimeoutError(
                        f"no matching message within {timeout}s; got "
                        f"{[_label(m) for m in self.received]}"
                    )
                self._recv_cond.wait(remaining)

    def count(self, pred: Callable[[pb.WorkerMessage], bool]) -> int:
        with self._recv_cond:
            return sum(1 for m in self.received if pred(m))


class FakeScheduler(pb_grpc.WorkerSchedulerServicer):
    def __init__(self, *, reject_unauthenticated: bool = False) -> None:
        self.connections: List[_Conn] = []
        self._conn_cond = threading.Condition()
        self.reject_unauthenticated = reject_unauthenticated
        self.file_base_url = "http://127.0.0.1:1/files"

    def Connect(self, request_iterator: Any, context: grpc.ServicerContext) -> Any:
        if self.reject_unauthenticated:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "bad worker jwt")

        first = next(request_iterator)
        assert first.WhichOneof("msg") == "hello", "first message must be Hello"
        conn = _Conn()
        conn.hello = first.hello
        # Queue the HelloAck BEFORE exposing the connection: the contract says
        # HelloAck precedes all other scheduler->worker traffic.
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            file_base_url=self.file_base_url,
        ))
        with self._conn_cond:
            self.connections.append(conn)
            self._conn_cond.notify_all()

        def _reader() -> None:
            try:
                for msg in request_iterator:
                    conn._record(msg)
            except Exception:
                pass
            finally:
                conn.client_done.set()
                conn._out.put(None)  # end the response stream too

        threading.Thread(target=_reader, daemon=True).start()
        while True:
            item = conn._out.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def wait_connection(self, index: int, timeout: float = _TIMEOUT) -> _Conn:
        deadline = time.monotonic() + timeout
        with self._conn_cond:
            while len(self.connections) <= index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"connection #{index} never arrived "
                        f"({len(self.connections)} so far)"
                    )
                self._conn_cond.wait(remaining)
            return self.connections[index]


class _Harness:
    def __init__(self, scheduler: FakeScheduler, port: int) -> None:
        self.scheduler = scheduler
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="e2e-worker",
            worker_jwt="",
        )
        self.worker = Worker(
            settings,
            ["e2e_endpoints"],
            gpu_slots=1,
            backoff_base_s=0.05,
            backoff_cap_s=0.2,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.exit_code = self.worker.run()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = _TIMEOUT) -> Optional[int]:
        self.worker.stop()
        self._thread.join(timeout)
        return self.exit_code

    def join(self, timeout: float = _TIMEOUT) -> Optional[int]:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "worker did not exit"
        return self.exit_code


@pytest.fixture
def scheduler_and_worker():
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    harness = _Harness(scheduler, port)
    harness.start()
    try:
        yield scheduler, harness
    finally:
        harness.stop()
        server.stop(grace=0)


def _is_result_for(rid: str):
    return lambda m: m.WhichOneof("msg") == "job_result" and m.job_result.request_id == rid


def _is_accept_for(rid: str):
    return lambda m: m.WhichOneof("msg") == "job_accepted" and m.job_accepted.request_id == rid


# ---------------------------------------------------------------------------


def test_full_contract_round_trip(scheduler_and_worker) -> None:
    scheduler, harness = scheduler_and_worker
    conn = scheduler.wait_connection(0)

    # ---- Hello -------------------------------------------------------------
    assert conn.hello is not None
    assert conn.hello.protocol_version == pb.PROTOCOL_VERSION_CURRENT
    assert conn.hello.worker_id == "e2e-worker"

    # ---- StateDelta advertises the functions once READY ----------------------
    ready = conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.phase == pb.WORKER_PHASE_READY
        and "echo" in m.state_delta.available_functions
    )
    assert set(ready.state_delta.available_functions) >= {"echo", "stream3", "slow", "sleepy"}

    # ---- dispatch -> JobAccepted -> JobResult(OK, inline) --------------------
    conn.send(run_job=pb.RunJob(
        request_id="r-echo", attempt=1, function_name="echo",
        input_payload=_msgpack("marco")))
    conn.wait_for(_is_accept_for("r-echo"))
    res = conn.wait_for(_is_result_for("r-echo")).job_result
    assert res.status == pb.JOB_STATUS_OK
    assert res.attempt == 1
    assert _decode_out(res.inline).response == "polo"
    assert res.metrics.runtime_ms >= 0

    # ---- invalid input -> INVALID (no retry) ---------------------------------
    conn.send(run_job=pb.RunJob(
        request_id="r-bad", attempt=1, function_name="echo",
        input_payload=_msgpack("not-marco")))
    res = conn.wait_for(_is_result_for("r-bad")).job_result
    assert res.status == pb.JOB_STATUS_INVALID

    # ---- unknown function -> INVALID -----------------------------------------
    conn.send(run_job=pb.RunJob(
        request_id="r-unknown", attempt=1, function_name="nope",
        input_payload=_msgpack("x")))
    res = conn.wait_for(_is_result_for("r-unknown")).job_result
    assert res.status == pb.JOB_STATUS_INVALID

    # ---- streaming: seq-ordered JobProgress then terminal JobResult ----------
    conn.send(run_job=pb.RunJob(
        request_id="r-stream", attempt=1, function_name="stream3",
        input_payload=_msgpack("marco")))
    conn.wait_for(_is_result_for("r-stream"))
    chunks = [
        m.job_progress for m in conn.received
        if m.WhichOneof("msg") == "job_progress" and m.job_progress.request_id == "r-stream"
    ]
    assert [c.seq for c in chunks] == [1, 2, 3]
    assert all(c.content_type == "application/json" for c in chunks)
    assert msgspec.json.decode(chunks[0].data)["response"] == "chunk-0"

    # ---- cancel: cooperative abort -> CANCELED --------------------------------
    conn.send(run_job=pb.RunJob(
        request_id="r-cancel", attempt=1, function_name="slow",
        input_payload=_msgpack("marco")))
    conn.wait_for(_is_accept_for("r-cancel"))
    conn.send(cancel_job=pb.CancelJob(request_id="r-cancel", attempt=1))
    res = conn.wait_for(_is_result_for("r-cancel")).job_result
    assert res.status == pb.JOB_STATUS_CANCELED

    # ---- retransmission of a live attempt re-acks, never re-executes ---------
    conn.send(run_job=pb.RunJob(
        request_id="r-echo", attempt=1, function_name="echo",
        input_payload=_msgpack("marco")))
    time.sleep(0.3)
    assert conn.count(_is_result_for("r-echo")) == 1

    # ---- stream-kill -> reconnect with backoff --------------------------------
    delays_before = len(harness.worker.transport.reconnect_delays)
    conn.kill()
    conn2 = scheduler.wait_connection(1)
    assert conn2.hello is not None
    assert len(harness.worker.transport.reconnect_delays) > delays_before

    # ---- kill mid-job: in_flight reconcile ships the result exactly once -----
    conn2.send(run_job=pb.RunJob(
        request_id="r-mid", attempt=2, function_name="sleepy",
        input_payload=_msgpack("marco")))
    conn2.wait_for(_is_accept_for("r-mid"))
    conn2.kill()
    conn3 = scheduler.wait_connection(2)
    assert conn3.hello is not None
    in_flight = {(j.request_id, j.attempt) for j in conn3.hello.in_flight}
    assert ("r-mid", 2) in in_flight
    res = conn3.wait_for(_is_result_for("r-mid")).job_result
    assert res.status == pb.JOB_STATUS_OK
    assert res.attempt == 2
    time.sleep(0.3)
    total = sum(c.count(_is_result_for("r-mid")) for c in scheduler.connections)
    assert total == 1, "buffered result must ship exactly once"

    # ---- drain round-trip ------------------------------------------------------
    conn3.send(run_job=pb.RunJob(
        request_id="r-last", attempt=1, function_name="sleepy",
        input_payload=_msgpack("marco")))
    conn3.wait_for(_is_accept_for("r-last"))
    conn3.send(drain=pb.Drain(deadline_ms=5000))
    conn3.send(run_job=pb.RunJob(
        request_id="r-after-drain", attempt=1, function_name="echo",
        input_payload=_msgpack("marco")))
    rejected = conn3.wait_for(_is_result_for("r-after-drain")).job_result
    assert rejected.status == pb.JOB_STATUS_RETRYABLE
    assert "draining" in rejected.safe_message
    assert conn3.count(_is_accept_for("r-after-drain")) == 0
    finished = conn3.wait_for(_is_result_for("r-last")).job_result
    assert finished.status == pb.JOB_STATUS_OK
    assert conn3.client_done.wait(_TIMEOUT), "worker must close the stream after drain"
    assert harness.join() == 0


def test_worker_reconnects_after_hub_restart() -> None:
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    harness = _Harness(scheduler, port)
    harness.start()
    replacement = None
    try:
        scheduler.wait_connection(0)
        before = len(harness.worker.transport.reconnect_delays)
        assert server.stop(grace=0).wait(_TIMEOUT)

        deadline = time.monotonic() + _TIMEOUT
        while len(harness.worker.transport.reconnect_delays) < before + 4:
            assert harness._thread.is_alive(), "worker exited while hub was down"
            assert time.monotonic() < deadline, "worker did not keep reconnecting"
            time.sleep(0.02)

        replacement_scheduler = FakeScheduler()
        replacement = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_WorkerSchedulerServicer_to_server(replacement_scheduler, replacement)
        assert replacement.add_insecure_port(f"127.0.0.1:{port}") == port
        replacement.start()

        assert replacement_scheduler.wait_connection(0).hello is not None
        assert harness.exit_code is None
    finally:
        harness.stop()
        server.stop(grace=0)
        if replacement is not None:
            replacement.stop(grace=0)


def test_deadline_marks_fatal_and_frees_the_worker(scheduler_and_worker) -> None:
    scheduler, harness = scheduler_and_worker
    conn = scheduler.wait_connection(0)
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.phase == pb.WORKER_PHASE_READY
    )
    conn.send(run_job=pb.RunJob(
        request_id="r-deadline", attempt=1, function_name="slow",
        input_payload=_msgpack("marco"), timeout_ms=300))
    res = conn.wait_for(_is_result_for("r-deadline")).job_result
    assert res.status == pb.JOB_STATUS_FATAL
    assert res.safe_message == "deadline exceeded"

    # The slot is free: a fresh job on the same worker completes normally.
    conn.send(run_job=pb.RunJob(
        request_id="r-next", attempt=1, function_name="echo",
        input_payload=_msgpack("marco")))
    res = conn.wait_for(_is_result_for("r-next")).job_result
    assert res.status == pb.JOB_STATUS_OK


def test_gpu_semaphore_serializes_cuda_jobs(scheduler_and_worker) -> None:
    scheduler, _harness = scheduler_and_worker
    conn = scheduler.wait_connection(0)
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.phase == pb.WORKER_PHASE_READY
    )
    cuda = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)
    t0 = time.monotonic()
    for rid in ("r-gpu-1", "r-gpu-2"):
        conn.send(run_job=pb.RunJob(
            request_id=rid, attempt=1, function_name="sleepy",
            input_payload=_msgpack("marco"), compute=cuda))
    for rid in ("r-gpu-1", "r-gpu-2"):
        res = conn.wait_for(_is_result_for(rid)).job_result
        assert res.status == pb.JOB_STATUS_OK
    elapsed = time.monotonic() - t0
    assert elapsed >= 1.0, f"cuda jobs must serialize on 1 gpu slot (took {elapsed:.2f}s)"


def test_gpu_slot_yield_lets_peer_run_during_upload_window(scheduler_and_worker) -> None:
    """#382: a GPU job inside `_gpu_slot_yielded` (the save_bytes upload
    window) must not starve the next GPU job on the single slot."""
    import e2e_endpoints as ep

    scheduler, _harness = scheduler_and_worker
    conn = scheduler.wait_connection(0)
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.phase == pb.WORKER_PHASE_READY
    )
    ep.SLOT_PROBE_STARTED.clear()
    ep.SLOT_PEER_RAN.clear()
    cuda = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)
    conn.send(run_job=pb.RunJob(
        request_id="r-slot-probe", attempt=1, function_name="slot-probe",
        input_payload=_msgpack("x"), compute=cuda))
    assert ep.SLOT_PROBE_STARTED.wait(timeout=10.0), "probe never started"
    # Probe holds the only GPU slot until it yields for its "upload".
    conn.send(run_job=pb.RunJob(
        request_id="r-slot-peer", attempt=1, function_name="slot-peer",
        input_payload=_msgpack("x"), compute=cuda))
    res = conn.wait_for(_is_result_for("r-slot-probe")).job_result
    assert res.status == pb.JOB_STATUS_OK
    assert _decode_out(res.inline).response == "overlapped"
    res = conn.wait_for(_is_result_for("r-slot-peer")).job_result
    assert res.status == pb.JOB_STATUS_OK


def test_gpu_slot_survives_cancel_during_yielded_window(scheduler_and_worker) -> None:
    """#382: cancelling a job while its slot is yielded must not leak or
    double-release the slot -- a follow-up GPU job still runs."""
    import e2e_endpoints as ep

    scheduler, _harness = scheduler_and_worker
    conn = scheduler.wait_connection(0)
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.phase == pb.WORKER_PHASE_READY
    )
    ep.SLOT_PROBE_STARTED.clear()
    ep.SLOT_PEER_RAN.clear()
    cuda = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)
    conn.send(run_job=pb.RunJob(
        request_id="r-yield-cancel", attempt=1, function_name="slot-probe",
        input_payload=_msgpack("x"), compute=cuda))
    assert ep.SLOT_PROBE_STARTED.wait(timeout=10.0)
    conn.send(cancel_job=pb.CancelJob(request_id="r-yield-cancel", attempt=1))
    ep.SLOT_PEER_RAN.set()  # unblock the probe's yielded window
    conn.wait_for(_is_result_for("r-yield-cancel"))
    # The slot must still be usable afterwards.
    conn.send(run_job=pb.RunJob(
        request_id="r-after-cancel", attempt=1, function_name="sleepy",
        input_payload=_msgpack("x"), compute=cuda))
    res = conn.wait_for(_is_result_for("r-after-cancel")).job_result
    assert res.status == pb.JOB_STATUS_OK


def test_finalize_release_overlaps_peer_compute_and_slot_survives(scheduler_and_worker) -> None:
    """gw#476/gw#516: a handler that terminally releases its GPU slot at the
    decode->finalize handoff lets the NEXT job's compute run while it is
    still encoding, completes without reacquiring, and leaves the semaphore
    balanced (a third GPU job still runs)."""
    import e2e_endpoints as ep

    scheduler, _harness = scheduler_and_worker
    conn = scheduler.wait_connection(0)
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.phase == pb.WORKER_PHASE_READY
    )
    ep.FINALIZE_PROBE_STARTED.clear()
    ep.FINALIZE_PEER_RAN.clear()
    cuda = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)
    conn.send(run_job=pb.RunJob(
        request_id="r-final-probe", attempt=1, function_name="finalize-probe",
        input_payload=_msgpack("x"), compute=cuda))
    assert ep.FINALIZE_PROBE_STARTED.wait(timeout=10.0), "probe never started"
    # gw#516: the terminal release makes the finalize tail hub-visible — a
    # StateDelta with finalizing_jobs=1 arrives while the probe encodes,
    # BEFORE any peer is dispatched (drain/retire gating signal).
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.finalizing_jobs == 1
    )
    # The probe is now inside its encode tail with the slot released; the
    # peer's compute must be able to run BEFORE the probe's handler returns.
    conn.send(run_job=pb.RunJob(
        request_id="r-final-peer", attempt=1, function_name="finalize-peer",
        input_payload=_msgpack("x"), compute=cuda))
    res = conn.wait_for(_is_result_for("r-final-probe")).job_result
    assert res.status == pb.JOB_STATUS_OK
    assert _decode_out(res.inline).response == "overlapped"
    # gw#516 typed metrics split: the probe held the slot only until its
    # handoff; the encode tail dominates its runtime.
    assert res.metrics.finalize_wall_ms > 0
    assert res.metrics.slot_held_ms + res.metrics.finalize_wall_ms <= res.metrics.runtime_ms + 1000
    assert res.metrics.slot_held_ms < res.metrics.runtime_ms
    res = conn.wait_for(_is_result_for("r-final-peer")).job_result
    assert res.status == pb.JOB_STATUS_OK
    # The finalize tail ended with the result: finalizing_jobs returns to 0.
    conn.wait_for(
        lambda m: m.WhichOneof("msg") == "state_delta"
        and m.state_delta.finalizing_jobs == 0
    )
    # Executor's post-handler release no-oped (already released) without
    # unbalancing the semaphore: a follow-up GPU job still gets the slot.
    conn.send(run_job=pb.RunJob(
        request_id="r-final-after", attempt=1, function_name="sleepy",
        input_payload=_msgpack("x"), compute=cuda))
    res = conn.wait_for(_is_result_for("r-final-after")).job_result
    assert res.status == pb.JOB_STATUS_OK
    # A job that never hands off holds its slot for ~its whole runtime.
    assert res.metrics.slot_held_ms >= res.metrics.runtime_ms - 1000
    assert res.metrics.finalize_wall_ms <= 1000


def test_marco_polo_example_serves_under_the_new_core() -> None:
    """#365 acceptance: examples/marco-polo runs against the fake scheduler."""
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "examples" / "marco-polo" / "src"
    sys.path.insert(0, str(src))
    try:
        scheduler = FakeScheduler()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="marco-polo-worker",
            worker_jwt="",
        )
        worker = Worker(settings, ["marco_polo.main"], backoff_base_s=0.05)
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        try:
            conn = scheduler.wait_connection(0)
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and m.state_delta.phase == pb.WORKER_PHASE_READY
                and "marco-polo" in m.state_delta.available_functions
            )
            conn.send(run_job=pb.RunJob(
                request_id="mp-1", attempt=1, function_name="marco-polo",
                input_payload=_msgpack("marco")))
            res = conn.wait_for(_is_result_for("mp-1")).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert _decode_out(res.inline).response == "polo"

            conn.send(run_job=pb.RunJob(
                request_id="mp-2", attempt=1, function_name="marco-polo-stream",
                input_payload=_msgpack("marco")))
            res = conn.wait_for(_is_result_for("mp-2")).job_result
            assert res.status == pb.JOB_STATUS_OK
            chunks = [
                m.job_progress for m in conn.received
                if m.WhichOneof("msg") == "job_progress"
                and m.job_progress.request_id == "mp-2"
            ]
            assert [c.seq for c in chunks] == [1, 2, 3, 4]
            assert msgspec.json.decode(chunks[-1].data)["response"] == "polo"
        finally:
            worker.stop()
            thread.join(_TIMEOUT)
            server.stop(grace=0)
    finally:
        sys.path.remove(str(src))


def test_auth_rejection_exits_instead_of_spinning(monkeypatch) -> None:
    # UNAUTHENTICATED can be transient hub-side; the fatal exit is gated on
    # BOTH a failure count and an elapsed window (#372). Shrink the window so
    # the test observes the exit quickly.
    import gen_worker.transport as transport_mod

    monkeypatch.setattr(transport_mod, "_AUTH_FAILURE_EXIT_WINDOW_S", 0.3)
    scheduler = FakeScheduler(reject_unauthenticated=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        harness = _Harness(scheduler, port)
        harness.start()
        assert harness.join(timeout=_TIMEOUT) == 1
    finally:
        server.stop(grace=0)


def test_auth_rejection_within_window_keeps_retrying() -> None:
    """3 quick UNAUTHENTICATED strikes must NOT kill the worker while the
    exit window has not elapsed — a hub pg blip is survivable (#372)."""
    scheduler = FakeScheduler(reject_unauthenticated=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        harness = _Harness(scheduler, port)
        harness.start()
        deadline = time.monotonic() + 3.0
        while len(harness.worker.transport.reconnect_delays) < 4:
            assert harness._thread.is_alive(), "worker exited inside the auth window"
            assert time.monotonic() < deadline, "worker never retried"
            time.sleep(0.02)
        assert harness.exit_code is None
    finally:
        harness.stop()
        server.stop(grace=0)


def test_permanent_precondition_exits_fast() -> None:
    """worker_id_mismatch cannot heal by retrying: exit for reap immediately
    instead of retrying forever (#372)."""

    class _Mismatch(FakeScheduler):
        def Connect(self, request_iterator, context):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          "worker_id_mismatch: hello=w1 jwt_sub=w2")

    scheduler = _Mismatch()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        harness = _Harness(scheduler, port)
        harness.start()
        assert harness.join(timeout=_TIMEOUT) == 1
    finally:
        server.stop(grace=0)


def test_hello_ack_deadline_reconnects_instead_of_hanging(monkeypatch) -> None:
    """A hub that accepts the stream but never sends HelloAck must not hang
    the worker forever (#372): the handshake deadline fires and the worker
    reconnects with backoff."""
    import gen_worker.transport as transport_mod

    monkeypatch.setattr(transport_mod, "_HELLO_ACK_TIMEOUT_S", 0.3)

    stalls = {"n": 0}

    class _Stall(FakeScheduler):
        def Connect(self, request_iterator, context):
            stalls["n"] += 1
            if stalls["n"] == 1:
                next(request_iterator)      # read Hello, then say nothing
                time.sleep(5.0)             # stall well past the deadline
                return iter(())
            return super().Connect(request_iterator, context)

    scheduler = _Stall()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    harness = _Harness(scheduler, port)
    harness.start()
    try:
        # The SECOND attempt (after the deadline fired) completes the handshake.
        conn = scheduler.wait_connection(0)
        assert conn.hello is not None
        assert stalls["n"] >= 2
    finally:
        harness.stop()
        server.stop(grace=0)


def test_not_leader_redirect_is_followed() -> None:
    """FAILED_PRECONDITION not_leader:<addr> redirects the worker to the
    leader immediately; schemeless targets keep the dialing TLS mode (#372,
    plaintext here on both ends)."""
    real = FakeScheduler()
    real_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_WorkerSchedulerServicer_to_server(real, real_server)
    real_port = real_server.add_insecure_port("127.0.0.1:0")
    real_server.start()

    class _NotLeader(FakeScheduler):
        def Connect(self, request_iterator, context):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                          f"not_leader:127.0.0.1:{real_port}")

    stale = _NotLeader()
    stale_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(stale, stale_server)
    stale_port = stale_server.add_insecure_port("127.0.0.1:0")
    stale_server.start()

    harness = _Harness(stale, stale_port)
    harness.start()
    try:
        conn = real.wait_connection(0)
        assert conn.hello is not None and conn.hello.worker_id == "e2e-worker"
    finally:
        harness.stop()
        real_server.stop(grace=0)
        stale_server.stop(grace=0)


def test_normalize_grpc_addr_inherits_tls_for_schemeless_redirects() -> None:
    from gen_worker.transport import normalize_grpc_addr

    assert normalize_grpc_addr("10.0.0.1:7000", default_tls=True) == ("10.0.0.1:7000", True)
    assert normalize_grpc_addr("10.0.0.1:7000", default_tls=False) == ("10.0.0.1:7000", False)
    # Explicit schemes always win.
    assert normalize_grpc_addr("grpc://10.0.0.1:7000", default_tls=True) == ("10.0.0.1:7000", False)
    assert normalize_grpc_addr("grpcs://10.0.0.1:7000", default_tls=False) == ("10.0.0.1:7000", True)
    # No hint: the bare :443 heuristic stands.
    assert normalize_grpc_addr("example.com:443") == ("example.com:443", True)
    assert normalize_grpc_addr("example.com:7000") == ("example.com:7000", False)


# ---------------------------------------------------------------------------
# Send-queue policy (#357): results are never dropped; progress sheds oldest.
# ---------------------------------------------------------------------------


def _result_msg(rid: str, attempt: int = 1) -> pb.WorkerMessage:
    return pb.WorkerMessage(job_result=pb.JobResult(
        request_id=rid, attempt=attempt, status=pb.JOB_STATUS_OK))


def _progress_msg(rid: str, seq: int) -> pb.WorkerMessage:
    return pb.WorkerMessage(job_progress=pb.JobProgress(
        request_id=rid, attempt=1, seq=seq))


def _host_capacity_msg(ref: str, state: int, generation: int) -> pb.WorkerMessage:
    return pb.WorkerMessage(model_event=pb.ModelEvent(
        ref=ref,
        state=state,
        error="insufficient_host_ram" if state == pb.MODEL_STATE_FAILED else "",
        host_ram_required_bytes=12 * 1024**3,
        host_ram_available_before_bytes=8 * 1024**3,
        host_ram_available_after_bytes=16 * 1024**3,
        host_ram_capacity_generation=generation,
    ))


def test_send_queue_drops_oldest_progress_never_results() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=2)
        await q.put(_progress_msg("p", 1))
        await q.put(_progress_msg("p", 2))
        await q.put(_progress_msg("p", 3))  # overflow: seq=1 dropped
        await q.put(_result_msg("r1"))      # results exempt from the bound
        kinds = []
        while len(q):
            kinds.append(await q.get())
        seqs = [m.job_progress.seq for k, m in kinds if k == "progress"]
        assert seqs == [2, 3]
        assert any(k == "result" for k, _m in kinds)

    asyncio.run(_run())


def test_pending_results_do_not_deadlock_pre_send_hello_ack_events() -> None:
    """Results are durable but do not consume bounded event capacity.

    HelloAck handlers run before the send loop and enqueue their state/event
    baseline. Two preserved results in a maxsize=1 queue must not block that
    enqueue forever during reconnect.
    """
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        await q.put(_result_msg("r1"))
        await q.put(_result_msg("r2"))
        state = pb.WorkerMessage(state_delta=pb.StateDelta())
        await asyncio.wait_for(q.put(state), 0.1)

    asyncio.run(_run())


def test_reconnect_capacity_replay_precedes_results_and_is_idempotent() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        failure = _host_capacity_msg("acme/blocked", pb.MODEL_STATE_FAILED, 1)
        progress = _host_capacity_msg(
            "acme/recovered", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        # The failure was queued on the old stream but not written. Reset
        # drops ordinary events and preserves both results.
        await q.put(failure)
        await q.put(_result_msg("r1"))
        await q.put(_result_msg("r2"))
        await q.reset_for_reconnect()

        # HelloAck replay is nonblocking, ahead of durable results, and a
        # duplicate midstream HelloAck cannot enqueue the same generations.
        await asyncio.wait_for(q.prepend_reconnect([failure, progress]), 0.1)
        await asyncio.wait_for(q.prepend_reconnect([failure, progress]), 0.1)
        got = [await q.get() for _ in range(4)]
        assert [msg.WhichOneof("msg") for _, msg in got] == [
            "model_event", "model_event", "job_result", "job_result",
        ]
        assert [
            msg.model_event.state for _, msg in got[:2]
        ] == [pb.MODEL_STATE_FAILED, pb.MODEL_STATE_HOST_CAPACITY_PROGRESS]
        assert len(q) == 0

    asyncio.run(_run())


def test_reconnect_capacity_keeps_failure_before_older_progress() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        active_failure = _host_capacity_msg(
            "acme/active", pb.MODEL_STATE_FAILED, 3,
        )
        undelivered_progress = _host_capacity_msg(
            "acme/satisfied", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        await q.put(_result_msg("r1"))
        await q.prepend_reconnect([active_failure, undelivered_progress])

        got = [await q.get() for _ in range(3)]
        assert [message for _kind, message in got] == [
            active_failure, undelivered_progress, _result_msg("r1"),
        ]

    asyncio.run(_run())


def test_reconnect_reorders_exact_pending_capacity_snapshot() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        active_failure = _host_capacity_msg(
            "acme/active", pb.MODEL_STATE_FAILED, 3,
        )
        undelivered_progress = _host_capacity_msg(
            "acme/satisfied", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        # A midstream producer left the exact entries pending in the inverse
        # order before HelloAck supplied its authoritative replay snapshot.
        await q.put(undelivered_progress)
        await q.put(active_failure)
        await q.prepend_reconnect([active_failure, undelivered_progress])
        await q.prepend_reconnect([active_failure, undelivered_progress])

        got = [await q.get() for _ in range(2)]
        assert [message for _kind, message in got] == [
            active_failure, undelivered_progress,
        ]
        assert len(q) == 0

    asyncio.run(_run())


def test_selected_capacity_is_not_duplicated_and_reset_restores_replay_order() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        active_failure = _host_capacity_msg(
            "acme/active", pb.MODEL_STATE_FAILED, 3,
        )
        undelivered_progress = _host_capacity_msg(
            "acme/satisfied", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        await q.put(undelivered_progress)
        selected = await q.get()
        await q.prepend_reconnect([active_failure, undelivered_progress])
        assert selected[1] == undelivered_progress
        assert await q.should_ship_capacity(selected[1])
        assert len(q) == 1  # only FAILED; selected PROGRESS was not copied

        # Cancellation/write failure resets the selected generation. The next
        # HelloAck reconstructs the authoritative order from executor state.
        await q.reset_for_reconnect()
        await q.prepend_reconnect([active_failure, undelivered_progress])
        got = [await q.get() for _ in range(2)]
        assert [message for _kind, message in got] == [
            active_failure, undelivered_progress,
        ]

    asyncio.run(_run())


def test_newer_hello_ack_fences_blocked_ordinary_state_delta() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        ordinary = pb.WorkerMessage(model_event=pb.ModelEvent(
            ref="acme/on-disk", state=pb.MODEL_STATE_ON_DISK,
        ))
        ready = pb.WorkerMessage(state_delta=pb.StateDelta(
            phase=pb.WORKER_PHASE_READY,
            available_functions=["generate"],
        ))
        error = pb.WorkerMessage(state_delta=pb.StateDelta(
            phase=pb.WORKER_PHASE_ERROR,
        ))
        await q.put(ordinary)
        stale_put = asyncio.create_task(q.put(ready))
        await asyncio.sleep(0)
        assert not stale_put.done()

        await q.prepend_reconnect([error])
        await asyncio.wait_for(stale_put, 0.1)
        first = await q.get()
        second = await q.get()
        assert first[1] == error
        assert second[1] == ordinary
        assert len(q) == 0

    asyncio.run(_run())


def test_capacity_evidence_bypasses_bound_and_newer_generation_wins() -> None:
    """Typed capacity is finite state, so it cannot wake stale after progress."""
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        ordinary = pb.WorkerMessage(model_event=pb.ModelEvent(
            ref="acme/on-disk", state=pb.MODEL_STATE_ON_DISK,
        ))
        failure = _host_capacity_msg("acme/blocked", pb.MODEL_STATE_FAILED, 1)
        await q.put(ordinary)
        capacity_put = asyncio.create_task(q.put(failure))
        await asyncio.sleep(0)
        assert capacity_put.done()
        another_ordinary = pb.WorkerMessage(model_event=pb.ModelEvent(
            ref="acme/also-on-disk", state=pb.MODEL_STATE_ON_DISK,
        ))
        blocked_ordinary = asyncio.create_task(q.put(another_ordinary))
        await asyncio.sleep(0)
        assert not blocked_ordinary.done()

        progress = _host_capacity_msg(
            "acme/blocked", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        await q.prepend_reconnect([progress])
        first = await q.get()
        await q.mark_event_shipped(first[1])
        assert not blocked_ordinary.done()
        second = await q.get()
        await q.mark_event_shipped(second[1])
        await asyncio.wait_for(blocked_ordinary, 0.1)
        third = await q.get()
        await q.mark_event_shipped(third[1])
        assert first[1] == progress
        assert second[1] == ordinary
        assert third[1] == another_ordinary
        assert len(q) == 0

    asyncio.run(_run())


def test_newer_capacity_generation_fences_selected_older_write() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        failure = _host_capacity_msg("acme/blocked", pb.MODEL_STATE_FAILED, 1)
        progress = _host_capacity_msg(
            "acme/blocked", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        await q.put(failure)
        selected = await q.get()
        await q.prepend_reconnect([progress])

        assert await q.should_ship_capacity(selected[1]) is False
        current = await q.get()
        assert current[1] == progress
        assert await q.should_ship_capacity(current[1]) is True

    asyncio.run(_run())


def test_prepend_replaces_stale_same_identity_across_all_lanes() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=2)
        ready = pb.WorkerMessage(state_delta=pb.StateDelta(
            phase=pb.WORKER_PHASE_READY,
            available_functions=["generate"],
        ))
        error = pb.WorkerMessage(state_delta=pb.StateDelta(
            phase=pb.WORKER_PHASE_ERROR,
        ))
        failure = _host_capacity_msg("acme/blocked", pb.MODEL_STATE_FAILED, 1)
        progress = _host_capacity_msg(
            "acme/blocked", pb.MODEL_STATE_HOST_CAPACITY_PROGRESS, 2,
        )
        await q.put(ready)
        # Reproduce an old disconnected capacity copy behind ordinary state;
        # prepend must replace by logical identity, not exact serialization.
        q._items.append(("event", failure))

        await q.prepend_reconnect([error, progress])
        first = await q.get()
        second = await q.get()
        assert first[1] == progress
        assert second[1] == error
        assert len(q) == 0

    asyncio.run(_run())


def test_capacity_get_prepend_race_is_fenced_and_reset_replays() -> None:
    """A sender-owned event is not copied; a new stream clears the fence."""
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        failure = _host_capacity_msg("acme/blocked", pb.MODEL_STATE_FAILED, 1)
        result = _result_msg("r1")
        await q.put(failure)
        await q.put(result)

        first = await q.get()  # the single sender now owns the normal copy
        assert first[1] == failure
        await q.prepend_reconnect([failure])
        await q.mark_event_shipped(first[1])
        second = await q.get()
        assert second[1] == result
        await q.mark_result_shipped(second[1])
        assert len(q) == 0

        await q.reset_for_reconnect()
        await q.prepend_reconnect([failure])
        replay = await q.get()
        assert replay[1] == failure
        assert len(q) == 0

    asyncio.run(_run())


def test_high_volume_progress_does_not_grow_reconnect_bookkeeping() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        for seq in range(2_000):
            messages = (
                _progress_msg("request", seq),
                pb.WorkerMessage(model_event=pb.ModelEvent(
                    ref="acme/downloading",
                    state=pb.MODEL_STATE_DOWNLOADING,
                    bytes_done=seq,
                    bytes_total=2_000,
                )),
            )
            for message in messages:
                await q.put(message)
                _kind, queued = await q.get()
                await q.mark_event_shipped(queued)

        assert q._reconnect_seen == {}
        assert q._in_flight == set()
        assert len(q) == 0

    asyncio.run(_run())


def test_distinct_shipped_capacity_events_leave_no_queue_fences() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=1)
        for generation in range(1, 5_001):
            message = _host_capacity_msg(
                f"acme/model-{generation}",
                pb.MODEL_STATE_HOST_CAPACITY_PROGRESS,
                generation,
            )
            await q.put(message)
            _kind, selected = await q.get()
            assert await q.should_ship_capacity(selected)
            await q.mark_event_shipped(selected)

        assert q._capacity == {}
        assert q._capacity_in_flight == {}
        assert q._reconnect_seen == {}
        assert q._in_flight == set()

    asyncio.run(_run())


def test_send_queue_results_survive_reconnect_until_shipped() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=4)
        await q.put(_progress_msg("p", 1))
        await q.put(_result_msg("r1"))
        await q.put(_result_msg("r2"))
        # Ship r1; the stream then dies before r2 goes out.
        while True:
            kind, msg = await q.get()
            if kind == "result" and msg.job_result.request_id == "r1":
                await q.mark_result_shipped(msg)
                break
        await q.reset_for_reconnect()
        assert q.pending_result_keys == [("r2", 1)]
        kind, msg = await q.get()
        assert kind == "result" and msg.job_result.request_id == "r2"
        assert len(q) == 0  # progress was shed on reconnect

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Desired-residency round-trip: a Tensorhub-bound function is gated until its
# exact instance is warm; ModelEvents follow the contract state machine.
# ---------------------------------------------------------------------------


def _is_model_event(ref: str, state: int):
    return lambda m: (
        m.WhichOneof("msg") == "model_event"
        and m.model_event.ref == ref
        and m.model_event.state == state
    )


def _is_exact_model_event(ref: str, state: int, digest: str, generation: int):
    return lambda m: (
        _is_model_event(ref, state)(m)
        and m.model_event.snapshot_digest == digest
        and m.model_event.residency_generation == generation
    )


def test_desired_residency_downloads_and_warms_round_trip(tmp_path, monkeypatch) -> None:
    import http.server

    from blake3 import blake3

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "hub-cache"))

    # Serve one real weight file over HTTP (the presigned-URL stand-in).
    payload = b"tiny-weights"
    moved_payload = b"moved-tag-weights"
    digest = blake3(payload).hexdigest()
    moved_digest = blake3(moved_payload).hexdigest()
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    (serve_dir / "blob").write_bytes(payload)
    (serve_dir / "blob-moved").write_bytes(moved_payload)

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    blob_url = f"http://127.0.0.1:{httpd.server_address[1]}/blob"
    moved_blob_url = f"http://127.0.0.1:{httpd.server_address[1]}/blob-moved"

    snapshot = pb.Snapshot(
        digest="e2e-snap-1",
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=len(payload),
            blake3=digest, url=blob_url,
        )],
    )
    moved_snapshot = pb.Snapshot(
        digest="e2e-snap-2",
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=len(moved_payload),
            blake3=moved_digest, url=moved_blob_url,
        )],
    )

    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    harness = _Harness(scheduler, port)
    harness.start()
    try:
        conn = scheduler.wait_connection(0)
        ready = conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and m.state_delta.phase == pb.WORKER_PHASE_READY
        )
        # Gated until its model loads: present in loading, absent from available.
        assert "model-echo" not in ready.state_delta.available_functions
        assert "model-echo" in ready.state_delta.loading_functions

        # One full desired state downloads the ref, warms the exact endpoint
        # instance, and reports the accepted generation.
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            desired_residency=pb.DesiredResidency(
                generation=1,
                disk_refs=["e2e/tiny"],
                hot=[pb.DesiredInstance(
                    function_name="model-echo",
                    models=[pb.ModelBinding(slot="model", ref="e2e/tiny")],
                )],
                snapshots={"e2e/tiny": snapshot},
            ),
        ))
        downloading = conn.wait_for(
            _is_model_event("e2e/tiny", pb.MODEL_STATE_DOWNLOADING)
        ).model_event
        on_disk = conn.wait_for(
            _is_model_event("e2e/tiny", pb.MODEL_STATE_ON_DISK)
        ).model_event
        in_ram = conn.wait_for(
            _is_model_event("e2e/tiny", pb.MODEL_STATE_IN_RAM)
        ).model_event
        for event in (downloading, on_disk, in_ram):
            assert event.snapshot_digest == "e2e-snap-1"
            assert event.residency_generation == 1
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and "model-echo" in m.state_delta.available_functions
            and m.state_delta.observed_residency_generation == 1
        )

        # The handler sees the materialized snapshot content.
        conn.send(run_job=pb.RunJob(
            request_id="r-model", attempt=1, function_name="model-echo",
            input_payload=_msgpack("marco")))
        res = conn.wait_for(_is_result_for("r-model")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert _decode_out(res.inline).response == payload.decode()

        # A mutable tag can keep the same wire ref while moving to new bytes.
        # Tensorhub prepositions disk residency before dispatching a job, so
        # prove the disk-only production path first. The worker must vacate A
        # and report B even though no DesiredInstance accompanies generation 2.
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            desired_residency=pb.DesiredResidency(
                generation=2,
                disk_refs=["e2e/tiny"],
                snapshots={"e2e/tiny": moved_snapshot},
            ),
        ))
        moved_downloading = conn.wait_for(
            _is_exact_model_event(
                "e2e/tiny", pb.MODEL_STATE_DOWNLOADING, "e2e-snap-2", 2,
            )
        ).model_event
        moved_disk = conn.wait_for(
            _is_exact_model_event(
                "e2e/tiny", pb.MODEL_STATE_ON_DISK, "e2e-snap-2", 2,
            )
        ).model_event

        # Dispatch intent advances the desired generation for the same B
        # bytes, then warms the exact instance without downloading again.
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            desired_residency=pb.DesiredResidency(
                generation=3,
                disk_refs=["e2e/tiny"],
                hot=[pb.DesiredInstance(
                    function_name="model-echo",
                    models=[pb.ModelBinding(slot="model", ref="e2e/tiny")],
                )],
                snapshots={"e2e/tiny": moved_snapshot},
            ),
        ))
        moved_ram = conn.wait_for(
            _is_exact_model_event(
                "e2e/tiny", pb.MODEL_STATE_IN_RAM, "e2e-snap-2", 3,
            )
        ).model_event
        for event in (moved_downloading, moved_disk):
            assert event.snapshot_digest == "e2e-snap-2"
            assert event.residency_generation == 2
        assert moved_ram.snapshot_digest == "e2e-snap-2"
        assert moved_ram.residency_generation == 3
        conn.send(run_job=pb.RunJob(
            request_id="r-model-moved", attempt=1, function_name="model-echo",
            input_payload=_msgpack("marco")))
        moved = conn.wait_for(_is_result_for("r-model-moved")).job_result
        assert moved.status == pb.JOB_STATUS_OK
        assert _decode_out(moved.inline).response == moved_payload.decode()

        # Reconnect baseline is the actual materialized B bytes, not whatever
        # tag target happens to be current when Hello is built.
        residencies = {
            r.ref: r for r in harness.worker.executor.store.residency_snapshot()
        }
        observed = residencies["e2e/tiny"]
        assert observed.snapshot_digest == "e2e-snap-2"
        assert observed.residency_generation == 3

    finally:
        harness.stop()
        server.stop(grace=0)
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Host-RAM admission failure crosses the real worker transport before the
# retry result. Only the largest staged ref fails; the smaller shared VAE
# remains usable by other functions (th#807).
# ---------------------------------------------------------------------------


def test_host_ram_failure_precedes_retryable_result_on_wire(tmp_path, monkeypatch) -> None:
    import http.server

    from blake3 import blake3

    from gen_worker.models import disk_gc
    from gen_worker.models import residency as residency_mod

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "hub-cache"))
    monkeypatch.setattr(residency_mod, "get_total_ram_gb", lambda: 31.0)
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 8.0)

    pipeline_payload = b"large-pipeline"
    vae_payload = b"small-shared-vae"
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    (serve_dir / "pipeline").write_bytes(pipeline_payload)
    (serve_dir / "vae").write_bytes(vae_payload)

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"

    def _snapshot(payload: bytes, name: str) -> pb.Snapshot:
        return pb.Snapshot(
            digest=f"ram-pressure-{name}",
            files=[pb.SnapshotFile(
                path="model.safetensors",
                size_bytes=len(payload),
                blake3=blake3(payload).hexdigest(),
                url=f"{base_url}/{name}",
            )],
        )

    def _tree_bytes(path) -> int:
        payload = (path / "model.safetensors").read_bytes()
        return (6 if payload == pipeline_payload else 1) * 1024**3

    monkeypatch.setattr(disk_gc, "tree_bytes", _tree_bytes)

    pipeline_ref = "e2e/ram-pressure-pipeline"
    vae_ref = "e2e/ram-pressure-shared-vae"
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    harness = _Harness(scheduler, port)
    harness.start()
    try:
        conn = scheduler.wait_connection(0)
        conn.wait_for(
            lambda message: message.WhichOneof("msg") == "state_delta"
            and message.state_delta.phase == pb.WORKER_PHASE_READY
        )
        conn.send(run_job=pb.RunJob(
            request_id="r-host-ram",
            attempt=1,
            function_name="ram-pressure",
            input_payload=_msgpack("marco"),
            snapshots={
                pipeline_ref: _snapshot(pipeline_payload, "pipeline"),
                vae_ref: _snapshot(vae_payload, "vae"),
            },
        ))
        result = conn.wait_for(_is_result_for("r-host-ram")).job_result
        assert result.status == pb.JOB_STATUS_RETRYABLE

        with conn._recv_cond:
            received = list(conn.received)
        failed = [
            (index, message.model_event)
            for index, message in enumerate(received)
            if message.WhichOneof("msg") == "model_event"
            and message.model_event.state == pb.MODEL_STATE_FAILED
        ]
        assert [
            (event.ref, event.error) for _, event in failed
        ] == [(pipeline_ref, "insufficient_host_ram")]
        failure = failed[0][1]
        assert failure.host_ram_required_bytes == pytest.approx(12.2 * 1024**3, rel=1e-6)
        assert failure.host_ram_available_before_bytes == 8 * 1024**3
        assert failure.host_ram_available_after_bytes == 8 * 1024**3
        assert list(failure.host_ram_evicted_refs) == []
        assert failure.host_ram_capacity_generation == 1
        result_index = next(
            index for index, message in enumerate(received)
            if message.WhichOneof("msg") == "job_result"
            and message.job_result.request_id == "r-host-ram"
        )
        assert failed[0][0] < result_index
        assert all(event.ref != vae_ref for _, event in failed)
    finally:
        harness.stop()
        server.stop(grace=0)
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Setup failure surfaces FnUnavailable (th#581 worker-side / ernie roster
# find): a function whose pipeline setup raises must NOT sit in
# loading_functions forever under a READY phase — it leaves BOTH lists and a
# terminal FnUnavailable{setup_failed} reaches the hub. Re-sending the same
# desired generation retries setup and re-advertises it after recovery.
# ---------------------------------------------------------------------------


def test_setup_failure_emits_fn_unavailable_and_recovers(tmp_path, monkeypatch) -> None:
    import http.server

    from blake3 import blake3

    import e2e_endpoints

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "hub-cache"))

    payload = b"tiny-weights"
    digest = blake3(payload).hexdigest()
    serve_dir = tmp_path / "www"
    serve_dir.mkdir()
    (serve_dir / "blob").write_bytes(payload)

    class _Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    blob_url = f"http://127.0.0.1:{httpd.server_address[1]}/blob"
    snapshot = pb.Snapshot(
        digest="e2e-snap-broken",
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=len(payload),
            blake3=digest, url=blob_url,
        )],
    )

    def _is_fn_unavailable(m: pb.WorkerMessage) -> bool:
        return (
            m.WhichOneof("msg") == "fn_unavailable"
            and m.fn_unavailable.function_name == "broken-echo"
            and m.fn_unavailable.reason == "setup_failed"
        )

    e2e_endpoints.BREAK_SETUP.set()
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    harness = _Harness(scheduler, port)
    harness.start()
    try:
        conn = scheduler.wait_connection(0)
        ready = conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and m.state_delta.phase == pb.WORKER_PHASE_READY
        )
        assert "broken-echo" in ready.state_delta.loading_functions

        desired = pb.DesiredResidency(
            generation=1,
            disk_refs=["e2e/broken"],
            hot=[pb.DesiredInstance(
                function_name="broken-echo",
                models=[pb.ModelBinding(slot="model", ref="e2e/broken")],
            )],
            snapshots={"e2e/broken": snapshot},
        )

        # Desired setup raises -> terminal per-function signal, and the fn
        # leaves BOTH available and loading (no more silent-ready limbo).
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            desired_residency=desired,
        ))
        sig = conn.wait_for(_is_fn_unavailable).fn_unavailable
        assert "pipeline exploded" in sig.detail
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and "broken-echo" not in m.state_delta.available_functions
            and "broken-echo" not in m.state_delta.loading_functions
        )
        # The residency path also reports the model failure itself.
        conn.wait_for(_is_model_event("e2e/broken", pb.MODEL_STATE_FAILED))

        # Same-generation full replacement is a retry (and URL refresh), so
        # setup succeeds without inventing an imperative command.
        e2e_endpoints.BREAK_SETUP.clear()
        conn.send(hello_ack=pb.HelloAck(
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            desired_residency=desired,
        ))
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and "broken-echo" in m.state_delta.available_functions
            and m.state_delta.observed_residency_generation == 1
        )
        assert "broken-echo" not in harness.worker.executor.unavailable

        # Serves for real after recovery.
        conn.send(run_job=pb.RunJob(
            request_id="r-broken", attempt=1, function_name="broken-echo",
            input_payload=_msgpack("marco")))
        res = conn.wait_for(_is_result_for("r-broken")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert _decode_out(res.inline).response == payload.decode()

    finally:
        e2e_endpoints.BREAK_SETUP.set()
        harness.stop()
        server.stop(grace=0)
        httpd.shutdown()
