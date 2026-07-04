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
            worker_disconnected_timeout_s=60,
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
    time.sleep(0.05)
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
                and "marco_polo" in m.state_delta.available_functions
            )
            conn.send(run_job=pb.RunJob(
                request_id="mp-1", attempt=1, function_name="marco_polo",
                input_payload=_msgpack("marco")))
            res = conn.wait_for(_is_result_for("mp-1")).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert _decode_out(res.inline).response == "polo"

            conn.send(run_job=pb.RunJob(
                request_id="mp-2", attempt=1, function_name="marco_polo_stream",
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


def test_auth_rejection_exits_instead_of_spinning() -> None:
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


# ---------------------------------------------------------------------------
# Send-queue policy (#357): results are never dropped; progress sheds oldest.
# ---------------------------------------------------------------------------


def _result_msg(rid: str, attempt: int = 1) -> pb.WorkerMessage:
    return pb.WorkerMessage(job_result=pb.JobResult(
        request_id=rid, attempt=attempt, status=pb.JOB_STATUS_OK))


def _progress_msg(rid: str, seq: int) -> pb.WorkerMessage:
    return pb.WorkerMessage(job_progress=pb.JobProgress(
        request_id=rid, attempt=1, seq=seq))


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
# ModelOp DOWNLOAD / LOAD / UNLOAD round-trip (#366): a tensorhub-bound
# function is gated until LOAD; ModelEvents follow the contract state machine.
# ---------------------------------------------------------------------------


def _is_model_event(ref: str, state: int):
    return lambda m: (
        m.WhichOneof("msg") == "model_event"
        and m.model_event.ref == ref
        and m.model_event.state == state
    )


def test_model_op_download_load_unload_round_trip(tmp_path, monkeypatch) -> None:
    import http.server

    from blake3 import blake3

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "hub-cache"))

    # Serve one real weight file over HTTP (the presigned-URL stand-in).
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
        digest="e2e-snap-1",
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=len(payload),
            blake3=digest, url=blob_url,
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
        assert "model_echo" not in ready.state_delta.available_functions
        assert "model_echo" in ready.state_delta.loading_functions

        # DOWNLOAD -> DOWNLOADING then ON_DISK.
        conn.send(model_op=pb.ModelOp(
            op=pb.MODEL_OP_KIND_DOWNLOAD, ref="e2e/tiny", snapshot=snapshot))
        conn.wait_for(_is_model_event("e2e/tiny", pb.MODEL_STATE_DOWNLOADING))
        conn.wait_for(_is_model_event("e2e/tiny", pb.MODEL_STATE_ON_DISK))

        # LOAD -> setup runs (typed str path injection); CPU host => IN_RAM;
        # the function flips to available.
        conn.send(model_op=pb.ModelOp(
            op=pb.MODEL_OP_KIND_LOAD, ref="e2e/tiny", snapshot=snapshot))
        conn.wait_for(_is_model_event("e2e/tiny", pb.MODEL_STATE_IN_RAM))
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and "model_echo" in m.state_delta.available_functions
        )

        # The handler sees the materialized snapshot content.
        conn.send(run_job=pb.RunJob(
            request_id="r-model", attempt=1, function_name="model_echo",
            input_payload=_msgpack("marco")))
        res = conn.wait_for(_is_result_for("r-model")).job_result
        assert res.status == pb.JOB_STATUS_OK
        assert _decode_out(res.inline).response == payload.decode()

        # Hello-shape sanity: the residency snapshot carries the ref.
        snap_refs = {r.ref for r in harness.worker.executor.store.residency_snapshot()}
        assert "e2e/tiny" in snap_refs

        # UNLOAD -> teardown; residency falls back to ON_DISK (a SECOND
        # on-disk event, after the download's); function gated again.
        conn.send(model_op=pb.ModelOp(op=pb.MODEL_OP_KIND_UNLOAD, ref="e2e/tiny"))
        deadline = time.monotonic() + _TIMEOUT
        while conn.count(_is_model_event("e2e/tiny", pb.MODEL_STATE_ON_DISK)) < 2:
            assert time.monotonic() < deadline, "no ON_DISK event after UNLOAD"
            time.sleep(0.02)
        conn.wait_for(
            lambda m: m.WhichOneof("msg") == "state_delta"
            and "model_echo" not in m.state_delta.available_functions
            and "model_echo" in m.state_delta.loading_functions
        )
    finally:
        harness.stop()
        server.stop(grace=0)
        httpd.shutdown()
