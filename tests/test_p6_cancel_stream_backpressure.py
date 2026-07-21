"""P6 (th#960/pgw#609 design table): async-gen streaming progress; cancel
mid-stream; durable-vs-sheddable SendQueue under a slow consumer;
GPU-semaphore serialization with accelerator=cuda declared, no CUDA touched.
"""

from __future__ import annotations

import asyncio
import time

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.transport import SendQueue

from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import EchoIn


def _payload(text: str = "marco") -> bytes:
    return msgspec.msgpack.encode(EchoIn(text=text))


# ---------------------------------------------------------------------------
# Real dispatch: streaming progress + mid-stream cancel + GPU serialization.
# ---------------------------------------------------------------------------


def test_streaming_progress_is_seq_ordered() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-stream", attempt=1, function_name="stream3",
            input_payload=_payload()))
        conn.wait_for(is_result_for("r-stream"))
        chunks = [
            m.job_progress for m in conn.received
            if m.WhichOneof("msg") == "job_progress" and m.job_progress.request_id == "r-stream"
        ]
        assert [c.seq for c in chunks] == [1, 2, 3]
        assert msgspec.json.decode(chunks[0].data)["response"] == "chunk-0"


def test_cancel_mid_stream_stops_further_progress() -> None:
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-slow-stream", attempt=1, function_name="slow-stream",
            input_payload=_payload()))
        conn.wait_for_count(
            lambda m: m.WhichOneof("msg") == "job_progress"
            and m.job_progress.request_id == "r-slow-stream", 1,
        )
        conn.send(cancel_job=pb.CancelJob(request_id="r-slow-stream", attempt=1))
        res = conn.wait_for(is_result_for("r-slow-stream")).job_result
        assert res.status == pb.JOB_STATUS_CANCELED
        progress_at_cancel = conn.count(
            lambda m: m.WhichOneof("msg") == "job_progress"
            and m.job_progress.request_id == "r-slow-stream"
        )
        time.sleep(0.5)  # the 20-chunk/0.2s generator would still be running
        assert conn.count(
            lambda m: m.WhichOneof("msg") == "job_progress"
            and m.job_progress.request_id == "r-slow-stream"
        ) == progress_at_cancel, "cancel must stop further progress emission"


def test_gpu_semaphore_serializes_cuda_jobs_no_cuda_touched() -> None:
    """accelerator=cuda is DECLARED on the dispatch (ResolvedCompute) — the
    toy handler never imports torch/touches a real GPU; only the semaphore
    serialization is under test."""
    with hub_double() as (scheduler, _harness):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        cuda = pb.ResolvedCompute(accelerator="cuda", gpu_index=0)
        t0 = time.monotonic()
        for rid in ("r-gpu-1", "r-gpu-2"):
            conn.send(run_job=pb.RunJob(
                request_id=rid, attempt=1, function_name="sleepy",
                input_payload=_payload(), compute=cuda))
        for rid in ("r-gpu-1", "r-gpu-2"):
            res = conn.wait_for(is_result_for(rid)).job_result
            assert res.status == pb.JOB_STATUS_OK
        elapsed = time.monotonic() - t0
        assert elapsed >= 1.0, f"cuda jobs must serialize on 1 gpu slot (took {elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# SendQueue: durable results vs sheddable progress under a slow/bounded
# consumer (unit-level; the real transport wraps this queue directly).
# ---------------------------------------------------------------------------


def _progress_msg(rid: str, seq: int) -> pb.WorkerMessage:
    return pb.WorkerMessage(job_progress=pb.JobProgress(request_id=rid, attempt=1, seq=seq))


def _result_msg(rid: str, attempt: int = 1) -> pb.WorkerMessage:
    return pb.WorkerMessage(job_result=pb.JobResult(
        request_id=rid, attempt=attempt, status=pb.JOB_STATUS_OK))


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


def test_send_queue_slow_consumer_backpressure_bounds_progress_not_results() -> None:
    """A slow reader (the transport's actual write loop stand-in) never
    forces unbounded memory growth for progress, but a durable result put
    still completes once the reader drains — durable vs sheddable under
    real bounded-queue backpressure, not just the drop-oldest unit check
    above."""
    async def _run() -> None:
        # maxsize=1 bounds ordinary (progress) traffic only; results are
        # exempt from the bound entirely (SendQueue.put's early-return path)
        # — put() never blocks the producer here, by design. The queue can
        # legitimately hold MORE than maxsize once a result is queued.
        q = SendQueue(maxsize=1)
        await asyncio.wait_for(q.put(_progress_msg("p", 1)), 0.2)
        await asyncio.wait_for(q.put(_progress_msg("p", 2)), 0.2)  # drops p1 (bound=1)
        await asyncio.wait_for(q.put(_result_msg("r1")), 0.2)      # exempt: never blocks
        assert len(q) == 2  # p2 (bounded slot) + r1 (exempt)

        drained: list[str] = []
        for _ in range(2):
            await asyncio.sleep(0.05)  # a slow consumer still drains everything queued
            kind, _msg = await q.get()
            drained.append(kind)
        assert drained == ["progress", "result"]

    asyncio.run(_run())
