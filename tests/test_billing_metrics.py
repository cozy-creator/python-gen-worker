"""pgw#512/#513: typed billing usage on JobMetrics + per-job VRAM peaks.

Settlement must read tokens/output-count from JobMetrics, never scavenge the
(possibly blob-ref'd) result payload by field name (pgw#512) — proven here by
computing metrics on a >64KB output, size-independent by construction since
the executor reads the Python object directly, before any msgpack
inline/blob_ref serialization decision. peak_vram_bytes must be a true
per-job peak: torch.cuda.reset_peak_memory_stats() runs at handler start,
under the GPU semaphore that serializes GPU jobs one-at-a-time (pgw#513).
"""

from __future__ import annotations

import asyncio
import time
import types
from typing import List

import msgspec

from gen_worker.api.streaming import StreamResult, TokenUsage
from gen_worker.api.types import ImageAsset
from gen_worker.api.decorators import Resources
from gen_worker.executor import INLINE_RESULT_MAX_BYTES, Executor, _output_token_usage, _scan_output_assets
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str = "x"


class _Out(msgspec.Struct):
    images: List[ImageAsset]


# ---------------------------------------------------------------------------
# _output_token_usage / _scan_output_assets — pure helpers
# ---------------------------------------------------------------------------


def test_output_token_usage_reads_stream_result_usage() -> None:
    usage = TokenUsage(prompt_tokens=100, cached_tokens=40, completion_tokens=25)
    out = StreamResult(text="hello", usage=usage)
    assert _output_token_usage(out) is usage


def test_output_token_usage_none_for_non_stream_output() -> None:
    assert _output_token_usage(_Out(images=[ImageAsset(ref="i")])) is None
    assert _output_token_usage(None) is None


def test_scan_output_assets_counts_images() -> None:
    out = _Out(images=[ImageAsset(ref="a"), ImageAsset(ref="b"), ImageAsset(ref="c")])
    duration_s, count = _scan_output_assets(out)
    assert (duration_s, count) == (0.0, 3)


# ---------------------------------------------------------------------------
# Executor._metrics — the JobMetrics assembly point. Doesn't touch `self`,
# so it's callable unbound; no live Executor/torch/GPU required.
# ---------------------------------------------------------------------------


def test_metrics_folds_token_usage_and_output_count() -> None:
    usage = TokenUsage(prompt_tokens=1000, cached_tokens=200, completion_tokens=500)
    out = StreamResult(text="a" * 10, usage=usage)
    m = Executor._metrics(None, queue_ms=5, started=time.monotonic(), concurrency_at_start=1,
                          gpu_index=0, output=out)
    assert m.input_tokens == 1000
    assert m.input_cached_tokens == 200
    assert m.output_tokens == 500
    assert m.output_count == 0  # a bare token stream has no output Assets


def test_metrics_folds_output_count_for_asset_output() -> None:
    out = _Out(images=[ImageAsset(ref="a"), ImageAsset(ref="b")])
    m = Executor._metrics(None, queue_ms=0, started=time.monotonic(), concurrency_at_start=0,
                          gpu_index=0, output=out)
    assert m.output_count == 2
    assert m.input_tokens == 0 and m.output_tokens == 0


def test_metrics_token_usage_survives_blob_ref_sized_output() -> None:
    """pgw#512 bug (b): outputs >64KB go blob_ref on the wire
    (executor.py _serialize_output), so settlement reading only
    result.GetInline() silently lost token counts on long generations.
    JobMetrics is computed from the raw Python output BEFORE that
    inline/blob_ref decision, so it is size-independent by construction —
    prove the output here really would have gone blob_ref, and that the
    metrics are still correct."""
    usage = TokenUsage(prompt_tokens=50_000, cached_tokens=10_000, completion_tokens=20_000)
    out = StreamResult(text="x" * 200_000, usage=usage)  # forces blob_ref sizing
    encoded = msgspec.msgpack.encode(out)
    assert len(encoded) > INLINE_RESULT_MAX_BYTES, "test output must exceed the inline cap"

    m = Executor._metrics(None, queue_ms=0, started=time.monotonic(), concurrency_at_start=0,
                          gpu_index=0, output=out)
    assert m.input_tokens == 50_000
    assert m.input_cached_tokens == 10_000
    assert m.output_tokens == 20_000


def test_job_metrics_proto_carries_new_fields() -> None:
    m = pb.JobMetrics(
        runtime_ms=1000, rss_at_end_bytes=123, peak_vram_bytes=456,
        input_tokens=1, input_cached_tokens=2, output_tokens=3, output_count=4,
    )
    assert (m.rss_at_end_bytes, m.peak_vram_bytes) == (123, 456)
    assert (m.input_tokens, m.input_cached_tokens, m.output_tokens, m.output_count) == (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# pgw#513: reset_peak_memory_stats() at handler start, under the GPU
# semaphore that serializes GPU jobs one-at-a-time.
# ---------------------------------------------------------------------------


def _fake_cuda_module(reset_calls: List[int]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda idx: None,
        reset_peak_memory_stats=lambda idx: reset_calls.append(idx),
        max_memory_allocated=lambda idx: 777,
    )


def test_reset_peak_memory_stats_runs_at_gpu_job_handler_start(monkeypatch) -> None:
    import gen_worker.executor as executor_mod

    reset_calls: List[int] = []
    fake_torch = types.SimpleNamespace(cuda=_fake_cuda_module(reset_calls))
    monkeypatch.setattr(executor_mod, "torch", fake_torch)

    def _handler(ctx, payload: _In) -> _Out:
        # By the time the handler runs, the reset must already have fired —
        # this job now exclusively owns the GPU under the semaphore.
        assert reset_calls == [0]
        return _Out(images=[])

    spec = EndpointSpec(
        name="fn", method=_handler, kind="inference", payload_type=_In,
        output_mode="single", output_type=_Out, resources=Resources(gpu=True),
    )

    async def _go() -> pb.JobResult:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        ex = executor_mod.Executor([spec], _send)
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="fn",
            input_payload=msgspec.msgpack.encode(_In())))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert results, f"no job_result sent; sent={sent}"
        return results[-1]

    result = asyncio.run(_go())
    assert result.status == pb.JOB_STATUS_OK
    assert reset_calls == [0]
    assert result.metrics.peak_vram_bytes == 777
