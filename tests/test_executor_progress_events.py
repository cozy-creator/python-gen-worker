"""th#640/J19: ctx.progress from an orchestrated handler must reach the hub.

The executor wires a RequestContext emitter that turns ctx.progress / ctx.log /
checkpoint events into JobProgress envelopes on the worker gRPC stream
(content_type application/x-request-event+json, JSON event verbatim in data).
tensorhub fans those to /v1/requests/:id/events SSE as output.delta with
payload.delta carrying the JSON — J19 parses "training step N/M loss=X" and
checkpoint markers out of exactly that.
"""

from __future__ import annotations

import asyncio
import json
from typing import Dict, List

import msgspec
import pytest

from gen_worker.executor import EVENT_CONTENT_TYPE, Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec
from gen_worker.request_context import TrainingContext


class _In(msgspec.Struct):
    steps: int = 500


class _Out(msgspec.Struct):
    ok: bool


def _train(ctx, payload: _In) -> _Out:
    for step in (100, 250):
        ctx.progress(step / payload.steps, f"training step {step}/{payload.steps} loss=0.0123")
    ctx.log("saving checkpoint", level="info")
    return _Out(ok=True)


def _run(kind: str, method) -> List[pb.WorkerMessage]:
    async def _go() -> List[pb.WorkerMessage]:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        spec = EndpointSpec(
            name="fn", method=method, kind=kind,
            payload_type=_In, output_mode="single",
        )
        ex = Executor([spec], _send)
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="fn",
            input_payload=msgspec.msgpack.encode(_In())))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        # Drain event coroutines scheduled from the handler thread.
        for _ in range(20):
            await asyncio.sleep(0)
        return sent

    return asyncio.run(_go())


def _events(sent: List[pb.WorkerMessage]) -> List[Dict]:
    out = []
    for m in sent:
        if m.WhichOneof("msg") != "job_progress":
            continue
        p = m.job_progress
        if p.content_type != EVENT_CONTENT_TYPE:
            continue
        out.append({"seq": p.seq, "event": json.loads(p.data)})
    return out


def test_training_ctx_progress_reaches_job_progress_stream() -> None:
    sent = _run("training", _train)
    events = _events(sent)
    assert len(events) == 3

    types = [e["event"]["type"] for e in events]
    assert types == ["request.progress", "request.progress", "request.log"]
    for e in events:
        assert e["event"]["request_id"] == "r1"

    stages = [e["event"]["payload"].get("stage", "") for e in events[:2]]
    assert stages == ["training step 100/500 loss=0.0123",
                      "training step 250/500 loss=0.0123"]
    assert events[0]["event"]["payload"]["progress"] == pytest.approx(0.2)

    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK


def test_inference_ctx_progress_also_flows() -> None:
    def _infer(ctx, payload: _In) -> _Out:
        ctx.progress(0.5, "denoise")
        return _Out(ok=True)

    events = _events(_run("inference", _infer))
    assert [e["event"]["type"] for e in events] == ["request.progress"]


def test_save_checkpoint_emits_checkpoint_event(tmp_path) -> None:
    """The trainer's periodic LoRA saves must surface as checkpoint events
    (J19 asserts markers at steps 250/500)."""
    src = tmp_path / "lora_000000250.safetensors"
    src.write_bytes(b"weights")
    captured: List[Dict] = []
    ctx = TrainingContext(
        request_id="r1",
        emitter=captured.append,
        local_output_dir=str(tmp_path / "outs"),
        worker_capability_token="cap-tok",
        execution_hints={"kind": "training"},
    )
    ctx.save_checkpoint(
        "checkpoints/lora_000000250.safetensors", str(src),
        step_number=250, output_kind="lora",
    )
    ckpt = [e for e in captured if e["type"] == "request.checkpoint"]
    assert len(ckpt) == 1
    payload = ckpt[0]["payload"]
    assert payload["step_number"] == 250
    assert payload["output_kind"] == "lora"
    assert payload["ref"].endswith("lora_000000250.safetensors")
    assert payload["size_bytes"] == len(b"weights")


def _train_metrics(ctx, payload: _In) -> _Out:
    # 10 fast steps: throttle (5s default) keeps only first + last.
    for step in range(1, 11):
        ctx.training_metric(step=step, total=10, loss=1.0 / step, lr=1e-4)
    return _Out(ok=True)


def test_training_metric_flows_first_and_last_throttled() -> None:
    """pgw#450: ctx.training_metric rides the same JobProgress envelope;
    the built-in min-interval throttle drops mid-run spam but always
    emits the first and the final (step==total) metric."""
    sent = _run("training", _train_metrics)
    events = [e for e in _events(sent)
              if e["event"]["type"] == "request.training_metric"]
    assert [e["event"]["payload"]["step"] for e in events] == [1, 10]

    first = events[0]["event"]["payload"]
    assert first == {"step": 1, "total": 10, "loss": 1.0, "lr": 1e-4}
    # Optional fields not passed must be absent, not null.
    assert "it_s" not in first and "eta_s" not in first

    results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
    assert results and results[-1].status == pb.JOB_STATUS_OK


def test_training_metric_unthrottled_payload_shape() -> None:
    """Direct-ctx: interval 0 emits every step with the full typed payload."""
    captured: List[Dict] = []
    ctx = TrainingContext(request_id="r1", emitter=captured.append)
    ctx.metric_min_interval_s = 0.0
    for step in (1, 2, 3):
        ctx.training_metric(step=step, total=100, loss=0.5,
                            lr=2e-5, it_s=1.7, eta_s=57.1)
    metrics = [e for e in captured if e["type"] == "request.training_metric"]
    assert len(metrics) == 3
    assert metrics[-1]["payload"] == {
        "step": 3, "total": 100, "loss": 0.5,
        "lr": 2e-5, "it_s": 1.7, "eta_s": 57.1,
    }
    assert all(e["request_id"] == "r1" for e in metrics)
