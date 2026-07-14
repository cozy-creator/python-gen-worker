"""#376: reserved-source materialization.

Producer-kind (conversion/dataset/training) payloads carrying the reserved
``source`` struct get a materialized local snapshot at ``ctx.source_path``
(and ``ctx.source`` metadata) before the handler runs. Failures classify like
model-binding downloads; missing source is a no-op."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import msgspec

from gen_worker.api.errors import RetryableError
from gen_worker.api.types import SourceRepo
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _ConvIn(msgspec.Struct):
    source: Optional[SourceRepo] = None
    dtype: str = "bf16"


class _Out(msgspec.Struct):
    source_path: str
    source_ref: str


def _convert(ctx, payload: _ConvIn) -> _Out:
    return _Out(
        source_path=str(ctx.source_path or ""),
        source_ref=str((ctx.source or {}).get("ref") or ""),
    )


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="cast-dtype", method=_convert, kind="conversion",
        payload_type=_ConvIn, output_mode="single",
    )


class _Harness:
    def __init__(self, tmp_path: Path, *, fail_with: Optional[BaseException] = None) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.ensured: List[str] = []
        self.tmp_path = tmp_path
        self.fail_with = fail_with

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        self.executor = Executor([_spec()], _send)

        async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
            self.ensured.append(ref)
            if self.fail_with is not None:
                raise self.fail_with
            return self.tmp_path

        self.executor.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]

    async def run(self, payload: _ConvIn) -> pb.JobResult:
        await self.executor.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="cast-dtype",
            input_payload=msgspec.msgpack.encode(payload)))
        job = self.executor.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in self.sent if m.WhichOneof("msg") == "job_result"]
        assert results, f"no job_result; sent={self.sent}"
        return results[-1]


def test_source_materialized_before_handler(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_ConvIn(source=SourceRepo(
            ref="acme/base-model:prod", attributes={"dtype": "fp32"})))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path == str(tmp_path)
        assert out.source_ref == "acme/base-model:prod"
        assert h.ensured == ["acme/base-model:prod"]

    asyncio.run(_run())


def test_missing_source_is_noop(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_ConvIn())
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path == ""
        assert h.ensured == []

    asyncio.run(_run())


def test_empty_source_ref_is_invalid(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_ConvIn(source=SourceRepo(ref="  ")))
        assert res.status == pb.JOB_STATUS_INVALID
        assert h.ensured == []

    asyncio.run(_run())


def test_source_download_failure_classifies_like_model_bindings(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path, fail_with=RetryableError("snapshot not provided"))
        res = await h.run(_ConvIn(source=SourceRepo(ref="acme/base-model:prod")))
        assert res.status == pb.JOB_STATUS_RETRYABLE

        h2 = _Harness(tmp_path, fail_with=ValueError("unsupported model ref"))
        res2 = await h2.run(_ConvIn(source=SourceRepo(ref="not-a-ref")))
        assert res2.status == pb.JOB_STATUS_INVALID

    asyncio.run(_run())


def test_inference_kind_ignores_reserved_source(tmp_path) -> None:
    class _InfOut(msgspec.Struct):
        ok: bool

    def _infer(ctx, payload: _ConvIn) -> _InfOut:
        return _InfOut(ok=not hasattr(ctx, "source_path"))

    spec = EndpointSpec(
        name="infer", method=_infer, kind="inference",
        payload_type=_ConvIn, output_mode="single",
    )

    async def _run() -> None:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        ex = Executor([spec], _send)

        async def _boom(ref, snapshot=None, *, binding=None) -> Path:
            raise AssertionError("inference must not materialize source")

        ex.store.ensure_local = _boom  # type: ignore[method-assign]
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="infer",
            input_payload=msgspec.msgpack.encode(
                _ConvIn(source=SourceRepo(ref="acme/base-model:prod")))))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert results[-1].status == pb.JOB_STATUS_OK

    asyncio.run(_run())
