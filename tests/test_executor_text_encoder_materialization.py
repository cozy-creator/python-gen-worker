"""pgw#594/te#70: second reserved model-input materialization.

Producer-kind payloads carrying the reserved ``text_encoder`` struct (same
``SourceRepo`` type as ``source``) get a materialized local snapshot at
``ctx.text_encoder_path``, independent of and never clobbering ``source`` /
``ctx.source_path``. Absent field is a no-op — every existing endpoint that
doesn't declare ``text_encoder`` sees no behavior change."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import msgspec

from gen_worker.api.types import SourceRepo
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _TrainIn(msgspec.Struct):
    source: Optional[SourceRepo] = None
    text_encoder: Optional[SourceRepo] = None
    dtype: str = "bf16"


class _Out(msgspec.Struct):
    source_path: str
    source_ref: str
    text_encoder_path: str
    text_encoder_ref: str


def _train(ctx, payload: _TrainIn) -> _Out:
    return _Out(
        source_path=str(ctx.source_path or ""),
        source_ref=str((ctx.source or {}).get("ref") or ""),
        text_encoder_path=str(ctx.text_encoder_path or ""),
        text_encoder_ref=str((ctx.text_encoder or {}).get("ref") or ""),
    )


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="video-lora-train", method=_train, kind="training",
        payload_type=_TrainIn, output_mode="single",
    )


class _Harness:
    """Returns a distinct local path per ref, so tests can assert `source`
    and `text_encoder` land in genuinely separate locations."""

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
            out = self.tmp_path / ref.replace("/", "_").replace(":", "_")
            out.mkdir(parents=True, exist_ok=True)
            return out

        self.executor.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]

    async def run(self, payload: _TrainIn) -> pb.JobResult:
        await self.executor.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="video-lora-train",
            input_payload=msgspec.msgpack.encode(payload)))
        job = self.executor.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in self.sent if m.WhichOneof("msg") == "job_result"]
        assert results, f"no job_result; sent={self.sent}"
        return results[-1]


def test_text_encoder_materialized_independent_of_source(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(
            source=SourceRepo(ref="acme/ltx2-dit:prod"),
            text_encoder=SourceRepo(ref="google/gemma-3-12b"),
        ))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_ref == "acme/ltx2-dit:prod"
        assert out.text_encoder_ref == "google/gemma-3-12b"
        assert out.source_path and out.text_encoder_path
        assert out.source_path != out.text_encoder_path
        assert set(h.ensured) == {"acme/ltx2-dit:prod", "google/gemma-3-12b"}

    asyncio.run(_run())


def test_missing_text_encoder_is_noop(tmp_path) -> None:
    """The common case: every existing endpoint that never declares
    `text_encoder` sees byte-for-byte unchanged behavior — no ensure_local
    call for it, ctx.text_encoder_path stays empty."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(source=SourceRepo(ref="acme/ltx2-dit:prod")))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path
        assert out.text_encoder_path == ""
        assert out.text_encoder_ref == ""
        assert h.ensured == ["acme/ltx2-dit:prod"]

    asyncio.run(_run())


def test_missing_both_reserved_fields_is_noop(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn())
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path == ""
        assert out.text_encoder_path == ""
        assert h.ensured == []

    asyncio.run(_run())


def test_empty_text_encoder_ref_is_invalid(tmp_path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(
            source=SourceRepo(ref="acme/ltx2-dit:prod"),
            text_encoder=SourceRepo(ref="  "),
        ))
        assert res.status == pb.JOB_STATUS_INVALID

    asyncio.run(_run())


def test_text_encoder_download_failure_classifies_like_source(tmp_path) -> None:
    from gen_worker.api.errors import RetryableError

    async def _run() -> None:
        h = _Harness(tmp_path, fail_with=RetryableError("snapshot not provided"))
        res = await h.run(_TrainIn(
            source=SourceRepo(ref="acme/ltx2-dit:prod"),
            text_encoder=SourceRepo(ref="google/gemma-3-12b"),
        ))
        assert res.status == pb.JOB_STATUS_RETRYABLE

    asyncio.run(_run())


def test_inference_kind_ignores_reserved_text_encoder(tmp_path) -> None:
    class _InfOut(msgspec.Struct):
        ok: bool

    def _infer(ctx, payload: _TrainIn) -> _InfOut:
        return _InfOut(ok=not hasattr(ctx, "text_encoder_path"))

    spec = EndpointSpec(
        name="infer", method=_infer, kind="inference",
        payload_type=_TrainIn, output_mode="single",
    )

    async def _run() -> None:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        ex = Executor([spec], _send)

        async def _boom(ref, snapshot=None, *, binding=None) -> Path:
            raise AssertionError("inference must not materialize text_encoder")

        ex.store.ensure_local = _boom  # type: ignore[method-assign]
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="infer",
            input_payload=msgspec.msgpack.encode(
                _TrainIn(text_encoder=SourceRepo(ref="google/gemma-3-12b")))))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert results[-1].status == pb.JOB_STATUS_OK

    asyncio.run(_run())
