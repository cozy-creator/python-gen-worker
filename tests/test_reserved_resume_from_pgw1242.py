"""pgw#1242/te#185: the fifth reserved repo field, `resume_from`.

THE HOLE IT CLOSES. `ctx.save_checkpoint` has always published a training
checkpoint to the job workspace, and there was no way to hand one BACK to a
handler: a model artifact reaches tenant code only as a reserved payload field,
and the set was hardcoded to source/destination/text_encoder/candidate. None of
them means "the adapter to continue from", and `source` is already the base
model. So the save half of resume worked, the load half had no door, and a
multi-hour training run restarted from zero on pod loss —
`image_lora_finetuner`'s "Resume v1 = clean restart" is that gap, not a
preference.

Shape copied verbatim from pgw#684's `candidate`, on the seam pgw#594
generalized (`_materialize_source` takes `field_name` + a `set_path` callback),
per pgw#684's own ruling: **"fourth hardcoded name NOW, declarative NEXT."**
pgw#690 remains the declarative successor and is explicitly not this change.

Absent field is a no-op: every existing endpoint is byte-for-byte unchanged.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

import msgspec

from gen_worker.api.types import SourceRepo
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _TrainIn(msgspec.Struct):
    source: Optional[SourceRepo] = None
    text_encoder: Optional[SourceRepo] = None
    candidate: Optional[SourceRepo] = None
    resume_from: Optional[SourceRepo] = None
    steps: int = 20


class _Out(msgspec.Struct):
    source_path: str
    text_encoder_path: str
    candidate_path: str
    resume_from_path: str
    resume_from_ref: str


def _train(ctx: Any, payload: _TrainIn) -> _Out:
    return _Out(
        source_path=str(ctx.source_path or ""),
        text_encoder_path=str(ctx.text_encoder_path or ""),
        candidate_path=str(ctx.candidate_path or ""),
        resume_from_path=str(ctx.resume_from_path or ""),
        resume_from_ref=str((ctx.resume_from or {}).get("ref") or ""),
    )


def _spec(kind: str = "training") -> EndpointSpec:
    return EndpointSpec(
        name="h3-trainer", method=_train, kind=kind,
        payload_type=_TrainIn, output_mode="single",
    )


class _Harness:
    """A distinct local path per ref, so each reserved field is provably
    landing somewhere of its own."""

    def __init__(
        self, tmp_path: Path, *, kind: str = "training",
        fail_with: Optional[BaseException] = None,
    ) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.ensured: List[str] = []
        self.tmp_path = tmp_path
        self.fail_with = fail_with

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        self.executor = Executor([_spec(kind)], _send)

        async def _fake_ensure_local(
            ref: str, snapshot: Any = None, *, binding: Any = None
        ) -> Path:
            self.ensured.append(ref)
            if self.fail_with is not None:
                raise self.fail_with
            out = self.tmp_path / ref.replace("/", "_").replace(":", "_")
            out.mkdir(parents=True, exist_ok=True)
            return out

        self.executor.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]

    async def run(self, payload: _TrainIn) -> pb.JobResult:
        await self.executor.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="h3-trainer",
            input_payload=msgspec.msgpack.encode(payload)))
        job = self.executor.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [
            m.job_result for m in self.sent if m.WhichOneof("msg") == "job_result"
        ]
        assert results, f"no job_result; sent={self.sent}"
        return results[-1]


def test_resume_from_materializes_for_a_training_endpoint(tmp_path: Path) -> None:
    """THE POINT: a training job names a published checkpoint and the handler
    gets a local path to it. Before this, `ctx.resume_from_path` did not exist
    and there was no reserved name that could carry one."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(
            source=SourceRepo(ref="acme/h3-dit-base"),
            resume_from=SourceRepo(ref="acme/h3-lora-step-250"),
        ))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.resume_from_path, "the checkpoint was named and never materialized"
        assert out.resume_from_ref == "acme/h3-lora-step-250"
        # It is its OWN artifact, not a decoration on the base model.
        assert out.resume_from_path != out.source_path
        assert set(h.ensured) == {"acme/h3-dit-base", "acme/h3-lora-step-250"}

    asyncio.run(_run())


def test_all_FIVE_reserved_inputs_coexist(tmp_path: Path) -> None:
    """A fifth name must not disturb the other four."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(
            source=SourceRepo(ref="acme/dit-base"),
            text_encoder=SourceRepo(ref="google/gemma-3-12b"),
            candidate=SourceRepo(ref="acme/dit-candidate"),
            resume_from=SourceRepo(ref="acme/lora-step-250"),
        ))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        paths = {out.source_path, out.text_encoder_path,
                 out.candidate_path, out.resume_from_path}
        assert "" not in paths
        assert len(paths) == 4, f"reserved paths collided: {out}"
        assert set(h.ensured) == {
            "acme/dit-base", "google/gemma-3-12b",
            "acme/dit-candidate", "acme/lora-step-250",
        }

    asyncio.run(_run())


def test_missing_resume_from_is_noop(tmp_path: Path) -> None:
    """The common case, and the whole safety argument for a hardcoded name:
    every existing endpoint that never declares `resume_from` is unchanged —
    no extra ensure_local, and the path stays empty."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(source=SourceRepo(ref="acme/dit-base")))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path
        assert out.resume_from_path == ""
        assert out.resume_from_ref == ""
        assert h.ensured == ["acme/dit-base"]

    asyncio.run(_run())


def test_resume_from_alone_without_source(tmp_path: Path) -> None:
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(resume_from=SourceRepo(ref="acme/lora-step-250")))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path == ""
        assert out.resume_from_path
        assert h.ensured == ["acme/lora-step-250"]

    asyncio.run(_run())


def test_empty_resume_from_ref_is_invalid_and_names_the_right_field(tmp_path: Path) -> None:
    """A caller who asks to resume from nothing must be told which field is
    wrong — a message naming `source` would send them to the base model."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_TrainIn(
            source=SourceRepo(ref="acme/dit-base"),
            resume_from=SourceRepo(ref="  "),
        ))
        assert res.status == pb.JOB_STATUS_INVALID
        assert "payload.resume_from.ref" in res.safe_message

    asyncio.run(_run())


def test_resume_from_download_failure_classifies_like_source(tmp_path: Path) -> None:
    """A checkpoint that cannot be fetched must be RETRYABLE, not a silent
    fresh start — restarting a paid continuation from zero is the failure this
    whole field exists to prevent."""
    async def _run() -> None:
        from gen_worker.api.errors import RetryableError

        h = _Harness(tmp_path, fail_with=RetryableError("snapshot not provided"))
        res = await h.run(_TrainIn(resume_from=SourceRepo(ref="acme/lora-step-250")))
        assert res.status == pb.JOB_STATUS_RETRYABLE

    asyncio.run(_run())


def test_inference_kind_ignores_reserved_resume_from(tmp_path: Path) -> None:
    """Reserved repo fields are producer-only."""
    class _InfOut(msgspec.Struct):
        has_resume_surface: bool

    def _infer(ctx: Any, payload: _TrainIn) -> _InfOut:
        return _InfOut(has_resume_surface=hasattr(ctx, "resume_from_path"))

    spec = EndpointSpec(
        name="infer", method=_infer, kind="inference",
        payload_type=_TrainIn, output_mode="single",
    )

    async def _run() -> None:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        ex = Executor([spec], _send)

        async def _boom(ref: str, snapshot: Any = None, *, binding: Any = None) -> Path:
            raise AssertionError("inference must not materialize resume_from")

        ex.store.ensure_local = _boom  # type: ignore[method-assign]
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="infer",
            input_payload=msgspec.msgpack.encode(
                _TrainIn(resume_from=SourceRepo(ref="acme/lora-step-250")))))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert results[-1].status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(results[-1].inline, type=_InfOut)
        assert out.has_resume_surface is False

    asyncio.run(_run())


def test_producer_ctx_resume_from_state_is_independent() -> None:
    """Setting one reserved path never moves another."""
    from gen_worker.request_context import JobContext

    ctx = JobContext(
        request_id="r1",
        source_info={"ref": "acme/dit-base"},
        candidate_info={"ref": "acme/dit-candidate"},
        resume_from_info={"ref": "acme/lora-step-250"},
    )
    assert ctx.resume_from == {"ref": "acme/lora-step-250"}
    assert ctx.resume_from_path is None
    ctx._set_resume_from_path("/models/lora-step-250")
    assert ctx.resume_from_path == "/models/lora-step-250"
    assert ctx.source_path is None and ctx.candidate_path is None
    ctx._set_candidate_path("/models/candidate")
    assert ctx.resume_from_path == "/models/lora-step-250"
    # The accessor hands out a copy, never the live dict.
    ctx.resume_from["ref"] = "mutated"
    assert ctx.resume_from == {"ref": "acme/lora-step-250"}
