"""pgw#684/te#121: the fourth reserved repo field, `candidate`.

A producer payload declaring the reserved ``candidate`` struct (same
``SourceRepo`` type as ``source``) gets a materialized local snapshot at
``ctx.candidate_path`` — independent of, and never clobbering, ``source`` or
``text_encoder``. That is what lets a two-ref quality eval point its candidate
arm at one of OUR OWN hub repos instead of only a public HF/Civitai coordinate.

Absent field is a no-op: every existing endpoint that never declares
``candidate`` sees byte-for-byte unchanged behavior.

Also re-covers the three-name reserved contract end to end — pgw#594's own
tests were swept in th#960/pgw#609 Phase 3b (`0b437aa`), leaving reserved-repo
materialization with no coverage at all in tests/.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import msgspec

from gen_worker.api.types import SourceRepo
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _EvalIn(msgspec.Struct):
    source: Optional[SourceRepo] = None
    text_encoder: Optional[SourceRepo] = None
    candidate: Optional[SourceRepo] = None
    steps: int = 20


class _Out(msgspec.Struct):
    source_path: str
    source_ref: str
    text_encoder_path: str
    candidate_path: str
    candidate_ref: str


def _evaluate(ctx, payload: _EvalIn) -> _Out:
    return _Out(
        source_path=str(ctx.source_path or ""),
        source_ref=str((ctx.source or {}).get("ref") or ""),
        text_encoder_path=str(ctx.text_encoder_path or ""),
        candidate_path=str(ctx.candidate_path or ""),
        candidate_ref=str((ctx.candidate or {}).get("ref") or ""),
    )


def _spec(kind: str = "conversion") -> EndpointSpec:
    return EndpointSpec(
        name="external-eval", method=_evaluate, kind=kind,
        payload_type=_EvalIn, output_mode="single",
    )


class _Harness:
    """Returns a distinct local path per ref, so the tests can assert each
    reserved field lands in a genuinely separate location."""

    def __init__(
        self, tmp_path: Path, *, kind: str = "conversion",
        fail_with: Optional[BaseException] = None,
    ) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.ensured: List[str] = []
        self.tmp_path = tmp_path
        self.fail_with = fail_with

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        self.executor = Executor([_spec(kind)], _send)

        async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
            self.ensured.append(ref)
            if self.fail_with is not None:
                raise self.fail_with
            out = self.tmp_path / ref.replace("/", "_").replace(":", "_")
            out.mkdir(parents=True, exist_ok=True)
            return out

        self.executor.store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]

    async def run(self, payload: _EvalIn) -> pb.JobResult:
        await self.executor.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="external-eval",
            input_payload=msgspec.msgpack.encode(payload)))
        job = self.executor.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [
            m.job_result for m in self.sent if m.WhichOneof("msg") == "job_result"
        ]
        assert results, f"no job_result; sent={self.sent}"
        return results[-1]


def test_all_three_reserved_inputs_coexist(tmp_path) -> None:
    """A fourth name must not disturb the other three: source, text_encoder
    and candidate each land in their own path in the same job, and each ref
    surfaces on its own ctx slot."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_EvalIn(
            source=SourceRepo(ref="acme/dit-base"),
            text_encoder=SourceRepo(ref="google/gemma-3-12b"),
            candidate=SourceRepo(ref="acme/dit-candidate"),
        ))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        paths = {out.source_path, out.text_encoder_path, out.candidate_path}
        assert "" not in paths
        assert len(paths) == 3, f"reserved paths collided: {out}"
        assert out.source_ref == "acme/dit-base"
        assert out.candidate_ref == "acme/dit-candidate"
        assert set(h.ensured) == {
            "acme/dit-base", "google/gemma-3-12b", "acme/dit-candidate",
        }

    asyncio.run(_run())


def test_missing_candidate_is_noop(tmp_path) -> None:
    """The common case: every existing endpoint that never declares
    `candidate` is byte-for-byte unchanged — no extra ensure_local call,
    ctx.candidate_path stays empty."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_EvalIn(source=SourceRepo(ref="acme/dit-base")))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path
        assert out.candidate_path == ""
        assert out.candidate_ref == ""
        assert h.ensured == ["acme/dit-base"]

    asyncio.run(_run())


def test_candidate_alone_without_source(tmp_path) -> None:
    """`candidate` is independent, not a `source` decoration — it materializes
    on its own."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_EvalIn(candidate=SourceRepo(ref="acme/mirrored-svdq")))
        assert res.status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(res.inline, type=_Out)
        assert out.source_path == ""
        assert out.candidate_path
        assert h.ensured == ["acme/mirrored-svdq"]

    asyncio.run(_run())


def test_empty_candidate_ref_is_invalid(tmp_path) -> None:
    """A blank ref is a payload error, and the message names the field the
    tenant actually wrote (not `source`)."""
    async def _run() -> None:
        h = _Harness(tmp_path)
        res = await h.run(_EvalIn(
            source=SourceRepo(ref="acme/dit-base"),
            candidate=SourceRepo(ref="  "),
        ))
        assert res.status == pb.JOB_STATUS_INVALID
        assert "payload.candidate.ref" in res.safe_message

    asyncio.run(_run())


def test_candidate_download_failure_classifies_like_source(tmp_path) -> None:
    async def _run() -> None:
        from gen_worker.api.errors import RetryableError

        h = _Harness(tmp_path, fail_with=RetryableError("snapshot not provided"))
        res = await h.run(_EvalIn(candidate=SourceRepo(ref="acme/mirrored-svdq")))
        assert res.status == pb.JOB_STATUS_RETRYABLE

    asyncio.run(_run())


def test_inference_kind_ignores_reserved_candidate(tmp_path) -> None:
    """Reserved repo fields are producer-only: an inference endpoint that
    happens to embed a SourceRepo never triggers materialization."""
    class _InfOut(msgspec.Struct):
        has_candidate_surface: bool

    def _infer(ctx, payload: _EvalIn) -> _InfOut:
        return _InfOut(has_candidate_surface=hasattr(ctx, "candidate_path"))

    spec = EndpointSpec(
        name="infer", method=_infer, kind="inference",
        payload_type=_EvalIn, output_mode="single",
    )

    async def _run() -> None:
        sent: List[pb.WorkerMessage] = []

        async def _send(msg: pb.WorkerMessage) -> None:
            sent.append(msg)

        ex = Executor([spec], _send)

        async def _boom(ref, snapshot=None, *, binding=None) -> Path:
            raise AssertionError("inference must not materialize candidate")

        ex.store.ensure_local = _boom  # type: ignore[method-assign]
        await ex.handle_run_job(pb.RunJob(
            request_id="r1", attempt=1, function_name="infer",
            input_payload=msgspec.msgpack.encode(
                _EvalIn(candidate=SourceRepo(ref="acme/mirrored-svdq")))))
        job = ex.jobs[("r1", 1)]
        assert job.task is not None
        await job.task
        results = [m.job_result for m in sent if m.WhichOneof("msg") == "job_result"]
        assert results[-1].status == pb.JOB_STATUS_OK
        out = msgspec.msgpack.decode(results[-1].inline, type=_InfOut)
        assert out.has_candidate_surface is False

    asyncio.run(_run())


def test_producer_ctx_candidate_state_is_independent() -> None:
    """Context-level contract: `candidate`/`candidate_path` are their own
    slots, and setting one reserved path never moves another."""
    from gen_worker.request_context import JobContext

    ctx = JobContext(
        request_id="r1",
        source_info={"ref": "acme/dit-base"},
        candidate_info={"ref": "acme/mirrored-svdq"},
    )
    assert ctx.candidate == {"ref": "acme/mirrored-svdq"}
    assert ctx.candidate_path is None
    ctx._set_candidate_path("/models/candidate")
    assert ctx.candidate_path == "/models/candidate"
    assert ctx.source_path is None
    ctx._set_source_path("/models/dit-base")
    assert ctx.source_path == "/models/dit-base"
    assert ctx.candidate_path == "/models/candidate"
    # The accessor hands out a copy, never the live dict.
    ctx.candidate["ref"] = "mutated"
    assert ctx.candidate == {"ref": "acme/mirrored-svdq"}
