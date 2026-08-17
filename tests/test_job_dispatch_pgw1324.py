"""pgw#1324 (JOBS program): a hub-dispatched `@job` REACHES ITS BODY.

The defect this closes, measured at `origin/master` `45711990` and in the
published `gen_worker-0.119.0` wheel: `execute_job`'s only caller was
`cli/job.py`, `Executor.specs` was `Dict[str, EndpointSpec]`, and the one wire
head resolved `RunJob.function_name` against that table alone. Every job te#218
migrated (27 conversion jobs + 2 trainers) was publishable, submittable and
UNRUNNABLE — a dispatch naming one resolved to nothing and came back
`unknown function`, indistinguishable from a typo.

Each section names the one-line edit that turns it RED:

1. **THE HEADLINE.** A real `RunJob` frame naming a JOB, through the real
   `Executor.handle_run_job`, reaches the body and returns a `JobResult` whose
   inline bytes decode as the job's declared result struct. RED by resolving
   `function_name` against `self.specs` alone in `_admit_dispatch` — which is
   exactly what master did.
2. **Two tables, one head, and a refusal that can tell them apart.** A name in
   NEITHER table is refused INVALID naming both inventories, and a name in BOTH
   is refused at boot — a dispatch carries a name and no intent kind, so one
   name cannot mean two things. RED by dropping the job inventory from the
   refusal message, or by allowing the collision.
3. **The declaration is the RELEASE's.** `execute_job._stamp_declaration` — not
   the dispatch head — projects `publishes`/`emits_media` onto the context, so
   an undeclared job's `save_checkpoint` refuses typed before a byte moves.
   RED by having the head build the context with `publishes=True`.
4. **The deadline is CARRIED.** `RunJob.timeout_ms` reaches
   `JobDispatch.deadline_s`. RED by dropping the conversion.
5. **GPU-vs-plan rung.** A `Resources(gpu=True)` job holds its group's GPU
   permit for the whole run; a `plan-*` `Resources(vcpus=N)` job holds none, so
   a CPU plan job cannot queue behind a card. RED by making the permit
   unconditional (or never taken).
6. **`recycle_child` is HONOURED, not ignored.** The run-once lifecycle is a
   fresh child per job: after the terminal result the compute child asks to be
   recycled. RED by deleting the `_recycle_after_job` call, or by clearing
   `recycle_child` on the outcome — the field then goes back to being one
   nobody reads.
7. **A jobs-only package BOOTS.** te#218's packages carry no `@endpoint` at
   all. RED by restoring `if not specs: raise` in `worker.py`.
8. **A body failure is TERMINAL, not infra.** The hub retries infra faults
   only; a typed exception from the body must not buy a second GPU boot.

Everything drives production code: the real `Executor`, the real registry
walk, the real `jobs.execute_job` harness, the real `JobContext`. The doubles
are a send-sink and a payload.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import pytest

from gen_worker import JobContext, Resources, endpoint, job
from gen_worker import procsplit
from gen_worker.api.errors import PublishNotDeclaredError
from gen_worker.executor import Executor
from gen_worker.jobs import JobDispatch, execute_job as _real_execute_job
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec, extract_job_spec


class In(msgspec.Struct):
    rung: str = "w8a8"


class Out(msgspec.Struct):
    rung: str
    saw: str = ""


# ---- the jobs under dispatch ----------------------------------------------
# Declared with the REAL decorator and extracted with the REAL registry walk,
# so what the executor holds is what a published release holds.

_SEEN: Dict[str, Any] = {}


@job(name="plan-h3-svdq", resources=Resources(vcpus=4))
def plan_h3_svdq(ctx: JobContext, payload: In) -> Out:
    _SEEN["ctx_class"] = type(ctx).__name__
    _SEEN["publishes"] = ctx.publishes
    _SEEN["emits_media"] = ctx.emits_media
    _SEEN["request_id"] = ctx.request_id
    ctx.progress(position=1, total=2, phase="plan")
    ctx.progress(position=2, total=2, phase="plan")
    return Out(rung=payload.rung, saw="planned")


@job(name="clone-huggingface", resources=Resources(gpu=True), publishes=True)
def clone_huggingface(ctx: JobContext, payload: In) -> Out:
    _SEEN["publishes"] = ctx.publishes
    _SEEN["gpu_permit_held"] = _permit_probe()
    return Out(rung=payload.rung, saw="cloned")


@job(name="score-benchmark", resources=Resources(vcpus=2), emits_media=True)
def score_benchmark(ctx: JobContext, payload: In) -> Out:
    _SEEN["emits_media"] = ctx.emits_media
    # Undeclared publisher surface: the refusal is the SDK's, before a byte
    # moves, because the hub minted no write grant for this release.
    weights = ctx.mktemp() / "adapter.safetensors"
    weights.write_bytes(b"\x00" * 8)
    try:
        ctx.save_checkpoint("model.safetensors", weights)
    except PublishNotDeclaredError as exc:
        _SEEN["publish_refusal"] = type(exc).__name__
    return Out(rung=payload.rung, saw="scored")


@job(name="explode", resources=Resources(vcpus=1))
def explode(ctx: JobContext, payload: In) -> Out:
    raise RuntimeError("the recipe is wrong and re-running it is money burnt")


_PERMIT_PROBE: List[Any] = []


def _permit_probe() -> Optional[bool]:
    return _PERMIT_PROBE[0].locked() if _PERMIT_PROBE else None


def _job_specs() -> List[Any]:
    return [
        extract_job_spec(fn)
        for fn in (plan_h3_svdq, clone_huggingface, score_benchmark, explode)
    ]


# ---- an endpoint, so the two tables are genuinely both populated -----------


def _serve(ctx: Any, payload: In) -> Out:
    return Out(rung=payload.rung, saw="served")


def _endpoint_spec() -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_serve, kind="inference",
        payload_type=In, output_mode="single",
    )


# ---- harness ---------------------------------------------------------------


class _Harness:
    """A real Executor with BOTH tables populated, and a send sink."""

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.exits: List[int] = []
        self.ex = Executor(
            [_endpoint_spec()], self._send, jobs=_job_specs())
        self.ex._process_exit = self.exits.append
        _PERMIT_PROBE[:] = [self.ex._gpu_permit_for_group(0)]

    async def _send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)

    def frame(self, name: str, **kw: Any) -> pb.RunJob:
        return pb.RunJob(
            request_id=kw.pop("request_id", "job-uuid-1"),
            attempt=int(kw.pop("attempt", 0)),
            function_name=name,
            input_payload=msgspec.msgpack.encode(In(rung=kw.pop("rung", "w8a8"))),
            **kw,
        )

    async def dispatch(self, name: str, **kw: Any) -> pb.JobResult:
        run = self.frame(name, **kw)
        await self.ex.handle_run_job(run)
        record = self.ex.jobs.get((run.request_id, run.attempt))
        if record is not None and record.task is not None:
            await record.task
        return self.results()[-1]

    def results(self) -> List[pb.JobResult]:
        return [m.job_result for m in self.sent
                if m.WhichOneof("msg") == "job_result"]

    def accepted(self) -> List[pb.JobAccepted]:
        return [m.job_accepted for m in self.sent
                if m.WhichOneof("msg") == "job_accepted"]

    def progress(self) -> List[pb.JobProgress]:
        return [m.job_progress for m in self.sent
                if m.WhichOneof("msg") == "job_progress"]


@pytest.fixture(autouse=True)
def _clean() -> Any:
    _SEEN.clear()
    yield
    _SEEN.clear()
    _PERMIT_PROBE.clear()


# ---- 1. THE HEADLINE: a dispatched job reaches its body --------------------


def test_a_dispatched_job_reaches_its_body_and_returns_a_result() -> None:
    """The whole issue, in one dispatch: master answered `unknown function`."""

    async def scenario() -> Tuple[pb.JobResult, _Harness]:
        h = _Harness()
        result = await h.dispatch("plan-h3-svdq", rung="fp8")
        return result, h

    result, h = asyncio.run(scenario())

    assert result.status == pb.JOB_STATUS_OK, result.safe_message
    assert msgspec.msgpack.decode(result.inline, type=Out) == Out(
        rung="fp8", saw="planned")
    # The BODY ran — registration alone is the shape that let this ship.
    assert _SEEN["ctx_class"] == "JobContext"
    assert _SEEN["request_id"] == "job-uuid-1"
    # Accepted before result, and the body's positions rode the wire.
    assert [a.request_id for a in h.accepted()] == ["job-uuid-1"]
    assert len(h.progress()) >= 2


def test_an_endpoint_still_dispatches_through_the_same_head() -> None:
    """Two tables, one head: the endpoint path is untouched."""

    async def scenario() -> pb.JobResult:
        h = _Harness()
        return await h.dispatch("generate")

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_OK
    assert msgspec.msgpack.decode(result.inline, type=Out).saw == "served"


# ---- 2. a name in NEITHER table is distinguishable -------------------------


def test_a_name_in_neither_table_names_both_inventories() -> None:
    async def scenario() -> pb.JobResult:
        h = _Harness()
        return await h.dispatch("no-such-thing")

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_INVALID
    msg = result.safe_message
    assert "no-such-thing" in msg
    # BOTH inventories, so "unknown name" and "known job name" stop looking
    # identical — which is how this stayed quiet for a whole migration.
    assert "generate" in msg and "plan-h3-svdq" in msg


def test_one_name_cannot_be_both_an_endpoint_and_a_job() -> None:
    """The wire carries a name and no intent kind, so a colliding name is a
    dispatch nobody can resolve. Refused at BOOT, where it is still fixable —
    not at 3am on a rented pod."""

    async def send(msg: pb.WorkerMessage) -> None:  # pragma: no cover
        raise AssertionError("nothing should be sent")

    collide = EndpointSpec(
        name="plan-h3-svdq", method=_serve, kind="inference",
        payload_type=In, output_mode="single",
    )
    with pytest.raises(ValueError, match="BOTH an @endpoint and a @job"):
        Executor([collide], send, jobs=_job_specs())


# ---- 3. the declaration is the RELEASE's -----------------------------------


def test_the_publish_declaration_is_stamped_from_the_spec_not_the_head() -> None:
    async def scenario() -> Tuple[pb.JobResult, pb.JobResult]:
        h = _Harness()
        declared = await h.dispatch("clone-huggingface", request_id="job-a")
        seen_declared = dict(_SEEN)
        undeclared = await h.dispatch("score-benchmark", request_id="job-b")
        _SEEN["declared_publishes"] = seen_declared["publishes"]
        return declared, undeclared

    declared, undeclared = asyncio.run(scenario())
    assert declared.status == pb.JOB_STATUS_OK
    assert undeclared.status == pb.JOB_STATUS_OK
    assert _SEEN["declared_publishes"] is True
    # The undeclared job saw publishes=False and was refused TYPED at the SDK.
    assert _SEEN["publish_refusal"] == "PublishNotDeclaredError"
    assert _SEEN["emits_media"] is True


def test_a_job_that_declares_nothing_gets_no_media_grant_either() -> None:
    async def scenario() -> pb.JobResult:
        h = _Harness()
        return await h.dispatch("plan-h3-svdq")

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_OK
    assert _SEEN["publishes"] is False
    assert _SEEN["emits_media"] is False


# ---- 4. the deadline is carried --------------------------------------------


def test_the_wire_deadline_reaches_the_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`RunJob.timeout_ms` is the hub's per-attempt deadline. It is RECORDED
    on the dispatch (the wall cap is hub-issued through the provider API), so
    losing it here loses the only statement of it the pod ever sees."""
    seen: List[JobDispatch] = []

    def spy(dispatch: JobDispatch, **kw: Any) -> Any:
        seen.append(dispatch)
        return _real_execute_job(dispatch, **kw)

    monkeypatch.setattr("gen_worker.executor.execute_job", spy)

    async def scenario() -> pb.JobResult:
        h = _Harness()
        return await h.dispatch("plan-h3-svdq", timeout_ms=1_800_000)

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_OK
    assert len(seen) == 1
    assert seen[0].deadline_s == 1800.0
    assert seen[0].job_name == "plan-h3-svdq"
    assert seen[0].job_id == "job-uuid-1"


# ---- 5. GPU-vs-plan rung ---------------------------------------------------


def test_a_gpu_job_holds_the_permit_and_a_plan_job_does_not() -> None:
    async def scenario() -> None:
        h = _Harness()
        await h.dispatch("clone-huggingface", request_id="gpu-job")
        gpu_held = _SEEN["gpu_permit_held"]
        await h.dispatch("plan-h3-svdq", request_id="cpu-job")
        _SEEN["gpu_held"] = gpu_held

    asyncio.run(scenario())
    assert _SEEN["gpu_held"] is True
    # And the permit is released again — a plan job never queues behind a card.
    assert _PERMIT_PROBE[0].locked() is False


def test_a_plan_job_runs_while_the_gpu_permit_is_taken() -> None:
    """The rung, stated as behaviour: a CPU plan job does not wait for a card
    another holder is using. Fail-fast-before-renting is a property of the
    inventory only if the runtime honours it."""

    async def scenario() -> pb.JobResult:
        h = _Harness()
        permit = h.ex._gpu_permit_for_group(0)
        await permit.acquire()
        try:
            return await asyncio.wait_for(h.dispatch("plan-h3-svdq"), timeout=20)
        finally:
            permit.release()

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_OK


# ---- 6. recycle_child is honoured ------------------------------------------


def test_the_child_is_recycled_after_a_job_and_not_after_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(procsplit, "is_compute_child", lambda: True)

    async def scenario() -> Tuple[List[int], List[int]]:
        h = _Harness()
        await h.dispatch("plan-h3-svdq", request_id="job-x")
        after_job = list(h.exits)
        await h.dispatch("generate", request_id="req-x")
        return after_job, list(h.exits)

    after_job, after_request = asyncio.run(scenario())
    assert after_job == [procsplit.EXIT_JOB_RECYCLE]
    # A serving REQUEST recycles nothing: the run-once lifecycle is the job's.
    assert after_request == after_job


def test_the_recycle_is_read_off_the_outcome_not_assumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`recycle_child` was a field nobody read. It is load-bearing only if
    clearing it actually keeps the process — otherwise the recycle is a habit
    that happens to agree with the contract."""
    monkeypatch.setattr(procsplit, "is_compute_child", lambda: True)

    def no_recycle(dispatch: JobDispatch, **kw: Any) -> Any:
        outcome = _real_execute_job(dispatch, **kw)
        return dataclasses.replace(outcome, recycle_child=False)

    monkeypatch.setattr("gen_worker.executor.execute_job", no_recycle)

    async def scenario() -> Tuple[pb.JobResult, List[int]]:
        h = _Harness()
        result = await h.dispatch("plan-h3-svdq")
        return result, list(h.exits)

    result, exits = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_OK
    assert exits == []


def test_a_worker_that_is_not_a_compute_child_does_not_kill_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`gen-worker serve` has no control parent to respawn it, so the recycle
    is REPORTED rather than executed — a process exit with nobody to notice is
    a dead pod, not a fresh child."""
    monkeypatch.setattr(procsplit, "is_compute_child", lambda: False)

    async def scenario() -> List[int]:
        h = _Harness()
        await h.dispatch("plan-h3-svdq")
        return list(h.exits)

    assert asyncio.run(scenario()) == []


# ---- 7. a jobs-only package boots ------------------------------------------


def test_a_package_with_jobs_and_no_endpoints_boots() -> None:
    """te#218 left every conversion package jobs-only. A worker that refuses
    to boot on one cannot run a single migrated job."""
    from gen_worker.config import load_settings
    from gen_worker.worker import Worker

    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id="pgw1324-jobs-only",
        worker_jwt="",
    )
    worker = Worker(settings, ["harness.job_pkg_pgw1324"])
    assert sorted(worker.executor.job_specs) == [
        "bake-h3-modulation", "plan-h3-svdq"]
    assert worker.executor.specs == {}


def test_a_package_with_neither_still_refuses() -> None:
    from gen_worker.config import load_settings
    from gen_worker.worker import Worker

    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id="pgw1324-empty",
        worker_jwt="",
    )
    with pytest.raises(ValueError, match="no @endpoint classes and no @job"):
        Worker(settings, ["harness.blob_host"])


# ---- 8. a body failure is terminal, not infra ------------------------------


def test_a_body_exception_is_terminal_never_retryable() -> None:
    """`max_retries` counts INFRA faults only — re-running a deterministic
    failure is money burnt (the jobs-system ruling, k8s semantics)."""

    async def scenario() -> pb.JobResult:
        h = _Harness()
        return await h.dispatch("explode")

    result = asyncio.run(scenario())
    assert result.status == pb.JOB_STATUS_FATAL
    assert result.status != pb.JOB_STATUS_RETRYABLE
    assert "recipe is wrong" in result.safe_message
