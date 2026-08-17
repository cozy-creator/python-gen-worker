"""pgw#1336 / pgw#1307 arm (8): the RunJob `compat-*` minter is GONE.

The arm's own blocker, quoted from the code it wanted to delete:
`ensure_local_intent`'s docstring said these intents *"cover legacy operations
that protocol v5 cannot yet own directly, such as a RunJob (the wire lacks a
job intent kind/owner field)"*. th#2052 grew that field — `DesiredIntentKind
.DESIRED_INTENT_KIND_RUN_JOB` plus `RunJob.intent_kind|intent_id|goal_id|
phase_budget_s`, fields 18-21, all additive — so a job dispatch now arrives
owning a hub-authored carrier and the fabrication has nothing left to do.

**What this file proves, and what each row goes RED on:**

1. **A JOB reports against the HUB's carrier.** The dispatch's `intent_id` /
   `goal_id` are what the registry holds and what every transition names; no
   `compat-*` id exists anywhere in the registry for it. RED by restoring
   `ensure_local_intent("job", ...)` in `Executor._dispatch_intent`.
2. **A dispatch WITHOUT the kind is DISTINGUISHABLE from one carrying it**,
   which is the whole point of the field: the same name, the same worker, two
   different outcomes. RED by routing on the name again (the pgw#1324 shape).
3. **A RUN_JOB frame with no carrier is REFUSED, not papered over.** The old
   code could always invent an id; `adopt_dispatch_intent` cannot and raises.
   RED by falling back to the compat minter on an empty `intent_id`.
4. **The SERVED-REQUEST half SURVIVES, deliberately.** th#2052 gave a carrier
   to jobs only. A served request is not an intent on this protocol — it is
   what an intent gets BLOCKED on (`IntentState.blocker_request`) — so the
   worker-local `compat-job-*` carrier for a request stays. RED by deleting
   that arm: the request path then has no reportable intent at all.
5. **`ensure_intent`'s compat arm SURVIVES**, for its own separate reason:
   re-verifying converged command work needs a REPORTABLE carrier, and without
   one `wait_idle` trips the unreported-wait timeout and drives a HEALTHY
   worker to `WORKER_PHASE_ERROR`. pgw#1307 arm (8) says explicitly that
   deleting both is the failure mode to avoid. RED by deleting it.
6. **`phase_budget_s` is the OPERATOR's number when stated.** A non-zero
   budget replaces the wheel's compiled `DEFAULT_PHASE_BUDGET_S`; zero keeps
   it. RED by dropping the conversion.

Everything drives production code: the real `Executor`, the real
`IntentRegistry`, the real registry walk and the real `jobs.execute_job`. The
doubles are a send-sink and a payload.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import pytest

from gen_worker import JobContext, Resources, job
from gen_worker.executor import Executor
from gen_worker.jobs import DEFAULT_PHASE_BUDGET_S, JobDispatch
from gen_worker.jobs import execute_job as _real_execute_job
from gen_worker.lifecycle_intents import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec, extract_job_spec

JOB_ID = "11111111-2222-3333-4444-555555555555"
#: Spelled exactly as tensorhub's `JobGoalID` / `JobIntentID` spell them
#: (`internal/orchestrator/grpc/job_dispatch_th2050.go`). If the hub's grammar
#: changes, this file is where the worker notices.
HUB_GOAL_ID = f"job-{JOB_ID}"
HUB_INTENT_ID = f"job-{JOB_ID}-0"


class In(msgspec.Struct):
    rung: str = "w8a8"


class Out(msgspec.Struct):
    rung: str


_SEEN: Dict[str, Any] = {}


@job(name="plan-h3-svdq", resources=Resources(vcpus=4))
def plan_h3_svdq(ctx: JobContext, payload: In) -> Out:
    _SEEN["request_id"] = ctx.request_id
    return Out(rung=payload.rung)


def _serve(ctx: Any, payload: In) -> Out:
    return Out(rung=payload.rung)


def _job_specs() -> List[Any]:
    return [extract_job_spec(plan_h3_svdq)]


def _executor(send: Any) -> Executor:
    return Executor(
        [EndpointSpec(
            name="generate", method=_serve, kind="inference",
            payload_type=In, output_mode="single",
        )],
        send,
        jobs=_job_specs(),
    )


class _Harness:
    """A real Executor with both tables populated and a bound registry."""

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.ex = _executor(self._send)
        self.ex._process_exit = lambda code: None
        self.registry = IntentRegistry("release-1", ["generate"])
        self.ex.bind_intent_registry(self.registry)

    async def _send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)

    def job_frame(self, name: str = "plan-h3-svdq", **kw: Any) -> pb.RunJob:
        """A JOB dispatch, stamped as `JobWire.Dispatch` stamps it hub-side."""
        kw.setdefault("intent_kind", pb.DESIRED_INTENT_KIND_RUN_JOB)
        kw.setdefault("intent_id", HUB_INTENT_ID)
        kw.setdefault("goal_id", HUB_GOAL_ID)
        return self.request_frame(name, **kw)

    def request_frame(self, name: str = "generate", **kw: Any) -> pb.RunJob:
        """A SERVED-REQUEST dispatch: no kind, no carrier — as the hub sends."""
        return pb.RunJob(
            request_id=kw.pop("request_id", JOB_ID),
            attempt=int(kw.pop("attempt", 0)),
            function_name=name,
            input_payload=msgspec.msgpack.encode(In()),
            **kw,
        )

    async def dispatch(self, run: pb.RunJob) -> Optional[pb.JobResult]:
        await self.ex.handle_run_job(run)
        record = self.ex.jobs.get((run.request_id, run.attempt))
        if record is not None and record.task is not None:
            await record.task
        results = self.results()
        return results[-1] if results else None

    def results(self) -> List[pb.JobResult]:
        return [m.job_result for m in self.sent
                if m.WhichOneof("msg") == "job_result"]

    def intent_ids(self) -> List[str]:
        return list(self.registry._intents)


@pytest.fixture(autouse=True)
def _clean() -> Any:
    _SEEN.clear()
    yield
    _SEEN.clear()


# ---- 1. a job reports against the HUB's carrier -----------------------------


def test_a_job_dispatch_reports_against_the_hub_authored_carrier() -> None:
    """The id the worker reports is the id the HUB authored — not a hash of
    (request_id, attempt) the hub has never heard of."""

    async def scenario() -> Tuple[Optional[pb.JobResult], List[str], str]:
        h = _Harness()
        run = h.job_frame()
        result = await h.dispatch(run)
        return result, h.intent_ids(), h.registry._intents[HUB_INTENT_ID].goal_id

    result, intent_ids, goal_id = asyncio.run(scenario())
    assert result is not None and result.status == pb.JOB_STATUS_OK
    assert HUB_INTENT_ID in intent_ids
    assert goal_id == HUB_GOAL_ID
    # THE DELETION, asserted as an absence: nothing in this registry was
    # fabricated for the job. This is the row that reds if the minter comes
    # back, whatever else still passes.
    assert not [i for i in intent_ids if i.startswith("compat-")], intent_ids


def test_an_adopted_carrier_is_never_renamed_on_a_redelivery() -> None:
    """The compat minter appended `-N` when it found the id taken, because the
    id was ITS OWN to choose. This id is the hub's: a redelivery reports under
    the same id or the hub cannot match it to the obligation it opened."""
    registry = IntentRegistry("release-1", [])
    first = registry.adopt_dispatch_intent(HUB_INTENT_ID, HUB_GOAL_ID)
    registry.transition(
        first,
        pb.LIFECYCLE_INTENT_STATUS_FAILED,
        pb.LIFECYCLE_INTENT_STAGE_FINALIZING,
    )
    again = registry.adopt_dispatch_intent(HUB_INTENT_ID, HUB_GOAL_ID)
    assert again == first == HUB_INTENT_ID
    assert list(registry._intents) == [HUB_INTENT_ID]
    # Replaced, not left terminal: a live obligation must be transitionable.
    state = registry._intents[HUB_INTENT_ID]
    assert state.status == pb.LIFECYCLE_INTENT_STATUS_ACCEPTED


# ---- 2. the kind is what routes, and it is observable -----------------------


def test_the_same_name_routes_differently_with_and_without_the_kind() -> None:
    """THE DISTINGUISHABILITY ROW. One name, one worker, two frames that differ
    only in `intent_kind` — and two different outcomes. Before th#2052 both
    frames were the same sentence and the head guessed from its tables."""

    async def scenario() -> Tuple[Optional[pb.JobResult], Optional[pb.JobResult]]:
        declared = _Harness()
        as_job = await declared.dispatch(declared.job_frame("plan-h3-svdq"))
        undeclared = _Harness()
        as_request = await undeclared.dispatch(
            undeclared.request_frame("plan-h3-svdq"))
        return as_job, as_request

    as_job, as_request = asyncio.run(scenario())
    assert as_job is not None and as_job.status == pb.JOB_STATUS_OK
    # The SAME name, without the kind, is a served request — and this release
    # declares no endpoint by that name.
    assert as_request is not None
    assert as_request.status == pb.JOB_STATUS_INVALID
    # And the refusal SAYS which way it crossed, so a submitter is not told to
    # rename a name that is perfectly correct.
    assert "declared in this release as a @job" in as_request.safe_message
    assert "asked for an @endpoint" in as_request.safe_message


def test_an_endpoint_name_dispatched_as_a_job_is_refused_the_other_way() -> None:
    async def scenario() -> Optional[pb.JobResult]:
        h = _Harness()
        return await h.dispatch(h.job_frame("generate"))

    result = asyncio.run(scenario())
    assert result is not None and result.status == pb.JOB_STATUS_INVALID
    assert "declared in this release as an @endpoint" in result.safe_message
    assert "asked for a @job" in result.safe_message


# ---- 3. a RUN_JOB frame with no carrier is refused --------------------------


def test_a_run_job_frame_with_no_carrier_cannot_be_papered_over() -> None:
    """The old code could always invent an id, so a hub bug here was invisible.
    Adoption cannot invent one, and says so."""
    registry = IntentRegistry("release-1", [])
    with pytest.raises(ValueError, match="hub-authored intent id"):
        registry.adopt_dispatch_intent("", HUB_GOAL_ID)
    with pytest.raises(ValueError, match="hub-authored intent id"):
        registry.adopt_dispatch_intent("   ", HUB_GOAL_ID)
    assert list(registry._intents) == []


# ---- 4. the served-request half survives, deliberately ----------------------


def test_a_served_request_still_mints_its_worker_local_carrier() -> None:
    """NOT a leftover. th#2052 gave a carrier to JOBS. A served request has no
    intent kind on this protocol and never had one — it is what an intent gets
    BLOCKED on, not an intent. Deleting this arm leaves the request path with
    no reportable intent, which is what `guard_await` fails closed on."""

    async def scenario() -> List[str]:
        h = _Harness()
        await h.dispatch(h.request_frame("generate"))
        return h.intent_ids()

    intent_ids = asyncio.run(scenario())
    assert [i for i in intent_ids if i.startswith("compat-job-")], intent_ids


# ---- 5. ensure_intent's compat arm survives ---------------------------------


def test_ensure_intent_still_mints_a_carrier_for_uncommanded_work() -> None:
    """pgw#1307 arm (8), verbatim: this twin *"SURVIVES and arm (8) says so
    explicitly"*. Its reason is not the missing wire field — it is that
    re-verifying converged command work needs a reportable carrier, and without
    one `wait_idle` trips the unreported-wait timeout and drives a HEALTHY
    worker to `WORKER_PHASE_ERROR`."""
    registry = IntentRegistry("release-1", ["generate"])
    intent_id = registry.ensure_intent(
        pb.DESIRED_INTENT_KIND_FUNCTION_READY, function_name="generate")
    assert intent_id.startswith("compat-"), intent_id
    assert registry.is_active(intent_id)
    # And it is REPORTABLE — the property the whole arm exists for.
    registry.transition(
        intent_id,
        pb.LIFECYCLE_INTENT_STATUS_RUNNING,
        pb.LIFECYCLE_INTENT_STAGE_VALIDATING,
    )
    assert registry._intents[intent_id].status == pb.LIFECYCLE_INTENT_STATUS_RUNNING


def test_a_blockerless_waiting_report_still_carries_a_deadline() -> None:
    """`_WAITING_DEADLINE_FALLBACK_MS`, re-read against `phase_budget_s` and
    KEPT (pgw#1336). They are not two spellings of one number: this fills a
    REQUIRED wire field on any blockerless WAITING report, whoever authored the
    intent, while `phase_budget_s` is one job's position-advance budget."""
    registry = IntentRegistry("release-1", [])
    intent_id = registry.adopt_dispatch_intent(HUB_INTENT_ID, HUB_GOAL_ID)
    registry.transition(
        intent_id,
        pb.LIFECYCLE_INTENT_STATUS_WAITING,
        pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK,
        reason=pb.LIFECYCLE_WAIT_REASON_SINGLE_FLIGHT_OWNER,
    )
    state = registry._intents[intent_id]
    assert state.deadline_at_unix_ms > 0


# ---- 6. phase_budget_s is the operator's number -----------------------------


def _spy_budget(monkeypatch: pytest.MonkeyPatch) -> List[JobDispatch]:
    seen: List[JobDispatch] = []

    def spy(dispatch: JobDispatch, **kw: Any) -> Any:
        seen.append(dispatch)
        return _real_execute_job(dispatch, **kw)

    monkeypatch.setattr("gen_worker.executor.execute_job", spy)
    return seen


def test_a_stated_phase_budget_replaces_the_wheels_compiled_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One question — "is this job advancing?" — had two numbers: the hub's
    liveness sweep read the operator's `jobs.progress_budget_s` and the worker
    read whatever this wheel was compiled with. The dispatch now states it."""
    seen = _spy_budget(monkeypatch)

    async def scenario() -> None:
        h = _Harness()
        await h.dispatch(h.job_frame(phase_budget_s=90))

    asyncio.run(scenario())
    assert len(seen) == 1
    assert seen[0].phase_budget_s == 90.0
    assert seen[0].phase_budget_s != DEFAULT_PHASE_BUDGET_S


def test_no_stated_budget_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 is "no instruction", not "no budget" — an unbounded phase is exactly
    what made pgw#1287's silent download indistinguishable from a healthy one."""
    seen = _spy_budget(monkeypatch)

    async def scenario() -> None:
        h = _Harness()
        await h.dispatch(h.job_frame())

    asyncio.run(scenario())
    assert len(seen) == 1
    assert seen[0].phase_budget_s == DEFAULT_PHASE_BUDGET_S
