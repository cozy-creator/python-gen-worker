"""pgw#888 (Paul, 2026-08-15): *"a worker should serve-eager if compilation
doesn't work. We want our worker to be as robust as possible, although it
should loudly report when it's performing in a degraded mode."*

The observable this closes, measured on the standing master stack 2026-08-02:
11 `tensorhub.request_events` rows reading
`request.failed / JOB_STATUS_RETRYABLE / required_compile_missing`, all
terminal, all at `assignment_attempt_epoch=5` — five retries burned per
request because the dispatch fence refused a pod whose pipeline, weights and
lane were exactly what the hub picked and whose only defect was that the
PINNED COMPILED GRAPH had gone (de-armed for cause per §4.31, revoked, or
superseded). A missing compiled graph is a speed fact, not a correctness one.

Drives the real production fence — `Executor._validate_required_compile`, the
callable wired as `_JobOrder.fence` and invoked at both intake and the last
pre-execution GPU turn — and the real `activity.emit_event` path through a
bound sink, so the assertion is on the `ActivityUpdate` the hub would bank in
`worker_activity_events`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, cast

import pytest

from gen_worker import Compile, Resources
from gen_worker import activity as activity_mod
from gen_worker import serving_mode as serving_mode_mod
from gen_worker.api.binding import Hub
from gen_worker.api.errors import RetryableError
from gen_worker.executor import Executor, _ClassRecord, _CompileTargetRecord
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


BARE = "acme/wai-illustrious"
PICK = "acme/wai-illustrious@sha256:" + "a1" * 32
SNAPSHOT = "sha256:" + "b2" * 32
INCARNATION = "incarnation-7"
CONTRACT = "contract-digest-1"

#: What the hub pinned.
PINNED_CELL = "cg:sdxl-1024-sm89"
PINNED_DIGEST = "sha256:" + "c3" * 32


class _Payload:
    pass


class _Endpoint:
    def setup(self, checkpoint: str) -> None:  # pragma: no cover - shape only
        pass

    def run(self, ctx, payload):  # pragma: no cover - shape only
        return None


class _LiveJob:
    """The four fields the fence writes and reads on a live job. A real `_Job`
    needs an asyncio loop and a hub transport; the fence touches neither."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.finished = False
        self.served_eager_fallback = False
        self.fallback_reason = ""
        self.pinned_compiled_graph_degrade_reported = False


class _Fence:
    """The Executor surface the dispatch fence reaches, and nothing else."""

    def __init__(self, execution_lane: str, target: _CompileTargetRecord) -> None:
        self._model_resolutions = {BARE: (PICK, "", execution_lane)}
        rec = _ClassRecord(cls=_Endpoint)
        rec.ready = True
        rec.compile_targets[target.incarnation_id] = target
        self._classes: Dict[Any, _ClassRecord] = {_Endpoint: rec}
        self.jobs: Dict[Any, _LiveJob] = {}

    _resolved_mandatory_execution_lane = Executor._resolved_mandatory_execution_lane
    _mandatory_execution_lane_of_bound = Executor._mandatory_execution_lane_of_bound
    _compile_target = Executor._compile_target
    _mark_request_eager_fallback = Executor._mark_request_eager_fallback
    _report_pinned_compiled_graph_unavailable = Executor._report_pinned_compiled_graph_unavailable
    _validate_required_compile = Executor._validate_required_compile
    _setup_slots = staticmethod(Executor._setup_slots)


def _serve(fence: "_Fence", spec: EndpointSpec, run: pb.RunJob) -> None:
    """The production call, through the real unbound method. `_Fence` is a
    surface, not an `Executor` subclass — an Executor needs a hub transport
    and an event loop to construct, and the fence touches neither."""
    Executor._validate_required_compile(cast(Executor, fence), spec, run)


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_Endpoint.run, kind="inference",
        payload_type=_Payload, output_mode="single", cls=_Endpoint,
        attr_name="run",
        # The DISPATCHED spec (`_dispatched_spec`): declared slots already
        # rebound to the hub's picks, which is what the fence always sees.
        models={"checkpoint": Hub(PICK)},
        resources=Resources(gpu=True),
        compile=Compile(family="sdxl", shapes=((1024, 1024),), text_len=0),
    )


def _target(
    weight_lane: str, active_compiled_graph: str, active_digest: str,
    *, held_ref: str = PICK,
) -> _CompileTargetRecord:
    return _CompileTargetRecord(
        incarnation_id=INCARNATION,
        spec=_spec(),
        pipeline=object(),
        pipeline_weight_lane=weight_lane,
        lora_bucket=0,
        contract_digest=CONTRACT,
        active_compile_ref=active_compiled_graph,
        active_compile_snapshot_digest=active_digest,
        function_names=("generate",),
        model_bindings=(("checkpoint", held_ref, SNAPSHOT),),
    )


def _run(
    *, cg_ref: str = PINNED_CELL, digest: str = PINNED_DIGEST,
    contract: str = CONTRACT,
) -> pb.RunJob:
    run = pb.RunJob(
        request_id="req-888",
        attempt=1,
        function_name="generate",
        models=[pb.ModelBinding(slot="checkpoint", ref=PICK)],
        required_compile=pb.RequiredCompileExecution(
            target_incarnation_id=INCARNATION,
            cell_ref=cg_ref,
            cell_snapshot_digest=digest,
            contract_digest=contract,
        ),
    )
    run.snapshots[PICK].digest = SNAPSHOT
    return run


@pytest.fixture()
def events() -> Iterator[List[pb.ActivityUpdate]]:
    captured: List[pb.ActivityUpdate] = []
    previous = activity_mod._sink  # noqa: SLF001
    activity_mod._sink = captured.append  # noqa: SLF001
    try:
        yield captured
    finally:
        activity_mod._sink = previous  # noqa: SLF001


def test_pinned_compiled_graph_gone_serves_instead_of_refusing(
    events: List[pb.ActivityUpdate],
) -> None:
    """RED on master: `required_compile_identity_mismatch` -> RetryableError
    -> `JOB_STATUS_RETRYABLE` -> the hub's five-retry loop. The pipeline is
    the right one on the plain lane; nothing about it is wrong except that
    the compiled graph it was pinned to is not armed."""
    fence = _Fence("fp8-w8a16+compiled", _target("fp8-w8a16", "", ""))
    job = _LiveJob("req-888")
    fence.jobs[("req-888", 1)] = job
    spec = _spec()

    _serve(fence, spec, _run())  # must not raise

    assert job.served_eager_fallback is True
    assert job.fallback_reason == (
        serving_mode_mod.FALLBACK_PINNED_CELL_UNAVAILABLE)


def test_the_degraded_event_carries_graph_key_mode_and_cause(
    events: List[pb.ActivityUpdate],
) -> None:
    """The confession is a TYPED event, not a log line — a hub-spawned worker
    exposes no stdout (pgw#760), so a log line reaches nobody."""
    fence = _Fence("fp8-w8a16+compiled", _target("fp8-w8a16", "", ""))
    fence.jobs[("req-888", 1)] = _LiveJob("req-888")

    _serve(fence, _spec(), _run())

    degrades = [e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]
    assert len(degrades) == 1
    event = degrades[0]
    # the degraded MODE
    assert event.phase == serving_mode_mod.FALLBACK_PINNED_CELL_UNAVAILABLE
    # the FAILED GRAPH KEY (proto field 19 is still spelled `cell_key`)
    assert event.cell_key == PINNED_CELL
    assert PINNED_CELL in event.detail
    # the CAUSE, and what was served instead
    assert "not armed on this target" in event.detail
    assert "serving eager" in event.detail
    assert "req-888" in event.detail


def test_a_superseded_cell_serves_compiled_and_is_not_an_eager_sample(
    events: List[pb.ActivityUpdate],
) -> None:
    """The pod holds a DIFFERENT armed compiled graph. Still degraded — the hub's pin
    was not honoured — but the request is not an eager latency sample, and
    charging it one would be the exact `serving_mode` contamination pgw#764
    exists to prevent."""
    fence = _Fence(
        "fp8-w8a16+compiled",
        _target("fp8-w8a16", "cg:sdxl-1024-sm89-v2", "sha256:" + "d4" * 32),
    )
    job = _LiveJob("req-888")
    fence.jobs[("req-888", 1)] = job

    _serve(fence, _spec(), _run())

    assert job.served_eager_fallback is False
    assert job.fallback_reason == ""
    degrades = [e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]
    assert len(degrades) == 1
    assert "serving the armed compiled graph" in degrades[0].detail
    assert "cg:sdxl-1024-sm89-v2" in degrades[0].detail


def test_mandatory_quantized_lane_still_refuses(
    events: List[pb.ActivityUpdate],
) -> None:
    """THE CONTROL, and the carve-out `hot_swap`'s docstring draws: on a
    declared-mandatory w8a8/w4a4 lane the author sanctioned compiled
    execution and nothing else, so a degrade here would answer with numerics
    the endpoint never sanctioned. That is a refusal, not a degrade."""
    fence = _Fence(
        "fp8-w8a8-dynamic+compiled", _target("w8a8-lora0", "", ""))
    job = _LiveJob("req-888")
    fence.jobs[("req-888", 1)] = job

    with pytest.raises(RetryableError, match="required_compile_identity_mismatch"):
        _serve(fence, _spec(), _run())

    assert job.served_eager_fallback is False
    assert not [
        e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]


def test_a_changed_execution_contract_still_requeues(
    events: List[pb.ActivityUpdate],
) -> None:
    """Identity, not availability: a different contract digest means the
    target's call ingress is not the one this dispatch was validated against,
    so serving it would run a different signature."""
    fence = _Fence(
        "fp8-w8a16+compiled", _target("fp8-w8a16", PINNED_CELL, PINNED_DIGEST))

    with pytest.raises(RetryableError, match="required_compile_contract_mismatch"):
        _serve(fence, _spec(), _run(contract="contract-digest-2"))


def test_a_different_model_binding_still_requeues(
    events: List[pb.ActivityUpdate],
) -> None:
    """pgw#888 acceptance box 2: the identity fence is SPLIT from compiled graph
    availability, and its half keeps requeuing. A merely same-family pipeline
    is not the model the hub picked."""
    fence = _Fence(
        "fp8-w8a16+compiled",
        _target("fp8-w8a16", "", "",
                held_ref="acme/wai-illustrious@sha256:" + "ee" * 32))

    with pytest.raises(RetryableError, match="required_compile_binding_mismatch"):
        _serve(fence, _spec(), _run())
    assert not [
        e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]


def test_an_exactly_matching_pin_is_silent(
    events: List[pb.ActivityUpdate],
) -> None:
    """The optimization fence still fences: a request the pod can serve on
    the exact pinned compiled graph must produce no degrade event at all."""
    fence = _Fence(
        "fp8-w8a16+compiled", _target("fp8-w8a16", PINNED_CELL, PINNED_DIGEST))
    job = _LiveJob("req-888")
    fence.jobs[("req-888", 1)] = job

    _serve(fence, _spec(), _run())

    assert job.served_eager_fallback is False
    assert not [
        e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]


def test_one_request_confesses_once(events: List[pb.ActivityUpdate]) -> None:
    """`_JobOrder.fence` is invoked TWICE for one job — at intake, and again
    as the last execution fence before the GPU turn. A degrade that persists
    across both must not double every hub-side count of it."""
    fence = _Fence("fp8-w8a16+compiled", _target("fp8-w8a16", "", ""))
    fence.jobs[("req-888", 1)] = _LiveJob("req-888")

    _serve(fence, _spec(), _run())   # intake
    _serve(fence, _spec(), _run())   # last execution fence

    assert len(
        [e for e in events if e.kind == activity_mod.KIND_SERVE_DEGRADE]) == 1


def test_pinned_compiled_graph_unavailable_is_a_wire_fallback_class() -> None:
    """A token `resolve()` does not recognise never reaches `metrics.
    fallback_reason` — the request would report the degrade nowhere."""
    served = serving_mode_mod.resolve(
        active_compile_ref="",
        verdict=serving_mode_mod.FALLBACK_PINNED_CELL_UNAVAILABLE,
    )
    assert served.served_eager_fallback is True
    assert served.fallback_reason == (
        serving_mode_mod.FALLBACK_PINNED_CELL_UNAVAILABLE)
