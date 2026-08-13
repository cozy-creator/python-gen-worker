"""ie#655: the EXECUTION axis of `metrics.lane` is reported, not derived.

The live defect (master stack, pod `sbzo69ff5t0l44`, an H100 80GB HBM3, both
billed wan-2.2 canary requests `774a84f3-…` / `e1baede2-…`, endpoint 0.4.13 on
gen-worker 0.103.0): the worker declined its own compile mint
(`self_mint_skipped reason=insufficient_vram headroom=37.68GiB
needed~=72.54GiB … this worker serves eager for the rest of its life`), said
`eager` three times in its own boot rows, served eager — and reported
`metrics.lane = fp8-w8a8-dynamic+compiled` on both requests.

pgw#1104 fixed the WEIGHTS axis by having the recipe REPORT what it applied.
The EXECUTION axis was left derived, and it erred in the FLATTERING direction:
`_served_execution_lane` ran the lane table's PLANNING coercion — the
`fp8-w8a8-dynamic` body is compiled-only because eager w8a8 is UNMEASURED, not
because it cannot happen — over an observed eager posture, and rewrote the
fact. The lane id feeds pricing, quant verdicts, serving floors and compiled graph
identity, so an over-claim there is the worst direction available.

REVERT-TURNS-RED: `test_an_eager_worker_reports_an_eager_lane_on_the_wire`
reads `fp8-w8a8-dynamic+compiled` on the pre-fix tree, over the real gRPC
terminal path, with `serving_mode=eager` on the same JobMetrics.

The invariant is now STRUCTURAL rather than checked: the lane's execution axis
IS `ServedIdentity.serving_mode`, the same object `metrics.serving_mode` is
stamped from, composed at the same instant. There is no second reading left to
disagree with.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import serving_mode
from gen_worker.models import execution_lanes as lanespec
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.applied_lane_endpoints_ie655 import RecipeIn
from harness.hub_double import hub_double, is_ready, is_result_for

_MODULES = ("harness.applied_lane_endpoints_ie655",)


def _run(request_id: str, **run_job: object) -> "pb.JobResult":
    with hub_double(modules=_MODULES) as (scheduler, _h):
        conn = scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id=request_id, attempt=1, function_name="render",
            input_payload=msgspec.msgpack.encode(RecipeIn(prompt="x")),
            **run_job))
        return conn.wait_for(is_result_for(request_id)).job_result


def test_an_eager_worker_reports_an_eager_lane_on_the_wire() -> None:
    """THE red test. A worker with no compiled graph must not report one."""
    res = _run("r-ie655-eager")
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    m = res.metrics
    # The weights half is pgw#1104's, still correct: the recipe reported it.
    assert m.lane.startswith("fp8-w8a8-dynamic")
    # The execution half is this issue's, and it is now the observed one.
    assert m.lane.endswith("+eager")
    assert m.lane == "fp8-w8a8-dynamic+eager"


def test_the_lane_and_the_serving_mode_are_one_reading() -> None:
    """Not "they agree" — they are composed from the SAME object. A test that
    only compared them would pass on the defect the day the two derivations
    happened to coincide."""
    m = _run("r-ie655-one").metrics
    assert m.serving_mode == serving_mode.MODE_EAGER
    assert m.fallback_reason == serving_mode.POSTURE_NO_COMPILE_DECLARED
    assert m.served_eager_fallback is False
    compiled = m.serving_mode != serving_mode.MODE_EAGER
    assert m.lane.endswith("+compiled" if compiled else "+eager")


def test_ctx_execution_lane_carries_the_same_string() -> None:
    """`ctx.lane` is the handler's copy of the reported lane (th#1050), so it
    must not be the flattering one either — an author kernel that branches on
    `+compiled` would otherwise take the compiled branch on an eager pod."""
    res = _run("r-ie655-ctx")
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert msgspec.msgpack.decode(res.inline)["lane"] == "fp8-w8a8-dynamic+eager"


def test_a_declared_instruction_owns_the_body_never_the_execution() -> None:
    """The hub asked for a compiled lane; the pod does not have one. What the
    hub ASKED FOR is not evidence of what ran — the instruction keeps the body
    it declared and loses the execution axis it cannot know."""
    res = _run("r-ie655-instructed", lane="fp8-w8a8-dynamic+compiled")
    assert res.status == pb.JOB_STATUS_OK, res.safe_message
    assert res.metrics.lane == "fp8-w8a8-dynamic+eager"


# --- the vocabulary half, without a worker --------------------------------



def test_most_quantized_body_outranks_a_bf16_binding() -> None:
    """A bf16 VAE riding a w8a8 pipeline is still the w8a8 lane."""
    assert lanespec.most_quantized_body(
        ["bf16-w16a16", "fp8-w8a8-dynamic"]) == "fp8-w8a8-dynamic"
    assert lanespec.most_quantized_body([]) == "bf16-w16a16"
    assert lanespec.most_quantized_body(["nonsense"]) == "bf16-w16a16"
    assert lanespec.execution_lane_body_of_binding("fp8") == "fp8-w8a16"
    assert lanespec.execution_lane_body_of_binding("") == "bf16-w16a16"
