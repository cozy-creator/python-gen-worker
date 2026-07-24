"""gw#596: the worker honors a hub-resolved per-request lane instruction.

Real code paths (Executor._lane_effective_spec / _served_lane /
apply_model_resolutions), never mocks of them:
  1. RunJob.lane="bf16" on a worker rebound to the fp8-w8a8 pick derives a
     spec bound to the declared base — a NEW instance key, so the warm fp8
     instance stays resident (gw#551 cycling) while the bf16 variant loads.
  2. RunJob.lane="fp8-w8a8-dynamic+compiled" keeps the pick.
  3. A w8a8 request with no w8a8 pick refuses TYPED, naming the lane.
  4. "fp8-w8a8-dynamic+eager" refuses (w8a8 is compiled-only).
  5. Family "fp8" without a stored pick expands to the local cast lane.
  6. An unknown lane string is a ValidationError (INVALID), not a crash.
  7. JobMetrics.lane reports the CONCRETE serving lane.
"""

from __future__ import annotations

from typing import List

import msgspec
import pytest

from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.errors import ValidationError
from gen_worker.executor import Executor
from gen_worker.models.lanes import LaneUnavailableError
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline, vae) -> None:  # pragma: no cover
        self.pipeline = pipeline

    def generate(self, ctx, payload: _In) -> dict:  # pragma: no cover
        return {}


BASE = "acme/z-image"
VAE = "acme/vae"
W8A8 = "acme/z-image#fp8-w8a8"


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub(BASE), "vae": Hub(VAE)},
    )


def _executor() -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    return Executor([_spec()], _send)


def _with_w8a8_pick(ex: Executor) -> None:
    ex.apply_model_resolutions({
        BASE: (W8A8, "", "fp8-w8a8-dynamic+compiled"),
    })


def test_bf16_override_rebinds_to_declared_base() -> None:
    ex = _executor()
    _with_w8a8_pick(ex)
    spec = ex.specs["generate"]
    assert wire_ref(spec.models["pipeline"]) == W8A8
    warm_key = spec.instance_key

    derived = ex._lane_effective_spec(spec, "bf16")
    assert wire_ref(derived.models["pipeline"]) == BASE
    assert derived.models["pipeline"].storage_dtype == ""
    assert wire_ref(derived.models["vae"]) == VAE
    # New instance key: the warm w8a8 instance is untouched (gw#551 cycling).
    assert derived.instance_key != warm_key
    assert wire_ref(spec.models["pipeline"]) == W8A8  # original spec unchanged


def test_w8a8_descriptor_keeps_the_pick() -> None:
    ex = _executor()
    _with_w8a8_pick(ex)
    spec = ex.specs["generate"]
    derived = ex._lane_effective_spec(spec, "fp8-w8a8-dynamic+compiled")
    assert wire_ref(derived.models["pipeline"]) == W8A8
    # Same bindings -> same spec object (no needless instance split).
    assert derived is spec


def test_w8a8_without_pick_refuses_typed() -> None:
    ex = _executor()  # no resolutions applied
    spec = ex.specs["generate"]
    with pytest.raises(LaneUnavailableError) as e:
        ex._lane_effective_spec(spec, "fp8-w8a8-dynamic+compiled")
    assert "lane_unavailable: fp8-w8a8-dynamic+compiled" in str(e.value)
    assert "generate" in str(e.value)


def test_w8a8_eager_refuses() -> None:
    ex = _executor()
    _with_w8a8_pick(ex)
    with pytest.raises(LaneUnavailableError) as e:
        ex._lane_effective_spec(ex.specs["generate"], "fp8-w8a8-dynamic+eager")
    assert "compiled-only" in str(e.value)


def test_fp8_family_without_stored_pick_expands_to_cast_lane() -> None:
    ex = _executor()
    # The hub laddered the pipeline but picked base (e.g. no fp8 flavor
    # published): family fp8 expands to the local cast lane on that ref.
    ex.apply_model_resolutions({BASE: (BASE, "", "bf16-w16a16+eager")})
    spec = ex.specs["generate"]
    derived = ex._lane_effective_spec(spec, "fp8")
    assert wire_ref(derived.models["pipeline"]) == BASE
    assert derived.models["pipeline"].storage_dtype == "fp8"
    # The never-laddered vae keeps its binding untouched.
    assert derived.models["vae"].storage_dtype == ""


def test_fp8_family_with_no_laddered_refs_refuses_typed() -> None:
    ex = _executor()  # hub delivered no resolutions at all
    with pytest.raises(LaneUnavailableError):
        ex._lane_effective_spec(ex.specs["generate"], "fp8")


def test_fp8_family_with_pick_uses_it() -> None:
    ex = _executor()
    _with_w8a8_pick(ex)
    spec = ex.specs["generate"]
    derived = ex._lane_effective_spec(spec, "fp8")
    assert wire_ref(derived.models["pipeline"]) == W8A8


def test_unknown_lane_is_invalid() -> None:
    ex = _executor()
    with pytest.raises(ValidationError):
        ex._lane_effective_spec(ex.specs["generate"], "int8-w8a8+eager")


def test_4bit_without_pick_refuses_typed() -> None:
    ex = _executor()
    with pytest.raises(LaneUnavailableError):
        ex._lane_effective_spec(ex.specs["generate"], "4bit")


def test_served_lane_reports_concrete_lane() -> None:
    ex = _executor()
    spec = ex.specs["generate"]
    # Declared base, no compile targets: bf16 eager.
    assert ex._served_lane(spec) == "bf16-w16a16+eager"
    # w8a8 pick applied: the pipeline's lane wins over the bf16 vae.
    _with_w8a8_pick(ex)
    assert ex._served_lane(ex.specs["generate"]) == "fp8-w8a8-dynamic+eager"
    # cast pick: w8a16.
    ex.apply_model_resolutions({BASE: (BASE, "fp8", "fp8-w8a16+eager")})
    assert ex._served_lane(ex.specs["generate"]) == "fp8-w8a16+eager"


def test_hello_ack_lane_field_flows_into_resolutions() -> None:
    ex = _executor()
    ack = pb.HelloAck(resolutions=[pb.ModelResolution(
        ref=BASE, resolved_ref=W8A8, cast="", lane="fp8-w8a8-dynamic+compiled",
    )])
    ex.apply_model_resolutions(
        {r.ref: (r.resolved_ref, r.cast, r.lane) for r in ack.resolutions})
    assert ex._model_resolutions[BASE] == (W8A8, "", "fp8-w8a8-dynamic+compiled")
    assert wire_ref(ex.specs["generate"].models["pipeline"]) == W8A8
