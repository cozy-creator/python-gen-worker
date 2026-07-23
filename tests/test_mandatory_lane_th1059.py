"""th#1059 twin (live master incident 2026-07-23): the `#fp8-w8a8` flavor
token names the STORAGE format, not the execution. SDXL's mixed variant is
`#fp8-w8a8` storage serving the w8a16 upcast lane (plain graphs), while
qwen's `#fp8-w8a8` executes real scaled_mm w8a8. The worker's mandatory-lane
admission (`_validate_required_compile`) refused every hub dispatch for the
mixed lane with `required_compile_missing` — Paul's live jobs failed at
21:54Z after the hub half (tensorhub th#1059) started dispatching them.

Mandatory-ness must follow the hub-delivered resolution lane when known;
the flavor token stays the fallback without lane evidence."""

from __future__ import annotations

import pytest

from gen_worker import Compile, Resources
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.errors import RetryableError
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


BARE = "acme/wai-illustrious:prod"
MIXED = "acme/wai-illustrious:prod#fp8-w8a8"


class _Resolutions:
    """Just enough Executor surface for the resolution-aware lane methods."""

    def __init__(self, resolutions):
        self._model_resolutions = resolutions

    _resolved_mandatory_lane = Executor._resolved_mandatory_lane
    _mandatory_lane_of_bound = Executor._mandatory_lane_of_bound
    _validate_required_compile = Executor._validate_required_compile
    _setup_slots = staticmethod(Executor._setup_slots)


class _Payload:
    pass


class _Endpoint:
    def setup(self, checkpoint: str) -> None:  # pragma: no cover - shape only
        pass

    def run(self, ctx, payload):  # pragma: no cover - shape only
        return None


def _spec() -> EndpointSpec:
    return EndpointSpec(
        name="generate-turbo", method=_Endpoint.run, kind="inference",
        payload_type=_Payload, output_mode="single", cls=_Endpoint,
        attr_name="run",
        models={"checkpoint": Hub("acme/wai-illustrious", tag="prod", flavor="fp8-w8a8")},
        resources=Resources(vram_gb=1.0),
        compile=Compile(family="sdxl", shapes=((1024, 1024),)),
    )


def test_w8a16_resolution_lane_is_not_mandatory() -> None:
    ex = _Resolutions({BARE: (MIXED, "", "fp8-w8a16+compiled")})
    assert ex._resolved_mandatory_lane(MIXED) == ""
    assert ex._resolved_mandatory_lane(BARE) == ""
    assert ex._mandatory_lane_of_bound([MIXED]) == ""


def test_w8a8_resolution_lane_stays_mandatory() -> None:
    ex = _Resolutions({BARE: (MIXED, "", "fp8-w8a8-dynamic+compiled")})
    assert ex._resolved_mandatory_lane(MIXED) == "w8a8"
    assert ex._mandatory_lane_of_bound([MIXED]) == "w8a8"


def test_flavor_token_fallback_without_lane_evidence() -> None:
    ex = _Resolutions({})
    assert ex._resolved_mandatory_lane(MIXED) == "w8a8"
    assert ex._resolved_mandatory_lane("acme/other:prod#nvfp4-w4a4") == "w4a4"
    assert ex._resolved_mandatory_lane(BARE) == ""
    empty_lane = _Resolutions({BARE: (MIXED, "", "")})
    assert empty_lane._resolved_mandatory_lane(MIXED) == "w8a8"


def test_conflicting_lane_evidence_fails_closed() -> None:
    ex = _Resolutions({
        BARE: (MIXED, "", "fp8-w8a16+compiled"),
        "acme/alias:prod": (MIXED, "", "fp8-w8a8-dynamic+compiled"),
    })
    assert ex._resolved_mandatory_lane(MIXED) == "w8a8"


def test_mixed_lane_dispatch_admits_without_required_compile() -> None:
    """The live failure shape: RunJob without required_compile for the mixed
    checkpoint must ADMIT (JIT setup), not raise required_compile_missing."""
    ex = _Resolutions({BARE: (MIXED, "", "fp8-w8a16+compiled")})
    spec = _spec()
    run = pb.RunJob(
        function_name=spec.name,
        models=[pb.ModelBinding(slot="checkpoint", ref=wire_ref(spec.models["checkpoint"]))],
    )
    ex._validate_required_compile(spec, run)  # must not raise


def test_w8a8_dispatch_without_required_compile_still_refuses() -> None:
    ex = _Resolutions({BARE: (MIXED, "", "fp8-w8a8-dynamic+compiled")})
    spec = _spec()
    run = pb.RunJob(
        function_name=spec.name,
        models=[pb.ModelBinding(slot="checkpoint", ref=wire_ref(spec.models["checkpoint"]))],
    )
    with pytest.raises(RetryableError, match="required_compile_missing"):
        ex._validate_required_compile(spec, run)
