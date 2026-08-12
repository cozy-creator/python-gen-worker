"""th#1059 twin (live master incident 2026-07-23): storage never implied
execution. SDXL's mixed fp8 variant serves the w8a16 upcast lane (plain
graphs) while qwen's serves real scaled_mm w8a8; the worker's mandatory-lane
admission (`_validate_required_compile`) refused every hub dispatch for the
mixed lane with `required_compile_missing` — Paul's live jobs failed at
21:54Z after the hub half (tensorhub th#1059) started dispatching them.

Mandatory-ness follows the hub-delivered resolution lane. pgw#1148 deleted the
`#flavor`-token FALLBACK that used to stand in for it (§1.32(d)): a token in a
ref was an assertion, never evidence, and with no resolved lane there is no
mandate — asserted below."""

from __future__ import annotations

import pytest

from gen_worker import Compile, Resources
from gen_worker.api.binding import Hub, wire_ref
from gen_worker.api.errors import RetryableError
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


BARE = "acme/wai-illustrious"
#: The hub's pick is a DIGEST since th#1803, never a `#flavor`.
MIXED = "acme/wai-illustrious@sha256:" + "a1" * 32


class _Resolutions:
    """Just enough Executor surface for the resolution-aware lane methods."""

    def __init__(self, resolutions):
        self._model_resolutions = resolutions

    _resolved_mandatory_execution_lane = Executor._resolved_mandatory_execution_lane
    _mandatory_execution_lane_of_bound = Executor._mandatory_execution_lane_of_bound
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
        models={"checkpoint": Hub("acme/wai-illustrious", tag="prod")},
        resources=Resources(gpu=True),
        compile=Compile(family="sdxl", shapes=((1024, 1024),), text_len=0),
    )


def test_w8a16_resolution_execution_lane_is_not_mandatory() -> None:
    ex = _Resolutions({BARE: (MIXED, "", "fp8-w8a16+compiled")})
    assert ex._resolved_mandatory_execution_lane(MIXED) == ""
    assert ex._resolved_mandatory_execution_lane(BARE) == ""
    assert ex._mandatory_execution_lane_of_bound([MIXED]) == ""


def test_w8a8_resolution_execution_lane_stays_mandatory() -> None:
    ex = _Resolutions({BARE: (MIXED, "", "fp8-w8a8-dynamic+compiled")})
    assert ex._resolved_mandatory_execution_lane(MIXED) == "w8a8"
    assert ex._mandatory_execution_lane_of_bound([MIXED]) == "w8a8"


def test_no_mandate_without_execution_lane_evidence_pgw1148() -> None:
    """pgw#1148 RED: the `#flavor` fallback is DELETED. Without a resolved
    lane there is no mandate — and a ref that still carries a flavor token
    cannot resurrect one, because the token is not an address any more."""
    ex = _Resolutions({})
    assert ex._resolved_mandatory_execution_lane(MIXED) == ""
    assert ex._resolved_mandatory_execution_lane("acme/other#nvfp4-w4a4") == ""
    assert ex._resolved_mandatory_execution_lane(BARE) == ""
    empty_execution_lane = _Resolutions({BARE: (MIXED, "", "")})
    assert empty_execution_lane._resolved_mandatory_execution_lane(MIXED) == ""


def test_conflicting_execution_lane_evidence_fails_closed() -> None:
    ex = _Resolutions({
        BARE: (MIXED, "", "fp8-w8a16+compiled"),
        "acme/alias": (MIXED, "", "fp8-w8a8-dynamic+compiled"),
    })
    assert ex._resolved_mandatory_execution_lane(MIXED) == "w8a8"


def test_mixed_execution_lane_dispatch_admits_without_required_compile() -> None:
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
