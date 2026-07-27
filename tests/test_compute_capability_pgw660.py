"""pgw#660: the hard GPU-architecture floor has a declared carrier again.

The v2 API freeze (pgw#647) deleted ``Resources.compute_capability`` on the
reasoning that "precision-per-card is the fit ladder's call, never a placement
gate". That reasoning holds for PRECISION SELECTION and does not hold for
INCAPABILITY: a producer whose kernel is ``torch._scaled_mm`` cannot run below
sm_89 at any precision, on any ladder rung, ever. With no carrier the hub's
builder — which still reads the key — emitted nothing, so
``requirement_payload_json`` lost ``compute_capability`` and the scheduler
placed the fp8 producer on sm_80 A100s (th#1155 x6; te#125 again, 2026-07-26).

Design 1 of the three in pgw#660, ruled by Paul: restore the DECLARATION.
These tapes go through the REAL discovery manifest builder, because the entire
defect class is a declaration that never reaches the builder's ``resources{}``
keys.
"""

from __future__ import annotations

from typing import Any, Dict

import msgspec
import pytest

from gen_worker import RequestContext, Resources, endpoint
from gen_worker.discovery.discover import _extract_entries
from gen_worker.families import GenerationDefaults


class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 30


class In(msgspec.Struct):
    prompt: str = "x"


class Out(msgspec.Struct):
    ok: bool = True


@endpoint(resources=Resources(compute_capability=8.9, vcpus=16, vram_gb_hint=80))
class FP8ProducerEndpoint:
    """The shape of conversion's modelopt producers: scaled_mm or nothing."""

    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


@endpoint(resources=Resources(gpu=True, vram_gb_hint=24))
class UndeclaredEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def _resources(cls: type) -> Dict[str, Any]:
    (fn,) = _extract_entries(cls, "testmod")
    res = fn["resources"]
    assert isinstance(res, dict)
    return res


def test_declaration_is_validated_and_normalized() -> None:
    # Dual-form: authors write 8.9 or "8.9" or "sm_89" — one canonical value.
    assert Resources(compute_capability=8.9).compute_capability == 8.9
    assert Resources(compute_capability="8.9").compute_capability == 8.9
    assert Resources(compute_capability="sm_89").compute_capability == 8.9
    assert Resources(compute_capability="sm89").compute_capability == 8.9
    assert Resources(compute_capability=12).compute_capability == 12.0
    assert Resources(compute_capability="12.0").compute_capability == 12.0
    for bad in (0, -8.9, "", "hopper", "8.9.1"):
        with pytest.raises(ValueError, match="compute_capability"):
            Resources(compute_capability=bad)


def test_an_arch_floor_implies_a_gpu() -> None:
    # A compute-capability floor is meaningless without a CUDA device, so it
    # implies gpu=True exactly like vram_gb_hint does. The builder infers the
    # same thing from the key, but the declaration should not depend on that.
    assert Resources(compute_capability=8.9).gpu is True


def test_the_floor_reaches_the_manifest_under_the_builders_own_key() -> None:
    res = _resources(FP8ProducerEndpoint)
    # `compute_capability` is the key internal/builder/function_requirements.go
    # already parses (scalar or {"min": ...}) into FunctionRequirements.
    # ComputeCapabilityMin -> requirement_payload_json's
    # `compute_capability.min` -> req.MinGPUCapSM -> the placement filters.
    assert res["compute_capability"] == 8.9
    assert res["gpu"] is True
    assert res["vram_gb_hint"] == 80.0


def test_undeclared_is_unchanged() -> None:
    # Migration contract: a function that declares no floor emits no key and
    # keeps today's behaviour (no gate).
    res = _resources(UndeclaredEndpoint)
    assert "compute_capability" not in res


def test_the_projection_is_owned_by_resources() -> None:
    # One mapping, in one place, so a second manifest consumer cannot drift.
    assert Resources(compute_capability="sm_89", vcpus=4).manifest_dict() == {
        "gpu": True,
        "compute_capability": 8.9,
        "vcpus": 4,
    }


def test_the_floor_is_not_a_hint() -> None:
    # Naming is the contract here: vram_gb_hint/ram_gb_hint are ALLOCATION-time
    # asks the platform may miss. This one is an incapability declaration, so
    # it carries no _hint suffix and there is no second spelling for it.
    assert "compute_capability_hint" not in Resources.__struct_fields__
    assert "compute_capability_min" not in Resources.__struct_fields__
    assert "compute_capability" in Resources.__struct_fields__
