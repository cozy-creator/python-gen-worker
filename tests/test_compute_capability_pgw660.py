"""pgw#660, re-spelled by pgw#1313: the hard GPU-architecture floor still has a
declared carrier — it is now a TERM in the one requirement vocabulary.

The v2 API freeze deleted ``Resources.compute_capability`` on the reasoning
that "precision-per-card is the fit ladder's call, never a placement gate".
That reasoning holds for PRECISION SELECTION and does not hold for
INCAPABILITY: a producer whose kernel is ``torch._scaled_mm`` cannot run below
sm_89 at any precision, on any ladder rung, ever. With no carrier the hub's
builder — which still reads the key — emitted nothing, so
``requirement_payload_json`` lost ``compute_capability`` and the scheduler
placed the fp8 producer on sm_80 A100s (th#1155 x6; te#125 again, 2026-07-26).

pgw#1313 folds the bespoke axis into ``Resources(requires=)`` and keeps the
floor byte-identical on the wire. These tapes go through the REAL discovery
manifest builder, because the entire defect class is a declaration that never
reaches the builder's ``resources{}`` keys.
"""

from __future__ import annotations

from typing import Any, Dict

import msgspec
import pytest

from gen_worker import (
    LayoutDeclarationError, LayoutRequirements, RequestContext, Resources,
    endpoint,
)
from gen_worker.discovery.discover import _extract_entries
from gen_worker.families import GenerationDefaults


class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 30


class In(msgspec.Struct):
    prompt: str = "x"


class Out(msgspec.Struct):
    ok: bool = True


@endpoint(resources=Resources(requires="sm89+", vcpus=16))
class FP8ProducerEndpoint:
    """The shape of conversion's modelopt producers: scaled_mm or nothing."""

    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


@endpoint(resources=Resources(gpu=True))
class UndeclaredEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def _requirement(resources: Resources) -> LayoutRequirements:
    """The parsed function-scope requirement, asserted present."""
    parsed = resources.requirement()
    assert parsed is not None
    return parsed


def _resources(cls: type) -> Dict[str, Any]:
    (fn,) = _extract_entries(cls, "testmod")
    res = fn["resources"]
    assert isinstance(res, dict)
    return res


def test_the_bespoke_axis_is_gone_and_the_term_replaced_it() -> None:
    """pgw#1313 row 12: `compute_capability` was a bespoke axis with its own
    parser, its own payload key and its own hub readers. It is DELETED as a
    declaration — no alias, no shim — and `min_sm` in the one vocabulary is
    where the floor lives now."""
    assert "compute_capability" not in Resources.__struct_fields__
    assert "compute_capability_min" not in Resources.__struct_fields__
    assert "compute_capability_hint" not in Resources.__struct_fields__
    assert "requires" in Resources.__struct_fields__
    with pytest.raises(TypeError):
        Resources(compute_capability=8.9)  # type: ignore[call-arg]


def test_declaration_is_validated_and_normalized() -> None:
    # ONE spelling now — tensorhub's own BARE sm code, the same spelling
    # `WorkerResources.gpu_sm` and `contractspec.DecodeEntry.MinSM` use. The
    # dotted 8.9 is gone rather than kept as a second stored form; the
    # `Resources.manifest_dict()` projection is the only place it appears.
    assert _requirement(Resources(requires="sm89+")).min_terms().min_sm == 89
    assert _requirement(
        Resources(requires={"min_sm": 100})).min_terms().min_sm == 100
    assert _requirement(Resources(requires=LayoutRequirements(
        minimum="sm89+"))).min_terms().min_sm == 89
    for bad in ("sm_89", "sm89", "8.9", "", {"min_sm": 0}, {"min_sm": -8}):
        with pytest.raises(LayoutDeclarationError):
            Resources(requires=bad)


def test_an_arch_floor_implies_a_gpu() -> None:
    # A compute-capability floor is meaningless without a CUDA device, so it
    # implies gpu=True — as does a VRAM floor. Host RAM does not.
    assert Resources(requires="sm89+").gpu is True
    assert Resources(requires="vram80g").gpu is True
    assert Resources(
        requires=LayoutRequirements(recommended="ram64g")).gpu is False


def test_the_floor_reaches_the_manifest_under_the_builders_own_key() -> None:
    res = _resources(FP8ProducerEndpoint)
    # The successor key: the requirement ROW, declared terms only, the same
    # shape the slot scope emits. th#2072 reads it.
    assert res["requires"] == {"min_sm": 89}
    # And it travels ONLY there. The `compute_capability` back-projection is
    # deleted: the hub prefers `requires` wherever present and this method
    # emitted the projection only when `min_sm` was declared, so the arm could
    # never be read for a wheel built from this source. The hub keeps its own
    # arm for PUBLISHED wheels that emit no `requires`; th#2074 retires it.
    assert "compute_capability" not in res
    assert res["gpu"] is True
    # th#1867: nothing VRAM-shaped rides beside it. `min_vram_gb` is declarable
    # in the vocabulary but is NOT projected to the builder's buy-side key —
    # arming that floor is th#2073's, with the fail-open closed in one change.
    assert not [k for k in res if "vram" in k]


def test_a_vram_floor_does_not_resurrect_the_buy_side_key() -> None:
    res = Resources(requires="sm90+, vram80g").manifest_dict()
    assert res["requires"] == {"min_sm": 90, "min_vram_gb": 80.0}
    assert "min_vram_gb" not in res and "vram_gb" not in res


def test_undeclared_is_unchanged() -> None:
    # Migration contract: a function that declares no floor emits no key and
    # keeps today's behaviour (no gate).
    res = _resources(UndeclaredEndpoint)
    assert "compute_capability" not in res
    assert "requires" not in res


def test_the_projection_is_owned_by_resources() -> None:
    # One mapping, in one place, so a second manifest consumer cannot drift.
    assert Resources(requires="sm89+", vcpus=4).manifest_dict() == {
        "gpu": True,
        "requires": {"min_sm": 89},
        "vcpus": 4,
    }


def test_no_key_but_requires_states_a_machine_requirement() -> None:
    # The class-wide statement of the cut: `requires` is the ONE key a
    # machine requirement rides. A second key restating the same fact is how
    # two spellings of a floor drift, and it is what th#2074 is left holding
    # on the hub side for already-published wheels.
    for decl in ("sm89+", "sm90+, vram80g", "sm100+"):
        row = Resources(requires=decl).manifest_dict()
        assert "requires" in row
        assert not [k for k in row if "capability" in k]
