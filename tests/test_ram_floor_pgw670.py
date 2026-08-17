"""pgw#670, re-ruled by pgw#1313: the measured HOST-RAM ask is a RECOMMENDATION.

Host RAM is not merely an opportunistic latency tier: a host-starved video
allocation measured 179-301 s mp4-encode and 147 s VAE-decode tails at
IDENTICAL GPU step-ms, which sized a 64 GB ask (ie#484/ie#492). That
measurement stands. What changed is what the platform may DO with it: Paul,
2026-07-11 — RunPod GPU pods cannot select or guarantee host RAM, so a declared
host-RAM MINIMUM was unenforceable theater, and the standing instruction is not
to rebuild a boot-time RAM gate or a read-back-and-reject loop. So:

  * ``min_host_ram_gb`` is declarable at ``recommended`` ONLY — a minimum is
    refused AT THE DECLARATION, naming the ruling and the move;
  * it does NOT imply ``gpu=True`` (a CPU encode lane needs host RAM too);
  * it is NOT projected to the builder's ``ram_gb`` pod-create minimum. A
    recommendation that becomes an allocation floor is th#1720 exactly, and
    that is the failure this whole program fences structurally;
  * omitting it changes nothing: no key, no behaviour.
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


@endpoint(resources=Resources(
    gpu=True, vcpus=16, requires=LayoutRequirements(recommended="ram64g")))
class FloorEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


@endpoint(resources=Resources(vcpus=8))
class NoFloorEndpoint:
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


def test_the_hint_named_axis_is_gone() -> None:
    """A field named `_hint` that the hub enforced as a pod-create minimum was
    a misnomer, not a nuance. It is deleted, with no alias."""
    assert "ram_gb_hint" not in Resources.__struct_fields__
    with pytest.raises(TypeError):
        Resources(ram_gb_hint=64)  # type: ignore[call-arg]


def test_host_ram_is_declarable_as_a_RECOMMENDATION_only() -> None:
    rec = Resources(requires=LayoutRequirements(recommended="ram64g"))
    assert _requirement(rec).recommended_terms().min_host_ram_gb == 64.0
    # ...and a MINIMUM is refused where it is written, naming the move.
    with pytest.raises(LayoutDeclarationError, match="RECOMMENDED only"):
        Resources(requires="ram64g")
    with pytest.raises(LayoutDeclarationError, match="RECOMMENDED only"):
        Resources(requires=LayoutRequirements(minimum="sm89+, ram64g"))
    with pytest.raises(LayoutDeclarationError):
        Resources(requires=LayoutRequirements(recommended="ram0g"))


def test_the_deleted_disk_axis_is_refused_as_UNKNOWN() -> None:
    """HARDCUT C3 / pgw#1281: `min_disk_gb` (pgw#732) shipped for two releases
    and no endpoint in any repo ever declared it, so the emitter is gone. It
    must refuse BY NAME rather than be accepted and dropped — a silently
    swallowed floor is a pod sized by nothing with nobody told."""
    with pytest.raises(TypeError, match="min_disk_gb"):
        Resources(min_disk_gb=1)  # type: ignore[call-arg]


def test_a_host_floor_does_not_imply_a_gpu() -> None:
    # Video encode is host-CPU/RAM bound (ie#484), with or without a GPU.
    assert Resources(requires="sm89+").gpu is True
    assert Resources(
        requires=LayoutRequirements(recommended="ram64g")).gpu is False
    assert Resources(vcpus=16).gpu is False


def test_the_recommendation_never_becomes_an_allocation_MINIMUM() -> None:
    """THE fence. `ram_gb` is what internal/builder/function_requirements.go
    maps to the scheduler's `min_ram_gb` — a pod-create minimum. A
    recommendation projected onto it is a learned monotone floor, which is
    th#1720 and is precisely what th#1867 deleted `recommended_vram_gb` for."""
    res = _resources(FloorEndpoint)
    assert "ram_gb" not in res and "ram_gb_hint" not in res
    assert res["requires"] == {"recommended": {"min_host_ram_gb": 64.0}}
    # The surviving host CPU ask keeps its own key, unchanged.
    assert res["vcpus"] == 16
    assert res["gpu"] is True


def test_omitting_the_floor_emits_nothing() -> None:
    res = _resources(NoFloorEndpoint)
    assert "ram_gb" not in res and "requires" not in res
    assert res["vcpus"] == 8


def test_projection_is_owned_by_resources_not_by_discovery() -> None:
    # The mapping lives in ONE place so a second manifest consumer cannot
    # drift from it.
    declared = Resources(
        requires=LayoutRequirements(recommended="ram64g"), vcpus=4)
    assert declared.manifest_dict() == {
        "requires": {"recommended": {"min_host_ram_gb": 64.0}}, "vcpus": 4}
    raw = msgspec.to_builtins(declared)
    assert raw == {
        "requires": {"recommended": {"min_host_ram_gb": 64.0}}, "vcpus": 4}
