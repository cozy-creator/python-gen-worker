"""pgw#670: a measured HOST-RAM floor has a successor again.

The v2 cut deleted ``Resources(ram_gb=)`` on the reasoning that host RAM is an
opportunistic latency tier. For ltx-video-2.3 it was neither opportunistic nor
a guess: ie#484 measured 179-301 s mp4-encode and 147 s VAE-decode tails on
host-starved allocations at IDENTICAL GPU step-ms, ie#492 sized the floor at
64 GB from that failure, and the hub consumed it as a pod-create minimum plus
th#740's read-back-and-reject. Without a carrier a starved allocation degrades
SILENTLY — a slow request, not a refused pod.

Tapes go through the REAL discovery manifest builder, because the whole defect
class (pgw#660) is a declaration that never reaches the builder's
``resources{}`` keys:

  * ``ram_gb_hint`` is declared, validated, and reaches the manifest under the
    key the builder actually reads (``ram_gb`` -> scheduler ``min_ram_gb``);
  * it does NOT imply ``gpu=True`` (a CPU encode lane needs host RAM too),
    unlike ``vram_gb_hint``;
  * omitting it changes nothing (``omit_defaults``): no key, no behaviour.
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


@endpoint(resources=Resources(gpu=True, vcpus=16, ram_gb_hint=64, vram_gb_hint=80))
class FloorEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


@endpoint(resources=Resources(vcpus=8))
class NoFloorEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def _resources(cls: type) -> Dict[str, Any]:
    (fn,) = _extract_entries(cls, "testmod")
    res = fn["resources"]
    assert isinstance(res, dict)
    return res


def test_declaration_is_validated() -> None:
    assert Resources(ram_gb_hint=64).ram_gb_hint == 64.0
    assert Resources(ram_gb_hint=1).ram_gb_hint == 1.0
    with pytest.raises(ValueError, match="ram_gb_hint must be positive"):
        Resources(ram_gb_hint=0)
    with pytest.raises(ValueError, match="ram_gb_hint must be positive"):
        Resources(ram_gb_hint=-8)


def test_a_host_floor_does_not_imply_a_gpu() -> None:
    # vram_gb_hint is a GPU-axis ask and forces gpu=True; a host-RAM floor is
    # not (video encode is host-CPU/RAM bound — ie#484 — with or without a GPU).
    assert Resources(vram_gb_hint=80).gpu is True
    assert Resources(ram_gb_hint=64).gpu is False
    assert Resources(vcpus=16).gpu is False


def test_the_floor_reaches_the_manifest_under_the_builders_own_key() -> None:
    res = _resources(FloorEndpoint)
    # `ram_gb` is what internal/builder/function_requirements.go maps to the
    # scheduler's `min_ram_gb` (pod-create minimum + th#740 read-back-and-
    # reject). One declaration, one wire name, one mapping.
    assert res["ram_gb"] == 64.0
    assert "ram_gb_hint" not in res, "the declaration name never rides the wire"
    # The surviving host CPU ask keeps its own key, unchanged.
    assert res["vcpus"] == 16
    assert res["gpu"] is True


def test_omitting_the_floor_emits_nothing() -> None:
    res = _resources(NoFloorEndpoint)
    assert "ram_gb" not in res and "ram_gb_hint" not in res
    assert res["vcpus"] == 8


def test_projection_is_owned_by_resources_not_by_discovery() -> None:
    # The mapping lives in ONE place so a second manifest consumer cannot
    # drift from it.
    projected = Resources(ram_gb_hint=64, vcpus=4).manifest_dict()
    assert projected == {"ram_gb": 64.0, "vcpus": 4}
    raw = msgspec.to_builtins(Resources(ram_gb_hint=64, vcpus=4))
    assert raw == {"ram_gb_hint": 64.0, "vcpus": 4}
