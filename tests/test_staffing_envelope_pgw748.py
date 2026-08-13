"""pgw#748/th#1285: the author ENVELOPE has an SDK carrier.

The hub builder has parsed ``resources["max_gpu_count"]`` and
``resources["parallel"]`` since th#1285 (``extractStaffingEnvelope``), and the
whole tier->degree product (typed admission, cohort-exact buys, degree-exact
dispatch) keys off ``requirement_payload["parallel"]`` — but ``Resources``
could not declare either field, so no real endpoint could opt into the fast
tier through the build path. These tapes go through the REAL discovery
manifest builder (the pgw#660/pgw#670 defect class: a declaration that never
reaches the builder's ``resources{}`` keys):

  * the envelope reaches the manifest under the keys the builder reads;
  * omitting it changes nothing (``omit_defaults``) — every existing
    release's payload stays byte-identical;
  * the builder's ingest refusals are mirrored at declaration time.
"""

from __future__ import annotations

from typing import Any, Dict

import msgspec
import pytest

from gen_worker import RequestContext, Resources, endpoint
from gen_worker.discovery.discover import _extract_compiled_graphs
from gen_worker.families import GenerationDefaults


class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 30


class In(msgspec.Struct):
    prompt: str = "x"


class Out(msgspec.Struct):
    ok: bool = True


@endpoint(resources=Resources(
    gpu=True, vram_gb_hint=80.0, max_gpu_count=2, parallel=("sequence",)))
class SPEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


@endpoint(resources=Resources(gpu=True, vram_gb_hint=80.0))
class PlainEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def _resources(cls: type) -> Dict[str, Any]:
    (fn,) = _extract_compiled_graphs(cls, "testmod")
    return fn["resources"]


def test_envelope_reaches_manifest_under_builder_keys() -> None:
    res = _resources(SPEndpoint)
    assert res["max_gpu_count"] == 2
    # In-memory the projection may hold a tuple; on the wire (JSON) it is a
    # list — the builder's normalizeParallelMechanisms accepts exactly that.
    assert list(res["parallel"]) == ["sequence"]
    assert msgspec.json.decode(msgspec.json.encode(res))["parallel"] == ["sequence"]


def test_undeclared_envelope_is_absent_not_defaulted() -> None:
    res = _resources(PlainEndpoint)
    assert "max_gpu_count" not in res
    assert "parallel" not in res


def test_max_gpu_count_implies_gpu() -> None:
    assert Resources(max_gpu_count=2).gpu is True


def test_parallel_tokens_normalized() -> None:
    r = Resources(gpu=True, max_gpu_count=4, parallel=(" Sequence ",))
    assert r.parallel == ("sequence",)


def test_unknown_mechanism_refused_at_declaration() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        Resources(gpu=True, max_gpu_count=2, parallel=("tensor",))


def test_parallel_without_headroom_refused() -> None:
    # No ceiling at all.
    with pytest.raises(ValueError, match="headroom"):
        Resources(gpu=True, parallel=("sequence",))
    # Ceiling equal to the floor — same contradiction the builder refuses.
    with pytest.raises(ValueError, match="headroom"):
        Resources(gpu=True, gpu_count=2, max_gpu_count=2, parallel=("sequence",))


def test_ceiling_below_floor_refused() -> None:
    with pytest.raises(ValueError, match="below gpu_count"):
        Resources(gpu=True, gpu_count=4, max_gpu_count=2)


# --- th#1426: the DEGREE axis, declared independently of the pod width ------
#
# `max_gpu_count` alone could not express a multi-group sharded pod: the hub
# derived the group degree from the ceiling and then the width from the ceiling
# over the degree, which forces width to 1 identically. The second declared
# number is what makes 2x2 askable.


@endpoint(resources=Resources(
    gpu=True, vram_gb_hint=80.0, max_gpu_count=4,
    max_gpus_per_execution_group=2, parallel=("sequence",)))
class TwoByTwoEndpoint:
    def generate(self, ctx: RequestContext[_Defaults], p: In) -> Out:
        return Out()


def test_group_width_reaches_manifest_under_the_builder_key() -> None:
    res = _resources(TwoByTwoEndpoint)
    # Both axes present and independent: 4 GPUs in the pod, 2 per request.
    assert res["max_gpu_count"] == 4
    assert res["max_gpus_per_execution_group"] == 2
    assert msgspec.json.decode(
        msgspec.json.encode(res))["max_gpus_per_execution_group"] == 2


def test_undeclared_group_width_is_absent_not_defaulted() -> None:
    # The whole existing fleet. Declaring the width axis must not start
    # emitting the degree axis with a defaulted legal value.
    assert "max_gpus_per_execution_group" not in _resources(SPEndpoint)
    assert "max_gpus_per_execution_group" not in _resources(PlainEndpoint)


def test_group_width_that_does_not_shard_is_refused() -> None:
    # 1 is the value a "default to today's behaviour" implementation would
    # have quietly accepted. It is outside the legal domain [2, ceiling], so
    # absence can never be confused with a declaration.
    with pytest.raises(ValueError, match="does not shard"):
        Resources(gpu=True, max_gpu_count=4,
                  max_gpus_per_execution_group=1, parallel=("sequence",))


def test_group_width_above_the_pod_ceiling_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds max_gpu_count"):
        Resources(gpu=True, max_gpu_count=4,
                  max_gpus_per_execution_group=8, parallel=("sequence",))
    with pytest.raises(ValueError, match="exceeds max_gpu_count"):
        Resources(gpu=True, max_gpus_per_execution_group=2,
                  parallel=("sequence",))


def test_group_width_without_a_mechanism_is_refused_as_inert() -> None:
    with pytest.raises(ValueError, match="inert"):
        Resources(gpu=True, max_gpu_count=4, max_gpus_per_execution_group=2)


def test_group_width_positive_control() -> None:
    # A validator that refused every group width would pass the table above
    # trivially. The legal shapes must construct.
    for width in (2, 3, 4):
        r = Resources(gpu=True, max_gpu_count=4,
                      max_gpus_per_execution_group=width,
                      parallel=("sequence",))
        assert r.max_gpus_per_execution_group == width
        assert r.gpu is True
