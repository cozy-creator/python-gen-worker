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
from gen_worker.discovery.discover import _extract_entries
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
    (fn,) = _extract_entries(cls, "testmod")
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
