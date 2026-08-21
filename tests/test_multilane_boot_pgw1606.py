"""pgw#1606 — a MULTI-LANE Model class can be resolved. Until this, it could
not boot at all.

The audit's headline finding, made executable. `LoadedEndpoint.lane(cls, "")`
raised `EndpointLoadError` for any model declaring more than one lane, and the
only writers of the `contract` that would have silenced it are the local CLI,
the daemon and `serving/__main__` — `entrypoint.py` builds `Worker(...)` with
no `lane=` at all. So on a pod, "declare two lanes" meant "do not boot", and
that is why no endpoint in the fleet has ever declared two.

These use the REAL sdxl contract documents that ship in tensorfs today
(`sdxl.diffusers-bf16@1` and `sdxl.diffusers-fp8-rowwise@1`), not fabricated
ones, so the class under test is one the fleet could actually write.
"""

from __future__ import annotations

from typing import Any

import pytest

from gen_worker._vendor.tensorfs import contracts
from gen_worker.serving import lane_ladder as L
from gen_worker.serving.loader import EndpointLoadError, LoadedEndpoint
from gen_worker.serving.model import Model
from gen_worker.models import SDXL

BF16 = contracts.SDXL_DIFFUSERS_BF16
FP8 = contracts.SDXL_DIFFUSERS_FP8_ROWWISE


class _Verdicts:
    def __init__(self, staged=(), sizes=None):
        self._staged = set(staged)
        self._sizes = dict(sizes or {})

    def verdict(self, contract_id: str) -> str:
        return (L.VERDICT_SATISFIES if contract_id in self._staged
                else L.VERDICT_ABSENT)

    def transfer_bytes(self, contract_id: str) -> int:
        return int(self._sizes.get(contract_id, 0))


class _Gates:
    def __init__(self, w8a8="rowwise", w4a4=""):
        self._a, self._b = w8a8, w4a4

    def w8a8_mode(self) -> str:
        return self._a

    def w4a4_mode(self) -> str:
        return self._b


class TwoLaneModel(Model[SDXL], lanes={BF16: "vram7g", FP8: "vram5g"}):
    """One Model class, two REAL lanes — the shape Paul's multi-lane example
    ratified and the shape that could not boot before this issue."""

    def load(self, ctx: Any) -> None:  # pragma: no cover — never run here
        self.pipe = ctx.load_pipeline(object)


class OneLaneModel(Model[SDXL], lanes={BF16: "vram7g"}):
    def load(self, ctx: Any) -> None:  # pragma: no cover
        self.pipe = ctx.load_pipeline(object)


def _loaded(*models: type) -> LoadedEndpoint:
    return LoadedEndpoint(module_name="fixture", entrypoints={},
                          models=tuple(models))


ADA = L.CardFacts(sm=89, vram_gb=8.0, name="RTX 4070 Laptop")
AMPERE = L.CardFacts(sm=86, vram_gb=24.0, name="RTX 3090")


def test_the_declaration_itself_is_accepted_with_two_lanes():
    from gen_worker.serving.model import model_lanes

    handles = {lane.stamp for lane in model_lanes(TwoLaneModel)}
    assert handles == {"sdxl.diffusers-bf16@1", "sdxl.diffusers-fp8-rowwise@1"}


def test_lane_still_refuses_to_GUESS_but_now_points_at_the_ladder():
    """`lane()` answering "the" lane of a multi-lane model would be a silent
    default, which is exactly what must not exist. It still refuses — but the
    refusal is now a wiring message, not a dead end."""
    with pytest.raises(EndpointLoadError, match="ask `resolve\\(\\)`"):
        _loaded(TwoLaneModel).lane(TwoLaneModel, "")


def test_resolve_picks_fp8_on_ada_and_names_the_rejected_rung():
    resolved = _loaded(TwoLaneModel).resolve(
        TwoLaneModel, card=ADA,
        verdicts=_Verdicts(staged=[BF16.stamp, FP8.stamp]), gates=_Gates(),
    )
    assert resolved.body == "fp8-w8a8-dynamic"
    assert resolved.contract_id == "sdxl.diffusers-fp8-rowwise@1"
    assert [r.reason for r in resolved.rejected] == [L.REJECT_OUTRANKED]
    assert "LANE=fp8-w8a8-dynamic" in resolved.confession()


def test_resolve_falls_to_bf16_on_ampere_using_the_DERIVED_floor():
    """No `min_sm` is written anywhere in `TwoLaneModel`'s header. The 89 comes
    from `capability_floor_for_dtype(float8_e4m3fn)`, which is the single
    producer pgw#1599 keeps."""
    resolved = _loaded(TwoLaneModel).resolve(
        TwoLaneModel, card=AMPERE,
        verdicts=_Verdicts(staged=[BF16.stamp, FP8.stamp]), gates=_Gates(),
    )
    assert resolved.body == "bf16-w16a16"
    (rung,) = resolved.rejected
    assert rung.reason == L.REJECT_SM_FLOOR
    assert "needs sm89, card is sm86" in rung.detail


def test_the_upcast_rung_on_a_real_two_lane_sdxl_class():
    """Ampere cannot run fp8, and the fp8 tree is half the bytes. Fetch those,
    upcast at load, serve full precision — and report the saving."""
    resolved = _loaded(TwoLaneModel).resolve(
        TwoLaneModel, card=AMPERE,
        verdicts=_Verdicts(staged=[BF16.stamp, FP8.stamp],
                           sizes={BF16.stamp: 6_900_000_000,
                                  FP8.stamp: 3_500_000_000}),
        gates=_Gates(),
    )
    assert resolved.reason == L.CHOSE_UPCAST
    assert resolved.fetch_contract == "sdxl.diffusers-fp8-rowwise@1"
    assert resolved.transfer_saved_bytes == 3_400_000_000


def test_an_operator_pin_narrows_the_candidate_set_and_still_walks_the_ladder():
    """`--lane` is an override, not a bypass: a pinned lane the card cannot run
    says WHY, here rather than deeper in the load."""
    resolved = _loaded(TwoLaneModel).resolve(
        TwoLaneModel, card=AMPERE,
        verdicts=_Verdicts(staged=[FP8.stamp]), gates=_Gates(),
        contract="sdxl.diffusers-fp8-rowwise@1",
    )
    # The only candidate is floored out, so nothing has runnable bytes — and
    # the answer is a conversion ask, never a refusal.
    assert resolved.conversion is not None
    assert [r.reason for r in resolved.rejected] == [L.REJECT_SM_FLOOR]


def test_pinning_a_lane_the_model_does_not_declare_refuses_by_name():
    with pytest.raises(EndpointLoadError, match="no lane .* to pin"):
        _loaded(TwoLaneModel).resolve(
            TwoLaneModel, card=ADA, verdicts=_Verdicts(), gates=_Gates(),
            contract="flux1.diffusers-bf16@1")


def test_the_single_lane_fleet_path_is_unchanged():
    """Every endpoint in the fleet is single-lane today. `lane()` must still
    answer directly, and `resolve()` must agree with it."""
    loaded = _loaded(OneLaneModel)
    assert loaded.lane(OneLaneModel, "").stamp == "sdxl.diffusers-bf16@1"
    resolved = loaded.resolve(
        OneLaneModel, card=ADA, verdicts=_Verdicts(staged=[BF16.stamp]),
        gates=_Gates(),
    )
    assert resolved.contract_id == "sdxl.diffusers-bf16@1"
    assert resolved.rejected == ()


def test_the_declared_lane_object_is_carried_not_rebuilt():
    """pgw#1599's ask: `request=` and `resident=` must travel intact to varena
    and `min_sm` must keep one producer, so the resolver carries the declared
    lane rather than re-deriving anything off the Contract."""
    resolved = _loaded(TwoLaneModel).resolve(
        TwoLaneModel, card=ADA, verdicts=_Verdicts(staged=[FP8.stamp]),
        gates=_Gates(),
    )
    declared = resolved.declared
    assert declared.contract is FP8, "the Contract object itself, not a copy"
    assert declared.contract_id == FP8.stamp
    assert declared.min_sm == 89
