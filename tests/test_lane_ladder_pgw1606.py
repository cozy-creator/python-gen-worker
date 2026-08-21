"""pgw#1606 — the boot lane-selection ladder, proven against fabricated cards.

Every case here is a REAL card state the fleet meets: an Ada laptop card that
has fp8 silicon, an Ampere one that does not, a Blackwell one that has fp4, and
a host whose fp8 GEMM benchmark declined. The card is a value object precisely
so these run on a CPU box — the thing under test is the DECISION, and a
decision does not need a GPU to be wrong.

What is deliberately NOT mocked: the ranked lane table
(`models.execution_lanes`), the RULE->body map (`lane_ladder._RULE_BODY`), and
the sm floors, which come from `capability_floor_for_rule` reading the ratified
quant-rule documents. Faking those would prove the test's own arithmetic
instead of the platform's.

pgw#1621: the rows below are keyed on the v2 `(topology, quant)` stamp pair and
the floor falls out of the QUANT RULE, not out of a dtype spelling.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pytest

from gen_worker.serving import lane_ladder as L


def declared(topology: str, quant: str) -> L.DeclaredLane:
    """A REAL pgw#1599/pgw#1621 `DeclaredLane`, not a stand-in.

    `dtype` and `min_sm` are DERIVED here exactly as the declaration surface
    derives them — one producer each, `rule_dtype` and
    `capability_floor_for_rule`, both reading the ratified quant-rule document
    — so a test can never disagree with the platform about a floor.
    """
    from gen_worker.demand import GiB, const
    from gen_worker.models.tensor_layout_contract import (
        LayoutId,
        capability_floor_for_rule,
        rule_dtype,
    )
    from gen_worker.serving.lane_spec import lane

    return L.DeclaredLane(
        layout=LayoutId(topology=topology, quant=quant),
        topology=topology,
        quant=quant,
        dtype=rule_dtype(quant),
        min_sm=capability_floor_for_rule(quant),
        spec=lane(request=const(GiB(1.0))),
    )


BF16 = declared("sdxl.diffusers@1", "plain.bf16@1")
FP8 = declared("sdxl.diffusers@1", "cozy.fp8-rowwise@1")
NVFP4 = declared("sdxl.diffusers@1", "cozy.nvfp4-flat@1")

ADA = L.CardFacts(sm=89, vram_gb=8.0, name="RTX 4070 Laptop")
AMPERE = L.CardFacts(sm=86, vram_gb=24.0, name="RTX 3090")
HOPPER = L.CardFacts(sm=90, vram_gb=80.0, name="H100")
BLACKWELL = L.CardFacts(sm=100, vram_gb=180.0, name="B200")
CPU_ONLY = L.CardFacts(sm=0, name="no-cuda")


class Verdicts:
    """What is staged, and what the bind gate said about what is not."""

    def __init__(
        self,
        staged: Sequence[str] = (),
        derivable: Sequence[str] = (),
        incompatible: Sequence[str] = (),
        sizes: Mapping[str, int] | None = None,
    ) -> None:
        self._staged = set(staged)
        self._derivable = set(derivable)
        self._incompatible = set(incompatible)
        self._sizes = dict(sizes or {})

    def verdict(self, contract_id: str) -> str:
        if contract_id in self._incompatible:
            return L.VERDICT_INCOMPATIBLE
        if contract_id in self._staged:
            return L.VERDICT_SATISFIES
        if contract_id in self._derivable:
            return L.VERDICT_DERIVABLE
        return L.VERDICT_ABSENT

    def transfer_bytes(self, contract_id: str) -> int:
        return int(self._sizes.get(contract_id, 0))


class Gates:
    def __init__(self, w8a8: str = "rowwise",
                 w4a4: str = "blockwise") -> None:
        self._w8a8, self._w4a4 = w8a8, w4a4

    def w8a8_mode(self) -> str:
        return self._w8a8

    def w4a4_mode(self) -> str:
        return self._w4a4


ALL_THREE = (BF16, FP8, NVFP4)


def _reasons(resolved: L.ResolvedLane) -> list[tuple[str, str]]:
    return [(r.body, r.reason) for r in resolved.rejected]


def test_ada_with_everything_staged_takes_fp8_and_names_what_it_passed_over():
    """fp8 is canonical: on sm_89 with a gate-passed artifact it wins, and the confession must name nvfp4's rejection BY REASON, not merely omit it."""
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=ADA,
        verdicts=Verdicts(staged=[l.contract_id for l in ALL_THREE]),
        gates=Gates(),
    )
    assert resolved.body == "fp8-w8a8-dynamic"
    assert resolved.reason == L.CHOSE_GATE_PASSED
    assert resolved.declared is FP8, "the DeclaredLane must be carried, not rebuilt"
    assert _reasons(resolved) == [
        ("nvfp4-w4a4-static", L.REJECT_SM_FLOOR),
        ("bf16-w16a16", L.REJECT_OUTRANKED),
    ]
    line = resolved.confession()
    assert "LANE=fp8-w8a8-dynamic" in line
    assert "sm100" in line and "sm89" in line, (
        "a rejection must carry its NUMBERS: 'needs sm100, card is sm89'")


def test_blackwell_still_prefers_fp8_because_the_table_ranks_it_first():
    """nvfp4 on Blackwell is a RUNG, not a preference."""
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=BLACKWELL,
        verdicts=Verdicts(staged=[l.contract_id for l in ALL_THREE]),
        gates=Gates(),
    )
    assert resolved.body == "fp8-w8a8-dynamic"
    assert ("nvfp4-w4a4-static", L.REJECT_OUTRANKED) in _reasons(resolved)


def test_blackwell_takes_nvfp4_when_fp8_has_no_bytes():
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=BLACKWELL,
        verdicts=Verdicts(staged=[NVFP4.contract_id, BF16.contract_id]),
        gates=Gates(),
    )
    assert resolved.body == "nvfp4-w4a4-static"
    assert ("fp8-w8a8-dynamic", L.REJECT_NO_ARTIFACT) in _reasons(resolved)


def test_ampere_falls_to_bf16_on_the_derived_sm_floor():
    """sm_86 has no fp8 tensor cores. The floor comes from
    `capability_floor_for_rule` (`cozy.fp8-rowwise@1` -> 89), read off the
    rule document, never from a hand-written number — which is the whole point
    of pgw#1599 deleting hand-written floors and of pgw#1621 moving the number
    onto the rule."""
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=AMPERE,
        verdicts=Verdicts(staged=[l.contract_id for l in ALL_THREE]),
        gates=Gates(),
    )
    assert resolved.body == "bf16-w16a16"
    assert _reasons(resolved) == [
        ("fp8-w8a8-dynamic", L.REJECT_SM_FLOOR),
        ("nvfp4-w4a4-static", L.REJECT_SM_FLOOR),
    ]


def test_a_cpu_host_floors_out_every_quantized_rung_without_a_special_case():
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=CPU_ONLY,
        verdicts=Verdicts(staged=[l.contract_id for l in ALL_THREE]),
        gates=Gates(w8a8="", w4a4=""),
    )
    assert resolved.body == "bf16-w16a16"
    assert all(r.reason == L.REJECT_SM_FLOOR for r in resolved.rejected)


def test_an_unqualified_fp8_gemm_rejects_fp8_even_on_hopper_with_bytes():
    """`w8a8_gemm_mode()` returning '' is a REAL veto (a 4096-cubed GEMM that did not clear 1.10x over bf16)."""
    resolved = L.resolve_lane(
        declared=(BF16, FP8), card=HOPPER,
        verdicts=Verdicts(staged=[BF16.contract_id, FP8.contract_id]),
        gates=Gates(w8a8=""),
    )
    assert resolved.body == "bf16-w16a16"
    (rung,) = resolved.rejected
    assert rung.reason == L.REJECT_KERNEL_UNQUALIFIED
    assert "w8a8_gemm_mode()" in rung.detail


def test_an_unknown_quant_rule_is_a_named_rejection_not_a_crash():
    """A rule the fleet has no executor for is passed over LOUDLY.

    Built by hand rather than through `declared()`: `capability_floor_for_rule`
    REFUSES a handle the corpus does not carry (that refusal is proven in
    `test_lane_dtype_fence_pgw1606`), so an unratified rule cannot reach a
    header at all. What is under test here is the ladder's own arm — a row that
    somehow arrives carrying a rule `_RULE_BODY` has no row for must be a named
    rejection, not a KeyError on a booting pod.
    """
    from gen_worker.demand import GiB, const
    from gen_worker.models.tensor_layout_contract import LayoutId
    from gen_worker.serving.lane_spec import lane

    weird = L.DeclaredLane(
        layout=LayoutId(topology="sdxl.diffusers@1", quant="x.int8@1"),
        topology="sdxl.diffusers@1",
        quant="x.int8@1",
        dtype="int8",
        min_sm=0,
        spec=lane(request=const(GiB(1.0))),
    )
    resolved = L.resolve_lane(
        declared=(weird, BF16), card=ADA,
        verdicts=Verdicts(staged=[BF16.contract_id, weird.contract_id]),
        gates=Gates(),
    )
    assert resolved.body == "bf16-w16a16"
    assert ("?", L.REJECT_UNKNOWN_RULE) in _reasons(resolved)


def test_upcast_rung_fetches_the_fp8_bytes_and_measures_the_saving():
    resolved = L.resolve_lane(
        declared=(BF16, FP8), card=AMPERE,
        verdicts=Verdicts(
            staged=[BF16.contract_id, FP8.contract_id],
            sizes={BF16.contract_id: 6_900_000_000,
                   FP8.contract_id: 3_500_000_000},
        ),
        gates=Gates(),
    )
    assert resolved.body == "bf16-w16a16"
    assert resolved.reason == L.CHOSE_UPCAST
    assert resolved.upcast is True
    assert resolved.fetch_contract == FP8.contract_id
    assert resolved.transfer_saved_bytes == 3_400_000_000
    assert "saved=3.40GB" in resolved.confession()


def test_the_upcast_rung_refuses_to_claim_a_saving_it_cannot_measure():
    """A rung that cannot show its saving does not get to claim one."""
    resolved = L.resolve_lane(
        declared=(BF16, FP8), card=AMPERE,
        verdicts=Verdicts(staged=[BF16.contract_id, FP8.contract_id]),
        gates=Gates(),
    )
    assert resolved.reason == L.CHOSE_BASELINE
    assert resolved.upcast is False


def test_the_upcast_rung_declines_when_the_quantized_tree_is_not_smaller():
    resolved = L.resolve_lane(
        declared=(BF16, FP8), card=AMPERE,
        verdicts=Verdicts(
            staged=[BF16.contract_id, FP8.contract_id],
            sizes={BF16.contract_id: 3_000_000_000,
                   FP8.contract_id: 3_100_000_000},
        ),
        gates=Gates(),
    )
    assert resolved.reason == L.CHOSE_BASELINE


def test_no_bytes_for_any_lane_yields_a_priced_conversion_never_a_refusal():
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=ADA,
        verdicts=Verdicts(derivable=[FP8.contract_id]),
        gates=Gates(),
    )
    assert resolved.conversion is not None, "degrade, never refuse"
    assert resolved.reason == L.CHOSE_PENDING_CONVERSION
    assert resolved.conversion.to_contract == FP8.contract_id
    assert resolved.conversion.from_contract == FP8.contract_id
    assert "conversion=" in resolved.confession()


def test_the_conversion_targets_a_lane_THIS_CARD_CAN_RUN():
    """Nothing staged, and the card is floored out of fp8 and nvfp4."""
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=AMPERE, verdicts=Verdicts(), gates=Gates(),
    )
    assert resolved.conversion is not None
    assert resolved.conversion.to_contract == BF16.contract_id
    assert resolved.body == "bf16-w16a16"


def test_the_conversion_prefers_a_DERIVABLE_target_over_a_merely_runnable_one():
    """On Ada both fp8 and bf16 are runnable."""
    resolved = L.resolve_lane(
        declared=ALL_THREE, card=ADA,
        verdicts=Verdicts(derivable=[FP8.contract_id]), gates=Gates(),
    )
    assert resolved.conversion is not None
    assert resolved.conversion.to_contract == FP8.contract_id


def test_an_incompatible_lane_is_distinguished_from_an_absent_one():
    resolved = L.resolve_lane(
        declared=(BF16, FP8), card=ADA,
        verdicts=Verdicts(staged=[BF16.contract_id],
                          incompatible=[FP8.contract_id]),
        gates=Gates(),
    )
    assert resolved.body == "bf16-w16a16"
    (rung,) = resolved.rejected
    assert rung.reason == L.REJECT_INCOMPATIBLE


def test_an_empty_candidate_set_is_a_declaration_defect():
    with pytest.raises(L.LaneLadderError, match="candidate set is EMPTY"):
        L.resolve_lane(declared=(), card=ADA, verdicts=Verdicts(),
                       gates=Gates())


def test_a_single_declared_lane_still_walks_the_ladder():
    """The fleet is all single-lane today."""
    resolved = L.resolve_lane(
        declared=(BF16,), card=ADA,
        verdicts=Verdicts(staged=[BF16.contract_id]), gates=Gates(),
    )
    assert resolved.body == "bf16-w16a16"
    assert resolved.reason == L.CHOSE_BASELINE
    assert resolved.rejected == ()
    assert "rejected=none" in resolved.confession()


def test_author_declaration_order_carries_no_priority():
    forward = L.resolve_lane(
        declared=(BF16, FP8), card=ADA,
        verdicts=Verdicts(staged=[BF16.contract_id, FP8.contract_id]),
        gates=Gates(),
    )
    backward = L.resolve_lane(
        declared=(FP8, BF16), card=ADA,
        verdicts=Verdicts(staged=[BF16.contract_id, FP8.contract_id]),
        gates=Gates(),
    )
    assert forward.body == backward.body == "fp8-w8a8-dynamic"


def test_the_confession_would_change_if_the_ladder_changed_its_mind():
    """A guard that cannot fail proves nothing."""
    staged = Verdicts(staged=[l.contract_id for l in ALL_THREE])
    ada = L.resolve_lane(declared=ALL_THREE, card=ADA, verdicts=staged,
                         gates=Gates()).confession()
    ampere = L.resolve_lane(declared=ALL_THREE, card=AMPERE, verdicts=staged,
                            gates=Gates()).confession()
    assert ada != ampere
    assert "LANE=fp8-w8a8-dynamic" in ada
    assert "LANE=bf16-w16a16" in ampere
