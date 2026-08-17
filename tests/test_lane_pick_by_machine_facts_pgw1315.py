"""pgw#1315 deliverable 4 — the machine's FACTS pick the lane.

`research/machine-compatibility-design.md`, Q2 in force: *"Rank the lanes the
release binds by how much of their declared `recommended`, then `minimum`, the
machine satisfies. Ties broken by the author's ladder order. Run the best one,
degraded as needed."*

Two properties, and both are load-bearing:

* **the facts FLIP the choice.** One two-lane fixture, two machines, and the
  pick swaps — which no amount of "the first accepted handle" can produce.
  Red-verify by making the selector ignore its facts: the flip arm reds and
  the always-picks-something arms stay green.
* **it is a PICK, never a gate.** Under-minimum on every candidate returns
  the least-bad lane and the caller warns. `select_lane` has no arm that
  returns nothing, because refusing here would re-answer question 1 — which
  this design gives to the author — and would break always-runs.

Worker-side by construction. This never reaches the hub as a gate: preference
has exactly one authority, the author's ordered (GPU, lane) ladder, which is
hub config. What the worker holds is the slot's accepted SET, whose order
carries NO preference (§1.33 point 2) — so the tie-break here is determinism,
and this file says so out loud.
"""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.api.slot import Slot
from gen_worker.executor import Executor
from gen_worker.hostfacts import HostFacts
from gen_worker.models import machine_fit
from gen_worker.models.tensor_layout_contract import (
    CONTRACT_COZY_SVDQ_NVFP4_LR8,
    CONTRACT_PLAIN_BF16,
    LayoutRequirements,
    parse_layout_requirements,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

_WHERE = "test"


def _req(value: Any) -> LayoutRequirements:
    return parse_layout_requirements(value, where=_WHERE)


#: THE two-lane fixture. A bf16 lane that wants a big card and a 4-bit lane
#: that wants a NEW one: neither dominates, so which lane wins is a fact about
#: the machine and about nothing else.
_BF16 = machine_fit.LaneCandidate(
    CONTRACT_PLAIN_BF16,
    _req(LayoutRequirements(minimum="sm80+, vram48g",
                            recommended="sm90+, vram80g")))
_NVFP4 = machine_fit.LaneCandidate(
    CONTRACT_COZY_SVDQ_NVFP4_LR8, _req("sm100+, vram24g"))
_LANES = (_BF16, _NVFP4)

#: An H100: new enough for neither 4-bit floor to bite, roomy enough for bf16.
_H100 = machine_fit.MachineFacts(sm=90, vram_gb=80.0, host_ram_gb=192.0)
#: A 5090: NEWER than the H100 and far smaller. bf16's 48 GiB floor fails
#: here and the 4-bit lane's sm100 floor does not.
_RTX5090 = machine_fit.MachineFacts(sm=120, vram_gb=32.0, host_ram_gb=64.0)
#: An A6000: old AND small. Under BOTH minimums — the least-bad case.
_A6000 = machine_fit.MachineFacts(sm=86, vram_gb=24.0, host_ram_gb=128.0)


# ---------------------------------------------------------------------------
# 1. the facts flip the choice
# ---------------------------------------------------------------------------


def test_the_same_two_lanes_pick_DIFFERENTLY_on_two_machines() -> None:
    """THE arm this deliverable exists for. Same candidates, same order, two
    machines, opposite picks — the machine's facts are doing the choosing.

    Red-verify by having `select_lane` ignore `facts`: every other arm in this
    file still passes, and this one cannot.
    """
    assert machine_fit.select_lane(_LANES, _H100).lane == CONTRACT_PLAIN_BF16
    assert machine_fit.select_lane(
        _LANES, _RTX5090).lane == CONTRACT_COZY_SVDQ_NVFP4_LR8


def test_the_flip_survives_the_candidates_being_handed_over_reversed() -> None:
    """The pick is not "the first one that fits". Reversing the input order
    changes nothing about which lane the facts satisfy."""
    reversed_lanes = tuple(reversed(_LANES))
    assert machine_fit.select_lane(
        reversed_lanes, _H100).lane == CONTRACT_PLAIN_BF16
    assert machine_fit.select_lane(
        reversed_lanes, _RTX5090).lane == CONTRACT_COZY_SVDQ_NVFP4_LR8


def test_a_satisfied_minimum_outranks_a_better_recommended() -> None:
    """`recommended` ranks WITHIN the satisfied set and can never promote an
    under-minimum lane above a satisfied one — the declaration side already
    refuses a `recommended` below its own `minimum`, so one lexicographic key
    expresses both levels without `recommended` ever becoming a floor."""
    choice = machine_fit.select_lane(_LANES, _H100)
    ranked = {row.lane: row for row in choice.ranked}
    # bf16 is picked despite MISSING its own recommendation (sm90 clears
    # sm90+, but 80 GiB is exactly the recommended VRAM, so both hold here);
    # the 4-bit lane is under its minimum and cannot win regardless.
    assert ranked[CONTRACT_COZY_SVDQ_NVFP4_LR8].minimum.shortfalls
    assert not ranked[CONTRACT_PLAIN_BF16].minimum.shortfalls
    assert choice.lane == CONTRACT_PLAIN_BF16
    assert choice.forced is False


def test_recommended_breaks_a_tie_between_two_satisfied_lanes() -> None:
    """Both minimums hold, so the ONLY thing left to separate them is how much
    of each `recommended` this machine affords. That is `recommended` doing
    the one job the design leaves it: informing a pick, gating nothing."""
    plain = machine_fit.LaneCandidate(
        CONTRACT_PLAIN_BF16,
        _req(LayoutRequirements(minimum="sm80+", recommended="sm100+")))
    nvfp4 = machine_fit.LaneCandidate(
        CONTRACT_COZY_SVDQ_NVFP4_LR8,
        _req(LayoutRequirements(minimum="sm80+", recommended="sm86+")))
    choice = machine_fit.select_lane((plain, nvfp4), _A6000)
    assert choice.lane == CONTRACT_COZY_SVDQ_NVFP4_LR8
    assert choice.forced is False, (
        "both minimums hold; a recommendation shortfall is not a shortfall "
        "against a floor"
    )


# ---------------------------------------------------------------------------
# 2. it always picks — under-minimum is a warning, not a veto
# ---------------------------------------------------------------------------


def test_under_minimum_on_EVERY_lane_still_returns_one() -> None:
    """Always-runs is the answer to question 2, and it does not get an
    exception for "no lane fits". The pick is the LEAST-BAD and the caller
    warns; `forced` is how the caller knows to."""
    choice = machine_fit.select_lane(_LANES, _A6000)
    assert choice.lane, "select_lane may never answer 'none'"
    assert choice.forced is True
    assert choice.under_minimum, (
        "the pick being forced is exactly when the confession is owed"
    )


def test_the_least_bad_lane_is_the_one_that_misses_LESS() -> None:
    """Both under their own minimum, one by more. `forced` says the caller
    owes a warning either way; the pick is still the better of two bad
    answers, which is what always-runs on a small old card looks like."""
    starving = machine_fit.LaneCandidate(
        CONTRACT_PLAIN_BF16, _req("sm100+, vram80g, cuda13.0+"))
    nearly = machine_fit.LaneCandidate(
        CONTRACT_COZY_SVDQ_NVFP4_LR8, _req("sm100+"))
    choice = machine_fit.select_lane((starving, nearly), _A6000)
    assert choice.lane == CONTRACT_COZY_SVDQ_NVFP4_LR8
    assert choice.forced is True
    assert len(choice.under_minimum) == 1


def test_a_pure_tie_keeps_the_CALLERS_order() -> None:
    """Identical declarations cannot be separated by any fact, so the caller's
    order decides — deterministically, and this is NOT a preference (§1.33
    point 2: the accepted SET's order carries none). A caller holding the
    author's ladder passes ladder order and gets the author's tie-break; a
    caller holding only the set is choosing arbitrarily, and honestly."""
    same = "sm100+, vram80g"
    lanes = (machine_fit.LaneCandidate(CONTRACT_PLAIN_BF16, _req(same)),
             machine_fit.LaneCandidate(CONTRACT_COZY_SVDQ_NVFP4_LR8, _req(same)))
    assert machine_fit.select_lane(lanes, _A6000).lane == CONTRACT_PLAIN_BF16
    assert machine_fit.select_lane(
        tuple(reversed(lanes)), _A6000).lane == CONTRACT_COZY_SVDQ_NVFP4_LR8


def test_an_UNDECLARED_recommendation_is_not_scored_as_a_failure() -> None:
    """The asymmetry, NAMED rather than left to be discovered: a lane that
    declares no recommendation contributes no recommended shortfall, so it can
    outrank a lane whose recommendation this machine misses. That is NO
    DEFAULTS applied consistently — scoring silence as a failure would invent
    the author's declaration — and it is harmless because ranking is not
    gating: both lanes run either way."""
    choice = machine_fit.select_lane(_LANES, _A6000)
    assert choice.lane == CONTRACT_COZY_SVDQ_NVFP4_LR8
    ranked = {row.lane: row for row in choice.ranked}
    assert len(ranked[CONTRACT_PLAIN_BF16].minimum.shortfalls) == 1
    assert len(ranked[CONTRACT_COZY_SVDQ_NVFP4_LR8].minimum.shortfalls) == 1
    assert ranked[CONTRACT_COZY_SVDQ_NVFP4_LR8].recommended.shortfalls == ()


def test_no_candidates_is_no_lane_and_no_crash() -> None:
    choice = machine_fit.select_lane((), _H100)
    assert choice.lane == "" and choice.ranked == ()
    assert choice.forced is False and choice.under_minimum == ()


# ---------------------------------------------------------------------------
# 3. candidates come from the slot the release binds
# ---------------------------------------------------------------------------


class _Pipe:
    pass


def test_lane_candidates_reads_the_slots_accepted_handles() -> None:
    slot: Slot[Any] = Slot(
        _Pipe, selected_by="model",
        layouts={"*": (CONTRACT_PLAIN_BF16, CONTRACT_COZY_SVDQ_NVFP4_LR8)},
        layout_requirements={
            CONTRACT_COZY_SVDQ_NVFP4_LR8: "sm100+, vram24g",
            CONTRACT_PLAIN_BF16: LayoutRequirements(
                minimum="sm80+, vram48g", recommended="sm90+, vram80g"),
        },
    )
    candidates = machine_fit.lane_candidates(slot)
    assert {c.lane for c in candidates} == {
        CONTRACT_PLAIN_BF16, CONTRACT_COZY_SVDQ_NVFP4_LR8}
    assert machine_fit.select_lane(
        candidates, _RTX5090).lane == CONTRACT_COZY_SVDQ_NVFP4_LR8


def test_a_slot_that_declares_NO_requirement_offers_NO_lanes() -> None:
    """NO DEFAULTS. Ranking lanes nobody declared anything about would be the
    platform inventing the author's answer to question 1, which is precisely
    what this ruling takes away from the platform."""
    slot: Slot[Any] = Slot(_Pipe, selected_by="model",
                layouts={"*": (CONTRACT_PLAIN_BF16,)})
    assert machine_fit.lane_candidates(slot) == ()
    assert machine_fit.lane_candidates(None) == ()


# ---------------------------------------------------------------------------
# 4. through the production gate
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def generate(self, ctx: Any, payload: _In) -> None:  # pragma: no cover
        return None


def _gate(gpu_sm: str, vram_gb: int) -> Any:
    async def _send(_msg: pb.WorkerMessage) -> None:  # pragma: no cover
        return None

    spec = EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/sdxl")},
        slots={"pipeline": Slot(
            _Pipe, selected_by="model",
            layouts={"*": (CONTRACT_PLAIN_BF16, CONTRACT_COZY_SVDQ_NVFP4_LR8)},
            layout_requirements={
                CONTRACT_COZY_SVDQ_NVFP4_LR8: "sm100+, vram24g",
                CONTRACT_PLAIN_BF16: LayoutRequirements(
                    minimum="sm80+, vram48g", recommended="sm90+, vram80g"),
            },
        )},
        resources=Resources(gpu=True),
    )
    ex = Executor([spec], _send)
    ex.gate_functions(HostFacts(
        vram_total_bytes=vram_gb * 1024 ** 3,
        vram_free_bytes=(vram_gb - 4) * 1024 ** 3,
        gpu_sm=gpu_sm, cuda_version="12.8", torch_version="2.9.0",
    ))
    return ex.serve_plans["generate"]


def test_the_EXECUTOR_GATE_records_the_lane_the_facts_picked() -> None:
    """ASSERT EXECUTION, NOT REGISTRATION: the same fixture, driven through
    `gate_functions` — the seam that actually decides what this pod serves —
    picks differently on the two machines and lands the pick on the plan."""
    assert _gate("90", 80).lane == CONTRACT_PLAIN_BF16
    assert _gate("120", 32).lane == CONTRACT_COZY_SVDQ_NVFP4_LR8


def test_the_gate_never_withdraws_a_function_over_a_lane_pick() -> None:
    """Even the machine that is under BOTH lanes' minimums keeps serving. A
    lane pick that could decline would be a gate, and this is a pick."""
    plan = _gate("86", 24)
    assert plan.serveable is True
    assert plan.lane, "a forced pick is still a pick"
    assert plan.degraded is True and plan.warning
