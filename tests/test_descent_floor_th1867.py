from __future__ import annotations

import pytest

from gen_worker.models import rung

ALL_TOKENS = (
    [None, "", "off", "vae_only", "auto", "resident", "not-a-rung"]
    + [r.name for r in rung.LADDER]
)

FLOORS = {
    rung.FLOOR_LADDER_EXHAUSTED,
}


@pytest.mark.parametrize("token", ALL_TOKENS)
def test_a_rung_or_a_floor_never_both_and_never_neither(token: str) -> None:
    nxt = rung.descend(token)
    floor = rung.descent_floor(token)
    assert (nxt is None) != (floor is None), (
        f"{token!r} -> rung={nxt}, floor={floor}: exactly one of the two "
        "must be set"
    )
    assert floor is None or floor in FLOORS


UPPER_RUNGS = [None, "", "off", "vae_only", "resident"] + [
    r.name for r in rung.LADDER if rung.descend(r.name) is not None
]


@pytest.mark.parametrize("token", UPPER_RUNGS)
def test_upper_rungs_descend_with_no_floor(token: str) -> None:
    """The half that matters as much as the token itself."""
    assert rung.descend(token) is not None
    assert rung.descent_floor(token) is None


def test_the_last_offload_rung_descends_to_cpu() -> None:
    assert rung.descend("sequential") is rung.CPU
    assert rung.descent_floor("sequential") is None


def test_the_cpu_unexecutable_floor_is_deleted_with_its_cause() -> None:
    assert not hasattr(rung, "FLOOR_CPU_RUNG_UNEXECUTABLE")
    assert rung.LADDER[-1] is rung.CPU
    assert rung.PLACEMENT_LADDER[-1] == rung.CPU.name
    assert any(rung.descend(t) is rung.CPU for t in ALL_TOKENS)


def test_standing_on_the_bottom_rung_does_not_climb() -> None:
    """``cpu`` is the end of the placement tail, so the resident-token arm must not claim it."""
    assert rung.descend("cpu") is None
    assert rung.descent_floor("cpu") == rung.FLOOR_LADDER_EXHAUSTED


def test_the_walk_only_ever_descends() -> None:
    for r in rung.LADDER:
        nxt = rung.descend(r.name)
        if nxt is None:
            continue
        assert rung.LADDER.index(nxt) > rung.LADDER.index(r), (
            f"{r.name} -> {nxt.name} climbs the ladder"
        )


@pytest.mark.parametrize("token", [None, "", "resident", "model_offload", "group_offload"])
def test_no_declaration_can_stop_the_walk_th1867(token: str) -> None:
    """``FLOOR_STRICT_VRAM_TRUNCATED`` is DELETED, and so is the only thing that could produce it."""
    assert not hasattr(rung, "FLOOR_STRICT_VRAM_TRUNCATED"), (
        "the declaration-truncation floor is back; §1.35 forbids an author "
        "declaration ending a descent the ladder could continue")
    assert rung.descend(token) is not None
    assert rung.descent_floor(token) is None


ABOLISHED = {
    "insufficient_vram", "compute_capability_unmet", "cuda_unavailable",
    "hardware_unmet", "gpu_capability_incompatible", "gpu_model_mismatch",
    "compute_size_mismatch", "no_runnable_precision",
    "execution_lane_ladder_exhausted", "hardware_unsatisfiable",
}


def test_no_floor_token_rejoins_the_abolished_vocabulary() -> None:
    """§1.35 second amendment: *"no new unsupported card/model combination vocabulary may be introduced"*."""
    assert FLOORS.isdisjoint(ABOLISHED)
    assert rung.FLOOR_LADDER_EXHAUSTED != "execution_lane_ladder_exhausted"
