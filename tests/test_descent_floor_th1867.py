"""th#1867 — A DESCENT THAT RUNS OUT MUST NAME ITS FLOOR.

The proactive fit ladder is being deleted (``Resources.vram_gb_hint`` is an
ESTIMATE deciding placement before anything is measured — §4.33), which makes
the reactive OOM walk in :mod:`gen_worker.models.rung` the ONLY ladder. Its
bottom therefore stops being theoretical, and falling off it must be a typed,
visible refusal naming OUR code rather than a silent slide into a rung nothing
can run — the loud-estimate-error-for-quiet-execution-error trade §1.35 and
§1.36 keep rejecting.

Deliberately torch-free: it imports ``rung`` and nothing else, so it runs on a
CPU-only runner and on a laptop with no CUDA. The floor it guards is exercised
on exactly the machines least able to reach it in production.
"""

from __future__ import annotations

import pytest

from gen_worker.models import rung

#: Every token a caller can be handed: the declared rungs, the resident
#: placement flavors ``memory`` sets, the never-prepped empty string, and junk.
ALL_TOKENS = (
    [None, "", "off", "vae_only", "auto", "resident", "not-a-rung"]
    + [r.name for r in rung.LADDER]
)

FLOORS = {
    rung.FLOOR_CPU_RUNG_UNEXECUTABLE,
    rung.FLOOR_LADDER_EXHAUSTED,
}


# --- the two halves cannot disagree ----------------------------------------

@pytest.mark.parametrize("token", ALL_TOKENS)
def test_a_rung_or_a_floor_never_both_and_never_neither(token: str) -> None:
    """th#1867 §3.0 — diagnosis and action read ONE verdict.

    A descent that returned a rung has nothing to explain; a descent that
    returned None must say why. Two implementations of the same question drift;
    this asserts the property that makes the drift impossible to express.
    """
    nxt = rung.descend(token)
    floor = rung.descent_floor(token)
    assert (nxt is None) != (floor is None), (
        f"{token!r} -> rung={nxt}, floor={floor}: exactly one of the two "
        "must be set"
    )
    assert floor is None or floor in FLOORS


# --- a token that fired everywhere would be noise ---------------------------

#: Every rung above the floor, DERIVED so a rung added or deleted is covered
#: automatically. Hardcoding this list is how it came to name `nf4` — a rung
#: pgw#1206 D deleted while this file sat on the merge queue, after which the
#: case still passed, for a different reason (an unknown token also descends to
#: `model_offload`). A green assertion about a symbol that no longer exists is
#: the defect class this file was written to guard.
UPPER_RUNGS = [None, "", "off", "vae_only", "resident"] + [
    r.name for r in rung.LADDER if rung.descend(r.name) is not None
]


@pytest.mark.parametrize("token", UPPER_RUNGS)
def test_upper_rungs_descend_with_no_floor(token: str) -> None:
    """The half that matters as much as the token itself.

    A floor reported on rungs that descend cleanly is noise, and noise is how a
    real floor stops being read.
    """
    assert rung.descend(token) is not None
    assert rung.descent_floor(token) is None


# --- the floor itself -------------------------------------------------------

def test_the_last_executable_rung_names_the_cpu_floor() -> None:
    """``sequential`` is the real bottom; ``cpu`` below it is plan-time only."""
    assert rung.descend("sequential") is None
    assert rung.descent_floor("sequential") == rung.FLOOR_CPU_RUNG_UNEXECUTABLE


def test_the_cpu_floor_exists_only_because_cpu_is_unreachable() -> None:
    """RED-PROOF FOR pgw#1212's IMPLEMENTER, not a restatement of the ladder.

    ``FLOOR_CPU_RUNG_UNEXECUTABLE`` is a confession that ``cpu`` is declared,
    priced at 40x and not executable. When pgw#1212 makes the rung real the
    token must be DELETED — a floor that no longer exists must not be left
    pointing somewhere else, which is how a diagnostic outlives its cause.
    This goes red the moment ``descend`` starts handing out ``cpu``.
    """
    assert rung.LADDER[-1] is rung.CPU
    assert rung.PLACEMENT_LADDER[-1] == rung.SEQUENTIAL.name
    assert rung.CPU.name not in rung.PLACEMENT_LADDER
    assert all(rung.descend(t) is not rung.CPU for t in ALL_TOKENS)


def test_standing_on_the_bottom_rung_does_not_climb() -> None:
    """``cpu`` is below the placement tail, so the resident-token arm must not
    claim it. Before th#1867 this returned ``model_offload`` — the walk went
    UP, which under pgw#1212 (where ``cpu`` becomes reachable) is a cycle."""
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


# --- opposite findings may not share a token --------------------------------

@pytest.mark.parametrize("token", [None, "", "resident", "model_offload", "group_offload"])
def test_no_declaration_can_stop_the_walk_th1867(token: str) -> None:
    """``FLOOR_STRICT_VRAM_TRUNCATED`` is DELETED, and so is the only thing
    that could produce it.

    The two findings this file kept apart were "the author's opt-out stopped
    the walk" and "our ladder ran out". §2.4 ruling 4 deleted the first one
    outright — it was a card requirement in softer words — so every remaining
    floor names OUR code by construction. From any of these tokens the walk
    must now find a real rung.
    """
    assert not hasattr(rung, "FLOOR_STRICT_VRAM_TRUNCATED"), (
        "the declaration-truncation floor is back; §1.35 forbids an author "
        "declaration ending a descent the ladder could continue")
    assert rung.descend(token) is not None
    assert rung.descent_floor(token) is None


# --- the floor names our code, never the card -------------------------------

#: The decline vocabulary §1.35's amendments abolish — every one of these says
#: "this card cannot run this model". th#1867 §2.6/§2.5 enumerates them.
ABOLISHED = {
    "insufficient_vram", "compute_capability_unmet", "cuda_unavailable",
    "hardware_unmet", "gpu_capability_incompatible", "gpu_model_mismatch",
    "compute_size_mismatch", "no_runnable_precision",
    "execution_lane_ladder_exhausted", "hardware_unsatisfiable",
}


def test_no_floor_token_rejoins_the_abolished_vocabulary() -> None:
    """§1.35 second amendment: *"no new unsupported card/model combination
    vocabulary may be introduced"*. These tokens are what the hub groups on, so
    the constraint belongs on the tokens and not only on the prose beside them.
    Each surviving floor names a thing WE did: a rung we cannot execute, the end
    of a ladder we wrote, an opt-out an author declared.
    """
    assert FLOORS.isdisjoint(ABOLISHED)
    assert rung.FLOOR_LADDER_EXHAUSTED != "execution_lane_ladder_exhausted"
