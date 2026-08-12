"""pgw#853 / ie#591: `unserved` meant five things; make the kind DECLARED.

THE MEASUREMENT THIS EXISTS FOR. A fleet audit asked of every `unserved` fork
arm whether it is closed by an ABSENT CODE PATH or by a DEFAULT VALUE. Seven
distinct forks, and the answer was that ``Fork(unserved=)`` was recording
**five materially different guarantees as one word**, only one of which means
"unreachable":

===================  ==========================================  ===========
kind                 fleet example                               strength
===================  ==========================================  ===========
absent_path          flux2 ``kv_cache`` — pipeline class never    the one the
                     imported; 0 mentions anywhere                word implies
unpassed_arg         ltx ``stg`` / ``isolate_modalities``        one code edit
default_value        ltx ``cfg`` — plumbed end to end, only the  **no code
                     number is 1.0                                edit at all**
checkpoint_config    wan ``expand_timesteps``                     moves with
                                                                  the ckpt
eager_by_choice      qwen ``condition_images`` — reachable AND    not a hazard
                     served, excluded from the COMPILED set       at all
===================  ==========================================  ===========

and ``why=`` was free prose doing a machine's job — prose that ltx was shown
wrong on TWICE in one day. A guarantee whose strength is only discoverable by
reading prose is not a guarantee; it is a note.

The property that makes the field worth having is that it makes the WEAK ones
greppable instead of readable: :func:`weak_arms` answers in a call what took a
day to answer by hand.
"""

from __future__ import annotations

import pytest

from gen_worker import Compile, Dim, Fork, GraphClass, Input
from gen_worker.api.export_contract import (
    FORK_REASONS, WEAK_FORK_REASONS, DeclarationError, export_declaration,
    register_export_declaration, reset_export_declarations, weak_arms,
    weak_arms_by_family,
)


def _decl(*forks: Fork, family: str = "harness-fork-family",
          blockers: tuple = ()) -> Compile:
    fork_names = {f.name for f in forks}
    return Compile(
        family=family, targets=("transformer",), text_len=0, blockers=blockers,
        shapes=((64, 64),),
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        forks=forks,
        classes=(GraphClass(
            dims={"B": 1},
            fork={f.name: f.served[0] for f in forks} if fork_names else {}),),
        inputs=(Input("hidden_states", shape=("B", 4, 6), dtype="model"),),
        shape_strategy="static-rows", warm_changes_key=False,
    )


# ---------------------------------------------------------------------------
# 1. THE VOCABULARY
# ---------------------------------------------------------------------------


def test_the_five_kinds_are_the_audit_s_five() -> None:
    assert FORK_REASONS == (
        "absent_path", "unpassed_arg", "default_value",
        "checkpoint_config", "eager_by_choice")


@pytest.mark.parametrize("kind", FORK_REASONS)
def test_every_declared_kind_is_accepted(kind: str) -> None:
    fork = Fork("f", served=(False,), unserved=(True,), reason=kind)
    assert fork.reason == kind


def test_an_undeclared_kind_is_refused_by_name() -> None:
    with pytest.raises(DeclarationError) as excinfo:
        Fork("f", served=(False,), unserved=(True,), reason="probably_fine")
    assert "probably_fine" in str(excinfo.value)
    assert "absent_path" in str(excinfo.value), "the refusal must list the kinds"


def test_a_reason_with_nothing_to_close_is_refused() -> None:
    """A reason explains what keeps an arm CLOSED. `eager_by_choice` is the
    one kind that can describe a fork with no unserved arm — qwen's edit lane
    is reachable and served, just not compiled."""
    with pytest.raises(DeclarationError):
        Fork("f", served=(False,), reason="absent_path")
    assert Fork("f", served=(0,), reason="eager_by_choice").reason == "eager_by_choice"


def test_prose_is_KEPT_alongside_the_kind_not_replaced() -> None:
    """The kind is for machines, the prose for humans. The failure this pair
    fixes was prose doing a machine's job, not prose existing."""
    fork = Fork("cfg", served=(False,), unserved=(True,),
                reason="default_value",
                why="guidance flows to the pipeline; only the value is 1.0")
    assert fork.reason == "default_value"
    assert "only the value is 1.0" in fork.why


# ---------------------------------------------------------------------------
# 2. THE QUERY — the reason the field exists
# ---------------------------------------------------------------------------


def test_weak_arms_finds_exactly_the_value_closed_ones() -> None:
    decl = _decl(
        Fork("kv_cache", served=(False,), unserved=(True,), reason="absent_path"),
        Fork("stg", served=(False,), unserved=(True,), reason="unpassed_arg"),
        Fork("cfg", served=(False,), unserved=(True,), reason="default_value"),
        Fork("expand", served=(False,), unserved=(True,), reason="checkpoint_config"),
    )
    assert [f.name for f in weak_arms(decl)] == ["stg", "cfg"]
    assert set(WEAK_FORK_REASONS) == {"unpassed_arg", "default_value"}


def test_the_UNANSWERED_state_is_now_UNCONSTRUCTIBLE() -> None:
    """`None` used to mean "the question has not been answered", and reading
    that as evidence of a weak guarantee would have been a guess. pgw#1157
    removes the ambiguity at its source instead of interpreting it: an
    unserved arm cannot be declared without saying what closes it, so
    `weak_arms` never sees an unanswered fork again."""
    with pytest.raises(DeclarationError, match="declares no reason"):
        _decl(Fork("mystery", served=(False,), unserved=(True,)))


def test_a_served_only_fork_is_never_weak() -> None:
    decl = _decl(Fork("edit", served=(0,), reason="eager_by_choice"))
    assert weak_arms(decl) == ()


def test_the_fleet_query_ANSWERS_for_a_family_whose_declaration_REFUSES() -> None:
    """A blocked family must not take a fleet-wide query down — pgw#853's
    whole point is that a refusal to MINT is not a refusal to everything else.

    pgw#1107 makes that strictly better than "skipped": the refusal is DATA on
    the declaration, so a blocked family's weak arms are still readable. A
    sweep asking "which families are one payload field from an unserved arm?"
    used to answer with a hole exactly where the answer mattered most.
    """
    from gen_worker.api.export_contract import MintBlocker

    blocker = MintBlocker(
        id="B1-harness", what="one open question",
        evidence="harness fixture", resolves_when="it is measured")

    reset_export_declarations()
    try:
        register_export_declaration(_decl(
            Fork("cfg", served=(False,), unserved=(True,),
                 reason="default_value"),
            family="harness-blocked", blockers=(blocker,)))
        register_export_declaration(_decl(
            Fork("cfg", served=(False,), unserved=(True,),
                 reason="default_value"), family="harness-weak"))

        found = weak_arms_by_family()

        assert [f.name for f in found["harness-weak"]] == ["cfg"]
        assert [f.name for f in found["harness-blocked"]] == ["cfg"]
        decl = export_declaration("harness-blocked")
        assert decl is not None
        assert [b.id for b in decl.open_blockers] == ["B1-harness"]
    finally:
        reset_export_declarations()


# ---------------------------------------------------------------------------
# 3. THE pgw#846 GATE — this changes what a declaration SAYS, never what it
#    traces. The fleet-wide half ran off-tree; this pins the mechanism.
# ---------------------------------------------------------------------------


def test_an_unannotated_fork_serialises_exactly_as_before() -> None:
    """`reason` is OMITTED from as_row() when absent, which is what makes
    every pre-existing declaration byte-identical. Verified across the real
    fleet too: flux2 4b/9b, wan x3, ltx, qwen, z-image and sdxl all
    fingerprinted identical before and after this field existed."""
    # Since pgw#1157 only a SERVED-ONLY fork may omit `reason` — it closes
    # nothing, so it has nothing to justify. That is the shape whose row must
    # stay byte-identical.
    row = Fork("kv_cache", served=(False,)).as_row()
    assert "reason" not in row
    assert set(row) == {"name", "served", "unserved", "source", "targets"}


def test_annotating_changes_neither_the_entries_nor_the_cell_contract() -> None:
    """The two mechanisms that make annotation admissible under pgw#846:
    entry names carry fork COORDINATES (from `served`), and the ck2 contract
    axis digests no fork rows at all."""
    from gen_worker.aot_declaration import cell_plans, plan_entry_name
    from gen_worker.compile_cache import declared_compile_facts

    plain = _decl(Fork("kv_cache", served=(False,), unserved=(True,),
                       reason="absent_path"))
    annotated = _decl(Fork("kv_cache", served=(False,), unserved=(True,),
                           reason="unpassed_arg", why="prose"))

    assert [plan_entry_name(p) for p in cell_plans(plain)] == \
        [plan_entry_name(p) for p in cell_plans(annotated)]
    assert declared_compile_facts(plain) == declared_compile_facts(annotated)


# ---------------------------------------------------------------------------
# 4. PHASE 2 — the shape it will take, pinned so it is not forgotten
# ---------------------------------------------------------------------------


def test_phase_2_IS_HERE_an_unserved_arm_must_name_what_closes_it() -> None:
    """The reminder this replaces said it "will need deleting when that lands
    — which is the point". pgw#1157 landed it.

    An unserved arm is a CLOSED arm, and a declaration that cannot say what
    closes it has not made the guarantee it appears to make. Refused where it
    is written, so it costs nothing at serve time.
    """
    with pytest.raises(DeclarationError, match="declares no reason"):
        _decl(Fork("mystery", served=(False,), unserved=(True,)))

    # ...and the refusal NAMES the arm and the vocabulary that would close it.
    with pytest.raises(DeclarationError) as err:
        Fork("mystery", served=(False,), unserved=(True,))
    assert "'True'" in str(err.value)
    assert "unpassed_arg" in str(err.value)

    # A served-only fork closes nothing and still needs no reason.
    assert _decl(Fork("edit", served=(0,))) is not None
