"""A declared target that NO Input row reaches is refused.

`target_inputs` scopes by ``not inp.targets or target in inp.targets``, so a
target that every row scopes AWAY from still gets a mint plan — one with no
declared inputs at all, whose trace has nothing to feed. The cost is not merely
a confusing declaration: a whole plan is minted for a target the author gave
nothing to trace.

Deliberately independent of the OTHER half — whether an UNTARGETED row should
fan out to every declared target is an open defaulting semantic. No defaulting
anyone would choose makes an input-less target correct, so refusing it cannot
freeze that question in either direction.
"""

from __future__ import annotations

import pytest

from gen_worker.api.decorators import Compile
from gen_worker.api.export_contract import (
    DeclarationError, Dim, GraphClass, Input)


def _decl(**over) -> Compile:
    base = dict(
        family="fam", text_len=512,
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        shape_strategy="static-rows", warm_changes_key=False)
    base.update(over)
    return Compile(**base)  # type: ignore[arg-type]


def test_a_target_no_input_row_reaches_is_REFUSED() -> None:
    """RED on master: ACCEPTED, and it mints a plan for the starved target."""
    with pytest.raises(DeclarationError, match="NO Input row reaches it"):
        _decl(
            targets=("transformer", "vae.decode"),
            inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                          targets=("transformer",)),))


def test_the_refusal_names_the_TARGET_and_where_the_rows_went() -> None:
    """An author reading it must be able to act without re-deriving the scope
    map: the starved target by name, and what every row scoped itself to."""
    with pytest.raises(DeclarationError) as err:
        _decl(
            targets=("transformer", "vae.decode"),
            inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                          targets=("transformer",)),))
    detail = str(err.value)
    assert "'vae.decode'" in detail, detail
    assert "['transformer']" in detail, detail
    # ...and all three ways out, because a refusal with one exit gets routed
    # around by whichever exit the author thought of first.
    assert "Scope a row" in detail and "drop the target" in detail, detail
    assert "untargeted" in detail, detail


# ---------------------------------------------------------------------------
# It must not widen: the shapes below are legitimate and stay constructible
# ---------------------------------------------------------------------------


def test_an_UNTARGETED_row_reaching_every_target_is_untouched() -> None:
    """Today's defaulting — the half that is NOT ruled here. Whatever it
    becomes, this row must keep constructing until that ruling lands, or this
    guard would have decided the open question by refusing one side of it."""
    decl = _decl(
        targets=("transformer", "vae.decode"),
        inputs=(Input("hidden_states", shape=("B", 4), dtype="model"),))
    assert decl.targets == ("transformer", "vae.decode")


def test_a_declaration_with_NO_inputs_at_all_is_a_different_case() -> None:
    """`inputs=()` is a declaration that states no ingress anywhere — already
    handled elsewhere and legal here. The guard asks "did the rows MISS this
    target", which is only a question when rows exist."""
    assert _decl(targets=("transformer",), inputs=()).inputs == ()


def test_every_target_scoped_explicitly_is_fine() -> None:
    """The shape the guard exists to require."""
    decl = _decl(
        targets=("transformer", "vae.decode"),
        inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                      targets=("transformer",)),
                Input("latent", shape=("B", 4), dtype="model",
                      targets=("vae.decode",))))
    assert len(decl.inputs) == 2


def test_the_guard_FIRES_on_the_shape_master_accepted() -> None:
    """The severance experiment, run on this guard: master accepted the
    construction below and minted a plan for the starved target. If this stops
    raising, the guard has been severed."""
    with pytest.raises(DeclarationError):
        _decl(
            targets=("transformer", "vae.decode"),
            inputs=(Input("hidden_states", shape=("B", 4), dtype="model",
                          targets=("transformer",)),))
        pytest.fail("the guard did not fire — it has been severed")
