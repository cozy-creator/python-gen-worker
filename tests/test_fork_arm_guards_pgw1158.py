"""pgw#1158 — a fork's ARMS must agree with the classes that cover them.

Found empirically by the endpoint-adaptation lane while writing sd15's and
sd2's declarations. It ran a severance experiment against the declaration's own
construction surface and found the SDK catches one error and silently accepts
two others:

    sd15's 14-class set applied to sd2   REFUSED  "graph class #0 sits on
                                                   UNSERVED arm cfg=True"
    a Fork claiming BOTH arms served,
      classes covering only one          ACCEPTED  <- no guard
    an unserved arm with NO `reason`     ACCEPTED  <- no guard

The second is a direction nothing asked. Every existing check reads
CLASS -> ARM ("does this class sit on an arm the fork declares?"), so a served
arm that no class covers passed silently: the declaration claims to serve it,
the mint traces nothing for it, and the first request on that arm finds a graph
that was never exported.

The third was a rule the docstring already asserted. `reason` was staged as
"optional BY DESIGN, FOR NOW ... phase 2 makes it required whenever `unserved`
is non-empty", and the reminder test even said it "will need deleting when that
lands — which is the point". Until it landed, the field read as a guarantee to
every endpoint author and enforced nothing.

Both refusals are DECLARATION-time, so they cost nothing at serve time.
"""

from __future__ import annotations

import pytest

from gen_worker import Compile, Dim, Fork, GraphClass
from gen_worker.api.export_contract import DeclarationError

_DIMS = (
    Dim("H", carried_by=(("hidden_states", 2),), multiple_of=2),
    Dim("B", carried_by=(("hidden_states", 0),)),
)


def _decl(forks, classes) -> Compile:
    return Compile(
        family="fam", targets=("transformer",), text_len=512,
        dims=_DIMS, forks=forks, classes=classes,
        shape_strategy="dynamic-collapse", warm_changes_key=False)


# ---------------------------------------------------------------------------
# (1) a SERVED arm no class covers
# ---------------------------------------------------------------------------


def test_a_served_arm_no_class_covers_is_REFUSED() -> None:
    """RED on master: ACCEPTED. The declaration says it serves cfg=True and
    nothing ever exports a graph for it."""
    with pytest.raises(DeclarationError, match="no graph class covers"):
        _decl(
            forks=(Fork("cfg", served=(False, True)),),
            classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),
                     GraphClass(dims={"H": 160, "B": 1}, fork={"cfg": False})))


def test_the_refusal_NAMES_the_arm_and_both_ways_out() -> None:
    """A refusal an author cannot act on is a refusal they route around."""
    with pytest.raises(DeclarationError) as err:
        _decl(
            forks=(Fork("cfg", served=(False, True)),),
            classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),))
    detail = str(err.value)
    assert "'cfg'" in detail and "'True'" in detail, detail
    assert "unserved" in detail and "reason" in detail, detail


def test_covering_BOTH_arms_constructs() -> None:
    """The guard must not refuse the shape it exists to require."""
    decl = _decl(
        forks=(Fork("cfg", served=(False, True)),),
        classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),
                 GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": True})))
    assert len(decl.classes) == 2


def test_a_declaration_with_NO_classes_is_silence_not_a_missing_class() -> None:
    """A shapes-declaring family has nothing that could cover an arm, and forks
    are legal there — they still key the graph. Caught by
    `test_registration_tightening_pgw1107`'s vocabulary sweep, which builds
    each field in isolation; an over-refusal here would have failed it."""
    decl = Compile(
        family="sweep", shapes=((64, 64),), targets=("transformer",),
        text_len=0,
        forks=(Fork("cfg", served=(False,), unserved=(True,),
                    reason="default_value"),))
    assert decl.forks[0].served == (False,)


# ---------------------------------------------------------------------------
# (2) an UNSERVED arm with no `reason`
# ---------------------------------------------------------------------------


def test_an_unserved_arm_without_a_reason_is_REFUSED() -> None:
    """RED on master: ACCEPTED, with `reason` left None. This is the rule the
    field's own docstring asserted and nothing enforced."""
    with pytest.raises(DeclarationError, match="declares no reason"):
        Fork("cfg", served=(False,), unserved=(True,))


def test_that_refusal_names_the_arm_and_the_vocabulary() -> None:
    with pytest.raises(DeclarationError) as err:
        Fork("cfg", served=(False,), unserved=(True,))
    detail = str(err.value)
    assert "'True'" in detail, detail
    assert "unpassed_arg" in detail and "checkpoint_config" in detail, detail


def test_a_served_only_fork_still_needs_no_reason() -> None:
    """It closes nothing, so it has nothing to justify — the guard must not
    widen into forks that make no guarantee."""
    assert Fork("edit", served=(0,)).reason is None


@pytest.mark.parametrize("reason", ["unpassed_arg", "absent_path",
                                    "default_value", "checkpoint_config"])
def test_every_declared_reason_is_accepted(reason: str) -> None:
    assert Fork("cfg", served=(False,), unserved=(True,),
                reason=reason).reason == reason


# ---------------------------------------------------------------------------
# The severance experiment, run on THESE guards
# ---------------------------------------------------------------------------


def test_the_guards_FIRE_on_the_two_shapes_master_accepted() -> None:
    """A validator that never fires is the thing this issue is about, so the
    experiment that found the gap is run against its own fix.

    Both constructions below are the ones measured as ACCEPTED on unmodified
    `origin/master`. If either stops raising, this guard has been severed and
    the declaration surface is back where it started.
    """
    accepted_on_master = (
        lambda: Fork("cfg", served=(False,), unserved=(True,)),
        lambda: _decl(
            forks=(Fork("cfg", served=(False, True)),),
            classes=(GraphClass(dims={"H": 90, "B": 1}, fork={"cfg": False}),)),
    )
    for i, construct in enumerate(accepted_on_master):
        with pytest.raises(DeclarationError):
            construct()
            pytest.fail(f"guard #{i} did not fire — it has been severed")
