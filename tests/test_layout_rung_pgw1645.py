"""pgw#1645 -- the layout rung a worker is ON, and the one it is EARNING.

Every arm reads the rung off ARTIFACT METADATA, which is where both facts
actually live (`declared_input_layout` and `layout_wishlist`, tcg#83), and
never off a caller's belief about the artifact. The deliverability arms move
the one real source of truth there is today: whether a layout-applying fill is
REACHABLE FROM THIS PROCESS. It is not (varena#13), so the decline arm is what
production takes; installing the capability is what makes EARNING reachable,
and that is the same branch production will take the day varena#13 lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

from gen_worker.serving import layout_rung as rung_mod
from gen_worker.serving.layout_rung import LayoutState, read_rung

IDENTITY = "torch.contiguous@1"
CHANNELS_LAST = "torch.channels_last-2d@1"


def _metadata(
    *, declared: str = IDENTITY, wishlist: List[Dict[str, Any]] | None = None
) -> Dict[str, Any]:
    return {
        "declared_input_layout": declared,
        "layout_wishlist": [] if wishlist is None else wishlist,
    }


def _wish(fqn: str, morphism: str, order: List[int]) -> Dict[str, Any]:
    return {"fqn": fqn, "morphism": morphism, "stride_order": order}


@pytest.fixture()
def fill_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a layout-applying fill in this process's reach.

    Moves the SOURCE the probe reads -- the vendored tensorfs module's own
    surface -- rather than handing the predicate an argument. varena#13 makes
    this attribute real; until then this is the only way the EARNING branch can
    be reached, and it must be reachable or the vocabulary carries a member
    nothing can produce.
    """

    from gen_worker._vendor import tensorfs as vendored

    monkeypatch.setattr(vendored, "fill", lambda *a, **k: None, raising=False)


# -- the four settled positions ----------------------------------------------


def test_no_wish_is_the_ideal_not_an_absence() -> None:
    """An artifact whose mint asked for nothing IS at its ideal layout. There
    is no rung above it, so nothing is owed and nothing should read as
    pending."""

    rung = read_rung("g", _metadata())
    assert rung.state is LayoutState.NO_WISH
    assert rung.settled and not rung.earning
    assert rung.ideal == ""
    assert rung.served == IDENTITY


def test_a_wish_the_artifact_already_satisfies_is_AT_IDEAL() -> None:
    rung = read_rung(
        "g",
        _metadata(
            declared=CHANNELS_LAST,
            wishlist=[_wish("conv.weight", CHANNELS_LAST, [3, 0, 2, 1])],
        ),
    )
    assert rung.state is LayoutState.AT_IDEAL
    assert rung.settled and not rung.earning


def test_an_unratified_wish_is_a_CANDIDATE_and_no_name_is_invented() -> None:
    """The permanent fallback path. The mint compiled against the stored layout
    and the order rides out for a human to ratify -- machines derive along
    ratified morphisms and never invent one."""

    rung = read_rung(
        "g", _metadata(wishlist=[_wish("w", "", [1, 0, 2, 3])])
    )
    assert rung.state is LayoutState.CANDIDATE
    assert rung.ideal == ""
    assert rung.settled and not rung.earning
    # The order survives, unnamed, so ratification has something to read.
    assert rung.wishes[0].stride_order == (1, 0, 2, 3)
    assert not rung.wishes[0].ratified


def test_two_ratified_wishes_are_NO_SINGLE_IDEAL_not_a_coin_flip() -> None:
    """A re-mint against one of two wanted arrangements MOVES the copies rather
    than deleting them, which is worse than not re-minting. Picking silently is
    the failure; saying there is no single ideal is the fix."""

    rung = read_rung(
        "g",
        _metadata(
            wishlist=[
                _wish("a", CHANNELS_LAST, [3, 0, 2, 1]),
                _wish("b", "torch.channels_last-3d@1", [4, 0, 3, 2, 1]),
            ]
        ),
    )
    assert rung.state is LayoutState.NO_SINGLE_IDEAL
    assert rung.ideal == ""
    assert rung.settled and not rung.earning
    assert CHANNELS_LAST in rung.detail


# -- deliverability: the two arms of one predicate ---------------------------


def test_with_no_fill_in_reach_a_ratified_wish_is_a_TYPED_DECLINE() -> None:
    """What production does TODAY, and it must not be silent. `tensorfs-py`
    exports no `fill` and varena implements no sink, so the arrangement cannot
    be delivered to VRAM at all -- and the confession says exactly that, names
    varena#13, and carries the tensorfs#157 measurement that will price the
    decision once a fill exists."""

    assert rung_mod.fill_path() is None, "a fill became reachable; retire this arm"
    rung = read_rung(
        "g", _metadata(wishlist=[_wish("conv.weight", CHANNELS_LAST, [3, 0, 2, 1])])
    )
    assert rung.state is LayoutState.DECLINED
    assert rung.ideal == CHANNELS_LAST
    assert rung.settled and not rung.earning
    assert "varena#13" in rung.detail
    assert "tensorfs#157" in rung.detail
    # A decline is a SETTLED position with a stated cause. Reading it as
    # pending would make a permanent state look like a stuck mint queue.
    assert rung.settled


def test_with_a_fill_in_reach_the_same_wish_is_EARNING(fill_installed: None) -> None:
    """The rung the design exists to make expressible: serving the stored
    layout while the ideal-layout mint is pending reads as EARNING, not as
    broken. Same metadata, same branch, one real capability moved."""

    assert rung_mod.fill_path() is not None
    rung = read_rung(
        "g", _metadata(wishlist=[_wish("conv.weight", CHANNELS_LAST, [3, 0, 2, 1])])
    )
    assert rung.state is LayoutState.EARNING
    assert rung.earning and not rung.settled
    assert rung.served == IDENTITY and rung.ideal == CHANNELS_LAST
    assert "EARNING" in rung.detail


def test_the_confession_names_both_layouts_and_survives_serialization(
    fill_installed: None,
) -> None:
    """An operator reading a slow pod's status has to be able to tell "on a
    lower rung, earning the higher one" from "broken", and the hub groups on
    the state value, so it has to reach the wire intact."""

    rung = read_rung(
        "g", _metadata(wishlist=[_wish("conv.weight", CHANNELS_LAST, [3, 0, 2, 1])])
    )
    line = rung.line()
    assert "LAYOUT_RUNG=earning" in line
    assert f"served={IDENTITY}" in line and f"ideal={CHANNELS_LAST}" in line

    facts = rung.facts()
    assert facts["state"] == "earning"
    assert facts["served"] == IDENTITY and facts["ideal"] == CHANNELS_LAST
    assert facts["wishes"] == [
        {"fqn": "conv.weight", "morphism": CHANNELS_LAST, "stride_order": [3, 0, 2, 1]}
    ]


def test_the_state_values_are_the_wire_contract_the_hub_groups_on() -> None:
    """`EagerPhase`'s rule, applied to this vocabulary before anything joins to
    it: a member may be ADDED, and renaming one orphans history."""

    assert sorted(str(state) for state in LayoutState) == [
        "at_ideal",
        "candidate",
        "declined",
        "earning",
        "no_single_ideal",
        "no_wish",
    ]


# -- the walk, against REAL torchcg structure --------------------------------


def test_the_rungs_are_read_off_the_ARTIFACT_THAT_IS_SERVING(tmp_path: Path) -> None:
    """`rungs_of` walks the adopt session's own dispatcher registry and reads
    each armed runner's VERIFIED metadata — the block the loader already
    validated on the way in. Nothing is re-materialized and nothing is
    re-derived, so the artifact answering requests is the artifact answering
    here.

    Driven through torchcg's real `_ForwardDispatcher` and a real packed
    artifact's real metadata rather than a hand-written dict, because the whole
    claim is about reaching the right object.
    """

    import tcg_artifacts

    from gen_worker._vendor.torchcg.adopt import _ForwardDispatcher
    from gen_worker._vendor.torchcg.artifact import read_metadata

    artifact = tcg_artifacts.build(tmp_path / "graph.tcg")
    metadata = read_metadata(artifact)
    # A real mint's real metadata: the layout axis is present because tcg#83
    # made it mandatory, and this build wished for nothing.
    assert metadata["declared_input_layout"] == IDENTITY
    assert metadata["layout_wishlist"] == []

    class _Module:
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return None

    class _Record:
        graph = "g0"
        lora_bucket = 0
        specialization_dims: tuple = ()
        fork: tuple = ()

    class _Graph:
        pass

    module = _Module()
    dispatcher = _ForwardDispatcher(module)
    graph = _Graph()
    graph.metadata = metadata  # type: ignore[attr-defined]
    runner = type("R", (), {"_graph": graph})()
    compiled = type("C", (), {"runner": runner})()
    dispatcher._entries.append((cast(Any, _Record()), compiled))

    session = type("S", (), {"_home": {"g0": dispatcher}})()
    rungs = rung_mod.rungs_of(session)
    assert len(rungs) == 1
    assert rungs[0].graph == "g0"
    assert rungs[0].state is LayoutState.NO_WISH
    assert rungs[0].served == IDENTITY


def test_a_session_that_never_armed_anything_reports_no_rungs() -> None:
    """Absent, not zero-with-a-shrug: a worker with nothing armed has no layout
    position, and inventing one would put a row in the status for a graph that
    is serving eager."""

    assert rung_mod.rungs_of(object()) == ()
    assert rung_mod.rungs_of(type("S", (), {"_home": {}})()) == ()
