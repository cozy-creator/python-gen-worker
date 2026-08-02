"""th#1375/pgw#856: the rename's skew guard — unrecognised topology REFUSES.

`group_degree` -> `gpus_per_execution_group` and `groups` -> `execution_groups`
is a wire rename across two sides that deploy independently. Prod pins
gen-worker 0.79.0 and every released tag (v0.78.0 .. v0.90.5) reads the old
spelling, so there is no flip day. The hazard is specific and it is silent:
**absent topology is legal and means one slot**, so a reader that shrugs at a
field it does not recognise reads the packing as absent and serves degree 1
while the hub bills the degree it bought.

Two mechanisms make that unreachable, and this module holds both to their word:

1. **Transitional read** (deleted by th#1376) — both spellings are accepted, so
   deploy ORDER cannot produce a wrong answer. Both present must AGREE.
2. **A closed field set** — anything NOT recognised is `topology_unknown_field`,
   a typed refusal. This is the guard that survives th#1376: once the legacy
   keys are dropped they become unknown keys, and an old hub is refused BY NAME
   instead of being served one slot.

The negative controls here are the point. Each of them returns a perfectly
valid single-slot topology under the pre-th#1375 decoder — that is exactly the
silent downgrade — so a green here is only meaningful because these were first
watched go RED against the shipped 0.79.0 decoder.
"""

from __future__ import annotations

import json

import pytest

from gen_worker.topology import (
    KEY_EXECUTION_GROUPS,
    KEY_GPUS_PER_GROUP,
    LEGACY_KEY_EXECUTION_GROUPS,
    LEGACY_KEY_GPUS_PER_GROUP,
    ExecutionTopology,
    TopologyError,
)


def _decode(**payload: object) -> ExecutionTopology:
    return ExecutionTopology.decode(json.dumps(payload))


# --- THE guard: present but unrecognised is a refusal, never one slot --------

@pytest.mark.parametrize("extra_key", [
    # The shape this rename would have taken had `parallel` been renamed too:
    # a field naming the packing that this build has never heard of.
    "gpus_per_group",
    "execution_group_count",
    # A future contract growth. The field set is CLOSED on purpose: growing it
    # is its own two-release transition, exactly like this rename.
    "placement_policy",
])
def test_unknown_field_refuses_typed_and_never_single_slots(extra_key: str) -> None:
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, **{extra_key: 2})
    assert caught.value.code == "topology_unknown_field"
    # The message must read as a version skew, not a config bug: it names the
    # offending key AND says what silence would have cost.
    assert extra_key in str(caught.value)
    assert "ONE" in str(caught.value)


def test_the_downgrade_this_guard_exists_to_stop() -> None:
    """The exact payload that used to single-slot in silence.

    A hub that emitted the new spelling to a reader that knew only the old one
    had `gpus_per_execution_group` read as absent -> degree 1. With BOTH the
    legacy aliases and the closed field set in place there is no payload naming
    a degree that this build accepts as a degree-1 pod.
    """
    # Recognised new spelling: served at the degree the hub bought.
    assert _decode(gpu_count=4, gpus_per_execution_group=2, execution_groups=2,
                   parallel="sequence").gpus_per_execution_group == 2
    # Recognised legacy spelling: same answer, not a downgrade.
    assert _decode(gpu_count=4, group_degree=2, groups=2,
                   parallel="sequence").gpus_per_execution_group == 2
    # UNrecognised spelling: refused, NOT served as 4x1.
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, gpus_per_group=2, parallel="sequence")
    assert caught.value.code == "topology_unknown_field"


def test_alias_disagreement_refuses_rather_than_choosing() -> None:
    """Both spellings present and contradicting is not a stale producer to be
    tolerated — it is two different packings, and picking either picks
    silently."""
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, gpus_per_execution_group=2, group_degree=1,
                parallel="sequence")
    assert caught.value.code == "topology_alias_disagree"

    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, gpus_per_execution_group=2, group_degree=2,
                execution_groups=2, groups=4, parallel="sequence")
    assert caught.value.code == "topology_alias_disagree"


def test_derived_count_still_cross_checked_under_either_spelling() -> None:
    for groups_key in (KEY_EXECUTION_GROUPS, LEGACY_KEY_EXECUTION_GROUPS):
        with pytest.raises(TopologyError) as caught:
            _decode(gpu_count=4, gpus_per_execution_group=2,
                    parallel="sequence", **{groups_key: 3})
        assert caught.value.code == "topology_execution_groups_disagree"


# --- positive controls: the transition must still SERVE ---------------------

@pytest.mark.parametrize("payload,want_g,want_d", [
    # new spelling only — what the hub emits after th#1376
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 2, KEY_EXECUTION_GROUPS: 2,
      "parallel": "sequence"}, 2, 2),
    # legacy spelling only — a pre-th#1375 hub against this build
    ({"gpu_count": 4, LEGACY_KEY_GPUS_PER_GROUP: 2,
      LEGACY_KEY_EXECUTION_GROUPS: 2, "parallel": "sequence"}, 2, 2),
    # BOTH, agreeing — what the hub emits during the transition
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 2, KEY_EXECUTION_GROUPS: 2,
      LEGACY_KEY_GPUS_PER_GROUP: 2, LEGACY_KEY_EXECUTION_GROUPS: 2,
      "parallel": "sequence"}, 2, 2),
    # data parallel, the shape that is spelling-invariant either way
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 1, KEY_EXECUTION_GROUPS: 4}, 4, 1),
    ({"gpu_count": 4, LEGACY_KEY_GPUS_PER_GROUP: 1,
      LEGACY_KEY_EXECUTION_GROUPS: 4}, 4, 1),
    # one slot spanning every card
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 4, KEY_EXECUTION_GROUPS: 1,
      "parallel": "internal"}, 1, 4),
], ids=["new", "legacy", "dual", "dp-new", "dp-legacy", "internal"])
def test_every_accepted_spelling_yields_the_same_G_and_D(
    payload: dict, want_g: int, want_d: int,
) -> None:
    topo = ExecutionTopology.decode(json.dumps(payload))
    assert (topo.execution_groups, topo.gpus_per_execution_group) == (want_g, want_d)
    # the partition invariant the docstring promises
    assert topo.gpu_count == topo.execution_groups * topo.gpus_per_execution_group


def test_legacy_only_is_accepted_but_says_so() -> None:
    """Accepted for exactly one release, and it must be findable in the logs —
    th#1376's precondition is knowing the fleet stopped sending it."""
    import logging

    logger = logging.getLogger("gen_worker.topology")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        topo = _decode(gpu_count=4, group_degree=2, groups=2, parallel="sequence")
    finally:
        logger.removeHandler(handler)
    assert topo.gpus_per_execution_group == 2
    warned = [r for r in records if r.levelno >= logging.WARNING]
    assert any("th#1376" in r.getMessage() for r in warned), (
        "the legacy spelling must announce its own removal"
    )


def test_produced_topology_is_readable_by_both_sides() -> None:
    """This worker PRODUCES topology too: the parent stamps one per split child
    (`procsplit/group.py`), and a rolling image swap can put a pre-th#1375
    reader on the other end of it."""
    emitted = ExecutionTopology(
        gpu_count=4, gpus_per_execution_group=2, parallel="sequence",
    ).as_dict()
    assert emitted[KEY_GPUS_PER_GROUP] == emitted[LEGACY_KEY_GPUS_PER_GROUP] == 2
    assert emitted[KEY_EXECUTION_GROUPS] == emitted[LEGACY_KEY_EXECUTION_GROUPS] == 2
    # and it round-trips through our own decoder (no key is unknown to us)
    assert ExecutionTopology.decode(json.dumps(emitted)).execution_groups == 2


def test_th1376_end_state_refuses_the_legacy_spelling_rather_than_single_slotting() -> None:
    """The removal must not reopen the hole it closed.

    After th#1376 drops the aliases, a legacy-only payload is a payload full of
    unknown keys. That MUST land on `topology_unknown_field` — not on a
    degree-1 pod. Proving it now means th#1376 is a deletion, not a redesign.
    """
    from gen_worker import topology as topo_mod

    post_removal = frozenset(
        k for k in topo_mod._KNOWN_KEYS
        if k not in (LEGACY_KEY_GPUS_PER_GROUP, LEGACY_KEY_EXECUTION_GROUPS)
    )
    original = topo_mod._KNOWN_KEYS
    topo_mod._KNOWN_KEYS = post_removal  # type: ignore[assignment]
    try:
        with pytest.raises(TopologyError) as caught:
            _decode(gpu_count=4, group_degree=2, groups=2, parallel="sequence")
    finally:
        topo_mod._KNOWN_KEYS = original  # type: ignore[assignment]
    assert caught.value.code == "topology_unknown_field"
    assert "group_degree" in str(caught.value)
