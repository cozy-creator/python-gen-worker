"""The rename's skew guard — unrecognised topology REFUSES.

`group_degree` -> `gpus_per_execution_group` and `groups` -> `execution_groups`
was a wire rename across two sides that deploy independently. The hazard is
specific and it is silent: **absent topology is legal and means one slot**, so a
reader that shrugs at a field it does not recognise reads the packing as absent
and serves degree 1 while the hub bills the degree it bought.

The transitional dual-accept is GONE (pre-launch hard cut; the hub emits one
spelling). What enforces the guard now is the **closed field set**: anything not
recognised is `topology_unknown_field`, a typed refusal — so the retired
spellings are refused BY NAME rather than being served as one slot.

The negative controls here are the point. Each returns a perfectly valid
single-slot topology under a decoder that shrugs at unknown keys — that is
exactly the silent downgrade.
"""

from __future__ import annotations

import json

import pytest

from gen_worker.topology import (
    KEY_EXECUTION_GROUPS,
    KEY_GPUS_PER_GROUP,
    ExecutionTopology,
    TopologyError,
)

# The pre-rename spellings. Named here, and ONLY here, so the tests can prove
# they are refused; `gen_worker.topology` no longer knows them.
RETIRED_KEY_GPUS_PER_GROUP = "group_degree"
RETIRED_KEY_EXECUTION_GROUPS = "groups"


def _decode(**payload: object) -> ExecutionTopology:
    return ExecutionTopology.decode(json.dumps(payload))


# --- THE guard: present but unrecognised is a refusal, never one slot --------

@pytest.mark.parametrize("extra_key", [
    # The shape this rename would have taken had `parallel` been renamed too:
    # a field naming the packing that this build has never heard of.
    "gpus_per_group",
    "execution_group_count",
    # A future contract growth. The field set is CLOSED on purpose: growing it
    # is its own transition.
    "placement_policy",
    # The retired spellings are now exactly this case.
    RETIRED_KEY_GPUS_PER_GROUP,
    RETIRED_KEY_EXECUTION_GROUPS,
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

    A hub emitting a spelling this reader does not know had the degree read as
    absent -> degree 1. With the closed field set there is no payload naming a
    degree that this build accepts as a degree-1 pod.
    """
    # Recognised spelling: served at the degree the hub bought.
    assert _decode(gpu_count=4, gpus_per_execution_group=2, execution_groups=2,
                   parallel="sequence").gpus_per_execution_group == 2
    # UNrecognised spelling: refused, NOT served as 4x1.
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, gpus_per_group=2, parallel="sequence")
    assert caught.value.code == "topology_unknown_field"


def test_retired_spelling_is_refused_not_translated() -> None:
    """The hard cut, stated as a test: a pre-rename hub is refused BY NAME.

    This is the property that replaced the dual-accept reader. Both retired
    keys, alone and together, and alongside the current spelling.
    """
    for payload in (
        {"gpu_count": 4, RETIRED_KEY_GPUS_PER_GROUP: 2, RETIRED_KEY_EXECUTION_GROUPS: 2,
         "parallel": "sequence"},
        {"gpu_count": 4, RETIRED_KEY_GPUS_PER_GROUP: 1, RETIRED_KEY_EXECUTION_GROUPS: 4},
        {"gpu_count": 4, KEY_GPUS_PER_GROUP: 2, KEY_EXECUTION_GROUPS: 2,
         RETIRED_KEY_GPUS_PER_GROUP: 2, RETIRED_KEY_EXECUTION_GROUPS: 2,
         "parallel": "sequence"},
    ):
        with pytest.raises(TopologyError) as caught:
            ExecutionTopology.decode(json.dumps(payload))
        assert caught.value.code == "topology_unknown_field"


def test_derived_count_still_cross_checked() -> None:
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, gpus_per_execution_group=2,
                parallel="sequence", **{KEY_EXECUTION_GROUPS: 3})
    assert caught.value.code == "topology_execution_groups_disagree"


# --- positive controls: the contract must still SERVE -----------------------

@pytest.mark.parametrize("payload,want_g,want_d", [
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 2, KEY_EXECUTION_GROUPS: 2,
      "parallel": "sequence"}, 2, 2),
    # data parallel
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 1, KEY_EXECUTION_GROUPS: 4}, 4, 1),
    # one slot spanning every card
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 4, KEY_EXECUTION_GROUPS: 1,
      "parallel": "internal"}, 1, 4),
], ids=["sequence", "dp", "internal"])
def test_accepted_payloads_yield_the_expected_G_and_D(
    payload: dict, want_g: int, want_d: int,
) -> None:
    topo = ExecutionTopology.decode(json.dumps(payload))
    assert (topo.execution_groups, topo.gpus_per_execution_group) == (want_g, want_d)
    # the partition invariant the docstring promises
    assert topo.gpu_count == topo.execution_groups * topo.gpus_per_execution_group
