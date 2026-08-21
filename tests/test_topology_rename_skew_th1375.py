"""The rename's skew guard — unrecognised topology REFUSES."""

from __future__ import annotations

import json

import pytest

from gen_worker.topology import (
    KEY_EXECUTION_GROUPS,
    KEY_GPUS_PER_GROUP,
    ExecutionTopology,
    TopologyError,
)

RETIRED_KEY_GPUS_PER_GROUP = "group_degree"
RETIRED_KEY_EXECUTION_GROUPS = "groups"


def _decode(**payload: object) -> ExecutionTopology:
    return ExecutionTopology.decode(json.dumps(payload))


@pytest.mark.parametrize("extra_key", [
    "gpus_per_group",
    "execution_group_count",
    "placement_policy",
    RETIRED_KEY_GPUS_PER_GROUP,
    RETIRED_KEY_EXECUTION_GROUPS,
])
def test_unknown_field_refuses_typed_and_never_single_slots(extra_key: str) -> None:
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, **{extra_key: 2})
    assert caught.value.code == "topology_unknown_field"
    assert extra_key in str(caught.value)
    assert "ONE" in str(caught.value)


def test_the_downgrade_this_guard_exists_to_stop() -> None:
    assert _decode(gpu_count=4, gpus_per_execution_group=2, execution_groups=2,
                   parallel="sequence").gpus_per_execution_group == 2
    with pytest.raises(TopologyError) as caught:
        _decode(gpu_count=4, gpus_per_group=2, parallel="sequence")
    assert caught.value.code == "topology_unknown_field"


def test_retired_spelling_is_refused_not_translated() -> None:
    """The hard cut, stated as a test: a pre-rename hub is refused BY NAME."""
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


@pytest.mark.parametrize("payload,want_g,want_d", [
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 2, KEY_EXECUTION_GROUPS: 2,
      "parallel": "sequence"}, 2, 2),
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 1, KEY_EXECUTION_GROUPS: 4}, 4, 1),
    ({"gpu_count": 4, KEY_GPUS_PER_GROUP: 4, KEY_EXECUTION_GROUPS: 1,
      "parallel": "internal"}, 1, 4),
], ids=["sequence", "dp", "internal"])
def test_accepted_payloads_yield_the_expected_G_and_D(
    payload: dict, want_g: int, want_d: int,
) -> None:
    topo = ExecutionTopology.decode(json.dumps(payload))
    assert (topo.execution_groups, topo.gpus_per_execution_group) == (want_g, want_d)
    assert topo.gpu_count == topo.execution_groups * topo.gpus_per_execution_group
