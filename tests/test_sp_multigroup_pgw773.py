"""pgw#773: two sequence-parallel groups must never share a process group.

Layer exercised: `topology.delivered_topology` (the worker's single boot
entry for the packing it will execute) and `parallel.group`/`parallel.runtime`
(the rank plumbing itself, on the gloo CPU rig).
"""

from __future__ import annotations

import pytest

from gen_worker.topology import (
    ExecutionTopology,
    TopologyError,
    delivered_topology,
    refuse_unless_groups_can_coexist,
)


def _env(gpu_count: int, degree: int, parallel: str = "sequence") -> dict:
    import json

    body = {"gpu_count": gpu_count, "group_degree": degree}
    if parallel:
        body["parallel"] = parallel
    return {"WORKER_EXECUTION_TOPOLOGY": json.dumps(body)}


def test_multi_group_degree_is_refused_typed_at_boot() -> None:
    # 4x H100 delivered as two degree-2 groups: a legal hub decision the
    # process cannot serve yet. Must be a NAMED refusal, never a silent
    # default-PG collision.
    with pytest.raises(TopologyError) as exc:
        delivered_topology(_env(4, 2), interconnect="nvlink")
    assert exc.value.code == "topology_multi_group_sequence_unsupported"


def test_single_group_and_pure_dp_shapes_are_untouched() -> None:
    # G=1 (every pod today), pure DP width, and a single degree-D group all
    # keep working exactly as before.
    assert delivered_topology({}, interconnect="").groups == 1
    dp = delivered_topology(_env(4, 1, parallel=""), interconnect="pcie")
    assert (dp.groups, dp.degree) == (4, 1)
    sp = delivered_topology(_env(2, 2), interconnect="nvlink")
    assert (sp.groups, sp.degree) == (1, 2)


def test_fabric_demotion_wins_before_the_refusal() -> None:
    # A 4x2 pod on a non-NVLink fabric demotes to 4x1 — which is servable,
    # so it must NOT be refused.
    topo = delivered_topology(_env(4, 2), interconnect="pcie")
    assert (topo.groups, topo.degree) == (4, 1)


def test_refusal_helper_is_exact() -> None:
    refuse_unless_groups_can_coexist(ExecutionTopology(4, 1))
    refuse_unless_groups_can_coexist(
        ExecutionTopology(2, 2, parallel="sequence"))
    with pytest.raises(TopologyError):
        refuse_unless_groups_can_coexist(
            ExecutionTopology(4, 2, parallel="sequence"))
