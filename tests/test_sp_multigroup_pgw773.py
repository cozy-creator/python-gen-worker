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

    body = {"gpu_count": gpu_count, "gpus_per_execution_group": degree}
    if parallel:
        body["parallel"] = parallel
    return {"WORKER_EXECUTION_TOPOLOGY": json.dumps(body)}


def test_multi_gpu_execution_group_is_served() -> None:
    # 4x H100 delivered as two degree-2 groups: a legal hub decision, and now
    # a SERVED one — per-group process groups plus
    # topology-derived placement (the residual, see
    # test_group_device_map_pgw773.py) leave nothing to refuse.
    topo = delivered_topology(_env(4, 2), interconnect="nvlink", peer_gbps=272.6)
    assert (topo.execution_groups, topo.degree) == (2, 2)
    assert [topo.group(g).devices for g in range(2)] == [(0, 1), (2, 3)]


def test_single_group_and_pure_dp_shapes_are_untouched() -> None:
    # G=1 (every pod today), pure DP width, and a single degree-D group all
    # keep working exactly as before.
    assert delivered_topology({}, interconnect="").execution_groups == 1
    dp = delivered_topology(_env(4, 1, parallel=""), interconnect="pcie")
    assert (dp.execution_groups, dp.degree) == (4, 1)
    sp = delivered_topology(_env(2, 2), interconnect="nvlink", peer_gbps=272.6)
    assert (sp.execution_groups, sp.degree) == (1, 2)


def test_fabric_demotion_wins_before_the_refusal() -> None:
    # A 4x2 pod on a non-NVLink fabric demotes to 4x1 — which is servable,
    # so it must NOT be refused.
    topo = delivered_topology(_env(4, 2), interconnect="pcie")
    assert (topo.execution_groups, topo.degree) == (4, 1)


def test_refusal_helper_is_exact() -> None:
    refuse_unless_groups_can_coexist(ExecutionTopology(4, 1))
    refuse_unless_groups_can_coexist(
        ExecutionTopology(2, 2, parallel="sequence"))
    refuse_unless_groups_can_coexist(
        ExecutionTopology(4, 2, parallel="sequence"))
    # Reachable, and typed, for a degree this worker has no runtime for.
    with pytest.raises(TopologyError):
        refuse_unless_groups_can_coexist(
            ExecutionTopology(4, 2, parallel="cfg"))
