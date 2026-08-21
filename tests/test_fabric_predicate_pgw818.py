"""Hub and worker must apply THE SAME fabric predicate."""

from __future__ import annotations

import pytest

from gen_worker.host_canary import (
    INTERCONNECT_NVLINK,
    SP_MIN_PEER_GBPS,
    is_fabric_wedge,
    sp_admits,
)
from gen_worker.topology import ENV_VAR, TopologyError, delivered_topology

_SP_2x2 = '{"gpu_count":4,"gpus_per_execution_group":2,"execution_groups":2,"parallel":"sequence"}'
_SP_1x4 = '{"gpu_count":4,"gpus_per_execution_group":4,"execution_groups":1,"parallel":"sequence"}'


def _env(raw: str) -> dict:
    return {ENV_VAR: raw}


def test_predicate_matches_the_hub_constant() -> None:
    assert SP_MIN_PEER_GBPS == 200.0
    assert sp_admits(INTERCONNECT_NVLINK, 241.9)
    assert not sp_admits(INTERCONNECT_NVLINK, 199.9)
    assert not sp_admits("pcie-p2p", 500.0)
    assert not sp_admits("", 500.0)


def test_disagreement_band_now_demotes_like_the_hub() -> None:
    for raw, want in ((_SP_2x2, (4, 1)), (_SP_1x4, (4, 1))):
        topo = delivered_topology(_env(raw), interconnect=INTERCONNECT_NVLINK, peer_gbps=30.2)
        assert (topo.gpu_count, topo.degree) == want, (
            f"worker kept a sharded group at 30.2 GB/s — the hub demoted; "
            f"got {topo}"
        )


def test_proven_fabric_keeps_the_group() -> None:
    topo = delivered_topology(
        _env(_SP_2x2), interconnect=INTERCONNECT_NVLINK, peer_gbps=241.9
    )
    assert (topo.execution_groups, topo.degree, topo.parallel) == (2, 2, "sequence")


def test_wedged_fabric_refuses_at_boot_typed() -> None:
    assert is_fabric_wedge(True, 0.0)
    assert not is_fabric_wedge(False, 0.0)
    assert not is_fabric_wedge(True, 30.2)

    for raw in (_SP_2x2, '{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"internal"}'):
        with pytest.raises(TopologyError) as err:
            delivered_topology(
                _env(raw),
                interconnect=INTERCONNECT_NVLINK,
                peer_gbps=0.0,
                peer_access=True,
            )
        assert err.value.code == "topology_fabric_wedged_peer_access_zero_bandwidth"


def test_internal_groups_still_never_bandwidth_demoted() -> None:
    raw = '{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"internal"}'
    topo = delivered_topology(_env(raw), interconnect="host-staged", peer_gbps=1.96)
    assert (topo.execution_groups, topo.degree, topo.parallel) == (1, 2, "internal")
