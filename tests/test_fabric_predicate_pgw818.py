"""pgw#818: hub and worker must apply THE SAME fabric predicate.

th#1285 interpretation 4 ruled out a HelloAck demote field because "the worker
gates on its OWN boot canary interconnect, which is the same measurement the
hub demotes on, so both sides agree by construction". The fleet survey then
added a bandwidth floor hub-side (`nvlink AND peer_gbps >= 200`,
tensorhub topology/interconnect.go SPMinPeerGbps) and the worker's gate kept
reading `interconnect` alone — so the construction broke. In the disagreement
band (`interconnect == "nvlink" AND peer_gbps < 200`):

  - a 2x2 pod: hub re-packs 4x1 and dispatches indices 0..3; the worker still
    holds 2x2, so `group_ordinal_exact` refuses indices 1 and 3 RETRYABLE,
    forever — half the pod is a permanent retry loop;
  - a 1x4 pod: hub sees 4 slots, worker arms 1 group — capacity overstated 4x.

The design stays two independent gates over one measurement (NO HelloAck
field); the fix is the worker adopting the same two-term predicate, plus the
survey's standing recommendation: a WEDGED fabric (peer access, exactly zero
bandwidth — the collective hangs with no error) refuses at boot, typed, for
any multi-GPU topology.
"""

from __future__ import annotations

import pytest

from gen_worker.host_canary import (
    INTERCONNECT_NVLINK,
    SP_MIN_PEER_GBPS,
    is_fabric_wedge,
    sp_admits,
)
from gen_worker.topology import ENV_VAR, TopologyError, delivered_topology

_SP_2x2 = '{"gpu_count":4,"group_degree":2,"groups":2,"parallel":"sequence"}'
_SP_1x4 = '{"gpu_count":4,"group_degree":4,"groups":1,"parallel":"sequence"}'


def _env(raw: str) -> dict:
    return {ENV_VAR: raw}


def test_predicate_matches_the_hub_constant() -> None:
    # The floor is the hub's SPMinPeerGbps, verbatim. The measured populations
    # it separates: NVLink 241.9-273.9 GB/s a2a (388.2-389.8 D2D) vs
    # everything else <= 30.2 (<= 52.9 D2D) — 200 sits inside both gaps.
    assert SP_MIN_PEER_GBPS == 200.0
    assert sp_admits(INTERCONNECT_NVLINK, 241.9)
    assert not sp_admits(INTERCONNECT_NVLINK, 199.9)
    assert not sp_admits("pcie-p2p", 500.0)
    assert not sp_admits("", 500.0)


def test_disagreement_band_now_demotes_like_the_hub() -> None:
    # THE pgw#818 band: class says nvlink, bandwidth says degraded (best NVL
    # host measured 30.2 GB/s). The hub demotes to G×1; pre-fix the worker
    # kept G×D and refused half of every dispatch forever.
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
    assert (topo.groups, topo.degree, topo.parallel) == (2, 2, "sequence")


def test_wedged_fabric_refuses_at_boot_typed() -> None:
    # Machine 8n9k05n0sz03, reproduced twice: peer access TRUE, 0.0 GB/s, the
    # collective HUNG with no error/timeout. classify_interconnect calls it
    # nvlink, so the class gate passes and the pod strands every request. A
    # typed boot refusal closes the race the hub-side drain can lose.
    assert is_fabric_wedge(True, 0.0)
    assert not is_fabric_wedge(False, 0.0)  # not measured, no verdict
    assert not is_fabric_wedge(True, 30.2)

    for raw in (_SP_2x2, '{"gpu_count":2,"group_degree":2,"groups":1,"parallel":"internal"}'):
        with pytest.raises(TopologyError) as err:
            delivered_topology(
                _env(raw),
                interconnect=INTERCONNECT_NVLINK,
                peer_gbps=0.0,
                peer_access=True,
            )
        assert err.value.code == "topology_fabric_wedged_peer_access_zero_bandwidth"


def test_internal_groups_still_never_bandwidth_demoted() -> None:
    # The devices are the model's, not the platform's: a slow fabric demotes
    # nothing on parallel="internal" (only a WEDGE refuses, above).
    raw = '{"gpu_count":2,"group_degree":2,"groups":1,"parallel":"internal"}'
    topo = delivered_topology(_env(raw), interconnect="host-staged", peer_gbps=1.96)
    assert (topo.groups, topo.degree, topo.parallel) == (1, 2, "internal")
