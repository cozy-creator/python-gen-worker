"""pgw#773 residual: a group's cards come from the TOPOLOGY, never the ordinal.

At degree D group ``g`` owns ``[g*D, (g+1)*D)``. The ordinal is the device only
at D == 1, so on a `2x2` pod (two degree-2 groups over four cards) group 1 owns
cards 2-3 — and the placement helpers used to say `cuda:1`, which is group 0's
FOLLOWER card. That was the last structural reason `G>1 ∧ D>1` was refused at
boot; these are the tests that the mapping is real and the refusal is gone.

Layer exercised: `topology` (the delivered packing and the placement helpers it
translates through) plus the devices a `SequenceRuntime` hands its ranks.
"""

from __future__ import annotations

import json

import pytest

from gen_worker.topology import (
    ExecutionTopology,
    TopologyError,
    delivered_topology,
    device_group_scope,
    group_devices,
    group_rank0_device,
    install_topology,
    pin_cuda_device_for_group,
    refuse_unless_groups_can_coexist,
)


def _env(gpu_count: int, degree: int, parallel: str = "sequence") -> dict:
    body = {"gpu_count": gpu_count, "gpus_per_execution_group": degree}
    if parallel:
        body["parallel"] = parallel
    return {"WORKER_EXECUTION_TOPOLOGY": json.dumps(body)}


@pytest.fixture(autouse=True)
def _no_installed_topology():
    install_topology(None)
    yield
    install_topology(None)


def test_two_pairs_map_every_rank_to_a_card_it_owns() -> None:
    topo = delivered_topology(_env(4, 2), interconnect="nvlink", peer_gbps=272.6)
    assert (topo.execution_groups, topo.degree) == (2, 2)
    install_topology(topo)

    # The whole defect in one assertion: group 1's rank-0 card is 2, not 1.
    assert group_devices(0) == (0, 1)
    assert group_devices(1) == (2, 3)
    assert (group_rank0_device(0), group_rank0_device(1)) == (0, 2)

    # No card is claimed by two groups, and every card is claimed once.
    owned = [d for g in range(topo.execution_groups) for d in group_devices(g)]
    assert sorted(owned) == [0, 1, 2, 3]

    # The ambient group (the contextvar every job stamps) resolves the same way.
    with device_group_scope(1):
        assert group_devices() == (2, 3)
        assert group_rank0_device() == 2


def test_the_ranks_a_sequence_runtime_forms_are_the_groups_own_cards() -> None:
    from gen_worker.parallel.runtime import SequenceRuntime

    topo = delivered_topology(_env(4, 2), interconnect="nvlink", peer_gbps=272.6)
    runtimes = [SequenceRuntime(topo.group(g).devices) for g in range(2)]
    assert [rt.devices for rt in runtimes] == [(0, 1), (2, 3)]
    # rank r of group g executes on devices[r] — group 1 rank 1 is card 3.
    assert runtimes[1].devices[1] == 3
    assert all(rt.degree == 2 for rt in runtimes)


def test_degree_one_and_no_topology_are_the_identity_mapping() -> None:
    # Every pod that has ever shipped: the ordinal IS the device.
    assert group_devices(0) == (0,)
    assert group_rank0_device(3) == 3
    install_topology(delivered_topology(_env(4, 1, parallel=""), interconnect="pcie"))
    assert [group_devices(g) for g in range(4)] == [(0,), (1,), (2,), (3,)]
    # A group the topology does not describe still resolves, never raises.
    assert group_rank0_device(9) == 9


def test_the_thread_is_pinned_to_the_groups_card_not_its_ordinal(monkeypatch) -> None:
    import torch

    install_topology(delivered_topology(_env(4, 2), interconnect="nvlink", peer_gbps=272.6))
    pinned: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: pinned.append(int(d)))

    with device_group_scope(1):
        pin_cuda_device_for_group()
    assert pinned == [2], "group 1 of a 2x2 pod must pin card 2, not card 1"

    # Group 0's rank-0 card is 0: still the thread default, still a no-op.
    with device_group_scope(0):
        pin_cuda_device_for_group()
    assert pinned == [2]


def test_multi_group_sequence_is_no_longer_refused() -> None:
    # The pgw#773 boot refusal is LIFTED for `sequence` at any shape.
    for gpus, degree in ((4, 2), (8, 2), (8, 4)):
        topo = delivered_topology(_env(gpus, degree), interconnect="nvlink", peer_gbps=272.6)
        assert (topo.execution_groups, topo.degree) == (gpus // degree, degree)
    refuse_unless_groups_can_coexist(ExecutionTopology(4, 2, parallel="sequence"))


def test_a_degree_without_a_runtime_is_still_refused_by_name() -> None:
    # `cfg` rides the same wire field and nothing in this worker splits a CFG
    # batch: holding 2 cards per slot and serving 1 card's worth is exactly
    # what a boot refusal exists to prevent.
    with pytest.raises(TopologyError) as exc:
        delivered_topology(_env(4, 2, parallel="cfg"), interconnect="nvlink", peer_gbps=272.6)
    assert exc.value.code == "topology_group_parallel_unsupported"
    with pytest.raises(TopologyError):
        refuse_unless_groups_can_coexist(ExecutionTopology(2, 2, parallel="cfg"))
    # ...and a cfg pod on a fabric that cannot keep the promise demotes first,
    # to a shape that IS servable.
    demoted = delivered_topology(_env(4, 2, parallel="cfg"), interconnect="pcie")
    assert (demoted.execution_groups, demoted.degree) == (4, 1)


def test_reading_the_real_environment_publishes_the_packing(monkeypatch) -> None:
    # The translation only happens if something installs the delivered
    # topology. The boot read IS that seam (`Worker.__init__` ->
    # `delivered_topology()`), so no placement path needs the topology threaded
    # through it — and a question asked about some OTHER pod (explicit env)
    # must publish nothing.
    assert group_rank0_device(1) == 1, "un-installed: the identity mapping"
    delivered_topology(_env(4, 2), interconnect="nvlink", peer_gbps=272.6)
    assert group_rank0_device(1) == 1, "an explicit env must not publish"

    monkeypatch.setenv("WORKER_EXECUTION_TOPOLOGY", _env(4, 2)["WORKER_EXECUTION_TOPOLOGY"])
    delivered_topology(interconnect="nvlink", peer_gbps=272.6)
    assert group_devices(1) == (2, 3)

    # A fabric miss demotes to 4x1 — and it is the DEMOTED packing that gets
    # published, because that is the one being served.
    delivered_topology(interconnect="pcie")
    assert group_devices(1) == (1,)
