"""pgw#748 phase 1 / th#1285: the worker half of the execution-topology wire.

The hub decides the packing and delivers it as ONE env var; the worker derives
slots, groups and devices from it and invents nothing. What is asserted here:

A. The contract as SHIPPED (`EXECUTION-TOPOLOGY-DESIGN.md` §2a) — every legal
   shape decodes, every illegal one is a TYPED refusal naming the hub's own
   code, and absent stays legal (one slot).
B. The slot math — `G = gpu_count/gpus_per_execution_group` IS the slot semaphore, and
   `gpu_index` names the group's rank-0 device so `DeviceGroup` falls straight
   out of it.
C. The group is part of instance identity, and is ELIDED at 0 — every key a
   single-group pod computes is byte-identical to what it computed before this
   existed.
D. One Residency registry per group, sharing only the disk tier.
E. The fabric gate: a pod bought for platform-installed sharding that measures
   anything but NVLink demotes to `G×1` rather than serving a promise the
   hardware cannot keep (SEQPAR-DESIGN §12: 2xRTX-4090 degree-2 measured
   **0.51x** — half the speed of one GPU).
"""

from __future__ import annotations

from typing import Tuple

import pytest

from gen_worker.models.residency import DeviceGroup
from gen_worker.topology import (
    ENV_VAR,
    ExecutionTopology,
    TopologyError,
    current_device_group,
    delivered_topology,
    device_group_scope,
)

_GiB = 1024 ** 3


def _env(raw: str) -> dict:
    return {ENV_VAR: raw}


# ---------------------------------------------------------------------------
# A. The contract as shipped.
# ---------------------------------------------------------------------------


def test_absent_topology_is_one_slot_and_never_an_error() -> None:
    # Every CPU pod and every pod created before this field existed.
    for env in ({}, {ENV_VAR: ""}, {ENV_VAR: "   "}):
        topo = ExecutionTopology.from_env(env)
        assert (topo.gpu_count, topo.gpus_per_execution_group, topo.execution_groups) == (1, 1, 1)
        assert topo.parallel == ""


def test_the_shipped_shapes_decode() -> None:
    dp = ExecutionTopology.from_env(
        _env('{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}'))
    assert (dp.execution_groups, dp.degree, dp.parallel) == (4, 1, "")

    sp = ExecutionTopology.from_env(
        _env('{"gpu_count":4,"gpus_per_execution_group":2,"execution_groups":2,"parallel":"sequence"}'))
    assert (sp.execution_groups, sp.degree, sp.parallel) == (2, 2, "sequence")

    internal = ExecutionTopology.from_env(
        _env('{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"internal"}'))
    assert (internal.execution_groups, internal.degree) == (1, 2)

    cfg = ExecutionTopology.from_env(
        _env('{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"cfg"}'))
    assert cfg.parallel == "cfg"


@pytest.mark.parametrize(
    "raw,code",
    [
        ('{"gpu_count":0,"gpus_per_execution_group":1,"execution_groups":0}', "topology_gpu_count_invalid"),
        ('{"gpu_count":4,"gpus_per_execution_group":0,"execution_groups":0}', "topology_degree_invalid"),
        ('{"gpu_count":3,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"sequence"}',
         "topology_degree_not_divisor"),
        ('{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1}', "topology_parallel_required"),
        ('{"gpu_count":2,"gpus_per_execution_group":1,"execution_groups":2,"parallel":"sequence"}',
         "topology_parallel_without_degree"),
        ('{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"tensor"}',
         "topology_parallel_unknown"),
        ('{"gpu_count":4,"gpus_per_execution_group":2,"execution_groups":3,"parallel":"sequence"}',
         "topology_execution_groups_disagree"),
        ("not json at all", "topology_decode_failed"),
        ("[1,2,3]", "topology_decode_failed"),
        ('{"gpu_count":"four","gpus_per_execution_group":1,"execution_groups":4}', "topology_decode_failed"),
        ('{"gpu_count":1.5,"gpus_per_execution_group":1,"execution_groups":1}', "topology_decode_failed"),
    ],
)
def test_a_present_but_illegal_topology_is_a_typed_refusal(raw: str, code: str) -> None:
    # NEVER a silent fallback to one slot: the hub validates before it emits,
    # so an illegal value can only mean a producer that is not the hub, and
    # serving one slot on a pod the hub dispatches G jobs to is the silent
    # half of a capacity bug.
    with pytest.raises(TopologyError) as exc:
        ExecutionTopology.from_env(_env(raw))
    assert exc.value.code == code


# ---------------------------------------------------------------------------
# B. The slot math and the device derivation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gpu_count,degree,groups",
    [(1, 1, 1), (2, 1, 2), (4, 1, 4), (4, 2, 2), (4, 4, 1), (8, 2, 4)],
)
def test_groups_is_gpu_count_over_degree(gpu_count: int, degree: int, groups: int) -> None:
    parallel = "sequence" if degree > 1 else ""
    topo = ExecutionTopology(
        gpu_count=gpu_count, gpus_per_execution_group=degree, parallel=parallel)
    assert topo.execution_groups == groups
    # G IS the slot count. Both sides compute it from the same two integers;
    # neither negotiates and the worker never advertises it back.
    assert topo.slots == groups


def test_gpu_index_names_the_groups_rank_zero_device() -> None:
    # §2a: gpu_index is UNCHANGED on the wire and is now 0, D, 2D, ...
    topo = ExecutionTopology(gpu_count=4, gpus_per_execution_group=2, parallel="sequence")
    assert topo.group_ordinal(0) == 0
    assert topo.group_ordinal(2) == 1
    assert topo.device_group(0) == DeviceGroup(devices=(0, 1))
    assert topo.device_group(2) == DeviceGroup(devices=(2, 3))
    assert topo.all_groups() == (
        DeviceGroup(devices=(0, 1)), DeviceGroup(devices=(2, 3)))


def test_degree_one_dispatch_is_byte_identical_to_today() -> None:
    topo = ExecutionTopology(gpu_count=4, gpus_per_execution_group=1)
    for idx in range(4):
        assert topo.group_ordinal(idx) == idx
        assert topo.device_group(idx) == DeviceGroup(devices=(idx,))


def test_platform_parallel_is_what_may_be_fabric_gated() -> None:
    # The internal/platform split is load-bearing in BOTH directions: only the
    # platform mechanisms make a bandwidth promise, and demoting an `internal`
    # group would take a model's own cards away.
    assert ExecutionTopology(
        gpu_count=2, gpus_per_execution_group=2, parallel="sequence").platform_parallel
    assert ExecutionTopology(
        gpu_count=2, gpus_per_execution_group=2, parallel="cfg").platform_parallel
    assert not ExecutionTopology(
        gpu_count=2, gpus_per_execution_group=2, parallel="internal").platform_parallel
    assert not ExecutionTopology(gpu_count=2, gpus_per_execution_group=1).platform_parallel


def test_placement_mode_follows_who_shards() -> None:
    # Ulysses replicates weights and shards activations, so a degree-2 group's
    # weight fit is MIN over members. `internal` genuinely splits the weights.
    assert ExecutionTopology(
        gpu_count=2, gpus_per_execution_group=2, parallel="sequence").group(0).placement_mode == "replicated"
    assert ExecutionTopology(
        gpu_count=2, gpus_per_execution_group=2, parallel="internal").group(0).placement_mode == "sharded"


# ---------------------------------------------------------------------------
# C. The group joins instance identity, elided at 0.
# ---------------------------------------------------------------------------


def test_group_zero_instance_keys_are_byte_identical() -> None:
    from gen_worker.registry import EndpointSpec

    def _handler(self, ctx, payload):  # pragma: no cover - never called
        return None

    class _Cls:
        pass

    base = EndpointSpec(
        name="f", method=_handler, kind="inference", payload_type=dict,
        output_mode="single", cls=_Cls,
    )
    grouped0 = type(base)(**{**base.__dict__, "device_group_ordinal": 0})
    grouped1 = type(base)(**{**base.__dict__, "device_group_ordinal": 1})

    # Nothing in the live fleet is re-keyed by this change.
    assert grouped0.instance_key == base.instance_key
    # Two groups are two cards, so they are two resident instances.
    assert grouped1.instance_key != base.instance_key
    assert grouped1.instance_key != grouped0.instance_key


# ---------------------------------------------------------------------------
# D. One residency registry per group, sharing the disk tier.
# ---------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, count: int, free_by_device: Tuple[int, ...]) -> None:
        self._count = count
        self._free = free_by_device

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return self._count

    def mem_get_info(self, device: int = 0) -> Tuple[int, int]:
        if not 0 <= int(device) < self._count:
            raise RuntimeError(f"invalid device ordinal {device}")
        return self._free[int(device)], 80 * _GiB


def test_one_registry_per_group_each_owning_its_own_devices() -> None:
    from gen_worker.models.store import ModelStore

    store = ModelStore(emit=None)  # type: ignore[arg-type]
    topo = ExecutionTopology(gpu_count=4, gpus_per_execution_group=2, parallel="sequence")
    store.bind_topology(topo)

    zero = store.residency_for(0)
    one = store.residency_for(1)
    assert zero is not one
    assert zero.device_group == DeviceGroup(devices=(0, 1))
    assert one.device_group == DeviceGroup(devices=(2, 3))
    # A group's promotions land on ITS card, not on whatever card the loading
    # thread happened to be pointing at.
    assert zero.vram_device == "cuda"
    assert one.vram_device == "cuda:2"
    assert len(store.all_residencies()) == 2


def test_residency_property_follows_the_current_group() -> None:
    from gen_worker.models.store import ModelStore

    store = ModelStore(emit=None)  # type: ignore[arg-type]
    store.bind_topology(ExecutionTopology(gpu_count=2, gpus_per_execution_group=1))
    assert current_device_group() == 0
    assert store.residency is store.residency_for(0)
    with device_group_scope(1):
        assert current_device_group() == 1
        assert store.residency is store.residency_for(1)
    assert store.residency is store.residency_for(0)


def test_disk_facts_union_across_groups(tmp_path) -> None:
    # §4.3 caveat 3: the CAS is ONE tree, hardlinked, with one page cache. A
    # per-group preserve set would drop the pages a sibling is mmapping.
    from gen_worker.models.store import ModelStore

    store = ModelStore(emit=None)  # type: ignore[arg-type]
    store.bind_topology(ExecutionTopology(gpu_count=2, gpus_per_execution_group=1))
    p0 = tmp_path / "a"
    p0.mkdir()
    p1 = tmp_path / "b"
    p1.mkdir()
    store.residency_for(0).track_disk("ref-a", p0)
    store.residency_for(1).track_disk("ref-b", p1)

    assert sorted(store.disk_refs()) == ["ref-a", "ref-b"]
    assert store.disk_local_path("ref-b") == p1
    assert not store.disk_ref_in_use("ref-b")


# ---------------------------------------------------------------------------
# E. The fabric gate — measured, never assumed.
# ---------------------------------------------------------------------------


def test_a_sharded_pod_without_nvlink_demotes_instead_of_serving_the_promise() -> None:
    raw = '{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"sequence"}'
    # The pod the hub bought in good faith. pgw#818: class alone no longer
    # keeps the group — the measured bandwidth must clear the floor too.
    nvlink = delivered_topology(_env(raw), interconnect="nvlink", peer_gbps=272.6)
    assert (nvlink.execution_groups, nvlink.degree, nvlink.parallel) == (1, 2, "sequence")

    # H100-PCIe measured `SYS` at 14.5 GB/s -> 1.40x, sold as 1.89x.
    # 2xRTX-4090 measured 1.96 GB/s -> 0.51x, SLOWER than one GPU.
    for miss in ("host-staged", "pcie-p2p", ""):
        demoted = delivered_topology(_env(raw), interconnect=miss, peer_gbps=14.5)
        assert (demoted.execution_groups, demoted.degree, demoted.parallel) == (2, 1, "")
    # And the class-passes-floor-fails band (pgw#818's disagreement band).
    demoted = delivered_topology(_env(raw), interconnect="nvlink", peer_gbps=30.2)
    assert (demoted.execution_groups, demoted.degree, demoted.parallel) == (2, 1, "")


def test_an_internal_group_is_never_demoted() -> None:
    # Demoting `internal` would take a multi-device model's own cards away and
    # break a release that has always worked.
    raw = '{"gpu_count":2,"gpus_per_execution_group":2,"execution_groups":1,"parallel":"internal"}'
    for link in ("nvlink", "host-staged", ""):
        topo = delivered_topology(_env(raw), interconnect=link, peer_gbps=14.5)
        assert (topo.execution_groups, topo.degree, topo.parallel) == (1, 2, "internal")


def test_free_vram_reports_the_tightest_group_never_the_pod_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The REPORTING half of pgw#648: a 4-card pod that reports the sum invites
    # the hub to admit a model that fits on no single card.
    #
    # pgw#776 corrected the direction: MAX across groups was honest per JOB and
    # dishonest per POD, because the hub admits G concurrent jobs against this
    # ONE scalar. The MIN is the only safe single number.
    torch = pytest.importorskip("torch")
    from gen_worker import lifecycle

    fake = _FakeCuda(4, (10 * _GiB, 10 * _GiB, 40 * _GiB, 20 * _GiB))
    monkeypatch.setattr(torch.cuda, "is_available", fake.is_available)
    monkeypatch.setattr(torch.cuda, "device_count", fake.device_count)
    monkeypatch.setattr(torch.cuda, "mem_get_info", fake.mem_get_info)

    monkeypatch.setenv(
        ENV_VAR, '{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}')
    # 4x1: four independent pools, and the hub may fill all four — so the
    # honest offer is the tightest one, not the roomiest (pgw#776).
    assert lifecycle.free_vram_bytes() == 10 * _GiB

    monkeypatch.setenv(
        ENV_VAR,
        '{"gpu_count":4,"gpus_per_execution_group":2,"execution_groups":2,"parallel":"sequence"}')
    # 2x2 replicated: group 0 = min(10,10) = 10, group 1 = min(40,20) = 20;
    # the pod can only promise 10 to each of the two jobs it may be given.
    assert lifecycle.free_vram_bytes() == 10 * _GiB

    monkeypatch.delenv(ENV_VAR, raising=False)
    assert lifecycle.free_vram_bytes() == 10 * _GiB


def test_pinned_budget_is_a_pod_budget_shared_by_the_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # §4.3 caveat 2: pinned host RAM is unswappable and process-global. At G
    # groups one group must not be able to claim the whole cap.
    from gen_worker.models import staging

    monkeypatch.setattr(staging, "get_total_ram_gb", lambda: 100.0)
    monkeypatch.setattr(staging, "get_available_ram_gb", lambda: 100.0)
    pool = staging.PinnedPool()
    pool.set_group_count(2)
    cap_each = int(100 * _GiB * staging._PINNED_TOTAL_FRACTION) // 2

    assert pool.try_reserve(cap_each)
    assert not pool.try_reserve(1 * _GiB)          # group 0 is at its share
    with device_group_scope(1):
        assert pool.try_reserve(cap_each)          # the sibling still has its own
        assert not pool.try_reserve(1 * _GiB)
