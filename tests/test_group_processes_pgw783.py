"""pgw#783: one child per EXECUTION GROUP — the group plan, the fan-in, and
the control-not-data invariant.

The load-bearing rows here are the two the design stands or falls on:

* **G == 1 IDENTITY** — a one-child worker must be byte-identical to what
  pgw#763 stage 1 shipped and measured. Empty env delta, stage 1's socket path,
  merges that return their input unchanged.
* **CONTROL, NOT DATA** — a job whose result crosses the parent's interpreter
  is a bug. Asserted green (a blob_ref result relays ~nothing) and RED (an
  inlined payload trips the guard, so the guard can actually fail).
"""

from __future__ import annotations

import pytest

from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import (
    ENV_GROUP_ORDINAL,
    ENV_HOST_SIBLINGS,
    ENV_TOPOLOGY,
    group_ordinal,
    host_siblings,
)
from gen_worker.procsplit.group import GroupPlan
from gen_worker.procsplit.merge import (
    merge_hello,
    merge_phase,
    merge_residency,
    merge_state_deltas,
)
from gen_worker.procsplit.seam import (
    CONTROL_FRAME_CEILING_BYTES,
    DIAL_PHASE,
    SeamAccountant,
)
from gen_worker.topology import ExecutionTopology

SOCK = "/tmp/gen-worker-compute-1234.sock"


def plan(gpu_count: int, degree: int = 1, parallel: str = "") -> GroupPlan:
    topo = ExecutionTopology(
        gpu_count=gpu_count, group_degree=degree, parallel=parallel
    )
    return GroupPlan.for_topology(topo, socket_path=SOCK)


# ---------------------------------------------------------------------------
# G == 1 IDENTITY — the claim the whole design rests on
# ---------------------------------------------------------------------------


def test_g1_child_env_delta_is_empty():
    """At G == 1 the child is spawned with EXACTLY stage 1's environment.

    Not "almost" — empty. No CUDA_VISIBLE_DEVICES (which would change CUDA
    ordinal semantics on every single-GPU pod in the fleet), no topology
    rewrite, no sibling count. If this row ever goes red, the one-child path
    stopped being the path pgw#763 measured on real silicon.
    """
    p = plan(1)
    assert p.groups == 1
    assert dict(p.child(0).env) == {}
    assert p.child(0).devices == (0,)


def test_g1_socket_path_is_stage_ones_verbatim():
    p = plan(1)
    assert p.child(0).socket_path == SOCK


def test_g1_routing_and_gpu_index_are_the_identity():
    p = plan(1)
    assert p.route(0) == 0
    assert p.route(None) == 0
    assert p.local_gpu_index(0) == 0


def test_g1_merges_return_their_input_unchanged():
    """Byte-identical on the serialized form, and the SAME object."""
    delta = pb.StateDelta(
        phase=pb.WORKER_PHASE_READY,
        available_functions=["a", "b"],
        free_vram_bytes=17,
        observed_config_generation=9,
    )
    merged = merge_state_deltas([delta])
    assert merged is delta
    assert merged.SerializeToString() == delta.SerializeToString()

    hello = pb.Hello(worker_id="w", state=delta, heartbeat_interval_ms=10_000)
    hello.in_flight.add(request_id="r1", attempt=1)
    assert merge_hello([hello]) is hello

    snap = [pb.ModelResidency(ref="x", tier=pb.RESIDENCY_TIER_VRAM, vram_bytes=5)]
    assert merge_residency([snap]) == list(snap)


def test_g1_absent_topology_is_still_one_child():
    """Every CPU pod and every pod created before th#1285 has no topology at
    all. That is a legal state and must stay a one-child worker."""
    p = GroupPlan.for_topology(ExecutionTopology.single(), socket_path=SOCK)
    assert p.groups == 1
    assert dict(p.child(0).env) == {}


# ---------------------------------------------------------------------------
# The group abstraction at G > 1
# ---------------------------------------------------------------------------


def test_children_own_disjoint_physical_devices():
    p = plan(4)
    assert p.groups == 4
    assert [c.devices for c in p.children] == [(0,), (1,), (2,), (3,)]
    seen = set()
    for c in p.children:
        assert not (seen & set(c.devices)), "two children share a card"
        seen |= set(c.devices)


def test_a_degree_2_group_is_one_child_with_two_cards():
    """A D>1 group is ONE logical accelerator — the model's own arrangement or
    a platform collective — so it stays ONE process. 4 GPUs at D=2 is TWO
    children, not four."""
    p = plan(4, degree=2, parallel="sequence")
    assert p.groups == 2
    assert [c.devices for c in p.children] == [(0, 1), (2, 3)]
    assert [c.degree for c in p.children] == [2, 2]


def test_each_child_is_a_single_group_worker_over_its_own_cards():
    """The core simplification: inside the child, G == 1 and the cards are
    local. That is why the N-child case reuses the most-tested path in the
    worker instead of growing a second one."""
    p = plan(4)
    env = dict(p.child(2).env)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    local = ExecutionTopology.decode(env[ENV_TOPOLOGY])
    assert local.groups == 1
    assert local.gpu_count == 1
    assert env[ENV_GROUP_ORDINAL] == "2"
    assert env[ENV_HOST_SIBLINGS] == "4"


def test_a_degree_2_childs_local_topology_keeps_its_parallel_mechanism():
    p = plan(4, degree=2, parallel="sequence")
    env = dict(p.child(1).env)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    local = ExecutionTopology.decode(env[ENV_TOPOLOGY])
    assert (local.groups, local.gpu_count, local.group_degree) == (1, 2, 2)
    assert local.parallel == "sequence"


def test_sibling_count_is_the_true_G_not_the_childs_rewritten_one():
    """pgw#782's cpu_budget divisor is the delivered group count, and this
    design rewrites that to 1 in every child. Without the sibling count each
    of G children would take the WHOLE cpu quota and reinstate exactly the
    192-threads-on-32-cores misconfiguration pgw#782 removed."""
    for gpus, degree, expected in ((4, 1, "4"), (4, 2, "2"), (8, 2, "4")):
        p = plan(gpus, degree=degree, parallel="sequence" if degree > 1 else "")
        for child in p.children:
            assert dict(child.env)[ENV_HOST_SIBLINGS] == expected


def test_children_get_distinct_sockets_only_beyond_one():
    p = plan(3)
    paths = [c.socket_path for c in p.children]
    assert len(set(paths)) == 3
    assert all(path.endswith(".sock") for path in paths)


def test_env_readers_default_to_the_single_worker_shape(monkeypatch):
    monkeypatch.delenv(ENV_HOST_SIBLINGS, raising=False)
    monkeypatch.delenv(ENV_GROUP_ORDINAL, raising=False)
    assert host_siblings() == 1
    assert group_ordinal() == 0
    monkeypatch.setenv(ENV_HOST_SIBLINGS, "4")
    monkeypatch.setenv(ENV_GROUP_ORDINAL, "3")
    assert host_siblings() == 4
    assert group_ordinal() == 3
    monkeypatch.setenv(ENV_HOST_SIBLINGS, "garbage")
    assert host_siblings() == 1


# ---------------------------------------------------------------------------
# Routing: the parent ROUTES, it does not schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_count,degree,parallel", [
    (1, 1, ""), (2, 1, ""), (4, 1, ""), (4, 2, "sequence"), (8, 4, "internal"),
])
def test_routing_agrees_with_the_executors_own_dispatch_derivation(
    gpu_count, degree, parallel,
):
    """The parent's route MUST be Executor._dispatch_group moved one process
    earlier — if the two ever disagree, a wide worker serves requests on a
    card the hub did not pick."""
    topo = ExecutionTopology(
        gpu_count=gpu_count, group_degree=degree, parallel=parallel
    )
    p = GroupPlan.for_topology(topo, socket_path=SOCK)
    for gpu_index in range(gpu_count):
        run = pb.RunJob(request_id="r", compute=pb.ResolvedCompute(gpu_index=gpu_index))
        expected = Executor._dispatch_group(
            type("S", (), {"topology": topo})(), run
        )
        assert p.route(gpu_index) == expected


def test_a_dispatch_without_a_compute_block_is_group_zero():
    p = plan(4)
    run = pb.RunJob(request_id="r")
    assert not run.HasField("compute")
    assert p.route(run.compute.gpu_index if run.HasField("compute") else None) == 0


def test_the_child_always_sees_local_gpu_index_zero():
    """Under CUDA_VISIBLE_DEVICES the child's world starts at 0, so the parent
    rewrites the one field as it routes. At G == 1 that rewrite is 0 -> 0."""
    p = plan(4, degree=2, parallel="cfg")
    for ordinal in range(p.groups):
        assert p.local_gpu_index(ordinal) == 0


# ---------------------------------------------------------------------------
# Fan-in — every rule, with the wrong answer it is guarding against
# ---------------------------------------------------------------------------


def _delta(**kw) -> pb.StateDelta:
    return pb.StateDelta(**kw)


def test_available_functions_intersect_and_loading_unions():
    merged = merge_state_deltas([
        _delta(available_functions=["a", "b", "c"], loading_functions=["d"]),
        _delta(available_functions=["a", "c"], loading_functions=["b"]),
        _delta(available_functions=["a", "b", "c"]),
    ])
    assert list(merged.available_functions) == ["a", "c"]
    # "b" is available in two groups and loading in one -> loading, never
    # available: the hub picks the slot, so a union would invite dispatch into
    # the group that cannot serve.
    assert list(merged.loading_functions) == ["b", "d"]


def test_free_vram_and_finalizing_jobs_sum():
    merged = merge_state_deltas([
        _delta(free_vram_bytes=10, finalizing_jobs=1),
        _delta(free_vram_bytes=30, finalizing_jobs=2),
    ])
    assert merged.free_vram_bytes == 40
    assert merged.finalizing_jobs == 3


def test_observed_generations_take_the_MINIMUM():
    """A max would tell the hub a config edit had landed while a group was
    still running the previous one."""
    merged = merge_state_deltas([
        _delta(observed_residency_generation=7, observed_config_generation=4),
        _delta(observed_residency_generation=5, observed_config_generation=9),
    ])
    assert merged.observed_residency_generation == 5
    assert merged.observed_config_generation == 4


def test_disk_usage_is_ONE_childs_report_and_never_the_sum():
    """THE TRAP. G children share ONE container filesystem. Summing would tell
    the hub the pod has G times the disk it has and break every residency
    budget on a wide pod."""
    report = pb.DiskUsageReport(capacity_generation=3)
    report.tiers.add(
        tier=pb.STORAGE_TIER_CONTAINER, mount_path="/", total_bytes=1000,
        free_bytes=400, used_bytes=600,
    )
    merged = merge_state_deltas([
        _delta(disk_usage=report), _delta(disk_usage=report), _delta(disk_usage=report),
    ])
    assert merged.disk_usage.tiers[0].total_bytes == 1000
    assert merged.disk_usage.tiers[0].free_bytes == 400
    assert merged.disk_usage.capacity_generation == 3


def test_phase_is_the_least_ready_group_and_ERROR_dominates():
    assert merge_phase([pb.WORKER_PHASE_READY, pb.WORKER_PHASE_READY]) == \
        pb.WORKER_PHASE_READY
    assert merge_phase([pb.WORKER_PHASE_READY, pb.WORKER_PHASE_WARMING]) == \
        pb.WORKER_PHASE_WARMING
    # ERROR is the LARGEST enum value but the LEAST ready state.
    assert merge_phase([pb.WORKER_PHASE_READY, pb.WORKER_PHASE_ERROR]) == \
        pb.WORKER_PHASE_ERROR


def test_compile_targets_union_and_cell_lookups_dedup():
    a = _delta(compile_targets=[pb.CompileTarget(incarnation_id="i1", family="f")],
               cell_lookups=[pb.CellLookup(family="f", cell_key="ck1-a")])
    b = _delta(compile_targets=[pb.CompileTarget(incarnation_id="i2", family="f")],
               cell_lookups=[pb.CellLookup(family="f", cell_key="ck1-a"),
                             pb.CellLookup(family="f", cell_key="ck1-b")])
    merged = merge_state_deltas([a, b])
    assert [t.incarnation_id for t in merged.compile_targets] == ["i1", "i2"]
    assert [cl.cell_key for cl in merged.cell_lookups] == ["ck1-a", "ck1-b"]


def test_residency_counts_only_refs_every_group_holds():
    merged = merge_residency([
        [pb.ModelResidency(ref="shared", tier=pb.RESIDENCY_TIER_VRAM, vram_bytes=100),
         pb.ModelResidency(ref="only-here", tier=pb.RESIDENCY_TIER_VRAM)],
        [pb.ModelResidency(ref="shared", tier=pb.RESIDENCY_TIER_DISK, vram_bytes=0)],
    ])
    assert [m.ref for m in merged] == ["shared"]
    # Weakest tier wins (the hub may dispatch to either group)...
    assert merged[0].tier == pb.RESIDENCY_TIER_DISK
    # ...but vram_bytes is a MEASURED pod footprint, so it sums.
    assert merged[0].vram_bytes == 100


def test_hello_merges_state_residency_in_flight_and_cadence():
    def hello(fns, gen, vram, beat, rid):
        h = pb.Hello(
            worker_id="w", release_id="r",
            state=_delta(phase=pb.WORKER_PHASE_READY, available_functions=fns,
                         free_vram_bytes=vram, observed_config_generation=gen),
            heartbeat_interval_ms=beat,
        )
        h.models.add(ref="m", tier=pb.RESIDENCY_TIER_VRAM, vram_bytes=vram)
        h.in_flight.add(request_id=rid, attempt=1)
        return h

    merged = merge_hello(
        [hello(["a", "b"], 5, 100, 10_000, "r1"),
         hello(["a"], 3, 200, 8_000, "r2")],
        worker_session_id="parent-session",
        extra_in_flight=[("r3", 2), ("r1", 1)],
    )
    assert list(merged.state.available_functions) == ["a"]
    assert merged.state.free_vram_bytes == 300
    assert merged.state.observed_config_generation == 3
    assert [m.vram_bytes for m in merged.models] == [300]
    assert merged.heartbeat_interval_ms == 8_000
    assert sorted((j.request_id, j.attempt) for j in merged.in_flight) == \
        [("r1", 1), ("r2", 1), ("r3", 2)]
    assert merged.worker_id == "w"


def test_worker_session_id_is_parent_minted():
    """It is uuid4 in intent_registry today, i.e. CHILD-minted, so it changes
    on every child respawn — and the hub rejects cross-session shadow state.
    With G children, one group's respawn must not invalidate the whole
    worker's."""
    a = pb.Hello(worker_id="w", worker_session_id="child-a")
    b = pb.Hello(worker_id="w", worker_session_id="child-b")
    merged = merge_hello([a, b], worker_session_id="parent-owns-this")
    assert merged.worker_session_id == "parent-owns-this"
    # And it applies at G == 1 too, where the same defect is latent.
    assert merge_hello([a], worker_session_id="parent-owns-this") \
        .worker_session_id == "parent-owns-this"


# ---------------------------------------------------------------------------
# CONTROL, NOT DATA — the invariant the 4x depends on
# ---------------------------------------------------------------------------


def _job_result_bytes(*, blob_ref: str = "", inline: bytes = b"") -> int:
    result = pb.JobResult(
        request_id="req-1", attempt=1, status=pb.JOB_STATUS_OK,
        safe_message=blob_ref,
    )
    msg = pb.WorkerMessage(job_result=result)
    raw = msg.SerializeToString()
    return len(raw) + len(inline)


def test_a_blob_ref_result_relays_almost_nothing_through_the_parent():
    """The green arm. The child produced a 24 MiB image and uploaded it to the
    object store itself; what crosses the parent is a reference."""
    seam = SeamAccountant()
    produced_bytes = 24 * 1024 * 1024
    for i in range(64):
        size = _job_result_bytes(blob_ref=f"r2://bucket/outputs/req-{i}.webp")
        assert seam.record("job_result", size, group=i % 4) is True
    assert seam.clean
    # 64 results whose payloads totalled 1.5 GiB crossed as a few KiB of refs.
    assert seam.job_payload_bytes < CONTROL_FRAME_CEILING_BYTES
    assert seam.job_payload_bytes * 1000 < produced_bytes * 64


def test_an_inlined_payload_is_a_VIOLATION_not_a_slow_path():
    """The RED arm — without it the green arm proves only that nothing was
    tested. A result carrying its bytes inline must be caught and named."""
    seam = SeamAccountant()
    ok = seam.record("job_result", 8 * 1024 * 1024, group=2)
    assert ok is False
    assert not seam.clean
    assert len(seam.violations) == 1
    v = seam.violations[0]
    assert v.kind == "job_result" and v.group == 2
    assert DIAL_PHASE in v.format()
    assert "blob_ref" in v.format()


def test_large_control_messages_are_not_violations():
    """A residency snapshot or a lifecycle projection is legitimately large and
    is not job data. The guard must discriminate, or it becomes noise and gets
    switched off."""
    seam = SeamAccountant()
    assert seam.record("lifecycle_snapshot", 4 * 1024 * 1024) is True
    assert seam.record("hello", 2 * 1024 * 1024) is True
    assert seam.clean
    assert seam.job_payload_bytes == 0


def test_the_ceiling_is_per_message_not_cumulative():
    """A worker that serves for hours legitimately relays gigabytes of small
    control frames. The invariant is about a SINGLE message carrying payload."""
    seam = SeamAccountant()
    for _ in range(100_000):
        assert seam.record("job_progress", 200) is True
    assert seam.clean
    assert seam.job_payload_bytes == 20_000_000
    assert "job_progress" in seam.summary()
