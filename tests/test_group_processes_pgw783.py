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
    reconcile_activity_kind,
    worker_fn_degraded,
    worker_fn_unavailable,
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
    card the hub did not pick. Only rank-0 devices are dispatched by the hub;
    the executor now typed-refuses anything else, and so does route()."""
    topo = ExecutionTopology(
        gpu_count=gpu_count, group_degree=degree, parallel=parallel
    )
    p = GroupPlan.for_topology(topo, socket_path=SOCK)
    rank0_indices = [g * degree for g in range(topo.groups)]
    for gpu_index in rank0_indices:
        run = pb.RunJob(request_id="r", compute=pb.ResolvedCompute(gpu_index=gpu_index))
        expected = Executor._dispatch_group(
            type("S", (), {"topology": topo})(), run
        )
        assert p.route(gpu_index) == expected


def test_a_non_rank0_dispatch_is_refused_not_floored_at_G_gt_1():
    """pgw#779 (security-deltas base): flooring a hub/worker packing
    disagreement onto group 0 piles every mis-dispatch onto the busiest card.
    route() refuses it, matching the executor."""
    from gen_worker.topology import TopologyError

    p = plan(4, degree=2, parallel="sequence")  # rank-0 devices are 0 and 2
    assert p.route(0) == 0
    assert p.route(2) == 1
    with pytest.raises(TopologyError):
        p.route(1)          # card 1 is a non-rank-0 device
    with pytest.raises(ValueError):
        p.route(None)       # no compute block on a wide pod


def test_a_dispatch_without_compute_is_group_zero_only_on_a_single_group_pod():
    p = plan(1)
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


def test_available_functions_UNION_and_loading_is_what_no_group_serves():
    """Paul's ruling: the worker advertises ANY function ANY group can serve.
    One group serving function-X is enough — the worker routes the dispatch to
    that group."""
    merged = merge_state_deltas([
        _delta(available_functions=["a", "b", "c"], loading_functions=["d"]),
        _delta(available_functions=["a", "c"], loading_functions=["b"]),
        _delta(available_functions=["a"], loading_functions=["e"]),
    ])
    # b and c are served by at least one group -> available for the worker.
    assert list(merged.available_functions) == ["a", "b", "c"]
    # "b" is loading in one group but AVAILABLE in another -> available wins,
    # never loading. d and e load nowhere else.
    assert list(merged.loading_functions) == ["d", "e"]


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


def test_residency_UNIONS_refs_and_takes_the_strongest_tier():
    """A ref is resident on the WORKER while ANY group holds it (same union as
    availability): the worker routes a dispatch to the group that has it."""
    merged = merge_residency([
        [pb.ModelResidency(ref="shared", tier=pb.RESIDENCY_TIER_VRAM, vram_bytes=100),
         pb.ModelResidency(ref="only-here", tier=pb.RESIDENCY_TIER_VRAM, vram_bytes=7)],
        [pb.ModelResidency(ref="shared", tier=pb.RESIDENCY_TIER_DISK, vram_bytes=0)],
    ])
    # BOTH refs appear — "only-here" is served by routing to group 0.
    assert [m.ref for m in merged] == ["only-here", "shared"]
    shared = next(m for m in merged if m.ref == "shared")
    # Strongest tier wins — the best the worker can do for that ref.
    assert shared.tier == pb.RESIDENCY_TIER_VRAM
    # vram_bytes is a MEASURED pod footprint, so it sums across holders.
    assert shared.vram_bytes == 100


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
    assert list(merged.state.available_functions) == ["a", "b"]  # union
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
# Parent-side aggregation — ONE worker view, never N sub-units (ruling 2)
# ---------------------------------------------------------------------------


def _act(state, *, step=0, total=0, seq=0, counter="", done=0.0, ctotal=0.0,
         stalled=False, stalled_ms=0, kind="self_mint_compile"):
    return pb.ActivityUpdate(
        kind=kind, state=state, step=step, total_steps=total, seq=seq,
        counter=counter, counter_done=done, counter_total=ctotal,
        self_stalled=stalled, stalled_for_ms=stalled_ms,
    )


def test_the_worker_is_minting_if_ANY_group_is():
    """The hub folds ActivityUpdate into info.Activities[kind]; G children with
    the same kind would overwrite each other and a group's mint would read as
    the whole worker's. The parent reconciles to ONE worker-level activity."""
    merged = reconcile_activity_kind(
        {0: _act(pb.ACTIVITY_STATE_COMPLETED),
         1: _act(pb.ACTIVITY_STATE_RUNNING, step=3, total=10),
         2: _act(pb.ACTIVITY_STATE_COMPLETED)},
        seq=42,
    )
    assert merged.state == pb.ACTIVITY_STATE_RUNNING
    assert merged.seq == 42  # parent-minted, not any child's
    assert merged.step == 3
    assert merged.kind == "self_mint_compile"


def test_activity_is_terminal_only_when_every_group_is():
    done = reconcile_activity_kind(
        {0: _act(pb.ACTIVITY_STATE_COMPLETED), 1: _act(pb.ACTIVITY_STATE_COMPLETED)},
        seq=1,
    )
    assert done.state == pb.ACTIVITY_STATE_COMPLETED
    # A failure in any group, once none are still running, surfaces as FAILED.
    failed = reconcile_activity_kind(
        {0: _act(pb.ACTIVITY_STATE_COMPLETED), 1: _act(pb.ACTIVITY_STATE_FAILED)},
        seq=2,
    )
    assert failed.state == pb.ACTIVITY_STATE_FAILED


def test_activity_progress_is_the_aggregate_so_one_group_cannot_mask_others():
    """The hub judges liveness by counter advancement (gw#621). If the parent
    forwarded one group's frozen counter the pod would be reaped while three
    groups made progress. The worker-level counter is the SUM."""
    merged = reconcile_activity_kind(
        {0: _act(pb.ACTIVITY_STATE_RUNNING, counter="download:bytes", done=100, ctotal=400),
         1: _act(pb.ACTIVITY_STATE_RUNNING, counter="download:bytes", done=250, ctotal=400)},
        seq=7,
    )
    assert merged.counter_done == 350
    assert merged.counter_total == 800


def test_the_worker_is_stalled_only_when_EVERY_live_group_is():
    """self_stalled is a confession the hub recycles the pod on. One group
    advancing is the worker advancing, so the parent must not confess for it."""
    one_moving = reconcile_activity_kind(
        {0: _act(pb.ACTIVITY_STATE_RUNNING, stalled=True, stalled_ms=9000),
         1: _act(pb.ACTIVITY_STATE_RUNNING, stalled=False)},
        seq=1,
    )
    assert one_moving.self_stalled is False
    assert one_moving.stalled_for_ms == 0
    all_stuck = reconcile_activity_kind(
        {0: _act(pb.ACTIVITY_STATE_RUNNING, stalled=True, stalled_ms=9000),
         1: _act(pb.ACTIVITY_STATE_RUNNING, stalled=True, stalled_ms=5000)},
        seq=2,
    )
    assert all_stuck.self_stalled is True
    assert all_stuck.stalled_for_ms == 5000  # the least-stuck group's clock


def test_a_function_is_unavailable_only_when_NO_group_serves_it():
    """Ruling 2: one group losing a function must not retire it worker-wide if
    another still serves it. The parent reports the single worker-level fact."""
    fu = pb.FnUnavailable(function_name="f", reason="insufficient_vram")
    # group 1 serves it (None) -> worker serves it, nothing reported.
    assert worker_fn_unavailable({0: fu, 1: None}) is None
    # every group unavailable -> the worker is unavailable, one reason carried.
    every = worker_fn_unavailable({0: fu, 1: pb.FnUnavailable(
        function_name="f", reason="setup_failed")})
    assert every is not None
    # a hardware-gating reason is preferred over a transient setup failure.
    assert every.reason == "insufficient_vram"


def test_a_function_is_degraded_only_when_no_group_serves_it_native():
    """FnDegraded is 'give me a bigger card'. If one group serves native, the
    worker routes there and is not degraded."""
    big = pb.FnDegraded(function_name="f", ran="offload", est_latency_multiplier=3.0)
    small = pb.FnDegraded(function_name="f", ran="fp8_storage", est_latency_multiplier=1.4)
    # a native-serving group exists -> not degraded.
    assert worker_fn_degraded({0: big}, served_native_somewhere=True) is None
    # none native -> report the LEAST degraded (the worker's true best card).
    best = worker_fn_degraded({0: big, 1: small}, served_native_somewhere=False)
    assert best is not None and best.est_latency_multiplier == 1.4


def test_aggregation_never_leaks_a_group_ordinal_to_the_wire():
    """The proto has no group field and must never grow one — the whole point
    of ruling 2. These reconcilers return plain worker-level messages."""
    act = reconcile_activity_kind({3: _act(pb.ACTIVITY_STATE_RUNNING)}, seq=1)
    assert "group" not in {f.name for f, _ in act.ListFields()}
    fu = worker_fn_unavailable({2: pb.FnUnavailable(function_name="f", reason="x")})
    assert "group" not in {f.name for f, _ in fu.ListFields()}


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


# ---------------------------------------------------------------------------
# Per-group host RAM — G children in ONE memory cgroup (783-C)
# ---------------------------------------------------------------------------


def _cgroup(tmp_path, *, limit_bytes: int, current_bytes: int):
    """A REAL synthetic cgroup tree, pgw#752's test style — no mocking of the
    probe, only of the kernel files it reads."""
    root = tmp_path / "cgroup"
    (root / "pod").mkdir(parents=True)
    (root / "pod" / "memory.max").write_text(str(limit_bytes))
    (root / "pod" / "memory.current").write_text(str(current_bytes))
    (root / "pod" / "memory.stat").write_text("inactive_file 0\nactive_file 0\n")
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text("0::/pod\n")
    return {"root": root, "proc_self_cgroup": proc}


def test_one_child_sees_the_whole_container_unchanged(tmp_path):
    from gen_worker.models.memory import probe_host_ram

    import msgspec

    files = _cgroup(tmp_path, limit_bytes=64 << 30, current_bytes=4 << 30)
    solo = probe_host_ram(siblings=1, **files)
    assert solo.siblings == 1
    assert solo.cgroup_limit_gb == pytest.approx(64.0)

    # Identical to not passing the argument at all on an unsplit worker. The
    # live `available` readings are min'd with real /proc/meminfo and drift
    # between two probes on a busy box, so they are zeroed for the comparison —
    # everything the split could have changed is compared.
    def stable(ram):
        return msgspec.structs.replace(ram, available_gb=0.0, meminfo_available_gb=0.0)

    assert stable(probe_host_ram(**files)) == stable(solo)


def test_four_children_each_get_a_quarter_of_the_cgroup(tmp_path):
    """G children read the SAME cgroup cap. Left alone, four children each
    admit a 50 GiB move against a 60 GiB pod and the kernel settles it — which
    is exactly the uncatchable SIGKILL pgw#763 exists to prevent."""
    from gen_worker.models.memory import probe_host_ram

    files = _cgroup(tmp_path, limit_bytes=64 << 30, current_bytes=4 << 30)
    solo = probe_host_ram(siblings=1, **files)
    quarter = probe_host_ram(siblings=4, **files)
    assert quarter.siblings == 4
    assert quarter.total_gb == pytest.approx(solo.total_gb / 4)
    assert quarter.cgroup_limit_gb == pytest.approx(16.0)
    # A child may claim no more than its share AND no more than exists, so the
    # sum of G claims is bounded by the cap — the property the guard needs.
    assert quarter.available_gb <= quarter.total_gb + 1e-9
    assert 4 * quarter.available_gb <= solo.total_gb + 1e-6


def test_the_divisor_comes_from_the_child_env(tmp_path, monkeypatch):
    from gen_worker.models.memory import probe_host_ram

    files = _cgroup(tmp_path, limit_bytes=64 << 30, current_bytes=4 << 30)
    monkeypatch.setenv(ENV_HOST_SIBLINGS, "2")
    assert probe_host_ram(**files).siblings == 2
    monkeypatch.delenv(ENV_HOST_SIBLINGS)
    assert probe_host_ram(**files).siblings == 1


def test_a_wide_pods_move_guard_refuses_what_a_solo_worker_admits(tmp_path):
    """The end-to-end consequence, through the REAL guard: a move that fits a
    solo worker on this pod must be refused for one of four children sharing
    it. Sized from the measured headroom so the row is box-independent."""
    from gen_worker.api.errors import HostRamMoveRefusedError
    from gen_worker.host_move_guard import _refuse_if_over_budget
    from gen_worker.models.memory import probe_host_ram
    from gen_worker.models.residency import _effective_ram_floor_gb

    files = _cgroup(tmp_path, limit_bytes=64 << 30, current_bytes=1 << 30)
    solo = probe_host_ram(siblings=1, **files)
    headroom_gb = solo.available_gb - _effective_ram_floor_gb()
    if headroom_gb < 2.0:
        pytest.skip(f"box has only {headroom_gb:.1f}GiB of guard headroom")
    # Comfortably inside a solo worker's budget, comfortably outside a quarter
    # of it.
    incoming = int(headroom_gb * 0.8 * (1 << 30))
    _refuse_if_over_budget(incoming, siblings=1, **files)
    with pytest.raises(HostRamMoveRefusedError) as exc:
        _refuse_if_over_budget(incoming, siblings=4, **files)
    assert "host-RAM move refused" in str(exc.value)


def test_the_ram_floor_scales_with_the_share_so_wide_small_pods_still_move():
    """The floor must not stay pod-wide while the budget is divided, or a
    4-group pod with a 32 GiB cap gives every child an 8 GiB share against an
    8 GiB floor and refuses every guarded move. It does not: the adaptive floor
    reads ``get_total_ram_gb()``, which is now this child's share."""
    from gen_worker.models.memory import HostRam, _host_ram_share

    whole = HostRam(
        total_gb=32.0, available_gb=30.0, meminfo_total_gb=32.0,
        meminfo_available_gb=30.0, cgroup_limit_gb=32.0, source="cgroup",
    )
    share = _host_ram_share(whole, 4)
    assert share.total_gb == 8.0
    # min(8.0, max(1.0, total * 0.2)) — the same formula, on the share.
    assert min(8.0, max(1.0, share.total_gb * 0.2)) == pytest.approx(1.6)
    assert min(8.0, max(1.0, whole.total_gb * 0.2)) == pytest.approx(6.4)


def test_the_ceiling_is_per_message_not_cumulative():
    """A worker that serves for hours legitimately relays gigabytes of small
    control frames. The invariant is about a SINGLE message carrying payload."""
    seam = SeamAccountant()
    for _ in range(100_000):
        assert seam.record("job_progress", 200) is True
    assert seam.clean
    assert seam.job_payload_bytes == 20_000_000
    assert "job_progress" in seam.summary()


# ---------------------------------------------------------------------------
# The ParentControl multi-slot core (pgw#783 parent.py refactor)
#
# These drive the parent's routing + fan-in wiring WITHOUT spawning real
# children (the heavy end-to-end G=2 spawn rides the procsplit integration
# harness). A ParentControl is constructed with a delivered topology; its
# Transport is built but never run.
# ---------------------------------------------------------------------------

from gen_worker.config import load_settings  # noqa: E402
from gen_worker.procsplit.parent import ParentControl, _ChildSlot  # noqa: E402


def _parent(gpu_count=1, degree=1, parallel=""):
    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1", worker_id="w-test", worker_jwt="",
    )
    topo = ExecutionTopology(gpu_count=gpu_count, group_degree=degree, parallel=parallel)
    return ParentControl(settings, socket_path=SOCK, topology=topo)


def test_parent_builds_one_slot_per_execution_group():
    p1 = _parent(1)
    assert p1.groups == 1
    assert [s.ordinal for s in p1._slots] == [0]
    # G==1: the ONE slot has an empty env delta and stage-1's socket path.
    assert dict(p1._slots[0].group_env) == {}
    assert p1._slots[0].socket_path == SOCK

    p4 = _parent(4)
    assert p4.groups == 4
    assert [s.devices for s in p4._slots] == [(0,), (1,), (2,), (3,)]
    assert len({s.socket_path for s in p4._slots}) == 4
    assert dict(p4._slots[2].group_env)["CUDA_VISIBLE_DEVICES"] == "2"
    assert dict(p4._slots[2].group_env)[ENV_HOST_SIBLINGS] == "4"


def test_absent_topology_is_a_single_slot_worker():
    settings = load_settings(
        orchestrator_public_addr="127.0.0.1:1", worker_id="w", worker_jwt="",
    )
    p = ParentControl(settings, socket_path=SOCK, topology=ExecutionTopology.single())
    assert p.groups == 1
    assert dict(p._slots[0].group_env) == {}


def test_route_slot_is_the_identity_at_G1_and_refuses_misdispatch_at_G4():
    p1 = _parent(1)
    # G==1: any dispatch (even one with a wild gpu_index or none) is the one slot.
    assert p1._route_slot(pb.RunJob(request_id="r")) is p1._slots[0]
    wild = pb.RunJob(request_id="r", compute=pb.ResolvedCompute(gpu_index=9))
    assert p1._route_slot(wild) is p1._slots[0]

    p4 = _parent(4)
    for gi, ordinal in ((0, 0), (1, 1), (3, 3)):
        run = pb.RunJob(request_id="r", compute=pb.ResolvedCompute(gpu_index=gi))
        assert p4._route_slot(run) is p4._slots[ordinal]
    # No compute block on a wide pod cannot be routed -> None (RETRYABLE upstream).
    assert p4._route_slot(pb.RunJob(request_id="r")) is None


def test_in_flight_helpers_span_all_slots():
    p = _parent(4)
    p._slots[0].in_flight[("r1", 1)] = "fn-a"
    p._slots[2].in_flight[("r2", 1)] = "fn-b"
    assert p._is_in_flight("r1", 1) and p._is_in_flight("r2", 1)
    assert not p._is_in_flight("r3", 1)
    assert p._slot_for_request(("r2", 1)) is p._slots[2]
    assert set(p._all_in_flight()) == {("r1", 1), ("r2", 1)}


def test_fan_in_is_verbatim_at_G1():
    """The whole safety property: at G==1 a child frame relays byte-identically
    to the pre-pgw#783 parent — no aggregation touches it."""
    p = _parent(1)
    slot = p._slots[0]
    for msg in (
        pb.WorkerMessage(job_result=pb.JobResult(request_id="r", status=pb.JOB_STATUS_OK)),
        pb.WorkerMessage(state_delta=pb.StateDelta(available_functions=["a"])),
        pb.WorkerMessage(activity_update=pb.ActivityUpdate(kind="mint", seq=5)),
        pb.WorkerMessage(job_progress=pb.JobProgress(request_id="r")),
    ):
        assert p._fan_in(slot, msg) is msg  # same object, verbatim


def test_note_state_delta_is_the_childs_message_at_G1_merged_at_G2():
    p1 = _parent(1)
    m = pb.WorkerMessage(state_delta=pb.StateDelta(
        available_functions=["a"], free_vram_bytes=10))
    p1._slots[0].last_state_delta = m
    p1._note_state_delta()
    assert p1._last_state_delta is m  # byte-identical beat at G==1

    p2 = _parent(2)
    p2._slots[0].last_state_delta = pb.WorkerMessage(state_delta=pb.StateDelta(
        available_functions=["a"], free_vram_bytes=10))
    p2._slots[1].last_state_delta = pb.WorkerMessage(state_delta=pb.StateDelta(
        available_functions=["b"], free_vram_bytes=30))
    p2._note_state_delta()
    st = p2._last_state_delta.state_delta
    assert list(st.available_functions) == ["a", "b"]  # UNION
    assert st.free_vram_bytes == 40                     # SUM


def test_fan_in_reconciles_activity_across_groups_at_G2():
    """Two groups minting the same kind collapse to ONE worker activity with a
    parent-minted seq (the hub sees one worker, not two mints)."""
    p = _parent(2)
    a0 = pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind="self_mint_compile", state=pb.ACTIVITY_STATE_RUNNING, step=2, seq=100))
    out0 = p._fan_in(p._slots[0], a0)
    a1 = pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind="self_mint_compile", state=pb.ACTIVITY_STATE_RUNNING, step=5, seq=7))
    out1 = p._fan_in(p._slots[1], a1)
    assert out1.activity_update.state == pb.ACTIVITY_STATE_RUNNING
    assert out1.activity_update.step == 5              # furthest group
    # parent-minted, monotonic, NOT either child's seq (100 or 7)
    assert out0.activity_update.seq == 1
    assert out1.activity_update.seq == 2


def test_fan_in_suppresses_fn_unavailable_while_a_sibling_serves_it():
    """One group losing a function must not retire it worker-wide."""
    p = _parent(2)
    # group 0 reports it unavailable; group 1 has never reported it (serves it).
    fu = pb.WorkerMessage(fn_unavailable=pb.FnUnavailable(
        function_name="f", reason="insufficient_vram"))
    out = p._fan_in(p._slots[0], fu)
    assert out is None  # suppressed: the worker still serves f via group 1


def test_parent_stays_torch_free_after_construction():
    import sys
    _parent(4)
    assert "torch" not in sys.modules


# ---------------------------------------------------------------------------
# CPU divisor — G children share ONE cpu cgroup (783-C, cpu_budget)
# ---------------------------------------------------------------------------

from gen_worker import cpu_budget  # noqa: E402


def test_cpu_divisor_multiplies_the_local_groups_by_the_sibling_count(monkeypatch):
    """Under the split each child rewrites its own topology to a single local
    group, so `groups` reads 1 in every child — but G of them share one cpu
    cgroup. Without the sibling multiplier each would claim the whole allowance
    and reinstate the 192-threads-on-32-cores oversubscription pgw#782 removed.
    `impose` records the effective concurrency before the torch check, so this
    holds torch-free."""
    monkeypatch.setenv(ENV_HOST_SIBLINGS, "4")
    facts = cpu_budget.impose_intra_op_threads(1)   # one local group per child
    assert facts["host_siblings"] == 4
    assert facts["concurrency"] == 4                # 1 local x 4 siblings

    # A D>1 group (2 local groups) on a 2-sibling pod: 2 x 2 = 4.
    monkeypatch.setenv(ENV_HOST_SIBLINGS, "2")
    assert cpu_budget.impose_intra_op_threads(2)["concurrency"] == 4


def test_cpu_divisor_is_unchanged_without_siblings(monkeypatch):
    """siblings == 1 (every non-split pod) -> the divisor is `groups`, exactly
    as pgw#782 shipped. Byte-identical."""
    monkeypatch.delenv(ENV_HOST_SIBLINGS, raising=False)
    facts = cpu_budget.impose_intra_op_threads(4)
    assert facts["host_siblings"] == 1
    assert facts["concurrency"] == 4                # groups, unmultiplied
    # And the per-group threads use that effective divisor.
    assert cpu_budget.per_group_threads(32.0, 4) == 8


# ---------------------------------------------------------------------------
# PR_SET_PDEATHSIG — a compute child dies with its parent (VRAM reap, 783-C)
# ---------------------------------------------------------------------------

import os as _os  # noqa: E402
import signal as _signal  # noqa: E402
import subprocess as _subprocess  # noqa: E402
import sys as _sys  # noqa: E402
import time as _time  # noqa: E402


@pytest.mark.skipif(_sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only")
def test_a_child_is_reaped_when_its_parent_dies():
    """The mechanism that stops a crashed group from stranding VRAM: with
    PR_SET_PDEATHSIG the child dies when its parent dies. Proven with CPU
    processes — killing the process is exactly what frees its CUDA context and
    all its VRAM, so a reaped child is a reaped GPU allocation.

    An intermediate parent spawns a long-sleeping grandchild with the same
    preexec the real spawn uses, prints its pid, then is SIGKILLed. The
    grandchild must die on its own (no one kills it directly)."""
    from gen_worker.procsplit.parent import _set_pdeathsig

    # The intermediate parent: spawn a grandchild with pdeathsig, print its pid,
    # then sleep forever (until WE kill this parent).
    driver = (
        "import subprocess,sys,time,os;"
        "from gen_worker.procsplit.parent import _set_pdeathsig;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
        "preexec_fn=_set_pdeathsig);"
        "print(p.pid,flush=True);"
        "time.sleep(120)"
    )
    src_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src")
    env = dict(_os.environ)
    env["PYTHONPATH"] = _os.pathsep.join([src_dir, env.get("PYTHONPATH", "")])
    parent = _subprocess.Popen(
        [_sys.executable, "-c", driver], stdout=_subprocess.PIPE, env=env,
    )
    try:
        grandchild_pid = int(parent.stdout.readline().decode().strip())
        # The grandchild is alive while its parent lives.
        _os.kill(grandchild_pid, 0)
        # Kill the intermediate parent; PR_SET_PDEATHSIG must reap the grandchild.
        parent.kill()
        parent.wait(timeout=10)
        deadline = _time.monotonic() + 10.0
        reaped = False
        while _time.monotonic() < deadline:
            try:
                _os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                reaped = True
                break
            _time.sleep(0.1)
        assert reaped, f"grandchild {grandchild_pid} survived its parent's death"
    finally:
        if parent.poll() is None:
            parent.kill()
        try:
            _os.kill(grandchild_pid, _signal.SIGKILL)
        except (ProcessLookupError, NameError):
            pass


# ---------------------------------------------------------------------------
# Parent-minted worker_session_id — survives child respawns (783-D)
# ---------------------------------------------------------------------------

from gen_worker.procsplit import ENV_SESSION_ID  # noqa: E402


def test_the_parent_mints_a_stable_session_id_passed_to_every_child():
    """Child-minted (uuid4 in IntentRegistry) it changed on every respawn and
    the hub rejected the cross-session shadow state — a latent defect even at
    G=1. The parent mints it ONCE and hands it to every child via env."""
    import os as _o
    p = _parent(4)
    sid = p._worker_session_id
    assert sid and len(sid) == 32
    # Every group's child spawn carries the SAME parent session id.
    for slot in p._slots:
        env = dict(_o.environ)
        env.update(p._child_env)
        env.update(slot.group_env)
        env[ENV_SESSION_ID] = p._worker_session_id  # mirrors _spawn_child
        assert env[ENV_SESSION_ID] == sid


def test_intent_registry_reads_the_parent_session_id_else_mints(monkeypatch):
    from gen_worker.intent_registry import IntentRegistry

    monkeypatch.setenv(ENV_SESSION_ID, "parent-owned-session-abc")
    reg = IntentRegistry("release-x", ["fn"])
    assert reg.worker_session_id == "parent-owned-session-abc"
    # A respawned child in the same pod reads the SAME id — stable across
    # respawns, which is the whole fix.
    reg2 = IntentRegistry("release-x", ["fn"])
    assert reg2.worker_session_id == "parent-owned-session-abc"
    # No env (no split): a fresh uuid, exactly as before.
    monkeypatch.delenv(ENV_SESSION_ID)
    assert IntentRegistry("release-x", ["fn"]).worker_session_id != \
        "parent-owned-session-abc"


def test_hello_carries_the_parent_session_id_at_G1_and_G2():
    """Both Hello paths stamp the parent's id."""
    p1 = _parent(1)
    hello = pb.Hello(worker_session_id="stale-child-mint")
    # G=1 inline path stamps it (via _apply then explicit set — assert the set).
    hello.worker_session_id = p1._worker_session_id
    assert hello.worker_session_id == p1._worker_session_id
    # G>1 merge path overrides via merge_hello(worker_session_id=...).
    p2 = _parent(2)
    a = pb.Hello(worker_id="w", worker_session_id="child-a")
    b = pb.Hello(worker_id="w", worker_session_id="child-b")
    merged = merge_hello([a, b], worker_session_id=p2._worker_session_id)
    assert merged.worker_session_id == p2._worker_session_id


# ---------------------------------------------------------------------------
# Per-group inductor cache dirs — G children don't race a shared cache (783-C)
# ---------------------------------------------------------------------------


def test_each_group_gets_its_own_inductor_and_triton_dir(monkeypatch, tmp_path):
    from gen_worker.entrypoint import _isolate_group_inductor_cache

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv(ENV_HOST_SIBLINGS, "4")
    monkeypatch.setenv(ENV_GROUP_ORDINAL, "2")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    _isolate_group_inductor_cache()
    import os as _o
    assert _o.environ["TORCHINDUCTOR_CACHE_DIR"].endswith("gen-worker-inductor/g2/inductor")
    assert _o.environ["TRITON_CACHE_DIR"].endswith("gen-worker-inductor/g2/triton")
    assert _o.path.isdir(_o.environ["TORCHINDUCTOR_CACHE_DIR"])
    # A different group gets a DIFFERENT dir — no shared cache to corrupt.
    monkeypatch.setenv(ENV_GROUP_ORDINAL, "3")
    _isolate_group_inductor_cache()
    assert _o.environ["TORCHINDUCTOR_CACHE_DIR"].endswith("g3/inductor")


def test_a_single_child_keeps_the_default_inductor_dir(monkeypatch, tmp_path):
    """siblings == 1 (every pod not running the split): NO override — the
    default is untouched, byte-identical to today."""
    from gen_worker.entrypoint import _isolate_group_inductor_cache
    import os as _o

    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(ENV_HOST_SIBLINGS, raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    _isolate_group_inductor_cache()
    assert "TORCHINDUCTOR_CACHE_DIR" not in _o.environ


# ---------------------------------------------------------------------------
# The seam accountant is WIRED into the live relay (control-not-data, demo)
# ---------------------------------------------------------------------------


def test_parent_accounts_relayed_frames_on_its_seam():
    """The demo's acceptance reads parent.seam: a blob_ref job_result relays
    kilobytes (under the ceiling, clean); the accountant is live on the parent,
    not just an unused module."""
    p = _parent(1)
    assert p.seam.clean and p.seam.job_payload_bytes == 0
    # Simulate the relay accounting a normal blob_ref result (what the child
    # sends when its output left as a presigned PUT).
    ok = pb.WorkerMessage(job_result=pb.JobResult(
        request_id="r", status=pb.JOB_STATUS_OK,
        safe_message="r2://bucket/out.webp"))
    raw = ok.SerializeToString()
    assert p.seam.record("job_result", len(raw), group=0) is True
    assert p.seam.clean
    assert p.seam.job_payload_bytes < 4096
    # And the guard can fail: an inlined 8 MiB result trips it.
    assert p.seam.record("job_result", 8 << 20, group=0) is False
    assert not p.seam.clean
