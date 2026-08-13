"""pgw#748 phase 1: the sequence-parallel runtime, proved on a CPU rig.

The ltx-cp-probe showed gloo/CPU 2-rank rigs catch the real bugs — rank
divergence, rendezvous, fatal propagation — because none of them are about
CUDA. What needs a GPU (NCCL bandwidth, actual Ulysses numerics) is a pod
measurement, not a test.

Asserted here:

A. A 2-rank group forms over gloo, broadcasts the plan, and every rank gets
   rank 0's decision — including a decision that is wrong for its own card.
B. Rank divergence is a TYPED FATAL, never a local fallback. This is the
   failure mode that dominates the design: same collectives + different
   weights corrupts SILENTLY.
C. A dead follower fails the request loudly instead of parking the group on a
   collective that can never complete.
D. Context parallelism refuses, by name, every configuration that would be
   silently wrong: no `_cp_plan`, CPU offload present, indivisible shapes, and
   a per-tensor w8a8 activation scale.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from gen_worker.parallel import (
    ContextParallelUnavailable,
    FollowerChannel,
    GroupPlan,
    RankDivergence,
    RankGroup,
    RankGroupError,
    RankSpec,
    init_rank,
    install_context_parallel,
    refuse_unless_divisible,
    refuse_unless_shard_invariant_quant,
)

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="needs torch.distributed with the gloo backend",
)


# ---------------------------------------------------------------------------
# A/B. Rank 0 decides; a follower that disagrees fails the group.
# ---------------------------------------------------------------------------


def _follower_echo(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    """The narrow follower loop: join, receive the plan, barrier, exit."""
    import torch.distributed as dist

    pg = init_rank(spec)
    cmd = chan.next_command(timeout=120)
    plan = cmd["plan"]
    assert plan.precision_execution_lane == "w8a8"
    assert plan.gemm_mode == "rowwise"
    chan.report_ready(spec.rank)
    dist.barrier(group=pg)


def _follower_disagrees(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    """A follower whose own card would have chosen differently. It must NOT
    serve its own choice — it must fail the group (pgw#774 made
    assert_agrees LIVE: the follower derives its own view and holds it
    against the delivered plan)."""
    init_rank(spec)
    cmd = chan.next_command(timeout=120)
    mine = GroupPlan(precision_execution_lane="bf16", sp_degree=2)
    mine.assert_agrees(cmd["plan"], rank=spec.rank)


def _follower_dies_before_joining(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    """The card is not there at all: this rank dies before the rendezvous."""
    raise RuntimeError("this rank's card is not there")


def _follower_dies_after_joining(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    """Joined the group, then died — the shape ``check_alive`` was written for."""
    init_rank(spec)
    raise RuntimeError("this rank's card fell off the bus")


def _cpu_group(compiled_graph: Any, degree: int = 2) -> RankGroup:
    return RankGroup(devices=tuple(range(degree)), backend="gloo", compiled_graph=compiled_graph)


def test_rank_zero_decides_and_every_rank_obeys() -> None:
    group = _cpu_group(_follower_echo)
    spec = group.form()
    try:
        assert spec.rank == 0 and spec.world_size == 2
        plan = GroupPlan(
            precision_execution_lane="w8a8", gemm_mode="rowwise", sp_degree=2,
            loras=(("style", 0.8),),
        )
        group.send({"op": "arm", "plan": plan})
        group.wait_armed()
        group.barrier()
    finally:
        group.close()


def test_a_follower_that_disagrees_fails_the_group() -> None:
    # The ruling: a rank that cannot honour the broadcast fails the WHOLE
    # group loudly; it NEVER adapts locally. Adapting locally is the quiet
    # corruption path (same collectives, different weights).
    group = _cpu_group(_follower_disagrees)
    group.form()
    try:
        group.send({"op": "arm",
                    "plan": GroupPlan(precision_execution_lane="w8a8", sp_degree=2)})
        with pytest.raises(RankGroupError):
            for _ in range(200):
                group.check_alive()
                import time

                time.sleep(0.05)
    finally:
        group.close()


def test_a_dead_follower_is_a_loud_failure_not_a_hang() -> None:
    # Died AFTER joining: form() succeeds, check_alive names the rank.
    group = _cpu_group(_follower_dies_after_joining)
    group.form()
    try:
        import time

        with pytest.raises(RankGroupError, match="rank 1|exited"):
            for _ in range(200):
                group.check_alive()
                time.sleep(0.05)
    finally:
        group.close()


def test_a_follower_that_dies_before_the_rendezvous_fails_form_bounded() -> None:
    # pgw#773/774: rank 0 must not enter the backend rendezvous blind. gloo
    # and NCCL block in `connect` for the whole collective timeout when a peer
    # is absent, so a follower that crashed on import turned "the group cannot
    # form" into a multi-minute stall INSIDE form() — with the collective
    # timeout (300s) as the only bound. Arrival is announced on the store and
    # polled against liveness, so this is bounded and TYPED.
    import time

    group = _cpu_group(_follower_dies_before_joining)
    start = time.monotonic()
    with pytest.raises(RankGroupError, match="rank 1|exited"):
        group.form()
    # The bound is the stall this replaces, not a number: the backend's own
    # collective timeout was the only thing that ended form() before, so
    # failing typed must cost a small fraction of it (pgw#845).
    from gen_worker.parallel import group as _group_mod
    blind_stall_s = float(_group_mod._COLLECTIVE_TIMEOUT_S)
    assert time.monotonic() - start < blind_stall_s / 5.0, (
        f"form() took longer than a fifth of the {blind_stall_s:.0f}s blind "
        f"rendezvous it replaces — it is not failing typed, it is stalling"
    )
    group.close()


def test_degree_one_group_forms_nothing() -> None:
    # A single-device group is exactly today's worker and must not pay a
    # process-group tax for a collective it will never make.
    group = RankGroup(devices=(0,), backend="gloo")
    spec = group.form()
    assert (spec.rank, spec.world_size) == (0, 1)
    assert group.degree == 1
    group.close()


def test_plan_agreement_names_the_exact_field() -> None:
    a = GroupPlan(precision_execution_lane="w8a8", compile_armed=True, sp_degree=2)
    b = GroupPlan(precision_execution_lane="w8a8", compile_armed=False, sp_degree=2)
    with pytest.raises(RankDivergence) as exc:
        a.assert_agrees(b, rank=1)
    assert exc.value.field_name == "compile_armed"
    assert exc.value.rank == 1
    a.assert_agrees(a, rank=1)  # agreement is silent


# ---------------------------------------------------------------------------
# D. Everything that would be silently wrong is refused by name.
# ---------------------------------------------------------------------------


def test_pertensor_w8a8_is_refused_under_sharding() -> None:
    # rowwise scales are per TOKEN (shard-invariant); pertensor derives one
    # scalar from the LOCAL shard, so two ranks quantize differently and the
    # group is silently wrong. Refuse, never downgrade.
    plan = GroupPlan(gemm_mode="pertensor", sp_degree=2)
    with pytest.raises(RankDivergence, match="rowwise"):
        plan.refuse_unless_cp_safe()
    # Degree 1 is untouched: pertensor is the correct Ada/sm_89 fast path.
    GroupPlan(gemm_mode="pertensor", sp_degree=1).refuse_unless_cp_safe()
    GroupPlan(gemm_mode="rowwise", sp_degree=2).refuse_unless_cp_safe()


class _FakeLinear:
    def __init__(self, mode: str) -> None:
        self.gemm_mode = mode

    def named_modules(self):
        yield "", self


class _FakeComponent:
    def __init__(self, mode: str) -> None:
        self._mode = mode

    def named_modules(self):
        yield "proj", _FakeLinear(self._mode)


class _FakePipeline:
    def __init__(self, mode: str = "rowwise") -> None:
        self.components = {"transformer": _FakeComponent(mode)}


def test_quant_mode_is_checked_on_the_live_modules() -> None:
    refuse_unless_shard_invariant_quant(_FakePipeline("rowwise"), degree=2)
    refuse_unless_shard_invariant_quant(_FakePipeline("pertensor"), degree=1)
    with pytest.raises(ContextParallelUnavailable, match="rowwise"):
        refuse_unless_shard_invariant_quant(_FakePipeline("pertensor"), degree=2)


def test_a_model_with_no_cp_plan_is_refused_not_sharded_anyway() -> None:
    with pytest.raises(ContextParallelUnavailable, match="_cp_plan"):
        install_context_parallel(_FakePipeline(), degree=2)
    # Degree 1 installs nothing and refuses nothing.
    assert install_context_parallel(_FakePipeline(), degree=1) == ()


def test_cpu_offload_and_context_parallelism_are_refused_together() -> None:
    # diffusers #12533: offload-then-parallelism raises a shape error AFTER
    # the first pipe call. Name it here instead.
    from gen_worker.parallel import cp as cp_mod

    class _Offloaded(_FakePipeline):
        _all_hooks = ["cpu_offload"]

    with pytest.raises(ContextParallelUnavailable, match="12533"):
        cp_mod._refuse_if_offloaded(_Offloaded())


def test_indivisible_shapes_are_refused_before_a_pod_is_committed() -> None:
    # Our T2V shapes divide cleanly at 2/4/8; diffusers #12536 is the crash
    # this replaces with a named refusal.
    refuse_unless_divisible(tokens=75600, heads=40, degree=2)
    refuse_unless_divisible(tokens=75600, heads=40, degree=8)
    with pytest.raises(ContextParallelUnavailable, match="sequence length"):
        refuse_unless_divisible(tokens=769, heads=40, degree=2)
    with pytest.raises(ContextParallelUnavailable, match="head count"):
        refuse_unless_divisible(tokens=75600, heads=40, degree=16)
    refuse_unless_divisible(tokens=769, heads=41, degree=1)  # degree 1: no-op


# ---------------------------------------------------------------------------
# E. WIDTH 4 — Paul, 2026-07-29: "4 workers for sequence-parallelism as well.
#    I just want to test these to make sure that they work right."
#
# Everything degree-4 that is not CUDA is provable here: a 4-rank rendezvous,
# a 4-way plan broadcast, and divergence/death in ONE of four ranks failing
# the whole group. NCCL bandwidth and Ulysses numerics need a pod; the
# control plane does not.
# ---------------------------------------------------------------------------


def _follower_echo4(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    import torch.distributed as dist

    pg = init_rank(spec)
    plan = chan.next_command(timeout=120)["plan"]
    assert plan.sp_degree == 4, plan
    assert plan.gemm_mode == "rowwise"
    assert plan.loras == (("style", 0.8),)
    chan.report_ready(spec.rank)
    dist.barrier(group=pg)


def _follower_rank3_dies(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    init_rank(spec)
    if spec.rank == 3:
        raise RuntimeError("rank 3's card fell off the bus")
    chan.next_command(timeout=120)
    chan.report_ready(spec.rank)


def test_degree_four_group_forms_and_every_rank_obeys_rank_zero() -> None:
    group = _cpu_group(_follower_echo4, degree=4)
    spec = group.form()
    try:
        assert (spec.rank, spec.world_size) == (0, 4)
        assert group.degree == 4
        plan = GroupPlan(
            precision_execution_lane="w8a8", gemm_mode="rowwise", sp_degree=4,
            compile_armed=False, loras=(("style", 0.8),),
        )
        group.send({"op": "arm", "plan": plan})
        group.wait_armed()
        group.barrier()   # all four ranks reached it
    finally:
        group.close()


def test_one_dead_rank_of_four_fails_the_whole_group() -> None:
    # Three healthy ranks must NOT quietly serve a degree-4 request as a
    # degree-3 one, and must not park forever on a rendezvous that can never
    # complete.
    import time

    group = _cpu_group(_follower_rank3_dies, degree=4)
    group.form()
    try:
        group.send({"op": "arm", "plan": GroupPlan(sp_degree=4)})
        with pytest.raises(RankGroupError):
            for _ in range(300):
                group.check_alive()
                time.sleep(0.05)
    finally:
        group.close()


def test_width_four_refusals_are_the_same_refusals() -> None:
    # The tested envelope is degree <= 4; nothing about the refusals is
    # degree-2-specific.
    with pytest.raises(RankDivergence, match="rowwise"):
        GroupPlan(gemm_mode="pertensor", sp_degree=4).refuse_unless_cp_safe()
    GroupPlan(gemm_mode="rowwise", sp_degree=4).refuse_unless_cp_safe()
    # Wan T2V divides cleanly at 4 — 40 heads, 75,600 tokens.
    refuse_unless_divisible(tokens=75600, heads=40, degree=4)
    with pytest.raises(ContextParallelUnavailable, match="head count"):
        refuse_unless_divisible(tokens=75600, heads=6, degree=4)
    with pytest.raises(ContextParallelUnavailable, match="_cp_plan"):
        install_context_parallel(_FakePipeline(), degree=4)
    with pytest.raises(ContextParallelUnavailable, match="rowwise"):
        refuse_unless_shard_invariant_quant(_FakePipeline("pertensor"), degree=4)


# ---------------------------------------------------------------------------
# F. The EXECUTOR WIRING — the pipeline call is the SPMD unit.
# ---------------------------------------------------------------------------


def test_the_call_gate_routes_the_pipeline_through_the_group() -> None:
    from gen_worker.parallel.runtime import SequenceRuntime, arm_sequence_gate

    calls: list = []

    class _Pipe:
        def __call__(self, *a, **kw):
            calls.append((a, kw))
            return "latent"

    class _FakeRuntime(SequenceRuntime):
        def __init__(self):
            super().__init__(devices=(0, 1, 2, 3))
            self.broadcast: list = []
            self._armed = True

        def call_with(self, base_call, pipe, *a, **kw):
            self.broadcast.append((a, kw))
            return base_call(pipe, *a, **kw)

    pipe, rt = _Pipe(), _FakeRuntime()
    assert arm_sequence_gate(pipe, rt)
    # The endpoint calls its pipeline exactly as it always has...
    assert pipe(3, steps=4) == "latent"
    # ...and the group saw the same call.
    assert rt.broadcast == [((3,), {"steps": 4})] == calls
    # isinstance/identity survive the wrap (same technique as gw#551).
    assert isinstance(pipe, _Pipe) and type(pipe).__name__ == "_Pipe"
    # Idempotent: re-arming swaps the runtime, never re-wraps.
    assert arm_sequence_gate(pipe, rt)

    # Degree 1 never gates: today's worker must be byte-identical.
    assert not arm_sequence_gate(_Pipe(), SequenceRuntime(devices=(0,)))


def test_call_marshalling_survives_the_wire() -> None:
    # The model call must arrive at every rank IDENTICAL. CUDA tensors go via
    # CPU (the followers own other cards) and a Generator is not picklable at
    # all — only its seed has to agree.
    import pickle

    from gen_worker.parallel.runtime import _dehydrate, _rehydrate

    payload = {
        "latents": torch.ones(2, 3),
        "steps": 4,
        "prompt": "a cat",
        "generator": torch.Generator().manual_seed(1234),
        "sigmas": [torch.zeros(2), 0.5],
    }
    wire = _dehydrate(payload)
    pickle.loads(pickle.dumps(wire))          # must cross the command channel
    back = _rehydrate(wire, device=0)
    assert torch.equal(back["latents"].cpu(), payload["latents"])
    assert back["steps"] == 4 and back["prompt"] == "a cat"
    assert back["generator"].initial_seed() == 1234
    assert torch.equal(back["sigmas"][0].cpu(), torch.zeros(2))
    assert back["sigmas"][1] == 0.5


def test_a_group_that_cannot_be_sharded_is_refused_not_served_at_degree_one() -> None:
    # Serving degree 1 against a degree-D promise delivers a fraction of the
    # tier that was sold, invisibly. Exactly one class-annotated pipeline slot
    # can be the SPMD unit.
    from gen_worker.executor import Executor, _ClassRecord
    from gen_worker.parallel import ContextParallelUnavailable
    from gen_worker.registry import EndpointSpec

    def _h(self, ctx, payload):  # pragma: no cover
        return None

    class _C:
        pass

    spec = EndpointSpec(name="f", method=_h, kind="inference", payload_type=dict,
                        output_mode="single", cls=_C)
    ex = Executor.__new__(Executor)
    rec = _ClassRecord(cls=_C)
    for pipes in ({}, {"a": object(), "b": object()}):
        rec.slot_pipelines = dict(pipes)
        with pytest.raises(ContextParallelUnavailable, match="exactly ONE"):
            ex._sequence_boot_slot(spec, rec)
    rec.slot_pipelines = {"pipeline": object()}
    assert ex._sequence_boot_slot(spec, rec) == "pipeline"


def test_async_handlers_are_refused_on_a_multi_group_worker() -> None:
    # ctx.device still resolves from the CALLING THREAD's current device. A
    # sync handler's thread is pinned to its group; the shared event loop is
    # not. Refuse by name rather than compute on a sibling's card.
    from gen_worker.executor import Executor
    from gen_worker.registry import EndpointSpec
    from gen_worker.topology import ExecutionTopology

    def _h(self, ctx, payload):  # pragma: no cover
        return None

    class _C:
        pass

    sync = EndpointSpec(name="s", method=_h, kind="inference", payload_type=dict,
                        output_mode="single", cls=_C)
    asy = type(sync)(**{**sync.__dict__, "name": "a", "is_async": True})

    ex = Executor.__new__(Executor)
    ex.topology = ExecutionTopology.single()
    assert ex._multi_group_handler_refusal(sync) == ""
    assert ex._multi_group_handler_refusal(asy) == ""   # one group: unchanged

    ex.topology = ExecutionTopology(gpu_count=4, gpus_per_execution_group=1)
    assert ex._multi_group_handler_refusal(sync) == ""
    refusal = ex._multi_group_handler_refusal(asy)
    assert "async handlers are not yet served" in refusal
    assert "ctx.device" in refusal
