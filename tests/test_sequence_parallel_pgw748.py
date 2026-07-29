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
    GroupPlan,
    RankDivergence,
    RankGroup,
    RankGroupError,
    RankSpec,
    broadcast_plan,
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


def _follower_echo(spec: RankSpec) -> None:  # pragma: no cover - spawned
    """The narrow follower loop: join, receive the plan, barrier, exit."""
    import torch.distributed as dist

    plan = broadcast_plan(None, rank=spec.rank)
    assert plan.precision_lane == "w8a8"
    assert plan.gemm_mode == "rowwise"
    dist.barrier()


def _follower_disagrees(spec: RankSpec) -> None:  # pragma: no cover - spawned
    """A follower whose own card would have chosen differently. It must NOT
    serve its own choice — it must fail the group."""
    mine = GroupPlan(precision_lane="bf16", sp_degree=2)
    broadcast_plan(mine, rank=spec.rank)


def _follower_dies(spec: RankSpec) -> None:  # pragma: no cover - spawned
    raise RuntimeError("this rank's card is not there")


def _cpu_group(entry: Any, degree: int = 2) -> RankGroup:
    return RankGroup(devices=tuple(range(degree)), backend="gloo", entry=entry)


def test_rank_zero_decides_and_every_rank_obeys() -> None:
    group = _cpu_group(_follower_echo)
    spec = group.form()
    try:
        assert spec.rank == 0 and spec.world_size == 2
        plan = GroupPlan(
            precision_lane="w8a8", gemm_mode="rowwise", sp_degree=2,
            loras=(("style", 0.8),),
        )
        decided = broadcast_plan(plan, rank=0)
        assert decided == plan
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
        broadcast_plan(
            GroupPlan(precision_lane="w8a8", sp_degree=2), rank=0)
        with pytest.raises(RankGroupError):
            for _ in range(200):
                group.check_alive()
                import time

                time.sleep(0.05)
    finally:
        group.close()


def test_a_dead_follower_is_a_loud_failure_not_a_hang() -> None:
    group = _cpu_group(_follower_dies)
    group.form()
    try:
        import time

        with pytest.raises(RankGroupError, match="rank 1|exited"):
            for _ in range(200):
                group.check_alive()
                time.sleep(0.05)
    finally:
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
    a = GroupPlan(precision_lane="w8a8", compile_armed=True, sp_degree=2)
    b = GroupPlan(precision_lane="w8a8", compile_armed=False, sp_degree=2)
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


def _follower_echo4(spec: RankSpec) -> None:  # pragma: no cover - spawned
    import torch.distributed as dist

    plan = broadcast_plan(None, rank=spec.rank)
    assert plan.sp_degree == 4, plan
    assert plan.gemm_mode == "rowwise"
    assert plan.loras == (("style", 0.8),)
    dist.barrier()


def _follower_rank3_dies(spec: RankSpec) -> None:  # pragma: no cover - spawned
    import torch.distributed as dist

    if spec.rank == 3:
        raise RuntimeError("rank 3's card fell off the bus")
    broadcast_plan(None, rank=spec.rank)
    dist.barrier()


def test_degree_four_group_forms_and_every_rank_obeys_rank_zero() -> None:
    group = _cpu_group(_follower_echo4, degree=4)
    spec = group.form()
    try:
        assert (spec.rank, spec.world_size) == (0, 4)
        assert group.degree == 4
        plan = GroupPlan(
            precision_lane="w8a8", gemm_mode="rowwise", sp_degree=4,
            compile_armed=False, loras=(("style", 0.8),),
        )
        assert broadcast_plan(plan, rank=0) == plan
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
