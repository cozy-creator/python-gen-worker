"""Per-group process groups and symmetric failure, on the gloo/CPU multi-rank
rig.

Layers exercised, per test:

- ``init_rank``/``RankGroup`` (the process-group construction itself): two
  degree-2 groups coexist in ONE parent process with independent worlds —
  the shape (``gpu_count=4, gpus_per_execution_group=2``) that joining the
  default group corrupts. Collectives run CONCURRENTLY on both groups
  through the real diffusers CP hooks (split/gather over our mesh) and the
  values prove no cross-talk.
- ``SequenceRuntime.call_with`` (the serving path): a rank-0 exception
  mid-call condemns the group (typed, torn down, other groups untouched)
  instead of wedging; an unpicklable argument is a typed per-request error
  with the group still healthy.
- The process group's collective timeout (the ONLY unwedging mechanism when
  a peer leaves an SPMD region): a follower that never joins the collective
  strands rank 0 for the timeout, then rank 0 fails TYPED.
- ``RankGroup.close``/``wait_armed``: teardown and arming never perform a
  collective, so a hung or dead follower cannot park them.

Barrier-reachability INSIDE handlers says nothing about group isolation, which
is why the rows above target the default-PG collision, the missing timeout and
the collective close directly.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="needs torch.distributed with the gloo backend",
)

from gen_worker.parallel import RankGroupError, wire  # noqa: E402
from gen_worker.parallel.group import FollowerChannel, RankSpec, init_rank  # noqa: E402
from gen_worker.parallel.plan import BootPlan, GroupPlan  # noqa: E402
from gen_worker.parallel.runtime import SequenceRuntime  # noqa: E402


# ---------------------------------------------------------------------------
# The toy SPMD unit: a real nn.Module with a real diffusers _cp_plan, so the
# rig runs the SAME hook machinery (EquipartitionSharder over our mesh /
# process group) a production transformer runs.
# ---------------------------------------------------------------------------


def _make_toy_pipe() -> Any:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
    )

    class _Inner(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states * 2.0

    class _Toy(torch.nn.Module):
        _cp_plan = {
            "inner": {
                "hidden_states": ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=False
                )
            },
            "": ContextParallelOutput(gather_dim=0, expected_dims=2),
        }

        def __init__(self) -> None:
            super().__init__()
            self.inner = _Inner()

        def enable_parallelism(self, **kwargs: Any) -> None:  # pragma: no cover
            raise AssertionError(
                "the default-PG path must never be taken (pgw#773)")

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.inner(hidden_states)

    class _Pipe:
        def __init__(self) -> None:
            self.components = {"transformer": _Toy()}
            self.transformer = self.components["transformer"]

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.transformer(hidden_states)

    return _Pipe()


def _rig_entry(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover - spawned
    """A follower that mirrors sequence_rank_main against the toy pipe."""
    from gen_worker.parallel import wire
    from gen_worker.parallel.cp import CpComms, gated_call, install_context_parallel

    pg = init_rank(spec)
    first = chan.next_command(timeout=120)
    assert isinstance(first, wire.Arm), first
    plan: GroupPlan = first.plan
    plan.refuse_unless_cp_safe()
    pipe = _make_toy_pipe()
    install_context_parallel(
        pipe, degree=plan.sp_degree,
        comms=CpComms(pg=pg, rank=spec.rank, device="cpu"),
    )
    chan.report_ready(spec.rank)
    while True:
        cmd = chan.next_command()
        if not isinstance(cmd, wire.Run):
            return
        args, _kwargs = wire.run_call(cmd, device="cpu")
        # Same gate `sequence_rank_main` enters: a follower's forward IS
        # rank-symmetric, so it is inside the gate.
        with torch.no_grad(), gated_call():
            pipe(*args)


def _rig_entry_never_ready(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover
    init_rank(spec)
    chan.next_command(timeout=120)
    time.sleep(600)


def _rig_entry_skips_collectives(spec: RankSpec, chan: FollowerChannel) -> None:  # pragma: no cover
    """Armed and alive, but never joins a RUN's collectives — the follower-
    hang case. Rank 0 must time out TYPED, not park forever."""
    from gen_worker.parallel import wire

    init_rank(spec)
    first = chan.next_command(timeout=120)
    assert isinstance(first, wire.Arm)
    chan.report_ready(spec.rank)
    while True:
        cmd = chan.next_command()
        if not isinstance(cmd, wire.Run):
            return
        time.sleep(600)  # a RUN arrived; ignore it


def _armed_runtime(
    devices: tuple, entry: Any = _rig_entry, timeout_s: float = 60.0,
) -> tuple:
    pipe = _make_toy_pipe()
    # arming no longer takes a duration at all — a follower that
    # keeps working arms for as long as its work takes, and one that stops
    # working is condemned by silence.
    rt = SequenceRuntime(
        devices, entry=entry, backend="gloo",
        collective_timeout_s=timeout_s,
    )
    installed = rt.arm(pipe, BootPlan(degree=len(devices)),
                       GroupPlan(sp_degree=len(devices)))
    assert installed == ("transformer",)
    return rt, pipe


def _base_call(pipe: Any, *args: Any, **kwargs: Any) -> Any:
    return pipe.__class__.__call__(pipe, *args, **kwargs)


def test_two_degree2_groups_coexist_with_no_cross_talk() -> None:
    # gpu_count=4, gpus_per_execution_group=2: the shape DPA-1 corrupts. Both groups arm
    # in ONE parent process and run their sharded forwards CONCURRENTLY;
    # each output must be its own group's input times 2, at full length
    # (the gather ran over the right world).
    rt0, pipe0 = _armed_runtime((0, 1))
    rt1, pipe1 = _armed_runtime((2, 3))
    try:
        x0 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        x1 = 100.0 + torch.arange(8, dtype=torch.float32).reshape(4, 2)
        results: dict = {}

        def drive(name: str, rt: SequenceRuntime, pipe: Any, x: torch.Tensor) -> None:
            try:
                results[name] = rt.call_with(_base_call, pipe, x)
            except Exception as exc:  # pragma: no cover - failure detail
                results[name] = exc

        threads = [
            threading.Thread(target=drive, args=("g0", rt0, pipe0, x0)),
            threading.Thread(target=drive, args=("g1", rt1, pipe1, x1)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        assert isinstance(results["g0"], torch.Tensor), results["g0"]
        assert isinstance(results["g1"], torch.Tensor), results["g1"]
        assert torch.equal(results["g0"], x0 * 2.0)
        assert torch.equal(results["g1"], x1 * 2.0)
    finally:
        rt0.close()
        rt1.close()


def test_closing_one_group_leaves_the_sibling_serving() -> None:
    # DPA-1's second instance: the old close() destroyed the DEFAULT process
    # group — vacating one record killed every other group's world.
    rt0, pipe0 = _armed_runtime((0, 1))
    rt1, pipe1 = _armed_runtime((2, 3))
    try:
        rt0.close()
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        out = rt1.call_with(_base_call, pipe1, x)
        assert torch.equal(out, x * 2.0)
    finally:
        rt1.close()


def test_rank0_exception_condemns_the_group_instead_of_wedging_it() -> None:
    # rank 0 raising mid-call (OOM/deadline/cancel) used to skip the
    # trailing barrier and park the followers forever; the next request then
    # blocked in a broadcast with no timeout. Now: the ORIGINAL error
    # surfaces, the group is condemned by name, teardown is bounded, and
    # further calls refuse typed instead of hanging.
    rt, pipe = _armed_runtime((0, 1))
    try:
        def _boom(pipe: Any, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("CUDA out of memory (simulated)")

        with pytest.raises(RuntimeError, match="out of memory"):
            rt.call_with(_boom, pipe, torch.ones(4, 2))
        assert rt.broken
        with pytest.raises(RankGroupError, match="condemned"):
            rt.call_with(_base_call, pipe, torch.ones(4, 2))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and rt._group is not None:
            time.sleep(0.1)
        assert rt._group is None, "condemned group was not torn down"
    finally:
        rt.close()


def test_sibling_group_survives_a_condemned_neighbour() -> None:
    rt0, pipe0 = _armed_runtime((0, 1))
    rt1, pipe1 = _armed_runtime((2, 3))
    try:
        def _boom(pipe: Any, *a: Any, **kw: Any) -> Any:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            rt0.call_with(_boom, pipe0, torch.ones(4, 2))
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        assert torch.equal(rt1.call_with(_base_call, pipe1, x), x * 2.0)
    finally:
        rt0.close()
        rt1.close()


def test_a_follower_that_skips_the_collective_times_out_typed() -> None:
    # `_RENDEZVOUS_TIMEOUT_S` was dead code — NO collective had a
    # timeout, so a hung follower parked rank 0 forever. The process group
    # now carries a real timeout and rank 0 fails typed, condemning the
    # group.
    collective_timeout_s = 5.0
    rt, pipe = _armed_runtime((0, 1), entry=_rig_entry_skips_collectives,
                              timeout_s=collective_timeout_s)
    try:
        start = time.monotonic()
        with pytest.raises(Exception):
            rt.call_with(_base_call, pipe, torch.ones(4, 2))
        # Bounded by the timeout the group was CONFIGURED with, not by a
        # constant that says nothing about this rig.
        assert time.monotonic() - start < collective_timeout_s * 4
        assert rt.broken
    finally:
        rt.close()


def test_an_uncrossable_argument_is_typed_and_does_not_condemn() -> None:
    # DPA-28's shape: a `callback_on_step_end` closure cannot cross the rank
    # boundary. The old broadcast_object_list raised AFTER the followers
    # were already in _recv, desynchronizing the group. Now the marshalling
    # runs eagerly on rank 0: typed error, group coherent, next call serves.
    rt, pipe = _armed_runtime((0, 1))
    try:
        with pytest.raises(RankGroupError, match="cannot cross the rank"):
            rt.call_with(
                _base_call, pipe, torch.ones(4, 2),
                callback=lambda *a: None,
            )
        assert not rt.broken
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        assert torch.equal(rt.call_with(_base_call, pipe, x), x * 2.0)
    finally:
        rt.close()


def test_arming_a_group_whose_follower_goes_silent_fails_typed(
    monkeypatch: Any,
) -> None:
    """RE-CUT for pgw#892. This used to assert that a never-ready follower
    failed within a multiple of `_ARM_TIMEOUT_S = 1800`, which is the flat
    wall clock that issue deletes: arming "covers a follower's full pipeline
    materialization (a cold model load)", the first self-mint measured ~28 min
    and pgw#846 projects 47 min - 6.26 h for sdxl, so the bound condemned
    healthy work and the RIGHT never-readies test is this one.

    The follower here parks in `time.sleep(600)` — alive, and doing nothing at
    all — so it produces no CPU and no I/O and the silence window fires. The
    window is shortened for the test because it is a window over a REAL
    signal, not a budget for the job."""
    from gen_worker.parallel import group as group_mod

    monkeypatch.setattr(group_mod, "_STAGING_SILENCE_WINDOW_S", 3.0)
    pipe = _make_toy_pipe()
    rt = SequenceRuntime(
        (0, 1), entry=_rig_entry_never_ready, backend="gloo",
        collective_timeout_s=30.0,
    )
    with pytest.raises(RankGroupError, match="no CPU or I/O"):
        rt.arm(pipe, BootPlan(degree=2), GroupPlan(sp_degree=2))
    rt.close()


def test_close_is_never_a_collective_and_completes_against_a_wedged_follower() -> None:
    rt, pipe = _armed_runtime((0, 1), entry=_rig_entry_skips_collectives,
                              timeout_s=5.0)
    # A RUN the follower ignores: it is now parked in a 600s sleep.
    parked_s = 600.0
    rt._group.send(wire.Run())
    start = time.monotonic()
    rt.close()   # must join/terminate, never broadcast
    # The property is that close() does NOT wait out the parked follower.
    assert time.monotonic() - start < parked_s / 20.0


def test_group_mesh_contract_matches_installed_diffusers() -> None:
    # Pins the diffusers surface `_GroupMesh` serves: setup() must derive the
    # flattened/ulysses meshes and local ranks from OUR mesh, so an upstream
    # change in ContextParallelConfig.setup breaks HERE, loudly.
    from diffusers.models._modeling_parallel import ContextParallelConfig

    from gen_worker.parallel.cp import _GroupMesh

    sentinel_pg = object()
    mesh = _GroupMesh(sentinel_pg, 2, 1)
    cfg = ContextParallelConfig(ulysses_degree=2)
    cfg.setup(rank=1, world_size=2, device=torch.device("cpu"), mesh=mesh)
    assert cfg._flattened_mesh.get_group() is sentinel_pg
    assert cfg._ulysses_mesh.get_group() is sentinel_pg
    assert cfg._ulysses_local_rank == 1
    assert cfg._ring_local_rank == 0
    from gen_worker.parallel import ContextParallelUnavailable

    with pytest.raises(ContextParallelUnavailable, match="ring"):
        cfg._ring_mesh.get_group()


def test_executor_close_sequence_group_runs_off_the_event_loop() -> None:
    # The recovery path used to run runtime.close() ON the event
    # loop, whose first act was a broadcast that could block forever — the
    # loop (and the heartbeat) stopped, presenting as a platform stall.
    import asyncio

    from gen_worker.executor import Executor, _ClassRecord

    ex = Executor.__new__(Executor)
    ex._sequence_plans = {}
    closed = threading.Event()
    loop_thread: dict = {}

    class _SlowRuntime:
        def close(self) -> None:
            loop_thread["thread"] = threading.current_thread()
            time.sleep(0.5)   # a bounded process join
            closed.set()

    rec = _ClassRecord(cls=object)
    rec.sp_runtime = _SlowRuntime()

    async def _drive() -> bool:
        ex._close_sequence_group(rec)
        # The property is that the call RETURNED without waiting for
        # the 0.5s close — proven by the close still being in flight when it
        # did, not by an elapsed-time budget that asserts the runner instead.
        returned_early = not closed.is_set()
        # The loop must stay responsive while close() runs elsewhere.
        await asyncio.sleep(0.05)
        return returned_early

    assert asyncio.run(_drive()), "close blocked the loop until it finished"
    assert closed.wait(5)
    assert loop_thread["thread"].name.startswith("sp-close")
    assert rec.sp_runtime is None
