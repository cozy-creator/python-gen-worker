"""pgw#775: at degree>1 every forward is rank-symmetric or refused BY NAME.

Installing context parallelism puts split/all-to-all/gather hooks ON THE
MODULES, so from that moment every forward through them issues collectives.
The only seam that supplies participants on the other ranks is the pipeline-
level ``__call__`` gate — so a hot-swap warm compile, a mint seed, a proof
warmup, an activation/degraded probe or an endpoint calling a component by hand
all forward on rank 0 alone and hang the whole group in NCCL with no timeout
anyone can observe.

Layers exercised:

- ``parallel/cp.py`` (the module hooks): a real degree-2 gloo group is armed
  through the real diffusers CP machinery, then a component is forwarded
  OUTSIDE the gate. It must raise ``UngatedShardedForward`` on the calling
  thread; without the check that forward blocks in the collective forever. The
  group must survive and the next gated call must serve.
- the SAME check from another thread, which is where the real strays come from
  (the shared shape-warm thread, a background mint turn).
- ``Executor`` construction (the "eager only" claim itself): on a
  ``parallel="sequence", degree>1`` topology no compile snapshot is fetched, no
  compile targets install (so nothing routes to the warm thread, no background
  mint, no adoption), and the arming scope is a no-op so an endpoint's own
  ``arm_compile`` inside setup() cannot arm one either. At degree 1 every one of
  those is untouched.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="needs torch.distributed with the gloo backend",
)

from gen_worker.parallel import UngatedShardedForward  # noqa: E402
from gen_worker.topology import ExecutionTopology  # noqa: E402

from test_sp_group_isolation_pgw773_774 import (  # noqa: E402
    _armed_runtime,
    _base_call,
)


def test_an_ungated_forward_is_refused_by_name_not_hung() -> None:
    rt, pipe = _armed_runtime((0, 1))
    try:
        with pytest.raises(UngatedShardedForward, match="outside the group's call gate"):
            pipe.transformer(torch.ones(4, 2))
        # The refusal is not a condemnation: nothing was sent to the followers,
        # so the group is still coherent and the next real call serves.
        assert not rt.broken
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        assert torch.equal(rt.call_with(_base_call, pipe, x), x * 2.0)
    finally:
        rt.close()


def test_a_stray_forward_from_the_warm_thread_is_refused_bounded() -> None:
    # The shape-warm thread's `job.compiled(*args)` targets a compile-capable
    # SUBMODULE, not the gated pipeline, and runs on its own thread — the
    # pgw#775 scenario verbatim. It must fail fast, not park the thread (which
    # in production holds rec.turn_mutex and takes the record down with it).
    rt, pipe = _armed_runtime((0, 1))
    try:
        seen: dict = {}

        def _warm() -> None:
            try:
                pipe.transformer(torch.ones(4, 2))
            except BaseException as exc:  # noqa: BLE001
                seen["exc"] = exc

        t = threading.Thread(target=_warm, name="shape-warm")
        t.start()
        t.join(30)
        assert not t.is_alive(), "the stray forward parked the warm thread"
        assert isinstance(seen.get("exc"), UngatedShardedForward), seen
    finally:
        rt.close()


# ---------------------------------------------------------------------------
# The executor half: "eager only at degree>1" by construction.
# ---------------------------------------------------------------------------


class _FakeCell:
    family = "toy"


class _FakeSpec:
    name = "toy_fn"
    compile = object()
    cls = object

    def compile_cell(self) -> Any:
        return _FakeCell()


def _executor_at(topology: ExecutionTopology) -> Any:
    from gen_worker.executor import Executor

    ex = Executor.__new__(Executor)
    ex.topology = topology
    return ex


def test_a_context_parallel_pod_installs_no_compile_targets() -> None:
    from gen_worker.executor import _ClassRecord

    sp = ExecutionTopology(gpu_count=2, gpus_per_execution_group=2, parallel="sequence")
    assert sp.degree == 2 and sp.execution_groups == 1
    ex = _executor_at(sp)
    assert ex._eager_only_reason()

    rec = _ClassRecord(cls=object)
    rec.compile_targets = {"stale": object()}
    ex._install_compile_targets(rec, _FakeSpec(), [object()])
    assert rec.compile_targets == {}, (
        "a degree>1 record must hold no compile target: a target is what routes"
        " a novel signature to the warm thread's ungated forward")


def test_degree_one_is_unchanged() -> None:
    # The G=1/D=1 invariant: nothing about eager-only may touch a pod that has
    # never heard of sequence parallelism.
    assert _executor_at(ExecutionTopology.single())._eager_only_reason() == ""
    dp = ExecutionTopology(gpu_count=4, gpus_per_execution_group=1)
    assert dp.degree == 1 and dp.execution_groups == 4
    assert _executor_at(dp)._eager_only_reason() == ""


def test_every_degree_above_one_is_eager_only() -> None:
    """pgw#1113/pgw#819 FLIPS this assertion, deliberately.

    It used to read: *"`parallel="internal"` means the MODEL spans the cards
    by its own arrangement — no CP hooks, so compile is still legal."* The
    premise is true and the conclusion did not follow. "No CP hooks" answers
    the HANG question (nothing forwards outside a collective gate) and says
    nothing about the IDENTITY question: a model whose own device map splits
    its modules across `cuda:0`/`cuda:1` has inductor bake that placement into
    the artifact, while the cell key scrubs the device index by design — so
    the 2-card cell and the 1-card cell published under one key and each pod
    adopted the other's, silently, in both directions (pgw#819, measured).

    The gate was an allowlist by mode NAME, so `cfg` — declared at
    `topology.PARALLEL_CFG` and platform-installed like `sequence` — was
    uncovered too, latent only because it has no serve-side implementation
    yet. The rule is `degree > 1`; a gate that must be widened once per new
    mode is not a rule.
    """
    # `parallel=""` is not in this list because it cannot be: a group wider
    # than one card with no parallel mechanism is refused by the topology
    # decoder itself (`topology_parallel_required`), so `degree > 1` and "some
    # mode is declared" are the same statement.
    for parallel in ("internal", "sequence", "cfg"):
        topo = ExecutionTopology(
            gpu_count=2, gpus_per_execution_group=2, parallel=parallel)
        assert topo.degree == 2
        assert _executor_at(topo)._eager_only_reason(), (
            f"degree-2 pod with parallel={parallel!r} must refuse to arm a "
            "compile cell: no cell can state the placement it was baked for")
