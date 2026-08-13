"""pgw#782: what four execution groups in ONE process are allowed to share.

The pgw#748 real-endpoint campaign measured width-4 data parallelism at 0.88x
of serial throughput and attributed it to "the host-side decode/encode tail".
The instrumented probe (4xA40, sdxl, `~/cozy/samples/dpfix/`) says otherwise:
the tail is 0.5 s of a 72 s request, the four model calls fully OVERLAP, all
four cards sit at ~21% utilization, and the process uses 2.3 of its 32.3
allowed cores with zero CFS throttling. Four SEPARATE worker processes on the
same pod ran the identical burst at **4.00x** (8.67-8.90 s solo -> 8.58-8.83 s
concurrent, 91-93% utilization on every card, per-step 275 ms unchanged). The
ceiling is the shared interpreter, not the tail and not a lock.

What is asserted here is the part that is fixable inside one process, plus the
property Paul suspected (and the audit disproved) so it can never regress:

A. The MODEL CALLS OF FOUR GROUPS OVERLAP. A per-class lock across the model
   call is the one shape that would serialize them; the group ordinal is part
   of instance identity precisely so it cannot. RED-verified by restoring the
   pre-pgw#748 behavior (the group does not join the key).
B. Intra-op threads are divided by the groups that share the process, and the
   allowance comes from the CGROUP, not the host's cpu count.
C. The buffered-finalize bound is per GROUP, not per process: at G=4 a flat 2
   made two of four groups wait on an unrelated card's encode.

Neither B nor C is the throughput fix — both measured neutral on the sdxl
launch path (37.1 s unfixed vs 36.7 s fixed at width 4). They remove a
misconfiguration and a cross-group serializer; the 4x lives in pgw#783.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import msgspec
import pytest

from gen_worker import RequestContext, Resources, endpoint, worker_function
from gen_worker import cpu_budget, video_encode
from gen_worker.executor import Executor, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.progress_wait import Cadence, await_count
from gen_worker.registry import extract_specs
from gen_worker.topology import ENV_VAR, ExecutionTopology

GROUPS = 4
#: The four handlers rendezvous instead of sleeping a fixed time: "all four
#: were inside the model call at once" is then a fact, not a wall-clock
#: inference that a loaded box can flake.
BARRIER_TIMEOUT_S = 5.0


@pytest.fixture(autouse=True)
def _restore_process_state() -> Any:
    """The impositions under test are deliberately process-global.

    That is precisely why every one of them must be put back. This fixture
    restored `torch` thread count and the encode semaphore but **not the
    INSTALLED TOPOLOGY**, and the tests here boot four-group workers
    (`{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}`). `monkeypatch` puts the ENV
    VAR back; it cannot put back the module global that the boot path writes
    through `install_topology()`.

    Measured cost: a leaked 4-group topology made
    `test_guard_miss_pgw680.py::test_tenant_guard_miss_end_to_end` fail 7 files
    later with `in-process mint refused at groups=4` — `fleet_compiled_graphs.py:1050`
    doing its job exactly right (pgw#777/DPA-8 forbids an in-process inductor
    capture on a multi-group worker) against state that belonged to this file.
    The victim passes alone and failed only in suite order, which is why it read
    as flaky for as long as nobody ran the full suite.

    The sibling `test_group_device_map_pgw773.py` already does this correctly;
    this is the same two lines.
    """
    from gen_worker.topology import install_topology

    try:
        import torch

        threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        torch = None  # type: ignore[assignment]
        threads = 0
    video_encode._finalize_sem = None
    install_topology(None)
    yield
    if torch is not None:
        torch.set_num_threads(threads)
    video_encode._finalize_sem = None
    install_topology(None)


# ---------------------------------------------------------------------------
# A. four groups' model calls overlap
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    group: int = -1


class _GroupHarness:
    """Four device groups over the REAL executor + handle_run_job lane. The
    handler is CPU-only (``Resources(gpu=False)``) so the tape measures the
    execute path's own concurrency on any box; the group ordinal still comes
    from ``ResolvedCompute.gpu_index`` exactly as on a GPU pod."""

    def __init__(self, tmp_path: Path) -> None:
        # (gpu_index, start, end) per model call, from inside the handler.
        self.calls: List[Tuple[int, float, float]] = []
        # Which groups met all their siblings INSIDE the model call.
        self.rendezvous: List[int] = []
        self.sent: List[pb.WorkerMessage] = []
        lock = threading.Lock()
        barrier = threading.Barrier(GROUPS)
        harness = self

        @endpoint(resources=Resources(gpu=False))
        class GroupEndpoint:
            def setup(self) -> None:
                self.ready = True

            @worker_function()
            def generate(self, ctx: RequestContext, payload: _In) -> _Out:
                group = int(payload.prompt)
                t0 = time.monotonic()
                # Real work would be kernel launches. The rendezvous stands in
                # for "this call occupies its group" AND proves concurrency
                # without a timing assumption: it can only trip if all four
                # groups are inside the model call simultaneously. A serialized
                # executor breaks the barrier instead (that is the RED tape).
                try:
                    barrier.wait(timeout=BARRIER_TIMEOUT_S)
                    met = True
                except threading.BrokenBarrierError:
                    met = False
                with lock:
                    harness.calls.append((group, t0, time.monotonic()))
                    if met:
                        harness.rendezvous.append(group)
                return _Out(group=group)

        self.specs = extract_specs(GroupEndpoint)
        (self.spec,) = [s for s in self.specs if s.name == "generate"]

        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        store = ModelStore(_send, cache_dir=tmp_path / "cas")
        topo = ExecutionTopology.from_env(
            {ENV_VAR: '{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}'})
        self.ex = Executor(self.specs, _send, store=store, topology=topo)

    def _run(self, group: int) -> pb.RunJob:
        return pb.RunJob(
            request_id=f"g{group}", attempt=1, function_name=self.spec.name,
            input_payload=msgspec.msgpack.encode(_In(prompt=str(group))),
            compute=pb.ResolvedCompute(accelerator="none", gpu_index=group),
        )

    async def burst(self) -> List[pb.JobResult]:
        runs = [self._run(g) for g in range(GROUPS)]
        for run in runs:
            await self.ex.handle_run_job(run)
        jobs = [self.ex.jobs[(r.request_id, r.attempt)] for r in runs]
        await asyncio.gather(*[j.task for j in jobs if j.task is not None])
        return [m.job_result for m in self.sent
                if m.WhichOneof("msg") == "job_result"]


def _max_overlap(calls: List[Tuple[int, float, float]]) -> int:
    """How many model calls were in flight at the busiest instant."""
    edges = sorted([(t0, 1) for _g, t0, _t1 in calls]
                   + [(t1, -1) for _g, _t0, t1 in calls])
    live = best = 0
    for _t, delta in edges:
        live += delta
        best = max(best, live)
    return best


def test_four_groups_model_calls_overlap(tmp_path: Path) -> None:
    h = _GroupHarness(tmp_path)
    results = asyncio.run(h.burst())

    assert [int(r.status) for r in results] == [pb.JOB_STATUS_OK] * GROUPS
    assert sorted(g for g, _s, _e in h.calls) == list(range(GROUPS))
    # The rendezvous tripped: all four groups were inside the model call at the
    # same instant. No wall-clock threshold, so a loaded box cannot flake it.
    assert sorted(h.rendezvous) == list(range(GROUPS)), h.rendezvous
    assert _max_overlap(h.calls) == GROUPS, h.calls
    # Four groups => four instances, and each key carries its ordinal.
    ordinals = sorted(
        next((int(p[1]) for p in key[2:]
              if isinstance(p, tuple) and p and p[0] == "device_group"), 0)
        for key in h.ex._classes)
    assert ordinals == list(range(GROUPS)), list(h.ex._classes)


def test_four_groups_serialize_if_the_group_leaves_instance_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED-verify for the shape that was suspected and is NOT what happens:
    with the group elided from identity (the pre-pgw#748 tree) all four
    dispatches share one record, so the per-instance single-flight gate makes
    the model calls take turns. This is the tape that would catch a
    regression back to a per-class serializer."""
    h = _GroupHarness(tmp_path)
    monkeypatch.setattr(
        Executor, "_group_effective_spec", lambda self, spec, group: spec)
    results = asyncio.run(h.burst())

    assert [int(r.status) for r in results] == [pb.JOB_STATUS_OK] * GROUPS
    # Nobody met a sibling: the four calls took turns.
    assert h.rendezvous == [], h.rendezvous
    assert _max_overlap(h.calls) == 1, h.calls
    assert len(h.ex._classes) == 1


# ---------------------------------------------------------------------------
# B. the intra-op thread budget
# ---------------------------------------------------------------------------


def test_cpu_allowance_is_the_cgroup_quota_not_the_host_cpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota = tmp_path / "cpu.max"
    quota.write_text("3230000 100000\n")  # the measured A40 pod: 32.3 cores
    monkeypatch.setattr(cpu_budget, "_CGROUP_V2", str(quota))
    monkeypatch.setattr(cpu_budget.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(cpu_budget.os, "sched_getaffinity", lambda _pid: set(range(96)))

    assert cpu_budget.cgroup_cpu_quota() == pytest.approx(32.3)
    assert cpu_budget.cpu_allowance() == pytest.approx(32.3)
    # 96 host processors is what torch would have used; the container has 32.3.
    assert cpu_budget.per_group_threads(32.3, 4) == 8
    assert cpu_budget.per_group_threads(32.3, 1) == 32
    # A fractional-core pod still gets a thread per group.
    assert cpu_budget.per_group_threads(0.5, 4) == 1


def test_unlimited_cgroup_falls_back_to_the_affinity_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000\n")
    monkeypatch.setattr(cpu_budget, "_CGROUP_V2", str(quota))
    monkeypatch.setattr(cpu_budget, "_CGROUP_V1_QUOTA", str(tmp_path / "absent"))
    monkeypatch.setattr(cpu_budget.os, "sched_getaffinity", lambda _pid: set(range(12)))
    monkeypatch.setattr(cpu_budget.os, "cpu_count", lambda: 96)

    assert cpu_budget.cgroup_cpu_quota() is None
    assert cpu_budget.cpu_allowance() == 12.0


def test_imposition_only_ever_de_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cpu_budget, "cpu_allowance", lambda: 32.0)

    torch.set_num_threads(16)
    facts = cpu_budget.impose_intra_op_threads(4)
    assert facts["budget"] == 8
    assert torch.get_num_threads() == 8 == facts["imposed"]

    # An allowance wider than torch's own pool leaves the pool alone: this is
    # a de-escalation, never a grant.
    torch.set_num_threads(4)
    facts = cpu_budget.impose_intra_op_threads(1)
    assert facts["imposed"] == 4
    assert torch.get_num_threads() == 4


def test_the_executor_imposes_the_budget_from_its_own_slot_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cpu_budget, "cpu_allowance", lambda: 32.0)
    torch.set_num_threads(32)

    async def _send(_msg: pb.WorkerMessage) -> None:
        return None

    topo = ExecutionTopology.from_env(
        {ENV_VAR: '{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}'})
    ex = Executor([], _send, store=ModelStore(_send, cache_dir=tmp_path / "cas"),
                  topology=topo)
    assert ex._gpu_slots == 4
    assert ex.cpu_budget["imposed"] == 8
    assert torch.get_num_threads() == 8


# ---------------------------------------------------------------------------
# D. the buffered-finalize bound is per group
# ---------------------------------------------------------------------------


def test_finalize_bound_is_two_per_group_and_single_group_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(video_encode.ENCODE_CONCURRENCY_ENV, raising=False)

    monkeypatch.delenv(ENV_VAR, raising=False)
    assert video_encode.finalize_concurrency() == 2

    monkeypatch.setenv(ENV_VAR, '{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}')
    assert video_encode.finalize_concurrency() == 8

    # A degree-4 sequence-parallel pod is ONE group: one request owns the pod,
    # so the bound stays where it was.
    monkeypatch.setenv(ENV_VAR, '{"gpu_count":4,"gpus_per_execution_group":4,"execution_groups":1}')
    assert video_encode.finalize_concurrency() == 2

    # An explicit operator budget still wins (gw#516 escape hatch).
    monkeypatch.setenv(video_encode.ENCODE_CONCURRENCY_ENV, "3")
    assert video_encode.finalize_concurrency() == 3


def test_the_semaphore_actually_admits_two_per_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(video_encode.ENCODE_CONCURRENCY_ENV, raising=False)
    monkeypatch.setenv(ENV_VAR, '{"gpu_count":4,"gpus_per_execution_group":1,"execution_groups":4}')
    video_encode._finalize_sem = None

    held: List[int] = []
    done = threading.Event()

    def _hold() -> None:
        with video_encode.finalize_permit():
            held.append(1)
            done.wait(5.0)

    workers = [threading.Thread(target=_hold, daemon=True) for _ in range(8)]
    for w in workers:
        w.start()
    # pgw#795: holders arriving is the progress signal; the give-up is "no
    # holder has arrived in a staleness window", not a 5s budget for eight
    # thread starts on a shared runner.
    try:
        await_count(
            lambda: len(held), 8,
            what="permit holders (2 per group at G=4)",
            cadence=Cadence(),
            gone=lambda: (
                None if any(w.is_alive() for w in workers)
                else "every holder thread exited"
            ),
        )
        assert len(held) == 8, "8 = 2 per group at G=4; a flat 2 would block 6"
    finally:
        done.set()
        for w in workers:
            w.join(timeout=5.0)
