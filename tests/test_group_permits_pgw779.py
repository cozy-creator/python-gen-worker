"""pgw#779: a slot is a GROUP's permit, not one of G interchangeable tickets.

`asyncio.Semaphore(G)` admits G concurrent jobs but binds none of them to a
group. The group came solely from `ResolvedCompute.gpu_index`, and a missing
`compute` floored to group 0 while an index that was not a rank-0 device was
floored to a real group with a `logger.warning`. So four dispatches naming card
0 all took a permit, all resolved to ONE record, and serialized on one
`run_lock` while three cards idled — and the pod reported four healthy slots.
The width-4 campaign never saw it because the harness assigned `gpu_index=g`
itself: the acceptance proved the worker CAN spread, not that it MUST.

Layers exercised:

- `Executor` permit construction: G independent permits, and `gpu_slots=`
  (the `cli serve`/test override) still means the concurrency it always meant.
- `_gpu_permit_for_group` / `_gpu_permit_for_record`: a job's permit and its
  card are the same fact; a background turn on group 2 cannot consume group 0's.
- `_dispatch_group`: a missing `compute` and an off-by-D `gpu_index` are typed
  refusals on a wide pod, and unchanged (group 0) on a single-group one.
- `_run_job`: the refusal ends the job with a terminal RETRYABLE state rather
  than raising out of the task.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import msgspec
import pytest

from gen_worker.api import Hub, Resources
from gen_worker.executor import (
    DispatchGroupUnresolved,
    EndpointSpec,
    Executor,
    _ClassRecord,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.topology import ExecutionTopology, TopologyError


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline: Any) -> None:  # pragma: no cover
        self.pipeline = pipeline

    def gen(self, ctx: Any, payload: _In) -> dict:  # pragma: no cover
        return {}


def _specs() -> List[EndpointSpec]:
    return [EndpointSpec(
        name="gen", method=_Fake.gen, kind="inference", payload_type=_In,
        output_mode="single", cls=_Fake, slots=("pipeline",),
        models={"pipeline": Hub("acme/sdxl")}, resources=Resources(gpu=True),
    )]


def _executor(topology: ExecutionTopology, gpu_slots: Any = None) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:  # pragma: no cover
        pass

    return Executor(_specs(), _send, topology=topology, gpu_slots=gpu_slots)


def test_a_wide_pod_has_one_permit_per_group_not_a_count() -> None:
    ex = _executor(ExecutionTopology(gpu_count=4, group_degree=1))
    assert len(ex._gpu_permits) == 4
    assert ex._gpu_permits_each == 1
    # Four distinct objects: taking group 0's permit cannot admit a job to
    # group 1, and taking all four of group 0's "tickets" is not expressible.
    assert len({id(p) for p in ex._gpu_permits}) == 4

    async def _drive() -> None:
        await ex._gpu_permit_for_group(0).acquire()
        # Group 0 is now busy; every other group is free.
        assert ex._gpu_permit_for_group(0).locked()
        for g in (1, 2, 3):
            assert not ex._gpu_permit_for_group(g).locked()
            await ex._gpu_permit_for_group(g).acquire()
        # A second job on group 0 must WAIT, not run on a sibling's card.
        second = asyncio.ensure_future(ex._gpu_permit_for_group(0).acquire())
        await asyncio.sleep(0.05)
        assert not second.done(), (
            "a count semaphore let a second same-group job run concurrently "
            "on one card")
        ex._gpu_permit_for_group(0).release()
        await asyncio.wait_for(second, 1)

    asyncio.run(_drive())


def test_a_single_group_pod_is_todays_pool_exactly() -> None:
    ex = _executor(ExecutionTopology.single())
    assert len(ex._gpu_permits) == 1
    assert ex._gpu_semaphore is ex._gpu_permits[0]
    assert ex._gpu_permits_each == 1


def test_an_explicit_gpu_slots_override_keeps_its_concurrency() -> None:
    # `gpu_slots=` is the `cli serve`/test override, never a production knob.
    # Its concurrency is divided among the groups, so at G==1 group 0's permit
    # IS the whole pool — the shape `tests/test_executor_adopt.py` relies on.
    ex = _executor(ExecutionTopology.single(), gpu_slots=2)
    assert len(ex._gpu_permits) == 1

    async def _drive() -> None:
        await ex._gpu_semaphore.acquire()
        await ex._gpu_semaphore.acquire()
        assert ex._gpu_semaphore.locked()

    asyncio.run(_drive())


def test_a_records_permit_is_its_own_groups() -> None:
    ex = _executor(ExecutionTopology(gpu_count=4, group_degree=1))
    spec = ex.specs["gen"]
    for g in range(4):
        rec = _ClassRecord(cls=_Fake)
        rec.specs = [ex._group_effective_spec(spec, g)]
        assert ex._gpu_permit_for_record(rec) is ex._gpu_permits[g], (
            "a background mint turn on group %d must not consume another "
            "group's slot" % g)


# ---------------------------------------------------------------------------
# gpu_index is the hub's ANSWER, and a wrong one is refused, not floored.
# ---------------------------------------------------------------------------


def _run(gpu_index: Any = None) -> pb.RunJob:
    run = pb.RunJob(request_id="r1", attempt=1)
    if gpu_index is not None:
        run.compute.gpu_index = int(gpu_index)
    return run


def test_a_missing_compute_is_refused_on_a_wide_pod() -> None:
    ex = _executor(ExecutionTopology(gpu_count=4, group_degree=1))
    with pytest.raises(DispatchGroupUnresolved, match="no resolved compute"):
        ex._dispatch_group(_run())


def test_an_off_by_degree_index_is_refused_not_floored() -> None:
    # 4 cards packed 2x2: the legal rank-0 devices are 0 and 2. gpu_index=1 is
    # a hub/worker disagreement about D — it used to be floored to group 0
    # with a warning, silently packing two jobs onto one group.
    topo = ExecutionTopology(gpu_count=4, group_degree=2, parallel="sequence")
    assert topo.groups == 2
    ex = _executor(topo)
    with pytest.raises(DispatchGroupUnresolved, match="not a rank-0 device"):
        ex._dispatch_group(_run(1))
    with pytest.raises(DispatchGroupUnresolved, match="not a rank-0 device"):
        ex._dispatch_group(_run(9))
    # The legal ones still resolve.
    assert ex._dispatch_group(_run(0)) == 0
    assert ex._dispatch_group(_run(2)) == 1


def test_group_ordinal_exact_refuses_where_group_ordinal_floors() -> None:
    topo = ExecutionTopology(gpu_count=4, group_degree=2, parallel="sequence")
    assert topo.group_ordinal(1) == 0          # the historical floor
    with pytest.raises(TopologyError, match="not a rank-0 device"):
        topo.group_ordinal_exact(1)


def test_a_single_group_pod_still_serves_a_computeless_dispatch() -> None:
    # Every CPU function and every pre-topology hub. Nothing here may move.
    ex = _executor(ExecutionTopology.single())
    assert ex._dispatch_group(_run()) == 0
    assert ex._dispatch_group(_run(0)) == 0
    # Even a nonsense index: there is exactly one group to mean.
    assert ex._dispatch_group(_run(7)) == 0


def test_the_refusal_ends_the_job_terminally() -> None:
    ex = _executor(ExecutionTopology(gpu_count=4, group_degree=1))
    finished: dict = {}

    async def _fake_finish(job: Any, status: Any, **kw: Any) -> None:
        finished["status"] = status
        finished["msg"] = kw.get("safe_message")

    ex._finish = _fake_finish  # type: ignore[method-assign]

    class _Job:
        spec = None

    job = _Job()
    job.spec = ex.specs["gen"]  # type: ignore[attr-defined]
    asyncio.run(ex._run_job(job, _run()))  # type: ignore[arg-type]
    assert finished["status"] == pb.JOB_STATUS_RETRYABLE, finished
    assert "no resolved compute" in finished["msg"]
