"""pgw#778: never ADVERTISE what dispatch will refuse.

`ctx.device` resolves from the calling thread's current CUDA device, which for
an ASYNC handler is the shared event loop's — not the job's group. That is
refused on a multi-group pod, correctly. What was wrong is WHERE: the refusal
fired at dispatch with `JOB_STATUS_INVALID` (the status that means the CALLER's
input was bad, so the hub neither retried elsewhere nor charged the worker)
while the function stayed in `available_functions`. The hub kept routing, every
request came back INVALID, and a wide pod with any async endpoint (every
LLM/vllm-shaped one) became a 100%-INVALID black hole blamed on the caller.

Layer exercised: `gate_functions` -> `available_functions`, i.e. the same seam
every other hardware refusal already uses, so the hub re-packs or routes
elsewhere. The dispatch-time check survives as belt-and-braces and now returns
RETRYABLE.
"""

from __future__ import annotations

from typing import Any, List

import msgspec
import pytest

from gen_worker.hostfacts import HostFacts
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker import Hub, Resources
from gen_worker.executor import EndpointSpec, Executor
from gen_worker.topology import ExecutionTopology


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline: Any) -> None:  # pragma: no cover
        self.pipeline = pipeline

    def sync_gen(self, ctx: Any, payload: _In) -> dict:  # pragma: no cover
        return {}

    async def async_gen(self, ctx: Any, payload: _In) -> dict:  # pragma: no cover
        return {}


def _specs() -> List[EndpointSpec]:
    common = dict(
        kind="inference", payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/sdxl")}, resources=Resources(gpu=True),
        slots=("pipeline",),
    )
    return [
        EndpointSpec(name="sync_fn", method=_Fake.sync_gen, **common),
        EndpointSpec(
            name="async_fn", method=_Fake.async_gen, is_async=True, **common),
    ]


def _executor(topology: ExecutionTopology) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:  # pragma: no cover
        pass

    return Executor(_specs(), _send, topology=topology)


_GPU = HostFacts(vram_total_bytes=40 * 1024**3, vram_free_bytes=40 * 1024**3,
                 gpu_sm="89")


def test_a_wide_pod_withdraws_the_async_function_instead_of_advertising_it() -> None:
    ex = _executor(ExecutionTopology(gpu_count=4, gpus_per_execution_group=1))
    assert ex.topology.execution_groups == 4
    ex.gate_functions(_GPU)

    code, detail, axes = ex.unavailable["async_fn"]
    assert code == "multi_group_async_handler"
    assert "async handlers are not yet served" in detail
    assert axes["execution_groups"] == "4"
    assert "async_fn" not in ex.available_functions(), (
        "the hub reads available_functions; advertising a function every "
        "dispatch refuses is the pgw#778 black hole")
    # The sync sibling is untouched: sync handlers run on a thread that DID
    # set_device to the group's rank-0 card.
    assert "sync_fn" not in ex.unavailable
    assert "sync_fn" in ex.available_functions()


def test_a_sequence_parallel_pod_withdraws_it_too() -> None:
    ex = _executor(ExecutionTopology(
        gpu_count=2, gpus_per_execution_group=2, parallel="sequence"))
    assert (ex.topology.execution_groups, ex.topology.degree) == (1, 2)
    ex.gate_functions(_GPU)
    assert ex.unavailable["async_fn"][0] == "multi_group_async_handler"
    assert "sync_fn" in ex.available_functions()


def test_a_single_group_pod_advertises_both() -> None:
    # G=1/D=1 — every pod today. Nothing here may move.
    ex = _executor(ExecutionTopology.single())
    ex.gate_functions(_GPU)
    assert ex.unavailable == {}
    assert set(ex.available_functions()) == {"sync_fn", "async_fn"}


def test_the_gate_mark_is_gate_owned_and_re_derived() -> None:
    ex = _executor(ExecutionTopology(gpu_count=4, gpus_per_execution_group=1))
    ex.gate_functions(_GPU)
    ex.gate_functions(_GPU)
    assert ex.unavailable["async_fn"][0] == "multi_group_async_handler"


def test_the_dispatch_refusal_is_retryable_not_invalid() -> None:
    # Belt-and-braces: a dispatch that raced the withdrawal must not blame the
    # caller. RETRYABLE lets the hub replan it onto a pod that can serve it.
    import asyncio

    ex = _executor(ExecutionTopology(gpu_count=4, gpus_per_execution_group=1))
    spec = ex.specs["async_fn"]
    finished: dict = {}

    async def _fake_finish(job: Any, status: Any, **kw: Any) -> None:
        finished["status"] = status
        finished["msg"] = kw.get("safe_message")

    ex._finish = _fake_finish  # type: ignore[method-assign]

    class _Job:
        def __init__(self) -> None:
            self.spec = spec

    asyncio.run(ex._run_job_grouped(_Job(), pb.RunJob()))  # type: ignore[arg-type]
    assert finished["status"] == pb.JOB_STATUS_RETRYABLE, finished
    assert "async handlers are not yet served" in finished["msg"]
    # And the refusal marks the function so the next state delta withdraws it.
    assert ex.unavailable["async_fn"][0] == "multi_group_async_handler"


@pytest.mark.parametrize("groups,degree", [(4, 1), (1, 2)])
def test_serve_plans_are_not_polluted_for_a_withdrawn_function(
    groups: int, degree: int,
) -> None:
    # The withdrawal `continue`s before plan_serve, so a function the pod will
    # never run does not publish a fit plan the hub could act on.
    ex = _executor(ExecutionTopology(
        gpu_count=groups * degree, gpus_per_execution_group=degree,
        parallel="sequence" if degree > 1 else ""))
    ex.gate_functions(_GPU)
    assert "async_fn" not in ex.serve_plans
    assert "sync_fn" in ex.serve_plans
