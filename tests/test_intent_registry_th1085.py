from __future__ import annotations

import asyncio
from pathlib import Path

import msgspec
import pytest

from gen_worker.executor import Executor, ModelStore
from gen_worker.intent_registry import IntentRegistry, UnreportedIntentWait
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


async def _noop_send(_message: pb.WorkerMessage) -> None:
    return None


def _state(registry: IntentRegistry, intent_id: str) -> pb.IntentState:
    return next(item for item in registry.snapshot().intents if item.intent_id == intent_id)


def test_long_unreported_await_fails_closed_and_replays_typed_failure() -> None:
    registry = IntentRegistry(
        "release-1",
        ["echo"],
        unreported_wait_timeout_s=0.01,
    )
    receipt = registry.apply_command(
        pb.DesiredStateCommand(
            worker_session_id=registry.worker_session_id,
            command_seq=1,
            goal_id="goal-1",
            release_id="release-1",
            intents=[
                pb.DesiredIntent(
                    intent_id="materialize-1",
                    kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
                    cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                    ref="owner/model:latest",
                    snapshot_digest=b"blake3:abc",
                    mandatory=True,
                )
            ],
            mandatory=True,
        )
    )
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    async def run() -> None:
        never = asyncio.Event()
        with pytest.raises(UnreportedIntentWait):
            await registry.guard_await(
                "materialize-1",
                never.wait(),
                operation="test materialization",
            )

    asyncio.run(run())

    state = _state(registry, "materialize-1")
    assert registry.protocol_rejected
    assert state.status == pb.LIFECYCLE_INTENT_STATUS_FAILED
    assert state.error_code == pb.LIFECYCLE_ERROR_CODE_PROTOCOL_UNREPORTED_WAIT


def test_materialization_waiter_names_the_active_ref_owner(tmp_path: Path) -> None:
    async def run() -> None:
        registry = IntentRegistry("release-1", [])
        store = ModelStore(_noop_send, cache_dir=tmp_path)
        store.bind_intent_registry(registry)
        ref = "owner/model:latest"
        lock = store._lock(ref)
        await lock.acquire()
        owner = asyncio.create_task(store.ensure_local(ref))
        await asyncio.sleep(0)
        owner_intent = store._materialize_active[ref]
        waiter = asyncio.create_task(store.ensure_local(ref))
        await asyncio.sleep(0)
        states = registry.snapshot().intents
        follower = next(item for item in states if item.blocker_intent_id == owner_intent)
        assert follower.status == pb.LIFECYCLE_INTENT_STATUS_WAITING
        assert follower.stage == pb.LIFECYCLE_INTENT_STAGE_WAIT_REF_LOCK
        assert follower.reason == pb.LIFECYCLE_WAIT_REASON_REF_LOCK
        owner.cancel()
        waiter.cancel()
        await asyncio.gather(owner, waiter, return_exceptions=True)
        assert _state(registry, owner_intent).status == pb.LIFECYCLE_INTENT_STATUS_CANCELED
        assert _state(registry, follower.intent_id).status == pb.LIFECYCLE_INTENT_STATUS_CANCELED
        lock.release()

    asyncio.run(run())


def test_setup_and_gpu_waits_are_typed_before_blocking() -> None:
    class Input(msgspec.Struct):
        value: int

    class Output(msgspec.Struct):
        value: int

    class Endpoint:
        def run(self, _ctx: object, payload: Input) -> Output:
            return Output(payload.value)

    spec = EndpointSpec(
        name="echo",
        method=Endpoint.run,
        kind="inference",
        payload_type=Input,
        output_type=Output,
        output_mode="single",
        cls=Endpoint,
        attr_name="run",
    )

    async def run() -> None:
        executor = Executor([spec], _noop_send)
        registry = IntentRegistry("release-1", ["echo"])
        executor.bind_intent_registry(registry)
        rec = executor._class_record(spec)
        await rec.lock.acquire()
        setup = asyncio.create_task(executor.ensure_setup(spec))
        await asyncio.sleep(0)
        setup_intent = executor._setup_intent(spec)
        setup_state = _state(registry, setup_intent)
        assert setup_state.status == pb.LIFECYCLE_INTENT_STATUS_WAITING
        assert setup_state.stage == pb.LIFECYCLE_INTENT_STAGE_WAIT_LOAD_LOCK
        assert setup_state.reason == pb.LIFECYCLE_WAIT_REASON_SINGLE_FLIGHT_OWNER
        setup.cancel()
        await asyncio.gather(setup, return_exceptions=True)
        rec.lock.release()

        gpu_intent = registry.ensure_local_intent("gpu-test", "one")
        await executor._gpu_semaphore.acquire()

        async def hold_gpu() -> None:
            async with executor._exclusive_gpu(gpu_intent):
                pass

        gpu = asyncio.create_task(hold_gpu())
        await asyncio.sleep(0)
        gpu_state = _state(registry, gpu_intent)
        assert gpu_state.status == pb.LIFECYCLE_INTENT_STATUS_WAITING
        assert gpu_state.stage == pb.LIFECYCLE_INTENT_STAGE_WAIT_GPU_SLOT
        assert gpu_state.reason == pb.LIFECYCLE_WAIT_REASON_GPU_SLOT
        gpu.cancel()
        await asyncio.gather(gpu, return_exceptions=True)
        assert _state(registry, gpu_intent).status == pb.LIFECYCLE_INTENT_STATUS_CANCELED
        executor._gpu_semaphore.release()

    asyncio.run(run())


def test_legacy_adoption_and_job_have_bounded_local_intents() -> None:
    class Input(msgspec.Struct):
        value: int

    class Output(msgspec.Struct):
        value: int

    def handler(ctx: object, payload: Input) -> Output:
        del ctx
        return Output(payload.value)

    async def run() -> None:
        sent: list[pb.WorkerMessage] = []

        async def send(message: pb.WorkerMessage) -> None:
            sent.append(message)

        spec = EndpointSpec(
            name="echo",
            method=handler,
            kind="inference",
            payload_type=Input,
            output_type=Output,
            output_mode="single",
        )
        executor = Executor([spec], send)
        registry = IntentRegistry("release-1", ["echo"])
        executor.bind_intent_registry(registry)

        adoption = pb.ModelOp(
            op=pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE,
            ref="compile-cache/family/cell",
        )
        adoption_intent = executor._adoption_intent(adoption)
        await executor.handle_model_op(adoption)
        adoption_state = _state(registry, adoption_intent)
        assert adoption_state.status == pb.LIFECYCLE_INTENT_STATUS_FAILED
        assert adoption_state.stage == pb.LIFECYCLE_INTENT_STAGE_VALIDATING

        request = pb.RunJob(
            request_id="request-1",
            attempt=1,
            function_name="echo",
            input_payload=msgspec.msgpack.encode(Input(7)),
        )
        await executor.handle_run_job(request)
        job = executor.jobs[("request-1", 1)]
        assert job.task is not None
        await job.task
        job_state = _state(registry, job.intent_id)
        assert job_state.status == pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED
        assert job_state.stage == pb.LIFECYCLE_INTENT_STAGE_READY
        assert not job_state.HasField("blocker_request")

    asyncio.run(run())
