from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

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


def test_config_classes_converge_separately_and_boot_change_stays_stale() -> None:
    registry = IntentRegistry("release-1", ["echo"], boot_config_generation=1)

    def apply(
        seq: int,
        generation: int,
        changed: pb.ConfigClassMask | None = None,
    ) -> None:
        command = pb.DesiredStateCommand(
            worker_session_id=registry.worker_session_id,
            command_seq=seq,
            goal_id=f"goal-{seq}",
            release_id="release-1",
            config_generation=generation,
            config_digest=f"digest-{generation}".encode(),
            parameter_snapshot=msgspec.msgpack.encode({}),
            first_action_by_unix_ms=9_000_000_000_000,
            intents=[
                pb.DesiredIntent(
                    intent_id=f"config-{generation}",
                    kind=pb.DESIRED_INTENT_KIND_CONFIG_APPLY,
                    cause=pb.DESIRED_INTENT_CAUSE_CONFIG_CHANGE,
                    mandatory=True,
                )
            ],
            mandatory=True,
        )
        if changed is not None:
            command.changed_config_classes.CopyFrom(changed)
        receipt = registry.apply_command(command)
        assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    apply(1, 1)
    registry.config_snapshot_applied(1)
    registry.bindings_applied(1)
    first = registry.snapshot().config_application
    assert first.state == pb.CONFIG_APPLICATION_STATE_CONVERGED
    assert (
        first.received_generation,
        first.parameter_snapshot_generation,
        first.binding_ready_generation,
        first.boot_generation,
    ) == (1, 1, 1, 1)
    runtime_config = SimpleNamespace(
        generation=1,
        parameter_snapshot_generation=1,
    )
    executor = SimpleNamespace(
        runtime_config=runtime_config,
        store=SimpleNamespace(residency_snapshot=lambda: []),
        available_functions=lambda: ["echo"],
        compile_targets=lambda: [],
        unavailable={},
    )
    desired = pb.DesiredResidency(
        hot=[pb.DesiredInstance(function_name="echo")],
    )
    registry.refresh_projection(executor, desired, {})
    capability = registry.snapshot().capabilities[0]
    assert capability.function_name == "echo"
    assert capability.release_id == "release-1"
    assert capability.config_generation == 1
    assert capability.binding_digest
    assert capability.state == pb.FUNCTION_CAPABILITY_STATE_READY

    apply(2, 2, pb.ConfigClassMask(parameters=True))
    applying = registry.snapshot().config_application
    assert applying.state == pb.CONFIG_APPLICATION_STATE_APPLYING
    assert applying.pending_classes.parameters
    assert not applying.pending_classes.bindings
    assert not applying.pending_classes.boot
    registry.config_snapshot_applied(2)
    assert registry.snapshot().config_application.state == pb.CONFIG_APPLICATION_STATE_CONVERGED

    apply(3, 3, pb.ConfigClassMask(boot=True))
    registry.config_snapshot_applied(3)
    stale = registry.snapshot()
    assert stale.config_application.state == pb.CONFIG_APPLICATION_STATE_BOOT_STALE
    assert stale.config_application.boot_generation == 2
    assert stale.config_application.pending_classes.boot
    config_state = _state(registry, "config-3")
    assert config_state.status == pb.LIFECYCLE_INTENT_STATUS_WAITING
    assert config_state.stage == pb.LIFECYCLE_INTENT_STAGE_WAIT_REPLACEMENT
    assert config_state.reason == pb.LIFECYCLE_WAIT_REASON_REPLACEMENT
    assert config_state.deadline_at_unix_ms == 9_000_000_000_000
    runtime_config.generation = 3
    runtime_config.parameter_snapshot_generation = 3
    registry.refresh_projection(executor, desired, {})
    assert registry.snapshot().capabilities[0].state == pb.FUNCTION_CAPABILITY_STATE_BOOT_STALE


def test_first_command_cannot_infer_an_unstamped_boot_generation() -> None:
    registry = IntentRegistry("release-1", ["echo"])
    receipt = registry.apply_command(
        pb.DesiredStateCommand(
            worker_session_id=registry.worker_session_id,
            command_seq=1,
            goal_id="goal-1",
            release_id="release-1",
            config_generation=4,
            config_digest=b"digest-4",
            parameter_snapshot=msgspec.msgpack.encode({}),
            first_action_by_unix_ms=9_000_000_000_000,
            intents=[
                pb.DesiredIntent(
                    intent_id="config-4",
                    kind=pb.DESIRED_INTENT_KIND_CONFIG_APPLY,
                    cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
                    mandatory=True,
                )
            ],
            mandatory=True,
        )
    )
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
    registry.config_snapshot_applied(4)
    registry.bindings_applied(4)
    application = registry.snapshot().config_application
    assert application.boot_generation == 0
    assert application.state == pb.CONFIG_APPLICATION_STATE_BOOT_STALE


def test_config_command_without_parameter_snapshot_is_rejected() -> None:
    registry = IntentRegistry("release-1", ["echo"])
    receipt = registry.apply_command(
        pb.DesiredStateCommand(
            worker_session_id=registry.worker_session_id,
            command_seq=1,
            goal_id="goal-1",
            release_id="release-1",
            config_generation=1,
            config_digest=b"digest-1",
            mandatory=True,
        )
    )
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_REJECTED
    assert receipt.error_code == pb.LIFECYCLE_ERROR_CODE_MISSING_MANDATORY_FIELD
    assert "parameter_snapshot" in receipt.detail
    assert registry.protocol_rejected
