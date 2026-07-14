"""Declarative model residency: full replace, exact hot picks, and observation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import msgspec

from gen_worker.api.binding import Civitai, wire_ref
from gen_worker.api.slot import Slot
from gen_worker.executor import Executor
from gen_worker.families.base import FamilyDefaults, family
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


@family("declarative-residency-test")
class _Defaults(FamilyDefaults):
    steps: int = 1


class _Pipeline:
    pass


class _Input(msgspec.Struct):
    prompt: str = ""


def _spec() -> EndpointSpec:
    default = Civitai("827184", version="2883731")

    class Endpoint:
        def setup(self, pipeline: str) -> None:  # pragma: no cover - replaced in tests
            self.pipeline = pipeline

        def generate(self, ctx: Any, payload: _Input) -> dict:  # pragma: no cover
            return {}

    return EndpointSpec(
        name="generate",
        method=Endpoint.generate,
        kind="inference",
        payload_type=_Input,
        output_mode="single",
        cls=Endpoint,
        attr_name="generate",
        models={"pipeline": default},
        slots={
            "pipeline": Slot(
                _Pipeline,
                default_checkpoint=default,
                default_config=_Defaults(),
            )
        },
        slot_family={"pipeline": "declarative-residency-test"},
    )


async def _noop_send(msg: pb.WorkerMessage) -> None:
    pass


class _Transport:
    def __init__(self) -> None:
        self.connected = True
        self.sent: list[pb.WorkerMessage] = []
        self.queue = SimpleNamespace(pending_result_keys=set())

    async def send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)


def _lifecycle() -> tuple[Lifecycle, Executor, _Transport]:
    executor = Executor([_spec()], _noop_send)
    lifecycle = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="worker", runpod_pod_id=""),
        executor,
    )
    transport = _Transport()
    lifecycle.transport = transport  # type: ignore[assignment]
    return lifecycle, executor, transport


def _snapshot(url: str) -> pb.Snapshot:
    return pb.Snapshot(
        digest="ab" * 32,
        files=[pb.SnapshotFile(
            path="model.safetensors",
            size_bytes=1,
            blake3="cd" * 32,
            url=url,
        )],
    )


def test_proto_field_numbers_match_tensorhub_contract() -> None:
    assert pb.PROTOCOL_VERSION_CURRENT == 3
    assert pb.HelloAck.DESIRED_RESIDENCY_FIELD_NUMBER == 5
    assert pb.DesiredResidency.GENERATION_FIELD_NUMBER == 1
    assert pb.DesiredResidency.DISK_REFS_FIELD_NUMBER == 2
    assert pb.DesiredResidency.HOT_FIELD_NUMBER == 3
    assert pb.DesiredResidency.SNAPSHOTS_FIELD_NUMBER == 4
    assert pb.DesiredInstance.FUNCTION_NAME_FIELD_NUMBER == 1
    assert pb.DesiredInstance.MODELS_FIELD_NUMBER == 2
    assert pb.StateDelta.OBSERVED_RESIDENCY_GENERATION_FIELD_NUMBER == 6
    assert not hasattr(pb, "MODEL_OP_KIND_DOWNLOAD")
    assert not hasattr(pb, "MODEL_OP_KIND_LOAD")
    assert not hasattr(pb, "MODEL_OP_KIND_UNLOAD")


def test_full_replace_supersedes_and_same_generation_refreshes_urls(monkeypatch) -> None:
    async def run() -> None:
        lifecycle, executor, transport = _lifecycle()
        calls: list[tuple[str, str]] = []
        old_started = asyncio.Event()
        refreshed = asyncio.Event()

        async def ensure_local(ref: str, snapshot=None, *, binding=None):
            url = snapshot.files[0].url if snapshot and snapshot.files else ""
            calls.append((ref, url))
            if ref == "acme/old":
                old_started.set()
                await asyncio.Event().wait()
            if url == "https://r2/new-2":
                refreshed.set()

        monkeypatch.setattr(executor.store, "ensure_local", ensure_local)

        # Tenant work owns the lane: reconciliation does not start while the
        # executor is busy.
        executor._idle.clear()
        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=1,
            disk_refs=["acme/old"],
            snapshots={"acme/old": _snapshot("https://r2/old")},
        )))
        await asyncio.sleep(0)
        assert calls == []
        assert transport.sent[-1].state_delta.observed_residency_generation == 1

        executor._idle.set()
        await old_started.wait()

        # A newer full replacement cancels the obsolete blocked reconcile.
        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=2,
            disk_refs=["acme/new"],
            snapshots={"acme/new": _snapshot("https://r2/new-1")},
        )))
        if lifecycle._residency_task is not None:
            await lifecycle._residency_task

        # Same generation is not stale: it may carry refreshed presigned URLs.
        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=2,
            disk_refs=["acme/new"],
            snapshots={"acme/new": _snapshot("https://r2/new-2")},
        )))
        await refreshed.wait()

        # An older generation cannot replace keep state or regress observation.
        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=1,
            disk_refs=["acme/stale"],
            snapshots={"acme/stale": _snapshot("https://r2/stale")},
        )))
        assert executor.store.keep == {"acme/new"}
        assert lifecycle._state_delta().observed_residency_generation == 2
        assert calls == [
            ("acme/old", "https://r2/old"),
            ("acme/new", "https://r2/new-1"),
            ("acme/new", "https://r2/new-2"),
        ]

    asyncio.run(run())


def test_hot_instance_uses_exact_dynamic_slot_binding(monkeypatch) -> None:
    async def run() -> None:
        executor = Executor([_spec()], _noop_send)
        base = executor.specs["generate"]
        captured: list[tuple[EndpointSpec, dict[str, pb.Snapshot]]] = []

        async def ensure_setup(spec, snapshots=None, promote_slots=None):
            captured.append((spec, snapshots or {}))

        monkeypatch.setattr(executor, "ensure_setup", ensure_setup)
        picked = "tensorhub/cyberrealistic-pony:prod"
        snapshot = _snapshot("https://r2/picked")
        await executor.ensure_desired_instance(
            pb.DesiredInstance(
                function_name="generate",
                models=[pb.ModelBinding(slot="pipeline", ref=picked)],
            ),
            {picked: snapshot},
        )

        assert len(captured) == 1
        effective, snapshots = captured[0]
        assert effective.instance_key != base.instance_key
        assert wire_ref(effective.models["pipeline"]) == picked
        assert snapshots[picked].files[0].url == "https://r2/picked"

    asyncio.run(run())


def test_run_job_preempts_then_resumes_current_desired_state(monkeypatch) -> None:
    async def run() -> None:
        lifecycle, executor, _ = _lifecycle()
        started = asyncio.Event()
        canceled = asyncio.Event()
        resumed = asyncio.Event()
        calls = 0

        async def ensure_local(ref: str, snapshot=None, *, binding=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    canceled.set()
                    raise
            resumed.set()

        async def handle_run_job(run_job: pb.RunJob) -> None:
            executor._idle.clear()
            await asyncio.sleep(0)
            assert canceled.is_set(), "background work must cancel before request setup"

        monkeypatch.setattr(executor.store, "ensure_local", ensure_local)
        monkeypatch.setattr(executor, "handle_run_job", handle_run_job)

        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=7,
            disk_refs=["acme/background"],
            snapshots={"acme/background": _snapshot("https://r2/background")},
        )))
        await started.wait()

        await lifecycle.on_message(pb.SchedulerMessage(run_job=pb.RunJob(
            request_id="request", attempt=1, function_name="generate",
        )))
        await asyncio.sleep(0)
        assert calls == 1, "background work must stay paused while a job is active"

        executor._idle.set()
        await resumed.wait()
        assert calls == 2, "current desired state must resume after the job becomes idle"

    asyncio.run(run())
