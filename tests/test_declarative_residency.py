"""Declarative model residency: full replace, exact hot picks, and observation."""

from __future__ import annotations

import asyncio
import re
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec

import gen_worker.executor as executor_mod
from gen_worker.api.binding import Civitai, wire_ref
from gen_worker.api.slot import Slot
from gen_worker.executor import Executor
from gen_worker.families.base import FamilyDefaults, family
from gen_worker.lifecycle import Lifecycle
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


@family("declarative-residency-test")
class _Defaults(FamilyDefaults):
    steps: int = 1


class _Pipeline:
    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "_Pipeline":
        return cls()


class _Input(msgspec.Struct):
    prompt: str = ""


def _spec() -> EndpointSpec:
    default = Civitai("827184", version="2883731")

    class Endpoint:
        def setup(self, pipeline: _Pipeline) -> None:  # pragma: no cover - replaced in tests
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
    assert pb.StateDelta.COMPILE_TARGETS_FIELD_NUMBER == 7
    assert pb.CompileTarget.INCARNATION_ID_FIELD_NUMBER == 1
    assert pb.CompileTarget.FAMILY_FIELD_NUMBER == 2
    assert pb.CompileTarget.PIPELINE_WEIGHT_LANE_FIELD_NUMBER == 3
    assert pb.CompileTarget.LORA_BUCKET_FIELD_NUMBER == 4
    assert pb.CompileTarget.CONTRACT_DIGEST_FIELD_NUMBER == 5
    assert pb.CompileTarget.ACTIVE_COMPILE_REF_FIELD_NUMBER == 6
    assert pb.CompileTarget.ACTIVE_COMPILE_SNAPSHOT_DIGEST_FIELD_NUMBER == 7
    assert pb.CompileTarget.FUNCTION_NAMES_FIELD_NUMBER == 8
    assert pb.CompileTarget.MODEL_BINDINGS_FIELD_NUMBER == 9
    assert pb.CompileTargetBinding.SLOT_FIELD_NUMBER == 1
    assert pb.CompileTargetBinding.REF_FIELD_NUMBER == 2
    assert pb.CompileTargetBinding.SNAPSHOT_DIGEST_FIELD_NUMBER == 3
    assert pb.ModelOp.TARGET_INCARNATION_ID_FIELD_NUMBER == 5
    assert pb.ModelEvent.TARGET_INCARNATION_ID_FIELD_NUMBER == 19
    assert pb.RunJob.REQUIRED_COMPILE_FIELD_NUMBER == 13
    assert pb.RequiredCompileExecution.TARGET_INCARNATION_ID_FIELD_NUMBER == 1
    assert pb.RequiredCompileExecution.CELL_REF_FIELD_NUMBER == 2
    assert pb.RequiredCompileExecution.CELL_SNAPSHOT_DIGEST_FIELD_NUMBER == 3
    assert pb.RequiredCompileExecution.CONTRACT_DIGEST_FIELD_NUMBER == 4
    assert not hasattr(pb, "MODEL_OP_KIND_DOWNLOAD")
    assert not hasattr(pb, "MODEL_OP_KIND_LOAD")
    assert not hasattr(pb, "MODEL_OP_KIND_UNLOAD")


def test_reconnect_snapshot_uses_disk_identity_before_transition_callback(
    tmp_path: Path, monkeypatch,
) -> None:
    _, executor, _ = _lifecycle()
    ref = "acme/moved-tag"
    old_ram = ("snapshot-a", 6)
    new_disk = ("snapshot-b", 7)

    executor.store.residency.track_disk(ref, tmp_path)
    executor.store.residency.track_ram(ref, object())
    with executor.store._identity_lock:
        executor.store._resident_identities[ref] = old_ram
        executor.store._disk_identities[ref] = new_disk

    callback_started = threading.Event()
    allow_callback = threading.Event()
    original_event = executor.store.residency._on_event

    def pause_disk_callback(*args) -> None:
        callback_started.set()
        assert allow_callback.wait(2)
        original_event(*args)

    monkeypatch.setattr(executor.store.residency, "_on_event", pause_disk_callback)
    monkeypatch.setattr("gen_worker.models.residency.flush_memory", lambda: None)
    transition = threading.Thread(
        target=executor.store.residency.release_to_disk,
        args=(ref,),
    )
    transition.start()
    assert callback_started.wait(2)

    # Residency already exposes DISK, while the synchronous callback has not
    # yet replaced resident A with disk B. Hello must still report DISK+B.
    disk = executor.store.residency_snapshot()[0]
    assert disk.tier == pb.RESIDENCY_TIER_DISK
    assert (disk.snapshot_digest, disk.residency_generation) == new_disk
    allow_callback.set()
    transition.join(2)
    assert not transition.is_alive()


def test_reconnect_snapshot_holds_identity_across_tier_capture(
    tmp_path: Path, monkeypatch,
) -> None:
    _, executor, _ = _lifecycle()
    ref = "acme/moved-tag"
    old_ram = ("snapshot-a", 6)
    new_disk = ("snapshot-b", 7)

    executor.store.residency.track_disk(ref, tmp_path)
    executor.store.residency.track_ram(ref, object())
    with executor.store._identity_lock:
        executor.store._resident_identities[ref] = old_ram
        executor.store._disk_identities[ref] = new_disk

    tier_captured = threading.Event()
    allow_snapshot = threading.Event()
    callback_started = threading.Event()
    original_snapshot = executor.store.residency.snapshot
    original_event = executor.store.residency._on_event

    def pause_after_tier_capture():
        rows = original_snapshot()
        tier_captured.set()
        assert allow_snapshot.wait(2)
        return rows

    def observe_callback(*args) -> None:
        callback_started.set()
        original_event(*args)

    monkeypatch.setattr(executor.store.residency, "snapshot", pause_after_tier_capture)
    monkeypatch.setattr(executor.store.residency, "_on_event", observe_callback)
    monkeypatch.setattr("gen_worker.models.residency.flush_memory", lambda: None)

    captured: list[pb.ModelResidency] = []
    snapshot = threading.Thread(
        target=lambda: captured.extend(executor.store.residency_snapshot())
    )
    snapshot.start()
    assert tier_captured.wait(2)
    transition = threading.Thread(
        target=executor.store.residency.release_to_disk,
        args=(ref,),
    )
    transition.start()
    assert callback_started.wait(2)
    assert transition.is_alive(), "identity callback must wait for Hello snapshot"
    allow_snapshot.set()
    snapshot.join(2)
    transition.join(2)
    assert not snapshot.is_alive()
    assert not transition.is_alive()

    ram = captured[0]
    assert ram.tier == pb.RESIDENCY_TIER_RAM
    assert (ram.snapshot_digest, ram.residency_generation) == old_ram
    monkeypatch.setattr(executor.store.residency, "snapshot", original_snapshot)
    disk = executor.store.residency_snapshot()[0]
    assert disk.tier == pb.RESIDENCY_TIER_DISK
    assert (disk.snapshot_digest, disk.residency_generation) == new_disk


def test_declared_protobuf_floor_imports_generated_code() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = next(
        dep for dep in project["project"]["dependencies"] if dep.startswith("protobuf>=")
    )
    floor = tuple(map(int, requirement.removeprefix("protobuf>=").split(".")))
    source = (root / "src/gen_worker/pb/worker_scheduler_pb2.py").read_text(
        encoding="utf-8"
    )
    match = re.search(r"Protobuf Python Version: (\d+)\.(\d+)\.(\d+)", source)
    assert match is not None
    assert floor >= tuple(map(int, match.groups()))
    assert pb.DESCRIPTOR.name == "worker_scheduler.proto"


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
        assert executor.store.keep == ["acme/new"]
        assert lifecycle._state_delta().observed_residency_generation == 2
        assert calls == [
            ("acme/old", "https://r2/old"),
            ("acme/new", "https://r2/new-1"),
            ("acme/new", "https://r2/new-2"),
        ]

    asyncio.run(run())


def test_non_idle_hello_ack_banks_snapshot_before_reconcile() -> None:
    async def run() -> None:
        lifecycle, executor, _ = _lifecycle()
        ref = "tensorhub/active-request"
        executor._idle.clear()
        waiting = asyncio.create_task(executor.store._await_hub_snapshot(ref))
        await asyncio.sleep(0)
        assert not waiting.done()

        await lifecycle.on_hello_ack(pb.HelloAck(
            desired_residency=pb.DesiredResidency(
                generation=1,
                disk_refs=[ref],
                snapshots={ref: _snapshot("https://r2/reminted")},
            )
        ))

        snapshot = await asyncio.wait_for(waiting, 0.1)
        assert snapshot.files[0].url == "https://r2/reminted"
        assert not executor._idle.is_set()
        assert lifecycle._residency_task is not None
        assert not lifecycle._residency_task.done()
        lifecycle._cancel_residency_reconcile()

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
        ref = "acme/background"
        desired_snapshot = _snapshot("https://r2/desired-b")
        desired_snapshot.digest = "bb" * 32
        priority_snapshot = _snapshot("https://r2/priority-a")
        priority_snapshot.digest = "aa" * 32

        async def ensure_local(ref: str, snapshot=None, *, binding=None):
            nonlocal calls
            calls += 1
            if snapshot is not None:
                executor.store.bank_snapshot(ref, snapshot)
            if calls == 1:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    canceled.set()
                    raise
            assert executor.store._snapshot_identity(ref, snapshot) == (
                desired_snapshot.digest,
                7,
            )
            resumed.set()

        async def handle_run_job(run_job: pb.RunJob) -> None:
            executor._idle.clear()
            await asyncio.sleep(0)
            assert canceled.is_set(), "background work must cancel before request setup"
            # Production RunJob materialization banks its exact snapshot with
            # no desired generation.  The mutable ref has temporarily moved
            # from desired B/gen7 back to request A/gen0.
            executor.store.bank_snapshot(ref, run_job.snapshots[ref])
            assert executor.store._snapshot_identity(
                ref, run_job.snapshots[ref]
            ) == (priority_snapshot.digest, 0)

        monkeypatch.setattr(executor.store, "ensure_local", ensure_local)
        monkeypatch.setattr(executor, "handle_run_job", handle_run_job)

        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=7,
            disk_refs=[ref],
            snapshots={ref: desired_snapshot},
        )))
        await started.wait()

        await lifecycle.on_message(pb.SchedulerMessage(run_job=pb.RunJob(
            request_id="request", attempt=1, function_name="generate",
            snapshots={ref: priority_snapshot},
        )))
        await asyncio.sleep(0)
        assert calls == 1, "background work must stay paused while a job is active"

        executor._idle.set()
        await resumed.wait()
        assert calls == 2, "current desired state must resume after the job becomes idle"

        # DesiredResidency is full replacement. Once B is removed, a later
        # generation-less request for the same digest must not resurrect gen7.
        await lifecycle.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=8,
        )))
        executor.store.bank_snapshot(ref, desired_snapshot)
        assert executor.store._snapshot_identity(
            ref, desired_snapshot
        ) == (desired_snapshot.digest, 0)

    asyncio.run(run())


def test_run_job_cancellation_waits_for_model_load_thread(monkeypatch, tmp_path: Path) -> None:
    async def run() -> None:
        lifecycle, executor, _ = _lifecycle()
        load_started = threading.Event()
        release_load = threading.Event()
        second_load_done = threading.Event()
        counter_lock = threading.Lock()
        active = 0
        max_active = 0
        calls = 0

        async def ensure_local(ref: str, **kwargs):
            return tmp_path

        def load_slot(*args, **kwargs):
            nonlocal active, max_active, calls
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
                calls += 1
                call = calls
            try:
                load_started.set()
                assert release_load.wait(2), "test model load was never released"
                return provision.SlotLoad(
                    obj=_Pipeline(), is_pipeline=True, placed={"mode": "cpu"}
                )
            finally:
                with counter_lock:
                    active -= 1
                if call == 2:
                    second_load_done.set()

        request_acquired_load_lock = asyncio.Event()

        async def handle_run_job(run_job: pb.RunJob) -> None:
            async with executor._load_lock:
                request_acquired_load_lock.set()

        monkeypatch.setattr(executor_mod, "ensure_local", ensure_local)
        monkeypatch.setattr(executor, "handle_run_job", handle_run_job)
        monkeypatch.setattr(provision, "load_slot", load_slot)

        picked = "tensorhub/cyberrealistic-pony:prod"
        snapshot = pb.Snapshot(
            digest="blake3:" + "a" * 64,
            files=[pb.SnapshotFile(
                path="model.safetensors", size_bytes=1,
                blake3="a" * 64, url="https://r2/model",
            )],
        )
        await lifecycle.on_hello_ack(pb.HelloAck(
            desired_residency=pb.DesiredResidency(
                generation=1,
                snapshots={picked: snapshot},
                hot=[pb.DesiredInstance(
                    function_name="generate",
                    models=[pb.ModelBinding(slot="pipeline", ref=picked)],
                )],
            )
        ))
        assert await asyncio.to_thread(load_started.wait, 2)

        request = asyncio.create_task(lifecycle.on_message(pb.SchedulerMessage(
            run_job=pb.RunJob(
                request_id="request", attempt=1, function_name="generate",
                models=[pb.ModelBinding(slot="pipeline", ref=picked)],
            )
        )))
        await asyncio.sleep(0.05)
        assert not request_acquired_load_lock.is_set()

        release_load.set()
        await asyncio.wait_for(request, 2)
        assert request_acquired_load_lock.is_set()
        assert await asyncio.to_thread(second_load_done.wait, 2)
        assert max_active == 1
        lifecycle._cancel_residency_reconcile()

    asyncio.run(run())
