"""th#938 (tensorhub P0, pgw#606): the th#912 boot-setup watcher ran
`ensure_setup` on the unmodified class-table spec for Slot functions,
materializing the IMAGE-BAKED code default over the hub-stamped release
binding. On the sdxl fleet template (pipeline=Civitai("827184") code default +
vae=Hub(tensorhub/...)) the watcher's `missing` set only counted the
tensorhub-sourced vae — the moment the hub delivered it to disk, setup fetched
the raw Civitai default -> civitai_not_found -> both class fns setup_failed ->
3/3 pods retired -> release-broken.

Contract under test (precedence): the hub-stamped binding is the ONLY setup
source when hub-connected — delivered via Hot DesiredInstance / RunJob, both
rebinding through `_effective_spec`. The code Slot default is the hub-less
bootstrap fallback and must never be fetched or set up at boot.

The fixture reproduces the live J31 spec shape and hub-delivery sequence
(stack th919live, hub log tensorhub-1784570340541369820.log); the store is
faked only at its network boundary and fails Civitai refs exactly as
production did.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Tuple

import msgspec

import gen_worker.lifecycle as lifecycle_mod
from gen_worker.api.binding import Civitai, Hub, wire_ref
from gen_worker.api.slot import Slot
from gen_worker.config.settings import Settings
from gen_worker.executor import Executor, _MaterializedLocal
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

_CODE_DEFAULT = Civitai("827184", version="2883731")
_VAE = Hub("tensorhub/sdxl-vae-fp16-fix", tag="prod")
_STAMPED = "tensorhub/wai-illustrious:prod"
_VAE_REF = wire_ref(_VAE)


class _In(msgspec.Struct):
    model: str = ""


class _Out(msgspec.Struct):
    pipeline_path: str


def _sdxl_spec(setup_calls: List[Tuple[str, str]]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str, vae: str) -> None:
            self.pipeline_path = pipeline
            setup_calls.append((pipeline, vae))

        def generate(self, ctx: Any, payload: _In) -> _Out:  # pragma: no cover
            return _Out(pipeline_path=self.pipeline_path)

    return EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate",
        models={"pipeline": _CODE_DEFAULT, "vae": _VAE},
        slots={
            "pipeline": Slot(str, selected_by="model", default_checkpoint=_CODE_DEFAULT),
            "vae": Slot(str, default_checkpoint=_VAE),
        },
    )


def _hardware() -> dict:
    return {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
            "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90", "installed_libs": []}


async def _noop_send(msg: pb.WorkerMessage) -> None:
    pass


def _snapshot(digest: str) -> pb.Snapshot:
    return pb.Snapshot(
        digest=digest * 32,
        files=[pb.SnapshotFile(
            path="model.safetensors", size_bytes=1, blake3="cd" * 32,
            url="https://r2/blob",
        )],
    )


def _fake_store_network(ex: Executor, tmp_path, monkeypatch, local: set) -> List[str]:
    """Fake the store at its network boundary with production semantics:
    tensorhub refs materialize once the hub delivered them; Civitai refs fail
    exactly as the live pod did (no mirror, no key)."""
    requested: List[str] = []

    async def _materialize(ref: str, snapshot=None, *, binding: Any = None):
        requested.append(ref)
        if binding is not None and binding.source == "civitai" or ref == "827184":
            raise ValueError("civitai_not_found")
        return _MaterializedLocal(path=tmp_path, identity=("", 0))

    async def _ensure_local(ref: str, snapshot=None, *, binding: Any = None):
        requested.append(ref)
        if ref == "827184":
            raise ValueError("civitai_not_found")
        local.add(ref)
        return tmp_path

    async def _revalidate(ref: str, snapshot=None) -> None:
        return None

    monkeypatch.setattr(ex.store, "_materialize_local", _materialize)
    monkeypatch.setattr(ex.store, "ensure_local", _ensure_local)
    monkeypatch.setattr(ex, "revalidate_snapshot_identity", _revalidate)
    monkeypatch.setattr(ex.store, "local_path",
                        lambda ref: tmp_path if ref in local else None)
    return requested


def test_boot_never_fetches_or_sets_up_the_image_baked_code_default(
    tmp_path, monkeypatch,
) -> None:
    """The live repro: vae (the only tensorhub-sourced slot default) lands on
    disk via hub delivery; nothing may fetch the Civitai code default or run
    setup — pre-fix this poisoned every fn on the class within one watch tick."""
    monkeypatch.setattr(lifecycle_mod, "_BOOT_SETUP_WATCH_INTERVAL_S", 0.02)

    setup_calls: List[Tuple[str, str]] = []
    ex = Executor([_sdxl_spec(setup_calls)], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    local: set = set()
    requested = _fake_store_network(ex, tmp_path, monkeypatch, local)

    async def _scenario() -> None:
        await lc.startup()
        assert lc._boot_setup_watch is None, (
            "Slot functions must not spawn a boot-setup watcher"
        )
        # Hub delivers the vae to disk (th#911 DesiredResidency disk plan).
        local.add(_VAE_REF)
        await asyncio.sleep(0.2)

    asyncio.run(_scenario())

    assert "827184" not in requested, (
        f"boot materialized the image-baked Civitai code default: {requested}"
    )
    assert setup_calls == []
    assert ex.unavailable == {}, (
        f"boot setup poisoned functions from the code default: {ex.unavailable}"
    )
    assert ex.available_functions() == ["generate"]


def test_hub_stamped_binding_outranks_code_default_via_hot_instance(
    tmp_path, monkeypatch,
) -> None:
    """Full hub-delivery path from the live log: HelloAck carries the disk
    plan (stamped base + vae) and a Hot DesiredInstance bound to the stamped
    refs. Setup must run exactly once, on the effective spec carrying the
    hub-stamped binding — never the code default."""
    setup_calls: List[Tuple[str, str]] = []
    ex = Executor([_sdxl_spec(setup_calls)], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    local: set = set()
    requested = _fake_store_network(ex, tmp_path, monkeypatch, local)

    captured: List[Tuple[EndpointSpec, dict]] = []

    async def _capture_setup(spec, snapshots=None, promote_slots=None):
        captured.append((spec, dict(snapshots or {})))
        rec = ex._classes[spec.instance_key]
        rec.ready = True
        return None

    monkeypatch.setattr(ex, "ensure_setup", _capture_setup)

    snapshots = {_STAMPED: _snapshot("ab"), _VAE_REF: _snapshot("ef")}

    async def _scenario() -> None:
        await lc.startup()
        assert setup_calls == [] and captured == []
        await lc.on_hello_ack(pb.HelloAck(desired_residency=pb.DesiredResidency(
            generation=4,
            disk_refs=[_STAMPED, _VAE_REF],
            snapshots=snapshots,
            hot=[pb.DesiredInstance(
                function_name="generate",
                models=[
                    pb.ModelBinding(slot="pipeline", ref=_STAMPED),
                    pb.ModelBinding(slot="vae", ref=_VAE_REF),
                ],
            )],
        )))
        task = lc._residency_task
        assert task is not None
        await asyncio.wait_for(task, 5)

    asyncio.run(_scenario())

    assert "827184" not in requested, requested
    assert len(captured) == 1, captured
    effective, got_snapshots = captured[0]
    assert wire_ref(effective.models["pipeline"]) == _STAMPED
    assert wire_ref(effective.models["vae"]) == _VAE_REF
    assert effective.instance_key != ex.specs["generate"].instance_key
    assert got_snapshots[_STAMPED].digest == snapshots[_STAMPED].digest
    assert ex.unavailable == {}


def test_all_tensorhub_slot_defaults_still_never_boot_setup(
    tmp_path, monkeypatch,
) -> None:
    """A Slot fn whose defaults are ALL tensorhub-sourced must also wait for
    hub delivery: the hub may stamp a DIFFERENT checkpoint than the code
    default, so a boot-time class-table setup is wrong even when fetchable."""
    monkeypatch.setattr(lifecycle_mod, "_BOOT_SETUP_WATCH_INTERVAL_S", 0.02)

    setup_calls: List[Tuple[str, str]] = []

    class Endpoint:
        def setup(self, pipeline: str) -> None:  # pragma: no cover
            setup_calls.append(("setup", pipeline))

        def generate(self, ctx: Any, payload: _In) -> _Out:  # pragma: no cover
            return _Out(pipeline_path="")

    default = Hub("tensorhub/qwen-image-edit-2511", tag="prod")
    spec = EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate",
        models={"pipeline": default},
        slots={"pipeline": Slot(str, selected_by="model", default_checkpoint=default)},
    )
    ex = Executor([spec], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    local: set = set()
    _fake_store_network(ex, tmp_path, monkeypatch, local)

    async def _scenario() -> None:
        await lc.startup()
        assert lc._boot_setup_watch is None
        local.add(wire_ref(default))
        await asyncio.sleep(0.2)

    asyncio.run(_scenario())

    assert setup_calls == []
    assert ex.unavailable == {}
    assert ex.available_functions() == ["generate"]
