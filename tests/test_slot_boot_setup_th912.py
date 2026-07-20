"""th#912: a Slot's default_checkpoint is a SEED for the hub's residency plan
(pgw#532) — the hub can deliver it to disk (th#911 seeds `DesiredResidency`
with a dynamic slot's default), but boot's startup scan skips every Slot
function outright ("set up on delivery, not at boot") and, unlike a plain
tensorhub binding, gw#591's `_setup_awaiting_functions` watcher never covered
it either. Live effect (e2e#188 run 15): both workers reached on-disk
residency for the slot's default checkpoint and then sat at 0% GPU forever —
no watcher ever ran `ensure_setup` for the default (no-override) pick, so the
worker never resolved this dispatch-eligibility gap on its own.

Covers:
  (a) boot never eagerly fetches/sets up a Slot whose default isn't local yet
      (unchanged pgw#532 invariant — no regression).
  (b) once the default's snapshot lands on disk (the watcher's poll observes
      `store.local_path` succeed), the watcher runs `ensure_setup` for the
      declared (no-override) pick and pushes a StateDelta.
  (c) a Slot whose default is a RAW upstream ref (non-tensorhub) is never
      queued onto the watcher (mirror-first, gw#465 — unchanged).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

import msgspec

from gen_worker.api.binding import HF, Hub
from gen_worker.api.slot import Slot
from gen_worker.config.settings import Settings
from gen_worker.executor import Executor
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    model: str = ""


class _Out(msgspec.Struct):
    pipeline_path: str


def _slot_spec(name: str, setup_calls: List[Tuple[str, str]], default_checkpoint) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            self.pipeline_path = pipeline
            setup_calls.append((name, pipeline))

        def generate(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(pipeline_path=self.pipeline_path)

    return EndpointSpec(
        name=name, method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate",
        models={"pipeline": default_checkpoint},
        slots={"pipeline": Slot(str, selected_by="model", default_checkpoint=default_checkpoint)},
    )


def _hardware() -> dict:
    return {"gpu_count": 1, "gpu_total_mem": 32 * 1024**3,
            "gpu_free_mem": 30 * 1024**3, "gpu_sm": "90", "installed_libs": []}


async def _noop_send(msg: pb.WorkerMessage) -> None:
    pass


def test_boot_never_eagerly_sets_up_a_slot_default_not_yet_local(caplog) -> None:
    setup_calls: List[Tuple[str, str]] = []
    spec = _slot_spec("generate", setup_calls, Hub("th912-fixture/slot-default", tag="prod"))
    ex = Executor([spec], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    with caplog.at_level(logging.WARNING, logger="gen_worker.lifecycle"):
        asyncio.run(lc.startup())

    assert setup_calls == [], f"boot eagerly set up a not-yet-local Slot default: {setup_calls}"
    # Slot functions always advertise available (pgw#532 — serveability is
    # per-dispatch), so this alone would never surface the deadlock; the
    # watcher's own diagnostic log is what makes it visible.
    assert ex.available_functions() == ["generate"]
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "generate" in m and "th912-fixture/slot-default:prod" in m and "DesiredResidency" in m
        for m in msgs
    ), msgs


def test_boot_setup_completes_once_slot_default_snapshot_lands(tmp_path, monkeypatch) -> None:
    import gen_worker.lifecycle as lifecycle_mod

    monkeypatch.setattr(lifecycle_mod, "_BOOT_SETUP_WATCH_INTERVAL_S", 0.02)

    setup_calls: List[Tuple[str, str]] = []
    spec = _slot_spec("generate", setup_calls, Hub("th912-fixture/slot-default", tag="prod"))
    ex = Executor([spec], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    deltas: List[bool] = []

    async def _fake_delta(**kwargs):
        deltas.append(True)

    monkeypatch.setattr(lc, "maybe_send_state_delta", _fake_delta)

    ran: List[str] = []

    async def _fake_setup(s, snapshots=None):
        ran.append(s.name)
        rec = ex._classes[s.instance_key]
        rec.ready = True
        return None

    monkeypatch.setattr(ex, "ensure_setup", _fake_setup)

    async def _scenario() -> None:
        await lc.startup()
        # Not local yet: the fake `ensure_setup` must not have run.
        assert ran == []
        # The hub's disk plan (th#911) lands the default checkpoint's bytes.
        monkeypatch.setattr(ex.store, "local_path", lambda ref: tmp_path)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if ran:
                break
        assert ran == ["generate"], "watcher never ran ensure_setup for the Slot default"
        assert deltas, "StateDelta push expected after late Slot boot setup"

    asyncio.run(_scenario())


def test_raw_upstream_slot_default_never_queued_on_watcher() -> None:
    """gw#465/mirror-first: a Slot default that ISN'T a tensorhub ref must
    never be watched for local disk arrival (nothing will ever place a raw
    upstream ref via DesiredResidency) — unchanged from before th#912."""
    setup_calls: List[Tuple[str, str]] = []
    spec = _slot_spec("generate", setup_calls, HF("th912-fixture/raw-model"))
    ex = Executor([spec], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    asyncio.run(lc.startup())

    assert lc._boot_setup_watch is None, "a raw-upstream Slot default must not spawn a boot-setup watcher"
    assert setup_calls == []


def test_mixed_slot_defaults_never_queued_on_watcher() -> None:
    """th#927: a spec with BOTH a raw-upstream slot default (sdxl's Civitai
    pipeline seed) and a tensorhub one (Hub vae) must not be watched — the
    watcher would run `ensure_setup` on the unmodified spec once the
    tensorhub refs land and self-fetch the raw default (mirror-first
    violation; live civitai_not_found boot crash-loop)."""
    setup_calls: List[Tuple[str, str]] = []

    class Endpoint:
        def setup(self, pipeline: str, vae: str) -> None:  # pragma: no cover
            setup_calls.append(("generate", pipeline))

        def generate(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(pipeline_path="")

    spec = EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate",
        models={
            "pipeline": HF("th927-fixture/raw-model"),
            "vae": Hub("th927-fixture/vae", tag="prod"),
        },
        slots={
            "pipeline": Slot(str, selected_by="model",
                             default_checkpoint=HF("th927-fixture/raw-model")),
            "vae": Slot(str, default_checkpoint=Hub("th927-fixture/vae", tag="prod")),
        },
    )
    ex = Executor([spec], _noop_send)
    lc = Lifecycle(Settings(orchestrator_public_addr="localhost:1"), ex)
    lc.hardware = _hardware()

    asyncio.run(lc.startup())

    assert lc._boot_setup_watch is None, "a mixed-default Slot spec must not spawn a boot-setup watcher"
    assert setup_calls == []
