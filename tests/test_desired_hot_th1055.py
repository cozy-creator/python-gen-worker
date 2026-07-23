"""th#1055: desired-hot warm on slot-only endpoints + loud failures.

Live root cause (master fleet, 2026-07-23): ``ensure_desired_instance``
demanded ``set(bindings) == set(spec.models)``, but every fleet endpoint now
declares deploy-bound Slots with NO code default (ie#524/th#980), so
``spec.models`` is EMPTY and every hub hot intent — gw#587 self-mint
prewarm, th#912 slot-default seeding, #567 compile-cell reload — was refused
with ``ValidationError("must bind exactly []")``, swallowed by
``_reconcile_pass`` as one pod-local warning. Zero events, zero VRAM, no
retry signal: the th#868 w8a8 fence never opened (qwen H100 rigs, sdxl
ie529.r2, ltx B200 all deadlocked) and precompiled cells never armed
fleet-wide.

Pinned here:
  1. a slot-only (default-less) compile endpoint warms + arms from a
     resolved-space DesiredInstance;
  2. declared-space hot bindings remap through the HelloAck precision picks
     (the hello_ack.go "hot bindings stay declared" contract) instead of
     silently undoing the pick;
  3. every desired-instance failure — including the pre-setup validation
     refusals — emits MODEL_STATE_FAILED for the instance refs so the stall
     is fleet-visible (the never-logs stall must be impossible).
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

import msgspec
import pytest

import gen_worker
from gen_worker import Compile
from gen_worker import compile_cache as cc
from gen_worker.api.binding import Hub
from gen_worker.api.slot import Slot
from gen_worker.api.errors import ValidationError
from gen_worker.executor import Executor
from gen_worker.models import provision
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

FAMILY = "th1055-fam"
AUTHORED = Hub("acme/qwen-image", tag="prod")
AUTHORED_REF = "acme/qwen-image:prod"
RESOLVED_REF = "acme/qwen-image:prod#fp8-w8a8"
ADAPTER_REF = "acme/qwen-lightning:prod"
CELL_REF = f"root/family-{FAMILY}#inductor-rtx-4090-torch2.9-w8a8"


class _Denoiser:
    def forward(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Denoiser()


class _CompilePipe(_Pipe):
    def __init__(self) -> None:
        super().__init__()
        self._cozy_weight_lane = "w8a8"


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


def _slot_only_spec(setup_calls: List[Tuple[str, str]]) -> EndpointSpec:
    """The fleet's real shape: deploy-bound Slots, EMPTY spec.models."""

    class Endpoint:
        def setup(self, pipeline: str, adapter: str) -> None:
            self.pipe = _CompilePipe()
            self.armed = gen_worker.arm_compile(self.pipe)
            setup_calls.append((pipeline, adapter))

        def warmup(self) -> None:
            signal = getattr(self.pipe, cc._MARKER_ATTR)["failure_signal"]
            with signal["lock"]:
                signal["successful_calls"] += 1
                signal["cache_hits"] += 1

        def generate(self, ctx: Any, payload: _In) -> _Out:
            return _Out()

    return EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={},
        slots={"pipeline": Slot(str), "adapter": Slot(str)},
        slot_family={"pipeline": FAMILY, "adapter": FAMILY},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )


def _default_binding_spec(setup_calls: List[str]) -> EndpointSpec:
    class Endpoint:
        def setup(self, pipeline: str) -> None:
            self.pipe = _CompilePipe()
            self.armed = gen_worker.arm_compile(self.pipe)
            setup_calls.append(pipeline)

        def warmup(self) -> None:
            signal = getattr(self.pipe, cc._MARKER_ATTR)["failure_signal"]
            with signal["lock"]:
                signal["successful_calls"] += 1
                signal["cache_hits"] += 1

        def generate(self, ctx: Any, payload: _In) -> _Out:
            return _Out()

    return EndpointSpec(
        name="generate", method=Endpoint.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="generate", models={"pipeline": AUTHORED},
        compile=Compile(shapes=((768, 768),), family=FAMILY),
    )


def _snapshot(digest: str) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])


def _cell_artifact(tmp_path: Path) -> Path:
    cap = tmp_path / "cap"
    (cap / "inductor" / "g").mkdir(parents=True)
    (cap / "inductor" / "g" / "code.py").write_text("x")
    (cap / "triton").mkdir()
    cfg = Compile(shapes=((768, 768),), family=FAMILY)
    signature, weight_contract = cc.execution_contract(_Pipe(), cfg)
    meta = cc.artifact_metadata(
        family=FAMILY, shapes=cfg.shapes, targets=cfg.targets,
        graph_signature=signature, weight_contract=weight_contract,
    )
    out = tmp_path / "minted"
    out.mkdir(exist_ok=True)
    return cc.pack(cap, out / "inductor-rtx-4090-torch2.9-w8a8.tar.gz", meta)


def _harness(tmp_path: Path, monkeypatch, specs: List[EndpointSpec]):
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(specs, _send)
    artifact = _cell_artifact(tmp_path)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        p = tmp_path / ref.replace("/", "_").replace(":", "_").replace("#", "_")
        p.mkdir(parents=True, exist_ok=True)
        if ref.startswith("root/"):
            shutil.copy(artifact, p / artifact.name)
        return p

    import gen_worker.executor as ex_mod
    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)

    enables: List[Tuple[Any, Optional[Path]]] = []

    def _fake_enable(pipe: Any, _cfg: Any, _cache_dir: Any, artifact_path: Any) -> bool:
        enables.append((pipe, artifact_path))
        setattr(pipe, cc._MARKER_ATTR, {
            "failure_signal": {
                "callback": None,
                "lock": threading.Lock(),
                "successful_calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "originals": [],
            "regional_mods": [],
        })
        return True

    monkeypatch.setattr(provision, "enable_compiled", _fake_enable)
    return ex, sent, enables


def _failed_model_events(sent: List[pb.WorkerMessage]) -> List[pb.ModelEvent]:
    return [
        m.model_event for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_FAILED
    ]


# ---------------------------------------------------------------------------
# 1. slot-only endpoint: the resolved-space hot intent warms and arms
# ---------------------------------------------------------------------------


def test_slot_only_desired_instance_warms_and_arms(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, enables = _harness(tmp_path, monkeypatch,
                                 [_slot_only_spec(setup_calls)])
    ex.apply_model_resolutions(
        {AUTHORED_REF: (RESOLVED_REF, "", "fp8-w8a8-dynamic+compiled")})

    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF),
                pb.ModelBinding(slot="adapter", ref=ADAPTER_REF)],
    )
    snapshots = {RESOLVED_REF: _snapshot("aa" * 32),
                 ADAPTER_REF: _snapshot("ab" * 32),
                 CELL_REF: _snapshot("bb" * 32)}
    asyncio.run(ex.ensure_desired_instance(desired, snapshots))

    assert len(setup_calls) == 1
    pipeline_path, adapter_path = setup_calls[0]
    assert "fp8-w8a8" in pipeline_path
    assert len(enables) == 1
    (target,) = ex.compile_targets()
    assert target.active_compile_ref == CELL_REF
    assert "generate" in ex.available_functions()
    assert not _failed_model_events(sent)


# ---------------------------------------------------------------------------
# 2. declared-space hot bindings remap through the precision picks
# ---------------------------------------------------------------------------


def test_declared_space_hot_bindings_remap_through_picks(tmp_path, monkeypatch) -> None:
    """seedDynamicSlotDefaults sends DECLARED refs ("the worker rebinds
    specs through the resolutions map", hello_ack.go th#697) — the warm
    instance must derive the RESOLVED binding, never silently undo the
    pick (and never download the declared artifact the disk plan skipped)."""
    setup_calls: List[str] = []
    ex, _sent, _enables = _harness(tmp_path, monkeypatch,
                                   [_default_binding_spec(setup_calls)])
    ex.apply_model_resolutions(
        {AUTHORED_REF: (RESOLVED_REF, "", "fp8-w8a8-dynamic+compiled")})

    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=AUTHORED_REF)],
    )
    snapshots = {RESOLVED_REF: _snapshot("aa" * 32),
                 CELL_REF: _snapshot("bb" * 32)}
    asyncio.run(ex.ensure_desired_instance(desired, snapshots))

    assert len(setup_calls) == 1
    assert "fp8-w8a8" in setup_calls[0], (
        "declared-space hot binding undid the precision pick")
    rec = ex._classes[ex.specs["generate"].instance_key]
    assert rec.held_refs == [RESOLVED_REF]


# ---------------------------------------------------------------------------
# 3. failures are LOUD: validation refusals emit MODEL_STATE_FAILED
# ---------------------------------------------------------------------------


def test_validation_refusal_emits_model_events(tmp_path, monkeypatch) -> None:
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, _enables = _harness(tmp_path, monkeypatch,
                                  [_slot_only_spec(setup_calls)])

    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF),
                pb.ModelBinding(slot="bogus", ref=ADAPTER_REF)],
    )
    with pytest.raises(ValidationError, match="declared slots"):
        asyncio.run(ex.ensure_desired_instance(desired, {}))

    failed = _failed_model_events(sent)
    assert {e.ref for e in failed} == {RESOLVED_REF, ADAPTER_REF}, (
        "a refused hot intent must be fleet-visible, not a pod-local warning")
    assert setup_calls == []


def test_unbound_default_less_slot_refuses_loud(tmp_path, monkeypatch) -> None:
    """A hot intent missing a default-less slot cannot fall back to code —
    refuse (loud), matching dispatch's mirror-first contract."""
    setup_calls: List[Tuple[str, str]] = []
    ex, sent, _enables = _harness(tmp_path, monkeypatch,
                                  [_slot_only_spec(setup_calls)])

    desired = pb.DesiredInstance(
        function_name="generate",
        models=[pb.ModelBinding(slot="pipeline", ref=RESOLVED_REF)],
    )
    with pytest.raises(ValidationError, match="declared slots"):
        asyncio.run(ex.ensure_desired_instance(desired, {}))
    assert _failed_model_events(sent)
    assert setup_calls == []
