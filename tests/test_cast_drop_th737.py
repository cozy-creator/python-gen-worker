"""th#737: a cast pick that cannot be satisfied is a STRUCTURAL degradation,
never a silent bf16 fallback.

Two layers under test:
- loading: the cast outcome is stamped on the pipe
  (``_cozy_fp8_storage_requested`` / ``_cozy_fp8_storage_ok``);
- executor: a denoiser-less snapshot drops the cast pre-load and records a
  ``FnDegraded``-shaped ServePlan (wanted=fp8, ran=bf16) via the real
  ensure_setup/injection path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from pathlib import Path

import msgspec

from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, ModelStore
from gen_worker.models.loading import load_from_pretrained
from gen_worker.models.serve_fit import ServePlan, cast_dropped
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


# --------------------------------------------------------------------------
# serve_fit: the structural plan
# --------------------------------------------------------------------------


def test_cast_dropped_plan_is_degraded() -> None:
    plan = cast_dropped(None, wanted="fp8", detail="no cast surface")
    assert plan.degraded
    assert plan.wanted == "fp8" and plan.ran == "bf16"
    assert plan.serveable


def test_native_matching_plan_is_not_degraded() -> None:
    plan = ServePlan(serveable=True, run_mode="native", fit="fits",
                     wanted="bf16", ran="bf16")
    assert not plan.degraded


# --------------------------------------------------------------------------
# loading: cast outcome stamped on the pipe
# --------------------------------------------------------------------------


class _Denoiser:
    def __init__(self) -> None:
        self.casting_calls: list = []

    def parameters(self):
        return iter(())

    def enable_layerwise_casting(self, *, storage_dtype, compute_dtype) -> None:
        self.casting_calls.append((storage_dtype, compute_dtype))


class _DenoiserPipe:
    def __init__(self) -> None:
        self.transformer = _Denoiser()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        return cls()


class _NoSurfacePipe:
    """Latent-upsampler shape: no transformer/unet, not a bare nn.Module."""

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        return cls()


def _write_safetensors(path: Path, dtype: str = "BF16", nbytes: int = 1024) -> None:
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [nbytes], "data_offsets": [0, nbytes]}}
    ).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def _snapshot(tmp_path: Path, components: dict) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    index = {"_class_name": "Pipe"}
    index.update(components)
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    _write_safetensors(tmp_path / "diffusion_pytorch_model.safetensors")
    return tmp_path


def test_load_stamps_cast_ok(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, {"transformer": ["x", "y"]})
    pipe = load_from_pretrained(_DenoiserPipe, snap, dtype="bf16",
                                storage_dtype="fp8")
    assert pipe._cozy_fp8_storage_requested is True
    assert pipe._cozy_fp8_storage_ok is True
    assert len(pipe.transformer.casting_calls) == 1


def test_load_stamps_cast_failure(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, {"latent_upsampler": ["x", "y"]})
    pipe = load_from_pretrained(_NoSurfacePipe, snap, dtype="bf16",
                                storage_dtype="fp8")
    assert pipe._cozy_fp8_storage_requested is True
    assert pipe._cozy_fp8_storage_ok is False


# --------------------------------------------------------------------------
# executor: pre-load drop + structural report through the real load path
# --------------------------------------------------------------------------


class _In(msgspec.Struct):
    x: str = "hi"


class _Out(msgspec.Struct):
    ok: bool = True


def _executor(spec: EndpointSpec, tmp_path: Path, snap: Path, sent: list) -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path, vram_budget_bytes=4 << 30)

    async def _fake_ensure_local(ref, snapshot=None, *, binding=None) -> Path:
        store.residency.track_disk(ref, snap)
        return snap

    store.ensure_local = _fake_ensure_local  # type: ignore[method-assign]
    return Executor([spec], _send, store=store)


class _UpsamplerShim:
    """No transformer/unet component; records that the loader saw NO cast."""

    to_calls: list = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        return cls()

    def to(self, *a, **k):
        type(self).to_calls.append((a, k))
        return self


def test_executor_drops_cast_on_denoiserless_snapshot(
        tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "0")

    class Endpoint:
        def setup(self, m: _UpsamplerShim) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="upsample", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"m": Hub("acme/upsampler", storage_dtype="fp8")},
        resources=Resources(vram_gb=1.0),
    )
    snap = _snapshot(tmp_path / "snapdir", {"latent_upsampler": ["a", "b"], "vae": ["a", "b"]})
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, snap, sent)
        with caplog.at_level(logging.WARNING):
            inst = await ex.ensure_setup(spec)
        # The cast was dropped BEFORE load: no fp8 stamp on the pipe.
        assert not getattr(inst.m, "_cozy_fp8_storage_requested", False)
        # Structural report, FnDegraded-shaped: wanted fp8, ran bf16.
        plan = ex.serve_plans["upsample"]
        assert plan.degraded
        assert plan.wanted == "fp8" and plan.ran == "bf16"
        assert "no denoiser/cast surface" in plan.warning
        assert "CAST_DROPPED" in caplog.text

    asyncio.run(_go())


class _DenoiserShim(_DenoiserPipe):
    def to(self, *a, **k):
        return self


def test_executor_keeps_cast_on_denoiser_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "0")

    class Endpoint:
        def setup(self, m: _DenoiserShim) -> None:
            self.m = m

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out()

    spec = EndpointSpec(
        name="generate", method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint, attr_name="run",
        models={"m": Hub("acme/z-image", storage_dtype="fp8")},
        resources=Resources(vram_gb=1.0),
    )
    snap = _snapshot(tmp_path / "snapdir", {"transformer": ["a", "b"], "vae": ["a", "b"]})
    sent: list = []

    async def _go() -> None:
        ex = _executor(spec, tmp_path, snap, sent)
        inst = await ex.ensure_setup(spec)
        assert inst.m._cozy_fp8_storage_requested is True
        assert inst.m._cozy_fp8_storage_ok is True
        plan = ex.serve_plans.get("generate")
        assert plan is None or not plan.degraded

    asyncio.run(_go())
