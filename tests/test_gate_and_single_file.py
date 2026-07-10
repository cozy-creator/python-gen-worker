"""VRAM recommendation gating + single-file checkpoint loading.

``Resources(vram_gb=X)`` recommends a card SIZE (total VRAM of the smallest
card the author targets), never a free-bytes requirement. A "24GB" card
reports ~23.99GiB total / ~23.6GiB free via torch.cuda.mem_get_info
(driver/framebuffer/CUDA-context reserve); neither the worker self-gate nor
the variant fit policy may disable ``vram_gb=24`` functions on exactly the
card they target (GPU_VRAM_OVERHEAD_GB absorbs the reserve). Single-file
repos (Illustrious-XL, civitai checkpoints) have no ``model_index.json`` and
must route through ``cls.from_single_file``.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker.api.binding import HF
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor
from gen_worker.models.loading import _single_file_checkpoint, load_from_pretrained
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


def _spec(name: str, vram_gb: float) -> EndpointSpec:
    class Endpoint:
        def setup(self, model: str) -> None:  # pragma: no cover
            pass

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    return EndpointSpec(
        name=name, method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": HF("acme/tiny")},
        resources=Resources(vram_gb=vram_gb),
    )


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


RTX_4090_TOTAL_BYTES = 25_757_220_864  # 23.99 GiB — what mem_get_info reports
RTX_4090_FREE_GB = 23.6  # idle card: total minus driver/framebuffer/context


def test_gate_accepts_card_of_exactly_recommended_size() -> None:
    ex = Executor([_spec("fits-24", 24.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert "fits-24" not in ex.unavailable


def test_gate_serves_bigger_model_via_runtime_quant_rungs() -> None:
    """th#683 P3: a model bigger than the card is NOT refused on the hint — it
    serves via the runtime quant rungs (on-GPU: fp8-E4M3 storage when the
    halved estimate fits, else the emergency 4-bit rung), regardless of the
    CPU-offload veto (neither is CPU-touching)."""
    # 32 GB on a 24 GB card: 32*0.55=17.6 fits ~24 free -> fp8 storage rung.
    ex = Executor([_spec("needs-32", 32.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert "needs-32" not in ex.unavailable
    plan = ex.serve_plans["needs-32"]
    assert plan.serveable and plan.degraded
    assert plan.run_mode == "fp8_storage"
    assert plan.warning  # honest-guidance surfaced

    # 48 GB on the same card: 48*0.55=26.4 doesn't fit but 48*0.45=21.6
    # does -> the emergency 4-bit rung.
    ex = Executor([_spec("needs-48", 48.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert "needs-48" not in ex.unavailable
    plan = ex.serve_plans["needs-48"]
    assert plan.serveable and plan.degraded
    assert plan.run_mode == "emergency_quant"
    assert plan.warning


def test_gate_offload_only_model_forbidden_here(monkeypatch) -> None:
    """A model so large even 4-bit won't fit needs CPU/disk offload. On a box
    that forbids CPU-touching placement it is unserveable HERE (belongs on the
    GPU lane) — with a distinct reason, never a silent 'too big'."""
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    ex = Executor([_spec("huge", 64.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert ex.unavailable["huge"][0] == "offload_forbidden"


def test_gate_offload_only_model_serves_when_allowed(monkeypatch) -> None:
    """The same too-large model serves via CPU/disk offload when offload is
    allowed (the production/GPU-lane case): fit over speed, never refused."""
    monkeypatch.delenv("GEN_WORKER_FORBID_CPU_OFFLOAD", raising=False)
    ex = Executor([_spec("huge", 64.0)], _noop_send)
    ex.gate_functions({"gpu_count": 1, "gpu_total_mem": RTX_4090_TOTAL_BYTES, "gpu_sm": "89"})
    assert "huge" not in ex.unavailable
    plan = ex.serve_plans["huge"]
    assert plan.serveable and plan.run_mode == "offload"
    assert plan.est_latency_multiplier > 1.0 and plan.warning


def test_gate_cpu_only_fallback_when_no_gpu(monkeypatch) -> None:
    """th#683 P3 CPU-only ultimate fallback: with no GPU, a GPU function is
    offered on CPU (very slow) behind a warning when allowed, and refused with
    a distinct reason only when CPU inference is forbidden here."""
    monkeypatch.delenv("GEN_WORKER_FORBID_CPU_OFFLOAD", raising=False)
    ex = Executor([_spec("needs-24", 24.0)], _noop_send)
    ex.gate_functions({"gpu_count": 0, "gpu_total_mem": 0, "gpu_sm": ""})
    assert "needs-24" not in ex.unavailable
    plan = ex.serve_plans["needs-24"]
    assert plan.serveable and plan.run_mode == "cpu"
    assert plan.est_latency_multiplier >= 10.0 and plan.warning

    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    ex2 = Executor([_spec("needs-24", 24.0)], _noop_send)
    ex2.gate_functions({"gpu_count": 0, "gpu_total_mem": 0, "gpu_sm": ""})
    assert ex2.unavailable["needs-24"][0] == "cuda_unavailable"


def test_variant_fit_counts_recommended_size_card_as_fitting() -> None:
    """The z-image footgun: vram_gb=24 on an idle 24GB card (~23.6GB free)
    must be FIT_FITS, not offload/unavailable."""
    from gen_worker.models.hub_policy import (
        FIT_FITS,
        TensorhubWorkerCapabilities,
        variant_fit,
    )

    caps = TensorhubWorkerCapabilities(
        cuda_version="12.8", gpu_sm=89, torch_version="2.8.0", installed_libs=[],
    )
    fit, reason = variant_fit(Resources(vram_gb=24.0), caps, RTX_4090_FREE_GB)
    assert (fit, reason) == (FIT_FITS, "")


def test_effective_vram_requirement_floor() -> None:
    from gen_worker.models.memory import (
        GPU_VRAM_OVERHEAD_GB,
        effective_vram_requirement_gb,
    )

    assert effective_vram_requirement_gb(24.0) == 24.0 - GPU_VRAM_OVERHEAD_GB
    assert effective_vram_requirement_gb(0.5) == 0.0  # never negative


def test_single_file_checkpoint_detection(tmp_path: Path) -> None:
    ckpt = tmp_path / "Illustrious-XL-v1.0.safetensors"
    ckpt.write_bytes(b"stub")
    assert _single_file_checkpoint(tmp_path) == ckpt
    assert _single_file_checkpoint(ckpt) == ckpt

    # A diffusers layout is NOT a single-file checkpoint.
    (tmp_path / "model_index.json").write_text("{}")
    assert _single_file_checkpoint(tmp_path) is None


def test_single_file_checkpoint_requires_exactly_one(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"a")
    (tmp_path / "b.safetensors").write_bytes(b"b")
    assert _single_file_checkpoint(tmp_path) is None
    assert _single_file_checkpoint(tmp_path / "missing") is None


def test_load_from_pretrained_routes_single_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"stub")
    calls: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):  # pragma: no cover
            raise AssertionError("single-file snapshot must not use from_pretrained")

        @classmethod
        def from_single_file(cls, path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return cls()

    out = load_from_pretrained(FakePipeline, tmp_path, dtype="fp16")
    assert isinstance(out, FakePipeline)
    assert calls["path"] == str(ckpt)
    assert "variant" not in calls["kwargs"]


def test_load_from_pretrained_still_uses_pretrained_for_layouts(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}")
    calls: dict[str, object] = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["path"] = path
            return cls()

        @classmethod
        def from_single_file(cls, path, **kwargs):  # pragma: no cover
            raise AssertionError("layout snapshot must not use from_single_file")

    out = load_from_pretrained(FakePipeline, tmp_path, dtype="bf16")
    assert isinstance(out, FakePipeline)
    assert calls["path"] == str(tmp_path)
