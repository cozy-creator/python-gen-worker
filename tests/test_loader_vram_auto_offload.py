"""Issue #20 fix 3: peak-aware VRAM auto-offload trigger.

Today's heuristic enables offload only when `disk_size > max_vram_gb`. For
FLUX.2-klein-4b (15GB on disk) on RTX 4090 (24GB total, ~20GB usable after
the 4GB safety margin) the check returns False — but from_pretrained's
peak VRAM (~1.4x disk size = 21GB) blows past max_vram and OOMs the worker
before any request-time offload can be applied.

New heuristics:
  - peak_estimate = disk_size * 1.4
  - if peak_estimate > max_vram → enable_model_cpu_offload
  - if disk_size >= max_vram * 0.7 → enable_sequential_cpu_offload (more
    aggressive — sequential reuses one slot at a time and is the safest
    bet when the model crowds the GPU)

And the offload methods are applied to the pipeline immediately after
from_pretrained returns, before `.to("cuda")` would otherwise move
weights and OOM.

These tests run without torch installed by injecting a stub `torch`
module — the production code does `import torch` inside `load()` and
references a small handful of attributes (cuda.is_available,
cuda.OutOfMemoryError, no_grad). The stub provides those.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub `torch` so the loader can be imported and exercised in an env that
# doesn't have the real torch wheel installed.
# ---------------------------------------------------------------------------


def _install_torch_stub(cuda_available: bool = False) -> None:
    """Install a minimal fake `torch` module in sys.modules for the
    duration of a test. Idempotent — replaces any existing stub."""
    fake_torch = types.ModuleType("torch")
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
        empty_cache=lambda: None,
        synchronize=lambda: None,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
    fake_torch.no_grad = lambda: _NullCtx()  # type: ignore[attr-defined]
    fake_torch.float16 = "torch.float16"  # type: ignore[attr-defined]
    fake_torch.bfloat16 = "torch.bfloat16"  # type: ignore[attr-defined]
    fake_torch.float32 = "torch.float32"  # type: ignore[attr-defined]
    sys.modules["torch"] = fake_torch


class _NullCtx:
    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, *_a: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def _torch_stub_fixture():
    """Auto-install the torch stub before each test in this module, then
    clean up so we don't leak the fake into other test files."""
    _install_torch_stub(cuda_available=False)
    yield
    sys.modules.pop("torch", None)


# Late imports — must happen after the autouse fixture would normally
# fire, but for module-level imports we just rely on the loader not
# touching torch at import time.
from gen_worker.pipeline import loader as loader_module  # noqa: E402
from gen_worker.pipeline.loader import PipelineConfig, PipelineLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Minimal stand-in for a diffusers pipeline. Records which offload
    method (if any) was called so the test can assert the order."""

    def __init__(self) -> None:
        self.moved_to: Any = None
        self.model_cpu_offload_called = False
        self.sequential_cpu_offload_called = False
        # VAE attribute presence — `_apply_vae_optimizations` checks for it.
        self.vae = None

    def to(self, device: str) -> "_FakePipeline":
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self) -> None:
        self.model_cpu_offload_called = True

    def enable_sequential_cpu_offload(self) -> None:
        self.sequential_cpu_offload_called = True


def _make_loader(max_vram_gb: float, tmp_path: Path) -> PipelineLoader:
    """Build a PipelineLoader with a fixed max_vram_gb and a tmp_path for
    the model dir. Bypasses the auto VRAM detection."""
    with patch.object(loader_module, "get_total_vram_gb", return_value=max_vram_gb + 4.0):
        loader = PipelineLoader(
            models_dir=str(tmp_path),
            max_vram_gb=max_vram_gb,
            vram_safety_margin_gb=4.0,
        )
    return loader


def _make_model_dir(tmp_path: Path) -> Path:
    """Build a minimal model dir that exists (the loader checks path.exists())."""
    model_dir = tmp_path / "acme--m"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _stub_load_path(loader: PipelineLoader, pipeline: _FakePipeline) -> None:
    """Stub out the actual diffusers load + format detection."""
    loader._detect_load_format = MagicMock(return_value="safetensors")  # type: ignore[method-assign]

    async def _fake_load_from_pretrained(
        path: Any,
        pipeline_class: Any,
        custom_pipeline: Any,
        torch_dtype: Any,
        variant: Any,
        quant_attributes: Any = None,
    ) -> _FakePipeline:
        return pipeline

    loader._load_from_pretrained = _fake_load_from_pretrained  # type: ignore[method-assign]
    # Skip VAE optimization (the fake pipeline has vae=None anyway).
    loader._apply_vae_optimizations = MagicMock()  # type: ignore[method-assign]


def _run_load(
    *,
    loader: PipelineLoader,
    model_dir: Path,
    config: PipelineConfig,
    disk_size_gb: float,
    available_vram_gb: float,
    cuda_available: bool = False,
) -> None:
    """Drive PipelineLoader.load() with all the heavy machinery stubbed."""
    _install_torch_stub(cuda_available=cuda_available)
    with patch.object(loader_module, "estimate_model_size_gb", return_value=disk_size_gb), \
         patch.object(loader_module, "get_available_vram_gb", return_value=available_vram_gb), \
         patch.object(loader_module, "get_pipeline_class", return_value=(MagicMock(__name__="P"), None)), \
         patch.object(loader_module, "_check_torch_available", return_value=True), \
         patch.object(loader_module, "_check_diffusers_available", return_value=True), \
         patch.object(loader_module, "get_torch_dtype", return_value="torch.bfloat16"):
        asyncio.run(loader.load("acme/m", model_path=str(model_dir), config=config))


# ---------------------------------------------------------------------------
# Heuristic decisions.
# ---------------------------------------------------------------------------


def test_small_model_no_offload_enabled(tmp_path: Path) -> None:
    """A model that comfortably fits (peak << max_vram) should NOT trigger
    auto-offload."""
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    config = PipelineConfig(model_path=str(model_dir), device="cpu", warmup_steps=0)
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=4.0,
        available_vram_gb=20.0,
    )

    # 4GB * 1.4 = 5.6 < 20, and 4 < 20 * 0.7 = 14 → no offload.
    assert config.enable_model_cpu_offload is False
    assert config.enable_sequential_cpu_offload is False


def test_flux_class_model_triggers_sequential_offload(tmp_path: Path) -> None:
    """FLUX.2-klein-4b shape: 15GB on disk, 20GB usable VRAM.

    Old check: 15 > 20 → False → no offload → OOMs.
    New check: 15 >= 20*0.7 = 14 → enable SEQUENTIAL offload (aggressive,
    since the model crowds the GPU and even with model-level offload the
    transient buffers can still push past the limit).
    """
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    config = PipelineConfig(model_path=str(model_dir), device="cuda", warmup_steps=0)
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=15.0,
        available_vram_gb=20.0,
        cuda_available=True,
    )

    # 15 >= 20 * 0.7 = 14 → sequential offload wins.
    assert config.enable_sequential_cpu_offload is True
    assert pipe.sequential_cpu_offload_called is True
    # And the pipeline must NOT have been moved to cuda directly (see
    # test_offload_skips_to_cuda_call for the dedicated assertion).
    assert pipe.moved_to is None


def test_threshold_lower_edge_no_offload(tmp_path: Path) -> None:
    """Right at the lower edge of the sequential threshold: no offload."""
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    config = PipelineConfig(model_path=str(model_dir), device="cuda", warmup_steps=0)
    # disk = 13.99, max_vram = 20. sequential threshold = 14. 13.99 < 14.
    # peak = 13.99 * 1.4 = 19.59 < 20 → no model offload either.
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=13.99,
        available_vram_gb=20.0,
        cuda_available=False,
    )

    assert config.enable_sequential_cpu_offload is False
    assert config.enable_model_cpu_offload is False


def test_disk_size_exactly_at_threshold_triggers_sequential(tmp_path: Path) -> None:
    """disk_size == max_vram * 0.7 should trigger sequential (>= check)."""
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    config = PipelineConfig(model_path=str(model_dir), device="cuda", warmup_steps=0)
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=14.0,
        available_vram_gb=20.0,
        cuda_available=False,
    )

    assert config.enable_sequential_cpu_offload is True


def test_explicit_offload_in_config_respected(tmp_path: Path) -> None:
    """The auto-trigger should be a no-op when the caller already set
    offload flags explicitly."""
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    # Tiny model — wouldn't trigger auto-offload — but caller forces sequential.
    config = PipelineConfig(
        model_path=str(model_dir),
        device="cuda",
        warmup_steps=0,
        enable_sequential_cpu_offload=True,
    )
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=2.0,
        available_vram_gb=20.0,
        cuda_available=True,
    )

    assert config.enable_sequential_cpu_offload is True
    assert pipe.sequential_cpu_offload_called is True


# ---------------------------------------------------------------------------
# Order: offload must be applied BEFORE .to("cuda") and skip the move.
# ---------------------------------------------------------------------------


def test_offload_skips_to_cuda_call(tmp_path: Path) -> None:
    """When auto-offload triggers, the pipeline must NOT be `.to("cuda")`'d
    afterwards — the offload hook owns device placement and rejects
    pipelines that have already been moved. Pre-fix, the order was
    backwards (move then maybe offload, which OOM'd before the offload
    could apply)."""
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    config = PipelineConfig(model_path=str(model_dir), device="cuda", warmup_steps=0)
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=15.0,
        available_vram_gb=20.0,
        cuda_available=True,
    )

    # Sequential offload triggered — `.to("cuda")` must NOT have been called.
    assert config.enable_sequential_cpu_offload is True
    assert pipe.sequential_cpu_offload_called is True
    assert pipe.moved_to is None, (
        f"pipeline was moved to {pipe.moved_to!r} despite offload being active — "
        "diffusers rejects this combination"
    )


def test_no_offload_path_still_moves_to_cuda(tmp_path: Path) -> None:
    """The happy path (small model, no offload needed) must still call
    `.to('cuda')` so the pipeline ends up on GPU."""
    loader = _make_loader(max_vram_gb=20.0, tmp_path=tmp_path)
    model_dir = _make_model_dir(tmp_path)
    pipe = _FakePipeline()
    _stub_load_path(loader, pipe)

    config = PipelineConfig(model_path=str(model_dir), device="cuda", warmup_steps=0)
    _run_load(
        loader=loader,
        model_dir=model_dir,
        config=config,
        disk_size_gb=2.0,
        available_vram_gb=20.0,
        cuda_available=True,
    )

    assert config.enable_model_cpu_offload is False
    assert config.enable_sequential_cpu_offload is False
    assert pipe.moved_to == "cuda"
    assert pipe.model_cpu_offload_called is False
    assert pipe.sequential_cpu_offload_called is False
