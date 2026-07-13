"""fp8-storage consumption + the emergency nf4 rung in the loading layer
(th#546 two-format policy / gw#389).

Real diffusers/CUDA are absent in CI — layerwise casting and the bnb config
handoff are tested against fakes; live measurements are the GPU matrix
campaign."""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker.models.loading import (
    apply_fp8_storage,
    detect_on_disk_dtype,
    emergency_quant_enabled,
    load_from_pretrained,
    snapshot_weight_bytes,
)


# --------------------------------------------------------------------------
# fakes / fixtures
# --------------------------------------------------------------------------


class _FakeDenoiser:
    def __init__(self) -> None:
        self.casting_calls: list = []

    def parameters(self):
        return iter(())

    def enable_layerwise_casting(self, *, storage_dtype: Any, compute_dtype: Any) -> None:
        self.casting_calls.append((storage_dtype, compute_dtype))


class _FakeDiffusionPipeline:
    pass


class _Pipe(_FakeDiffusionPipeline):
    calls: list = []

    def __init__(self) -> None:
        self.transformer = _FakeDenoiser()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
        cls.calls.append(kwargs)
        return cls()


class _FakePipelineQuantizationConfig:
    def __init__(self, quant_backend: str, quant_kwargs: Dict[str, Any],
                 components_to_quantize: list) -> None:
        self.quant_backend = quant_backend
        self.quant_kwargs = quant_kwargs
        self.components_to_quantize = components_to_quantize


@pytest.fixture
def fake_diffusers(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    root = types.ModuleType("diffusers")
    quantizers = types.ModuleType("diffusers.quantizers")
    qc_mod = types.ModuleType("diffusers.quantizers.quantization_config")
    root.DiffusionPipeline = _FakeDiffusionPipeline
    quantizers.PipelineQuantizationConfig = _FakePipelineQuantizationConfig
    qc_mod.BitsAndBytesConfig = lambda **kw: ("bnb", kw)
    root.quantizers = quantizers
    quantizers.quantization_config = qc_mod
    monkeypatch.setitem(sys.modules, "diffusers", root)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.quantization_config", qc_mod)
    # the nf4 rung is gated on bitsandbytes availability (gw#469)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    return root


def _write_safetensors(path: Path, dtype: str, nbytes: int) -> None:
    """Header-only safetensors: detect/size helpers read headers, not data."""
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [nbytes], "data_offsets": [0, nbytes]}}
    ).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def _snapshot(tmp_path: Path, dtype: str = "BF16", nbytes: int = 1024) -> Path:
    """Realistic component layout (gw#521): the fit ladder derives quant
    targets from the tree's REAL component names, so the denoiser weights
    live under transformer/ and model_index.json declares it."""
    (tmp_path / "model_index.json").write_text(
        '{"_class_name": "Pipe", "transformer": ["diffusers", "X"]}')
    (tmp_path / "transformer").mkdir(exist_ok=True)
    _write_safetensors(
        tmp_path / "transformer" / "diffusion_pytorch_model.safetensors",
        dtype, nbytes)
    return tmp_path


# --------------------------------------------------------------------------
# fp8 storage
# --------------------------------------------------------------------------


def test_detect_on_disk_dtype_reads_fp8(tmp_path: Path) -> None:
    assert detect_on_disk_dtype(_snapshot(tmp_path, "F8_E4M3")) == "fp8"


def test_apply_fp8_storage_targets_denoiser() -> None:
    import torch

    pipe = _Pipe()
    assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16) is True
    ((storage, compute),) = pipe.transformer.casting_calls
    assert storage is torch.float8_e4m3fn
    assert compute is torch.bfloat16


def test_apply_fp8_storage_on_bare_module_defaults_bf16() -> None:
    import torch

    mod = _FakeDenoiser()
    assert apply_fp8_storage(mod) is True
    ((storage, compute),) = mod.casting_calls
    assert storage is torch.float8_e4m3fn
    assert compute is torch.bfloat16


def test_binding_storage_dtype_applies_fp8(tmp_path: Path) -> None:
    import torch

    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, _snapshot(tmp_path), dtype="bf16",
                                storage_dtype="fp8")
    (kwargs,) = _Pipe.calls
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert "quantization_config" not in kwargs
    ((storage, compute),) = pipe.transformer.casting_calls
    assert storage is torch.float8_e4m3fn
    assert compute is torch.bfloat16


def test_fp8_stored_flavor_auto_preserved(tmp_path: Path) -> None:
    """An #fp8 flavor snapshot loads at bf16 compute and gets its storage
    precision restored — never silently upcast into 2x the VRAM."""
    import torch

    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, _snapshot(tmp_path, "F8_E4M3"))
    (kwargs,) = _Pipe.calls
    assert kwargs["torch_dtype"] is torch.bfloat16
    ((storage, _compute),) = pipe.transformer.casting_calls
    assert storage is torch.float8_e4m3fn


def test_no_storage_dtype_means_no_casting(tmp_path: Path) -> None:
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, _snapshot(tmp_path), dtype="bf16")
    assert pipe.transformer.casting_calls == []


# --------------------------------------------------------------------------
# emergency nf4 rung
# --------------------------------------------------------------------------


@pytest.fixture
def emergency_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA host with 10GB free — the rung is automatic there (gw#420)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    from gen_worker.models import memory

    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 10.0)


def test_snapshot_weight_bytes_reads_headers(tmp_path: Path) -> None:
    assert snapshot_weight_bytes(_snapshot(tmp_path, "BF16", 4096)) == 4096


def test_emergency_always_on_cuda_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    # gw#420: no env flag — the rung is armed iff the host has CUDA.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert emergency_quant_enabled() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert emergency_quant_enabled() is True


def test_emergency_rung_engages_when_flavor_cannot_fit(
    fake_diffusers: Any, emergency_on: None, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The canonical case: a 24GB fp8 flavor on a 10GB-free card — the rung
    quantizes the denoiser to nf4 with the loud warning."""
    snap = _snapshot(tmp_path, "F8_E4M3", 24 << 30)
    _Pipe.calls = []
    with caplog.at_level("WARNING"):
        pipe = load_from_pretrained(_Pipe, snap)
    (kwargs,) = _Pipe.calls
    qc = kwargs["quantization_config"]
    assert isinstance(qc, _FakePipelineQuantizationConfig)
    assert qc.quant_backend == "bitsandbytes_4bit"
    assert qc.quant_kwargs["bnb_4bit_quant_type"] == "nf4"
    # gw#521: the target is the snapshot's REAL denoiser, never a guess.
    assert qc.components_to_quantize == ["transformer"]
    assert "EMERGENCY 4-bit quantization" in caplog.text
    assert "below platform standards" in caplog.text
    # nf4 supersedes the fp8-storage rung — no layerwise casting on top
    assert pipe.transformer.casting_calls == []


def test_emergency_rung_counts_planned_fp8_halving(
    fake_diffusers: Any, emergency_on: None, tmp_path: Path,
) -> None:
    """A 14GB bf16 snapshot with storage_dtype=fp8 -> ~7GB resident fits
    10GB free: fp8 rung wins, emergency stays out."""
    snap = _snapshot(tmp_path, "BF16", 14 << 30)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap, dtype="bf16", storage_dtype="fp8")
    (kwargs,) = _Pipe.calls
    assert "quantization_config" not in kwargs
    assert len(pipe.transformer.casting_calls) == 1


def test_fp8_storage_rung_engages_before_nf4(
    fake_diffusers: Any, emergency_on: None, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """th#683 adaptive fit: a 14GB bf16 snapshot on a 10GB-free card doesn't
    fit as stored, but halved by fp8-E4M3 storage (~7GB) it does — the fp8
    rung engages INSTEAD of dropping straight to nf4."""
    snap = _snapshot(tmp_path, "BF16", 14 << 30)
    _Pipe.calls = []
    with caplog.at_level("WARNING"):
        pipe = load_from_pretrained(_Pipe, snap, dtype="bf16")
    (kwargs,) = _Pipe.calls
    assert "quantization_config" not in kwargs
    ((storage, _compute),) = pipe.transformer.casting_calls
    import torch

    assert storage is torch.float8_e4m3fn
    assert "fp8-E4M3 emergency weight storage engaged" in caplog.text


def test_nf4_rung_engages_when_even_fp8_estimate_cannot_fit(
    fake_diffusers: Any, emergency_on: None, tmp_path: Path,
) -> None:
    """A 20GB bf16 snapshot on a 10GB-free card: halved (~10GB) still exceeds
    the 8GB budget -> the nf4 emergency rung, not fp8 storage."""
    snap = _snapshot(tmp_path, "BF16", 20 << 30)
    _Pipe.calls = []
    pipe = load_from_pretrained(_Pipe, snap, dtype="bf16")
    (kwargs,) = _Pipe.calls
    qc = kwargs["quantization_config"]
    assert isinstance(qc, _FakePipelineQuantizationConfig)
    assert qc.quant_kwargs["bnb_4bit_quant_type"] == "nf4"
    assert pipe.transformer.casting_calls == []


def test_nf4_rung_skipped_without_bitsandbytes(
    fake_diffusers: Any, emergency_on: None, tmp_path: Path,
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gw#469: bitsandbytes absent from the endpoint image — the nf4 rung is
    SKIPPED with a logged reason (the offload ladder carries the load), never
    attempted into a PackageNotFoundError setup_failed."""
    import importlib.util
    import sys

    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "bitsandbytes"
                        else importlib.util._bootstrap._find_spec(name, None))
    snap = _snapshot(tmp_path, "BF16", 20 << 30)  # nf4 territory (fp8 can't fit)
    _Pipe.calls = []
    with caplog.at_level("WARNING"):
        load_from_pretrained(_Pipe, snap, dtype="bf16")
    (kwargs,) = _Pipe.calls
    assert "quantization_config" not in kwargs, "unavailable rung was attempted"
    assert "bitsandbytes not installed" in caplog.text


def test_emergency_rung_stays_out_without_cuda(
    fake_diffusers: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    snap = _snapshot(tmp_path, "F8_E4M3", 27 << 30)
    _Pipe.calls = []
    load_from_pretrained(_Pipe, snap)
    (kwargs,) = _Pipe.calls
    assert "quantization_config" not in kwargs


# --------------------------------------------------------------------------
# fit ladder (hub_policy)
# --------------------------------------------------------------------------


def test_variant_fit_runtime_rung_verdicts() -> None:
    # gw#420: the rungs are automatic on CUDA hosts — no flag anywhere.
    from gen_worker.api import Resources
    from gen_worker.models.hub_policy import (
        FIT_EMERGENCY,
        FIT_EMERGENCY_FP8,
        FIT_OFFLOAD,
        TensorhubWorkerCapabilities,
        variant_fit,
    )

    caps = TensorhubWorkerCapabilities(
        cuda_version="12.8", gpu_sm=89, torch_version="2.11", installed_libs=[])
    res = Resources(vram_gb=34)  # klein-9b-class on a 24GB card, 20 free

    # 34*0.55=18.7 <= 20 -> the fp8-storage rung outranks nf4 (th#683).
    fit, reason = variant_fit(res, caps, 20.0)
    assert fit == FIT_EMERGENCY_FP8
    assert "fp8" in reason
    # 40*0.55=22 > 20 but 40*0.45=18 <= 20 -> the nf4 emergency rung.
    fit, reason = variant_fit(Resources(vram_gb=40), caps, 20.0)
    assert fit == FIT_EMERGENCY
    assert "emergency quality" in reason
    # 4-bit estimate still too big -> offload even on a CUDA host
    assert variant_fit(Resources(vram_gb=60), caps, 20.0)[0] == FIT_OFFLOAD
