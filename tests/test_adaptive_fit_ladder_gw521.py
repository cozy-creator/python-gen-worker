"""gw#521: the emergency-quant rung targets the snapshot's REAL components.

Root-caused on the 4070 dogfood: the rung's PipelineQuantizationConfig named
components the pipeline didn't have, diffusers silently ignored them, and the
"EMERGENCY 4-bit quantization engaged" line was a lie on every model — the
worker then fell to (forbidden) CPU offload. These tests pin the ladder-core
contract with fakes; tests/test_emergency_quant_lands_gw521.py proves the
quant lands on real tiny pipelines of both archetypes (CUDA + bnb).

Also here: offload-hooked pipelines book RAM tier in residency (the 0.03GB
bogus-VRAM registration facet).
"""

from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker.models.loading import (
    _adaptive_fit_rung,
    emergency_quantization_config,
    model_index_components,
    snapshot_component_weight_bytes,
)

_GiB = 1 << 30


class _FakeDiffusionPipeline:
    pass


class _Pipe(_FakeDiffusionPipeline):
    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
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
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    return root


@pytest.fixture
def cuda_10gb_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA host with 10GB free -> an 8GB emergency budget (2GB margin)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    from gen_worker.models import memory

    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 10.0)


def _write_safetensors(path: Path, dtype: str, nbytes: int) -> None:
    header = json.dumps(
        {"w": {"dtype": dtype, "shape": [nbytes], "data_offsets": [0, nbytes]}}
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def _tree(tmp_path: Path, components: Dict[str, int], dtype: str = "BF16") -> Path:
    index: Dict[str, Any] = {"_class_name": "Pipe"}
    for name, nbytes in components.items():
        index[name] = ["diffusers", "X"]
        _write_safetensors(
            tmp_path / name / "diffusion_pytorch_model.safetensors", dtype, nbytes)
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    return tmp_path


# --------------------------------------------------------------------------
# target derivation: the snapshot's real names, never an archetype guess
# --------------------------------------------------------------------------


def test_unet_archetype_targets_unet(
    fake_diffusers: Any, cuda_10gb_free: None, tmp_path: Path,
) -> None:
    # SDXL shape: 20GB unet + 1GB TE. nf4(unet) = 21 - 14 = 7GB <= 8GB budget.
    snap = _tree(tmp_path, {"unet": 20 * _GiB, "text_encoder": 1 * _GiB})
    mode, qc = _adaptive_fit_rung(_Pipe, snap, fp8_planned=True)
    assert mode == "nf4"
    assert qc.components_to_quantize == ["unet"]


def test_transformer_archetype_targets_transformer(
    fake_diffusers: Any, cuda_10gb_free: None, tmp_path: Path,
) -> None:
    snap = _tree(tmp_path, {"transformer": 20 * _GiB, "text_encoder": 1 * _GiB})
    mode, qc = _adaptive_fit_rung(_Pipe, snap, fp8_planned=True)
    assert mode == "nf4"
    assert qc.components_to_quantize == ["transformer"]


def test_text_encoders_join_only_when_denoiser_alone_is_not_enough(
    fake_diffusers: Any, cuda_10gb_free: None, tmp_path: Path,
) -> None:
    # klein shape: 9GB transformer + 7GB TE. nf4(denoiser) = 16 - 6.3 = 9.7GB
    # > 8GB budget; +TE = 9.7 - 4.9 = 4.8GB fits -> both quantized.
    snap = _tree(tmp_path, {"transformer": 9 * _GiB, "text_encoder": 7 * _GiB})
    mode, qc = _adaptive_fit_rung(_Pipe, snap, fp8_planned=True)
    assert mode == "nf4"
    assert qc.components_to_quantize == ["transformer", "text_encoder"]


def test_rung_skipped_when_even_nf4_cannot_fit(
    fake_diffusers: Any, cuda_10gb_free: None, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 30GB denoiser: nf4 est 9GB > 8GB budget -> keep full precision (the
    # offload ladder carries it) instead of paying quality for nothing.
    snap = _tree(tmp_path, {"transformer": 30 * _GiB})
    with caplog.at_level("WARNING"):
        mode, qc = _adaptive_fit_rung(_Pipe, snap, fp8_planned=True)
    assert (mode, qc) == ("", None)
    assert "even 4-bit weights" in caplog.text


def test_rung_skipped_on_denoiser_less_tree(
    fake_diffusers: Any, cuda_10gb_free: None, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    snap = _tree(tmp_path, {"vae": 20 * _GiB})
    with caplog.at_level("WARNING"):
        mode, qc = _adaptive_fit_rung(_Pipe, snap, fp8_planned=True)
    assert (mode, qc) == ("", None)
    assert "no denoiser" in caplog.text


def test_emergency_config_refuses_empty_components(
    fake_diffusers: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert emergency_quantization_config(_Pipe, components=[]) is None


def test_component_helpers_read_the_tree(tmp_path: Path) -> None:
    snap = _tree(tmp_path, {"unet": 4096, "vae": 1024})
    assert model_index_components(snap) == {"unet", "vae"}
    assert snapshot_component_weight_bytes(snap) == {"unet": 4096, "vae": 1024}


# --------------------------------------------------------------------------
# residency: offload-hooked pipelines never book bogus VRAM (0.03GB live)
# --------------------------------------------------------------------------


class _HookedPipe:
    _cozy_low_vram_mode = "group_offload"

    def to(self, *a: Any, **k: Any) -> "_HookedPipe":
        return self


def test_track_vram_books_hooked_pipeline_in_ram_tier() -> None:
    from gen_worker.models.residency import IN_RAM, Residency, Tier

    events: list = []
    res = Residency(on_event=lambda *e: events.append(e))
    res.track_vram("acme/model", _HookedPipe(), vram_bytes=32 << 20)  # 0.03GB lie
    assert res.tier("acme/model") is Tier.RAM
    assert res.vram_bytes("acme/model") == 0
    assert events[-1][:3] == ("acme/model", IN_RAM, 0)


def test_track_vram_unhooked_still_books_vram() -> None:
    from gen_worker.models.residency import IN_VRAM, Residency, Tier

    class _Plain:
        def to(self, *a: Any, **k: Any) -> "_Plain":
            return self

    events: list = []
    res = Residency(on_event=lambda *e: events.append(e))
    res.track_vram("acme/model", _Plain(), vram_bytes=3 << 30)
    assert res.tier("acme/model") is Tier.VRAM
    assert res.vram_bytes("acme/model") == 3 << 30
    assert events[-1][:3] == ("acme/model", IN_VRAM, 3 << 30)
