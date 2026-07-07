"""Binding-level quantize= plumbing in the loading layer (#389).

Real diffusers/CUDA are absent in CI — the from_pretrained handoff is tested
against a faked diffusers module; real quantized loads are proven in the GPU
matrix campaign (th#546)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker.models.loading import (
    binding_quantization_config,
    load_from_pretrained,
)


class _FakePipelineQuantizationConfig:
    def __init__(self, quant_backend: str, quant_kwargs: Dict[str, Any],
                 components_to_quantize: list) -> None:
        self.quant_backend = quant_backend
        self.quant_kwargs = quant_kwargs
        self.components_to_quantize = components_to_quantize


class _FakeComponentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeDiffusionPipeline:
    pass


@pytest.fixture
def fake_diffusers(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    root = types.ModuleType("diffusers")
    quantizers = types.ModuleType("diffusers.quantizers")
    qc_mod = types.ModuleType("diffusers.quantizers.quantization_config")
    root.DiffusionPipeline = _FakeDiffusionPipeline
    quantizers.PipelineQuantizationConfig = _FakePipelineQuantizationConfig
    qc_mod.BitsAndBytesConfig = _FakeComponentConfig
    qc_mod.TorchAoConfig = _FakeComponentConfig
    root.quantizers = quantizers
    quantizers.quantization_config = qc_mod
    monkeypatch.setitem(sys.modules, "diffusers", root)
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", quantizers)
    monkeypatch.setitem(
        sys.modules, "diffusers.quantizers.quantization_config", qc_mod
    )
    return root


@pytest.fixture
def cuda_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


class _Pipe(_FakeDiffusionPipeline):
    calls: list = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
        cls.calls.append(kwargs)
        return cls()


def _snapshot(tmp_path: Path, quantized: bool = False) -> Path:
    index: Dict[str, Any] = {"_class_name": "Pipe"}
    if quantized:
        index["quantization_config"] = {"quant_method": "bitsandbytes"}
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    return tmp_path


def test_no_cuda_falls_back_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("CPU-only path")
    with caplog.at_level("WARNING"):
        assert binding_quantization_config("nf4", _Pipe) is None
    assert "CUDA unavailable" in caplog.text


def test_pipeline_config_synthesis(fake_diffusers: Any, cuda_on: None) -> None:
    import torch

    qc = binding_quantization_config("nf4", _Pipe)
    assert isinstance(qc, _FakePipelineQuantizationConfig)
    assert qc.quant_backend == "bitsandbytes_4bit"
    assert qc.quant_kwargs["bnb_4bit_quant_type"] == "nf4"
    assert qc.quant_kwargs["bnb_4bit_compute_dtype"] is torch.bfloat16
    assert "transformer" in qc.components_to_quantize

    qc8 = binding_quantization_config("int8", _Pipe)
    assert qc8.quant_backend == "bitsandbytes_8bit"
    assert qc8.quant_kwargs == {"load_in_8bit": True}

    for method, quant_type in (
        ("int8-torchao", "int8wo"),
        ("int4-torchao", "int4wo"),
        ("fp8-torchao", "float8wo_e4m3"),
    ):
        q = binding_quantization_config(method, _Pipe)
        assert q.quant_backend == "torchao"
        assert q.quant_kwargs == {"quant_type": quant_type}


def test_component_class_gets_component_config(
    fake_diffusers: Any, cuda_on: None
) -> None:
    class NotAPipeline:
        pass

    qc = binding_quantization_config("nf4", NotAPipeline)
    assert isinstance(qc, _FakeComponentConfig)
    assert qc.kwargs["load_in_4bit"] is True
    qao = binding_quantization_config("fp8-torchao", NotAPipeline)
    assert qao.kwargs == {"quant_type": "float8wo_e4m3"}


def test_load_injects_binding_quantization_config(
    fake_diffusers: Any, cuda_on: None, tmp_path: Path
) -> None:
    _Pipe.calls = []
    load_from_pretrained(_Pipe, _snapshot(tmp_path), dtype="bf16", quantize="nf4")
    (kwargs,) = _Pipe.calls
    qc = kwargs["quantization_config"]
    assert isinstance(qc, _FakePipelineQuantizationConfig)
    assert qc.quant_backend == "bitsandbytes_4bit"


def test_on_disk_quant_config_wins_over_binding_quantize(
    fake_diffusers: Any, cuda_on: None, tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _Pipe.calls = []
    with caplog.at_level("WARNING"):
        load_from_pretrained(
            _Pipe, _snapshot(tmp_path, quantized=True), quantize="nf4"
        )
    (kwargs,) = _Pipe.calls
    assert "quantization_config" not in kwargs
    assert "already carries" in caplog.text


def test_binding_quantize_is_never_silently_dropped(
    fake_diffusers: Any, cuda_on: None, tmp_path: Path
) -> None:
    class Rejects(_FakeDiffusionPipeline):
        calls: list = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            cls.calls.append(kwargs)
            if "quantization_config" in kwargs:
                raise TypeError("no quantization_config here")
            return cls()

    with pytest.raises(TypeError):
        load_from_pretrained(Rejects, _snapshot(tmp_path), quantize="int8")
    # both attempts carried the config — the retry didn't strip it
    assert all("quantization_config" in k for k in Rejects.calls)

    # without quantize, the legacy retry still strips loader-rejected kwargs
    Rejects.calls = []
    out = load_from_pretrained(Rejects, _snapshot(tmp_path))
    assert isinstance(out, Rejects)


def test_no_quantize_means_no_config(tmp_path: Path) -> None:
    _Pipe.calls = []
    load_from_pretrained(_Pipe, _snapshot(tmp_path))
    (kwargs,) = _Pipe.calls
    assert "quantization_config" not in kwargs
