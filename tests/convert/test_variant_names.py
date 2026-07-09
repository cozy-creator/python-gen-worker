"""Variant-named mirrors (repo-cas clones with a dtype preference keep HF's
``.fp16.``/``.bf16.`` filename suffix) must load through every consumer.
Found live: e2e J9 repackage-singlefile went FATAL on an fp16-named sd15
mirror because _load_component_state_dict only tried exact names."""

from __future__ import annotations

from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from gen_worker.convert.repackage import _load_component_state_dict
from gen_worker.convert.source import Source


def _tensors() -> dict[str, torch.Tensor]:
    return {"w": torch.zeros(2, 2)}


class TestLoadComponentStateDictVariants:
    def test_exact_name_still_preferred(self, tmp_path: Path) -> None:
        save_file(_tensors(), str(tmp_path / "diffusion_pytorch_model.safetensors"))
        save_file({"v": torch.ones(1)}, str(tmp_path / "diffusion_pytorch_model.fp16.safetensors"))
        sd = _load_component_state_dict(tmp_path, safetensors_bases=["diffusion_pytorch_model"])
        assert "w" in sd

    def test_fp16_variant_name(self, tmp_path: Path) -> None:
        save_file(_tensors(), str(tmp_path / "diffusion_pytorch_model.fp16.safetensors"))
        sd = _load_component_state_dict(tmp_path, safetensors_bases=["diffusion_pytorch_model"])
        assert "w" in sd

    def test_bf16_variant_text_encoder_base(self, tmp_path: Path) -> None:
        save_file(_tensors(), str(tmp_path / "model.bf16.safetensors"))
        sd = _load_component_state_dict(tmp_path, safetensors_bases=["model", "pytorch_model"])
        assert "w" in sd

    def test_missing_still_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_component_state_dict(tmp_path, safetensors_bases=["diffusion_pytorch_model"])


class TestSourceDiffusersVariant:
    def _diffusers_tree(self, tmp_path: Path, weight_name: str) -> Path:
        root = tmp_path / "model"
        (root / "unet").mkdir(parents=True)
        (root / "model_index.json").write_text("{}")
        save_file(_tensors(), str(root / "unet" / weight_name))
        return root

    def test_detects_fp16(self, tmp_path: Path) -> None:
        root = self._diffusers_tree(tmp_path, "diffusion_pytorch_model.fp16.safetensors")
        assert Source(root).diffusers_variant() == "fp16"

    def test_plain_names_no_variant(self, tmp_path: Path) -> None:
        root = self._diffusers_tree(tmp_path, "diffusion_pytorch_model.safetensors")
        assert Source(root).diffusers_variant() is None
