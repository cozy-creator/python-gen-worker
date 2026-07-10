"""Published cast trees carry canonical diffusers filenames (no variant
tokens). Found live (J23 cloud validation): the passthrough+reshard path
published juggernaut-xl's sharded unet index as
``diffusion_pytorch_model.fp16.safetensors.index.json`` — diffusers'
``_add_variant`` looks for ``diffusion_pytorch_model.safetensors.index.fp16.json``,
so ``variant="fp16"`` serve loads failed."""

from __future__ import annotations

import json
from pathlib import Path

from gen_worker.convert.clone import _normalize_variant_filenames


def _mk(tree: Path, rel: str, data: bytes = b"x") -> Path:
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


class TestNormalizeVariantFilenames:
    def test_sharded_variant_unet_and_index(self, tmp_path: Path) -> None:
        _mk(tmp_path, "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors")
        _mk(tmp_path, "unet/diffusion_pytorch_model.fp16-00002-of-00002.safetensors")
        idx = _mk(
            tmp_path, "unet/diffusion_pytorch_model.fp16.safetensors.index.json",
            json.dumps({
                "metadata": {"total_size": 2},
                "weight_map": {
                    "a": "diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
                    "b": "diffusion_pytorch_model.fp16-00002-of-00002.safetensors",
                },
            }).encode(),
        )
        del idx
        _normalize_variant_filenames(tmp_path)
        unet = tmp_path / "unet"
        assert (unet / "diffusion_pytorch_model-00001-of-00002.safetensors").exists()
        assert (unet / "diffusion_pytorch_model-00002-of-00002.safetensors").exists()
        canon_idx = unet / "diffusion_pytorch_model.safetensors.index.json"
        assert canon_idx.exists()
        assert not (unet / "diffusion_pytorch_model.fp16.safetensors.index.json").exists()
        wm = json.loads(canon_idx.read_text())["weight_map"]
        assert wm == {
            "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
            "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
        }

    def test_plain_variant_files(self, tmp_path: Path) -> None:
        _mk(tmp_path, "text_encoder/model.fp16.safetensors")
        _mk(tmp_path, "vae/diffusion_pytorch_model.bf16.safetensors")
        _normalize_variant_filenames(tmp_path)
        assert (tmp_path / "text_encoder/model.safetensors").exists()
        assert (tmp_path / "vae/diffusion_pytorch_model.safetensors").exists()

    def test_canonical_names_untouched(self, tmp_path: Path) -> None:
        _mk(tmp_path, "unet/diffusion_pytorch_model.safetensors")
        _mk(tmp_path, "unet/config.json")
        _normalize_variant_filenames(tmp_path)
        assert (tmp_path / "unet/diffusion_pytorch_model.safetensors").exists()

    def test_canonical_twin_blocks_directory(self, tmp_path: Path) -> None:
        # A dual-dtype tree must not collapse: leave the whole dir alone.
        _mk(tmp_path, "unet/diffusion_pytorch_model.safetensors", b"fp32")
        _mk(tmp_path, "unet/diffusion_pytorch_model.fp16.safetensors", b"fp16")
        _normalize_variant_filenames(tmp_path)
        assert (tmp_path / "unet/diffusion_pytorch_model.fp16.safetensors").exists()
        assert (tmp_path / "unet/diffusion_pytorch_model.safetensors").read_bytes() == b"fp32"

    def test_quant_tokens_not_matched(self, tmp_path: Path) -> None:
        _mk(tmp_path, "transformer/diffusion_pytorch_model.fp8.safetensors")
        _normalize_variant_filenames(tmp_path)
        assert (tmp_path / "transformer/diffusion_pytorch_model.fp8.safetensors").exists()
