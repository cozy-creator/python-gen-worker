"""fp8 storage flavors for transformers-BACKBONE snapshots (ie#478).

A sharded-transformers repo whose single root weight set IS the model
(pixel-space UiT like HiDream-O1: no VAE, no external text encoders) gets a
block-scoped fp8-E4M3 cast: only >=2-D ``.weight`` tensors living under a
repeated-block container (``.<idx>.`` path segment) that miss the skip
patterns are cast — a strict subset of the runtime block-window walk, so
everything stored fp8 is re-armed by any consumer. Multi-set singlefile
bundles still refuse (component identity is ambiguous).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from gen_worker.convert.convert import run_inline_conversion  # noqa: E402
from gen_worker.convert.writer import (  # noqa: E402
    ConversionImplementationError,
    streaming_fp8_snapshot,
)


def _backbone_snapshot(tmp_path: Path) -> Path:
    """Two root shards + index + config/tokenizer sidecars, HiDream-O1 shape."""
    src = tmp_path / "src"
    src.mkdir(parents=True)

    shard1 = {
        "model.embed_tokens.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.randn(8, dtype=torch.bfloat16),
    }
    shard2 = {
        "model.layers.1.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(8, dtype=torch.bfloat16),
        # 2-D, misses every skip pattern, but OUTSIDE any repeated block:
        # block scoping must keep it at source precision.
        "lm_head.weight": torch.randn(16, 8, dtype=torch.bfloat16),
    }
    save_file(shard1, src / "model-00001-of-00002.safetensors")
    save_file(shard2, src / "model-00002-of-00002.safetensors")
    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map |= {k: "model-00002-of-00002.safetensors" for k in shard2}
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps({"architectures": ["FakeUiT"]}))
    (src / "tokenizer_config.json").write_text("{}")
    return src


def _stored_dtypes(out_dir: Path) -> dict[str, str]:
    from safetensors import safe_open

    dtypes: dict[str, str] = {}
    for f in sorted(out_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            import struct

            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        for k, v in header.items():
            if k != "__metadata__":
                dtypes[k] = v["dtype"]
    return dtypes


CAST = {
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
}
KEPT = {
    "model.embed_tokens.weight",          # skip pattern: embed
    "model.layers.0.input_layernorm.weight",  # 1-D + norm
    "model.norm.weight",                  # 1-D + norm
    "lm_head.weight",                     # outside repeated blocks
}


def test_backbone_snapshot_block_scoped_cast(tmp_path: Path) -> None:
    src = _backbone_snapshot(tmp_path)
    out = tmp_path / "out"

    stats = streaming_fp8_snapshot(src, out, file_layout="singlefile")

    dtypes = _stored_dtypes(out)
    assert set(dtypes) == CAST | KEPT
    assert {k for k, v in dtypes.items() if v == "F8_E4M3"} == CAST
    for k in KEPT:
        assert dtypes[k] == "BF16"
    assert stats["converted_count"] == len(CAST)
    # Sidecars ride along; the source index is superseded by the re-shard.
    assert (out / "config.json").is_file()
    assert (out / "tokenizer_config.json").is_file()


def test_multi_weight_set_still_refuses(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    save_file({"a.0.weight": torch.randn(4, 4, dtype=torch.bfloat16)},
              src / "diffusion_model.safetensors")
    save_file({"b.0.weight": torch.randn(4, 4, dtype=torch.bfloat16)},
              src / "text_encoder.safetensors")

    with pytest.raises(ConversionImplementationError, match="weight set"):
        streaming_fp8_snapshot(src, tmp_path / "out", file_layout="singlefile")


def test_diffusers_lane_unchanged(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "transformer").mkdir(parents=True)
    save_file(
        {
            "blocks.0.attn.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "proj_out.weight": torch.randn(8, 8, dtype=torch.bfloat16),
        },
        src / "transformer" / "diffusion_pytorch_model.safetensors",
    )
    (src / "model_index.json").write_text("{}")

    streaming_fp8_snapshot(src, tmp_path / "out", file_layout="diffusers")
    dtypes = _stored_dtypes(tmp_path / "out" / "transformer")
    assert dtypes["blocks.0.attn.to_q.weight"] == "F8_E4M3"
    assert dtypes["proj_out.weight"] == "BF16"


def test_run_inline_conversion_block_scope_flag(tmp_path: Path) -> None:
    src = _backbone_snapshot(tmp_path)
    out = tmp_path / "out"

    result = run_inline_conversion(
        source_path=src / "model.safetensors.index.json",
        out_dir=out, target_dtype="fp8", fp8_block_scope=True,
    )

    assert result.attributes["dtype"] == "fp8"
    dtypes = _stored_dtypes(out)
    assert {k for k, v in dtypes.items() if v == "F8_E4M3"} == CAST
    assert dtypes["lm_head.weight"] == "BF16"
