"""ie681: `streaming_w8a8_snapshot` forwards the module-path skip patterns.

`streaming_w8a8_cast` has taken `skip_patterns` since gw#557, but the
SNAPSHOT-level entry point the conversion endpoint actually invokes did not
forward them, so every produce run was locked to the architecture-agnostic
default set. That default knows `adaln_single` and not MiniMax-H3's
`adaln_proj`, so a bare `cast-dtype dtypes=["w8a8"]` on H3 quantizes 50
modulation projections (26.01 GB, 39% of the DiT) that H3's own serve recipe
keeps bf16 and that te#171's AdaLN-skip precompute consumes as bf16.

Model-specific selection is INVOKE data (the `weight_set_patterns`
precedent), never a name added to the shared default — so the fix is the
parameter, not a wider default. RED before it: the `adaln_proj` weight comes
back F8_E4M3 with a `weight_scale` twin.

No mocks: the real producer over real safetensors, headers read back off
disk.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from safetensors.torch import save_file

from gen_worker.convert.writer import (
    W8A8_SKIP_TENSOR_PATTERNS,
    streaming_w8a8_snapshot,
)


def _header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def _h3_shaped_denoiser(tmp_path: Path) -> Path:
    """A diffusers tree whose `transformer` carries H3's module names: an
    attention projection (must quantize) and an `adaln_proj.linear`
    (modulation — must be skippable by invoke data)."""
    root = tmp_path / "src"
    (root / "transformer").mkdir(parents=True)
    save_file(
        {
            "transformer_blocks.0.attn.to_q.weight": torch.randn(32, 32, dtype=torch.bfloat16),
            "transformer_blocks.0.adaln_proj.linear.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "transformer_blocks.0.adaln_proj.linear.bias": torch.randn(32, dtype=torch.bfloat16),
            "transformer_blocks.0.norm1.weight": torch.randn(32, dtype=torch.bfloat16),
        },
        str(root / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    (root / "transformer" / "config.json").write_text(json.dumps({"_class_name": "X"}))
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "MiniMaxH3ModularPipeline",
        "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
    }))
    return root


def _produce(tmp_path: Path, name: str, **kw) -> dict:
    out = tmp_path / name
    streaming_w8a8_snapshot(
        _h3_shaped_denoiser(tmp_path / name / "in"), out,
        file_layout="multi-file", components=("transformer",), **kw)
    return _header(out / "transformer" / "diffusion_pytorch_model.safetensors")


def test_default_patterns_quantize_the_modulation_projection(tmp_path: Path) -> None:
    """The premise, measured rather than asserted: the shared default set does
    NOT protect `adaln_proj`. This is why the parameter has to exist."""
    header = _produce(tmp_path, "default")
    assert header["transformer_blocks.0.attn.to_q.weight"]["dtype"] == "F8_E4M3"
    assert header["transformer_blocks.0.adaln_proj.linear.weight"]["dtype"] == "F8_E4M3"
    assert "transformer_blocks.0.adaln_proj.linear.weight_scale" in header


def test_invoke_supplied_pattern_keeps_the_modulation_projection_bf16(tmp_path: Path) -> None:
    header = _produce(
        tmp_path, "skipped",
        skip_patterns=W8A8_SKIP_TENSOR_PATTERNS + ("adaln",))
    # The attention projection still quantizes...
    assert header["transformer_blocks.0.attn.to_q.weight"]["dtype"] == "F8_E4M3"
    assert "transformer_blocks.0.attn.to_q.weight_scale" in header
    # ...and the modulation projection is untouched, with NO orphan scale.
    assert header["transformer_blocks.0.adaln_proj.linear.weight"]["dtype"] == "BF16"
    assert "transformer_blocks.0.adaln_proj.linear.weight_scale" not in header
    # Passthrough tensors keep source precision either way.
    assert header["transformer_blocks.0.norm1.weight"]["dtype"] == "BF16"
    assert header["transformer_blocks.0.adaln_proj.linear.bias"]["dtype"] == "BF16"
