"""A component override must load at the dtype the base tree's LOAD LANE
computes at — never at the tree's MAJORITY on-disk dtype.

A `#fp8-w8a8` flavor quantizes only the repeated-block Linears (F8_E4M3 + F32
scale twins) and passes every other tensor through at SOURCE precision, so a
tree built from an fp16-stored fine-tune header-counts majority-F16 while
`load_w8a8_pipeline` loads the composition at its bf16 compute default. A
majority sniff then loads a Half override into a bf16-activation pipeline and
every warm/serve forward dies with::

    RuntimeError: Input type (c10::BFloat16) and bias type (c10::Half)
    should be the same

These tapes rebuild that composition for real on CPU: the REAL producer (`streaming_w8a8_cast`) makes the w8a8 tree from an
fp16 source, the REAL `load_component_override` loads a REAL tiny diffusers
`AutoencoderKL` override, and the decode forward runs bf16 latents through
it — the exact crash path.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from gen_worker.convert.writer import streaming_w8a8_cast
from gen_worker.models.loading import (
    QUANT_EXECUTION_LANE_COMPUTE_DEFAULT,
    composition_compute_dtype,
    load_component_override,
)

# ---------------------------------------------------------------------------
# Fixture: a majority-fp16 #fp8-w8a8 base tree, built by the REAL producer.
# ---------------------------------------------------------------------------

_TINY_VAE_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",),
    up_block_types=("UpDecoderBlock2D",),
    block_out_channels=(4,),
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=4,
    sample_size=8,
)


def _w8a8_base_tree(tmp_path: Path) -> Path:
    """Diffusers root whose unet is a REAL streaming_w8a8_cast product of an
    fp16 source: one eligible repeated-block Linear becomes F8+F32-scale;
    enough passthrough tensors stay F16 that F16 is the tree majority —
    the exact shape of the production `#fp8-w8a8` sdxl mirrors."""
    src = tmp_path / "src-unet"
    src.mkdir()
    tensors = {
        # Eligible: 2-D .weight, 16-aligned, inside a `.N.` block container.
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight":
            torch.randn(16, 16, dtype=torch.float16),
    }
    # Passthrough majority (norms/biases/convs keep source precision).
    for i in range(6):
        tensors[f"passthrough.norm_{i}.weight"] = torch.randn(
            8, dtype=torch.float16)
    save_file(tensors, str(src / "model.safetensors"))

    root = tmp_path / "base-w8a8"
    out = streaming_w8a8_cast(src / "model.safetensors", root / "unet")
    assert int(out["converted_count"]) == 1
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    return root


def _fp32_vae_override(tmp_path: Path) -> Path:
    """A REAL tiny AutoencoderKL saved fp32 — the sdxl-vae-fp16-fix mirror is
    fp32-stored (334MB on the wire), so the override's own on-disk dtype must
    never win either."""
    override = tmp_path / "override-vae"
    vae = AutoencoderKL(**_TINY_VAE_CONFIG)
    vae.save_pretrained(str(override))
    return override


# ---------------------------------------------------------------------------
# The tapes
# ---------------------------------------------------------------------------


def test_composition_compute_dtype_is_execution_lane_aware_on_w8a8(
    tmp_path: Path,
) -> None:
    """RED pre-fix: the majority sniff returned "fp16" for a w8a8 tree whose
    lane computes bf16."""
    root = _w8a8_base_tree(tmp_path)
    assert composition_compute_dtype(root) == QUANT_EXECUTION_LANE_COMPUTE_DEFAULT
    # A declared binding dtype still wins outright.
    assert composition_compute_dtype(root, "fp16") == "fp16"


def test_override_decodes_bf16_latents_on_the_w8a8_execution_lane(
    tmp_path: Path,
) -> None:
    """The exact production crash path, for real on CPU: load the override
    through `load_component_override` against the w8a8 base (hub bindings
    carry no dtype), then run bf16 latents through its decode. RED pre-fix
    with the exact live signature: `Input type (c10::BFloat16) and bias type
    (c10::Half) should be the same`."""
    root = _w8a8_base_tree(tmp_path)
    override = _fp32_vae_override(tmp_path)

    vae = load_component_override(root, "vae", override)
    dtypes = {p.dtype for p in vae.parameters()}
    assert dtypes == {torch.bfloat16}, (
        f"override must land at the w8a8 lane's bf16 compute dtype, got "
        f"{dtypes} — a Half override in a bf16 composition is the pgw#675 "
        "warmup/serve fatal"
    )
    latents = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
    with torch.inference_mode():
        decoded = vae.decode(latents).sample
    assert decoded.dtype == torch.bfloat16
