"""A CONFIG-ONLY checkpoint tree for the structure-only forge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DENOISER_COMPONENT = "transformer"

DENOISER_CONFIG: Dict[str, Any] = {
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 1,
    "block_out_channels": (32, 32),
    "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
    "cross_attention_dim": 32,
    "attention_head_dim": 8,
    "norm_num_groups": 8,
}


def build_config_only_tree(root: Path) -> Path:
    """Write ``model_index.json`` + one config-only component dir; return it."""
    from diffusers import UNet2DConditionModel

    root = Path(root)
    component = root / DENOISER_COMPONENT
    component.mkdir(parents=True, exist_ok=True)
    UNet2DConditionModel(**DENOISER_CONFIG).save_config(  # type: ignore[attr-defined]
        str(component))
    (root / "model_index.json").write_text(json.dumps(
        {
            "_class_name": "StableDiffusionPipeline",
            DENOISER_COMPONENT: ["diffusers", "UNet2DConditionModel"],
        },
        indent=2, sort_keys=True,
    ))
    return root
