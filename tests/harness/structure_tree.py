"""A CONFIG-ONLY checkpoint tree for the structure-only forge.

``structure_only.build_component`` resolves a component's CLASS out of
``model_index.json`` and builds it from ``config.json`` alone, holding no
weights. It therefore needs a tree, and until pgw#1373 the tree came from
``examples/micro-diffusion`` — a hand-written ``MicroDenoiser``/``MicroDecoder``
pair. 56d89b7f deleted ``examples/`` (every one declared against the v1 SDK) and
took the fixture with it, which errored the only remaining test of a live
production module (pgw#1438).

The replacement is a REAL ``diffusers`` class rather than a re-vendored toy, and
that is an improvement, not a substitution:

* ``build_component`` resolves ``[library, class]`` through the installed
  library, so a diffusers row exercises the resolution production takes; a
  private module in the test tree exercised an import path no pod has.
* the class ships its own ``load_config``/``from_config``, which is the exact
  surface ``_require_config_surface`` fences — a toy satisfying it by hand
  proves the toy, not the fence.
* there is nothing to keep in sync when diffusers moves.

No weights are written and none move: every file here is JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

#: The denoiser component name, from ``gen_worker.component_vocab``'s denoiser
#: vocabulary — ``quantize_tree_w8a8`` and ``detect_w8a8_artifacts`` both scan
#: for a directory named from it, so a tree whose denoiser is called anything
#: else is invisible to the whole w8a8 path (pgw#1014).
DENOISER_COMPONENT = "transformer"

#: Small enough that instantiating it costs milliseconds, wide enough that it
#: carries cross-attention — the Linears a quantizer re-wraps are the whole
#: subject of the virtuality fences, so a Linear-free config tests nothing.
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
    # `save_config` rather than a hand-written dict: the config the class
    # ACCEPTS is the class's business, and hand-writing it is how a fixture
    # starts lying about a library that moved.
    UNet2DConditionModel(**DENOISER_CONFIG).save_config(str(component))
    (root / "model_index.json").write_text(json.dumps(
        {
            "_class_name": "StableDiffusionPipeline",
            DENOISER_COMPONENT: ["diffusers", "UNet2DConditionModel"],
        },
        indent=2, sort_keys=True,
    ))
    return root
