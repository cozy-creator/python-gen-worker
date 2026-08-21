from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import diffusers
from diffusers.modular_pipelines.modular_pipeline import (
    ModularPipeline,
    ModularPipelineBlocks,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

import tiny_tree

COMPONENTS = {
    "unet": ("diffusers", "UNet2DConditionModel"),
    "vae": ("diffusers", "AutoencoderKL"),
    "text_encoder": ("transformers", "CLIPTextModel"),
    "tokenizer": ("transformers", "CLIPTokenizer"),
    "scheduler": ("diffusers", "DDIMScheduler"),
}


class TinyModularStep(ModularPipelineBlocks):  # type: ignore[misc]
    model_name = "tiny-modular"  # type: ignore[assignment]

    @property
    def description(self) -> str:
        return "the block whose expected_components ARE the pipeline's"

    @property
    def expected_components(self) -> list[Any]:
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        return [
            ComponentSpec("unet", UNet2DConditionModel),
            ComponentSpec("vae", AutoencoderKL),
            ComponentSpec("text_encoder", CLIPTextModel),
            ComponentSpec("tokenizer", CLIPTokenizer),
            ComponentSpec("scheduler", DDIMScheduler),
        ]

    @property
    def inputs(self) -> list[Any]:
        return []

    @property
    def intermediate_outputs(self) -> list[Any]:
        return []

    def __call__(self, components: Any, state: Any) -> Any:
        return components, state


class TinyModularBlocks(SequentialPipelineBlocks):  # type: ignore[misc]
    model_name = "tiny-modular"  # type: ignore[assignment]
    block_classes = [TinyModularStep]
    block_names = ["tiny"]


class TinyStreamingPipeline(ModularPipeline):  # type: ignore[misc]
    """The adapter every modular endpoint writes (minimax-h3's, verbatim)."""

    default_blocks_name = "TinyModularBlocks"  # type: ignore[assignment]

    _OWN = (
        "blocks",
        "pretrained_model_name_or_path",
        "components_manager",
        "collection",
        "modular_config_dict",
        "config_dict",
    )

    def __init__(self, **kwargs: Any) -> None:
        own = {name: kwargs.pop(name) for name in self._OWN if name in kwargs}
        super().__init__(**own)
        components = {
            name: value
            for name, value in kwargs.items()
            if name in self._component_specs and value is not None
        }
        if components:
            self.update_components(**components)
        left = sorted(name for name in kwargs if name not in self._component_specs)
        if left:
            raise TypeError(
                f"{type(self).__name__}: {left} are neither pipeline arguments "
                f"nor components of this pipeline"
            )


for _cls in (TinyModularBlocks, TinyStreamingPipeline):
    setattr(diffusers, _cls.__name__, _cls)


def save_config_only(tree: Path) -> Path:
    """The tiny config-only tree plus a modular index that points INTO it."""

    root = tiny_tree.save_config_only(tree)
    index: dict[str, Any] = {
        "_class_name": TinyStreamingPipeline.__name__,
        "_blocks_class_name": TinyModularBlocks.__name__,
        "_diffusers_version": diffusers.__version__,
    }
    for name, (library, class_name) in COMPONENTS.items():
        index[name] = [
            library,
            class_name,
            {
                "repo": str(root),
                "subfolder": name,
                "type_hint": [library, class_name],
                "variant": None,
                "revision": None,
            },
        ]
    (root / "modular_model_index.json").write_text(json.dumps(index, indent=2))
    return root
