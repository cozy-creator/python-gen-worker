"""
z-image-lora: Dynamic LoRA loading example

This example demonstrates the z-image pattern for loading custom LoRAs at runtime.
LoRAs are passed as Assets in the request payload, downloaded by the worker, and
applied to the base Z-Image Turbo pipeline. LoRAs are unloaded after each request to avoid
memory accumulation.

Pattern inspired by fal.ai's z-image/turbo/lora endpoint.
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Annotated, List, Optional

import msgspec
from diffusers import DiffusionPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class LoraSpec(msgspec.Struct):
    """Specification for a LoRA to apply."""
    file: Asset  # LoRA weights file (safetensors format)
    weight: float = 1.0  # LoRA strength/scale
    adapter_name: Optional[str] = None  # Optional name for the adapter


class GenerateInput(msgspec.Struct):
    """Input for image generation with LoRAs."""
    prompt: str
    loras: List[LoraSpec] = []  # LoRAs to apply (passed as Assets)
    negative_prompt: str = ""
    # Turbo models typically want low step counts and guidance_scale=0.
    num_inference_steps: int = 9
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class GenerateOutput(msgspec.Struct):
    """Output containing the generated image."""
    image: Asset


@worker_function(ResourceRequirements())
def generate_with_loras(
    ctx: ActionContext,
    pipeline: Annotated[
        DiffusionPipeline, ModelRef(Src.DEPLOYMENT, "z-image-turbo")
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    Generate an image with Z-Image Turbo and dynamically loaded LoRAs.

    LoRAs are:
    1. Downloaded from the Asset URLs (materialized by the worker)
    2. Loaded into the pipeline
    3. Applied during inference
    4. Unloaded after the request completes

    This pattern allows each request to use different LoRAs without
    keeping them all in VRAM.
    """
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    logger.info("[run_id=%s] Generating with %d LoRAs", ctx.run_id, len(payload.loras))

    # Track loaded adapter names for cleanup
    loaded_adapters: List[str] = []

    try:
        # Load each LoRA from the Assets
        for i, lora_spec in enumerate(payload.loras):
            adapter_name = lora_spec.adapter_name or f"lora_{i}"

            # The Asset's local_path is populated by the worker after materialization
            lora_path = lora_spec.file.local_path
            if not lora_path:
                raise ValueError(f"LoRA {adapter_name} was not materialized (no local_path)")

            logger.info("[run_id=%s] Loading LoRA %s from %s (weight=%.2f)",
                       ctx.run_id, adapter_name, lora_path, lora_spec.weight)

            if not hasattr(pipeline, "load_lora_weights"):
                raise RuntimeError(
                    "This pipeline does not support LoRA loading (missing load_lora_weights). "
                    "If you intended to use Z-Image Turbo + LoRA, you likely need a newer diffusers build."
                )

            # Prefer passing the safetensors file path directly (diffusers supports this for LoRA adapters).
            # For directory-style adapters, lora_path can also be a directory path.
            pipeline.load_lora_weights(  # type: ignore[attr-defined]
                lora_path,
                adapter_name=adapter_name,
            )
            loaded_adapters.append(adapter_name)

        # Set adapter weights if we have LoRAs
        if loaded_adapters:
            weights = [
                lora.weight for lora in payload.loras[:len(loaded_adapters)]
            ]
            if not hasattr(pipeline, "set_adapters"):
                raise RuntimeError("This pipeline does not support multi-adapter LoRA selection (missing set_adapters).")
            pipeline.set_adapters(loaded_adapters, adapter_weights=weights)  # type: ignore[attr-defined]
            logger.info("[run_id=%s] Applied adapters: %s with weights: %s",
                       ctx.run_id, loaded_adapters, weights)

        # Set up generator for reproducibility
        import torch
        generator = None
        if payload.seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(payload.seed)

        # Generate the image
        logger.info("[run_id=%s] Running inference: prompt=%r, steps=%d",
                   ctx.run_id, payload.prompt[:50], payload.num_inference_steps)

        call_kwargs: dict[str, Any] = {
            "prompt": payload.prompt,
            "num_inference_steps": payload.num_inference_steps,
            "guidance_scale": payload.guidance_scale,
            "width": payload.width,
            "height": payload.height,
            "generator": generator,
        }
        if payload.negative_prompt:
            call_kwargs["negative_prompt"] = payload.negative_prompt

        try:
            result = pipeline(**call_kwargs)
        except TypeError:
            # Some pipelines don't accept negative_prompt (and turbo models typically don't use CFG anyway).
            call_kwargs.pop("negative_prompt", None)
            result = pipeline(**call_kwargs)
        image: Image.Image = result.images[0]

        # Save the output
        buf = BytesIO()
        image.save(buf, format="PNG")
        output_asset = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())

        return GenerateOutput(image=output_asset)

    finally:
        # Always unload LoRAs to free VRAM for next request
        if loaded_adapters:
            logger.info("[run_id=%s] Unloading %d LoRAs", ctx.run_id, len(loaded_adapters))
            try:
                if hasattr(pipeline, "unload_lora_weights"):
                    pipeline.unload_lora_weights()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("[run_id=%s] Error unloading LoRAs: %s", ctx.run_id, e)
