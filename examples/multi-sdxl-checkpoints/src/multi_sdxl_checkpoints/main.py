"""
multi-sdxl-checkpoints: Payload-based model selection example

This example demonstrates how to support multiple model fine-tunes (checkpoints)
efficiently. The request payload specifies which model to use via a key, and the
orchestrator's scheduler routes requests to workers that already have that model
loaded in VRAM.

Key concepts:
- Models are declared in cozy.toml [models] with endpoint-local keys
- ModelRef(Src.PAYLOAD, "model_key") resolves the model from the request
- Scheduler uses vram_models/disk_models heartbeat data for smart routing
- LRU eviction manages VRAM when switching between models
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Annotated, Optional

import msgspec
from diffusers import DiffusionPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class GenerateInput(msgspec.Struct):
    """Input for image generation with model selection."""
    prompt: str
    model_key: str = "sdxl-base"  # Key from cozy.toml [models]
    negative_prompt: str = ""
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class GenerateOutput(msgspec.Struct):
    """Output containing the generated image."""
    image: Asset
    model_used: str  # Echo back which model was used


@worker_function(ResourceRequirements())
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        DiffusionPipeline,
        ModelRef(Src.PAYLOAD, "model_key")  # Model key comes from payload.model_key
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    Generate an image using the model specified in the payload.

    The model_key in the payload must be a key in `cozy.toml [models]`.

    The scheduler routes this request to a worker that has the requested
    model already loaded in VRAM (hot) or on disk (warm). If no worker has
    the model, any capable worker will download and load it (cold start).
    """
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    logger.info("[run_id=%s] Generating with model_key=%s, prompt=%r",
               ctx.run_id, payload.model_key, payload.prompt[:50])

    # Set up generator for reproducibility
    import torch
    generator = None
    if payload.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    # Adjust parameters for turbo models (they need fewer steps)
    steps = payload.num_inference_steps
    guidance = payload.guidance_scale
    if "turbo" in payload.model_key.lower():
        # Turbo models work best with fewer steps and lower guidance
        steps = min(steps, 4)
        guidance = 0.0
        logger.info("[run_id=%s] Using turbo settings: steps=%d, guidance=%.1f",
                   ctx.run_id, steps, guidance)

    # Generate the image
    result = pipeline(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=payload.width,
        height=payload.height,
        generator=generator,
    )
    image: Image.Image = result.images[0]

    # Save the output
    buf = BytesIO()
    image.save(buf, format="PNG")
    output_asset = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())

    return GenerateOutput(
        image=output_asset,
        model_used=payload.model_key,
    )
