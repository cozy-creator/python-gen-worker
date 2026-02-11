"""
multi-sdxl-checkpoints: Payload-based model selection example

This example demonstrates how to support multiple model fine-tunes (checkpoints)
efficiently using a single endpoint.

Key concepts:
- Models are declared in cozy.toml:
  - global [models] applies to all endpoints by default
  - optional [endpoints.<name>.models] overrides per-endpoint model keyspaces
- ModelRef(Src.PAYLOAD, "model_key") resolves the model from the request payload
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
    # Optional. If empty, we apply a light default negative prompt to reduce
    # watermarks/text artifacts (common in SDXL outputs).
    negative_prompt: str = ""
    # Default SDXL quality target.
    num_inference_steps: int = 25
    # Default SDXL guidance: 7.0 is a common "production" default across SDXL fine-tunes.
    guidance_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    # "auto" applies quality defaults for SDXL checkpoints.
    # "manual" uses provided fields as-is (bounded to sane limits).
    preset: str = "auto"


class GenerateOutput(msgspec.Struct):
    """Output containing the generated image."""
    image: Asset
    model_used: str  # Echo back which model was used


def _set_seed_and_perf(seed: Optional[int]) -> None:
    import torch

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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

    _set_seed_and_perf(payload.seed)
    import torch

    preset = (payload.preset or "").strip().lower() or "auto"

    # If preset=manual, we don't override user-supplied settings.

    steps = int(payload.num_inference_steps)
    guidance = float(payload.guidance_scale)
    width = int(payload.width)
    height = int(payload.height)
    neg = (payload.negative_prompt or "").strip()

    if preset == "auto":
        # SDXL quality defaults: 25+ steps, moderate CFG.
        steps = max(25, min(80, steps))
        if not neg:
            neg = "text, watermark, logo, low quality, blurry, jpeg artifacts"
    else:
        # Manual: respect provided inputs, but keep them sane.
        steps = max(1, min(80, steps))
        width = max(64, min(2048, width))
        height = max(64, min(2048, height))
        if not neg and guidance > 1.0:
            neg = "text, watermark, logo, low quality, blurry, jpeg artifacts"

    # SDXL uses dual text encoders. We intentionally mirror the single user prompt
    # pair into both channels instead of exposing secondary payload prompt fields.
    prompt_2 = payload.prompt
    neg_2 = neg

    # On small GPUs (common on dev laptops), SDXL at 1024 can OOM the driver.
    # These toggles reduce peak VRAM with minimal quality impact.
    try:
        if torch.cuda.is_available():
            vram_bytes = int(torch.cuda.get_device_properties(0).total_memory)
            if vram_bytes <= 10 * 1024**3:
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_slicing"):
                    pipeline.vae.enable_slicing()
                elif hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
                    pipeline.vae.enable_tiling()
                elif hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()
                # On <=8GB GPUs, keep SDXL memory stable by offloading modules to CPU.
                if vram_bytes <= 8 * 1024**3 and hasattr(pipeline, "enable_model_cpu_offload"):
                    try:
                        pipeline.enable_model_cpu_offload()
                    except TypeError:
                        pipeline.enable_model_cpu_offload(gpu_id=0)
                torch.cuda.empty_cache()
    except Exception:
        pass

    # Scheduler: model-aware defaults.
    #
    # Note: we mutate the pipeline's scheduler in-place. The worker caches the
    # pipeline, so we avoid re-instantiating schedulers on every call.
    try:
        from diffusers import (  # type: ignore
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
        )

        # Quality-first default: Euler a is robust around ~25-35 steps across SDXL checkpoints.
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(  # type: ignore[attr-defined]
            pipeline.scheduler.config  # type: ignore[attr-defined]
        )
        _ = DPMSolverMultistepScheduler  # keep import validated for future scheduler knobs
    except Exception:
        # Scheduler config is best-effort; fall back to checkpoint default.
        pass

    # Generate the image
    call_kwargs = dict(
        prompt=payload.prompt,
        negative_prompt=neg,
        prompt_2=prompt_2,
        negative_prompt_2=neg_2,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
    )

    # Some diffusers versions add `guidance_rescale` to reduce overexposure at
    # higher CFG; use it when available, otherwise retry without it.
    # Keep inference allocations as tight as possible on small VRAM cards.
    # SDXL decode can OOM late in the call if the CUDA caching allocator holds stale blocks.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        with torch.inference_mode():
            result = pipeline(**call_kwargs)
    except TypeError:
        call_kwargs.pop("prompt_2", None)
        call_kwargs.pop("negative_prompt_2", None)
        with torch.inference_mode():
            result = pipeline(**call_kwargs)
    image: Image.Image = result.images[0]

    # Save the output
    buf = BytesIO()
    image.save(buf, format="PNG")
    output_asset = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())

    return GenerateOutput(
        image=output_asset,
        model_used=payload.model_key,
    )
