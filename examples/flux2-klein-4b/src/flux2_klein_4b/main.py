from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Annotated, Optional

import msgspec
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.payload_constraints import Clamp
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class GenerateInput(msgspec.Struct):
    prompt: str
    # Turbo model: keep the range tight for latency/cost predictability.
    # Accept int/float and clamp to [4, 8] (rounded to nearest int).
    num_inference_steps: Annotated[int | float, Clamp(4, 8, cast="int")] = 8
    guidance_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class GenerateOutput(msgspec.Struct):
    image: Asset


def _should_enable_seq_offload() -> bool:
    raw = (os.getenv("COZY_DISABLE_SEQUENTIAL_CPU_OFFLOAD") or "").strip().lower()
    return raw not in {"1", "true", "yes", "y", "t"}


@worker_function(ResourceRequirements())
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b")  # Key from cozy.toml [models]
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    steps = int(payload.num_inference_steps)
    logger.info(
        "[run_id=%s] flux2-klein-4b prompt=%r steps=%s (requested=%s)",
        ctx.run_id,
        payload.prompt,
        steps,
        payload.num_inference_steps,
    )

    # FLUX.2-klein-4B can exceed 8GB VRAM; use sequential CPU offload by default.
    if torch.cuda.is_available() and _should_enable_seq_offload():
        if not getattr(pipeline, "_cozy_seq_offload_enabled", False):
            pipeline.enable_sequential_cpu_offload(gpu_id=0)
            setattr(pipeline, "_cozy_seq_offload_enabled", True)

    generator = None
    if payload.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    result = pipeline(
        prompt=payload.prompt,
        num_inference_steps=steps,
        guidance_scale=payload.guidance_scale,
        width=payload.width,
        height=payload.height,
        generator=generator,
    )
    image: Image.Image = result.images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    out = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())
    return GenerateOutput(image=out)


@worker_function(ResourceRequirements())
def generate_fp8(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b_fp8")  # Key from cozy.toml [models]
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    FP8 endpoint.

    This endpoint is intended to run against an fp8-weight-only artifact (or an artifact
    that the worker can load with torchao-backed fp8 quantization enabled).
    """
    return generate(ctx, pipeline, payload)


@worker_function(ResourceRequirements())
def generate_int8(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b_int8")  # Key from cozy.toml [models]
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    INT8 endpoint (weight-only).

    This endpoint is intended to run against an int8-weight-only artifact (or an artifact
    that the worker can load with torchao-backed int8 quantization enabled).
    """
    return generate(ctx, pipeline, payload)


@worker_function(ResourceRequirements())
def generate_int4(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b_int4")  # Key from cozy.toml [models]
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    INT4 endpoint (weight-only).

    This endpoint is experimental; expect quality regressions or incompatibilities.
    """
    return generate(ctx, pipeline, payload)
