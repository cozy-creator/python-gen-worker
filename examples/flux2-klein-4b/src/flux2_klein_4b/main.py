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
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class GenerateInput(msgspec.Struct):
    prompt: str
    # Keep the range tight for latency/cost predictability; clamped in code.
    num_inference_steps: int | float = 8
    guidance_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class GenerateOutput(msgspec.Struct):
    image: Asset


def _should_enable_seq_offload() -> bool:
    raw = (os.getenv("COZY_DISABLE_SEQUENTIAL_CPU_OFFLOAD") or "").strip().lower()
    return raw not in {"1", "true", "yes", "y", "t"}


def _generate(
    ctx: ActionContext,
    pipeline,
    payload: GenerateInput,
    model_key: str,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    steps = max(4, min(8, int(payload.num_inference_steps)))
    logger.info(
        "[run_id=%s] %s prompt=%r steps=%s (requested=%s)",
        ctx.run_id,
        model_key,
        payload.prompt,
        steps,
        payload.num_inference_steps,
    )

    # FLUX.2-klein variants can exceed 8GB VRAM; use sequential CPU offload by default.
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
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(
            Src.FIXED,
            "flux2-klein-4b",
            ref="black-forest-labs/FLUX.2-klein-4B",
            dtypes=("bf16",),
        ),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-4b")


@worker_function(ResourceRequirements())
def generate_fp8(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(
            Src.FIXED,
            "flux2-klein-4b_fp8",
            ref="black-forest-labs/FLUX.2-klein-4B",
            dtypes=("fp8",),
        ),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    FP8 endpoint.

    This endpoint is intended to run against an fp8-weight-only artifact (or an artifact
    that the worker can load with torchao-backed fp8 quantization enabled).
    """
    return _generate(ctx, pipeline, payload, "flux2-klein-4b_fp8")


@worker_function(ResourceRequirements())
def generate_9b(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(
            Src.FIXED,
            "flux2-klein-9b",
            ref="black-forest-labs/FLUX.2-klein-9B",
            dtypes=("bf16",),
        ),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b")


@worker_function(ResourceRequirements())
def generate_9b_fp8(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(
            Src.FIXED,
            "flux2-klein-9b_fp8",
            ref="black-forest-labs/FLUX.2-klein-9B",
            dtypes=("fp8",),
        ),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b_fp8")


@worker_function(ResourceRequirements())
def generate_int8(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(
            Src.FIXED,
            "flux2-klein-4b_int8",
            ref="black-forest-labs/FLUX.2-klein-4B",
            dtypes=("int8",),
        ),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    INT8 endpoint (weight-only).

    This endpoint is intended to run against an int8-weight-only artifact (or an artifact
    that the worker can load with torchao-backed int8 quantization enabled).
    """
    return _generate(ctx, pipeline, payload, "flux2-klein-4b_int8")


@worker_function(ResourceRequirements())
def generate_int4(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(
            Src.FIXED,
            "flux2-klein-4b_int4",
            ref="black-forest-labs/FLUX.2-klein-4B",
            dtypes=("int4",),
        ),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    INT4 endpoint (weight-only).

    This endpoint is experimental; expect quality regressions or incompatibilities.
    """
    return _generate(ctx, pipeline, payload, "flux2-klein-4b_int4")
