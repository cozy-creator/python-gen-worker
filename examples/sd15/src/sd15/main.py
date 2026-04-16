from __future__ import annotations

import logging
import threading
from io import BytesIO
from typing import Annotated, Optional

import msgspec
from diffusers import StableDiffusionPipeline
from PIL import Image

from gen_worker import RequestContext, ResourceRequirements, worker_function
from gen_worker.api.injection import ModelRef, ModelRefSource as Src
from gen_worker.api.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_sd15_resources = ResourceRequirements()
_pipeline_locks_guard = threading.Lock()
_pipeline_locks: dict[int, threading.Lock] = {}


def _lock_for_pipeline(pipeline: object) -> threading.Lock:
    key = id(pipeline)
    with _pipeline_locks_guard:
        lock = _pipeline_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _pipeline_locks[key] = lock
    return lock


class GenerateInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None


class GenerateOutput(msgspec.Struct):
    image: Asset


@worker_function(_sd15_resources)
def generate(
    ctx: RequestContext,
    pipeline: Annotated[
        StableDiffusionPipeline,
        ModelRef(Src.FIXED, "sd15"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    requested_steps = payload.num_inference_steps
    steps = requested_steps
    if steps < 25:
        steps = 25

    logger.info(
        "[request_id=%s] sd15 prompt=%r steps=%s (requested=%s)",
        ctx.request_id,
        payload.prompt,
        steps,
        requested_steps,
    )

    generator = None
    if payload.seed is not None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    with _lock_for_pipeline(pipeline):
        result = pipeline(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            num_inference_steps=steps,
            guidance_scale=payload.guidance_scale,
            width=payload.width,
            height=payload.height,
            generator=generator,
        )
    image: Image.Image = result.images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    out = ctx.save_bytes(f"jobs/{ctx.request_id}/outputs/image.png", buf.getvalue())
    return GenerateOutput(image=out)


@worker_function(_sd15_resources)
def generate_fp8(
    ctx: RequestContext,
    pipeline: Annotated[
        StableDiffusionPipeline,
        ModelRef(Src.FIXED, "sd15_fp8"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    FP8 function.

    This endpoint is intended to run against an fp8-weight-only artifact (or an artifact
    that the worker can load with torchao-backed fp8 quantization enabled).
    """
    return generate(ctx, pipeline, payload)


@worker_function(_sd15_resources)
def generate_int8(
    ctx: RequestContext,
    pipeline: Annotated[
        StableDiffusionPipeline,
        ModelRef(Src.FIXED, "sd15_int8"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    INT8 function (weight-only).

    This endpoint is intended to run against an int8-weight-only artifact (or an artifact
    that the worker can load with torchao-backed int8 quantization enabled).
    """
    return generate(ctx, pipeline, payload)


@worker_function(_sd15_resources)
def generate_int4(
    ctx: RequestContext,
    pipeline: Annotated[
        StableDiffusionPipeline,
        ModelRef(Src.FIXED, "sd15_int4"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    """
    INT4 function (weight-only).

    This endpoint is experimental; many diffusion pipelines are not validated at int4.
    """
    return generate(ctx, pipeline, payload)
