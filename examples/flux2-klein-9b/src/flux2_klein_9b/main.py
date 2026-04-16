from __future__ import annotations

import logging
import os
import threading
from io import BytesIO
from typing import Annotated, Optional

import msgspec
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from gen_worker import RequestContext, ResourceRequirements, worker_function
from gen_worker.api.injection import ModelRef, ModelRefSource as Src
from gen_worker.api.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_flux_resources = ResourceRequirements()
_nvfp4_resources = ResourceRequirements(compute_capability_min=10.0)
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
    ctx: RequestContext,
    pipeline,
    payload: GenerateInput,
    model_key: str,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    steps = max(4, min(8, int(payload.num_inference_steps)))
    logger.info(
        "[request_id=%s] %s prompt=%r steps=%s (requested=%s)",
        ctx.request_id,
        model_key,
        payload.prompt,
        steps,
        payload.num_inference_steps,
    )

    with _lock_for_pipeline(pipeline):
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
    out = ctx.save_bytes(f"jobs/{ctx.request_id}/outputs/image.png", buf.getvalue())
    return GenerateOutput(image=out)


@worker_function(_flux_resources)
def generate(
    ctx: RequestContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(Src.FIXED, "flux2-klein-9b-base"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b-base")


@worker_function(_flux_resources)
def generate_turbo(
    ctx: RequestContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(Src.FIXED, "flux2-klein-9b-turbo"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b-turbo")


@worker_function(_flux_resources)
def generate_fp8(
    ctx: RequestContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(Src.FIXED, "flux2-klein-9b-base_fp8"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b-base_fp8")


@worker_function(_flux_resources)
def generate_turbo_fp8(
    ctx: RequestContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(Src.FIXED, "flux2-klein-9b-turbo_fp8"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b-turbo_fp8")


@worker_function(_nvfp4_resources)
def generate_nvfp4(
    ctx: RequestContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(Src.FIXED, "flux2-klein-9b-base_nvfp4"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b-base_nvfp4")


@worker_function(_nvfp4_resources)
def generate_turbo_nvfp4(
    ctx: RequestContext,
    pipeline: Annotated[
        Flux2KleinPipeline,
        ModelRef(Src.FIXED, "flux2-klein-9b-turbo_nvfp4"),
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    return _generate(ctx, pipeline, payload, "flux2-klein-9b-turbo_nvfp4")
