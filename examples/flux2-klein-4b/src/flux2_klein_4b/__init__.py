from __future__ import annotations

import logging
from io import BytesIO
from typing import Annotated, Optional

import msgspec
from diffusers import Flux2KleinPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class GenerateInput(msgspec.Struct):
    prompt: str
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class GenerateOutput(msgspec.Struct):
    image: Asset


@worker_function(ResourceRequirements())
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b")  # Key from [tool.cozy.models]
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    logger.info("[run_id=%s] flux2-klein-4b prompt=%r", ctx.run_id, payload.prompt)

    generator = None
    if payload.seed is not None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

    result = pipeline(
        prompt=payload.prompt,
        num_inference_steps=payload.num_inference_steps,
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
