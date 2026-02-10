from __future__ import annotations

import logging
from io import BytesIO
from typing import Annotated, Optional

import msgspec
from diffusers import StableDiffusionPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


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


@worker_function(ResourceRequirements())
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        StableDiffusionPipeline, ModelRef(Src.DEPLOYMENT, "sd15")  # Key from cozy.toml [models]
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
        "[run_id=%s] sd15 prompt=%r steps=%s (requested=%s)",
        ctx.run_id,
        payload.prompt,
        steps,
        requested_steps,
    )

    generator = None
    if payload.seed is not None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(payload.seed)

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
    out = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())
    return GenerateOutput(image=out)
