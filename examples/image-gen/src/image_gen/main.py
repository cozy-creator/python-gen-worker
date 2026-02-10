from __future__ import annotations

import logging
from io import BytesIO
from typing import Annotated

import msgspec
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

sdxl_resources = ResourceRequirements()

class GenerateInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024


class GenerateOutput(msgspec.Struct):
    image: Asset


@worker_function(sdxl_resources)
def generate_image(
    ctx: ActionContext,
    pipeline: Annotated[
        StableDiffusionXLPipeline, ModelRef(Src.DEPLOYMENT, "sdxl")  # Key from cozy.toml [models]
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    logger.info("[run_id=%s] image-gen prompt=%r", ctx.run_id, payload.prompt)

    result = pipeline(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        num_inference_steps=payload.num_inference_steps,
        guidance_scale=payload.guidance_scale,
        width=payload.width,
        height=payload.height,
    )
    image: Image.Image = result.images[0]

    buf = BytesIO()
    image.save(buf, format="PNG")
    out = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/image.png", buf.getvalue())
    
    return GenerateOutput(image=out)


# @worker_function(resources=sdxl_resources)
# def generate_image(ctx: ActionContext, pipeline: DiffusionPipeline = None, prompt_details: dict = None) -> bytes:
#     """
#     Legacy function: Generates an image and returns raw bytes.
#     Consider using generate_and_upload_image for complete workflows.
#     """
#     if pipeline is None:
#         logger.error(f"Run {ctx.run_id} for generate_image received a None pipeline object.")
#         raise ValueError("Pipeline object cannot be None for image generation.")

#     prompt = prompt_details.get("prompt", "a beautiful landscape")
#     seed = int(prompt_details.get("seed", 42))

#     logger.info(f"[run_id={ctx.run_id}] Generating image for prompt: '{prompt}', seed: {seed}")

#     if ctx.is_canceled():
#         raise InterruptedError("Image generation cancelled")

#     generator = torch.Generator(device=device).manual_seed(seed)

#     with torch.inference_mode():
#         image_result = pipeline(
#             prompt=prompt,
#             generator=generator,
#             num_inference_steps=28
#         ).images[0]

#     buffer = BytesIO()
#     image_result.save(buffer, format="PNG")
#     img_bytes = buffer.getvalue()
    
#     logger.info(f"[run_id={ctx.run_id}] Image generation complete ({len(img_bytes)} bytes).")
#     return img_bytes


# s3_upload_resources = ResourceRequirements()

# @worker_function(resources=s3_upload_resources)
# def upload_image_to_s3(ctx: ActionContext, upload_details: dict) -> Dict[str, str]:
#     """
#     Legacy function: Uploads image bytes to S3.
#     Consider using generate_and_upload_image for complete workflows.
#     """
#     logger.info(f"[run_id={ctx.run_id}] Received S3 upload request.")

#     image_bytes = upload_details.get("image_bytes")
#     filename = upload_details.get("filename")

#     if not image_bytes or not isinstance(image_bytes, bytes):
#         raise ValueError("Missing or invalid 'image_bytes' in upload_details")
    
#     if ctx.is_canceled():
#         raise InterruptedError("Upload cancelled")
    
#     s3_url = _upload_image_to_s3(ctx, image_bytes, filename)
#     return {"s3_url": s3_url}
