import os
import io
import time
import logging
import base64
import torch
from diffusers import StableDiffusionXLPipeline
from pydantic import BaseModel, Field
from typing import List, Optional

import msgpack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenInput(BaseModel):
    prompt: str = Field(..., description="Prompt for image generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt (optional)")
    steps: int = Field(30, description="Number of inference steps")
    num_images: int = Field(1, description="Number of images to generate")
    guidance_scale: float = Field(7.5, description="Guidance scale for classifier-free guidance")


class ImageGenOutput(BaseModel):
    urls: List[str] = Field(..., description="Local file URLs of generated images")


# Global pipeline instance (lazy-loaded)
_global_pipeline: StableDiffusionXLPipeline = None

def get_pipeline() -> StableDiffusionXLPipeline:
    """
    Lazy-load and return the Stable Diffusion XL pipeline on GPU.
    """
    global _global_pipeline
    if _global_pipeline is None:
        logger.info("Loading StableDiffusionXLPipeline onto GPU...")
        _global_pipeline = StableDiffusionXLPipeline.from_pretrained(
            "John6666/fucktastic-anime-checkpoint-15-sdxl",
            torch_dtype=torch.float16,
            # variant="fp16"
        )
        _global_pipeline.to("cuda")
        logger.info("Pipeline loaded successfully.")
    return _global_pipeline


def image_gen_action(ctx, data: dict) -> dict:
    """
    Main action function for image generation.
    """
    # Check if the action was canceled
    if ctx.is_canceled():
        return {"error": "Action was canceled before start."}
    
    # Parse and validate input
    try:
        input_data = ImageGenInput(**data)
    except Exception as e:
        logger.error(f"Invalid input: {e}")
        return {"error": str(e)}
    
    # Load the pipeline (loads only once)
    pipeline = get_pipeline()

    try:
        logger.info(f"Generating images with prompt: {input_data.prompt}")
        with torch.inference_mode():
            output = pipeline(
                prompt=input_data.prompt,
                negative_prompt=input_data.negative_prompt,
                num_inference_steps=input_data.steps,
                num_images_per_prompt=input_data.num_images,
                guidance_scale=input_data.guidance_scale
            )
            images = output.images

        # Save images locally and construct file URLs
        output_dir = "./generated_images"
        os.makedirs(output_dir, exist_ok=True)
        urls = []
        for idx, img in enumerate(images):
            filename = f"img_{int(time.time())}_{idx}.png"
            file_path = os.path.join(output_dir, filename)
            img.save(file_path, format="PNG")
            abs_path = os.path.abspath(file_path)
            urls.append(f"file://{abs_path}")
        
        logger.info("Image generation successful.")
        return ImageGenOutput(urls=urls).dict()
    
    except Exception as e:
        logger.exception("Error during image generation:")
        return {"error": str(e)}
    

def register_functions(worker):
    """
    Register the image generation function with the provided worker.
    This should be called by the worker's import-and-register process.
    """
    worker.register_function(image_gen_action, "image_gen_action")
    logger.info("Registered image_gen_action with the worker.")
        
