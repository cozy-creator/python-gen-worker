import time
from PIL import Image
from io import BytesIO
import logging
import torch
from diffusers import StableDiffusionXLPipeline
from typing import Optional, Dict
from gen_worker import worker_function, ResourceRequirements, ActionContext
import os
import boto3
import blake3
from botocore.client import Config
import threading
from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Combined resources for both image generation and S3 upload
sdxl_resources = ResourceRequirements(
    model_name="John6666/holy-mix-illustriousxl-vibrant-anime-checkpoint-v1-sdxl",
    min_vram_gb=8.0,
    recommended_vram_gb=12.0,
    expects_pipeline_arg=True
)

@worker_function(resources=sdxl_resources)
def generate_and_upload_image(ctx: ActionContext, pipeline: DiffusionPipeline = None, request_details: dict = None) -> Dict[str, str]:
    """
    Generates an image based on a prompt and uploads it directly to S3.

    Args:
        ctx: The ActionContext provided by the worker.
        request_details: A dictionary containing:
            - 'prompt' (str): Text prompt for image generation
            - 'seed' (int, optional): Random seed for generation
            - 'filename' (str, optional): Desired filename for S3 upload
            - Other SDXL parameters (negative_prompt, num_inference_steps, etc.)

    Returns:
        A dictionary containing 's3_url' and generation metadata.
    """
    
    if pipeline is None:
        logger.error(f"Run {ctx.run_id} for generate_and_upload_image received a None pipeline object.")
        raise ValueError("Pipeline object cannot be None for image generation.")

    # Extract generation parameters
    prompt = request_details.get("prompt", "a beautiful landscape")
    negative_prompt = request_details.get("negative_prompt", "")
    seed = int(request_details.get("seed", 42))
    num_inference_steps = int(request_details.get("num_inference_steps", 28))
    guidance_scale = float(request_details.get("guidance_scale", 7.5))
    width = int(request_details.get("width", 1024))
    height = int(request_details.get("height", 1024))
    filename = request_details.get("filename")

    logger.info(f"[run_id={ctx.run_id}] Starting combined generate+upload for prompt: '{prompt}', seed: {seed}")

    try:
        # Step 1: Generate Image
        logger.info(f"[run_id={ctx.run_id}] Generating image on device {device}")
        
        # Check for cancellation before starting inference
        if ctx.is_canceled():
            logger.warning(f"[run_id={ctx.run_id}] Cancellation detected before starting generation.")
            raise InterruptedError("Image generation cancelled before start")

        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            image_result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height
            ).images[0]

        if ctx.is_canceled():
            logger.warning(f"[run_id={ctx.run_id}] Cancellation detected after generation finished.")
            raise InterruptedError("Image generation cancelled after processing")

        # Convert PIL Image to PNG bytes
        buffer = BytesIO()
        image_result.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        logger.info(f"[run_id={ctx.run_id}] Image generation complete ({len(img_bytes)} bytes).")

        if ctx.is_canceled():
            logger.warning(f"[run_id={ctx.run_id}] Cancellation detected before upload.")
            raise InterruptedError("Upload cancelled before starting")

        # Step 2: Upload to S3
        logger.info(f"[run_id={ctx.run_id}] Uploading generated image to S3")
        
        s3_url = _upload_image_to_s3(ctx, img_bytes, filename)
        
        logger.info(f"[run_id={ctx.run_id}] Complete workflow finished successfully")
        
        return {
            "s3_url": s3_url,
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "image_size_bytes": len(img_bytes)
        }

    except InterruptedError:
        logger.warning(f"[run_id={ctx.run_id}] Workflow was cancelled")
        raise
    except Exception as e:
        logger.exception(f"[run_id={ctx.run_id}] Error during generate+upload workflow")
        raise


def _upload_image_to_s3(ctx: ActionContext, image_bytes: bytes, filename: Optional[str] = None) -> str:
    """
    Internal helper function to upload image bytes to S3.
    
    Args:
        ctx: The ActionContext
        image_bytes: The raw image data
        filename: Optional filename for the S3 object
        
    Returns:
        The S3 URL of the uploaded image
    """
    if not image_bytes or not isinstance(image_bytes, bytes):
        raise ValueError("Missing or invalid image_bytes for S3 upload")
    
    # Get S3 configuration from environment
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    region = os.environ.get("S3_REGION")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")

    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is not set for this worker.")
    
    s3_client = _get_s3_client()
    if not s3_client:
        raise RuntimeError("Failed to initialize S3 client. Check worker logs and environment variables.")
    
    s3_key = _generate_s3_key(image_bytes, filename)
    content_type = "image/png"

    logger.info(f"[run_id={ctx.run_id}] Uploading {len(image_bytes)} bytes to s3://{bucket_name}/{s3_key}")

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=image_bytes,
            ContentType=content_type,
            ACL='public-read'
        )

        file_url = f"https://{bucket_name}.{region}.digitaloceanspaces.com/{s3_key}"
        logger.info(f"[run_id={ctx.run_id}] Upload successful. URL: {file_url}")
        return file_url
    
    except Exception as e:
        logger.exception(f"[run_id={ctx.run_id}] Error during S3 upload")
        raise


def _get_s3_client() -> Optional[boto3.client]:
    """Initializes and returns an S3 client using environment variables."""
    access_key = os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    region = os.environ.get("S3_REGION")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")

    if not all([access_key, secret_key, bucket_name, region]):
        logger.error("Missing one or more S3 environment variables (S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME)")
        return None
    
    try:
        s3_client = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4")
        )
        return s3_client
    except Exception as e:
        logger.exception("Failed to initialize S3 client")
        return None
    

def _generate_s3_key(image_bytes: bytes, filename: Optional[str] = None) -> str:
    """Generates a unique S3 key for the image."""
    if filename:
        safe_filename = "".join(c if c.isalnum() or c in ['.', '-', '_'] else '_' for c in filename)
        hash_prefix = blake3.blake3(image_bytes).hexdigest()[:16]
        return f"uploads/{hash_prefix}-{safe_filename}"
    else:
        hash_hex = blake3.blake3(image_bytes).hexdigest()
        return f"uploads/{hash_hex}"


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