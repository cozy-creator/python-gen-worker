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

pipe = None
device = None
pipeline_lock = threading.Lock()

def _initialize_pipeline():
    """Loads the SDXL pipeline onto the appropriate device."""
    global pipe, device
    if pipe is not None:
        return

    logger.info("Initializing Stable Diffusion XL Pipeline (stabilityai/stable-diffusion-xl-base-1.0)...")
    try:
        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            logger.info("CUDA available, setting device to GPU and dtype to float16.")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            logger.warning("CUDA not available, setting device to CPU and dtype to float32. Inference will be slow.")

        print(f"Device: {device}")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "John6666/holy-mix-illustriousxl-vibrant-anime-checkpoint-v1-sdxl",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            # variant="fp16" if torch_dtype == torch.float16 else None
        ).to(device)
        
        logger.info(f"Pipeline loaded successfully and moved to device '{device}'.")

    except ImportError as ie:
         logger.exception("ImportError during pipeline initialization. Are diffusers/transformers/accelerate installed?")
         pipe = None
         device = None
         raise ie
    except Exception as e:
        logger.exception("Failed to initialize Stable Diffusion XL Pipeline.")
        pipe = None
        device = None
        raise e

sdxl_resources = ResourceRequirements(
    model_name="John6666/holy-mix-illustriousxl-vibrant-anime-checkpoint-v1-sdxl",
    min_vram_gb=8.0,
    recommended_vram_gb=12.0,
    expects_pipeline_arg=True
)

@worker_function(resources=sdxl_resources)
def generate_image(ctx: ActionContext, pipeline: DiffusionPipeline, prompt_details: dict) -> bytes:
    """
    Generates an image based on a prompt using the pre-loaded SDXL pipeline.

    Args:
        ctx: The ActionContext provided by the worker.
        prompt_details: A dictionary containing 'prompt' (str) and 'seed' (int).

    Returns:
        PNG image data as bytes.
    """

    # _initialize_pipeline()

    # global pipe, device, pipeline_lock

    if pipeline is None:
        # This check is a safeguard; Worker core should ensure a valid pipeline is passed
        # if the function is designated as needing one and a required_model_id was processed.
        logger.error(f"Run {ctx.run_id} for generate_image received a None pipeline object.")
        raise ValueError("Pipeline object cannot be None for image generation.")

    prompt = prompt_details.get("prompt", "a beautiful landscape")
    negative_prompt = prompt_details.get("negative_prompt", "")
    seed = int(prompt_details.get("seed", 42))
    num_inference_steps = int(prompt_details.get("num_inference_steps", 28))
    guidance_scale = float(prompt_details.get("guidance_scale", 7.5))
    width = int(prompt_details.get("width", 1024))
    height = int(prompt_details.get("height", 1024))

    img_bytes: Optional[bytes] = None
    try:
        logger.info(f"[run_id={ctx.run_id}] Generating image for prompt: '{prompt}', seed: {seed} on device {device}")

        # Check for cancellation before starting inference
        if ctx.is_canceled():
            logger.warning(f"[run_id={ctx.run_id}] Cancellation detected before starting generation.")
            raise InterruptedError("Image generation cancelled before start")

        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.inference_mode():
            image_result = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=28
            ).images[0]

        if ctx.is_canceled():
            logger.warning(f"[run_id={ctx.run_id}] Cancellation detected after generation finished.")
            raise InterruptedError("Image generation cancelled after processing")

        # Convert PIL Image to PNG bytes
        buffer = BytesIO()
        image_result.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        logger.info(f"[run_id={ctx.run_id}] Image generation complete ({len(img_bytes)} bytes).")

    except Exception as e:
        logger.exception(f"[run_id={ctx.run_id}] Error during SDXL inference or image saving.")
        raise

    if img_bytes is None:
         raise RuntimeError("Image generation failed, resulting bytes are None.")

    return img_bytes


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
    

s3_upload_resources = ResourceRequirements()

@worker_function(resources=s3_upload_resources)
def upload_image_to_s3(ctx: ActionContext, upload_details: dict) -> Dict[str, str]:
    """
    Uploads provided image bytes to an S3 bucket configured via environment variables.

    Args:
        ctx: The ActionContext.
        upload_details: A dictionary containing:
            - 'image_bytes': The raw image data (bytes).
            - 'filename': (Optional) Desired filename base for the S3 object.

    Returns:
        A dictionary containing 's3_url' on success.
    """
    logger.info(f"[run_id={ctx.run_id}] Received S3 upload request.")

    image_bytes = upload_details.get("image_bytes")
    filename = upload_details.get("filename")

    if not image_bytes or not isinstance(image_bytes, bytes):
        raise ValueError("Missing or invalid 'image_bytes' in upload_details")
    
    if ctx.is_canceled():
        raise InterruptedError("Upload cancelled before starting")
    
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
        return {"s3_url": file_url}
    
    except Exception as e:
        logger.exception(f"[run_id={ctx.run_id}] Error during S3 upload.")
        raise


        
    

