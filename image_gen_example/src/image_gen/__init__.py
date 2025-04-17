import time
from PIL import Image
from io import BytesIO
import logging
import torch
from diffusers import StableDiffusionXLPipeline
from typing import Optional
from gen_worker import worker_function, ResourceRequirements, ActionContext


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

pipe = None
device = None

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
    recommended_vram_gb=12.0
)

@worker_function(resources=sdxl_resources)
def generate_image(ctx: ActionContext, prompt_details: dict) -> bytes:
    """
    Generates an image based on a prompt using the pre-loaded SDXL pipeline.

    Args:
        ctx: The ActionContext provided by the worker.
        prompt_details: A dictionary containing 'prompt' (str) and 'seed' (int).

    Returns:
        PNG image data as bytes.
    """

    _initialize_pipeline()

    global pipe, device

    if pipe is None or device is None:
        logger.error("SDXL Pipeline is not initialized. Cannot generate image.")
        raise RuntimeError("SDXL Pipeline failed to initialize during worker startup.")

    prompt = prompt_details.get("prompt", "a default prompt")
    seed = prompt_details.get("seed", 42)

    logger.info(f"[run_id={ctx.run_id}] Generating image for prompt: '{prompt}', seed: {seed} on device {device}")

    # Check for cancellation before starting inference
    if ctx.is_canceled():
        logger.warning(f"[run_id={ctx.run_id}] Cancellation detected before starting generation.")
        raise InterruptedError("Image generation cancelled before start")

    generator = torch.Generator(device=device).manual_seed(seed)

    img_bytes: Optional[bytes] = None
    try:
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