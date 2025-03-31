import os
import sys
import time
import json
import inspect
import traceback
import logging
import torch
import tempfile
import numpy as np
from tqdm import tqdm
import asyncio

from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    FluxPipeline,
    FluxControlNetPipeline,
)
from compel import Compel, ReturnedEmbeddingsType
import boto3
import tempfile
import blake3
from io import BytesIO
from botocore.client import Config


# Dummy aspect ratio conversion
def aspect_ratio_to_dimensions(aspect_ratio: str, class_name: str):
    if aspect_ratio == "1/1":
        return 1024, 1024
    elif aspect_ratio == "16/9":
        return 1024, 576
    else:
        return 1024, 1024

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class ProgressCallback:
    def __init__(self, num_steps, num_images):
        self.num_steps = num_steps
        self.num_images = num_images
        self.total_steps = num_steps * num_images
        self.pbar = tqdm(total=num_steps, desc="Generating images")
        self.last_update = 0

    def on_step_end(self, pipeline, step_number, timestep, callback_kwargs):
        overall_step = self.last_update + step_number
        scaled_step = int((overall_step / self.total_steps) * self.num_steps)
        if scaled_step > self.pbar.n:
            self.pbar.update(scaled_step - self.pbar.n)
        return {"latents": callback_kwargs.get("latents")}

    def on_image_complete(self):
        self.last_update += self.num_steps

    def close(self):
        self.pbar.n = self.num_steps
        self.pbar.refresh()
        self.pbar.close()

async def image_gen_action(ctx, data: dict, model_manager) -> dict:
    """
    Image generation action for generating images with a diffusion pipeline.
    
    Expects input data with keys:
      - model_id: Identifier for the model (default "sdxl_base")
      - positive_prompt: The text prompt for generation
      - negative_prompt: (Optional) Negative prompt
      - aspect_ratio: e.g. "1/1" or "16/9"
      - num_images: Number of images to generate
      - enhance_prompt: Whether to enhance the prompt (bool)
      - style: Prompt style if enhancing
      - random_seed: (Optional) Seed for generation
      - num_inference_steps: (Optional) Steps for diffusion (default 30)
      - guidance_scale: (Optional) Guidance scale (default 7.5)
      - (Optional) controlnet and LoRA parameters as needed.
      
    It loads the model via model_manager, sets up inference parameters,
    and returns a dict with a list of file:// URLs where generated images are saved.
    """
    try:
        # Extract input parameters with defaults.
        model_id = data.get("model_id", "sdxl_base")
        positive_prompt = data.get("positive_prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        aspect_ratio = data.get("aspect_ratio", "1/1")
        num_images = data.get("num_images", 1)
        enhance_prompt = data.get("enhance_prompt", False)
        style = data.get("style", "cinematic")
        random_seed = data.get("random_seed", None)
        num_inference_steps = data.get("num_inference_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        
        logger.info(f"Image generation request: {positive_prompt}")
        
        # Load model via memory manager.
        pipeline = await model_manager.load(model_id)
        if pipeline is None:
            return {"error": f"Failed to load model {model_id}"}
        class_name = pipeline.__class__.__name__
        logger.info(f"Using pipeline class: {class_name}")
        
        # Optional prompt enhancement. TODO: Implement prompt enhancement from previous code.
        if enhance_prompt:
            positive_prompt = positive_prompt + " in " + style + " style"
        
        # Initialize Compel for embedding generation if applicable.
        compel = None
        if class_name in ["StableDiffusionPipeline", "StableDiffusionXLPipeline"]:
            try:
                compel = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
            except Exception as e:
                logger.error(f"Error initializing Compel: {e}")
        
        # Determine image dimensions.
        width, height = aspect_ratio_to_dimensions(aspect_ratio, class_name)
        
        # Setup generation parameters.
        gen_params = {
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "output_type": "pil",
            "guidance_scale": guidance_scale,
        }
        if random_seed is not None:
            gen_params["generator"] = torch.Generator().manual_seed(random_seed)
        
        # Use Compel if available.
        if compel:
            if negative_prompt:
                conditioning, pooled = compel([positive_prompt, negative_prompt])
                gen_params["prompt_embeds"] = conditioning[0:1]
                gen_params["pooled_prompt_embeds"] = pooled[0:1]
                gen_params["negative_prompt_embeds"] = conditioning[1:2]
                gen_params["negative_pooled_prompt_embeds"] = pooled[1:2]
            else:
                conditioning, pooled = compel([positive_prompt])
                gen_params["prompt_embeds"] = conditioning
                gen_params["pooled_prompt_embeds"] = pooled
        else:
            gen_params["prompt"] = positive_prompt
            if negative_prompt:
                gen_params["negative_prompt"] = negative_prompt
        
        # Setup progress callback.
        progress_callback = ProgressCallback(num_inference_steps, num_images)
        pipeline.set_progress_bar_config(disable=True)
        
        # Run inference.
        generated_images = []
        for _ in range(num_images):
            if "callback_on_step_end" in inspect.signature(pipeline.__call__).parameters:
                with torch.no_grad():
                    output = pipeline(
                        **gen_params,
                        callback_on_step_end=progress_callback.on_step_end,
                        callback_on_step_end_tensor_inputs=["latents"]
                    ).images
                progress_callback.on_image_complete()
            else:
                with torch.no_grad():
                    output = pipeline(**gen_params).images
            generated_images.append(output[0])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        progress_callback.close()


        # Upload images to S3
        upload_to_s3 = os.environ.get("UPLOAD_TO_S3", "false").lower() == "true"
        if upload_to_s3:
            file_urls = upload_images_to_s3(generated_images)
        else:
            output_dir = "./generated_images"
            os.makedirs(output_dir, exist_ok=True)
            file_urls = []
            for img in generated_images:
                hashed_name = get_hashed_filename(img)
                file_path = os.path.join(output_dir, hashed_name)
                img.save(file_path, format="PNG")
                abs_path = os.path.abspath(file_path)
                file_urls.append(f"file://{abs_path}")
        
        
        return {"urls": file_urls}
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error generating images: {e}")
    

def get_hashed_filename(img, extension=".png"):
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    hash_hex = blake3.blake3(data).hexdigest()
    return f"{hash_hex}{extension}"
    

def upload_images_to_s3(generated_images):
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")
    
    region = os.environ.get("S3_REGION")
    
    s3_client = boto3.client("s3",
        region_name=region,
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version='s3v4')
    )
    
    file_urls = []
    for idx, img in enumerate(generated_images):
        # get the hash of the file using blake3 so as to avoid duplicates
        file_hash = get_hashed_filename(img)

        s3_key = f"orchestrator_images_test/{file_hash}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            img.save(temp_file.name, format="PNG")
            tmp_file = temp_file.name

        s3_client.upload_file(
            tmp_file,
            bucket_name,
            s3_key,
            ExtraArgs={"ACL": "public-read", "ContentType": "image/png"}
        )

        file_url = f"https://{bucket_name}.{region}.digitaloceanspaces.com/{s3_key}"
        file_urls.append(file_url)
        os.remove(tmp_file)

    return file_urls
        
        

    
    

def register_functions(worker, model_manager) -> None:
    """
    Register image generation function with the worker.
    Wrap image_gen_action to include the model memory manager.
    """
    def wrapped_image_gen_action(ctx, data: dict) -> dict:
        return asyncio.run(image_gen_action(ctx, data, model_manager))
    
    worker.register_function(wrapped_image_gen_action, "image_gen_action")
    logger.info("Registered image_gen_action with the worker using ModelMemoryManager.")
