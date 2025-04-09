# example_functions.py

import time
from PIL import Image
from io import BytesIO
import logging

# Import the decorator and resource specifier from the gen_worker package
# Assumes 'gen_worker' is installed or available in the Python path
try:
    from gen_worker import worker_function, ResourceRequirements, ActionContext
except ImportError:
    # Fallback for local testing if path isn't set up perfectly
    print("WARNING: Could not import from gen_worker. Using dummy decorators.")
    class ResourceRequirements: # Dummy
        def __init__(self, **kwargs): pass
    def worker_function(resources=None): # Dummy
        def decorator(func): return func
        return decorator
    class ActionContext: # Dummy
         def is_canceled(self): return False
         @property
         def run_id(self): return "dummy_run_id"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Configure logging for the example

# --- Example 1: Function with specific resource requirements ---

# Define the resources needed
sdxl_resources = ResourceRequirements(
    model_name="stabilityai/stable-diffusion-xl-base-1.0",
    min_vram_gb=8.0,
    recommended_vram_gb=12.0
)

@worker_function(resources=sdxl_resources)
def generate_image_demo(ctx: ActionContext, prompt_details: dict) -> bytes:
    """
    Generates an image based on a prompt.
    (Actual model loading/inference would happen here or be managed by the worker)

    Args:
        ctx: The ActionContext provided by the worker.
        prompt_details: A dictionary containing 'prompt' (str) and 'seed' (int).

    Returns:
        PNG image data as bytes.
    """
    prompt = prompt_details.get("prompt", "a default prompt")
    seed = prompt_details.get("seed", 42)

    logger.info(f"[run_id={ctx.run_id}] Generating image for prompt: '{prompt}', seed: {seed}")

    # --- Placeholder for actual Stable Diffusion Inference ---
    # Check for cancellation periodically during long tasks
    if ctx.is_canceled():
        logger.warning(f"[run_id={ctx.run_id}] Cancellation detected before starting generation.")
        raise InterruptedError("Image generation cancelled before start")

    print("Simulating model loading/inference...")
    time.sleep(2) # Simulate work

    if ctx.is_canceled():
        logger.warning(f"[run_id={ctx.run_id}] Cancellation detected during generation.")
        raise InterruptedError("Image generation cancelled during processing")

    # Create a dummy image (replace with real model output)
    img = Image.new('RGB', (256, 256), color = 'purple')
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    # --- End Placeholder ---

    logger.info(f"[run_id={ctx.run_id}] Image generation complete.")
    return img_bytes

# --- Example 2: Function with minimal/no specific resources ---

@worker_function() # No specific resources needed, uses default empty ResourceRequirements
def simple_text_processor(ctx: ActionContext, text_input: str) -> dict:
    """
    Processes text input.

    Args:
        ctx: The ActionContext.
        text_input: The input string.

    Returns:
        A dictionary with processing results.
    """
    logger.info(f"[run_id={ctx.run_id}] Processing text: '{text_input}'")
    if ctx.is_canceled():
         raise InterruptedError("Text processing cancelled")
    processed_text = text_input.upper()
    char_count = len(text_input)
    time.sleep(0.5) # Simulate work
    logger.info(f"[run_id={ctx.run_id}] Text processing complete.")
    return {"original": text_input, "processed": processed_text, "length": char_count}

# --- Example 3: A function that might fail ---

@worker_function()
def potentially_failing_task(ctx: ActionContext, data: dict) -> str:
    """
    A task that might fail based on input.
    """
    logger.info(f"[run_id={ctx.run_id}] Running potentially failing task with data: {data}")
    if data.get("should_fail", False):
        logger.error(f"[run_id={ctx.run_id}] Task instructed to fail!")
        raise ValueError("This task was instructed to fail.")

    if ctx.is_canceled():
        raise InterruptedError("Failing task cancelled")

    logger.info(f"[run_id={ctx.run_id}] Task completed successfully.")
    return "Task succeeded!"

# You would typically *not* call these directly. The Worker class discovers
# and calls them based on scheduler requests.
# The code below is just for demonstrating the functions exist.
if __name__ == "__main__":
    print("Demonstrating functions locally (Worker/Runner not involved):")

    # Create dummy context for local testing
    dummy_ctx = ActionContext("local-test-run")

    # Example 1 call
    try:
        print("\nTesting generate_image...")
        png_bytes = generate_image_demo(dummy_ctx, {"prompt": "a cat", "seed": 123})
        print(f"generate_image returned {len(png_bytes)} bytes")
    except Exception as e:
        print(f"generate_image failed: {e}")

    # Example 2 call
    try:
        print("\nTesting simple_text_processor...")
        result = simple_text_processor(dummy_ctx, "hello world")
        print(f"simple_text_processor returned: {result}")
    except Exception as e:
        print(f"simple_text_processor failed: {e}")

    # Example 3 call (success)
    try:
        print("\nTesting potentially_failing_task (success case)...")
        result = potentially_failing_task(dummy_ctx, {"should_fail": False})
        print(f"potentially_failing_task returned: {result}")
    except Exception as e:
        print(f"potentially_failing_task failed unexpectedly: {e}")

    # Example 3 call (failure)
    try:
        print("\nTesting potentially_failing_task (failure case)...")
        result = potentially_failing_task(dummy_ctx, {"should_fail": True})
        print(f"potentially_failing_task returned: {result}")
    except Exception as e:
        print(f"potentially_failing_task failed as expected: {e}")
