import logging
from typing import List, Optional
import time

from pydantic import BaseModel, Field
from gen_orchestrator import Worker, ActionContext


# Configure logging
logging.basicConfig(level=logging.INFO)


class ImageGenInput(BaseModel):
    """Input parameters for image generation."""
    positive_prompt: str = Field(..., description="Prompt describing what to generate")
    negative_prompt: Optional[str] = Field(None, description="Prompt specifying what to avoid")
    model_id: str = Field(..., description="Which model to use for generation")
    num_images: int = Field(..., description="How many images to generate")
    aspect_ratio: str = Field(..., description="Width/height ratio (e.g. '1:1', '16:9')")
    steps: int = Field(..., description="Number of steps in the diffusion process")
    cfg: float = Field(..., description="Classifier-free guidance scale")
    seed: int = Field(0, description="Seed for the random number generator")


class ImageGenOutput(BaseModel):
    """Output from image generation."""
    urls: List[str] = Field(..., description="URLs of the generated images")


def image_gen_action(ctx: ActionContext, data: dict) -> dict:
    """Image generation function that processes requests similar to the Go example."""
    # Parse and validate input using Pydantic
    try:
        input_data = ImageGenInput(**data)
    except Exception as e:
        return {"error": f"Invalid input: {str(e)}"}
    
    # Log processing information
    logging.info(f"Processing image generation request for model: {input_data.model_id}")
    logging.info(f"Prompt: {input_data.positive_prompt}")
    
    # In a real implementation, we would actually generate images here
    # For now, just create dummy URLs
    urls = [f"http://example.com/generated-image-{i+1}.png" for i in range(input_data.num_images)]
    
    # Create and validate output
    output = ImageGenOutput(urls=urls)
    
    # Return as dictionary for msgpack serialization
    return output.model_dump()


def add_numbers(ctx: ActionContext, data: dict[str, int]) -> dict[str, int]:
    """Example function that adds two numbers"""
    # Extract inputs from the data
    a = data.get('a', 0)
    b = data.get('b', 0)
    
    # Perform computation
    result = a + b
    
    # Simulate some work
    time.sleep(1)
    
    # Return result
    return {'result': result}


def multiply_numbers(ctx: ActionContext, data: dict[str, int]) -> dict[str, int]:
    """Example function that multiplies two numbers"""
    # Extract inputs from the data
    a = data.get('a', 0)
    b = data.get('b', 0)
    
    # Perform computation
    result = a * b
    
    # Simulate some work
    time.sleep(1)
    
    # Return result
    return {'result': result}


def main():
    """Run the example worker"""
    # Create a worker that connects to the scheduler
    worker = Worker(
        scheduler_addr="localhost:8080",  # Change to your scheduler address
        worker_id="image-gen-worker-1"
    )
    
    # Register functions
    worker.register_function(image_gen_action, "github.com/cozy-creator/gen-orchestrator/pkg/action.ActionFunc[github.com/cozy-creator/gen-orchestrator/benchmarks/image_gen/app.ImageGenInput,github.com/cozy-creator/gen-orchestrator/benchmarks/image_gen/app.ImageGenOutput]")
    worker.register_function(add_numbers)
    worker.register_function(multiply_numbers)
    
    # Run the worker (this blocks until interrupted)
    try:
        worker.run()
    except KeyboardInterrupt:
        print("Worker interrupted, shutting down")
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
