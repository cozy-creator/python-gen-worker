import time
from typing import Iterator, List

import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function


class ImageGenInput(msgspec.Struct):
    """Input parameters for image generation."""

    positive_prompt: str
    model_ref: str
    num_images: int
    aspect_ratio: str
    steps: int
    cfg: float
    negative_prompt: str | None = None
    seed: int = 0


class ImageGenOutput(msgspec.Struct):
    """Output from image generation."""

    urls: List[str]


@worker_function(ResourceRequirements())
def image_gen_action(ctx: ActionContext, data: ImageGenInput) -> ImageGenOutput:
    """Example image generation function (dummy output)."""
    if ctx.is_canceled():
        raise InterruptedError("Task cancelled")

    urls = [
        f"https://example.com/generated-image-{i+1}.png"
        for i in range(data.num_images)
    ]
    return ImageGenOutput(urls=urls)


class AddInput(msgspec.Struct):
    a: int = 0
    b: int = 0


class AddOutput(msgspec.Struct):
    result: int


@worker_function(ResourceRequirements())
def add_numbers(ctx: ActionContext, data: AddInput) -> AddOutput:
    """Example function that adds two numbers."""
    if ctx.is_canceled():
        raise InterruptedError("Task cancelled")
    time.sleep(0.1)
    return AddOutput(result=data.a + data.b)


class MultiplyInput(msgspec.Struct):
    a: int = 0
    b: int = 0


class MultiplyOutput(msgspec.Struct):
    result: int


@worker_function(ResourceRequirements())
def multiply_numbers(ctx: ActionContext, data: MultiplyInput) -> MultiplyOutput:
    """Example function that multiplies two numbers."""
    if ctx.is_canceled():
        raise InterruptedError("Task cancelled")
    time.sleep(0.1)
    return MultiplyOutput(result=data.a * data.b)


class TokenDelta(msgspec.Struct):
    delta: str


class StreamInput(msgspec.Struct):
    text: str
    delay_ms: int = 25


@worker_function(ResourceRequirements())
def token_stream(ctx: ActionContext, data: StreamInput) -> Iterator[TokenDelta]:
    """Example incremental-output function (LLM-style token deltas)."""
    for ch in data.text:
        if ctx.is_canceled():
            raise InterruptedError("Task cancelled")
        yield TokenDelta(delta=ch)
        if data.delay_ms > 0:
            time.sleep(data.delay_ms / 1000.0)
