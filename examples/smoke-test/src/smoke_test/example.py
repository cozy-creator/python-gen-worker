import time
import base64
from typing import Iterator, List

import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.types import Asset


class ImageGenInput(msgspec.Struct):
    """Input parameters for image generation."""

    prompt: str = "a tiny test image"
    width: int = 1
    height: int = 1
    num_images: int = 1


class ImageGenOutput(msgspec.Struct):
    """Output from image generation (real file output)."""

    images: List[Asset]


# 1x1 PNG (valid image bytes). This is deterministic and avoids pulling in Pillow.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAAEnnX8AAAAASUVORK5CYII="
)


@worker_function(ResourceRequirements())
def image_gen_action(ctx: ActionContext, data: ImageGenInput) -> ImageGenOutput:
    """Example image generation function that returns real output Assets.

    This is a smoke test: it does not run ML inference. It validates:
    - request/response msgspec serialization
    - output file creation + upload path wiring (ctx.save_bytes)
    """
    if ctx.is_canceled():
        raise InterruptedError("Task cancelled")

    out: List[Asset] = []
    n = max(1, int(data.num_images))
    for i in range(n):
        ref = f"runs/{ctx.run_id}/outputs/image-{i+1}.png"
        out.append(ctx.save_bytes(ref, _PNG_1X1))
    return ImageGenOutput(images=out)


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
