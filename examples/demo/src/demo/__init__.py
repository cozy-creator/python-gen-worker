import logging
from io import BytesIO

import msgspec
from PIL import Image

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.types import Asset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class GenerateImageInput(msgspec.Struct):
    prompt: str = "a default prompt"
    seed: int = 42

class GenerateImageOutput(msgspec.Struct):
    image: Asset

class TextProcessInput(msgspec.Struct):
    text: str

class TextProcessOutput(msgspec.Struct):
    original: str
    processed: str
    length: int

class FailInput(msgspec.Struct):
    should_fail: bool = False

class FailOutput(msgspec.Struct):
    message: str


@worker_function(ResourceRequirements(requires_gpu=False))
def generate_image_demo(ctx: ActionContext, payload: GenerateImageInput) -> GenerateImageOutput:
    """Dummy image generator that returns a saved Asset (no model inference)."""
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    logger.info(
        "[run_id=%s] generate_image_demo prompt=%r seed=%d",
        ctx.run_id,
        payload.prompt,
        payload.seed,
    )

    img = Image.new("RGB", (256, 256), color="purple")
    buf = BytesIO()
    img.save(buf, format="PNG")
    out = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/demo.png", buf.getvalue())
    return GenerateImageOutput(image=out)


@worker_function(ResourceRequirements())
def simple_text_processor(ctx: ActionContext, payload: TextProcessInput) -> TextProcessOutput:
    logger.info("[run_id=%s] simple_text_processor", ctx.run_id)
    if ctx.is_canceled():
        raise InterruptedError("canceled")
    processed = payload.text.upper()
    return TextProcessOutput(original=payload.text, processed=processed, length=len(payload.text))


@worker_function(ResourceRequirements())
def potentially_failing_task(ctx: ActionContext, payload: FailInput) -> FailOutput:
    logger.info("[run_id=%s] potentially_failing_task", ctx.run_id)
    if ctx.is_canceled():
        raise InterruptedError("canceled")
    if payload.should_fail:
        raise ValueError("task was instructed to fail")
    return FailOutput(message="Task succeeded!")
