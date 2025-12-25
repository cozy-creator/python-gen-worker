from __future__ import annotations

from typing import List, Tuple

import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function


class ImageGenInput(msgspec.Struct):
    prompt: str
    model_ref: str
    aspect_ratio: str = "1/1"
    num_images: int = 1
    steps: int = 30
    cfg: float = 7.5
    negative_prompt: str | None = None
    seed: int | None = None


class ImageGenOutput(msgspec.Struct):
    urls: List[str]


def aspect_ratio_to_dimensions(aspect_ratio: str) -> Tuple[int, int]:
    if aspect_ratio == "16/9":
        return 1024, 576
    if aspect_ratio == "9/16":
        return 576, 1024
    return 1024, 1024


@worker_function(ResourceRequirements(requires_gpu=True))
def generate_image(ctx: ActionContext, payload: ImageGenInput) -> ImageGenOutput:
    if ctx.is_canceled():
        raise InterruptedError("canceled")

    if payload.num_images < 1 or payload.num_images > 8:
        raise ValueError("num_images must be between 1 and 8")

    width, height = aspect_ratio_to_dimensions(payload.aspect_ratio)
    urls = [
        f"https://example.com/gen/{payload.model_ref}/{width}x{height}/{i}.png"
        for i in range(payload.num_images)
    ]
    return ImageGenOutput(urls=urls)
