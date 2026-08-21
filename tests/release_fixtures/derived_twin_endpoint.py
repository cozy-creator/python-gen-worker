from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import (
    STATIC,
    Adapter,
    DistillationAdapter,
    ImageAsset,
    LoadContext,
    Model,
    RequestContext,
    ValidationError,
    entrypoint,
    lane,
)
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


_BUCKETS: dict[Size, int] = {Size.SMALL: 32, Size.LARGE: 64}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.LARGE
    guidance_scale: float | None = None
    num_inference_steps: int | None = None


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model_used: str


class TinyModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)
        self.defaults = ctx.defaults()


def _run(model: TinyModel, ctx: Any, *, steps: int, guidance: float,
         side: int, prompt: str) -> ImageAsset:
    with torch.inference_mode():
        result = model.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=side,
            height=side,
            callback_on_step_end=ctx.step_callback(steps),
            output_type="pil",
        )
    return ctx.save_image(result.images[0], format="png")


@entrypoint
def generate(ctx: RequestContext, payload: GenerateInput, model: TinyModel,
             turbo: DistillationAdapter | None,
             loras: list[Adapter]) -> ImageOutput:
    """Contract-file shape: ctx-first order, platform-injected adapter slots."""
    ctx.raise_if_cancelled()
    d = model.defaults
    if turbo is not None and not d.cfg:
        raise ValidationError(
            "this checkpoint is already distilled; a distillation adapter "
            "cannot be stacked on it"
        )
    config = turbo.defaults if turbo is not None else d
    distilled = not config.cfg
    steps = config.steps.resolve(payload.num_inference_steps or 16, ctx)
    guidance = 0.0 if distilled else d.guidance.resolve(payload.guidance_scale, ctx)
    image = _run(model, ctx, steps=int(steps), guidance=guidance,
                 side=_BUCKETS[payload.size], prompt=payload.prompt.strip())
    return ImageOutput(image=image, model_used=ctx.checkpoint_ref)


@entrypoint
def generate_turbo(ctx: RequestContext, payload: TurboInput,
                   model: TinyModel) -> ImageOutput:
    ctx.raise_if_cancelled()
    image = _run(model, ctx, steps=2, guidance=0.0,
                 side=_BUCKETS[payload.size], prompt=payload.prompt.strip())
    return ImageOutput(image=image, model_used=ctx.checkpoint_ref)
