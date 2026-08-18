"""A main_v2-shaped Model + entrypoints over the tiny pipeline (derive fixture).

Deliberately spelled like the Paul-reviewed sdxl ``main_v2.py``: the stateful
half is a ``Model[SDXL]`` subclass with ``lanes=`` class kwargs and an
imperative ``ctx.compile`` mark inside ``load``; the stateless half is free
``@entrypoint`` functions ``(payload, model, ctx)``. Trace coverage is
auto-enumerated from the payload schemas (the ``Size`` enum is this
fixture's aspect-ratio analogue); no samples surface, no catalog.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import Adapter, ImageAsset, Model, RequestContext, ValidationError, entrypoint
from gen_worker.models import SDXL
from gen_worker.models.model_types import register_contract_dtype

LANE = "tiny.diffusers-fp32@1"
register_contract_dtype(LANE, torch.float32)


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


_BUCKETS: dict[Size, int] = {Size.SMALL: 32, Size.LARGE: 64}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL
    guidance_scale: float | None = None
    num_inference_steps: int | None = None


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model_used: str


class TinyModel(Model[SDXL], lanes=(LANE,)):
    def load(self, ctx: Any) -> None:
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


@entrypoint  # type: ignore[operator]
def generate(ctx: RequestContext, payload: GenerateInput, model: TinyModel,
             turbo: Adapter | None, loras: list[Adapter]) -> ImageOutput:
    """Contract-file shape: ctx-first order, platform-injected facts.

    A riding distillation adapter (or a cfg-off checkpoint) serves the
    guidance-free batch-1 arm -- the derive's binding enumeration reaches it
    without any real adapter bytes.
    """
    ctx.raise_if_cancelled()
    d = model.defaults
    if turbo is not None and not d.cfg:
        raise ValidationError(
            "this checkpoint is already distilled; a distillation adapter "
            "cannot be stacked on it"
        )
    recipe = turbo.defaults if turbo is not None else d
    distilled = not recipe.cfg
    steps = recipe.steps.resolve(payload.num_inference_steps or 2, ctx)
    guidance = 0.0 if distilled else d.guidance.resolve(payload.guidance_scale, ctx)
    image = _run(model, ctx, steps=int(steps), guidance=guidance,
                 side=_BUCKETS[payload.size], prompt=payload.prompt.strip())
    # checkpoint_ref lands on the serving RequestContext with pgw#1372.
    return ImageOutput(image=image, model_used=ctx.checkpoint_ref)  # type: ignore[attr-defined]


@entrypoint  # type: ignore[operator]
def generate_turbo(payload: TurboInput, model: TinyModel, ctx: Any) -> ImageOutput:
    ctx.raise_if_cancelled()
    image = _run(model, ctx, steps=2, guidance=0.0,
                 side=_BUCKETS[payload.size], prompt=payload.prompt.strip())
    return ImageOutput(image=image, model_used=ctx.checkpoint_ref)
