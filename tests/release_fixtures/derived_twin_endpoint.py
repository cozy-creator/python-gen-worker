"""tiny_endpoint's TWIN, identical but for the LANE DECLARATION (pgw#1488).

Every line below is `tiny_endpoint.py`'s, with one difference: the model class
declares `lanes=()` — no layout contract at all — where the original names
`TINY_DIFFUSERS_FP32`. So the two derives run the same code, the same
payload enumeration and the same precision, and differ only in what the lane
is CALLED.

That is the wire-compatibility measurement the ruling needs. `cg-graph-v1` is
a content hash of the canonical trace plus its ingress and passes, and
`cg-key-v1` is (graph, sm, toolchain): the contract handle is in NEITHER. The
test asserts what that predicts — byte-identical graph hashes across the two
fixtures, under two different lane names — which is why declaring a contract
can be made optional without rekeying a single existing artifact.
"""


from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import (
    Adapter,
    DistillationAdapter,
    ImageAsset,
    LoadContext,
    Model,
    RequestContext,
    ValidationError,
    entrypoint,
)
from gen_worker.models import SDXL


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


_BUCKETS: dict[Size, int] = {Size.SMALL: 32, Size.LARGE: 64}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    # LARGE deliberately: the payload DEFAULT differs from enum declaration
    # order, so pgw#1384's default-first document ordering is observable.
    size: Size = Size.LARGE
    guidance_scale: float | None = None
    num_inference_steps: int | None = None


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model_used: str


class TinyModel(Model[SDXL], lanes=()):
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
    """Contract-file shape: ctx-first order, platform-injected adapter slots.

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
