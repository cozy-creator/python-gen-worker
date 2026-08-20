"""sd15's SHAPE FAN, in miniature: three aspects x two CFG modes (pgw#1548).

The real sd15 endpoint derives 14 graph specializations — 2 (CFG) x 7 (aspect
bucket) — and sdxl 18. Nothing about that count is declared anywhere; it falls
out of the payload enumeration driving the marked UNet at a different shape
each pass. This fixture reproduces exactly that structure at fixture scale so
the collapse can be measured through the REAL derive, not a stand-in.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import LoadContext, Model, RequestContext, entrypoint
from gen_worker.models import SDXL
from gen_worker.models.model_types import SD15_DIFFUSERS_BF16

class Aspect(StrEnum):
    """The aspect axis, spelled the way sd15's own `AspectRatio` is.

    A shape-bearing string axis is a StrEnum, never a string Literal: the
    derive enumerates enums and NUMERIC literals, and deliberately refuses to
    cross-product string literals (they name host-side policy, not shape).
    """

    SQUARE = "square"
    TALL = "tall"
    WIDE = "wide"


#: latent side = pixel side here (the fixture VAE has one block), so these are
#: the three aspect buckets the UNet actually sees.
SHAPES: dict[str, tuple[int, int]] = {
    Aspect.SQUARE: (64, 64),
    Aspect.TALL: (80, 48),
    Aspect.WIDE: (48, 80),
}


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "a cat"
    #: The aspect axis. Each value is one bucket, and the enumeration drives
    #: the marked module once per value.
    aspect: Aspect = Aspect.SQUARE
    #: The CFG axis: guided runs concatenate cond+uncond (batch 2), the
    #: guidance-free path does not (batch 1). Spelled as an int enum because
    #: msgspec refuses a bool Literal.
    guided: Literal[1, 0] = 1


class Out(msgspec.Struct):
    model_used: str


class FanShaped(Model[SDXL], lanes=(SD15_DIFFUSERS_BF16,)):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: FanShaped) -> Out:
    ctx.raise_if_cancelled()
    height, width = SHAPES[payload.aspect]
    with torch.inference_mode():
        model.pipe(
            prompt=payload.prompt,
            num_inference_steps=1,
            guidance_scale=7.5 if payload.guided else 1.0,
            width=width,
            height=height,
            callback_on_step_end=ctx.step_callback(1),
            output_type="latent",
        )
    return Out(model_used=ctx.checkpoint_ref)
