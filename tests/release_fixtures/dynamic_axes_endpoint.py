from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import DYNAMIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
#: THE REAL RATIFIED PAIR (pgw#1621): a lane is `(topology, quant)`, both
#: halves documents in the vendored `spec/v2` corpus. `SD15_DIFFUSERS_BF16`
#: was a v1 Contract OBJECT and is deleted with the v1 vocabulary; the
#: spelling it used to carry survives only as a display name.
SD15_DIFFUSERS_BF16 = ("sd15.diffusers@1", "plain.bf16@1")

class Aspect(StrEnum):
    """The aspect axis, spelled the way sd15's own `AspectRatio` is."""

    SQUARE = "square"
    TALL = "tall"
    WIDE = "wide"


SHAPES: dict[str, tuple[int, int]] = {
    Aspect.SQUARE: (64, 64),
    Aspect.TALL: (80, 48),
    Aspect.WIDE: (48, 80),
}


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "a cat"
    aspect: Aspect = Aspect.SQUARE
    guided: Literal[1, 0] = 1


class Out(msgspec.Struct):
    model_used: str


class FanShaped(
    Model[SDXL],
    lanes={SD15_DIFFUSERS_BF16: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": DYNAMIC},
):
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
