"""One entrypoint, TWO model slots, backed by TWO checkpoints (pgw#1508).

h3's `generate` shape: `video` is the primary (a diffusers pipeline out of the
DiT's tree) and `aide` is an auxiliary model with a checkpoint of its own --
h3's is the RIFE interpolator at `rife-4.25`. The serving binding table has
been per-slot since 0.9.0; the derive used to hand every slot the PRIMARY's
tree, so the aide tried to build itself out of the wrong checkpoint.

The aide here loads a bare `UNet2DConditionModel` from its own tree, which the
primary tree does NOT contain at the top level -- so pointing it at the primary
fails the way h3's RIFE did, and pointing it at its own succeeds.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from gen_worker import LoadContext, Model, RequestContext, entrypoint
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class VideoModel(Model[SDXL], lanes=(TINY_DIFFUSERS_FP32,)):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


class AideModel(Model[SDXL], eager_only="an interpolator has no AOT story here"):
    """Eager-permanent, and it STILL hydrates -- the drive calls it.

    Skipping hydration for an eager_only slot was the tempting shortcut and is
    wrong: the entrypoint body runs at trace and calls this model, so a `None`
    here fails one layer later with a worse message.
    """

    net: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.net = ctx.load(UNet2DConditionModel)


@entrypoint
def generate(
    ctx: RequestContext, payload: In, video: VideoModel, aide: AideModel
) -> Out:
    ctx.raise_if_cancelled()
    with torch.inference_mode():
        video.pipe(
            prompt=payload.prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            width=32,
            height=32,
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )
        # The aide is CALLED, which is why it has to be real at trace.
        assert aide.net is not None
    return Out(model_used=ctx.checkpoint_ref)
