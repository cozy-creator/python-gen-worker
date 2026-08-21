from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class VideoModel(
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


class AideModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(32)))},
):
    """Marks nothing, and it STILL hydrates -- the drive calls it."""

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
        assert aide.net is not None
    return Out(model_used=ctx.checkpoint_ref)
