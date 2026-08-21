from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class Out(msgspec.Struct):
    model_used: str


class UnservableModel(
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


def _side_for(size: Size) -> int:

    if size is Size.LARGE:
        raise ValueError("this checkpoint cannot serve the large bucket")
    return 32


@entrypoint
def generate(ctx: RequestContext, payload: In, model: UnservableModel) -> Out:
    ctx.raise_if_cancelled()
    side = _side_for(payload.size)
    with torch.inference_mode():
        model.pipe(
            prompt=payload.prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            width=side,
            height=side,
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )
    return Out(model_used=ctx.checkpoint_ref)
