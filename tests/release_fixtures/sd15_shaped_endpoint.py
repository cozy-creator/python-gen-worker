"""sd15's shape exactly: a bf16 lane whose patterns are UNET-INTERNAL.

pgw#1528. The real library contract `sd15.diffusers-bf16@1` states its tensors
as `conv_in.weight`, `down_blocks.…`, `mid_block.…` — the DENOISER's own
parameter names, with no `unet.` prefix. Every fixture written before this one
used component-prefixed patterns (`unet.conv_out.weight`), which is a spelling
the shipped contracts do not use.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from gen_worker.models.model_types import SD15_DIFFUSERS_BF16

#: THE REAL SHIPPED CONTRACT, imported — not a path outside the repo and not a
#: hand-written stand-in. pgw#1530 happened because a fixture invented a
#: pattern spelling the fleet does not use, so this fixture takes the object.
SD15_BF16 = SD15_DIFFUSERS_BF16


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class Sd15Shaped(
    Model[SDXL],
    lanes={SD15_BF16: lane(request=const(MiB(64)) + per_mp_batch(MiB(16)))},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: Sd15Shaped) -> Out:
    ctx.raise_if_cancelled()
    with torch.inference_mode():
        model.pipe(
            prompt=payload.prompt,
            num_inference_steps=2,
            guidance_scale=7.5,
            width=64,
            height=64,
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )
    return Out(model_used=ctx.checkpoint_ref)
