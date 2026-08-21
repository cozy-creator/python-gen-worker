"""sd15's shape exactly: a bf16 lane whose patterns are UNET-INTERNAL."""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
#: THE REAL RATIFIED PAIR (pgw#1621): a lane is `(topology, quant)`, both
#: halves documents in the vendored `spec/v2` corpus. `SD15_DIFFUSERS_BF16`
#: was a v1 Contract OBJECT and is deleted with the v1 vocabulary; the
#: spelling it used to carry survives only as a display name.
SD15_DIFFUSERS_BF16 = ("sd15.diffusers@1", "plain.bf16@1")

#: THE REAL RATIFIED PAIR, not a hand-written stand-in. pgw#1530 happened
#: because a fixture invented a spelling the fleet does not use; under v2 a
#: fixture CANNOT invent one — both halves must be in the vendored corpus or
#: `parse_lane_stamp` refuses at class definition.
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
