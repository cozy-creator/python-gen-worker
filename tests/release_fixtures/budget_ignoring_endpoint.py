"""An endpoint whose sampling loop NEVER calls the step callback (pgw#1671).

The shape se#840 shipped and could not see: the entrypoint asks for no
``ctx.step_callback`` at all, so the derive's step budget has nothing to raise
out of and the drive runs the whole schedule against fake weights — silently.
On a 17.5B model at 2048x2048 that is 139 s per payload and reads as a hung
``torch.export`` the drive never reached.

The unet call count is written to ``$GEN_WORKER_ROUND_PROBE`` so a test can
assert what the drive actually did, not what it reported.
"""

from __future__ import annotations

import os
from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32

#: What the drive is asked for. Every one of these is a full pass of the marked
#: module, and without a budget the drive pays all of them.
STEPS = 8


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


def _count_one(*_: Any, **__: Any) -> None:
    probe = os.environ.get("GEN_WORKER_ROUND_PROBE")
    if not probe:
        return
    with open(probe, "a", encoding="utf-8") as handle:
        handle.write("1\n")


class BudgetIgnoringModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet.register_forward_pre_hook(_count_one)
        self.pipe.unet = ctx.compile(self.pipe.unet)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: BudgetIgnoringModel) -> Out:
    ctx.raise_if_cancelled()
    with torch.inference_mode():
        # NO `callback_on_step_end=ctx.step_callback(...)`. That omission is
        # the whole fixture.
        model.pipe(
            prompt=payload.prompt,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            width=32,
            height=32,
            output_type="pil",
        )
    return Out(model_used=ctx.checkpoint_ref)
