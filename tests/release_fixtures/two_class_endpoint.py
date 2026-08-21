"""TWO compile-marking model classes on ONE lane — the qwen-image shape (pgw#1650).

Both classes own ``.unet``, both call ``ctx.compile`` on it, and both declare
the SAME lane: a compile target is an attribute PATH on one pipeline object, so
the two arms cannot be one class (pgw#1112's finding), and the two checkpoints
are the same layout, so they cannot be two lanes either. Each class binds its
OWN checkpoint tree.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class EditIn(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    strength: float = 0.5


class Out(msgspec.Struct):
    model_used: str


class EditModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(128)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    """The edit checkpoint — its own tree, `--checkpoint EditModel=<path>`."""

    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


class PrimaryModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    """The text-to-image checkpoint — the bare `--checkpoint` tree."""

    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


def _run(model: Any, ctx: Any, prompt: str) -> None:
    with torch.inference_mode():
        model.pipe(
            prompt=prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            width=32,
            height=32,
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )


@entrypoint
def generate(ctx: RequestContext, payload: In, model: PrimaryModel) -> Out:
    ctx.raise_if_cancelled()
    _run(model, ctx, payload.prompt)
    return Out(model_used=ctx.checkpoint_ref)


@entrypoint
def edit(ctx: RequestContext, payload: EditIn, model: EditModel) -> Out:
    ctx.raise_if_cancelled()
    _run(model, ctx, payload.prompt)
    return Out(model_used=ctx.checkpoint_ref)
