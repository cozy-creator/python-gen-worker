"""Red fixture: ctx.compile of an attribute that does not exist."""

from __future__ import annotations

from typing import Any

import msgspec
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class BadMark(
    Model[Any],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[Any]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.does_not_exist = ctx.compile(self.pipe.does_not_exist)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: BadMark) -> Out:
    model.pipe(
        prompt=payload.prompt, num_inference_steps=2, guidance_scale=0.0,
        height=32, width=32, output_type="pil",
    )
    return Out(model_used=ctx.checkpoint_ref)
