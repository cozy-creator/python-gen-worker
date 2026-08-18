"""Red fixture: ctx.compile of an attribute that does not exist.

The imperative marking is typed by construction -- a typo is a real
AttributeError at the author's own line, surfaced by the derive with the
failing name in the message.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import Model, entrypoint
from gen_worker.models.model_types import register_contract_dtype

LANE = "tiny.diffusers-fp32@1"
register_contract_dtype(LANE, torch.float32)


class In(msgspec.Struct):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class BadMark(Model[Any], lanes=(LANE,)):
    def load(self, ctx: Any) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.does_not_exist = ctx.compile(self.pipe.does_not_exist)


@entrypoint  # type: ignore[operator]
def generate(payload: In, model: BadMark, ctx: Any) -> Out:
    model.pipe(
        prompt=payload.prompt, num_inference_steps=2, guidance_scale=0.0,
        height=32, width=32, output_type="pil",
    )
    return Out(model_used=ctx.checkpoint_ref)
