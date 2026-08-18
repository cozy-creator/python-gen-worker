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

from gen_worker import Endpoint, endpoint
from gen_worker.models.model_types import register_contract_dtype

LANE = "tiny.diffusers-fp32@1"
register_contract_dtype(LANE, torch.float32)


class In(msgspec.Struct):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


@endpoint(lanes=(LANE,))
class BadMark(Endpoint):
    def setup(self, ctx: Any) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            ctx.checkpoint_dir, torch_dtype=ctx.lane.dtype
        )
        self.pipe.does_not_exist = ctx.compile(self.pipe.does_not_exist)

    def generate(self, ctx: Any, payload: In) -> Out:
        self.pipe(
            prompt=payload.prompt, num_inference_steps=2, guidance_scale=0.0,
            height=32, width=32, output_type="pil",
        )
        return Out(model_used=ctx.checkpoint_ref)
