"""Red fixture: a marked module the code never CALLS.

``vae`` is real and markable, but the pipeline calls ``vae.decode()``, which
bypasses ``Module.__call__`` -- the hook observes nothing. The derive must
refuse with the name in the message, never emit a silent zero-graph lane.
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


class UnobservedMark(Model[Any], lanes=(LANE,)):
    def load(self, ctx: Any) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)
        self.pipe.vae = ctx.compile(self.pipe.vae)  # .decode() bypasses __call__


@entrypoint  # type: ignore[operator]
def generate(payload: In, model: UnobservedMark, ctx: Any) -> Out:
    model.pipe(
        prompt=payload.prompt, num_inference_steps=2, guidance_scale=0.0,
        height=32, width=32, output_type="pil",
    )
    return Out(model_used=ctx.checkpoint_ref)
