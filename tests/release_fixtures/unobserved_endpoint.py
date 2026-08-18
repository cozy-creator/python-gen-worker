"""Red fixture: a lane naming a module the code never CALLS.

``vae`` exists, but the pipeline calls ``vae.decode()``, which bypasses
``Module.__call__`` -- the hook observes nothing. The derive must refuse
with the path in the message, never emit a silent zero-graph lane.
"""

from __future__ import annotations

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import RequestContext, endpoint
from torchcg import Lane


class In(msgspec.Struct):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


@endpoint(
    lanes=(
        Lane("fp32", compile=("unet", "vae"), contract="plain.fp32@1",
             dtype=torch.float32),
    ),
)
class UnobservedTarget:
    def setup(self, ctx: RequestContext) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            ctx.checkpoint_dir, torch_dtype=ctx.lane.dtype
        )

    def generate(self, ctx: RequestContext, payload: In) -> Out:
        self.pipe(
            prompt=payload.prompt, num_inference_steps=2, guidance_scale=0.0,
            height=32, width=32, output_type="pil",
        )
        return Out(model_used=ctx.checkpoint_ref)
