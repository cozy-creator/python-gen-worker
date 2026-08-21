"""Red fixture: a marked module the code never CALLS.

``vae`` is real and markable, but the pipeline calls ``vae.decode()``, which
bypasses ``Module.__call__`` -- the hook observes nothing. The derive must
refuse with the name in the message, never emit a silent zero-graph lane.
"""

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


class UnobservedMark(
    Model[Any],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    # diffusers pipelines compose their components dynamically; the static
    # class carries no `unet`/`vae`.
    pipe: Any

    def load(self, ctx: LoadContext[Any]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)
        self.pipe.vae = ctx.compile(self.pipe.vae)  # .decode() bypasses __call__


@entrypoint
def generate(ctx: RequestContext, payload: In, model: UnobservedMark) -> Out:
    model.pipe(
        prompt=payload.prompt, num_inference_steps=2, guidance_scale=0.0,
        height=32, width=32, output_type="pil",
    )
    return Out(model_used=ctx.checkpoint_ref)
