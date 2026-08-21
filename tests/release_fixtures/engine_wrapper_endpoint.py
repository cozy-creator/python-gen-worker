from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class Engine:
    """A runtime wrapper with state of its own -- not a pass-through."""

    def __init__(self, model: Any, pipeline: Any) -> None:
        self.model = model
        self.pipeline = pipeline
        self.step_cache: dict[str, Any] = {}

    def compile_dit(self, ctx: Any) -> None:
        for name in ("unet", "unet_ref"):
            module = getattr(self.pipeline, name, None)
            if module is not None:
                setattr(self.pipeline, name, ctx.compile(module))
                return


class EngineModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
):
    engine: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = Engine(self, ctx.load(StableDiffusionPipeline))
        self.engine.compile_dit(ctx)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: EngineModel) -> Out:
    ctx.raise_if_cancelled()
    ctx.log("engine wrapper: resolved pipeline", level="info", side=32)
    with torch.inference_mode():
        model.engine.pipeline(
            prompt=payload.prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            width=32,
            height=32,
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )
    return Out(model_used=ctx.checkpoint_ref)
