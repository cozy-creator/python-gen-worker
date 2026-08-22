"""minimax-h3's SHAPE, in miniature — pgw#1655.

TWO model classes in one module. One compiles, and its mark is DELEGATED: the
ENGINE owns the partition (se#827 — the engine, not the model, knows which DiT
this checkpoint carries), so `load()` hands `ctx.compile` on rather than calling
it in place. The other is an auxiliary model another slot drives and marks
nothing.

Before pgw#1655 the AST reader could not see the delegated mark, so BOTH classes
read as unmarked and pgw#1650's subject gate refused the whole release:

    error: derive: module '...' has more than one model class
    (['AuxModel', 'EngineDrivenModel']) and NONE of them marks a compile target

The mark did not change; the reader's weight did. This fixture is that refusal's
red arm.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class Engine:
    """Owns the partition, so it applies the mark — and takes the MARK."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def compile_dit(self, mark: Any) -> None:
        for name in ("unet", "unet_ref"):
            module = getattr(self.pipeline, name, None)
            if module is not None:
                setattr(self.pipeline, name, mark(module))
                return


class EngineDrivenModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(64)))},
    shapes={"aspect": STATIC},
):
    engine: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = Engine(ctx.load(StableDiffusionPipeline))
        self.engine.compile_dit(ctx.compile)


class AuxModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(32)))},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)


@entrypoint
def generate(
    ctx: RequestContext, payload: In, model: EngineDrivenModel, aux: AuxModel
) -> Out:
    ctx.raise_if_cancelled()
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
