"""The model owns an ENGINE and the engine owns the pipeline (pgw#1506).

minimax-h3's shape: `H3Model.engine` carries the AdaLN cache, the conditioner
buffers and the serve recipe, and the pipeline lives on the engine. So the DiT
is at `model.engine.pipeline.components['transformer']` -- depth 2 -- while
sdxl and sd15 happen to hold their pipeline directly.

The engine also back-references its model, which is how a runtime engine
normally reaches request state. That makes the object graph CYCLIC and is why
the walk has to be identity-visited rather than merely bounded.
"""

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
        self.model = model  # the back-reference: the cycle is deliberate
        self.pipeline = pipeline
        self.step_cache: dict[str, Any] = {}

    def compile_dit(self, ctx: Any) -> None:
        # h3's own line: mark through the engine, because WHICH partition a
        # checkpoint carries is the checkpoint's business.
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
    # NO `shapes=`, and that is the fixture's second finding: the mark here is
    # `Engine.compile_dit`'s, not `load`'s, so the AST reader that decides
    # whether `shapes=` is required or refused (`load_marks_compile`) sees no
    # `ctx.compile` in this class's `load` and declaring an axis would be a
    # refusal. The derive still observes the graph — the mark is real at
    # runtime — so a delegated mark keys graphs no header claims.
):
    engine: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = Engine(self, ctx.load(StableDiffusionPipeline))
        self.engine.compile_dit(ctx)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: EngineModel) -> Out:
    ctx.raise_if_cancelled()
    # pgw#1510: the documented operator-diagnostic line. It is CORRECT at
    # serve, and it used to kill the drive with "'Logger' object is not
    # callable" — so a real derive has to execute it.
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
