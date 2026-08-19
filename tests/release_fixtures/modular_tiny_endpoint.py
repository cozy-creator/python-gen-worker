"""A main_v2-shaped endpoint over a MODULAR pipeline (pgw#1450 fixture).

Same surface as ``tiny_endpoint``: contract-object ``lanes=``, an imperative
``ctx.compile`` mark inside ``load``, free ctx-first entrypoints whose payload
schema is what the derive enumerates. The one difference is the pipeline
class -- a ``ModularPipeline`` subclass that attaches the components it is
constructed with, which is what every modular endpoint in the fleet ships.

The entrypoint calls the marked denoiser directly rather than a full modular
``__call__``: the subject is whether the pipeline CARRIES the module at all,
and a fixture that also had to be a working modular workflow would put a lot
of diffusers between the defect and the assertion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from modular_tiny_tree import TinyStreamingPipeline

from gen_worker import (
    LoadContext,
    Model,
    RequestContext,
    entrypoint,
)
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


_BUCKETS: dict[Size, int] = {Size.SMALL: 4, Size.LARGE: 8}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class LatentOutput(msgspec.Struct):
    model_used: str


class ModularModel(Model[SDXL], lanes=(TINY_DIFFUSERS_FP32,)):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(TinyStreamingPipeline)
        # The author's line, unchanged from the classic endpoint's. It reads
        # the component off the pipeline -- which is exactly the read that
        # found `None` before the loader finished the build.
        self.pipe.unet = ctx.compile(self.pipe.unet)


@entrypoint
def generate(
    ctx: RequestContext, payload: GenerateInput, model: ModularModel
) -> LatentOutput:
    ctx.raise_if_cancelled()
    side = _BUCKETS[payload.size]
    unet = model.pipe.unet
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    with torch.inference_mode():
        model.pipe.unet(
            torch.zeros(1, 4, side, side, device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
            encoder_hidden_states=torch.zeros(1, 77, 16, device=device, dtype=dtype),
        )
    return LatentOutput(model_used=ctx.checkpoint_ref)
