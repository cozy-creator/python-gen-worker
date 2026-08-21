from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from modular_tiny_tree import TinyStreamingPipeline

from gen_worker import (
    STATIC,
    LoadContext,
    Model,
    RequestContext,
    entrypoint,
    lane,
)
from gen_worker.demand import MiB, const, per_mp_batch
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


class ModularModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(TinyStreamingPipeline)
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
