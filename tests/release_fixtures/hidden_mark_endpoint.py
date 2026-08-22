"""wan-2.2's SHAPE — the mark exists and the reader CANNOT follow it (pgw#1655).

`load()` hands the LOAD CONTEXT ITSELF to a helper, so no reader that does not
execute author code can say whether this class compiles. Before pgw#1655 that
"cannot tell" was answered `False`: wan-2.2's two MoE classes were dropped from
the subject set silently, and a module whose only marks were this shape refused
outright. It is now STATED at the class declaration, so the module does not
import until the author hands the MARK instead of the context.
"""

from __future__ import annotations

from typing import Any

import msgspec
from diffusers import StableDiffusionPipeline

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


def _mark_moe_targets(ctx: Any, pipeline: Any) -> None:
    pipeline.unet = ctx.compile(pipeline.unet)


class HiddenMarkModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(64)))},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        _mark_moe_targets(ctx, self.pipe)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: HiddenMarkModel) -> Out:
    ctx.raise_if_cancelled()
    return Out(model_used=ctx.checkpoint_ref)
