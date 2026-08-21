"""TWO model classes and NOT ONE compile mark — the refusal that SURVIVES pgw#1650.

Subjecthood is read off the MARK. With no mark anywhere there is nothing to
tell a release subject from an auxiliary model another slot drives, so the
derive refuses rather than guessing.
"""

from __future__ import annotations

from typing import Any

import msgspec
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class Out(msgspec.Struct):
    model_used: str


class OneModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(64)))},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)


class OtherModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(32)))},
):
    net: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.net = ctx.load(UNet2DConditionModel)


@entrypoint
def generate(
    ctx: RequestContext, payload: In, one: OneModel, other: OtherModel
) -> Out:
    ctx.raise_if_cancelled()
    return Out(model_used=ctx.checkpoint_ref)
