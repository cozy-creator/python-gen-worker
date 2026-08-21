"""A declared lane, no compile mark: it TRACES, and measures zero graphs."""

from __future__ import annotations

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from lane_contracts import TINY_DIFFUSERS_FP32


class Whatever(msgspec.Struct):
    """A model type with no canonical contract — tensorfs publishes none."""

    steps: int = 4


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


class UnmarkedModel(
    Model[Whatever],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(64)))},
):
    """A real `lanes=`, no `ctx.compile` — and no refusal."""

    def load(self, ctx: LoadContext[Whatever]) -> None:
        self.loaded = True


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: UnmarkedModel) -> Out:
    return Out(echoed=payload.text)
