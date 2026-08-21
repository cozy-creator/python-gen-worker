from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from lane_contracts import TINY_DIFFUSERS_FP32


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


class EagerModel(
    Model[Any],
    lanes={TINY_DIFFUSERS_FP32: lane(request=const(MiB(64)))},
):
    """No `load` at all, so no `ctx.compile` — nothing here is ever keyed."""


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: EagerModel) -> Out:
    return Out(echoed=payload.text)
