"""A module that MARKS NOTHING: a real lane, and zero graphs (pgw#1599).

There is no `eager_only=` any more. Paul, 2026-08-20: *"If you do not want the
model compiled, simply do not include any ctx.compile() invocations in your
model's 'load' method."* The keyword conflated two independent axes — a lane
answers checkpoint COMPATIBILITY and lane SELECTION whether or not anything
compiles — so this class declares its lane like every other model and the
ABSENCE of a mark is the whole statement. The empty `graphs.lanes` this derives
is a measurement, not a declaration.
"""

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
    # No `shapes=`: this class marks no compile target, and the fixture's
    # subject IS the no-compile document.
):
    """No `load` at all, so no `ctx.compile` — nothing here is ever keyed."""


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: EagerModel) -> Out:
    return Out(echoed=payload.text)
