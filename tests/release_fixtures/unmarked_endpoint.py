"""A declared lane, no compile mark: it TRACES, and measures zero graphs.

pgw#1488's middle state, re-based on pgw#1599. `Whatever` is a model type with
no CANONICAL contract, which used to leave three bad options: refuse the class,
borrow a contract the model type does not have, or write `lanes=()` and
silently disable compilation. All three are gone. A lane is a property of the
CHECKPOINT, not of the model family, so this class names the contract its
checkpoint actually carries — a model type with no canonical lane is no
obstacle to that — and marks nothing in `load`.

The result is the honest one: the author marked nothing, so there is nothing to
compile, and the empty `graphs.lanes` is measured rather than declared.
"""

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
    # No `shapes=`: `load` marks no compile target, so nothing is keyed and
    # declaring an axis would be a refusal.
):
    """A real `lanes=`, no `ctx.compile` — and no refusal."""

    def load(self, ctx: LoadContext[Whatever]) -> None:
        self.loaded = True


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: UnmarkedModel) -> Out:
    return Out(echoed=payload.text)
