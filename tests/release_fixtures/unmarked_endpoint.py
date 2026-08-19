"""No contract, no eager declaration, no compile mark: it TRACES anyway.

pgw#1488's middle state. `Whatever` is a model type with no canonical
contract, so before this change the derive refused the class outright ("omits
lanes= and its model type has no canonical contract yet"); the remedy the
refusal named — `lanes=()` — then disabled compilation with no output at all.

Now the lane is DERIVED, `load` runs, and the honest answer is recorded: the
author marked nothing, so there is nothing to compile. Zero graphs measured,
not assumed.
"""

from __future__ import annotations

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint


class Whatever(msgspec.Struct):
    """A model type with no canonical contract — tensorfs publishes none."""

    steps: int = 4


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


class UnmarkedModel(Model[Whatever]):
    """No `lanes=`, no `ctx.compile` — and no refusal."""

    def load(self, ctx: LoadContext[Whatever]) -> None:
        self.loaded = True


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: UnmarkedModel) -> Out:
    return Out(echoed=payload.text)
