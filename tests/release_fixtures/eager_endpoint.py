"""A lanes=() module: eager-permanent, stated by an explicit empty document."""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import Model, RequestContext, entrypoint


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


class EagerModel(Model[Any], lanes=()):
    """lanes=() -> nothing compiles, ever; the document says so."""


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: EagerModel) -> Out:
    return Out(echoed=payload.text)
