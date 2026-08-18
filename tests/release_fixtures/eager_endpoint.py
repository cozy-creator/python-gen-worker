"""A no-lane module: eager-permanent, stated by an explicit empty document."""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import Model, entrypoint


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


class EagerModel(Model[Any]):
    """No lanes= -> nothing compiles, ever; the document says so."""


@entrypoint  # type: ignore[operator]
def analyze(payload: In, model: EagerModel, ctx: Any) -> Out:
    return Out(echoed=payload.text)
