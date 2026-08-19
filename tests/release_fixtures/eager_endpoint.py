"""An `eager_only=` module: eager-permanent, DECLARED, with the reason.

pgw#1488: `lanes=()` no longer says this. An absent lane declaration means
"no layout contract stated", which traces; eager-forever is its own word and
carries the author's reason for it.
"""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import Model, RequestContext, entrypoint


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


class EagerModel(
    Model[Any],
    eager_only="the fixture's subject IS the no-compile document",
):
    """eager_only= -> nothing compiles, ever; the document says so."""


@entrypoint
def analyze(ctx: RequestContext, payload: In, model: EagerModel) -> Out:
    return Out(echoed=payload.text)
