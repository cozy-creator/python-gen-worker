"""A no-lane endpoint: eager-permanent, stated by an explicit empty document."""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, endpoint


class In(msgspec.Struct):
    text: str


class Out(msgspec.Struct):
    echoed: str


@endpoint
def analyze(ctx: RequestContext, payload: In) -> Out:
    return Out(echoed=payload.text)
