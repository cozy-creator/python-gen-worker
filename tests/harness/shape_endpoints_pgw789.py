"""pgw#789 harness endpoint: a payload carrying the SHAPE axes.

The toy endpoints echo text, so none of them can prove that the executed
``num_inference_steps``/``width``/``height`` reach ``JobMetrics``. This module
exists only to give that assertion a real endpoint to run through — a real
registry compiled graph, real msgspec decoding (so struct defaults are applied exactly
as in production), real dispatch. It lives in its own file so no sibling lane's
uncommitted work in ``toy_endpoints.py`` is disturbed.
"""

from __future__ import annotations

import msgspec

from gen_worker import RequestContext, endpoint


class ShapedIn(msgspec.Struct):
    prompt: str = ""
    # Defaults are the point: a caller that omits `height` must still produce a
    # height dimension on the metrics row, because the DEFAULT is what executed.
    num_inference_steps: int = 20
    width: int = 512
    height: int = 768


class ShapedOut(msgspec.Struct):
    pixels: int


@endpoint
class Shaped:
    def render(self, ctx: RequestContext, data: ShapedIn) -> ShapedOut:
        return ShapedOut(pixels=data.width * data.height)
