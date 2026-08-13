"""th#1871 P1 harness endpoint: a setup() that reports its POSTURE.

ie#707's shape, minus the GPU. The lane declares flash attention; the image
does not have it; sdpa runs. Under the old surface the endpoint could not say
so at all — `report_applied_attention` validates against the SPARSITY grammar
(`dense` / `sparse-k<N>`) and raises on `"sdpa"` — so the only honest thing an
author could do was stay silent, and 23 of 29 families did.

Its own file so no sibling lane's harness module is disturbed.
"""

from __future__ import annotations

import msgspec

import gen_worker
from gen_worker import RequestContext, endpoint


class PostureIn(msgspec.Struct):
    prompt: str = ""


class PostureOut(msgspec.Struct):
    lane: str = ""


@endpoint(handles=["fp8-w8a8-dynamic"])
class ServeWithFallback:
    def setup(self) -> None:
        gen_worker.report_applied_lane(
            "transformer", "fp8-w8a8-dynamic", modules=400, kept_bf16=6)
        # The whole of ie#707, in one call the SDK could not previously accept.
        gen_worker.report_attention_backend(
            "transformer", "sdpa", wanted="flash_attention_2")

    def render(self, ctx: RequestContext, data: PostureIn) -> PostureOut:
        return PostureOut(lane=ctx.execution_lane)
