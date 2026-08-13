"""th#1871 P1 harness endpoint: the CONTROL — the kernel asked for is the one
that ran, so `wanted == applied` and the measurement is clean.

Its own module because two endpoint classes declaring the same `handles` in one
module resolve to one instance, and the point of this pair is that they are two
instances with two setups.
"""

from __future__ import annotations

import gen_worker
from gen_worker import RequestContext, endpoint

from .posture_endpoints_pgw1225 import PostureIn, PostureOut


@endpoint(handles=["fp8-w8a8-dynamic"])
class ServeClean:
    """The control: the kernel that was asked for is the kernel that ran."""

    def setup(self) -> None:
        gen_worker.report_applied_lane(
            "transformer", "fp8-w8a8-dynamic", modules=400, kept_bf16=6)
        gen_worker.report_attention_backend(
            "transformer", "sdpa", wanted="sdpa")

    def render_clean(self, ctx: RequestContext, data: PostureIn) -> PostureOut:
        return PostureOut(lane=ctx.execution_lane)
