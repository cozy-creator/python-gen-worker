"""th#1871 P1 harness endpoint: the DECLARED-COMPILED / RAN-EAGER shape.

minimax-h3's specimen, minus the GPU: the hub demands a compiled cell, the
worker serves eager, and for hours nothing in any schema could say so. The
worker's own half of that fact is `compile_state_wanted`, and on the
ModelResolution dispatch path (`RunJob.lane` empty — "" = policy) the ONLY thing
that states it is the `required_compile` fence.
"""

from __future__ import annotations

import gen_worker
from gen_worker import RequestContext, endpoint

from .posture_endpoints_pgw1225 import PostureIn, PostureOut


@endpoint(handles=["fp8-w8a8-dynamic"])
class ServeDeclaredCompiled:
    def setup(self) -> None:
        gen_worker.report_applied_lane(
            "transformer", "fp8-w8a8-dynamic", modules=400, kept_bf16=6)
        gen_worker.report_attention_backend(
            "transformer", "sdpa", wanted="sdpa")

    def render_declared(self, ctx: RequestContext, data: PostureIn) -> PostureOut:
        return PostureOut(lane=ctx.execution_lane)
