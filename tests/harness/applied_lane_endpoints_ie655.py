"""ie#655 harness endpoint: a serve-time recipe on a worker that never compiles.

wan-2.2's shape, minus the GPU: `setup()` quantizes the weights and REPORTS
the applied body (pgw#1104), and nothing arms a compiled graph — so the honest
lane is `fp8-w8a8-dynamic+eager` and the honest serving mode is `eager`. Its
own file so no sibling lane's uncommitted `toy_endpoints.py` is disturbed.
"""

from __future__ import annotations

import msgspec

import gen_worker
from gen_worker import RequestContext, endpoint


class RecipeIn(msgspec.Struct):
    prompt: str = ""


class RecipeOut(msgspec.Struct):
    lane: str = ""


#: Declared so the instructed-lane case is exercised too: the hub may ASK for
#: `fp8-w8a8-dynamic+compiled`; only the body is the author's to honor.
@endpoint(handles=["fp8-w8a8-dynamic"])
class ServeQuantized:
    def setup(self) -> None:
        gen_worker.report_applied_lane(
            "transformer", "fp8-w8a8-dynamic", modules=400, kept_bf16=6)

    def render(self, ctx: RequestContext, data: RecipeIn) -> RecipeOut:
        return RecipeOut(lane=ctx.execution_lane)
