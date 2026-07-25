"""pgw#655 fixture: one endpoint bound to a WORKER-FETCHED (hf) model that
never materializes, plus one model-free endpoint that must keep serving.

Its own module (not ``toy_endpoints``) because adding an endpoint there
changes the world every hub-double test observes.
"""

from __future__ import annotations

from gen_worker import HF, RequestContext, endpoint

from harness.toy_endpoints import EchoIn, EchoOut

UPSTREAM_REF = "harness/pgw655-upstream"


@endpoint(model=HF("harness/pgw655-upstream"))
class HfBoundEndpoint:
    def setup(self, model: str) -> None:
        self.model_path = model

    def hf_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        return EchoOut(response=self.model_path)


@endpoint
class ModelFreeEndpoint:
    def plain_echo(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
        return EchoOut(response=data.text)
