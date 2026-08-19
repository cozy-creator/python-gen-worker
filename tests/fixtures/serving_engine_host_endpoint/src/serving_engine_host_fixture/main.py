"""pgw#1421 host fixture: ONE engine-hosted model, whose engine really boots.

Separate from `serving_engine_endpoint` on purpose. `EndpointHost.setup()`
loads EVERY model class the endpoint references, so a fixture that also
declared the llama.cpp and vLLM arms would try to exec `llama-server` and
`vllm` on this box — the declaration arms and the supervision arm cannot
share a package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint
from gen_worker.models import SDXL
from gen_worker.serving.engine_runtime import EngineCommand, EngineSpec

#: Set by the test before boot — the stand-in engine script.
STAND_IN_SCRIPT = ""
#: Appended by `unload`, so ordering (author unload FIRST, then the
#: structural engine stop) is observable.
ORDER: List[str] = []


class In(msgspec.Struct):
    prompt: str = ""


class Out(msgspec.Struct):
    text: str


class StandInSpec(EngineSpec, frozen=True, kw_only=True):
    """A spec whose engine really runs, so the host supervises a real process
    rather than a stub of one."""

    runtime = "stand-in"

    def ladder(self, checkpoint_dir: Path) -> List[EngineCommand]:
        port = self._port()
        return [EngineCommand(
            argv=(sys.executable, STAND_IN_SCRIPT, str(port), "0", "0.1", "-1"),
            port=port,
        )]


class StandInModel(
    Model[SDXL],
    eager_only="an external engine process owns the weights and the graph",
    self_loading="served by an external engine process over HTTP",
):
    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(StandInSpec())

    def unload(self, ctx: LoadContext[SDXL]) -> None:
        ORDER.append("author_unload")
        # An author bug, deliberately: the engine must be reaped anyway.
        raise RuntimeError("author unload bug — must not strand the engine")


@entrypoint
def probe(ctx: RequestContext, payload: In, model: StandInModel) -> Out:
    return Out(text=model.engine.base_url)
