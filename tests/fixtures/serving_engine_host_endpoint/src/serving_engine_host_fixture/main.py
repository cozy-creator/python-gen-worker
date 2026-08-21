from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import msgspec

from gen_worker import LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const
from gen_worker.models import SDXL
#: THE REAL RATIFIED PAIR (pgw#1621). A lane is `(topology, quant)`; both
#: halves are documents in the vendored `spec/v2` corpus, so this fixture
#: cannot invent one. The v1 constant it replaces is deleted.
SDXL_DIFFUSERS_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")
from gen_worker.serving.engine_runtime import EngineCommand, EngineSpec

STAND_IN_SCRIPT = ""
ORDER: List[str] = []


class In(msgspec.Struct):
    prompt: str = ""


class Out(msgspec.Struct):
    text: str


class StandInSpec(EngineSpec, frozen=True, kw_only=True):
    """A spec whose engine really runs, so the host supervises a real process rather than a stub of one."""

    runtime = "stand-in"

    def ladder(self, checkpoint_dir: Path) -> List[EngineCommand]:
        port = self._port()
        return [EngineCommand(
            argv=(sys.executable, STAND_IN_SCRIPT, str(port), "0", "0.1", "-1"),
            port=port,
        )]


class StandInModel(
    Model[SDXL],
    lanes={SDXL_DIFFUSERS_BF16: lane(request=const(MiB(64)))},
    self_loading="served by an external engine process over HTTP",
):
    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.engine = ctx.engine(StandInSpec())

    def unload(self, ctx: LoadContext[SDXL]) -> None:
        ORDER.append("author_unload")
        raise RuntimeError("author unload bug — must not strand the engine")


@entrypoint
def probe(ctx: RequestContext, payload: In, model: StandInModel) -> Out:
    return Out(text=model.engine.base_url)
