"""Single-endpoint module in the shape EVERY real release has (pgw#797).

`spec.compile is not None` is the branch that made the released 0.78.0 boot
ladder empty, and no existing harness module carries it:

  * `Lifecycle.startup()` routes a compile spec to `dynamic` and does NOT set
    it up — so neither of pgw#789's two `pipeline_load` call sites is reached;
  * unlike a `Slot` function, a compile function is NOT advertised until
    `rec.ready` (`Executor.available_functions`), so the boot genuinely is not
    over when `startup()` returns — which is what pgw#789's milestone claimed.

That combination is the defect, so the regression fixture has to boot on it.
Deliberately ONE endpoint: with a plain sibling in the module the worker
advertises something at boot, the boot legitimately closes there, and the
hub-delivered setup that follows is steady state rather than boot.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker import Compile, Hub, RequestContext, endpoint
from gen_worker.families.base import GenerationDefaults, family


@family("harness-pgw797-testfam")
class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 3


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


class ToyPipeline:
    """A worker-LOADED slot: `compile=` only arms on a slot the worker loads
    itself (a class exposing `from_pretrained`), so a `str` annotation would be
    rejected at walk time and this fixture would not be the shape under test."""

    def __init__(self, weights: str) -> None:
        self.weights = weights

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "ToyPipeline":
        return cls((Path(path) / "model.safetensors").read_text())

    def to(self, device: str) -> "ToyPipeline":
        return self


#: Never reachable from the code default: only a hub-stamped DesiredResidency
#: snapshot materializes it, which is the ordering the ladder has to survive.
COMPILE_MODEL = Hub("harness/pgw797-compile-model", tag="prod")


@endpoint(
    model=COMPILE_MODEL,
    # `text_len=0` declares "no text conditioning" explicitly — the walk-time
    # lint requires every compile endpoint to state the axis.
    compile=Compile(family="harness-pgw797", shapes=((64, 64),), text_len=0),
)
class CompileBoundEndpoint:
    def setup(self, model: ToyPipeline) -> None:
        self.pipe = model

    def compile_echo(
        self, ctx: RequestContext[_Defaults], data: EchoIn,
    ) -> EchoOut:
        return EchoOut(response=self.pipe.weights)
