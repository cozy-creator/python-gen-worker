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

from gen_worker import Compile, Hub, RequestContext, Resources, endpoint
from gen_worker.families.base import GenerationDefaults, register_family


def _slot_weight_bytes(slot_dir: object) -> str:
    """What the slot's weights file HOLDS, on any tree shape.

    A harness endpoint is the closest thing this repo has to a pgw#1303 author
    slot: it is handed a DIRECTORY and reads raw weight bytes out of it. After
    pgw#1308 step ⑥ that directory is a projected tree, so the file at the
    path is a ~128 B pointer stub — which is exactly what a real third-party
    loader gets, and exactly why #1303 is a ruling rather than a cleanup.

    So these fixtures go through the SAME seam a gated production site now
    goes through (`models.materialized_view.third_party_dir`), which is what
    makes them evidence about the gate instead of a workaround for it.
    """

    from gen_worker.models.materialized_view import third_party_dir

    real = third_party_dir(Path(str(slot_dir)), why="harness author slot")
    return (real / "model.safetensors").read_text()


class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 3


register_family("harness-pgw797-testfam", _Defaults)


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
        return cls(_slot_weight_bytes(path))

    def to(self, device: str) -> "ToyPipeline":
        return self


#: Never reachable from the code default: only a hub-stamped DesiredResidency
#: snapshot materializes it, which is the ordering the ladder has to survive.
COMPILE_MODEL = Hub("harness/pgw797-compile-model", release="prod")


@endpoint(
    model=COMPILE_MODEL,
    # `text_len=0` declares "no text conditioning" explicitly — the walk-time
    # lint requires every compile endpoint to state the axis.
    compile=Compile(family="harness-pgw797", shapes=((64, 64),), text_len=0),
    # pgw#797: `warmup.plan` only schedules handlers whose spec declares a
    # GPU (`resources.gpu`), so WITHOUT this the warm plan is empty and the
    # per-iteration rows have no real-path coverage at all.
    resources=Resources(gpu=True),
)
class CompileBoundEndpoint:
    def setup(self, model: ToyPipeline) -> None:
        self.pipe = model

    def compile_echo(
        self, ctx: RequestContext[_Defaults], data: EchoIn,
    ) -> EchoOut:
        return EchoOut(response=self.pipe.weights)
