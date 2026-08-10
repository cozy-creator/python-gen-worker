"""A boot fixture with MULTIPLE weight components (pgw#1087).

The pgw#797 ladder fixture ships one weight file, which is exactly the shape
that cannot exercise the thing pgw#1087 adds: with one file there is one
component, one `component_fetch` row, and "is the decomposition per component"
is unfalsifiable. This module's checkpoint is laid out the way a real
diffusers repo is — `transformer/`, `text_encoder/`, `vae/` plus a top-level
`model_index.json` — so a boot produces four component rows inside one
`weights_fetch`, and their intervals can be compared for overlap.

Otherwise deliberately identical in SHAPE to the pgw#797 fixture: ONE compile
endpoint, a worker-LOADED slot, `resources.gpu` so a warm plan exists. Those
are the branches that made the released ladder empty, and a decomposition
fixture that avoided them would measure a boot production never runs.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker import Compile, Hub, RequestContext, Resources, endpoint
from gen_worker.families.base import GenerationDefaults, family

#: The component directories the fixture snapshot carries, in the order a
#: reader should expect them. `(root)` is the fourth, synthesized by
#: `cozy_snapshot._component_of` for `model_index.json`.
COMPONENTS = ("text_encoder", "transformer", "vae")


@family("harness-pgw1087-testfam")
class _Defaults(GenerationDefaults, frozen=True):
    steps: int = 3


class EchoIn(msgspec.Struct):
    text: str = ""


class EchoOut(msgspec.Struct):
    response: str


class MultiComponentPipeline:
    """A worker-LOADED slot whose weights are spread over components."""

    def __init__(self, parts: dict) -> None:
        self.parts = parts

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "MultiComponentPipeline":
        root = Path(path)
        parts = {
            name: (root / name / "weights.safetensors").read_bytes()
            for name in COMPONENTS
            if (root / name / "weights.safetensors").is_file()
        }
        return cls(parts)

    def to(self, device: str) -> "MultiComponentPipeline":
        return self


COMPILE_MODEL = Hub("harness/pgw1087-multi-component", tag="prod")


@endpoint(
    model=COMPILE_MODEL,
    compile=Compile(family="harness-pgw1087", shapes=((64, 64),), text_len=0),
    resources=Resources(gpu=True),
)
class ColdBootDecompositionEndpoint:
    def setup(self, model: MultiComponentPipeline) -> None:
        self.pipe = model

    def compile_echo(
        self, ctx: RequestContext[_Defaults], data: EchoIn,
    ) -> EchoOut:
        return EchoOut(response=",".join(sorted(self.pipe.parts)))
