"""The ie#637 pad-to-32 variant's worker function.

Its own FAMILY (`micro-pad32`) with its own traced signature, for the same
reason `main_4d` is separate: two families in one discovery namespace buys
nothing. Everything else — a catalog slot with no code default, the
declaration registered at import, `ctx.slots` dereferenced first — is
deliberately identical to `main_4d`, so a difference in outcome is a
difference in the PAD and nothing else.
"""

from __future__ import annotations

from typing import List

import msgspec
import torch

from gen_worker import Compile, RequestContext, Resources, Slot, endpoint
from gen_worker.families import GenerationDefaults, family

from .aot_declaration_pad32 import (  # noqa: F401 — registers at import
    ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    LATENT_ROWS,
    PIXEL_ROWS,
)
from .pipeline import MicroPad32Pipeline


@family(FAMILY)
class MicroPad32Defaults(GenerationDefaults, frozen=True):
    steps: int = 2


class MicroPad32In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class MicroPad32Out(msgspec.Struct):
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroPad32Pipeline, selected_by="model",
        layouts={"*": ("plain.bf16@1",)})},
    compile=Compile(
        family=FAMILY, targets=("transformer",), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class GeneratePad32:
    def setup(self, pipeline: MicroPad32Pipeline) -> None:
        self.pipe = pipeline

    def generate_pad32(
        self, ctx: RequestContext[MicroPad32Defaults], data: MicroPad32In,
    ) -> MicroPad32Out:
        resolved = ctx.slots["pipeline"]
        grid = LATENT_ROWS[0]
        device = self.pipe.device
        config = self.pipe.config
        generator = ctx.generator(637)
        x: List[torch.Tensor] = [
            torch.randn(config.in_channels, grid, grid, generator=generator,
                        device=device, dtype=torch.float32)
            for _ in range(ARITY)
        ]
        t = torch.full((ARITY,), 100.0, device=device, dtype=torch.float32)
        cond: List[torch.Tensor] = [
            torch.randn(COND_LEN, config.cond_dim, generator=generator,
                        device=device, dtype=torch.float32)
            for _ in range(ARITY)
        ]
        with torch.no_grad():
            out = self.pipe.transformer(x, t, cond)
        return MicroPad32Out(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in out.shape)))


__all__ = ["DECLARATION", "FAMILY", "GeneratePad32", "MicroPad32Defaults",
           "MicroPad32In", "MicroPad32Out"]
