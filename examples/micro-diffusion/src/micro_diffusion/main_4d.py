"""The pgw#998 variant's worker function — same family shape, GRID latents.

Separate from `main` because it is a different FAMILY (`micro-4d`) with a
different traced signature; sharing a module would put two families' endpoints
in one discovery namespace for no benefit. Everything else — catalog slot with
no code default, declaration registered at import, `ctx.slots` dereferenced
first — is deliberately identical to `main`, so a difference in outcome is a
difference in the SHAPE under test and nothing else.
"""

from __future__ import annotations

from typing import List

import msgspec
import torch

from gen_worker import Compile, RequestContext, Resources, Slot, endpoint
from gen_worker.families import GenerationDefaults, family

from .aot_declaration_4d import (  # noqa: F401 — registers at import
    ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    LATENT_ROWS,
    PIXEL_ROWS,
)
from .pipeline import MicroGridPipeline


@family(FAMILY)
class Micro4dDefaults(GenerationDefaults, frozen=True):
    steps: int = 2


class Micro4dIn(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class Micro4dOut(msgspec.Struct):
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroGridPipeline, selected_by="model",
        layouts={"*": ("plain.bf16@1",)})},
    compile=Compile(
        family=FAMILY, targets=("transformer",), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class Generate4d:
    def setup(self, pipeline: MicroGridPipeline) -> None:
        self.pipe = pipeline

    def generate_4d(
        self, ctx: RequestContext[Micro4dDefaults], data: Micro4dIn,
    ) -> Micro4dOut:
        resolved = ctx.slots["pipeline"]
        grid = LATENT_ROWS[0]
        device = self.pipe.device
        config = self.pipe.config
        generator = ctx.generator(998)
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
        return Micro4dOut(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in out.shape)))


__all__ = ["DECLARATION", "FAMILY", "Generate4d", "Micro4dDefaults",
           "Micro4dIn", "Micro4dOut"]
