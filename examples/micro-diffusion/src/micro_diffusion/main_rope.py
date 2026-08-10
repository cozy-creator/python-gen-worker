"""The pgw#1080 RED control's worker function — the pinned-rope graph.

A separate FAMILY (`micro-rope`) whose only difference from the base family
is the denoiser's lazily-built, CPU-PINNED table. It exists to be REFUSED by
the meta-instantiation gate, so its gauntlet expectation is `red`; the base
family is the green twin (ie#630's registered buffer), and the pair is the
control the gate is judged by.
"""

from __future__ import annotations

from typing import List

import msgspec
import torch

from gen_worker import Compile, RequestContext, Resources, Slot, endpoint
from gen_worker.families import GenerationDefaults, family

from .aot_declaration_rope import (  # noqa: F401 — registers at import
    ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    PIXEL_ROWS,
    TOKEN_ROWS,
)
from .pipeline import MicroRopePipeline


@family(FAMILY)
class MicroRopeDefaults(GenerationDefaults, frozen=True):
    steps: int = 2


class MicroRopeIn(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class MicroRopeOut(msgspec.Struct):
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroRopePipeline, selected_by="model")},
    compile=Compile(
        family=FAMILY, targets=("transformer",), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class GenerateRope:
    def setup(self, pipeline: MicroRopePipeline) -> None:
        self.pipe = pipeline

    def generate_rope(
        self, ctx: RequestContext[MicroRopeDefaults], data: MicroRopeIn,
    ) -> MicroRopeOut:
        resolved = ctx.slots["pipeline"]
        tokens = TOKEN_ROWS[0]
        device = self.pipe.device
        config = self.pipe.config
        generator = ctx.generator(1080)
        x: List[torch.Tensor] = [
            torch.randn(tokens, config.in_channels, generator=generator,
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
        return MicroRopeOut(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in out.shape)))


__all__ = ["DECLARATION", "FAMILY", "GenerateRope", "MicroRopeDefaults",
           "MicroRopeIn", "MicroRopeOut"]
