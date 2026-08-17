"""The RED twin: ie#637's UPSTREAM pad spelling, expected to be REFUSED.

Identical to `main_pad32` in every respect but the denoiser class — this one
branches on the pad, which is what pins the declared symbols and what ie#566
G3's declared-range gate exists to refuse. It is here so the fixed member's
green means something: without a red twin on the same declaration, a pass
proves only that this graph never reached the gate.
"""

from __future__ import annotations

from typing import List

import msgspec
import torch

from gen_worker import Compile, RequestContext, Resources, Slot, endpoint
from gen_worker.families import GenerationDefaults, register_family

from .aot_declaration_pad32_branchy import (  # noqa: F401 — registers at import
    ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    LATENT_ROWS,
    PIXEL_ROWS,
)
from .pipeline import MicroPad32BranchyPipeline


class MicroPad32BranchyDefaults(GenerationDefaults, frozen=True):
    steps: int = 2


register_family(FAMILY, MicroPad32BranchyDefaults)


class MicroPad32BranchyIn(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class MicroPad32BranchyOut(msgspec.Struct):
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroPad32BranchyPipeline, selected_by="model",
        layouts={"*": ("plain.bf16@1",)})},
    compile=Compile(
        family=FAMILY, targets=("transformer",), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class GeneratePad32Branchy:
    def setup(self, pipeline: MicroPad32BranchyPipeline) -> None:
        self.pipe = pipeline

    def generate_pad32_branchy(
        self, ctx: RequestContext[MicroPad32BranchyDefaults], data: MicroPad32BranchyIn,
    ) -> MicroPad32BranchyOut:
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
        return MicroPad32BranchyOut(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in out.shape)))


__all__ = ["DECLARATION", "FAMILY", "GeneratePad32Branchy", "MicroPad32BranchyDefaults",
           "MicroPad32BranchyIn", "MicroPad32BranchyOut"]
