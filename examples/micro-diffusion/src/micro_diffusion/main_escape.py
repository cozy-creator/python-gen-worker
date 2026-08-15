"""The pgw#1062 variant's worker function — the escape-hatch graph.

A separate FAMILY (`micro-escape`) with its own traced signature, mirroring
`main_4d`'s structure exactly: catalog slot with no code default, declaration
registered at import, `ctx.slots` dereferenced first. A difference in outcome
against `micro` is therefore a difference in the OPS under test — the custom
op, the `triton_op` kernel and the raw Triton call — and nothing else.
"""

from __future__ import annotations

from typing import List

import msgspec
import torch

from gen_worker import Compile, RequestContext, Resources, Slot, endpoint
from gen_worker.families import GenerationDefaults, family

from .aot_declaration_escape import (  # noqa: F401 — registers at import
    ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    PIXEL_ROWS,
    TOKEN_ROWS,
)
from .pipeline import MicroEscapePipeline


@family(FAMILY)
class MicroEscapeDefaults(GenerationDefaults, frozen=True):
    steps: int = 2


class MicroEscapeIn(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class MicroEscapeOut(msgspec.Struct):
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroEscapePipeline, selected_by="model",
        layouts={"*": ("plain.bf16@1",)})},
    compile=Compile(
        family=FAMILY, targets=("transformer",), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class GenerateEscape:
    def setup(self, pipeline: MicroEscapePipeline) -> None:
        self.pipe = pipeline

    def generate_escape(
        self, ctx: RequestContext[MicroEscapeDefaults], data: MicroEscapeIn,
    ) -> MicroEscapeOut:
        resolved = ctx.slots["pipeline"]
        tokens = TOKEN_ROWS[0]
        device = self.pipe.device
        config = self.pipe.config
        generator = ctx.generator(1062)
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
        return MicroEscapeOut(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in out.shape)))


__all__ = ["DECLARATION", "FAMILY", "GenerateEscape", "MicroEscapeDefaults",
           "MicroEscapeIn", "MicroEscapeOut"]
