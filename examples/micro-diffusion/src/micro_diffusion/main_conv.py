"""The pgw#1073 conv variant's worker function — static-rows, mixed dtype.

Separate module for a separate FAMILY (`micro-conv`), the `main_4d` precedent:
everything about the endpoint shape — catalog slot with no code default,
declaration registered at import, `ctx.slots` dereferenced first — is
deliberately identical to `main`, so a difference in outcome is a difference
in the GRAPH CLASS under test (conv-bearing static-rows with an int64
timestep) and nothing else.
"""

from __future__ import annotations

import msgspec
import torch

from gen_worker import Compile, RequestContext, Resources, Slot, endpoint
from gen_worker.families import GenerationDefaults, family

from .aot_declaration_conv import (  # noqa: F401 — registers at import
    CFG_ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    LATENT_ROWS,
    PIXEL_ROWS,
)
from .model_conv import NUM_TRAIN_TIMESTEPS
from .pipeline import MicroConvPipeline


@family(FAMILY)
class MicroConvDefaults(GenerationDefaults, frozen=True):
    steps: int = 2


class MicroConvIn(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class MicroConvOut(msgspec.Struct):
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroConvPipeline, selected_by="model")},
    compile=Compile(
        family=FAMILY, targets=("unet",), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class GenerateConv:
    def setup(self, pipeline: MicroConvPipeline) -> None:
        self.pipe = pipeline

    def generate_conv(
        self, ctx: RequestContext[MicroConvDefaults], data: MicroConvIn,
    ) -> MicroConvOut:
        resolved = ctx.slots["pipeline"]
        grid = LATENT_ROWS[0]
        device = self.pipe.device
        config = self.pipe.config
        generator = ctx.generator(1073)
        sample = torch.randn(
            CFG_ARITY, config.in_channels, grid, grid,
            generator=generator, device=device, dtype=torch.float32)
        # int64 and structural: it indexes the embedding table.
        timestep = torch.randint(
            0, NUM_TRAIN_TIMESTEPS, (CFG_ARITY,), generator=generator,
            device=device, dtype=torch.int64)
        cond = torch.randn(
            CFG_ARITY, COND_LEN, config.cond_dim,
            generator=generator, device=device, dtype=torch.float32)
        with torch.no_grad():
            out = self.pipe.unet(sample, timestep, cond)
        return MicroConvOut(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in out.shape)))


__all__ = ["DECLARATION", "FAMILY", "GenerateConv", "MicroConvDefaults",
           "MicroConvIn", "MicroConvOut"]
