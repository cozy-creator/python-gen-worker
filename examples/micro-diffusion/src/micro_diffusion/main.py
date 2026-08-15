"""The micro-diffusion worker function — a REAL org worker, deliberately small.

It exists so that a full production-path AOT mint cycle costs minutes instead
of the ~95 that sdxl's 36 entries cost on a pod. Everything about its SHAPE is
production: a catalog slot with no code default, a declaration registered at
import, two served arms that map onto the two declared fork coordinates, and a
Dockerfile-first build. The only thing that is small is the model.

Two shapes here are load-bearing and neither is incidental:

**The slot is a CATALOG slot** — ``Slot(MicroPipeline, selected_by="model")``
with no ``default_checkpoint=``. That is sdxl's shape and it is the shape
pgw#969 died on: the mint child re-runs discovery in a fresh interpreter, so a
catalog slot rediscovers NOTHING and ``ctx.slots["pipeline"]`` raises unless
the parent's ``MintSlot`` crossed the delegation boundary intact.

**The handler dereferences ``ctx.slots`` first.** That is the exact call that
crashed 0.0 s into ``warmup_forward`` on two L40S pods.
"""

from __future__ import annotations

from typing import List

import msgspec
import torch

from gen_worker import (
    Compile,
    RequestContext,
    Resources,
    Slot,
    StringEnum,
    endpoint,
)
from gen_worker.families import GenerationDefaults, family

from .aot_declaration import (  # noqa: F401 — registers at import, as a real endpoint does
    CFG_ARITY,
    COND_LEN,
    DECLARATION,
    FAMILY,
    LATENT_ROWS,
    PIXEL_ROWS,
    TOKEN_ROWS,
    VAE_SCALE,
)
from .pipeline import MicroPipeline


class Size(StringEnum):
    """The served latent rows, as the payload's legality oracle (pgw#669).

    A bare width/height int would be a free shape-affecting field — the thing
    every fleet endpoint is linted against — so the rows are an enum and the
    latent math lives in the declaration, once.
    """

    SMALL = "256x256"
    LARGE = "384x384"


_LATENT_BY_SIZE = {
    Size.SMALL: LATENT_ROWS[0],
    Size.LARGE: LATENT_ROWS[1],
}


@family(FAMILY)
class MicroDefaults(GenerationDefaults, frozen=True):
    steps: int = 2
    guidance: float = 4.0


class MicroIn(msgspec.Struct):
    prompt: str = ""
    size: Size = Size.SMALL
    #: ``selected_by="model"`` binds the catalog slot off this field.
    model: str = ""


class MicroOut(msgspec.Struct):
    #: The saved image, or "" on the boot warm pass (which saves nothing).
    image: str = ""
    #: Which checkpoint actually served, read off the resolved slot — so a
    #: cycle can PROVE which bytes were traced rather than trusting that some
    #: were.
    checkpoint: str = ""
    shape: str = ""


@endpoint(
    models={"pipeline": Slot(MicroPipeline, selected_by="model",
        layouts={"*": ("plain.bf16@1",)})},
    compile=Compile(
        family=FAMILY, targets=("transformer", "decoder"), shapes=PIXEL_ROWS,
        text_len=COND_LEN),
    resources=Resources(gpu=True),
)
class Generate:
    def setup(self, pipeline: MicroPipeline) -> None:
        self.pipe = pipeline

    def generate(self, ctx: RequestContext[MicroDefaults], data: MicroIn) -> MicroOut:
        """The guided arm — ``cfg=true``, container arity 2."""
        return self._run(ctx, data, arity=CFG_ARITY,
                         guidance=ctx.defaults.guidance)

    def generate_turbo(
        self, ctx: RequestContext[MicroDefaults], data: MicroIn,
    ) -> MicroOut:
        """The turbo arm — guidance pinned to 0.0, so ``cfg=false`` and the
        container arity is 1. A separate METHOD rather than a flag, because
        the two arms are two traced graphs."""
        return self._run(ctx, data, arity=1, guidance=0.0)

    def _run(
        self, ctx: RequestContext[MicroDefaults], data: MicroIn, *,
        arity: int, guidance: float,
    ) -> MicroOut:
        resolved = ctx.slots["pipeline"]
        grid = _LATENT_BY_SIZE[Size(data.size)]
        device = self.pipe.device
        config = self.pipe.config
        generator = ctx.generator(abs(hash(data.prompt)) % (2 ** 31))
        # Token sequences, not latent grids: patchify is the PIPELINE's job
        # here, the way it is in flux and qwen.
        latents: List[torch.Tensor] = [
            torch.randn(grid * grid, config.in_channels,
                        generator=generator, device=device, dtype=torch.float32)
            for _ in range(arity)
        ]
        cond: List[torch.Tensor] = [
            torch.randn(COND_LEN, config.cond_dim, generator=generator,
                        device=device, dtype=torch.float32)
            for _ in range(arity)
        ]
        steps = 1 if ctx.boot_warmup else ctx.defaults.steps
        pixels = self.pipe(latents, cond, grid=grid, steps=steps,
                           guidance=guidance)
        out = MicroOut(
            checkpoint=str(resolved.ref.path),
            shape=str(tuple(int(n) for n in pixels.shape)))
        if ctx.boot_warmup:
            # The warm pass proves the GRAPH, not the encoder. Saving here
            # would put an image encode on the boot critical path for nothing.
            return out
        out.image = ctx.save_image(_to_image(pixels[0])).ref
        return out


def _to_image(pixels: torch.Tensor) -> object:
    from PIL import Image

    array = ((pixels.detach().float().clamp(-1, 1) + 1) * 127.5)
    array = array.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array)


__all__ = [
    "DECLARATION",
    "TOKEN_ROWS",
    "FAMILY",
    "VAE_SCALE",
    "Generate",
    "MicroDefaults",
    "MicroIn",
    "MicroOut",
    "Size",
]
