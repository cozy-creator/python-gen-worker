"""sd15's SHAPE FAN, in miniature: three aspects x two CFG modes (pgw#1548).

The real sd15 endpoint derives 14 graph specializations — 2 (CFG) x 7 (aspect
bucket) — and sdxl 18. The COUNT still falls out of the payload enumeration
driving the marked UNet at a different shape each pass, so this fixture
reproduces that structure at fixture scale and the collapse is measured through
the REAL derive, not a stand-in.

What changed (pgw#1599): the aspect axis is DECLARED, not FLAGGED. The global
`--dynamic-axes` derive flag is deleted — a whole-run switch could only ever be
right for every model in the run at once — and the choice now lives on the
model class as `shapes={"aspect": DYNAMIC}`, which is where the person who
knows what a symbolic aspect dim costs THIS model writes it down. This fixture
is the dynamic arm; the static arm is every other fixture's
`shapes={"aspect": STATIC}`. CFG/batch is NOT declarable in either: it is a
permanently static fork (Paul, 2026-08-20), so the x2 stays whatever the aspect
choice is.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import DYNAMIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
#: THE REAL RATIFIED PAIR (pgw#1621): a lane is `(topology, quant)`, both
#: halves documents in the vendored `spec/v2` corpus. `SD15_DIFFUSERS_BF16`
#: was a v1 Contract OBJECT and is deleted with the v1 vocabulary; the
#: spelling it used to carry survives only as a display name.
SD15_DIFFUSERS_BF16 = ("sd15.diffusers@1", "plain.bf16@1")

class Aspect(StrEnum):
    """The aspect axis, spelled the way sd15's own `AspectRatio` is.

    A shape-bearing string axis is a StrEnum, never a string Literal: the
    derive enumerates enums and NUMERIC literals, and deliberately refuses to
    cross-product string literals (they name host-side policy, not shape).
    """

    SQUARE = "square"
    TALL = "tall"
    WIDE = "wide"


#: latent side = pixel side here (the fixture VAE has one block), so these are
#: the three aspect buckets the UNet actually sees.
SHAPES: dict[str, tuple[int, int]] = {
    Aspect.SQUARE: (64, 64),
    Aspect.TALL: (80, 48),
    Aspect.WIDE: (48, 80),
}


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "a cat"
    #: The aspect axis. Each value is one bucket, and the enumeration drives
    #: the marked module once per value.
    aspect: Aspect = Aspect.SQUARE
    #: The CFG axis: guided runs concatenate cond+uncond (batch 2), the
    #: guidance-free path does not (batch 1). Spelled as an int enum because
    #: msgspec refuses a bool Literal.
    guided: Literal[1, 0] = 1


class Out(msgspec.Struct):
    model_used: str


class FanShaped(
    Model[SDXL],
    lanes={SD15_DIFFUSERS_BF16: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    # The declaration under test: one artifact over a symbolic aspect dim,
    # instead of one baked bucket per aspect.
    shapes={"aspect": DYNAMIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: FanShaped) -> Out:
    ctx.raise_if_cancelled()
    height, width = SHAPES[payload.aspect]
    with torch.inference_mode():
        model.pipe(
            prompt=payload.prompt,
            num_inference_steps=1,
            guidance_scale=7.5 if payload.guided else 1.0,
            width=width,
            height=height,
            callback_on_step_end=ctx.step_callback(1),
            output_type="latent",
        )
    return Out(model_used=ctx.checkpoint_ref)
