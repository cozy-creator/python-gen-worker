"""Three entrypoints; two carry a payload field the enumerator cannot reach.

minimax-h3's exact shape (pgw#1449): `generate` enumerates, while
`generate_long.slots: list[LongVideoSlot]` and
`reference_to_video.references: list[ReferenceInput]` are required
list-of-struct fields with no default and no axes. Under fail-the-module,
ONE of those cost the whole derive — and this endpoint has two, so the
"drop the other decorator" workaround was iterative.

The enumerable entrypoint carries a real enum axis so the document has
specializations to show, and every entrypoint drives the SAME marked module:
the point is that a refused signature costs the module nothing, not that the
survivors are trivial.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import MiB, const, per_mp_batch
from gen_worker.models import SDXL
from lane_contracts import TINY_DIFFUSERS_FP32


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


_BUCKETS: dict[Size, int] = {Size.SMALL: 32, Size.LARGE: 64}


class Slot(msgspec.Struct):
    """A struct the enumerator has no way to invent a value for."""

    prompt: str
    seconds: float


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class LongInput(msgspec.Struct, forbid_unknown_fields=True):
    #: REQUIRED, no default, no axes — h3's `generate_long.slots`.
    slots: list[Slot]
    size: Size = Size.SMALL


class ReferenceInput(msgspec.Struct, forbid_unknown_fields=True):
    #: The second one, so the fixture proves the fix is not "skip the first".
    references: list[Slot]


class Out(msgspec.Struct):
    model_used: str


class MixedModel(
    Model[SDXL],
    lanes={TINY_DIFFUSERS_FP32: lane(
        request=const(MiB(64)) + per_mp_batch(MiB(16)),
    )},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


def _run(model: MixedModel, ctx: Any, side: int, prompt: str) -> None:
    with torch.inference_mode():
        model.pipe(
            prompt=prompt,
            num_inference_steps=2,
            guidance_scale=0.0,
            width=side,
            height=side,
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )


@entrypoint
def generate(ctx: RequestContext, payload: GenerateInput, model: MixedModel) -> Out:
    ctx.raise_if_cancelled()
    _run(model, ctx, _BUCKETS[payload.size], payload.prompt)
    return Out(model_used=ctx.checkpoint_ref)


@entrypoint
def generate_long(ctx: RequestContext, payload: LongInput, model: MixedModel) -> Out:
    ctx.raise_if_cancelled()
    _run(model, ctx, _BUCKETS[payload.size], payload.slots[0].prompt)
    return Out(model_used=ctx.checkpoint_ref)


@entrypoint
def reference_to_video(
    ctx: RequestContext, payload: ReferenceInput, model: MixedModel
) -> Out:
    ctx.raise_if_cancelled()
    _run(model, ctx, 32, payload.references[0].prompt)
    return Out(model_used=ctx.checkpoint_ref)
