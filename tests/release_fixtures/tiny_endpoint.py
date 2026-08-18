"""A main_v2-shaped lanes endpoint over the tiny pipeline (derive fixture).

Deliberately spelled like the Paul-reviewed sdxl ``main_v2.py``: code as-is
(stock diffusers serve host), ``lanes=`` as the whole compile surface, trace
coverage auto-enumerated from the payload schemas (the ``Size`` enum is this
fixture's aspect-ratio analogue), ``ctx.checkpoint_defaults(SDXL)`` +
``Knob.resolve`` for serving values, no models=, no catalog.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import ImageAsset, RequestContext, endpoint
from gen_worker.models import SDXL
from torchcg import Lane


class Size(StrEnum):
    SMALL = "small"
    LARGE = "large"


_BUCKETS: dict[Size, int] = {Size.SMALL: 32, Size.LARGE: 64}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL
    guidance_scale: float | None = None
    num_inference_steps: int | None = None


class TurboInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    size: Size = Size.SMALL


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    model_used: str


@endpoint(
    lanes=(
        Lane("fp32", compile=("unet",), contract="plain.fp32@1",
             dtype=torch.float32),
    ),
    # Trace coverage auto-enumerates the Size enum through both handlers:
    # 2 CFG batch-2 graphs + 2 batch-1 graphs = 4 graph classes.
)
class TinyDiffusion:
    def setup(self, ctx: RequestContext) -> None:
        self.pipe = StableDiffusionPipeline.from_pretrained(
            ctx.checkpoint_dir, torch_dtype=ctx.lane.dtype
        ).to("cuda")
        self.defaults = ctx.checkpoint_defaults(SDXL)

    def _run(self, ctx: RequestContext, *, steps: int, seed: Optional[int],
             **call_kwargs) -> ImageAsset:
        generator = (
            torch.Generator("cuda").manual_seed(seed) if seed is not None else None
        )
        with torch.inference_mode():
            result = self.pipe(
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=ctx.step_callback(steps),
                output_type="pil",
                **call_kwargs,
            )
        return ctx.save_image(result.images[0], format="png")

    def generate(self, ctx: RequestContext, payload: GenerateInput) -> ImageOutput:
        ctx.raise_if_cancelled()
        d = self.defaults
        steps = d.steps.resolve(payload.num_inference_steps or 2, ctx)
        guidance = d.guidance.resolve(payload.guidance_scale, ctx)
        side = _BUCKETS[payload.size]
        image = self._run(
            ctx, steps=int(steps), seed=None,
            prompt=payload.prompt.strip(), guidance_scale=guidance,
            width=side, height=side,
        )
        return ImageOutput(image=image, model_used=ctx.checkpoint_ref)

    def generate_turbo(self, ctx: RequestContext, payload: TurboInput) -> ImageOutput:
        ctx.raise_if_cancelled()
        side = _BUCKETS[payload.size]
        image = self._run(
            ctx, steps=2, seed=None,
            prompt=payload.prompt.strip(), guidance_scale=0.0,
            width=side, height=side,
        )
        return ImageOutput(image=image, model_used=ctx.checkpoint_ref)
