"""sd15-image — real GPU inference: Stable Diffusion 1.5 text-to-image.

The platform downloads the HF snapshot through ``ensure_local`` (fp16
variant only), constructs the pipeline from the ``setup()`` annotation
(dtype + placement are worker policy), and the handler returns the PNG
through the typed-parts output path (>64KB rides blob_ref, not inline).
"""

from __future__ import annotations

import msgspec

from diffusers import StableDiffusionPipeline
from gen_worker import HF, RequestContext, Resources, ValidationError, endpoint
from gen_worker import io as gw_io
from gen_worker.api.types import Asset


class SD15Input(msgspec.Struct):
    prompt: str = ""
    steps: int = 12
    width: int = 512
    height: int = 512
    seed: int = 0


class SD15Output(msgspec.Struct):
    image: Asset
    width: int
    height: int
    seed: int


@endpoint(
    model=HF(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        dtype="fp16",
        files=("*.json", "*.txt", "*.fp16.safetensors"),
    ),
    resources=Resources(vram_gb=6),
)
class SD15Image:
    def setup(self, model: StableDiffusionPipeline) -> None:
        self._pipe = model

    def generate(self, ctx: RequestContext, p: SD15Input) -> SD15Output:
        import torch

        if not str(p.prompt or "").strip():
            raise ValidationError("prompt required")
        if p.width % 8 or p.height % 8 or not (64 <= p.width <= 1024) or not (64 <= p.height <= 1024):
            raise ValidationError("width/height must be multiples of 8 in [64, 1024]")
        steps = max(1, min(int(p.steps), 50))

        ctx.raise_if_cancelled()
        generator = torch.Generator(device=self._pipe.device).manual_seed(int(p.seed))
        image = self._pipe(
            p.prompt,
            num_inference_steps=steps,
            width=p.width,
            height=p.height,
            generator=generator,
        ).images[0]

        asset = gw_io.write_image(ctx, "image", image)
        return SD15Output(image=asset, width=image.width, height=image.height, seed=p.seed)


__all__ = ["SD15Image", "SD15Input", "SD15Output"]
