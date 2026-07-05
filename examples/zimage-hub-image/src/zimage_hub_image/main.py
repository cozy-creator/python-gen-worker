"""zimage-hub-image — Z-Image-Turbo text-to-image from OUR R2-backed repo-cas.

The model binding is ``Hub(...)``: the platform resolves the ref to a
repo-cas snapshot and downloads every blob through presigned R2 GETs,
blake3-verified — the worker never talks to huggingface.co. Driven by the
cozy e2e J8 full-cycle journey: a prior mirror step (``clone-huggingface``,
``outputs=[bf16]``) publishes ``Tongyi-MAI/Z-Image-Turbo`` (fp32 DiT on HF,
cast to bf16 at ingest) into repo-cas under this ref.

Z-Image-Turbo is step-distilled: ~9 steps, guidance disabled.
"""

from __future__ import annotations

from typing import Annotated, Optional

import msgspec

from diffusers import ZImagePipeline
from gen_worker import Hub, RequestContext, Resources, ValidationError, endpoint
from gen_worker import io as gw_io
from gen_worker.api.types import ImageAsset

HUB_REF = "tensorhub/z-image-turbo"


class ZImageInput(msgspec.Struct):
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: Annotated[int, msgspec.Meta(ge=1, le=16)] = 9
    guidance_scale: float = 0.0
    seed: Optional[int] = None


class ZImageOutput(msgspec.Struct):
    image: ImageAsset
    width: int
    height: int


@endpoint(model=Hub(HUB_REF, tag="prod"), resources=Resources(vram_gb=22))
class ZImageTurbo:
    def setup(self, model: ZImagePipeline) -> None:
        self._pipe = model

    def generate(self, ctx: RequestContext, p: ZImageInput) -> ZImageOutput:
        import torch

        if not str(p.prompt or "").strip():
            raise ValidationError("prompt required")
        if p.width % 16 or p.height % 16 or not (512 <= p.width <= 2048) or not (512 <= p.height <= 2048):
            raise ValidationError("width/height must be multiples of 16 in [512, 2048]")

        ctx.raise_if_cancelled()
        with torch.no_grad():
            image = self._pipe(
                prompt=p.prompt,
                negative_prompt=p.negative_prompt,
                width=p.width,
                height=p.height,
                num_inference_steps=int(p.steps),
                guidance_scale=float(p.guidance_scale),
                generator=ctx.generator(p.seed) if p.seed is not None else None,
            ).images[0]

        asset = gw_io.write_image(ctx, "image", image, format="png", as_type=ImageAsset)
        return ZImageOutput(image=asset, width=image.width, height=image.height)


__all__ = ["ZImageTurbo", "ZImageInput", "ZImageOutput", "HUB_REF"]
