"""qwen-edit-hub-image — Qwen-Image-Edit-2511 editing from OUR R2-backed repo-cas.

The model binding is ``Hub(...)``: the platform resolves the ref to a
repo-cas snapshot (mirrored bf16 from ``Qwen/Qwen-Image-Edit-2511``, the
newest edit iteration — 20B MMDiT + Qwen2.5-VL-7B text encoder, ~54GB) and
downloads every blob through presigned R2 GETs — zero huggingface.co
traffic. 2509+ edit iterations use ``QwenImageEditPlusPipeline``.

Exercises the typed-media INPUT path: ``image`` arrives as a hub-approved
URL-ref Asset the worker materializes to ``local_path`` before the handler
runs (gen-worker input-asset materialization).
"""

from __future__ import annotations

from typing import Annotated, Optional

import msgspec

from diffusers import QwenImageEditPlusPipeline
from gen_worker import Hub, RequestContext, Resources, ValidationError, endpoint
from gen_worker import io as gw_io
from gen_worker.api.types import ImageAsset

HUB_REF = "tensorhub/qwen-image-edit-2511"


class EditInput(msgspec.Struct):
    image: ImageAsset
    prompt: str = ""
    negative_prompt: str = " "
    steps: Annotated[int, msgspec.Meta(ge=1, le=60)] = 28
    true_cfg_scale: Annotated[float, msgspec.Meta(ge=1.0, le=12.0)] = 4.0
    seed: Optional[int] = None


class EditOutput(msgspec.Struct):
    image: ImageAsset
    width: int
    height: int


@endpoint(model=Hub(HUB_REF, tag="prod"), resources=Resources(vram_gb=60))
class QwenImageEdit:
    def setup(self, model: QwenImageEditPlusPipeline) -> None:
        self._pipe = model

    def edit(self, ctx: RequestContext, p: EditInput) -> EditOutput:
        import torch

        if not str(p.prompt or "").strip():
            raise ValidationError("prompt (edit instruction) required")
        source = gw_io.read_image(p.image, mode="RGB")

        ctx.raise_if_cancelled()
        with torch.inference_mode():
            image = self._pipe(
                image=[source],
                prompt=p.prompt,
                negative_prompt=p.negative_prompt or " ",
                true_cfg_scale=float(p.true_cfg_scale),
                num_inference_steps=int(p.steps),
                num_images_per_prompt=1,
                generator=ctx.generator(p.seed) if p.seed is not None else None,
            ).images[0]

        asset = gw_io.write_image(ctx, "image", image, as_type=ImageAsset)
        return EditOutput(image=asset, width=image.width, height=image.height)


__all__ = ["QwenImageEdit", "EditInput", "EditOutput", "HUB_REF"]
