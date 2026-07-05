"""flux2-klein-image — real GPU inference: FLUX.2-klein-4B (turbo) text-to-image.

Minimal single-function port of inference-endpoints/flux.2-klein-4b's
``generate_turbo`` onto gen-worker's v2 authoring surface (``@endpoint`` /
``HF`` / ``Resources`` — the source repo predates #368 and uses the retired
``@inference``/``Case``/``HFRepo`` surface, which does not exist here).

The FLUX.2-klein-4B repo ships a redundant root-level single-file checkpoint
(``flux-2-klein-4b.safetensors``, ~7.75GB) alongside the real diffusers-layout
weights (``transformer/``, ``text_encoder/``, ``vae/``) diffusers actually
loads via ``model_index.json``. ``files=`` narrows the snapshot download to
just the layout diffusers needs (~16GB) instead of both copies (~24GB).
"""

from __future__ import annotations

import msgspec
import torch

from diffusers import Flux2KleinPipeline
from gen_worker import HF, RequestContext, Resources, ValidationError, endpoint
from gen_worker import io as gw_io
from gen_worker.api.types import ImageAsset


class KleinTurboInput(msgspec.Struct):
    prompt: str = ""
    steps: int = 4
    width: int = 768
    height: int = 768
    seed: int = 0


class KleinTurboOutput(msgspec.Struct):
    image: ImageAsset
    width: int
    height: int
    seed: int


@endpoint(
    model=HF(
        "black-forest-labs/FLUX.2-klein-4B",
        dtype="bf16",
        files=(
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/*",
        ),
    ),
    resources=Resources(vram_gb=20),
)
class Flux2KleinTurbo:
    def setup(self, model: Flux2KleinPipeline) -> None:
        self._pipe = model

    def generate_turbo(self, ctx: RequestContext, p: KleinTurboInput) -> KleinTurboOutput:
        if not str(p.prompt or "").strip():
            raise ValidationError("prompt required")
        if p.width % 16 or p.height % 16 or not (512 <= p.width <= 1360) or not (512 <= p.height <= 1360):
            raise ValidationError("width/height must be multiples of 16 in [512, 1360]")
        steps = max(4, min(int(p.steps), 8))

        ctx.raise_if_cancelled()
        generator = torch.Generator(device=self._pipe.device).manual_seed(int(p.seed))
        image = self._pipe(
            p.prompt,
            num_inference_steps=steps,
            width=p.width,
            height=p.height,
            generator=generator,
        ).images[0]

        asset = gw_io.write_image(ctx, "image", image, format="png", as_type=ImageAsset)
        return KleinTurboOutput(image=asset, width=image.width, height=image.height, seed=p.seed)


__all__ = ["Flux2KleinTurbo", "KleinTurboInput", "KleinTurboOutput"]
