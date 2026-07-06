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
from gen_worker import Compile, HF, RequestContext, Resources, ValidationError, endpoint
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
    # Opt into torch.compile (#384). Safe by construction: the worker arms
    # compile ONLY when a verified per-(model, SKU, torch, triton) cache
    # artifact is seeded (GEN_WORKER_COMPILE_CACHE[_URL]); otherwise eager.
    compile=Compile(shapes=((768, 768), (1024, 1024))),
)
class Flux2KleinTurbo:
    def setup(self, model: Flux2KleinPipeline) -> None:
        self._pipe = model

    def warmup(self) -> None:
        """When compile is armed, pay the (cache-served) compile cost at boot
        over the declared shapes so no request ever sees the stall."""
        info = getattr(self._pipe, "_cozy_compile", None)
        if not info:
            return
        for w, h in info.get("shapes") or ():
            self._pipe(
                prompt="warmup",
                num_inference_steps=1,
                width=int(w),
                height=int(h),
                generator=torch.Generator(device=self._pipe.device).manual_seed(0),
            )

    def generate_turbo(self, ctx: RequestContext, p: KleinTurboInput) -> KleinTurboOutput:
        if not str(p.prompt or "").strip():
            raise ValidationError("prompt required")
        if p.width % 16 or p.height % 16 or not (512 <= p.width <= 1360) or not (512 <= p.height <= 1360):
            raise ValidationError("width/height must be multiples of 16 in [512, 1360]")
        steps = max(4, min(int(p.steps), 8))

        ctx.raise_if_cancelled()
        generator = torch.Generator(device=self._pipe.device).manual_seed(int(p.seed))
        # Flux2KleinPipeline.__call__'s FIRST positional parameter is `image`
        # (it is a text-to-image AND editing pipeline) — prompt must be a
        # keyword or it lands in the image slot and the pipeline rejects the
        # call with "Provide either `prompt` or `prompt_embeds`".
        image = self._pipe(
            prompt=p.prompt,
            num_inference_steps=steps,
            width=p.width,
            height=p.height,
            generator=generator,
        ).images[0]

        # WEBP q90: measured visually identical to PNG for these outputs at
        # ~15% of the bytes and ~2.7x less encode CPU (#382) - the upload, not
        # the GPU, dominated per-image latency with PNG.
        asset = gw_io.write_image(ctx, "image", image, format="webp", as_type=ImageAsset)
        return KleinTurboOutput(image=asset, width=image.width, height=image.height, seed=p.seed)


__all__ = ["Flux2KleinTurbo", "KleinTurboInput", "KleinTurboOutput"]
