"""sd15-hub-image — serve Stable Diffusion 1.5 from OUR R2-backed repo-cas.

Same handler as ``examples/sd15-image`` but the model binding is
``Hub(...)`` (tensorhub-provider) instead of ``HF(...)``: the platform
resolves the ref to a repo-cas snapshot and downloads every blob through
presigned R2 GETs, blake3-verified — the worker never talks to
huggingface.co. Driven by the cozy e2e J7 ingest-then-serve journey: a
prior mirror step (``clone-huggingface``) publishes
``stable-diffusion-v1-5/stable-diffusion-v1-5`` into repo-cas under this
ref; this endpoint serves straight from that mirror.
"""

from __future__ import annotations

import msgspec

from diffusers import StableDiffusionPipeline
from gen_worker import Hub, RequestContext, Resources, ValidationError, endpoint
from gen_worker import io as gw_io
from gen_worker.api.types import Asset

# The e2e J7 journey mirrors stable-diffusion-v1-5 into this exact repo
# ref (owner "tensorhub" — the harness-seeded root org, tag "prod").
HUB_REF = "tensorhub/sd15-mirror"


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
    model=Hub(HUB_REF, tag="prod"),
    resources=Resources(vram_gb=6),
)
class SD15HubImage:
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


__all__ = ["SD15HubImage", "SD15Input", "SD15Output", "HUB_REF"]
