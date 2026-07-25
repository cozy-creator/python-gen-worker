"""th#1111 harness endpoint: an image-shaped request with real stage structure.

Mirrors what an sdxl-class handler actually does — text encode, a stepped
denoise loop driving the SHARED ``diffusers_step_callback``, an unbracketed
VAE-decode-shaped gap, then the framework's own encode path via
``gw_io.write_image`` — so the executor produces a real ``stage_ms`` map over
the real serving path. No torch, no GPU, no weights.
"""

from __future__ import annotations

import time

import msgspec
from PIL import Image

from gen_worker import RequestContext, diffusers_step_callback, endpoint
from gen_worker import io as gw_io
from gen_worker.api.types import ImageAsset

#: Stage durations the test asserts against (seconds).
TEXT_ENCODE_S = 0.06
STEP_S = 0.04
STEPS = 4
#: Deliberately NOT bracketed: stands in for the VAE decode no endpoint
#: brackets today, which must show up as resid.tail rather than vanish.
DECODE_S = 0.05


class GenIn(msgspec.Struct):
    prompt: str = ""


class GenOut(msgspec.Struct):
    image: ImageAsset


@endpoint
class Staged:
    def staged_generate(self, ctx: RequestContext, data: GenIn) -> GenOut:
        with ctx.stage("text_encode"):
            time.sleep(TEXT_ENCODE_S)

        on_step = diffusers_step_callback(ctx, STEPS)
        for i in range(STEPS):
            time.sleep(STEP_S)
            on_step(None, i, None, {})

        time.sleep(DECODE_S)  # un-bracketed on purpose
        # th#1130 made webp the default with a faster shared encode core, and a 16x16
        # webp encodes in under the 1ms stage_ms resolution -> image_encode rounded to 0
        # and th#1111's `stages["image_encode"] > 0` failed on fast CI runners. Use a
        # size whose encode is reliably measurable; noise, not gradients, so it does not
        # compress to nothing.
        image = Image.effect_noise((512, 512), 64).convert("RGB")
        asset = gw_io.write_image(
            ctx, f"outputs/{ctx.request_id}/image.webp", image,
            format="webp", as_type=ImageAsset,
        )
        return GenOut(image=asset)
