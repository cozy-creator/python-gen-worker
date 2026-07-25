"""th#1130 harness endpoints: image handlers whose encode+upload tail must run
AFTER the GPU permit is released.

Every handler here calls ``ctx.save_image`` — the surface the whole live image
fleet uses (19 of 19 call sites) — and marks its own timeline into
:data:`EVENTS` with ``time.monotonic()``. The hub-double runs the worker
in-process, so those marks are directly comparable with the test's own clock:
that is how "request B's GPU phase started while request A was still encoding"
is proven without a GPU.

No torch, no weights. Payload sizes are kept under the 64 KiB inline-result cap
so the encoded bytes come back on the wire and can be decoded and asserted.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Tuple

import msgspec
from PIL import Image

from gen_worker import RequestContext, diffusers_step_callback, endpoint
from gen_worker.api.types import ImageAsset

#: Handler shape the tests assert against (seconds / pixels).
STEP_S = 0.02
STEPS = 4
#: A smooth 1024^2 frame: ~250ms of webp q95/method=6 encode for ~29 KB of
#: output — a tail long enough to observe, small enough to ride the inline
#: result. (Noise of the same size is 370ms but 800 KB.)
SLOW_PX = 1024
#: GPU-shaped work the N-image loop does BETWEEN saves: if any save released
#: the permit, a peer job would start during these.
LOOP_GPU_WORK_S = 0.15
LOOP_IMAGES = 3

EVENTS: List[Tuple[str, str, float]] = []
_lock = threading.Lock()


def mark(label: str, ctx: RequestContext) -> float:
    now = time.monotonic()
    with _lock:
        EVENTS.append((label, ctx.request_id, now))
    return now


def at(label: str, request_id: str) -> float:
    """The monotonic time of one mark; raises if it never happened."""
    with _lock:
        for name, rid, when in EVENTS:
            if name == label and rid == request_id:
                return when
    raise AssertionError(f"no {label!r} mark for {request_id!r} in {EVENTS}")


def counts() -> Dict[str, int]:
    with _lock:
        out: Dict[str, int] = {}
        for name, _rid, _when in EVENTS:
            out[name] = out.get(name, 0) + 1
        return out


def reset() -> None:
    with _lock:
        EVENTS.clear()


def gradient(px: int) -> Image.Image:
    """A smooth RGB frame — cheap to build, slow to webp-encode, small once
    encoded."""
    base = Image.merge("RGB", (
        Image.linear_gradient("L"),
        Image.linear_gradient("L").rotate(90),
        Image.radial_gradient("L"),
    ))
    return base.resize((px, px), Image.Resampling.BILINEAR)


class _ExplodingImage:
    """PIL-shaped enough for the encode core, and it fails IN THE TAIL."""

    mode = "RGB"

    def copy(self) -> "_ExplodingImage":
        return self

    def save(self, buf: object, **_kwargs: object) -> None:
        raise ValueError("encoder exploded")


class GenIn(msgspec.Struct):
    prompt: str = ""


class GenOut(msgspec.Struct):
    image: ImageAsset


class MultiOut(msgspec.Struct):
    images: list[ImageAsset]


class SizeOut(msgspec.Struct):
    ref: str
    size_bytes: int


@endpoint
class DeferredImages:
    def slow_encode(self, ctx: RequestContext, data: GenIn) -> GenOut:
        """A diffusion-shaped handler: stepped denoise, then one save whose
        encode is the whole tail."""
        mark("handler_start", ctx)
        on_step = diffusers_step_callback(ctx, STEPS)
        for i in range(STEPS):
            time.sleep(STEP_S)
            on_step(None, i, None, {})
        asset = ctx.save_image(
            gradient(SLOW_PX), format="webp", quality=95, method=6)
        mark("handler_end", ctx)
        return GenOut(image=asset)

    def fast_peer(self, ctx: RequestContext, data: GenIn) -> GenOut:
        """Request B. Its handler_start mark is the moment it holds the GPU
        permit (the executor acquires before calling the handler)."""
        mark("handler_start", ctx)
        asset = ctx.save_image(gradient(64), format="webp")
        mark("handler_end", ctx)
        return GenOut(image=asset)

    def n_images(self, ctx: RequestContext, data: GenIn) -> MultiOut:
        """N saves with GPU-shaped work AFTER each one — the case a blanket
        terminal release on save would break."""
        mark("handler_start", ctx)
        assets = []
        for i in range(LOOP_IMAGES):
            assets.append(ctx.save_image(
                gradient(256), f"outputs/{ctx.request_id}/img{i}", format="webp"))
            mark(f"saved_{i}", ctx)
            time.sleep(LOOP_GPU_WORK_S)  # stands in for more GPU work
        mark("handler_end", ctx)
        return MultiOut(images=assets)

    def fails_in_tail(self, ctx: RequestContext, data: GenIn) -> GenOut:
        asset = ctx.save_image(_ExplodingImage(), "outputs/boom.webp")
        mark("handler_end", ctx)
        return GenOut(image=asset)

    def mutates_after_save(self, ctx: RequestContext, data: GenIn) -> GenOut:
        """Saves a RED frame, then keeps painting on the same PIL object."""
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        asset = ctx.save_image(img, "outputs/mutated", format="png")
        img.paste((0, 0, 255), (0, 0, 32, 32))
        mark("handler_end", ctx)
        return GenOut(image=asset)

    def reads_back(self, ctx: RequestContext, data: GenIn) -> SizeOut:
        """Reads a bytes field inside the handler — must be REAL, which means
        the deferred encode is forced inline rather than answered with None."""
        asset = ctx.save_image(gradient(64), "outputs/readback", format="png")
        mark("handler_end", ctx)
        return SizeOut(ref=asset.ref, size_bytes=int(asset.size_bytes or 0))
