"""THE FULLY-DECLARING ENDPOINT — every author-side declaration at once.

The control for ``tests/test_manifest_declarations_pgw1579_1580.py``. Its job is
to be the row that carries EVERYTHING, so a fence can pin the emitted key set
and fail the moment a hardcut drops one of them again. ``producer_endpoint.py``
is its opposite number and pins the row that declares NOTHING.

* ``make_video`` — ``child_calls=True`` (pgw#1579). The
  ``private-inference-endpoints/dj-pipeline`` shape, verbatim in structure: an
  ordinary CPU body whose whole purpose is composing OTHER endpoints out of
  child requests. Weightless, so it also proves the declaration does not depend
  on a model slot.
* ``render`` — ``handles=`` (pgw#1580) plus the ``ExpectedOutput`` annotations
  on its return struct, in both shapes the fleet writes: a single asset with a
  payload-ref aspect (``anima``'s exact annotation) and a list whose count is a
  payload ref (``sdxl``'s shape).
* ``everything`` — every declaration at once, including pgw#1576's
  ``streams=``. Nothing in the fleet looks like this; it exists so ONE row can
  pin the complete emitted key set.
* ``describe`` — declares NOTHING and returns an un-annotated struct.
"""

from __future__ import annotations

from typing import Annotated

import msgspec

from gen_worker import (
    ExpectedOutput,
    ImageAsset,
    RequestContext,
    Resources,
    TokenDelta,
    VideoAsset,
    entrypoint,
)


class DjInput(msgspec.Struct, forbid_unknown_fields=True):
    track_url: str
    duration_s: int = 30


class DjOutput(msgspec.Struct):
    video: Annotated[
        VideoAsset,
        ExpectedOutput(media_type="video", mime_type="video/mp4",
                       duration_s="input.duration_s"),
    ]


class RenderInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    num_images: int = 1
    aspect_ratio: str = "1:1"
    width: int = 1024
    height: int = 1024


class RenderOutput(msgspec.Struct):
    #: `anima`'s annotation, character for character — the row measured on the
    #: live hub as `{"type":"image","count":1,"field":"image",
    #: "aspect_ratio":"input.aspect_ratio"}` before its pointer moved to v2,
    #: and as `null` after.
    image: Annotated[ImageAsset, ExpectedOutput(aspect_ratio="input.aspect_ratio")]
    #: The multi-output shape: count and dimensions are payload refs.
    thumbnails: Annotated[
        list[ImageAsset],
        ExpectedOutput(count="input.num_images", width="input.width",
                       height="input.height", mime_type="image/webp"),
    ]
    model: str


class DescribeInput(msgspec.Struct, forbid_unknown_fields=True):
    ref: str


class DescribeResult(msgspec.Struct):
    summary: str


@entrypoint(child_calls=True)
def make_video(ctx: RequestContext, payload: DjInput) -> DjOutput:
    """`dj-pipeline/main.py:498`, re-decorated: composes the pipeline out of
    child requests to music-analysis / ltx-video-2.3 / dj-utils."""
    raise NotImplementedError


@entrypoint(handles=("fp8-w8a8-dynamic", "bf16-w16a16"))
def render(ctx: RequestContext, payload: RenderInput) -> RenderOutput:
    """Branches on the executing lane, and says so."""
    raise NotImplementedError


@entrypoint(
    kind="conversion",
    resources=Resources(vcpus=8),
    publishes=True,
    env=("HF_TOKEN",),
    emits_media=True,
    child_calls=True,
    handles=("fp8-w8a8-dynamic",),
    streams=TokenDelta,
)
def everything(ctx: RequestContext, payload: RenderInput) -> RenderOutput:
    """EVERY declaration at once. Nothing in the fleet looks like this; it
    exists so one row can pin the complete emitted key set, and so dropping any
    single emission fails a fence instead of a production request."""
    raise NotImplementedError


@entrypoint
def describe(ctx: RequestContext, payload: DescribeInput) -> DescribeResult:
    """Declares nothing — the undeclared control."""
    return DescribeResult(summary=payload.ref)
