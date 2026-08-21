"""THE FULLY-DECLARING ENDPOINT — every author-side declaration at once."""

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
    image: Annotated[ImageAsset, ExpectedOutput(aspect_ratio="input.aspect_ratio")]
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
    """`dj-pipeline/main.py:498`, re-decorated: composes the pipeline out of child requests to music-analysis / ltx-video-2.3 / dj-utils."""
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
    """EVERY declaration at once."""
    raise NotImplementedError


@entrypoint
def describe(ctx: RequestContext, payload: DescribeInput) -> DescribeResult:
    """Declares nothing — the undeclared control."""
    return DescribeResult(summary=payload.ref)
