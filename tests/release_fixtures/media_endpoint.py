from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import msgspec

from gen_worker import (
    AudioAsset,
    PromptRole,
    RequestContext,
    ValidationError,
    VideoAsset,
    entrypoint,
)


class Cue(msgspec.Struct, forbid_unknown_fields=True):
    """A NESTED asset, reached through a list — the shape whose field path is `cues[].clip` and the one a flat walk gets wrong."""

    clip: VideoAsset
    note: Annotated[str, PromptRole("negative")] = ""


class ExtractFrameInput(msgspec.Struct, forbid_unknown_fields=True):
    video: VideoAsset
    caption: Annotated[str, PromptRole("positive")] = ""
    cues: list[Cue] = []
    at_seconds: float = 0.0


class ExtractFrameOutput(msgspec.Struct):
    local_path: str
    size_bytes: int
    nested_paths: list[str]


class AnalyzeInput(msgspec.Struct, forbid_unknown_fields=True):
    audio: AudioAsset
    hint: Optional[str] = None


class AnalyzeOutput(msgspec.Struct):
    size_bytes: int


def _local_path(asset: object, what: str) -> str:
    path = getattr(asset, "local_path", None)
    if not path:
        raise ValidationError(f"{what} asset not materialized")
    return str(path)


@entrypoint
def extract_frame(
    ctx: RequestContext, payload: ExtractFrameInput
) -> ExtractFrameOutput:
    path = _local_path(payload.video, "video")
    nested = [_local_path(cue.clip, "cue") for cue in payload.cues]
    return ExtractFrameOutput(
        local_path=path,
        size_bytes=len(Path(path).read_bytes()),
        nested_paths=nested,
    )


@entrypoint
def analyze(ctx: RequestContext, payload: AnalyzeInput) -> AnalyzeOutput:
    path = _local_path(payload.audio, "audio")
    return AnalyzeOutput(size_bytes=len(Path(path).read_bytes()))
