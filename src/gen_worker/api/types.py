from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO, Any, Dict, Literal, Optional

import msgspec


class Asset(msgspec.Struct):
    """Reference to a file in the invoking owner's file store."""

    ref: str
    owner: Optional[str] = None
    local_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    blake3: Optional[str] = None
    media_id: Optional[str] = None
    download_token: Optional[str] = None
    stream_mode: Optional[str] = None
    inline_bytes: Optional[bytes] = None
    url_max_bytes: Optional[int] = None
    url_allowed_mime_types: tuple[str, ...] = ()
    url_max_width: Optional[int] = None
    url_max_height: Optional[int] = None
    url_max_pixels: Optional[int] = None
    url_validation_context: Optional[str] = None

    def __fspath__(self) -> str:
        if self.local_path is None:
            raise ValueError("Asset.local_path is not set (file not materialized)")
        return self.local_path


class MediaAsset(Asset):
    """Reference to user-supplied media bytes."""


class ImageAsset(MediaAsset):
    """Reference to image media bytes."""


class VideoAsset(MediaAsset):
    """Reference to video media bytes, plus probed container metadata."""

    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    has_audio: Optional[bool] = None
    sample_rate: Optional[int] = None


class AudioAsset(MediaAsset):
    """Reference to audio media bytes."""




@dataclass(frozen=True)
class ExpectedOutput:
    """Planning metadata for an output media field."""

    count: int | str = 1
    width: int | str | None = None
    height: int | str | None = None
    aspect_ratio: str | None = None
    mime_type: str | None = None
    media_type: Literal["image", "video", "audio", "file", "other"] | None = None
    duration_s: int | str | None = None


@dataclass(frozen=True)
class PromptRole:
    role: Literal["positive", "negative"]

    def __post_init__(self) -> None:
        if self.role not in ("positive", "negative"):
            raise ValueError("PromptRole.role must be 'positive' or 'negative'")


class Tensors(msgspec.Struct):
    """Reference to checkpoint/model-weight artifacts."""

    ref: str
    owner: Optional[str] = None
    local_path: Optional[str] = None
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    blake3: Optional[str] = None
    blob_digest: Optional[str] = None
    blob_domain: Optional[str] = None
    blob_path: Optional[str] = None
    snapshot_digest: Optional[str] = None
    download_token: Optional[str] = None
    stream_mode: Optional[str] = None

    def __fspath__(self) -> str:
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        return self.local_path

    def open(self, mode: str = "rb") -> IO[bytes]:
        if "b" not in mode:
            raise ValueError("Tensors.open only supports binary modes")
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        return open(self.local_path, mode)

    def exists(self) -> bool:
        if self.local_path is None:
            return False
        return os.path.exists(self.local_path)

    def read_bytes(self, max_bytes: Optional[int] = None) -> bytes:
        if self.local_path is None:
            raise ValueError("Tensors.local_path is not set (file not materialized)")
        with open(self.local_path, "rb") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError("tensors file too large to read into memory")
        return data


class SourceRepo(msgspec.Struct):
    """Reserved-name source descriptor for conversion/training job payloads."""

    ref: str
    checkpoint_id: Optional[str] = None
    attributes: Dict[str, Any] = msgspec.field(default_factory=dict)


class DatasetRef(msgspec.Struct):
    """Reserved-name dataset descriptor for transform-kind job payloads."""

    ref: str
    checkpoint_id: Optional[str] = None
    attributes: Dict[str, Any] = msgspec.field(default_factory=dict)
    split: str = "train"


class OutputSpec(msgspec.Struct):
    """Describes one variant a conversion endpoint will emit into the destination checkpoint."""

    attributes: Dict[str, Any] = msgspec.field(default_factory=dict)


