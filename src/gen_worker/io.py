"""I/O codecs for endpoint payloads — read/write Assets to/from common formats."""

from __future__ import annotations

import io as _io
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, Literal, Optional

from .api.errors import ValidationError
from .api.types import Asset
from .output_integrity import (
    StreamCollector, enforce, guard_frames, guard_image, judged,
)
from .stage_timing import stage_of as _stage
import os
import tempfile

if TYPE_CHECKING:
    import numpy as np


def _local_path(asset: Asset) -> Path:
    p = asset.local_path
    if p is None:
        raise ValidationError(
            f"Asset {getattr(asset, 'ref', '?')!r} has no local_path; "
            "the platform did not materialize it."
        )
    return Path(p)


def read_bytes(asset: Asset) -> bytes:
    """Return the raw bytes of an Asset's materialized file."""
    return _local_path(asset).read_bytes()


def open(asset: Asset, mode: str = "rb") -> IO[Any]:
    """Open the Asset's materialized file."""
    return _local_path(asset).open(mode)


def exists(asset: Asset) -> bool:
    """Return True if the Asset's ``local_path`` is set and points to a real file."""
    p = asset.local_path
    return p is not None and Path(p).exists()


def read_image(asset: Asset, mode: str = "RGB") -> Any:
    """Decode an Asset as a PIL image."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "gen_worker.io.read_image requires Pillow. "
            "Install with `pip install gen-worker[images]`."
        ) from e
    img: "Image.Image" = Image.open(_local_path(asset))
    if mode and img.mode != mode:
        img = img.convert(mode)
    return img


def read_audio(
    asset: Asset,
    target_sample_rate: int | None = None,
    mono: bool = True,
) -> tuple["np.ndarray", int]:
    """Decode an Asset as a float32 numpy array + sample rate."""
    try:
        import numpy  # noqa: F401
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "gen_worker.io.read_audio requires soundfile + numpy. "
            "Install with `pip install gen-worker[audio]`."
        ) from e
    data, sr = sf.read(_local_path(asset), always_2d=False, dtype="float32")
    if mono and data.ndim == 2:
        data = data.mean(axis=1)
    if target_sample_rate is not None and sr != target_sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr), int(target_sample_rate))
        data = resample_poly(
            data, int(target_sample_rate) // g, int(sr) // g
        ).astype("float32")
        sr = target_sample_rate
    return data, sr


ImageFormat = Literal["webp", "png", "jpg", "jpeg"]

DEFAULT_IMAGE_FORMAT: Final[ImageFormat] = "webp"
DEFAULT_IMAGE_QUALITY = 95

IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "webp": ("WEBP", ".webp"),
    "png": ("PNG", ".png"),
    "jpg": ("JPEG", ".jpg"),
    "jpeg": ("JPEG", ".jpg"),
}


def image_format(format: ImageFormat = DEFAULT_IMAGE_FORMAT) -> tuple[str, str]:
    """``(PIL format, file extension)`` for a tenant-facing format name."""
    fmt = str(format or DEFAULT_IMAGE_FORMAT).strip().lower()
    try:
        return IMAGE_FORMATS[fmt]
    except KeyError:
        raise ValidationError(
            f"unsupported image format {format!r}; expected one of "
            f"{', '.join(sorted(IMAGE_FORMATS))}"
        ) from None


def encode_image(
    image: Any,
    *,
    format: ImageFormat = DEFAULT_IMAGE_FORMAT,
    quality: int = DEFAULT_IMAGE_QUALITY,
    lossless: bool = False,
    method: int = 4,
    **encode_kwargs: Any,
) -> tuple[bytes, str]:
    """THE image encode core — returns ``(payload_bytes, file_extension)``."""
    pil_format, ext = image_format(format)

    img = image
    if pil_format == "JPEG" and getattr(img, "mode", "") in {"RGBA", "LA", "P"}:
        img = img.convert("RGB")

    save_kwargs: dict[str, Any] = {}
    if pil_format in {"JPEG", "WEBP"}:
        save_kwargs["quality"] = max(1, min(int(quality), 100))
    if pil_format == "WEBP":
        save_kwargs["lossless"] = bool(lossless)
        save_kwargs["method"] = int(method)
    save_kwargs.update(encode_kwargs)

    buf = _io.BytesIO()
    img.save(buf, format=pil_format, **save_kwargs)
    return buf.getvalue(), ext


def write_image(
    ctx: Any,
    ref: str,
    image: Any,
    *,
    format: ImageFormat = DEFAULT_IMAGE_FORMAT,
    quality: int = DEFAULT_IMAGE_QUALITY,
    as_type: Optional[type] = None,
    **encode_kwargs: Any,
) -> Asset:
    """Encode ``image`` via :func:`encode_image` and save via ``ctx.save_bytes``."""
    from .video_encode import finalize_permit

    with finalize_permit():
        _release_gpu_slot_for_finalize(ctx)
        if judged(ctx):
            with _stage(ctx, "output_integrity"):
                guard_image(image, ref=ref)
        with _stage(ctx, "image_encode"):
            payload, ext = encode_image(
                image, format=format, quality=quality, **encode_kwargs
            )
    if Path(ref).suffix == "":
        ref = f"{ref}{ext}"
    out = ctx.save_bytes(ref, payload)
    if as_type is not None and type(out) is not as_type:
        import msgspec
        return as_type(**msgspec.to_builtins(out))
    return out


def probe_video(source: "bytes | str | Path") -> dict[str, Any]:
    """Probe a video container (PyAV) for media metadata."""
    try:
        import av
    except ImportError:
        return {}
    out: dict[str, Any] = {}
    try:
        opened = av.open(_io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source))
    except Exception:
        return {}
    try:
        video = next(iter(opened.streams.video), None)
        if video is not None:
            if video.width:
                out["width"] = int(video.width)
            if video.height:
                out["height"] = int(video.height)
            rate = video.average_rate or video.base_rate
            if rate:
                out["fps"] = float(rate)
            container_duration = getattr(opened, "duration", None)
            if video.duration is not None and video.time_base is not None:
                out["duration_s"] = float(video.duration * video.time_base)
            elif container_duration is not None:
                out["duration_s"] = float(container_duration / av.time_base)
        audio = next(iter(opened.streams.audio), None)
        out["has_audio"] = audio is not None
        if audio is not None and audio.sample_rate:
            out["sample_rate"] = int(audio.sample_rate)
    except Exception:
        pass
    finally:
        opened.close()
    return out


def write_video(
    ctx: Any,
    ref: str,
    frames: Any,
    *,
    fps: float,
    audio: Any = None,
    audio_sample_rate: Optional[int] = None,
) -> Asset:
    """Encode frames (+ optional audio) to an H.264/AAC mp4 and save via ctx.save_video(ref=...). Requires PyAV + numpy (gen-worker[video]). Frames may be PIL list, [F,H,W,C] numpy/torch, or an iterator of chunks (encoded as produced, never rebuffered). For array input the request's GPU slot is TERMINALLY released once frames are on host so the encode+upload overlaps the next request's denoise — do NOT run more GPU work on ctx after this call; for iterator input the release happens when the iterator is exhausted."""
    try:
        import numpy as np  # noqa: F401  (hard dep of the encode path)

        from .video_encode import StreamingVideoEncoder, finalize_permit, frames_to_uint8
    except ImportError as e:
        raise ImportError(
            "gen_worker.io.write_video requires PyAV + numpy. "
            "Install with `pip install gen-worker[video]`."
        ) from e
    sample_rate = 0
    if audio is not None:
        if not audio_sample_rate:
            raise ValidationError("write_video: audio_sample_rate is required with audio")
        sample_rate = int(audio_sample_rate)

    streaming = _is_chunk_iterator(frames)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        tmp_path = handle.name
    try:
        with _stage(ctx, "video_encode"), StreamingVideoEncoder(
            tmp_path, fps=fps, audio_sample_rate=sample_rate or None
        ) as encoder:
            if streaming:
                from .media_transfer import staged_uint8_chunks

                integrity = StreamCollector()
                for chunk in staged_uint8_chunks(frames):
                    integrity.observe(chunk)
                    encoder.add(chunk)
                _release_gpu_slot_for_finalize(ctx)
                if judged(ctx):
                    with _stage(ctx, "output_integrity"):
                        enforce(integrity.verdict(), ref=ref, kind="video")
                encoder.finish(audio)
            else:
                arr = frames_to_uint8(frames)
                with finalize_permit():
                    _release_gpu_slot_for_finalize(ctx)
                    if judged(ctx):
                        with _stage(ctx, "output_integrity"):
                            guard_frames(arr, ref=ref)
                    encoder.add(arr)
                    del arr
                    encoder.finish(audio)
        return ctx.save_video(tmp_path, ref, format="mp4")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _is_chunk_iterator(frames: Any) -> bool:
    if isinstance(frames, (list, tuple)) or hasattr(frames, "__array__") or hasattr(frames, "detach"):
        return False
    return hasattr(frames, "__next__") or hasattr(frames, "__iter__")


def _release_gpu_slot_for_finalize(ctx: Any) -> None:
    release = getattr(ctx, "_release_gpu_slot_for_finalize", None)
    if callable(release):
        release()


__all__ = [
    "read_bytes",
    "open",
    "exists",
    "read_image",
    "read_audio",
    "encode_image",
    "image_format",
    "write_image",
    "DEFAULT_IMAGE_FORMAT",
    "DEFAULT_IMAGE_QUALITY",
    "IMAGE_FORMATS",
    "write_video",
    "probe_video",
]
