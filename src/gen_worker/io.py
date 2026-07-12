"""I/O codecs for endpoint payloads — read/write Assets to/from common formats.

Free functions over methods so :class:`Asset` stays a small data struct.
Optional codec deps (Pillow, soundfile) are lazy-imported so the core wheel
doesn't drag them in.

Usage::

    from gen_worker import io as gw_io

    speech, sr = gw_io.read_audio(payload.audio, target_sample_rate=16000)
    img = gw_io.read_image(payload.image)
    out = gw_io.write_image(ctx, "out", img, format="webp", quality=90)
"""

from __future__ import annotations

import io as _io
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Optional

from .api.errors import ValidationError
from .api.types import Asset

if TYPE_CHECKING:
    import numpy as np


def _local_path(asset: Asset) -> Path:
    """Coerce ``asset.local_path`` to :class:`Path`; raise :class:`ValidationError` if missing."""
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
    """Open the Asset's materialized file. Defaults to binary read mode."""
    return _local_path(asset).open(mode)


def exists(asset: Asset) -> bool:
    """Return True if the Asset's ``local_path`` is set and points to a real file."""
    p = asset.local_path
    return p is not None and Path(p).exists()


def read_image(asset: Asset, mode: str = "RGB") -> Any:
    """Decode an Asset as a PIL image.

    Requires Pillow (``pip install gen-worker[images]``). If ``mode`` is set
    and differs from the source image, the image is converted via
    ``PIL.Image.convert(mode)``.
    """
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
    """Decode an Asset as a float32 numpy array + sample rate.

    Requires soundfile + numpy (``pip install gen-worker[audio]``).

    - If ``mono`` is True (default), multi-channel audio is mixed down to mono.
    - If ``target_sample_rate`` is set and differs from the source rate, the
      signal is resampled in-process. Uses :func:`scipy.signal.resample_poly`
      when scipy is available, otherwise falls back to a pure-numpy linear
      resample (acceptable for typical speech-input pipelines).
    """
    try:
        import numpy as np
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
        try:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), int(target_sample_rate))
            data = resample_poly(
                data, int(target_sample_rate) // g, int(sr) // g
            ).astype("float32")
        except ImportError:
            # Pure-numpy linear resample fallback.
            duration = data.shape[0] / sr
            new_n = int(round(duration * target_sample_rate))
            old_t = np.linspace(0.0, duration, num=data.shape[0], endpoint=False)
            new_t = np.linspace(0.0, duration, num=new_n, endpoint=False)
            data = np.interp(new_t, old_t, data).astype("float32")
        sr = target_sample_rate
    return data, sr


def write_image(
    ctx: Any,
    ref: str,
    image: Any,
    *,
    format: str = "webp",
    quality: int = 90,
    as_type: Optional[type] = None,
    **encode_kwargs: Any,
) -> Asset:
    """Encode ``image`` to bytes and save via ``ctx.save_bytes(ref, ...)``.

    Replaces the undocumented ``ctx.save_image()``. ``format`` and ``quality``
    are passed to :meth:`PIL.Image.Image.save`; any extra ``encode_kwargs`` are
    forwarded too (e.g. ``method=6`` for higher-effort WebP).

    ``as_type`` optionally re-wraps the returned ``Asset`` as a subclass such as
    :class:`~gen_worker.api.types.ImageAsset`, so endpoints whose output struct
    is typed ``ImageAsset`` don't have to round-trip through
    ``msgspec.to_builtins``.
    """
    buf = _io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": format.upper()}
    if format.lower() in ("webp", "jpeg", "jpg"):
        save_kwargs["quality"] = quality
    save_kwargs.update(encode_kwargs)
    image.save(buf, **save_kwargs)
    out = ctx.save_bytes(ref, buf.getvalue())
    if as_type is not None and type(out) is not as_type:
        import msgspec
        return as_type(**msgspec.to_builtins(out))
    return out


def probe_video(source: "bytes | str | Path") -> dict[str, Any]:
    """Probe a video container (PyAV) for media metadata.

    Returns a dict with any of ``duration_s``, ``fps``, ``width``, ``height``,
    ``has_audio``, ``sample_rate`` — empty when PyAV is missing or the probe
    fails. Used by ``ctx.save_video`` to fill :class:`VideoAsset` metadata.
    """
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
    """Encode ``frames`` (+ optional ``audio``) to an H.264/AAC mp4 and save
    via ``ctx.save_video(ref=...)``.

    Requires PyAV + numpy (``pip install gen-worker[video]``). Mirrors
    diffusers' ltx2 ``export_utils.encode_video`` so video endpoints stop
    hand-rolling tempfile + ``export_to_video`` and audio survives the mux.

    - ``frames``: list of PIL images, a numpy array ``[F, H, W, C]`` (float in
      [0, 1] or uint8), a torch tensor of the same shape, OR an iterator/
      generator of such chunks — chunks are encoded as they are produced
      (VAE framewise decode seam) and the full clip is never rebuffered.
    - ``audio``: waveform ``[channels, samples]`` (numpy or torch, float in
      [-1, 1]); mono is duplicated to stereo. ``audio_sample_rate`` is
      required when audio is given.

    Encoder selection + GPU-slot handoff (gw#476 / gw#516): the backend is
    NVENC when the card has the encoder block (probed once per process),
    else libx264 at a fast preset. For array input the request's GPU slot is
    terminally released once the frames are on the host — the CPU encode and
    the upload overlap the next request's denoise instead of idling the GPU
    (measured up to 179s of idle on a CPU-contended host). Do not run more
    GPU work on ``ctx`` after this call. For iterator input the release
    happens when the iterator is exhausted (the producer is still decoding
    on the GPU while chunks stream into the encoder).
    """
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

    import os
    import tempfile

    streaming = _is_chunk_iterator(frames)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        tmp_path = handle.name
    try:
        with StreamingVideoEncoder(
            tmp_path, fps=fps, audio_sample_rate=sample_rate or None
        ) as encoder:
            if streaming:
                # Chunks arrive while the producer still owns the GPU; the
                # encode interleaves (NVENC costs zero SMs). Release the slot
                # only once decode is done, before the flush + upload tail.
                for chunk in frames:
                    encoder.add(chunk)
                _release_gpu_slot_for_finalize(ctx)
                encoder.finish(audio)
            else:
                arr = frames_to_uint8(frames)
                # Bounded CPU-finalize admission BEFORE the slot release:
                # back-pressure holds the GPU slot rather than stacking raw
                # frame buffers in host RAM (gw#516).
                with finalize_permit():
                    _release_gpu_slot_for_finalize(ctx)
                    encoder.add(arr)
                    del arr
                    encoder.finish(audio)
        return ctx.save_video(tmp_path, ref, format="mp4")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _is_chunk_iterator(frames: Any) -> bool:
    """True for generators/iterators of frame chunks (streaming input).

    Lists/tuples (PIL frames), numpy arrays, and torch tensors are buffered
    input; everything else with ``__next__`` streams.
    """
    if isinstance(frames, (list, tuple)) or hasattr(frames, "__array__") or hasattr(frames, "detach"):
        return False
    return hasattr(frames, "__next__") or hasattr(frames, "__iter__")


def _release_gpu_slot_for_finalize(ctx: Any) -> None:
    """Terminal GPU-slot release at the decode->finalize handoff (gw#516)."""
    release = getattr(ctx, "_release_gpu_slot_for_finalize", None)
    if callable(release):
        release()


__all__ = [
    "read_bytes",
    "open",
    "exists",
    "read_image",
    "read_audio",
    "write_image",
    "write_video",
    "probe_video",
]
