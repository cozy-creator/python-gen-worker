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
    img = Image.open(_local_path(asset))
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

            from scipy.signal import resample_poly  # type: ignore

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
      [0, 1] or uint8), or a torch tensor of the same shape.
    - ``audio``: waveform ``[channels, samples]`` (numpy or torch, float in
      [-1, 1]); mono is duplicated to stereo. ``audio_sample_rate`` is
      required when audio is given.
    """
    try:
        import av
        import numpy as np
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

    arr = _frames_to_uint8(frames, np)
    height, width = int(arr.shape[1]), int(arr.shape[2])
    # libx264/yuv420p needs even dimensions; crop a stray row/column.
    height -= height % 2
    width -= width % 2
    arr = arr[:, :height, :width]

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        tmp_path = handle.name
    try:
        container = av.open(tmp_path, mode="w")
        try:
            stream = container.add_stream("libx264", rate=round(fps))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            # Both streams must exist before the first mux writes the header.
            audio_stream = (
                _prepare_audio_stream(container, sample_rate, av)
                if audio is not None
                else None
            )
            for frame_array in arr:
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            if audio_stream is not None:
                _mux_audio(container, audio_stream, audio, sample_rate, av, np)
        finally:
            container.close()
        return ctx.save_video(tmp_path, ref, format="mp4")
    finally:
        import os

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _frames_to_uint8(frames: Any, np: Any) -> Any:
    """Coerce PIL list / float array / torch tensor to uint8 [F, H, W, C]."""
    if isinstance(frames, (list, tuple)):
        if not frames:
            raise ValidationError("write_video: frames is empty")
        frames = np.stack([np.asarray(f.convert("RGB") if hasattr(f, "convert") else f) for f in frames])
    if hasattr(frames, "detach"):  # torch tensor
        frames = frames.detach().to("cpu").float().numpy()
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValidationError(
            f"write_video: frames must be [F, H, W, 3], got shape {arr.shape}"
        )
    if arr.dtype != np.uint8:
        arr = arr.astype("float32")
        if float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr.round(), 0, 255).astype("uint8")
    return arr


def _prepare_audio_stream(container: Any, sample_rate: int, av: Any) -> Any:
    from fractions import Fraction

    stream = container.add_stream("aac", rate=sample_rate)
    cc = stream.codec_context
    cc.sample_rate = sample_rate
    cc.layout = "stereo"
    cc.time_base = Fraction(1, sample_rate)
    return stream


def _mux_audio(container: Any, stream: Any, audio: Any, sample_rate: int, av: Any, np: Any) -> None:
    """Append an AAC stereo track (mirrors diffusers ltx2 export_utils)."""
    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").float().numpy()
    wave = np.asarray(audio, dtype="float32")
    if wave.ndim == 1:
        wave = wave[None, :]
    if wave.ndim != 2:
        raise ValidationError(f"write_video: audio must be [channels, samples], got shape {wave.shape}")
    if wave.shape[0] == 1:
        wave = np.repeat(wave, 2, axis=0)
    elif wave.shape[0] > 2:
        wave = wave[:2]
    wave = np.ascontiguousarray(np.clip(wave, -1.0, 1.0))

    cc = stream.codec_context
    # One packed-s16 input frame; the resampler converts to the encoder's
    # format and assigns pts (mirrors diffusers ltx2 export_utils._write_audio).
    pcm = (wave.T * 32767.0).astype("int16")  # [samples, 2] interleaved
    frame_in = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(pcm.reshape(1, -1)), format="s16", layout="stereo"
    )
    frame_in.sample_rate = sample_rate

    resampler = av.audio.resampler.AudioResampler(
        format=cc.format or "fltp", layout=cc.layout or "stereo", rate=cc.sample_rate or sample_rate
    )
    next_pts = 0
    for rframe in resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = next_pts
        next_pts += rframe.samples
        rframe.sample_rate = sample_rate
        container.mux(stream.encode(rframe))
    for packet in stream.encode():
        container.mux(packet)


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
