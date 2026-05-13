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
from typing import IO, TYPE_CHECKING, Any

from .api.errors import ValidationError
from .api.types import Asset

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as _PILImage


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


def read_image(asset: Asset, mode: str = "RGB") -> "_PILImage":
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
    image: "_PILImage",
    *,
    format: str = "webp",
    quality: int = 90,
) -> Asset:
    """Encode ``image`` to bytes and save via ``ctx.save_bytes(ref, ...)``.

    Replaces the undocumented ``ctx.save_image()``. ``format`` and ``quality``
    are passed to :meth:`PIL.Image.Image.save`.
    """
    buf = _io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": format.upper()}
    if format.lower() in ("webp", "jpeg", "jpg"):
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return ctx.save_bytes(ref, buf.getvalue())


__all__ = [
    "read_bytes",
    "open",
    "exists",
    "read_image",
    "read_audio",
    "write_image",
]
