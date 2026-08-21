"""Video encode backend selection + streaming encoder."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from .api.errors import ValidationError
import io as _io
from fractions import Fraction

logger = logging.getLogger(__name__)

ENCODE_CONCURRENCY_ENV = "GEN_WORKER_VIDEO_ENCODE_CONCURRENCY"
DEFAULT_ENCODE_CONCURRENCY_PER_GROUP = 2

X264_OPTIONS: Dict[str, str] = {"preset": "veryfast", "crf": "18"}
NVENC_OPTIONS: Dict[str, str] = {"preset": "p4", "tune": "hq", "rc": "vbr", "cq": "19"}


@dataclass(frozen=True)
class EncoderChoice:
    codec: str
    options: Dict[str, str] = field(default_factory=dict)
    hardware: bool = False


_detect_lock = threading.Lock()
_detected: Optional[EncoderChoice] = None


def _x264() -> EncoderChoice:
    return EncoderChoice("libx264", dict(X264_OPTIONS), hardware=False)


def _probe_nvenc() -> bool:
    """One tiny REAL encode: codec presence in the PyAV build is not enough — opening h264_nvenc needs the driver's libnvidia-encode AND a card whose silicon has the encoder block (absent on H100/A100/B200) AND a tenancy whose driver grants encode sessions. The probe frame is 256x256 because NVENC enforces minimum encode dimensions (H.264 min 145x49) — a 64x64 probe fails "Frame Dimension less than the minimum supported value" on GENUINELY capable cards."""

    try:
        import av
        import numpy as np
    except ImportError:
        return False
    buf = _io.BytesIO()
    try:
        packets = 0
        with av.open(buf, mode="w", format="mp4") as container:
            stream: Any = container.add_stream("h264_nvenc", rate=8, options=dict(NVENC_OPTIONS))
            stream.width = 256
            stream.height = 256
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(
                np.zeros((256, 256, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
                packets += 1
            for packet in stream.encode():
                container.mux(packet)
                packets += 1
        return packets > 0
    except Exception as exc:
        logger.info("NVENC probe negative (%s: %s)", type(exc).__name__, exc)
        return False


def detect_encoder(*, refresh: bool = False) -> EncoderChoice:
    """Pick the video encoder for this process."""
    global _detected
    with _detect_lock:
        if _detected is not None and not refresh:
            return _detected
        if _probe_nvenc():
            _detected = EncoderChoice("h264_nvenc", dict(NVENC_OPTIONS), hardware=True)
        else:
            _detected = _x264()
        logger.info(
            "video encoder selected: %s %s", _detected.codec, _detected.options)
        return _detected


_finalize_sem: Optional[threading.BoundedSemaphore] = None
_finalize_sem_lock = threading.Lock()


def finalize_concurrency() -> int:
    """Concurrent buffered CPU encodes this PROCESS allows: two per execution group."""
    raw = os.environ.get(ENCODE_CONCURRENCY_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        from .topology import delivered_topology

        groups = max(1, int(delivered_topology().execution_groups))
    except Exception:  # noqa: BLE001
        groups = 1
    return DEFAULT_ENCODE_CONCURRENCY_PER_GROUP * groups


def _finalize_semaphore() -> threading.BoundedSemaphore:
    global _finalize_sem
    with _finalize_sem_lock:
        if _finalize_sem is None:
            _finalize_sem = threading.BoundedSemaphore(finalize_concurrency())
        return _finalize_sem


@contextmanager
def finalize_permit() -> Iterator[None]:
    """Bound concurrent buffered CPU encodes."""
    sem = _finalize_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()


def frames_to_uint8(frames: Any) -> Any:
    """Coerce PIL list / float array / torch tensor to uint8 ``[F, H, W, 3]``."""
    import numpy as np

    if isinstance(frames, (list, tuple)):
        if not frames:
            raise ValidationError("write_video: frames is empty")
        frames = np.stack(
            [np.asarray(f.convert("RGB") if hasattr(f, "convert") else f) for f in frames]
        )
    if hasattr(frames, "detach"):
        if getattr(frames, "is_cuda", False):
            from .media_transfer import cuda_tensor_to_uint8_host

            frames, _ = cuda_tensor_to_uint8_host(frames)
        else:
            frames = frames.detach().to("cpu").float().numpy()
    arr = np.asarray(frames)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr[None]
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


class StreamingVideoEncoder:
    """Incremental H.264/AAC mp4 encoder — feed frame chunks as they exist."""

    def __init__(
        self,
        path: "str | os.PathLike[str]",
        *,
        fps: float,
        audio_sample_rate: Optional[int] = None,
        encoder: Optional[EncoderChoice] = None,
    ) -> None:
        import av

        self._av = av
        self._path = str(path)
        self._fps = float(fps)
        self._sample_rate = int(audio_sample_rate) if audio_sample_rate else 0
        self._encoder = encoder or detect_encoder()
        self._container: Any = None
        self._stream: Any = None
        self._audio_stream: Any = None
        self._frames = 0
        self._closed = False
        self._zero_copy = hasattr(av.VideoFrame, "from_numpy_buffer")

    @property
    def encoder(self) -> EncoderChoice:
        return self._encoder

    @property
    def frames_encoded(self) -> int:
        return self._frames

    def _open(self, height: int, width: int) -> None:
        av = self._av
        self._container = av.open(self._path, mode="w")
        try:
            self._stream = self._add_video_stream(self._encoder, width, height)
        except Exception as exc:
            if not self._encoder.hardware:
                raise
            logger.warning(
                "hardware encoder %s failed to open (%s: %s); "
                "falling back to libx264 for this encode",
                self._encoder.codec, type(exc).__name__, exc)
            try:
                self._container.close()
            except Exception:
                logger.debug("failed hardware encoder container close", exc_info=True)
            self._container = av.open(self._path, mode="w")
            self._encoder = _x264()
            self._stream = self._add_video_stream(self._encoder, width, height)
        if self._sample_rate:
            self._audio_stream = _prepare_audio_stream(
                self._container, self._sample_rate, av)

    def _add_video_stream(self, enc: EncoderChoice, width: int, height: int) -> Any:
        stream = self._container.add_stream(
            enc.codec, rate=round(self._fps), options=dict(enc.options))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.open(strict=False)
        return stream

    def add(self, frames: Any) -> int:
        """Encode one chunk (``[F, H, W, 3]`` array / torch tensor / PIL list / single frame)."""
        if self._closed:
            raise RuntimeError("StreamingVideoEncoder is finished")
        arr = frames_to_uint8(frames)
        if self._container is None:
            height = int(arr.shape[1]) - int(arr.shape[1]) % 2
            width = int(arr.shape[2]) - int(arr.shape[2]) % 2
            if height <= 0 or width <= 0:
                raise ValidationError(
                    f"write_video: frames too small to encode ({arr.shape})")
            self._open(height, width)
        arr = arr[:, : self._stream.height, : self._stream.width]
        for frame_array in arr:
            frame = self._video_frame(frame_array)
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
            self._frames += 1
        return self._frames

    def _video_frame(self, frame_array: Any) -> Any:
        if self._zero_copy and getattr(frame_array, "flags", None) is not None \
                and frame_array.flags["C_CONTIGUOUS"]:
            try:
                return self._av.VideoFrame.from_numpy_buffer(frame_array, format="rgb24")
            except Exception:
                self._zero_copy = False
                logger.debug("from_numpy_buffer failed; using from_ndarray", exc_info=True)
        return self._av.VideoFrame.from_ndarray(frame_array, format="rgb24")

    def finish(self, audio: Any = None) -> str:
        """Flush the video stream, mux ``audio`` (waveform ``[C, samples]``) when given, close the container."""
        if self._closed:
            return self._path
        if self._container is None:
            raise ValidationError("write_video: no frames were encoded")
        if audio is not None and self._audio_stream is None:
            raise ValidationError(
                "write_video: audio given but audio_sample_rate was not "
                "declared at encoder construction")
        self._closed = True
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
            if audio is not None:
                import numpy as np

                _mux_audio(self._container, self._audio_stream, audio,
                           self._sample_rate, self._av, np)
        finally:
            self._container.close()
        return self._path

    def abort(self) -> None:
        """Close without flushing (error paths); output file is not usable."""
        self._closed = True
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                logger.debug("encoder abort close failed", exc_info=True)

    def __enter__(self) -> "StreamingVideoEncoder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self.abort()


def _prepare_audio_stream(container: Any, sample_rate: int, av: Any) -> Any:

    stream = container.add_stream("aac", rate=sample_rate)
    cc = stream.codec_context
    cc.sample_rate = sample_rate
    cc.layout = "stereo"
    cc.time_base = Fraction(1, sample_rate)
    return stream


def _mux_audio(container: Any, stream: Any, audio: Any, sample_rate: int, av: Any, np: Any) -> None:
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
    pcm = (wave.T * 32767.0).astype("int16")
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
    "ENCODE_CONCURRENCY_ENV",
    "EncoderChoice",
    "StreamingVideoEncoder",
    "detect_encoder",
    "finalize_permit",
    "frames_to_uint8",
]
