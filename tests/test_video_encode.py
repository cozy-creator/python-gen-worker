"""gw#476 video encode path — encoder selection, streaming encode, fast
presets, and the terminal GPU-slot release at the decode->finalize handoff.

Integration-style through the real code path: real PyAV encodes (CPU x264 —
the CI fallback rung), real container probes, a real RequestContext. NVENC
itself is proven on GPU pods (evidence run), not here.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from gen_worker import RequestContext, ValidationError, io as gw_io
from gen_worker import video_encode as ve

av = pytest.importorskip("av")


@pytest.fixture(autouse=True)
def _fresh_encoder_state(monkeypatch):
    """Each test starts with a cold detection cache + default concurrency."""
    ve._detected = None
    ve._finalize_sem = None
    yield
    ve._detected = None
    ve._finalize_sem = None


def _frames(count: int = 24, height: int = 64, width: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((count, height, width, 3), dtype=np.float32)


def _decode_frame_count(path: str) -> int:
    with av.open(path) as container:
        return sum(1 for _ in container.decode(video=0))


# ---- encoder selection -------------------------------------------------------


def test_env_x264_skips_probe_and_uses_fast_preset(monkeypatch) -> None:
    monkeypatch.setenv(ve.ENCODER_ENV, "x264")
    choice = ve.detect_encoder(refresh=True)
    assert choice.codec == "libx264"
    assert choice.hardware is False
    assert choice.options["preset"] == "veryfast"
    assert "crf" in choice.options


def test_detection_is_cached(monkeypatch) -> None:
    monkeypatch.setenv(ve.ENCODER_ENV, "x264")
    first = ve.detect_encoder(refresh=True)
    calls = []
    monkeypatch.setattr(ve, "_probe_nvenc", lambda: calls.append(1) or False)
    assert ve.detect_encoder() is first
    assert calls == []  # cached — no re-probe


def test_forced_nvenc_falls_back_when_probe_fails(monkeypatch) -> None:
    monkeypatch.setenv(ve.ENCODER_ENV, "nvenc")
    monkeypatch.setattr(ve, "_probe_nvenc", lambda: False)
    choice = ve.detect_encoder(refresh=True)
    assert choice.codec == "libx264"  # never refuse to serve


def test_auto_picks_nvenc_when_probe_succeeds(monkeypatch) -> None:
    monkeypatch.setenv(ve.ENCODER_ENV, "auto")
    monkeypatch.setattr(ve, "_probe_nvenc", lambda: True)
    choice = ve.detect_encoder(refresh=True)
    assert choice.codec == "h264_nvenc"
    assert choice.hardware is True
    assert choice.options == ve.NVENC_OPTIONS


def test_hardware_open_failure_falls_back_per_encode(tmp_path, monkeypatch) -> None:
    """A positive boot probe + a failing open at encode time (NVENC session
    limits) must fall back to x264 for THAT encode, not fail the request."""
    fake_hw = ve.EncoderChoice("h264_nvenc_bogus_codec", {}, hardware=True)
    out = tmp_path / "fallback.mp4"
    enc = ve.StreamingVideoEncoder(out, fps=24, encoder=fake_hw)
    enc.add(_frames(count=4))
    enc.finish()
    assert enc.encoder.codec == "libx264"
    assert _decode_frame_count(str(out)) == 4


# ---- streaming encoder -------------------------------------------------------


def test_streaming_chunks_equal_buffered_output(tmp_path) -> None:
    frames = _frames(count=25)
    buffered = tmp_path / "buffered.mp4"
    chunked = tmp_path / "chunked.mp4"

    enc = ve.StreamingVideoEncoder(buffered, fps=24)
    enc.add(frames)
    enc.finish()

    enc = ve.StreamingVideoEncoder(chunked, fps=24)
    for i in range(0, 25, 8):  # uneven tail chunk on purpose
        enc.add(frames[i : i + 8])
    assert enc.frames_encoded == 25
    enc.finish()

    assert _decode_frame_count(str(buffered)) == 25
    assert _decode_frame_count(str(chunked)) == 25
    # Same encoder, same input, same chunk-invariant output size ballpark.
    assert abs(buffered.stat().st_size - chunked.stat().st_size) < max(
        2048, buffered.stat().st_size // 4
    )


def test_streaming_encoder_accepts_single_frames_and_odd_dims(tmp_path) -> None:
    out = tmp_path / "odd.mp4"
    enc = ve.StreamingVideoEncoder(out, fps=12)
    for i in range(5):
        enc.add(_frames(count=1, height=33, width=65, seed=i)[0])  # [H, W, 3]
    enc.finish()
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (64, 32)


def test_finish_without_frames_raises(tmp_path) -> None:
    enc = ve.StreamingVideoEncoder(tmp_path / "empty.mp4", fps=24)
    with pytest.raises(ValidationError):
        enc.finish()


def test_audio_requires_declared_sample_rate(tmp_path) -> None:
    enc = ve.StreamingVideoEncoder(tmp_path / "a.mp4", fps=24)
    enc.add(_frames(count=2))
    with pytest.raises(ValidationError):
        enc.finish(audio=np.zeros((1, 100), dtype=np.float32))


# ---- write_video: streaming input + slot handoff -----------------------------


class _FakeLease:
    def __init__(self) -> None:
        self.releases = 0

    def yield_slot(self) -> bool:
        self.releases += 1
        return self.releases == 1  # once-only transition, like _GpuSlotLease

    def reacquire(self) -> None:  # pragma: no cover - must NOT be called
        raise AssertionError("terminal finalize release must never reacquire")


def _ctx_with_lease(tmp_path, request_id: str = "r-enc") -> tuple[RequestContext, _FakeLease]:
    ctx = RequestContext(request_id=request_id, local_output_dir=str(tmp_path))
    lease = _FakeLease()
    ctx._gpu_slot_lease = lease
    return ctx, lease


def test_write_video_releases_slot_before_encode(tmp_path, monkeypatch) -> None:
    ctx, lease = _ctx_with_lease(tmp_path)
    released_at_encode: list[int] = []

    real_add = ve.StreamingVideoEncoder.add

    def spying_add(self, frames):
        released_at_encode.append(lease.releases)
        return real_add(self, frames)

    monkeypatch.setattr(ve.StreamingVideoEncoder, "add", spying_add)
    asset = gw_io.write_video(ctx, "outputs/r-enc/v", _frames(count=8), fps=24)
    assert asset.local_path and asset.local_path.endswith(".mp4")
    # Slot was terminally released BEFORE the first frame hit the encoder,
    # exactly once, and never reacquired.
    assert lease.releases >= 1
    assert released_at_encode and released_at_encode[0] == 1


def test_write_video_iterator_streams_and_releases_after_exhaustion(tmp_path) -> None:
    ctx, lease = _ctx_with_lease(tmp_path)
    seen_at_chunk: list[int] = []

    def chunks():
        for i in range(3):
            seen_at_chunk.append(lease.releases)
            yield _frames(count=5, seed=i)

    asset = gw_io.write_video(ctx, "outputs/r-enc/stream", chunks(), fps=24)
    assert _decode_frame_count(asset.local_path) == 15
    # While the producer was still decoding, the slot was HELD (releases=0);
    # the terminal release happened only after exhaustion.
    assert seen_at_chunk == [0, 0, 0]
    assert lease.releases >= 1


def test_write_video_failure_fails_its_own_request(tmp_path) -> None:
    ctx, _lease = _ctx_with_lease(tmp_path)

    def exploding_chunks():
        yield _frames(count=2)
        raise RuntimeError("VAE decode died mid-stream")

    with pytest.raises(RuntimeError, match="VAE decode died"):
        gw_io.write_video(ctx, "outputs/r-enc/boom", exploding_chunks(), fps=24)


def test_write_video_without_lease_is_a_noop_release(tmp_path) -> None:
    ctx = RequestContext(request_id="r-plain", local_output_dir=str(tmp_path))
    asset = gw_io.write_video(ctx, "outputs/r-plain/v", _frames(count=4), fps=24)
    assert asset.width == 64


def test_write_video_audio_survives_the_new_path(tmp_path) -> None:
    ctx = RequestContext(request_id="r-audio", local_output_dir=str(tmp_path))
    sr = 24000
    audio = np.sin(np.linspace(0, 440 * 2 * np.pi, sr))[None, :].astype(np.float32)
    asset = gw_io.write_video(
        ctx, "outputs/r-audio/v", _frames(), fps=24, audio=audio, audio_sample_rate=sr
    )
    assert asset.has_audio is True and asset.sample_rate == sr
    with av.open(asset.local_path) as container:
        assert container.streams.audio[0].codec_context.name == "aac"


# ---- bounded finalize concurrency (gw#516) -----------------------------------


def test_finalize_permit_bounds_concurrency(monkeypatch) -> None:
    monkeypatch.setenv(ve.ENCODE_CONCURRENCY_ENV, "1")
    ve._finalize_sem = None
    active = 0
    peak = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal active, peak
        with ve.finalize_permit():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_x264_stream_uses_fast_preset_options(tmp_path) -> None:
    """The stream really carries the veryfast/crf options (not silently
    dropped by PyAV): an unknown option would raise at add_stream."""
    out = tmp_path / "preset.mp4"
    enc = ve.StreamingVideoEncoder(
        out, fps=24, encoder=ve.EncoderChoice("libx264", dict(ve.X264_OPTIONS)))
    enc.add(_frames(count=4))
    enc.finish()
    assert _decode_frame_count(str(out)) == 4
