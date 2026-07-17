"""gen_worker.io.write_video + VideoAsset metadata probe (gw#387).

End-to-end through the real code path: av-encode synthetic frames (+ audio)
-> ctx.save_video -> probe fills VideoAsset media metadata.
"""

from __future__ import annotations

from typing import Annotated

import msgspec
import numpy as np
import pytest

from gen_worker import ExpectedOutput, RequestContext, io as gw_io
from gen_worker.api.types import VideoAsset

av = pytest.importorskip("av")


class DurationIn(msgspec.Struct):
    duration_s: int = 10


class DurationOut(msgspec.Struct):
    video: Annotated[VideoAsset, ExpectedOutput(duration_s="input.duration_s", mime_type="video/mp4")]


def _frames(count: int = 24, height: int = 64, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((count, height, width, 3), dtype=np.float32)


def test_write_video_with_audio_probes_metadata(tmp_path) -> None:
    ctx = RequestContext(request_id="r1", local_output_dir=str(tmp_path))
    sr = 24000
    audio = np.sin(np.linspace(0, 440 * 2 * np.pi, sr))[None, :].astype(np.float32)  # 1s mono

    asset = gw_io.write_video(
        ctx, "outputs/r1/video", _frames(), fps=24, audio=audio, audio_sample_rate=sr
    )

    assert isinstance(asset, VideoAsset)
    assert asset.local_path and asset.local_path.endswith(".mp4")
    assert asset.width == 64 and asset.height == 64
    assert asset.fps == pytest.approx(24.0)
    assert asset.duration_s == pytest.approx(1.0, abs=0.25)
    assert asset.has_audio is True
    assert asset.sample_rate == sr

    # The stored container really carries an AAC audio stream (the mux is not
    # metadata-only), and the video stream survived intact.
    with av.open(asset.local_path) as container:
        assert len(container.streams.audio) == 1
        assert container.streams.audio[0].codec_context.name == "aac"
        assert len(container.streams.video) == 1


def test_write_video_without_audio(tmp_path) -> None:
    ctx = RequestContext(request_id="r2", local_output_dir=str(tmp_path))
    asset = gw_io.write_video(ctx, "outputs/r2/silent", _frames(count=9), fps=24)
    assert asset.has_audio is False
    assert asset.sample_rate is None
    assert asset.width == 64 and asset.height == 64


def test_write_video_accepts_torch_and_odd_dims(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    ctx = RequestContext(request_id="r3", local_output_dir=str(tmp_path))
    frames = torch.rand(9, 33, 65, 3)  # odd dims -> cropped to even
    asset = gw_io.write_video(ctx, "outputs/r3/odd", frames, fps=12)
    assert asset.width == 64 and asset.height == 32


def test_write_video_requires_sample_rate_with_audio(tmp_path) -> None:
    from gen_worker import ValidationError

    ctx = RequestContext(request_id="r4", local_output_dir=str(tmp_path))
    with pytest.raises(ValidationError):
        gw_io.write_video(ctx, "x", _frames(count=2), fps=24, audio=np.zeros((2, 100), dtype=np.float32))


def test_save_video_probe_is_best_effort(tmp_path) -> None:
    ctx = RequestContext(request_id="r5", local_output_dir=str(tmp_path))
    asset = ctx.save_video(b"not a real container", "outputs/r5/garbage")
    assert isinstance(asset, VideoAsset)  # save succeeds; metadata stays None
    assert asset.duration_s is None and asset.has_audio is None


def test_expected_output_duration_ref_compiles() -> None:
    from gen_worker.discovery.discover import _collect_expected_output_metadata

    items = _collect_expected_output_metadata(DurationIn, DurationOut)
    assert items == [
        {"field": "video", "type": "video", "count": 1, "duration_s": "input.duration_s", "mime_type": "video/mp4"}
    ]


def test_scan_output_assets_sums_nested_video_durations_and_counts() -> None:
    import msgspec

    from gen_worker.api.types import AudioAsset, ImageAsset
    from gen_worker.executor import _scan_output_assets

    class Out(msgspec.Struct):
        videos: list[VideoAsset]
        extras: dict

    v1 = VideoAsset(ref="a", duration_s=10.5)
    v2 = VideoAsset(ref="b", duration_s=2.0)
    unprobed = VideoAsset(ref="c")  # probe failed: duration_s None
    out = Out(videos=[v1, unprobed], extras={"more": (v2, ImageAsset(ref="i"), AudioAsset(ref="s"))})
    # 5 total Assets (v1, unprobed, v2, image, audio); only videos with a
    # probed duration_s contribute media seconds (pgw#512: output_count is
    # the ONLY per_output settlement source, replacing field-name scavenging
    # of "images"/"videos"/"audios" keys).
    assert _scan_output_assets(out) == (12.5, 5)
    assert _scan_output_assets(None) == (0.0, 0)
    assert _scan_output_assets({"images": [ImageAsset(ref="i")]}) == (0.0, 1)
