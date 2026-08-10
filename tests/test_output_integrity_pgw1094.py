"""pgw#1094: the serve-path output-integrity floor.

REAL fixtures, synthesized here so the suite carries no media: a clean render,
VAE-decoded-style noise, a black clip, a cut-heavy clip, a MELTED clip, a
non-finite clip. The candidate the floor exists for (ie#634's noise) must FAIL
and must never reach ``ctx.save_video``; the melted clip must PASS, and the
assertion is that it scores HIGHER than the clean one — that inversion is this
gate's scope boundary and it is pinned here so nobody re-reads a green
integrity line as a quality verdict.
"""

from __future__ import annotations

import numpy as np
import pytest

from gen_worker import io as gw_io
from gen_worker import output_integrity as oi
from gen_worker.api.errors import OutputIntegrityError

# ---------------------------------------------------------------------------
# fixtures — real pixel arrays, the (F, H, W, 3) uint8 the encode path holds
# ---------------------------------------------------------------------------

H, W = 256, 448


def _scene(h: int = H, w: int = W) -> np.ndarray:
    """One textured still: gradients + a checker + a bright disc."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (yy / h) * 120.0 + (xx / w) * 90.0
    checker = ((yy // 16 + xx // 16) % 2) * 40.0
    disc = ((yy - h * 0.4) ** 2 + (xx - w * 0.45) ** 2) < (min(h, w) * 0.18) ** 2
    img = base + checker + disc * 70.0
    rgb = np.stack([img, img * 0.85 + 20.0, img * 0.6 + 40.0], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def clean_clip(frames: int = 24) -> np.ndarray:
    """A real render's shape: one scene panning, plus the per-frame
    high-frequency variation every real render carries (grain, sampler jitter,
    VAE ringing). Adjacent frames are the self-similar-but-not-identical pair
    the floor is calibrated on — and the grain is what the melt class removes.
    """
    still = _scene(H, W + frames).astype(np.float32)
    rng = np.random.default_rng(4242)
    clip = np.stack([still[:, t:t + W] for t in range(frames)])
    clip = clip + rng.normal(0.0, 13.0, size=clip.shape).astype(np.float32)
    return np.clip(clip, 0, 255).astype(np.uint8)


def noise_clip(frames: int = 24) -> np.ndarray:
    """ie#634's candidate: every frame independently sampled, so consecutive
    frames are unrelated. This is what VAE-decoded noise looks like."""
    rng = np.random.default_rng(1094)
    return rng.integers(0, 256, size=(frames, H, W, 3), dtype=np.uint8)


def black_clip(frames: int = 24) -> np.ndarray:
    return np.zeros((frames, H, W, 3), dtype=np.uint8)


def melted_clip(frames: int = 24) -> np.ndarray:
    """The fp8-melt class: the clean clip with its high-frequency detail
    smeared away. Structurally intact, visibly wrong, and INVISIBLE here."""
    clip = clean_clip(frames).astype(np.float32)
    for _ in range(6):  # separable box blur, repeated -> heavy smear
        clip = (clip + np.roll(clip, 1, axis=1) + np.roll(clip, -1, axis=1)) / 3.0
        clip = (clip + np.roll(clip, 1, axis=2) + np.roll(clip, -1, axis=2)) / 3.0
    return np.clip(clip, 0, 255).astype(np.uint8)


def cut_clip(frames: int = 24) -> np.ndarray:
    """Two unrelated real scenes spliced: one adjacent pair correlates at ~0
    and the MEDIAN is what keeps the clip servable."""
    a = clean_clip(frames // 2)
    b = np.stack([_scene(H, W)[::-1, ::-1]] * (frames // 2))
    b = (b.astype(np.float32) * 0.7 + 30.0).astype(np.uint8)
    b = np.stack([np.roll(f, i * 3, axis=1) for i, f in enumerate(b)])
    return np.concatenate([a, b])


# ---------------------------------------------------------------------------
# the floor itself
# ---------------------------------------------------------------------------


def test_clean_clip_passes():
    r = oi.check_frames(clean_clip())
    assert r.verdict == oi.PASS, r.summary()
    assert r.adjacent_frame_corr > oi.NOISE_CORR_FLOOR
    assert oi.SCOPE_NOTE in r.detail  # a PASS always carries its scope limit


def test_noise_clip_is_rejected_as_noise():
    r = oi.check_frames(noise_clip())
    assert r.verdict == oi.NOISE, r.summary()
    assert r.rejected and not r.ok
    # ie#634 measured 0.29 on the production clip; independent sampling here is
    # even lower. Either way it is nowhere near the 0.6 floor.
    assert r.adjacent_frame_corr < 0.3, r.summary()


def test_black_clip_is_rejected_as_blank_not_noise():
    r = oi.check_frames(black_clip())
    assert r.verdict == oi.BLANK, r.summary()
    assert r.frame_std_min < oi.BLANK_STD_FLOOR


def test_nonfinite_pixels_are_their_own_verdict():
    """A NaN latent decodes to NaN pixels — the save-path-observable form of
    the pre-decode NaN check. Seen on the SAMPLED, decimated planes, which is
    the whole clip's worth of NaN a real overflow produces, not one stray
    pixel a strided sample could miss."""
    clip = clean_clip().astype(np.float32) / 255.0
    clip[0] = np.nan
    r = oi.check_frames(clip)
    assert r.verdict == oi.NONFINITE, r.summary()


def test_a_hard_cut_still_serves():
    """The MEDIAN over spread pairs is what makes this safe — a cut drives one
    pair to ~0 while the rest stay high."""
    r = oi.check_frames(cut_clip())
    assert r.verdict == oi.PASS, r.summary()
    assert min(r.corr_series) < 0.6, "the fixture must actually contain a cut"


def test_melt_blindness_is_the_scope_boundary():
    """THE PIN. A melted render PASSES and scores HIGHER than a clean one.

    Smearing REMOVES high-frequency temporal variation, so adjacent-frame
    correlation goes UP. This gate is a noise/blank floor and can never be
    quoted as a quality verdict; fine-detail damage belongs to cozy-eval's
    detail detectors and the VLM rubric.
    """
    clean = oi.check_frames(clean_clip())
    melted = oi.check_frames(melted_clip())
    assert melted.verdict == oi.PASS, melted.summary()
    assert melted.adjacent_frame_corr > clean.adjacent_frame_corr, (
        f"melt inversion is the documented blind spot: "
        f"melted {melted.adjacent_frame_corr:.4f} vs clean "
        f"{clean.adjacent_frame_corr:.4f}"
    )


def test_single_frame_is_judged_on_the_blank_half_alone():
    flat = oi.check_frames(np.zeros((1, H, W, 3), np.uint8))
    assert flat.verdict == oi.BLANK
    good = oi.check_frames(_scene()[None])
    assert good.verdict == oi.PASS
    assert np.isnan(good.adjacent_frame_corr)


def test_image_floor_rejects_a_flat_render_and_passes_a_real_one():
    from PIL import Image

    assert oi.check_image(Image.new("RGB", (512, 512), (0, 0, 0))).verdict == oi.BLANK
    assert oi.check_image(Image.fromarray(_scene())).verdict == oi.PASS


def test_unmeasurable_output_is_never_a_pass():
    r = oi.check_frames(np.zeros((4, 8, 8, 7), np.uint8))  # not RGB
    assert r.verdict == oi.UNMEASURED
    assert not r.ok and not r.rejected  # confesses, does not refuse


def test_the_verdict_is_decimation_invariant():
    """The ~96-row decimation is what makes this affordable on the serve path,
    and it does not change the ANSWER: the statistic is a coarse whole-frame
    correlation, so it lives far from the resolution it is measured at. (That
    is also exactly why it can never see fine-detail damage.)"""
    for fixture, want in ((clean_clip(), oi.PASS), (noise_clip(), oi.NOISE),
                          (black_clip(), oi.BLANK)):
        full_h = fixture.shape[1]
        assert oi.check_frames(fixture).verdict == want
        assert oi.check_frames(fixture, target_h=full_h).verdict == want
    clip = clean_clip()
    small = oi.check_frames(clip).adjacent_frame_corr
    full = oi.check_frames(clip, target_h=clip.shape[1]).adjacent_frame_corr
    assert abs(small - full) < 0.05, (small, full)


# ---------------------------------------------------------------------------
# cost — the serve path pays this on EVERY render
# ---------------------------------------------------------------------------


def test_cost_on_a_full_size_clip_is_single_digit_ms(capsys):
    """121 frames at 1344x768 uint8 — 0.37 GB of pixels, the ie#634 shape."""
    rng = np.random.default_rng(7)
    still = rng.integers(0, 256, size=(768, 1344 + 121, 3), dtype=np.uint8)
    clip = np.stack([still[:, t:t + 1344] for t in range(121)])
    oi.check_frames(clip)  # warm numpy
    best = min(oi.check_frames(clip).seconds for _ in range(5))
    with capsys.disabled():
        print(f"\npgw#1094 integrity floor: {best * 1000:.2f} ms on "
              f"{clip.shape} uint8 ({clip.nbytes / 1e9:.2f} GB)")
    # Naive full-resolution on this clip is 555 ms. The bound is deliberately
    # loose for CI hardware; the printed number is the measurement.
    assert best < 0.060, f"{best * 1000:.1f} ms"


# ---------------------------------------------------------------------------
# the streaming collector (gw#476 chunk seam)
# ---------------------------------------------------------------------------


def _stream(clip: np.ndarray, chunk: int = 6) -> oi.OutputIntegrity:
    c = oi.StreamCollector()
    for i in range(0, len(clip), chunk):
        c.observe(clip[i:i + chunk])
    return c.verdict()


def test_streaming_collector_rejects_noise_and_passes_a_render():
    assert _stream(clean_clip(48)).verdict == oi.PASS
    assert _stream(noise_clip(48)).verdict == oi.NOISE
    assert _stream(black_clip(48)).verdict == oi.BLANK


def test_single_chunk_stream_equals_the_buffered_answer():
    clip = clean_clip(24)
    assert _stream(clip, chunk=len(clip)).adjacent_frame_corr == pytest.approx(
        oi.check_frames(clip).adjacent_frame_corr)


def test_streaming_cost_stays_bounded_on_a_long_clip():
    """The clip length is unknown until the producer is done, so the collector
    thins: kept pairs halve and the chunk stride doubles at the budget."""
    clip = clean_clip(64)
    c = oi.StreamCollector()
    for i in range(0, 4000, 4):  # 1000 chunks of the same 4 frames
        c.observe(clip[i % 60:i % 60 + 4])
    r = c.verdict()
    assert r.verdict == oi.PASS
    assert r.frames_sampled < 12 * oi.STREAM_PAIR_BUDGET, r.frames_sampled
    assert r.seconds < 0.5, r.seconds


# ---------------------------------------------------------------------------
# the save path — nothing garbage reaches the upload
# ---------------------------------------------------------------------------


class _Ctx:
    """The save surface ``io.write_video`` / ``io.write_image`` touch."""

    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_video(self, path, ref, format="mp4"):
        self.saved.append(ref)
        from gen_worker.api.types import VideoAsset
        return VideoAsset(ref=ref, owner="t")

    def save_bytes(self, ref, data):
        self.saved.append(ref)
        from gen_worker.api.types import Asset
        return Asset(ref=ref, owner="t", size_bytes=len(data))


def test_write_video_refuses_noise_before_it_can_be_uploaded():
    ctx = _Ctx()
    with pytest.raises(OutputIntegrityError) as exc:
        gw_io.write_video(ctx, "outputs/r/video.mp4", noise_clip(), fps=24)
    assert exc.value.verdict == oi.NOISE
    assert "output-integrity floor" in str(exc.value)
    assert ctx.saved == [], "a rejected render must never reach the upload"


def test_write_video_refuses_a_black_clip():
    ctx = _Ctx()
    with pytest.raises(OutputIntegrityError) as exc:
        gw_io.write_video(ctx, "outputs/r/video.mp4", black_clip(), fps=24)
    assert exc.value.verdict == oi.BLANK
    assert ctx.saved == []


def test_write_video_still_serves_a_real_render():
    ctx = _Ctx()
    asset = gw_io.write_video(ctx, "outputs/r/video.mp4", clean_clip(), fps=24)
    assert asset.ref == "outputs/r/video.mp4"
    assert ctx.saved == ["outputs/r/video.mp4"]


def test_write_video_screens_the_streaming_seam_too():
    ctx = _Ctx()
    clip = noise_clip(48)
    chunks = (clip[i:i + 8] for i in range(0, len(clip), 8))
    with pytest.raises(OutputIntegrityError):
        gw_io.write_video(ctx, "outputs/r/video.mp4", chunks, fps=24)
    assert ctx.saved == []


def test_write_image_refuses_a_flat_render():
    from PIL import Image

    ctx = _Ctx()
    with pytest.raises(OutputIntegrityError) as exc:
        gw_io.write_image(ctx, "outputs/r/image", Image.new("RGB", (256, 256)))
    assert exc.value.verdict == oi.BLANK
    assert ctx.saved == []
    assert gw_io.write_image(ctx, "outputs/r/image",
                             Image.fromarray(_scene())).ref.endswith(".webp")


# ---------------------------------------------------------------------------
# the typed fault and the event
# ---------------------------------------------------------------------------


def test_the_fault_maps_fatal_and_never_invalid():
    """BLAME: a render is produced by release code, model state AND payload
    together, so this is neither a payload verdict (INVALID) nor a
    release-declared fault. FATAL is the honest class — the hub's
    `jobResultEvidence` FATAL arm reads it as EvidenceExecutionFatal, "the
    release's CODE answering a REQUEST", with no hub change."""
    from gen_worker import executor
    from gen_worker.api.errors import RetryableError, ValidationError
    from gen_worker.pb import worker_scheduler_pb2 as pb

    exc = OutputIntegrityError(oi.NOISE, ref="outputs/r/video.mp4",
                               kind="video", summary="integrity noise")
    assert not isinstance(exc, (ValidationError, RetryableError))
    status, msg = executor._map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert msg.startswith("OutputIntegrityError:")


def test_reject_and_unmeasured_emit_a_typed_event_and_pass_does_not():
    from gen_worker import activity

    rows: list = []
    activity.reset_for_tests()
    try:
        for result, kind in (
            (oi.check_frames(noise_clip()), oi.NOISE),
            (oi.check_frames(np.zeros((4, 8, 8, 7), np.uint8)), oi.UNMEASURED),
            (oi.check_frames(clean_clip()), oi.PASS),
        ):
            before = len(rows)
            with _capture(activity, rows):
                try:
                    oi.enforce(result, ref="outputs/r/out", kind="video")
                except OutputIntegrityError:
                    pass
            if kind is oi.PASS:
                assert len(rows) == before, "a PASS buys no row"
            else:
                assert len(rows) == before + 1
                assert rows[-1].kind == activity.KIND_OUTPUT_INTEGRITY
                assert rows[-1].phase == kind
                assert "adjacent_frame_corr" in rows[-1].detail
    finally:
        activity.reset_for_tests()


class _capture:
    def __init__(self, activity, rows):
        self._a, self._rows = activity, rows

    def __enter__(self):
        self._prev = self._a._sink
        self._a._sink = self._rows.append
        return self

    def __exit__(self, *exc):
        self._a._sink = self._prev
        return False


def test_measured_floors_match_the_eval_half():
    """The eval half (cozy-eval ce#10, metric_set @7) and this serve half share
    the floors by construction; a disagreement would be a bug in one of them."""
    assert oi.NOISE_CORR_FLOOR == 0.6
    assert oi.BLANK_STD_FLOOR == 0.01
    assert oi.INTEGRITY_PAIRS == 5
    assert oi.INTEGRITY_TARGET_H == 96
