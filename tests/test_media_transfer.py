"""gw#549 media transfer + gw#550 host canary — CPU-fallback coverage.

The CUDA staging pipeline itself is proven on GPU pods (evidence run); CI
proves the device-agnostic conversion semantics, the passthrough/ordering
contract of the staged iterator, the zero-copy encoder handoff, and that the
canary measures-and-caches without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from gen_worker import io as gw_io
from gen_worker import media_transfer as mt
from gen_worker.video_encode import StreamingVideoEncoder, frames_to_uint8

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")


# ---- gpu_frames_to_uint8 (device-agnostic semantics, run on CPU) -----------

def test_uint8_conversion_matches_numpy_reference() -> None:
    rng = np.random.default_rng(7)
    arr = rng.random((3, 32, 32, 3), dtype=np.float32)
    want = frames_to_uint8(arr.copy())
    got = mt.gpu_frames_to_uint8(torch.from_numpy(arr)).numpy()
    np.testing.assert_array_equal(got, want)


def test_uint8_conversion_scales_unit_floats_and_clips_others() -> None:
    unit = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
    assert mt.gpu_frames_to_uint8(unit).numpy().max() == 128  # 0.5*255 rounded
    big = torch.full((1, 4, 4, 3), 300.0, dtype=torch.float32)
    assert mt.gpu_frames_to_uint8(big).numpy().max() == 255
    neg = torch.full((1, 4, 4, 3), -3.0, dtype=torch.float32)
    assert mt.gpu_frames_to_uint8(neg).numpy().min() == 0


def test_uint8_conversion_crops_to_even_dims_and_expands_single_frame() -> None:
    t = torch.zeros(5, 5, 3, dtype=torch.uint8)  # single [H, W, 3] frame
    out = mt.gpu_frames_to_uint8(t)
    assert tuple(out.shape) == (1, 4, 4, 3)
    assert out.is_contiguous()


def test_uint8_conversion_passes_through_uint8_and_rejects_bad_shapes() -> None:
    t = (torch.arange(2 * 4 * 4 * 3, dtype=torch.int64) % 256).to(torch.uint8)
    out = mt.gpu_frames_to_uint8(t.reshape(2, 4, 4, 3))
    assert out.dtype == torch.uint8
    with pytest.raises(Exception):
        mt.gpu_frames_to_uint8(torch.zeros(4, 4))


def test_bf16_frames_convert_exactly_like_float32() -> None:
    rng = np.random.default_rng(11)
    arr = rng.random((2, 8, 8, 3), dtype=np.float32)
    f32 = mt.gpu_frames_to_uint8(torch.from_numpy(arr))
    bf16 = mt.gpu_frames_to_uint8(torch.from_numpy(arr).to(torch.bfloat16))
    # bf16 storage loses mantissa bits; values must stay within 1 step.
    assert int((f32.int() - bf16.int()).abs().max()) <= 1


# ---- staged_uint8_chunks (passthrough + ordering; CUDA path pod-proven) ----

def test_staged_chunks_pass_non_cuda_input_through_in_order() -> None:
    rng = np.random.default_rng(3)
    chunks = [
        rng.random((2, 16, 16, 3), dtype=np.float32),          # numpy
        torch.from_numpy(rng.random((1, 16, 16, 3), dtype=np.float32)),  # cpu tensor
        rng.integers(0, 255, (3, 16, 16, 3), dtype=np.uint8),  # uint8
    ]
    out = list(mt.staged_uint8_chunks(iter(chunks)))
    assert len(out) == 3
    for got, want in zip(out, chunks):
        assert got is want  # passthrough, no copies


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_staged_chunks_cuda_pipeline_matches_reference() -> None:
    rng = np.random.default_rng(5)
    ref = [rng.random((4, 32, 32, 3), dtype=np.float32) for _ in range(5)]
    chunks = [torch.from_numpy(a).cuda() for a in ref]
    out = []
    for host in mt.staged_uint8_chunks(iter(chunks)):
        out.append(np.array(host, copy=True))  # staging buffers are reused
    assert len(out) == len(ref)
    for got, want in zip(out, ref):
        np.testing.assert_array_equal(got, frames_to_uint8(want))


# ---- write_video end-to-end through the staged path (CPU chunks) -----------

class _Ctx:
    """Minimal ctx: save_video captures the temp path's probe result."""

    def __init__(self) -> None:
        self.saved: dict = {}

    def save_video(self, path, ref, format="mp4"):
        self.saved = gw_io.probe_video(path)
        self.saved["path"] = str(path)
        return self.saved


def test_write_video_streaming_iterator_still_encodes_every_frame() -> None:
    rng = np.random.default_rng(9)
    chunks = [rng.random((6, 64, 64, 3), dtype=np.float32) for _ in range(4)]
    ctx = _Ctx()
    gw_io.write_video(ctx, "clip", iter(chunks), fps=24.0)
    assert ctx.saved["width"] == 64 and ctx.saved["height"] == 64
    assert abs(ctx.saved["duration_s"] - 1.0) < 0.15  # 24 frames @ 24fps


# ---- zero-copy encoder handoff ----------------------------------------------

def test_encoder_zero_copy_and_fallback_produce_identical_streams(tmp_path) -> None:
    rng = np.random.default_rng(13)
    arr = (rng.random((8, 64, 64, 3)) * 255).astype(np.uint8)

    def encode(path, force_fallback: bool) -> bytes:
        enc = StreamingVideoEncoder(path, fps=24.0)
        if force_fallback:
            enc._zero_copy = False
        enc.add(arr)
        enc.finish()
        return open(path, "rb").read()

    fast = encode(str(tmp_path / "zc.mp4"), force_fallback=False)
    slow = encode(str(tmp_path / "nd.mp4"), force_fallback=True)
    assert fast == slow


def test_encoder_handles_noncontiguous_crop_input() -> None:
    # Odd dims force the encoder-side crop slice (non-contiguous view).
    rng = np.random.default_rng(17)
    arr = (rng.random((4, 33, 47, 3)) * 255).astype(np.uint8)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        enc = StreamingVideoEncoder(f.name, fps=8.0)
        enc.add(arr)
        enc.finish()
        info = gw_io.probe_video(f.name)
        assert info["width"] == 46 and info["height"] == 32


# ---- frames_to_uint8 CUDA branch (buffered-input byte cut) ------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_frames_to_uint8_converts_cuda_tensor_on_device() -> None:
    rng = np.random.default_rng(19)
    arr = rng.random((2, 16, 16, 3), dtype=np.float32)
    got = frames_to_uint8(torch.from_numpy(arr).cuda())
    np.testing.assert_array_equal(got, frames_to_uint8(arr))


# ---- host canary (gw#550) ---------------------------------------------------

def test_host_canary_measures_cpu_axes_without_cuda() -> None:
    from gen_worker import host_canary as hc

    report = hc.measure_host_canary()
    assert report.memcpy_gbps > 0
    assert report.cpu_single_mbps > 0
    assert report.cpu_multi_mbps >= report.cpu_single_mbps * 0.5
    assert report.vcpus >= 1
    assert report.ram_total_gb > 0
    assert 0 < report.duration_ms < 30_000
    if not torch.cuda.is_available():
        assert report.h2d_gbps == 0.0 and report.d2h_gbps == 0.0
        assert report.pinned_alloc_ok is False


def test_host_canary_is_cached_per_process(monkeypatch) -> None:
    from gen_worker import host_canary as hc

    monkeypatch.setattr(hc, "_cached", None)
    first = hc.get_host_canary()
    assert hc.get_host_canary() is first


def test_host_canary_rides_hello_worker_resources(monkeypatch) -> None:
    from gen_worker import host_canary as hc
    from gen_worker import lifecycle as lc

    monkeypatch.setattr(
        hc, "_cached",
        hc.HostCanaryReport(memcpy_gbps=8.5, cpu_single_mbps=400.0,
                            cpu_multi_mbps=3000.0, vcpus=32,
                            ram_total_gb=62.0, duration_ms=900),
    )
    resources = lc.Lifecycle.build_resources(_FakeLifecycle())  # type: ignore[arg-type]
    assert resources.host_canary.memcpy_gbps == 8.5
    assert resources.host_canary.vcpus == 32
    assert resources.host_canary.duration_ms == 900
    assert resources.torch_version == "2.13.0+cu130"


class _FakeLifecycle:
    """Just enough state for build_resources."""

    hardware: dict = {"gpu_count": 0, "gpu_total_mem": 0, "gpu_name": "",
                      "gpu_sm": "", "torch_version": "2.13.0+cu130",
                      "installed_libs": []}

    class _settings:  # noqa: D106 - namespace stub
        worker_image_digest = ""
        runpod_pod_id = ""
