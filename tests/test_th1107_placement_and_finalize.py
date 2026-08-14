"""th#1107 — two serve-path defects found by the inference-implementation audit.

1. The ``vae_only`` rung is selected when the pipeline FITS and only headroom is
   tight, yet it applied VAE tiling + attention slicing — VRAM tools that cost
   latency on a card that never needed them.
2. (RETIRED by th#1867.) ``strict_vram`` was enforced only on the reactive
   OOM-demotion path, so an auto SELECTION straight into an offload rung walked
   past the author's opt-out silently. §2.4 ruling 4 deleted the declaration
   itself — an author asserting a card requirement in softer words — so there
   is no opt-out left for a selection to walk past. What replaces the guard is
   `test_no_vram_markers_th1867.py`, which reds if the refusal comes back.
3. The image write path never signalled gw#516's decode->finalize handoff, so the
   encode+upload tail held the GPU slot on every image endpoint (measured:
   ``finalize_wall_ms == 0`` on 19 of 20 endpoints in the master hub DB).
"""

from __future__ import annotations

import pytest

from gen_worker.models import memory as mem


class _FakeVAE:
    def __init__(self) -> None:
        self.slicing = False
        self.tiling = False


class _FakePipeline:
    """Diffusers-shaped pipeline exposing the memory-saver entry points."""

    def __init__(self) -> None:
        self.vae = _FakeVAE()
        self.attention_slicing = False
        self.moved_to: list[str] = []

    def enable_vae_slicing(self) -> None:
        self.vae.slicing = True

    def enable_vae_tiling(self) -> None:
        self.vae.tiling = True

    def enable_attention_slicing(self) -> None:
        self.attention_slicing = True

    def to(self, device: str) -> "_FakePipeline":
        self.moved_to.append(device)
        return self


def test_vae_only_keeps_the_fast_attention_and_decode_paths() -> None:
    """vae_only means "it fits, headroom is tight" — not "shrink at any cost"."""
    pipe = _FakePipeline()
    applied = mem.apply_low_vram_config(pipe, mode="vae_only")

    assert applied["mode"] == "vae_only"
    # Slicing is a no-op at batch 1 and is kept for genuine batch decodes.
    assert pipe.vae.slicing is True
    # Tiling re-runs the VAE per tile and blends overlaps; attention slicing
    # replaces the fused SDPA/flash kernel with a chunked loop. Neither belongs
    # on a pipeline that fits.
    assert pipe.vae.tiling is False, "vae_only must not tile a pipeline that fits"
    assert pipe.attention_slicing is False, (
        "vae_only must not disable the fused attention path"
    )
    assert applied["vae_tiling"] is False
    assert applied["attention_slicing"] is False


def test_offload_rungs_still_get_every_memory_saver() -> None:
    """The rungs that exist because the model does NOT fit are unchanged."""
    pipe = _FakePipeline()
    mem._apply_vae_and_attention(pipe, {}, memory_bound=True)

    assert pipe.vae.slicing is True
    assert pipe.vae.tiling is True
    assert pipe.attention_slicing is True


def test_an_auto_selected_offload_rung_is_APPLIED_never_refused(monkeypatch) -> None:
    """th#1867: the selection that used to be refused is the normal path.

    This is the same scenario the deleted `strict_vram` test drove — a 40 GB
    pipeline on a 24 GB card, auto-selecting `model_offload`. It used to raise.
    Under §1.35 a card that looks too small is a card whose best rung is
    further down the ladder, so it now PLACES."""
    pipe = _FakePipeline()

    monkeypatch.setattr(mem, "select_auto_mode", lambda **_: "model_offload")
    monkeypatch.setattr(mem, "estimate_pipeline_size_gb", lambda _p: 40.0)
    monkeypatch.setattr(mem, "get_available_vram_gb", lambda: 24.0)
    monkeypatch.setattr(mem, "apply_low_vram_config",
                        lambda *a, **k: {"mode": "model_offload"})

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert mem.place_pipeline(
        pipe, mode="auto", ref="tensorhub/qwen-image")["mode"] == "model_offload"


def test_write_image_releases_the_gpu_slot_before_encode_and_upload() -> None:
    """gw#516's handoff must fire on the image path, as it does for video."""
    from gen_worker import io as gio

    events: list[str] = []

    class _Ctx:
        def _release_gpu_slot_for_finalize(self) -> None:
            events.append("release")

        def save_bytes(self, ref: str, data: bytes):
            events.append("upload")
            assert data, "encoded payload must reach the upload"
            return {"ref": ref, "size": len(data)}

    class _Image:
        def save(self, buf, **kwargs) -> None:
            events.append("encode")
            buf.write(b"webp-bytes")

    gio.write_image(_Ctx(), "out.webp", _Image(), format="webp")

    assert events == ["release", "encode", "upload"], (
        "the slot must be released BEFORE the encode+upload tail so it overlaps "
        "the next request's denoise"
    )
