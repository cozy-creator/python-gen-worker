"""Memory-tier auto-selection — model-size-aware, not just total-VRAM thresholds.

The key behavior: offload ONLY when a model won't fit the available VRAM (minus
an activation margin). A model that fits stays FULLY RESIDENT even on a modest
card (no offload penalty); a model that doesn't fit gets model_offload. This is
the "efficient when there's enough VRAM, low-VRAM mode only when necessary" rule.
"""

from gen_worker.inference_memory import select_auto_mode

# select_auto_mode probes a pipeline only when model_size_gb is omitted; passing
# both available_vram_gb (FREE VRAM) and model_size_gb exercises the decision
# logic with no GPU. The ladder is derived from free VRAM, never total capacity.
_NO_PIPELINE = object()


def _mode(avail_gb, model_gb):
    return select_auto_mode(
        pipeline=_NO_PIPELINE, available_vram_gb=avail_gb, model_size_gb=model_gb,
    )


def _mode_peak(avail_gb, model_gb, peak_gb):
    return select_auto_mode(
        pipeline=_NO_PIPELINE, available_vram_gb=avail_gb,
        model_size_gb=model_gb, peak_vram_gb=peak_gb,
    )


def test_no_cuda_is_off():
    assert _mode(0.0, 4.0) == "off"


def test_model_that_fits_stays_resident_even_on_modest_card():
    # sd1.5 (~2.5GB) on an 8GB card: fits within 8 - 2 margin -> NO offload.
    assert _mode(8.0, 2.5) == "vae_only"


def test_model_too_big_for_the_card_offloads():
    # SDXL (~6.9GB) on an 8GB card: 6.9 > 8 - 2 -> model_offload (the SDXL case).
    assert _mode(8.0, 6.9) == "model_offload"


def test_big_model_fits_on_big_card_no_offload():
    # SDXL with 24GB FREE: fits with generous headroom (22 - 6.9 = 15.1 >= 8 GB
    # OFF_HEADROOM) -> fully unoptimized "off"; VAE slicing/tiling would only
    # trade speed for memory we demonstrably don't need.
    assert _mode(24.0, 6.9) == "off"


def test_very_low_vram_is_aggressive_even_for_small_models():
    # <=6GB total: even a fitting model gets aggressive group offload.
    assert _mode(4.0, 1.5) == "group_offload"


def test_unknown_model_size_falls_back_to_conservative_threshold():
    # Can't estimate size on a modest card -> conservative model_offload.
    assert _mode(8.0, 0.0) == "model_offload"


# --- #339: declared Resources.peak_vram_per_request_gb drives the decision ---


def test_declared_peak_forces_offload_for_a_small_model():
    # A model whose WEIGHTS fit easily (2.5GB on a 24GB card -> normally
    # resident) but whose DECLARED per-request peak is huge (23GB) must offload:
    # requirement = max(2.5, 23) = 23 > 24 - 2 margin.
    assert _mode_peak(24.0, 2.5, 23.0) == "model_offload"
    # Same model, no declaration: 22 - 2.5 = 19.5 >= 8 GB OFF_HEADROOM -> fully
    # resident and unoptimized ("off").
    assert _mode(24.0, 2.5) == "off"


def test_declared_peak_below_weights_never_lowers_the_requirement():
    # A too-small declared peak must not trick a big model into staying resident:
    # requirement = max(6.9, 1.0) = 6.9 > 8 - 2 -> still model_offload.
    assert _mode_peak(8.0, 6.9, 1.0) == "model_offload"


def test_modest_declared_peak_that_still_fits_stays_resident():
    # sd1.5 weights (2.5GB) + a declared 4GB peak on an 8GB card: max(2.5,4)=4
    # <= 8 - 2 -> stays fully resident (no needless offload penalty).
    assert _mode_peak(8.0, 2.5, 4.0) == "vae_only"


def test_no_declared_peak_is_unchanged():
    # peak_vram_gb=None must reproduce the model-size-only decision exactly.
    assert _mode_peak(8.0, 2.5, None) == _mode(8.0, 2.5) == "vae_only"
    assert _mode_peak(8.0, 6.9, None) == _mode(8.0, 6.9) == "model_offload"
