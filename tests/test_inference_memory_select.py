"""Memory-tier auto-selection — model-size-aware, not just total-VRAM thresholds.

The key behavior: offload ONLY when a model won't fit the available VRAM (minus
an activation margin). A model that fits stays FULLY RESIDENT even on a modest
card (no offload penalty); a model that doesn't fit gets model_offload. This is
the "efficient when there's enough VRAM, low-VRAM mode only when necessary" rule.
"""

from gen_worker.inference_memory import select_auto_mode

# select_auto_mode probes a pipeline only when model_size_gb is omitted; passing
# both total_vram_gb and model_size_gb exercises the decision logic with no GPU.
_NO_PIPELINE = object()


def _mode(total_gb, model_gb):
    return select_auto_mode(
        pipeline=_NO_PIPELINE, total_vram_gb=total_gb, model_size_gb=model_gb,
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
    # SDXL on a 24GB card: fits easily -> full residency, no offload (production).
    assert _mode(24.0, 6.9) == "vae_only"


def test_very_low_vram_is_aggressive_even_for_small_models():
    # <=6GB total: even a fitting model gets aggressive group offload.
    assert _mode(4.0, 1.5) == "group_offload"


def test_unknown_model_size_falls_back_to_conservative_threshold():
    # Can't estimate size on a modest card -> conservative model_offload.
    assert _mode(8.0, 0.0) == "model_offload"
