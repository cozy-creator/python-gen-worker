"""Resources merged-struct invariants.

Resources collapsed `ResourceRequirements` + `ScalingHints` (and dropped
`kind`). This test exercises the merged surface end-to-end.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import Resources


def test_static_envelope_fields() -> None:
    r = Resources(
        accelerator="cuda",
        requires_gpu=True,
        min_vram_gb=22.0,
        cuda_compute_min=8.0,
        required_libraries=("flash_attn", "torchao"),
    )
    assert r.accelerator == "cuda"
    assert r.requires_gpu is True
    assert r.min_vram_gb == 22.0
    assert r.cuda_compute_min == 8.0
    assert r.required_libraries == ("flash_attn", "torchao")
    assert r.compute_capability == {"min": "8.0"}


def test_dynamic_cost_shape_fields_present() -> None:
    r = Resources(
        vram_must_fit="full_model",
        vram_base=500 * 1024 * 1024,
        vram_size_multiplier=1.1,
        vram_scales_with=("width", "height"),
        runtime_scales_with=("num_inference_steps",),
    )
    assert r.vram_must_fit == "full_model"
    assert r.vram_base == 500 * 1024 * 1024
    assert r.vram_size_multiplier == 1.1
    assert r.vram_scales_with == ("width", "height")
    assert r.runtime_scales_with == ("num_inference_steps",)


def test_accelerator_gpu_normalizes_to_cuda() -> None:
    r = Resources(accelerator="gpu")
    assert r.accelerator == "cuda"
    assert r.requires_gpu is True


def test_accelerator_cpu_normalizes_to_none() -> None:
    r = Resources(accelerator="cpu")
    assert r.accelerator == "none"


def test_resources_no_kind_field() -> None:
    """The legacy `kind` field is gone."""
    with pytest.raises(TypeError):
        Resources(kind="training")  # type: ignore[call-arg]


def test_resources_no_scaling_hints_field() -> None:
    """Resources doesn't carry a nested scaling_hints field — those merged in."""
    r = Resources()
    assert not hasattr(r, "scaling_hints")


def test_resources_omit_defaults_keeps_wire_shape_tight() -> None:
    """msgspec.to_builtins skips default fields so the wire shape stays sparse."""
    r = Resources(min_vram_gb=4.0)
    wire = msgspec.to_builtins(r)
    # min_vram_gb is set, but vram_base / vram_scales_with / etc are at defaults.
    assert wire == {"min_vram_gb": 4.0}


def test_resources_rejects_negative_vram_base() -> None:
    with pytest.raises(ValueError, match="vram_base"):
        Resources(vram_base=-1)


def test_resources_rejects_bad_must_fit() -> None:
    with pytest.raises(ValueError, match="vram_must_fit"):
        Resources(vram_must_fit="invalid")  # type: ignore[arg-type]


def test_resources_rejects_zero_or_negative_min_vram_gb() -> None:
    with pytest.raises(ValueError, match="min_vram_gb"):
        Resources(min_vram_gb=0.0)
    with pytest.raises(ValueError, match="min_vram_gb"):
        Resources(min_vram_gb=-4.0)


def test_resources_rejects_bad_accelerator() -> None:
    with pytest.raises(ValueError, match="accelerator"):
        Resources(accelerator="quantum")  # type: ignore[arg-type]
