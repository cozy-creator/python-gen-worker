"""@training_function scaling-hints kwargs still flow into Resources.

Resources merged the old `ScalingHints` fields (vram_must_fit, vram_base,
vram_size_multiplier, vram_scales_with, runtime_scales_with) in 0.7.0; the
training decorator now stores a Resources value on `_scaling_hints` rather
than the old dedicated `ScalingHints` type.
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import Resources
from gen_worker.conversion import ConversionContext, ProducedFlavor, Source, training_function


def test_training_function_accepts_direct_scaling_hint_fields() -> None:
    @training_function(
        kind="quantization",
        vram_must_fit="largest_component",
        vram_base=500,
        vram_size_multiplier=1.25,
        vram_scales_with=["specs[0].scheme"],
        runtime_scales_with=["specs[0].scheme"],
    )
    def quantize(ctx: ConversionContext, source: Source) -> list[ProducedFlavor]:
        return []

    hints = getattr(quantize, "_scaling_hints")
    assert msgspec.to_builtins(hints) == {
        "vram_must_fit": "largest_component",
        "vram_base": 500,
        "vram_size_multiplier": 1.25,
        "vram_scales_with": ("specs[0].scheme",),
        "runtime_scales_with": ("specs[0].scheme",),
    }


def test_training_function_rejects_mixed_scaling_hint_forms() -> None:
    with pytest.raises(TypeError, match="use one form"):

        @training_function(
            scaling_hints=Resources(vram_must_fit="full_model"),
            vram_must_fit="largest_component",
        )
        def quantize(ctx: ConversionContext, source: Source) -> list[ProducedFlavor]:
            return []


def test_resources_normalizes_frozen_struct_fields() -> None:
    req = Resources(
        accelerator="gpu",
        cuda_compute_min=8,
        min_vram_gb=16,
        required_libraries=(" torch ", "", "bitsandbytes"),
    )

    assert req.accelerator == "cuda"
    assert req.requires_gpu is True
    assert req.compute_capability == {"min": "8.0"}
    assert req.min_vram_gb == 16.0
    assert req.required_libraries == ("torch", "bitsandbytes")
