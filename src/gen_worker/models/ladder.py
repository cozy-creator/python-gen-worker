"""Precision classes — the VOCABULARY a producer declares its class from."""

from __future__ import annotations

CLASS_BASE = "base"
CLASS_FP8 = "fp8"
CLASS_SVDQ_FP4 = "svdq-fp4"
CLASS_SVDQ_INT4 = "svdq-int4"
CLASS_NVFP4 = "nvfp4"
CLASS_NVFP4_W4A4 = "nvfp4-w4a4"
CLASS_GGUF = "gguf"

PRECISION_CLASSES = frozenset({
    CLASS_BASE,
    CLASS_FP8,
    CLASS_GGUF,
    CLASS_NVFP4,
    CLASS_NVFP4_W4A4,
    CLASS_SVDQ_FP4,
    CLASS_SVDQ_INT4,
})


__all__ = [
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_GGUF",
    "CLASS_NVFP4",
    "CLASS_NVFP4_W4A4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "PRECISION_CLASSES",
]
