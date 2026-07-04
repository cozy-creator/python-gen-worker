"""Canonical dtype vocabulary + fuzzy bit-width resolution.

The HTTP API caller declares `OutputSpec.dtype` as either:
- A concrete value: `bf16`, `fp16`, `fp32`, `fp8:e4m3`, `fp8:e5m2`, `nvfp4`,
  `int8`, `int4:awq`, `int4:gptq`, `q2_k`, `q3_k_m`, `q4_0`, `q4_k_m`, `q4_k_s`,
  `q5_k_m`, `q5_k_s`, `q6_k`, `q8_0`, `f16`, `bf16`, `f32`.
- A fuzzy bit-width token: `2bit`, `3bit`, `4bit`, `5bit`, `6bit`, `8bit`,
  `16bit`, `32bit`. Resolves to the best concrete dtype available in the
  source repo (best = highest-fidelity / most common quant scheme).

The classifier resolves fuzzy tokens at selection time; only concrete values
land on the destination checkpoint's `attributes.dtype`.
"""

from __future__ import annotations

from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# Concrete dtype set (case-sensitive, lowercase canonical)
# ---------------------------------------------------------------------------

# Floating-point with no scheme parameterization
_CONCRETE_FP_BASE: frozenset[str] = frozenset({
    "fp32", "fp16", "bf16",
    "f32", "f16",  # GGUF spelling — same numerical type as fp32/fp16
})

# Floating-point with scheme parameterization (fp8:e4m3 etc.)
_CONCRETE_FP_PARAM: frozenset[str] = frozenset({
    "fp8:e4m3", "fp8:e5m2",
    "nvfp4",
})

# Integer / quantized
_CONCRETE_INT: frozenset[str] = frozenset({
    "int8", "int4",
    "int4:awq", "int4:gptq", "int4:gptq:g32", "int4:gptq:g64", "int4:gptq:g128",
    "int8:awq", "int8:gptq",
})

# GGUF quant tokens (per llama.cpp naming).
# q1_0 / iq1_* are extreme low-bit experimentals (rare; some community
# repos like Bonsai-8B ship them). Including for completeness — the
# server-side validator rejects unknown tokens, so the set must mirror
# every quant a HF repo might legitimately advertise.
_CONCRETE_GGUF: frozenset[str] = frozenset({
    "q1_0",
    "q2_k",
    "q3_k_s", "q3_k_m", "q3_k_l", "q3_k_xl",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q4_k_l",
    "q4_0_4_4", "q4_0_4_8", "q4_0_8_8",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q5_k_l",
    "q6_k", "q6_k_l",
    "q8_0",
    # i-quants (importance-matrix quants) — newer llama.cpp variants.
    "iq1_s", "iq1_m",
    "iq2_xs", "iq2_xxs", "iq2_s", "iq2_m",
    "iq3_xs", "iq3_xxs", "iq3_s", "iq3_m",
    "iq4_xs", "iq4_nl",
})

CONCRETE_DTYPES: frozenset[str] = (
    _CONCRETE_FP_BASE | _CONCRETE_FP_PARAM | _CONCRETE_INT | _CONCRETE_GGUF
)


# ---------------------------------------------------------------------------
# Fuzzy bit-width resolution per source kind
# ---------------------------------------------------------------------------

# Source kind → (fuzzy token → ordered list of concrete fallbacks).
# First match available in the source repo wins.
_FUZZY_RESOLUTION: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "gguf": {
        "32bit": ("f32",),
        "16bit": ("bf16", "f16"),
        "8bit":  ("q8_0",),
        "6bit":  ("q6_k_l", "q6_k"),
        "5bit":  ("q5_k_l", "q5_k_m", "q5_k_s", "q5_1", "q5_0"),
        "4bit":  ("q4_k_l", "q4_k_m", "q4_k_s", "q4_1", "q4_0"),
        "3bit":  ("q3_k_xl", "q3_k_l", "q3_k_m", "q3_k_s"),
        "2bit":  ("q2_k",),
    },
    "transformers": {
        "32bit": ("fp32",),
        "16bit": ("bf16", "fp16"),
        "8bit":  ("fp8:e4m3", "fp8:e5m2", "int8"),
        "4bit":  ("int4:awq", "int4:gptq", "nvfp4"),
        # 2/3/5/6/7-bit don't have canonical transformers schemes today
    },
    "diffusers": {
        "32bit": ("fp32",),
        "16bit": ("bf16", "fp16"),
        "8bit":  ("fp8:e4m3", "fp8:e5m2"),
        "4bit":  ("nvfp4", "int4"),
    },
    "diffusers-lora": {
        "32bit": ("fp32",),
        "16bit": ("bf16", "fp16"),
        "8bit":  ("fp8:e4m3",),
    },
    "peft": {
        "32bit": ("fp32",),
        "16bit": ("bf16", "fp16"),
        "8bit":  ("fp8:e4m3",),
    },
    "sentence-transformers": {
        "32bit": ("fp32",),
        "16bit": ("bf16", "fp16"),
    },
    "aio_singlefile": {
        "32bit": ("fp32",),
        "16bit": ("bf16", "fp16"),
    },
}


# Per-source-kind precision-priority order (used when no `outputs` is
# specified — pick the highest-fidelity dtype available).
_DEFAULT_PRECISION_ORDER: Mapping[str, tuple[str, ...]] = {
    "gguf": (
        "f16", "bf16",
        "q8_0",
        "q6_k_l", "q6_k",
        "q5_k_l", "q5_k_m", "q5_k_s", "q5_1", "q5_0",
        "q4_k_l", "q4_k_m", "q4_k_s", "q4_1", "q4_0",
        "q3_k_xl", "q3_k_l", "q3_k_m", "q3_k_s",
        "q2_k",
        "f32",
    ),
    "transformers": ("bf16", "fp16", "fp32"),
    "diffusers":    ("bf16", "fp16", "fp32"),
    "diffusers-lora": ("bf16", "fp16", "fp32"),
    "peft": ("bf16", "fp16", "fp32"),
    "sentence-transformers": ("bf16", "fp16", "fp32"),
    "aio_singlefile": ("bf16", "fp16", "fp32"),
}


def is_concrete_dtype(value: str) -> bool:
    """True when `value` is a recognized concrete dtype token."""
    return str(value or "").strip().lower() in CONCRETE_DTYPES


def is_fuzzy_bitwidth(value: str) -> bool:
    """True when `value` is a fuzzy bit-width token (`2bit`, `4bit`, `8bit`, ...)."""
    s = str(value or "").strip().lower()
    if not s.endswith("bit"):
        return False
    head = s[:-3]
    return head.isdigit() and 1 <= len(head) <= 2


def resolve_fuzzy_to_concrete(
    value: str,
    *,
    source_kind: str,
    available_dtypes: Sequence[str],
) -> str | None:
    """Resolve a fuzzy `<N>bit` token to a concrete dtype available in the
    source repo.

    Returns the resolved concrete dtype (lowercase), or None if no fallback
    matches what's available. The caller decides whether None means
    "fall through to QUANTIZE_FROM_BF16" or "raise DtypeUnavailable" based
    on source kind.

    `available_dtypes` is the list of concrete dtypes the classifier found
    in the source repo (e.g. for a GGUF repo: ["f16", "q8_0", "q4_k_m"]).
    """
    if not is_fuzzy_bitwidth(value):
        return None
    kind = str(source_kind or "").strip().lower()
    fuzzy = str(value).strip().lower()
    table = _FUZZY_RESOLUTION.get(kind, {}).get(fuzzy, ())
    avail = {a.strip().lower() for a in available_dtypes if a}
    for candidate in table:
        if candidate in avail:
            return candidate
    return None


def default_dtype_for_source(
    source_kind: str,
    available_dtypes: Sequence[str],
) -> str | None:
    """Pick the highest-fidelity dtype available for this source kind.

    Used when the invoker doesn't specify `outputs` — we default to the
    best precision the source ships. default change
    (was: lowest-precision quant for GGUF; now: highest-precision).
    """
    kind = str(source_kind or "").strip().lower()
    order = _DEFAULT_PRECISION_ORDER.get(kind, ("bf16", "fp16", "fp32"))
    avail = {a.strip().lower() for a in available_dtypes if a}
    for d in order:
        if d in avail:
            return d
    return None


def normalize_dtype(value: str) -> str:
    """Canonicalize a dtype token: lowercase, strip whitespace.
    Returns "" for empty input. Does NOT validate — caller checks
    `is_concrete_dtype` or `is_fuzzy_bitwidth` for that.
    """
    return str(value or "").strip().lower()


__all__ = [
    "CONCRETE_DTYPES",
    "is_concrete_dtype",
    "is_fuzzy_bitwidth",
    "resolve_fuzzy_to_concrete",
    "default_dtype_for_source",
    "normalize_dtype",
]
