"""Per-COMPONENT load-time dtype facts."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

LOAD_DTYPES = ("fp32", "fp16", "bf16")


@dataclass(frozen=True)
class ComponentDtype:
    """One component class's required LOAD dtype, with the reason it exists."""

    dtype: str
    reason: str

    def __post_init__(self) -> None:
        if self.dtype not in LOAD_DTYPES:
            raise ValueError(
                f"ComponentDtype.dtype must be one of {LOAD_DTYPES}, got {self.dtype!r}"
            )
        if not str(self.reason or "").strip():
            raise ValueError("ComponentDtype requires a non-empty reason")


COMPONENT_DTYPES: Dict[str, ComponentDtype] = {
    "AutoencoderKLWan": ComponentDtype(
        "fp32",
        "the Wan VAE is numerically fragile: bf16 latents degrade decoded "
        "frames visibly, and the diffusers Wan reference usage loads it fp32 "
        "alongside a bf16 transformer (ie#546 wave-2 / pgw#667)",
    ),
    "AutoencoderKLMiniMaxH3": ComponentDtype(
        "fp32",
        "H3's released VAE is fp32 and its verified decode recipe is fp16 "
        "AUTOCAST OVER fp32 weights, so the class pins "
        "_keep_in_fp32_modules=[encoder, decoder, quant_conv, "
        "post_quant_conv]. The compute is already fp16 — narrowing the "
        "weights buys bandwidth only, at 3 fewer mantissa bits than the "
        "arithmetic. Measured: the fp16-WEIGHT arm (strictly wider than "
        "bf16) scored 74.97 dB PSNR min / 0.0186 max abs against the fp32 "
        "decode and ran 0.94x — worse on both axes (ie#621, ie#718)",
    ),
    "AutoencoderKLMiniMaxH3Audio": ComponentDtype(
        "fp32",
        "H3's audio VAE is a DAC/BigVGAN stack (weight-normalized "
        "convolutions, Snake activations) that upstream measures as "
        "~20 dB QUIETER under bfloat16, and it decodes at its own parameter "
        "dtype with no autocast to widen it back. H3's audio is also the "
        "half two independent probes found fragile, so a video-only review "
        "would not catch it (te#191/te#192, ie#718)",
    ),
}


def component_dtype_for_class(class_name: str) -> Optional[ComponentDtype]:
    """The load-dtype fact for one diffusers component class, or None."""
    return COMPONENT_DTYPES.get(str(class_name or "").strip())


def component_classes(pipeline_cls: Any) -> Dict[str, str]:
    """``{part_name: component class NAME}`` from a pipeline class's ``__init__`` annotations."""
    if not isinstance(pipeline_cls, type):
        return {}
    try:
        init = inspect.getattr_static(pipeline_cls, "__init__")
    except AttributeError:
        return {}
    raw = getattr(init, "__annotations__", None) or {}
    out: Dict[str, str] = {}
    for name, ann in raw.items():
        if name in ("self", "return", "args", "kwargs"):
            continue
        if isinstance(ann, type):
            out[str(name)] = ann.__name__
        elif isinstance(ann, str) and ann.isidentifier():
            out[str(name)] = ann
    return out


def component_dtypes_for_classes(
    classes: Mapping[str, str],
) -> Dict[str, ComponentDtype]:
    """``{part_name: ComponentDtype}`` for the parts of ``{part_name: class name}`` that carry a fact."""
    out: Dict[str, ComponentDtype] = {}
    for part, class_name in classes.items():
        fact = component_dtype_for_class(class_name)
        if fact is not None:
            out[str(part)] = fact
    return out


__all__ = [
    "COMPONENT_DTYPES",
    "LOAD_DTYPES",
    "ComponentDtype",
    "component_classes",
    "component_dtype_for_class",
    "component_dtypes_for_classes",
]
