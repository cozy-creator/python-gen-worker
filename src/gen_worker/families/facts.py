"""Per-COMPONENT load-time dtype facts.

A component's dtype is part of its RESIDENT IDENTITY, decided AT LOAD — not a
post-load adjustment. Upcasting a bf16-loaded tensor recovers no precision; it
only makes the truncation invisible. Some architectures therefore have a
component that must come off disk WIDER than the composition's compute dtype:
wan-2.2's ``AutoencoderKLWan`` is the fleet's first (the diffusers Wan
reference usage loads the VAE fp32 while the transformer loads bf16, and a
bf16-loaded Wan VAE degrades frames visibly).

This module is the ONE home for that knowledge, keyed by the diffusers
COMPONENT CLASS name rather than by family or by endpoint:

* the class is what the fact is true OF — every pipeline class composing
  ``AutoencoderKLWan`` (T2V / I2V / V2V / TI2V) and every fine-tune of the
  family inherits it with zero per-endpoint declaration;
* it is derivable at BOTH ends — from the pipeline class's ``__init__``
  annotations at build time (published into the manifest's component tree) and
  from the snapshot's own ``model_index.json`` at load time (authoritative: a
  fine-tune may substitute a class);
* endpoint-private tables must not exist, for the same reason
  :data:`gen_worker.view.SAMPLERS` is the one sampler table — two endpoints
  disagreeing about the Wan VAE's load dtype would make one checkpoint mean
  different math depending on which one served it.

Deliberately NOT catalog recipe data: a numeric-stability floor is an
architecture fact, identical for every checkpoint of the family, and a per-repo
value would let one bad metadata row silently degrade output. Recipe data keeps
what a checkpoint genuinely varies (steps, guidance, shift).

Deliberately NOT a ``Slot(component_dtypes=...)`` endpoint declaration either:
that is the sibling-as-part shape SDK v2 deleted (``gen_worker.api.tree``), and
it would put the same fact in N endpoints to drift independently. A new fragile
component adds a row HERE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

# Load dtypes a fact may name. Storage/quant lanes (fp8, nvfp4, nf4) are the
# fit ladder's business and are never expressed here: this axis exists only to
# widen a component the composition's compute dtype would truncate.
LOAD_DTYPES = ("fp32", "fp16", "bf16")


@dataclass(frozen=True)
class ComponentDtype:
    """One component class's required LOAD dtype, with the reason it exists.

    The reason is carried as data, not a comment: the loader logs it when it
    widens a component, so an operator reading a boot log sees WHY one part of
    the composition is resident at a different precision.
    """

    dtype: str
    reason: str

    def __post_init__(self) -> None:
        if self.dtype not in LOAD_DTYPES:
            raise ValueError(
                f"ComponentDtype.dtype must be one of {LOAD_DTYPES}, got {self.dtype!r}"
            )
        if not str(self.reason or "").strip():
            raise ValueError("ComponentDtype requires a non-empty reason")


# diffusers component CLASS name -> its required load dtype.
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


def component_dtypes_for_classes(
    classes: Mapping[str, str],
) -> Dict[str, ComponentDtype]:
    """``{part_name: ComponentDtype}`` for the parts of ``{part_name: class
    name}`` that carry a fact. Parts with no fact are absent (an empty result
    means "the composition loads uniformly at its compute dtype")."""
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
    "component_dtype_for_class",
    "component_dtypes_for_classes",
]
