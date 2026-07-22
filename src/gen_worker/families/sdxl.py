"""SDXL family vocabulary (pgw#520 / pgw#516) — the shape tensorhub
validates SDXL repo metadata against, and an SDXL
:class:`~gen_worker.api.slot.Slot`'s fallback preset type.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from .base import FamilyDefaults, family

SdxlScheduler = Literal[
    "euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras",
    # th#1017: distilled-regime schedulers (few-step, near-zero CFG).
    "lcm", "euler_trailing",
]


@family("sdxl")
class SdxlDefaults(FamilyDefaults, frozen=True):
    """SDXL inference defaults + constraints.

    ``max_guidance`` is a CLAMP constraint (e.g. a distilled/turbo lineage
    pins guidance low) — it bounds the payload's ``guidance`` field, it
    never reshapes the wire contract (pgw#520 boundary: repo metadata may
    clamp within a function's payload schema, never add/remove fields).
    """

    scheduler: SdxlScheduler = "euler_a"
    steps: int = 28
    guidance: float = 6.0
    quality_preamble: str = ""
    negative: str = ""
    max_guidance: Optional[float] = None


@family("sdxl", kind="lora")
class SdxlLoraDefaults(FamilyDefaults, frozen=True):
    """SDXL LoRA overlay recipe opinions (pgw#516 settled foundation).

    A LoRA repo's OWN metadata — trained-in trigger words, the weight the
    trainer/curator recommends, and any recipe opinions the adapter carries
    (a distillation LoRA pinning ``steps=4``/``guidance=0``). EVERY field
    but ``trigger_words``/``schema_version`` defaults to ``None``: "no
    opinion" — a LoRA that is purely a style/subject overlay with no
    recipe requirements sets nothing, and the checkpoint's own recipe
    passes through untouched.

    Composition (th#767b, the SETTLED part of pgw#516): when this LoRA
    rides a pick, its non-``None`` fields OVERRIDE the base checkpoint's
    resolved recipe FIELD BY FIELD (not whole-object like the repo-metadata-
    over-fallback precedence above) — see
    ``gen_worker.api.slot.resolve_slot``'s ``lora_metadata_json`` merge and
    ``proto/CONTRACT.md``'s ``LoraOverlay.inference_defaults`` row. Only the
    fields this struct shares with :class:`SdxlDefaults` (``scheduler``,
    ``steps``, ``guidance``, ``max_guidance``) participate in that merge;
    ``trigger_words``/``recommended_weight`` have no checkpoint-recipe
    analog and are NOT applied to ``ctx.slots[slot].defaults`` — an
    endpoint reads them off the resolved lora's own object (the
    endpoint-authoring surface for that is explicitly OUT of this issue's
    settled foundation).
    """

    trigger_words: Tuple[str, ...] = ()
    recommended_weight: Optional[float] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    max_guidance: Optional[float] = None
    scheduler: Optional[SdxlScheduler] = None


__all__ = ["SdxlDefaults", "SdxlLoraDefaults", "SdxlScheduler"]
