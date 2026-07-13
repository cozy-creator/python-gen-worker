"""SDXL family vocabulary (pgw#520) — the shape tensorhub validates SDXL repo
metadata against, and an SDXL :class:`~gen_worker.api.slot.Slot`'s fallback
preset type.
"""

from __future__ import annotations

from typing import Literal, Optional

from .base import FamilyDefaults, family

SdxlScheduler = Literal["euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"]


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


__all__ = ["SdxlDefaults", "SdxlScheduler"]
