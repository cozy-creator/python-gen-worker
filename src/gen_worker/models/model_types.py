"""Thin model types for the ship-code-as-is author surface (pgw#1370 seam).

``main_v2``-shape endpoints say ``ctx.checkpoint_defaults(SDXL)`` and read a
typed ``SDXL.Defaults``: per-checkpoint serving values are MUTABLE PLATFORM
DEPLOY STATE (hub rows validated against the schema the release EXPORTS —
``gen-worker release derive`` stamps ``msgspec`` JSON Schema of this struct),
and the zero-arg construction IS the platform fallback.

This module is deliberately minimal: it is the seam pgw#1370's derive and the
Paul-reviewed sdxl ``main_v2.py`` build against. The full defaults design
program (pgw#1376) and the model-type package shape (pgw#1377) own the final
form; coordinate there before widening it.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import msgspec

Number = Union[int, float]


class Knob(msgspec.Struct, frozen=True):
    """One resolvable serving knob: a default plus the checkpoint's range.

    ``resolve(None) -> default``; a caller value is CLAMPED into
    ``[lo, hi]`` caller-visibly via ``ctx.clamp`` — the API declares no
    bounds of its own, the checkpoint's knob owns the constraint.
    """

    default: Number
    lo: Optional[Number] = None
    hi: Optional[Number] = None
    name: str = ""

    def resolve(self, value: Optional[Number], ctx: Any) -> Number:
        if value is None:
            return self.default
        clamped = ctx.clamp(
            self.name or "value",
            value,
            lo=self.lo,
            hi=self.hi,
            reason="outside the checkpoint's declared range",
        )
        # ctx.clamp answers float; keep an int knob's integer arithmetic.
        if isinstance(self.default, int) and float(clamped).is_integer():
            return int(clamped)
        return clamped


class SDXL:
    """The SDXL model type: today, its checkpoint-defaults schema."""

    class Defaults(msgspec.Struct, frozen=True):
        # cfg=False marks a distilled checkpoint (guidance-off serving).
        cfg: bool = True
        steps: Knob = msgspec.field(
            default_factory=lambda: Knob(
                default=28, lo=1, hi=80, name="num_inference_steps"
            )
        )
        guidance: Knob = msgspec.field(
            default_factory=lambda: Knob(
                default=6.0, lo=1.0, hi=15.0, name="guidance_scale"
            )
        )
        positive_preamble: str = ""
        negative_preamble: str = ""


__all__ = ["Knob", "Number", "SDXL"]
