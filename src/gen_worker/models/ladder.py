"""Precision-ladder spec (th#697) — flavor precision classes + placement requirements.

A flavor's *precision class* names its quantization lane (``fp8``,
``svdq-int4``, ...). A :class:`Placement` states which silicon can run it:
a discrete SM allow-list (fail-closed — kernel wheels are per-arch), an
open-ended SM floor, and the importable engine libraries the lane needs.

Produced flavors carry their placement in ``checkpoints.metadata["placement"]``
(stamped at publish by :func:`gen_worker.convert.publish.publish_flavors`).
Unstamped/mirrored rows fall back to the token-derived defaults here — the
same defaults the stamping writes, so both paths agree. The ladder walk
itself (rung ordering per arch class) lands with the shared Go+Py vector
spec; this module is the classification + placement half both sides share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .svdq import SVDQ_FP4_SMS, SVDQ_INT4_SMS

CLASS_BASE = "base"  # bare bf16/fp16/fp32 row — runs anywhere a card fits it
CLASS_FP8 = "fp8"  # fp8-E4M3 storage; universal (bf16-upcast path needs no fp8 silicon)
CLASS_SVDQ_FP4 = "svdq-fp4"  # nunchaku SVDQuant fp4 — consumer Blackwell only
CLASS_SVDQ_INT4 = "svdq-int4"  # nunchaku SVDQuant int4 — sm_75-89
CLASS_NVFP4 = "nvfp4"  # plain nvfp4 artifact — Blackwell datacenter, TRT lane (not a diffusers rung)

_BASE_TOKENS = ("", "bf16", "fp16", "fp32")


@dataclass(frozen=True)
class Placement:
    """Arch requirements for one flavor. Empty fields = unconstrained."""

    precision_class: str
    sm_allowed: tuple[int, ...] = ()  # discrete allow-list (gpu_sm as int, e.g. 89, 120)
    sm_min: int = 0  # open-ended floor; 0 = none
    engines: tuple[str, ...] = ()  # importable libraries required to load

    def admits_sm(self, gpu_sm: int) -> bool:
        if self.sm_allowed and gpu_sm not in self.sm_allowed:
            return False
        if self.sm_min and gpu_sm < self.sm_min:
            return False
        return True


def classify_flavor_token(flavor: str) -> str:
    """Flavor token -> precision class; "" when unrecognized (gguf/trt/etc.
    stay opaque — never ladder rungs)."""
    token = str(flavor or "").strip().lower()
    if token in _BASE_TOKENS:
        return CLASS_BASE
    if token.startswith("svdq-fp4"):
        return CLASS_SVDQ_FP4
    if token.startswith("svdq-int4"):
        return CLASS_SVDQ_INT4
    if token == "fp8" or token.startswith("fp8-"):
        return CLASS_FP8
    if token == "nvfp4" or token.startswith("nvfp4-"):
        return CLASS_NVFP4
    return ""


def default_placement(precision_class: str) -> Optional[Placement]:
    """Token-derived placement defaults — the fallback for unstamped rows
    and the source the publish-time stamp writes."""
    if precision_class == CLASS_BASE:
        return Placement(CLASS_BASE)
    if precision_class == CLASS_FP8:
        return Placement(CLASS_FP8)  # fp8-storage serves on any silicon
    if precision_class == CLASS_SVDQ_FP4:
        return Placement(CLASS_SVDQ_FP4, sm_allowed=tuple(SVDQ_FP4_SMS), engines=("nunchaku",))
    if precision_class == CLASS_SVDQ_INT4:
        return Placement(CLASS_SVDQ_INT4, sm_allowed=tuple(SVDQ_INT4_SMS), engines=("nunchaku",))
    if precision_class == CLASS_NVFP4:
        return Placement(CLASS_NVFP4, sm_min=100)
    return None


def placement_for_flavor(flavor: str) -> Optional[Placement]:
    return default_placement(classify_flavor_token(flavor))


def placement_to_metadata(p: Placement) -> dict[str, Any]:
    """The ``checkpoints.metadata["placement"]`` wire/storage shape."""
    out: dict[str, Any] = {"precision_class": p.precision_class}
    if p.sm_allowed:
        out["sm_allowed"] = list(p.sm_allowed)
    if p.sm_min:
        out["sm_min"] = p.sm_min
    if p.engines:
        out["engines"] = list(p.engines)
    return out


def placement_from_metadata(meta: Mapping[str, Any] | None) -> Optional[Placement]:
    """Parse a checkpoint metadata mapping (the whole bag or the placement
    block itself). Unknown keys ignored; malformed values fail soft (None)."""
    if not isinstance(meta, Mapping):
        return None
    block = meta.get("placement", meta)
    if not isinstance(block, Mapping):
        return None
    cls = str(block.get("precision_class", "") or "").strip().lower()
    if not cls:
        return None
    try:
        sm_allowed = tuple(int(v) for v in (block.get("sm_allowed") or ()))
        sm_min = int(block.get("sm_min") or 0)
    except (TypeError, ValueError):
        return None
    engines = tuple(
        s for s in (str(e).strip() for e in (block.get("engines") or ())) if s
    )
    return Placement(cls, sm_allowed=sm_allowed, sm_min=sm_min, engines=engines)


__all__ = [
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_NVFP4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "Placement",
    "classify_flavor_token",
    "default_placement",
    "placement_for_flavor",
    "placement_from_metadata",
    "placement_to_metadata",
]
