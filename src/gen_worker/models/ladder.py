"""Precision-ladder spec — precision classes + placement requirements.

A quant token's *precision class* names its quantization lane (``fp8``,
``svdq-int4``, ...). A :class:`Placement` states which silicon can run it:
a discrete SM allow-list (fail-closed — kernel wheels are per-arch), an
open-ended SM floor, and the importable engine libraries the lane needs.

Produced flavors carry their placement in ``checkpoints.metadata["placement"]``
(stamped at publish by :func:`gen_worker.convert.publish.publish_flavors`).
Unstamped/mirrored rows fall back to the token-derived defaults here — the
same defaults the stamping writes, so both paths agree. The ladder WALK
(rung ordering per arch class) lives hub-side (tensorhub's
internal/orchestrator/precision resolver) and delivers picks via HelloAck;
this module is the classification + placement half, plus the family lane
policy. There is deliberately no local walk and no local AUTO fp8 fold:
locally, fit is the loading layer's job (runtime fp8 rung + the offload
ladder), and selection within a tag group is §1.33 contract compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .svdq import SVDQ_FP4_SMS, SVDQ_INT4_SMS

CLASS_BASE = "base"  # bare bf16/fp16/fp32 row — runs anywhere a card fits it
CLASS_FP8 = "fp8"  # fp8-E4M3 storage; universal (bf16-upcast path needs no fp8 silicon)
CLASS_SVDQ_FP4 = "svdq-fp4"  # nunchaku SVDQuant fp4 — consumer Blackwell only
CLASS_SVDQ_INT4 = "svdq-int4"  # nunchaku SVDQuant int4 — sm_75-89
CLASS_NVFP4 = "nvfp4"  # plain nvfp4 artifact — Blackwell datacenter, no serving lane (not a diffusers rung)
# Calibrated nvfp4 with two-level scales: torch fp4 blockwise
# scaled_mm serve lane. Blackwell-only (sm_100+ incl. sm_120 consumer) —
# no fp4 silicon below, and the 4x dequant blow-up erases the fit story.
CLASS_NVFP4_W4A4 = "nvfp4-w4a4"

_BASE_TOKENS = ("", "bf16", "fp16", "fp32")


@dataclass(frozen=True)
class Placement:
    """Arch requirements for one flavor. Empty fields = unconstrained.

    ``sm_min = 0`` is not an absence collapse: "unconstrained" is this class's
    STATED meaning for every empty field (`sm_allowed=()` says the same thing),
    the placement is the flavor's own declaration rather than an operator knob,
    and a missing floor cannot admit anything ``sm_allowed`` and the engine
    list do not already admit.
    """

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
    """Flavor token -> precision class; "" when unrecognized (gguf/etc.
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
    if token == "nvfp4-w4a4" or token.startswith("nvfp4-w4a4-"):
        return CLASS_NVFP4_W4A4
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
    if precision_class == CLASS_NVFP4_W4A4:
        return Placement(CLASS_NVFP4_W4A4, sm_min=100)
    if precision_class == CLASS_NVFP4:
        return Placement(CLASS_NVFP4, sm_min=100)
    return None



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



# Native fp8 tensor-core compute exists on SM >= 89 (sm_89 Ada, sm_90 Hopper,
# sm_100+/120 Blackwell). Below that, fp8 storage still SERVES (bf16-upcast
# path) — this floor gates only the fp8-over-bf16 PREFERENCE, never admission.
FP8_COMPUTE_MIN_SM = 89


# --- Family-root policy — twin of tensorhub's modelfamily.Root ----

# Families whose root is not derivable by normalization alone. Roots collapse
# fine-tune/scheduler/distillation variants that keep the weight envelope.
_FAMILY_ROOT_OVERRIDES = {
    "sd14": "sd1", "sd15": "sd1",
    "sdxl-turbo": "sdxl", "sdxl-pony": "sdxl", "sdxl-illustrious": "sdxl",
    "sdxl-lightning": "sdxl", "sdxl-hyper": "sdxl", "sdxl-refiner": "sdxl",
    "sd35-large-turbo": "sd35-large",
    "flux1-dev": "flux1", "flux1-schnell": "flux1",
    "flux1-kontext": "flux1", "flux1-krea": "flux1",
    "flux2-dev": "flux2", "flux2-pro": "flux2",
    "z-image-turbo": "z-image",
    "svd-xt": "svd",
}



# Conv-UNet roots get no fp8-GEMM win (torch scaled_mm is Linear-only, and
# SDXL w8a8 measured 1.9-2.7x slower than bf16): their fp8-w8a8 rows
# are AUTO-ineligible and the scale-free #fp8 row is the family table-best
# on sm_89+; bf16 stays the sub-floor default. Explicit pins still resolve
# w8a8. Twin of tensorhub precision.convUNetW8A8ExcludedRoots.
CONV_UNET_W8A8_EXCLUDED_ROOTS = frozenset({"sd1", "sd2", "sdxl"})




# The flavor-token parses (classify_flavor_token, placement_for_flavor) are
# package-internal choke points, not public API. They die when typed
# descriptors are backfilled; nothing new may grow on them.
__all__ = [
    "FP8_COMPUTE_MIN_SM",
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_NVFP4",
    "CLASS_NVFP4_W4A4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "CONV_UNET_W8A8_EXCLUDED_ROOTS",
    "Placement",
    "default_placement",
    "placement_to_metadata",
]
