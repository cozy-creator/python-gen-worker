"""Precision-ladder spec (th#697) — flavor precision classes + placement requirements.

A flavor's *precision class* names its quantization lane (``fp8``,
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
policy (th#964) the local CLI fold shares with the hub. The former local
walk (``resolve``/``resolve_local_bindings``) was deleted with pgw#515 —
locally, fit is the loading layer's job (runtime fp8/nf4 rungs + the
offload ladder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .lanes import is_w8a8_flavor
from .svdq import SVDQ_FP4_SMS, SVDQ_INT4_SMS

CLASS_BASE = "base"  # bare bf16/fp16/fp32 row — runs anywhere a card fits it
CLASS_FP8 = "fp8"  # fp8-E4M3 storage; universal (bf16-upcast path needs no fp8 silicon)
CLASS_SVDQ_FP4 = "svdq-fp4"  # nunchaku SVDQuant fp4 — consumer Blackwell only
CLASS_SVDQ_INT4 = "svdq-int4"  # nunchaku SVDQuant int4 — sm_75-89
CLASS_NVFP4 = "nvfp4"  # plain nvfp4 artifact — Blackwell datacenter, TRT lane (not a diffusers rung)
# Calibrated nvfp4 with two-level scales (gw#540): torch fp4 blockwise
# scaled_mm serve lane. Blackwell-only (sm_100+ incl. sm_120 consumer) —
# no fp4 silicon below, and the 4x dequant blow-up erases the fit story.
CLASS_NVFP4_W4A4 = "nvfp4-w4a4"

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


# Native fp8 tensor-core compute exists on SM >= 89 (sm_89 Ada, sm_90 Hopper,
# sm_100+/120 Blackwell). Below that, fp8 storage still SERVES (bf16-upcast
# path) — this floor gates only the fp8-over-bf16 PREFERENCE, never admission.
FP8_COMPUTE_MIN_SM = 89


# --- Family-root policy (th#964) — twin of tensorhub's modelfamily.Root ----

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


def family_root(family: str) -> str:
    """Architecture root of a family name; "" for empty. Unrecognized
    families root to their own normalized spelling."""
    n = str(family or "").strip().lower().replace(".", "").replace(" ", "")
    if not n:
        return ""
    return _FAMILY_ROOT_OVERRIDES.get(n, n)


# Conv-UNet roots get no fp8-GEMM win (torch scaled_mm is Linear-only;
# th#927 measured SDXL w8a8 1.9-2.7x slower than bf16): their fp8-w8a8 rows
# are AUTO-ineligible and the scale-free #fp8 row is the family table-best
# on sm_89+; bf16 stays the sub-floor default. Explicit pins still resolve
# w8a8. Twin of tensorhub precision.convUNetW8A8ExcludedRoots.
CONV_UNET_W8A8_EXCLUDED_ROOTS = frozenset({"sd1", "sd2", "sdxl"})


def w8a8_excluded_for_family(family: str) -> bool:
    """Whether AUTO selection policy-excludes fp8-w8a8 rows for this family
    (any spelling — rooted internally)."""
    return family_root(family) in CONV_UNET_W8A8_EXCLUDED_ROOTS


def pick_family_fp8_flavor(
    rows: Iterable[Any],
    *,
    model_family: str,
    gpu_sm: int,
    free_vram_gb: float,
    installed_libs: Sequence[str] = (),
) -> str:
    """th#964 AUTO pick over a resolve's sibling flavor rows: for conv-UNet
    families on sm_89+, the best scale-free #fp8 row (smallest token, hub
    tiebreak) when it fits free VRAM. "" = keep the declared binding (bf16
    sub-floor default, non-excluded family, or no admissible fitting row).
    """
    if not w8a8_excluded_for_family(model_family):
        return ""
    if gpu_sm < FP8_COMPUTE_MIN_SM:
        return ""
    libs = frozenset(installed_libs)
    candidates: list[tuple[int, str, int]] = []
    for row in rows:
        token = str(getattr(row, "flavor", "") or "").strip().lower()
        size = int(getattr(row, "size_bytes", 0) or 0)
        if size <= 0 or classify_flavor_token(token) != CLASS_FP8:
            continue
        if is_w8a8_flavor(token):
            continue
        placement = (
            placement_from_metadata(getattr(row, "placement", None))
            or default_placement(CLASS_FP8)
        )
        if placement is None or not placement.admits_sm(gpu_sm):
            continue
        if any(lib not in libs for lib in placement.engines):
            continue
        candidates.append((len(token), token, size))
    if not candidates:
        return ""
    # Hub walk gates only the single best fp8 rung on fit, then falls to bf16.
    candidates.sort()
    _, token, size = candidates[0]
    return token if size / 1e9 <= float(free_vram_gb) else ""


def maybe_rebind_family_fp8(
    binding: Any,
    *,
    resolved: Any,
    slot_family: str = "",
    gpu_sm: int = 0,
    free_vram_gb: float = 0.0,
    installed_libs: Sequence[str] = (),
) -> Any:
    """Fail-open local CLI fold (th#964): rebind a bare conv-UNet-family
    binding to its scale-free #fp8 sibling, matching the hub's AUTO pick.
    Family comes from the resolve's model_family, slot-declared family as
    fallback. Production resolution remains hub-owned."""
    family = (
        str(getattr(resolved, "model_family", "") or "").strip()
        or str(slot_family or "").strip()
    )
    try:
        flavor = pick_family_fp8_flavor(
            getattr(resolved, "sibling_flavors", ()) or (),
            model_family=family,
            gpu_sm=int(gpu_sm or 0),
            free_vram_gb=float(free_vram_gb or 0.0),
            installed_libs=tuple(installed_libs or ()),
        )
        if not flavor:
            return binding
        from ..api.binding import rebind_pick

        return rebind_pick(binding, flavor=flavor)
    except Exception:
        return binding

EMERGENCY_NF4_VRAM_FACTOR = 0.45  # nf4 denoiser, encoders/VAE at compute dtype

# Per-component resident-bytes factor for a bnb-nf4 quantized module vs its
# bf16/fp16 stored bytes (4-bit packed + double-quant + quant-state overhead).
# Used by the load-time fit ladder's per-component estimate (gw#521); the
# whole-model EMERGENCY_NF4_VRAM_FACTOR above stays the hub-side coarse spec.
NF4_WEIGHT_BYTES_FACTOR = 0.30


__all__ = [
    "FP8_COMPUTE_MIN_SM",
    "CLASS_BASE",
    "CLASS_FP8",
    "CLASS_NVFP4",
    "CLASS_NVFP4_W4A4",
    "CLASS_SVDQ_FP4",
    "CLASS_SVDQ_INT4",
    "CONV_UNET_W8A8_EXCLUDED_ROOTS",
    "EMERGENCY_NF4_VRAM_FACTOR",
    "NF4_WEIGHT_BYTES_FACTOR",
    "Placement",
    "classify_flavor_token",
    "default_placement",
    "family_root",
    "maybe_rebind_family_fp8",
    "pick_family_fp8_flavor",
    "placement_for_flavor",
    "placement_from_metadata",
    "placement_to_metadata",
    "w8a8_excluded_for_family",
]
