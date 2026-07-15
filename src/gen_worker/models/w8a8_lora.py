"""Runtime LoRA on the w8a8 serve lane (gw#547).

Fp8ScaledLinear holds fp8 buffers + torch._scaled_mm — peft can't target it,
so adapters ride a bf16 SIDE-BRANCH instead: ``y += B(A @ x)`` reading the
original bf16 activation and adding onto the bf16 output. Quantized weights
are never touched; there is no upcast/requant round-trip.

Graph stability: every quantized Linear gets a branch (canonical placement —
zeroed slots for layers an adapter doesn't cover), and the concatenated rank
of the active adapter set is padded to a fixed bucket (:data:`RANK_BUCKETS`).
Every adapter set inside one bucket shares ONE traced graph; hot-swap is a
buffer copy. Multiple active adapters rank-concat into one A/B pair (the
gw#430 svdq trick); per-adapter scale (alpha/rank x user weight) is folded
into the B copy.

The branch-bearing pipeline stamps ``_cozy_weight_lane = "w8a8-lora<bucket>"``
so the existing SYMMETRIC ``compile_cache.lane_drift`` guard keeps LoRA-bearing
pipelines and branchless compile cells apart in both directions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..api.errors import RefCompatibilitySurprise, ValidationError

logger = logging.getLogger(__name__)

# Padded-rank buckets: one traced graph per bucket. 16/32 cover the bulk of
# the civitai catalog; 64/128 the high-rank tail (survey follow-up in gw#547).
RANK_BUCKETS = (16, 32, 64, 128)

_BUCKET_ATTR = "_cozy_lora_bucket"
_ACTIVE_ATTR = "_cozy_lora_active"
_DENOISER_PREFIXES = ("unet.", "transformer.", "lora_unet_", "lora_transformer_")
# (normalized suffix marker, is_down) — dotted forms after key normalization.
_DOWN_SUFFIXES = (".lora_down.weight", ".lora.down.weight", ".lora_A.weight")
_UP_SUFFIXES = (".lora_up.weight", ".lora.up.weight", ".lora_B.weight")


def rank_bucket(total_rank: int) -> int:
    """Smallest bucket covering ``total_rank``."""
    for b in RANK_BUCKETS:
        if total_rank <= b:
            return b
    raise RefCompatibilitySurprise(
        f"active LoRA set needs rank {total_rank} > max bucket {RANK_BUCKETS[-1]}",
        axis="state_dict",
    )


def branch_target(pipe: Any) -> Optional[Any]:
    """The pipeline's denoiser when it serves on the scaled_mm w8a8 lane
    (branch-capable), else None (plain/dequant pipelines keep the peft path)."""
    for name in ("transformer", "unet"):
        denoiser = getattr(pipe, name, None)
        if denoiser is not None and getattr(denoiser, "_cozy_w8a8_mode", "") == "scaled_mm":
            return denoiser
    return None


def quantized_modules(model: Any) -> Dict[str, Any]:
    """name -> Fp8ScaledLinear for every quantized Linear in the denoiser."""
    from .w8a8 import fp8_scaled_linear_class

    cls = fp8_scaled_linear_class()
    return {n: m for n, m in model.named_modules() if isinstance(m, cls)}


def branch_bucket(model: Any) -> int:
    """The enabled bucket, 0 when branches are not enabled."""
    return int(getattr(model, _BUCKET_ATTR, 0) or 0)


def lora_lane(bucket: int) -> str:
    return f"w8a8-lora{int(bucket)}"


def split_state_dict(sd: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(denoiser keys, everything else). Text-encoder halves stay on the
    peft path — TEs are not quantized."""
    den: Dict[str, Any] = {}
    rest: Dict[str, Any] = {}
    for k, v in sd.items():
        (den if k.startswith(_DENOISER_PREFIXES) else rest)[k] = v
    return den, rest


def _base_and_kind(key: str) -> Tuple[str, str]:
    """(base module name, 'down'|'up'|'alpha'|'') for one adapter key.
    Handles adapter-scoped peft keys (``.lora_A.<name>.weight``)."""
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], "alpha"
    for suf in _DOWN_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], "down"
    for suf in _UP_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], "up"
    # adapter-scoped peft: ...lora_A.<adapter>.weight
    if key.endswith(".weight"):
        stem = key[: -len(".weight")]
        head, _, _scope = stem.rpartition(".")
        if head.endswith(".lora_A"):
            return head[: -len(".lora_A")], "down"
        if head.endswith(".lora_B"):
            return head[: -len(".lora_B")], "up"
    return "", ""


def _kohya_to_diffusers(sd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a kohya/sd-scripts adapter (incl. LDM block naming) through
    diffusers' own converter — the exact mapping the bf16 peft path uses.
    None when the converter can't handle it (caller falls back / fails)."""
    try:
        from diffusers.loaders.lora_conversion_utils import (
            _convert_non_diffusers_lora_to_diffusers,
        )

        converted, alphas = _convert_non_diffusers_lora_to_diffusers(dict(sd))
    except Exception:
        return None
    out = dict(converted)
    out.update(alphas or {})  # "<base>.alpha" -> float
    return out


def _group_keys(
    den_sd: Dict[str, Any], mods: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    flat = {p.replace(".", "_"): p for p in mods}

    def resolve(base: str) -> str:
        for pref in ("unet.", "transformer."):
            if base.startswith(pref) and base[len(pref):] in mods:
                return base[len(pref):]
        for pref in ("lora_unet_", "lora_transformer_"):
            if base.startswith(pref) and base[len(pref):] in flat:
                return flat[base[len(pref):]]
        if base in mods:
            return base
        return ""

    groups: Dict[str, Dict[str, Any]] = {}
    unresolved: List[str] = []
    for key, tensor in den_sd.items():
        base, kind = _base_and_kind(key)
        path = resolve(base) if kind else ""
        if not path:
            unresolved.append(key)
            continue
        groups.setdefault(path, {})[kind] = tensor
    return groups, unresolved


def map_adapter(
    den_sd: Dict[str, Any], model: Any, *, ref: str = ""
) -> Dict[str, Tuple[Any, Any, float]]:
    """Resolve one adapter's denoiser keys onto the model's quantized
    modules: module path -> (A [r, in], B [out, r], alpha_scale).

    Dotted diffusers/peft names resolve after stripping the component
    prefix; kohya flattened names resolve against the model's own module
    paths, falling back to diffusers' kohya converter (LDM block naming,
    SDXL sd-scripts). Any key that does not land on a branch-capable module
    is a hard error — a silently-dropped block would change the adapter's
    output."""
    mods = quantized_modules(model)
    groups, unresolved = _group_keys(den_sd, mods)
    if unresolved and any(k.startswith("lora_unet_") for k in den_sd):
        converted = _kohya_to_diffusers(den_sd)
        if converted is not None:
            groups, unresolved = _group_keys(converted, mods)
    if unresolved:
        raise RefCompatibilitySurprise(
            f"{len(unresolved)} adapter key(s) target no w8a8-quantized module "
            f"(e.g. {', '.join(sorted(unresolved)[:3])}) — the w8a8 branch "
            "cannot apply this adapter without changing its output",
            ref=ref, axis="state_dict",
        )

    out: Dict[str, Tuple[Any, Any, float]] = {}
    for path, g in groups.items():
        a, b = g.get("down"), g.get("up")
        if a is None or b is None:
            raise RefCompatibilitySurprise(
                f"adapter is missing the down/up pair for {path!r}",
                ref=ref, axis="state_dict",
            )
        rank = int(a.shape[0])
        mod = mods[path]
        if int(a.shape[1]) != mod.in_features or int(b.shape[0]) != mod.out_features:
            raise RefCompatibilitySurprise(
                f"adapter shapes for {path!r} do not match the base "
                f"({tuple(a.shape)}/{tuple(b.shape)} vs "
                f"in={mod.in_features} out={mod.out_features})",
                ref=ref, axis="state_dict",
            )
        alpha = g.get("alpha")
        alpha_scale = (float(alpha) / rank) if alpha is not None else 1.0
        out[path] = (a, b, alpha_scale)
    return out


def alloc_branch_buffers(mod: Any, bucket: int) -> None:
    """Zeroed A/B buffers on one Fp8ScaledLinear (persistent=False — they
    move with the module, never enter state_dict)."""
    import torch

    dev = mod.weight.device
    dtype = mod.bias.dtype if mod.bias is not None else torch.bfloat16
    for name in ("lora_a", "lora_b"):
        mod._buffers.pop(name, None)
        mod.__dict__.pop(name, None)
    mod.register_buffer("lora_a", torch.zeros(
        bucket, mod.in_features, dtype=dtype, device=dev), persistent=False)
    mod.register_buffer("lora_b", torch.zeros(
        mod.out_features, bucket, dtype=dtype, device=dev), persistent=False)


def enable_lora_branches(model: Any, bucket: int) -> None:
    """Allocate branch buffers on EVERY quantized Linear (canonical
    placement). Idempotent at the same bucket; a different bucket
    reallocates (a new graph family)."""
    if bucket not in RANK_BUCKETS:
        raise ValidationError(f"invalid lora rank bucket {bucket} (valid: {RANK_BUCKETS})")
    if branch_bucket(model) == bucket:
        return
    for mod in quantized_modules(model).values():
        alloc_branch_buffers(mod, bucket)
    setattr(model, _BUCKET_ATTR, int(bucket))
    setattr(model, _ACTIVE_ATTR, False)


def disable_lora_branches(model: Any) -> None:
    """Drop the branch buffers entirely (back to the branchless graph
    family). Used on demote/teardown, never between requests."""
    for mod in quantized_modules(model).values():
        for name in ("lora_a", "lora_b"):
            mod._buffers.pop(name, None)
            mod.__dict__.pop(name, None)
        mod.lora_a = None
        mod.lora_b = None
    if hasattr(model, _BUCKET_ATTR):
        delattr(model, _BUCKET_ATTR)
    setattr(model, _ACTIVE_ATTR, False)


def clear_branch_adapters(model: Any) -> None:
    """Zero the active set (B only — the addend is exactly 0). Buffers and
    the traced graph stay; the next swap is still just a copy."""
    if not branch_bucket(model):
        return
    for mod in quantized_modules(model).values():
        if getattr(mod, "lora_b", None) is not None:
            mod.lora_b.zero_()
    setattr(model, _ACTIVE_ATTR, False)


def branches_active(model: Any) -> bool:
    return bool(getattr(model, _ACTIVE_ATTR, False))


def apply_branch_adapters(
    model: Any,
    adapters: Sequence[Tuple[Dict[str, Any], float, str]],
    *,
    allow_resize: bool = True,
    request_id: str = "",
) -> Dict[str, Any]:
    """Make exactly ``adapters`` (state_dict, user weight, ref) the denoiser's
    active branch set. Rank-concat across adapters, pad to the bucket, fold
    ``alpha/rank * weight`` into the B copy. Returns swap stats.

    ``allow_resize=False`` (compiled pipelines) refuses bucket changes —
    a resize is a new graph family, and prod never compiles at runtime."""
    import torch

    t0 = time.monotonic()
    if not adapters:
        clear_branch_adapters(model)
        return {"bucket": branch_bucket(model), "resized": False, "covered": 0,
                "modules": 0, "copied_bytes": 0, "swap_ms": 0}
    mapped = [(map_adapter(sd, model, ref=ref), w, ref) for sd, w, ref in adapters]
    per_layer: Dict[str, int] = {}
    for m, _w, _ref in mapped:
        for path, (a, _b, _s) in m.items():
            per_layer[path] = per_layer.get(path, 0) + int(a.shape[0])
    needed = max(per_layer.values(), default=0)
    bucket = rank_bucket(max(needed, 1))
    current = branch_bucket(model)
    if current >= bucket and current:
        bucket = current  # never shrink — stay on the already-traced graph
    if current != bucket:
        if not allow_resize:
            raise ValidationError(
                f"active LoRA set needs rank bucket {bucket} but the compiled "
                f"pipeline traced bucket {current or 'none'} — recompile at "
                "swap time is never allowed; publish a matching lora cell"
            )
        enable_lora_branches(model, bucket)

    copied = 0
    covered = 0
    mods = quantized_modules(model)
    with torch.no_grad():
        for path, mod in mods.items():
            rows: List[Any] = []
            cols: List[Any] = []
            for m, weight, _ref in mapped:
                hit = m.get(path)
                if hit is None:
                    continue
                a, b, alpha_scale = hit
                rows.append(a.to(dtype=mod.lora_a.dtype))
                cols.append(b.to(dtype=mod.lora_b.dtype) * (alpha_scale * weight))
            if not rows:
                mod.lora_b.zero_()
                continue
            a_cat = torch.cat(rows, dim=0)
            b_cat = torch.cat(cols, dim=1)
            r = int(a_cat.shape[0])
            mod.lora_a.zero_()
            mod.lora_b.zero_()
            mod.lora_a[:r].copy_(a_cat.to(mod.lora_a.device), non_blocking=True)
            mod.lora_b[:, :r].copy_(b_cat.to(mod.lora_b.device), non_blocking=True)
            copied += a_cat.numel() * a_cat.element_size()
            copied += b_cat.numel() * b_cat.element_size()
            covered += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    setattr(model, _ACTIVE_ATTR, True)
    stats = {
        "bucket": bucket, "resized": current != bucket, "covered": covered,
        "modules": len(mods), "copied_bytes": copied,
        "swap_ms": int((time.monotonic() - t0) * 1000),
    }
    logger.info(
        "[request_id=%s] w8a8 lora branch swap: adapters=%d bucket=%d "
        "covered=%d/%d copied_bytes=%d resized=%s swap_ms=%d",
        request_id, len(adapters), bucket, covered, len(mods), copied,
        stats["resized"], stats["swap_ms"],
    )
    return stats


def stamp_lane(pipe: Any, model: Any) -> None:
    """Keep the compile-cache graph key honest: branch-bearing pipelines are
    a different graph family per bucket (lane_drift guards both directions)."""
    bucket = branch_bucket(model)
    try:
        pipe._cozy_weight_lane = lora_lane(bucket) if bucket else "w8a8"
    except Exception:
        pass


__all__ = [
    "RANK_BUCKETS",
    "apply_branch_adapters",
    "branch_bucket",
    "branch_target",
    "branches_active",
    "clear_branch_adapters",
    "disable_lora_branches",
    "enable_lora_branches",
    "lora_lane",
    "map_adapter",
    "quantized_modules",
    "rank_bucket",
    "split_state_dict",
    "stamp_lane",
]
