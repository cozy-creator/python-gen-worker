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
_SPARSE_ATTR = "_cozy_lora_sparse"
_MAPCACHE_ATTR = "_cozy_lora_mapcache"
_MAPCACHE_MAX = 8
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


def lora_lane(bucket: int, sparse: bool = False) -> str:
    # Sparse (eager-only) placement is a different graph per coverage
    # pattern — the "-sparse" suffix can never match a produced cell label.
    return f"w8a8-lora{int(bucket)}" + ("-sparse" if sparse else "")


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


def _kohya_sgm_normalize(sd: Dict[str, Any], model: Any) -> Optional[Dict[str, Any]]:
    """Rename SGM/LDM block indices (input_blocks_4_1 ...) to diffusers block
    paths using the REAL unet config — diffusers' own pre-pass, the same one
    the bf16 peft path runs. Keys stay kohya-flat afterwards and resolve
    against the model's module paths directly (down/up/alpha handled
    natively; the full non-diffusers converter is NOT used — it emits legacy
    attn-processor names that match no real module)."""
    try:
        from diffusers.loaders.lora_conversion_utils import (
            _maybe_map_sgm_blocks_to_diffusers,
        )

        mapped = _maybe_map_sgm_blocks_to_diffusers(dict(sd), model.config)
    except Exception:
        logger.warning("w8a8 lora: SGM block normalization failed", exc_info=True)
        return None
    # The SGM pass renumbers blocks but keeps the sgm family names; the
    # family rename is _convert_unet_lora_key's job in diffusers — do just
    # that part here.
    return {
        k.replace("input_blocks", "down_blocks")
         .replace("middle_block", "mid_block")
         .replace("output_blocks", "up_blocks"): v
        for k, v in mapped.items()
    }


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
    if unresolved and any(
            p in k for k in den_sd
            for p in ("input_blocks", "middle_block", "output_blocks")):
        converted = _kohya_sgm_normalize(den_sd, model)
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
    placement — one traced graph over all coverage patterns; the compiled
    lane). Idempotent at the same bucket; a different bucket reallocates
    (a new graph family)."""
    if bucket not in RANK_BUCKETS:
        raise ValidationError(f"invalid lora rank bucket {bucket} (valid: {RANK_BUCKETS})")
    if branch_bucket(model) == bucket and not getattr(model, _SPARSE_ATTR, False):
        return
    for mod in quantized_modules(model).values():
        alloc_branch_buffers(mod, bucket)
    setattr(model, _BUCKET_ATTR, int(bucket))
    setattr(model, _SPARSE_ATTR, False)
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
    setattr(model, _SPARSE_ATTR, False)
    setattr(model, _ACTIVE_ATTR, False)


def clear_branch_adapters(model: Any) -> None:
    """Deactivate. Canonical (compiled) placement zeroes B — the addend is
    exactly 0 and the traced graph stays. Sparse (eager) placement DROPS the
    buffers instead: eager pays per-kernel launch cost even for zeroed
    branches, so bare requests go back to exactly branchless speed."""
    if not branch_bucket(model):
        return
    if getattr(model, _SPARSE_ATTR, False):
        disable_lora_branches(model)
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
    uniform: bool = False,
    request_id: str = "",
) -> Dict[str, Any]:
    """Make exactly ``adapters`` (state_dict, user weight, ref) the denoiser's
    active branch set. Rank-concat across adapters, pad to the bucket, fold
    ``alpha/rank * weight`` into the B copy. Returns swap stats.

    ``uniform=True`` (compiled pipelines) keeps canonical placement — every
    quantized Linear carries a branch, zeroed slots for uncovered layers,
    never shrinking the bucket; ``allow_resize=False`` additionally refuses
    bucket changes (a resize is a new graph family, and prod never compiles
    at runtime). ``uniform=False`` (eager) allocates branches ONLY on
    covered layers and drops stale ones — eager pays a per-kernel launch
    tax even for zeroed slots, so sparse placement keeps uncovered layers
    at exactly branchless speed."""
    import torch

    t0 = time.monotonic()
    if not adapters:
        clear_branch_adapters(model)
        return {"bucket": branch_bucket(model), "resized": False, "covered": 0,
                "modules": 0, "copied_bytes": 0, "swap_ms": 0}
    # Mapping is pure in the state dict; repeat swaps of a resident adapter
    # (the AdapterCache serves the SAME dict object) skip the key-mapping /
    # kohya-conversion pass entirely.
    cache: Dict[Any, Any] = getattr(model, _MAPCACHE_ATTR, None) or {}
    mapped = []
    for sd, w, ref in adapters:
        key = (ref, id(sd), len(sd))
        m = cache.get(key)
        if m is None:
            m = map_adapter(sd, model, ref=ref)
            cache[key] = m
            while len(cache) > _MAPCACHE_MAX:
                cache.pop(next(iter(cache)))
        mapped.append((m, w, ref))
    setattr(model, _MAPCACHE_ATTR, cache)
    per_layer: Dict[str, int] = {}
    for m, _w, _ref in mapped:
        for path, (a, _b, _s) in m.items():
            per_layer[path] = per_layer.get(path, 0) + int(a.shape[0])
    needed = max(per_layer.values(), default=0)
    bucket = rank_bucket(max(needed, 1))
    current = branch_bucket(model)
    was_sparse = bool(getattr(model, _SPARSE_ATTR, False))
    if uniform:
        if current >= bucket and current and not was_sparse:
            bucket = current  # never shrink — stay on the already-traced graph
        if current != bucket or was_sparse:
            if not allow_resize:
                raise ValidationError(
                    f"active LoRA set needs rank bucket {bucket} but the compiled "
                    f"pipeline traced bucket {current or 'none'} — recompile at "
                    "swap time is never allowed; publish a matching lora cell"
                )
            enable_lora_branches(model, bucket)
    else:
        covered_paths: set[str] = set()
        for m, _w, _ref in mapped:
            covered_paths.update(m)
        for path, mod in quantized_modules(model).items():
            if path in covered_paths:
                if (getattr(mod, "lora_a", None) is None
                        or int(mod.lora_a.shape[0]) != bucket):
                    alloc_branch_buffers(mod, bucket)
            elif getattr(mod, "lora_a", None) is not None:
                for name in ("lora_a", "lora_b"):
                    mod._buffers.pop(name, None)
                    mod.__dict__.pop(name, None)
                mod.lora_a = None
                mod.lora_b = None
        setattr(model, _BUCKET_ATTR, int(bucket))
        setattr(model, _SPARSE_ATTR, True)

    copied = 0
    covered = 0
    mods = quantized_modules(model)
    with torch.no_grad():
        # Stage on CPU, ship in ONE H2D transfer, then device-side slice
        # copies — 2x739 individual pageable copies measured seconds at SDXL
        # scale; one flat transfer is tens of ms.
        plan: List[Tuple[Any, Any, Any]] = []
        stage_dtype = None
        for path, mod in mods.items():
            if getattr(mod, "lora_a", None) is None:
                continue  # sparse placement: uncovered layer has no branch
            rows: List[Any] = []
            cols: List[Any] = []
            for m, weight, _ref in mapped:
                hit = m.get(path)
                if hit is None:
                    continue
                a, b, alpha_scale = hit
                rows.append(a)
                cols.append(b.float() * (alpha_scale * weight))
            if not rows:
                mod.lora_b.zero_()
                continue
            if stage_dtype is None:
                stage_dtype = mod.lora_a.dtype
            a_cat = torch.cat(rows, dim=0).to(stage_dtype)
            b_cat = torch.cat(cols, dim=1).to(stage_dtype)
            plan.append((mod, a_cat, b_cat))
            covered += 1
        if plan:
            device = plan[0][0].lora_a.device
            flat = torch.cat(
                [t.reshape(-1) for _mod, a, b in plan for t in (a, b)])
            copied = flat.numel() * flat.element_size()
            flat = flat.to(device, non_blocking=True)
            off = 0
            for mod, a, b in plan:
                r = int(a.shape[0])
                mod.lora_a.zero_()
                mod.lora_b.zero_()
                n = a.numel()
                mod.lora_a[:r].copy_(flat[off:off + n].view_as(a))
                off += n
                n = b.numel()
                mod.lora_b[:, :r].copy_(flat[off:off + n].view_as(b))
                off += n
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    setattr(model, _ACTIVE_ATTR, True)
    stats = {
        "bucket": bucket, "resized": current != bucket,
        "sparse": not uniform, "covered": covered,
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
    sparse = bool(getattr(model, _SPARSE_ATTR, False))
    try:
        pipe._cozy_weight_lane = lora_lane(bucket, sparse) if bucket else "w8a8"
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
