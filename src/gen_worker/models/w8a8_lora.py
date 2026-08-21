"""Runtime LoRA additive branches."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import activity as activity_mod
from ..component_vocab import denoiser_components
from ..api.errors import RefCompatibilitySurprise, ValidationError
from ..media_transfer import PIN_MAX_BYTES as _PIN_MAX_BYTES
from . import adapter_fidelity
from .fp8_storage import structural_base
from .w8a8 import fp8_scaled_linear_class
import inspect
from .loading import pipeline_weight_lane
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

RANK_BUCKETS = (16, 32, 64, 128)

_BUCKET_ATTR = "_cozy_lora_bucket"
_ACTIVE_ATTR = "_cozy_lora_active"
_SPARSE_ATTR = "_cozy_lora_sparse"
_MAPCACHE_ATTR = "_cozy_lora_mapcache"
_MAPCACHE_MAX = 8
_KOHYA_FLAT_PREFIXES = ("lora_unet_", "lora_transformer_")
_denoiser_components = denoiser_components


def _denoiser_prefixes() -> tuple[str, ...]:
    return tuple(f"{c}." for c in _denoiser_components()) + _KOHYA_FLAT_PREFIXES


def _component_prefixes() -> tuple[tuple[str, str], ...]:
    return tuple(
        (f"{c}.", c)
        for c in sorted(_denoiser_components(), key=len, reverse=True)
    )
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


def branch_targets(pipe: Any) -> Dict[str, Any]:
    """component name -> denoiser module for EVERY branch-capable denoiser the pipeline carries, in stamp order."""
    out: Dict[str, Any] = {}
    for name in _denoiser_components():
        denoiser = getattr(pipe, name, None)
        if denoiser is not None and hasattr(denoiser, "named_modules"):
            out[name] = denoiser
    return out


def declared_component(key: str) -> str:
    """The pipeline component one adapter key NAMES, or ``""`` when the key carries no component prefix (bare/kohya-flat module paths)."""
    for prefix, comp in _component_prefixes():
        if key.startswith(prefix):
            return comp
    return ""


def require_component_declaration(
    components: Iterable[str], raw_sd: Mapping[str, Any], *, ref: str = "",
) -> None:
    """Fail-closed, checked against the adapter's RAW keys."""
    comps = tuple(components)
    if len(comps) < 2:
        return
    if any(declared_component(k) for k in raw_sd):
        return
    raise RefCompatibilitySurprise(
        f"this pipeline has {len(comps)} denoiser experts "
        f"({', '.join(comps)}) and the adapter does not name the one it "
        "adapts — its keys carry no component prefix, and the diffusers "
        "converter would rewrite them all onto the high-noise expert. "
        "Publish per-expert adapters with their component prefix "
        "(e.g. 'transformer.blocks.0…' / 'transformer_2.blocks.0…')",
        ref=ref, axis="state_dict",
    )


def route_denoiser_keys(
    den_sd: Dict[str, Any], components: Iterable[str], *, ref: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Partition one adapter's denoiser keys by the component they target ."""
    comps = tuple(components)
    if not den_sd:
        return {}
    routed: Dict[str, Dict[str, Any]] = {}
    bare: Dict[str, Any] = {}
    foreign: Dict[str, str] = {}
    for key, tensor in den_sd.items():
        comp = declared_component(key)
        if comp and comp not in comps:
            foreign.setdefault(comp, key)
        elif not comp:
            bare[key] = tensor
        else:
            routed.setdefault(comp, {})[key] = tensor
    if not foreign and bare and len(comps) == 1:
        routed.setdefault(comps[0], {}).update(bare)
        bare = {}
    if bare and not foreign:
        raise RefCompatibilitySurprise(
            f"{len(bare)} adapter key(s) name no denoiser component "
            f"(e.g. {', '.join(sorted(bare)[:3])}) but this pipeline has "
            f"{len(comps)} experts ({', '.join(comps)}) — which one they "
            "adapt is ambiguous and guessing would silently leave an "
            "expert unadapted",
            ref=ref, axis="state_dict",
        )
    if foreign:
        detail = ", ".join(f"{c} (e.g. {k})" for c, k in sorted(foreign.items()))
        raise RefCompatibilitySurprise(
            f"adapter targets component(s) this pipeline does not carry: "
            f"{detail} — it has {', '.join(comps)}",
            ref=ref, axis="component_missing",
        )
    return routed


def branch_execution_lane(model: Any) -> str:
    """The denoiser's base weight lane for branch policy/stamping: ``"w8a8"`` | ``"fp8-hooks"`` | ``"gguf"`` | ``""`` (plain resident)."""
    if getattr(model, "_cozy_w8a8_mode", "") in ("rowwise", "pertensor"):
        return "w8a8"
    if getattr(model, "_cozy_fp8_storage_applied", False):
        return "fp8-hooks"
    from .gguf_torch import is_gguf_leaf

    if any(is_gguf_leaf(m) for _, m in model.named_modules()):
        return "gguf"
    return ""


def branch_modules(model: Any) -> Dict[str, Any]:
    """name -> branch-capable module for the denoiser: Fp8ScaledLinear, plain nn.Linear, or plain nn.Conv2d (the curated sdxl distill adapters carry conv pairs; convs are never quantized, so their branch ..."""
    import torch.nn as nn

    fp8_cls = fp8_scaled_linear_class()
    return {
        n: m for n, m in model.named_modules()
        if isinstance(m, fp8_cls) or structural_base(m) in (nn.Linear, nn.Conv2d)
    }


def branch_bucket(model: Any) -> int:
    """The enabled bucket, 0 when branches are not enabled."""
    return int(getattr(model, _BUCKET_ATTR, 0) or 0)


def lora_execution_lane(bucket: int, sparse: bool = False, base: str = "w8a8") -> str:
    prefix = f"{base}-" if base else ""
    return f"{prefix}lora{int(bucket)}" + ("-sparse" if sparse else "")


def split_state_dict(sd: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(denoiser keys, everything else)."""
    den: Dict[str, Any] = {}
    rest: Dict[str, Any] = {}
    for k, v in sd.items():
        (den if k.startswith(_denoiser_prefixes()) else rest)[k] = v
    return den, rest


def _base_and_kind(key: str) -> Tuple[str, str]:
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], "alpha"
    for suf in _DOWN_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], "down"
    for suf in _UP_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)], "up"
    if key.endswith(".weight"):
        stem = key[: -len(".weight")]
        head, _, _scope = stem.rpartition(".")
        if head.endswith(".lora_A"):
            return head[: -len(".lora_A")], "down"
        if head.endswith(".lora_B"):
            return head[: -len(".lora_B")], "up"
    return "", ""


def _kohya_sgm_normalize(sd: Dict[str, Any], model: Any) -> Optional[Dict[str, Any]]:
    try:
        from diffusers.loaders.lora_conversion_utils import (
            _maybe_map_sgm_blocks_to_diffusers,
        )

        mapped = _maybe_map_sgm_blocks_to_diffusers(dict(sd), model.config)
    except Exception:
        logger.warning("w8a8 lora: SGM block normalization failed", exc_info=True)
        return None
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
        for pref in ("unet.", "transformer_2.", "transformer."):
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
    """Resolve one adapter's denoiser keys onto the model's quantized modules: module path -> (A [r, in], B [out, r], alpha_scale)."""
    mods = branch_modules(model)
    groups, unresolved = _group_keys(den_sd, mods)
    if unresolved and any(
            p in k for k in den_sd
            for p in ("input_blocks", "middle_block", "output_blocks")):
        converted = _kohya_sgm_normalize(den_sd, model)
        if converted is not None:
            groups, unresolved = _group_keys(converted, mods)
    if unresolved:
        raise RefCompatibilitySurprise(
            f"{len(unresolved)} adapter key(s) target no branch-capable "
            f"Linear (e.g. {', '.join(sorted(unresolved)[:3])}) — the "
            "additive branch cannot apply this adapter without changing "
            "its output",
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
        a, b = _validated_pair(path, mod, a, b, ref=ref)
        alpha = g.get("alpha")
        alpha_scale = (float(alpha) / rank) if alpha is not None else 1.0
        out[path] = (a, b, alpha_scale)
    return out


def _validated_pair(path: str, mod: Any, a: Any, b: Any, *, ref: str) -> Tuple[Any, Any]:
    import torch.nn as nn

    def bad(detail: str) -> RefCompatibilitySurprise:
        return RefCompatibilitySurprise(
            f"adapter shapes for {path!r} do not match the base "
            f"({tuple(a.shape)}/{tuple(b.shape)}): {detail}",
            ref=ref, axis="state_dict",
        )

    rank = int(a.shape[0])
    if isinstance(mod, nn.Conv2d):
        if mod.groups != 1:
            raise bad(f"grouped conv (groups={mod.groups}) is not a branch target")
        if a.dim() != 4 or int(a.shape[1]) != mod.in_channels or (
                tuple(int(v) for v in a.shape[2:]) != tuple(mod.kernel_size)):
            raise bad(
                f"want down [r, {mod.in_channels}, "
                f"{mod.kernel_size[0]}, {mod.kernel_size[1]}]")
        if int(b.shape[0]) != mod.out_channels or int(b.shape[1]) != rank or (
                b.dim() not in (2, 4)) or (
                b.dim() == 4 and tuple(int(v) for v in b.shape[2:]) != (1, 1)):
            raise bad(f"want up [{mod.out_channels}, {rank}(, 1, 1)]")
        return a, b.reshape(mod.out_channels, rank, 1, 1)
    if a.dim() != 2 or b.dim() != 2 or int(a.shape[1]) != mod.in_features or (
            int(b.shape[0]) != mod.out_features or int(b.shape[1]) != rank):
        raise bad(f"want down [r, {mod.in_features}] / up [{mod.out_features}, r]")
    return a, b


def _stage_adapter(mapped: Dict[str, Tuple[Any, Any, float]]) -> Dict[str, Any]:
    import torch

    by_dtype: Dict[Any, List[Tuple[str, str, Any]]] = {}
    for path in sorted(mapped):
        a, b, _alpha = mapped[path]
        by_dtype.setdefault(a.dtype, []).append((path, "a", a))
        by_dtype.setdefault(b.dtype, []).append((path, "b", b))
    flat: Dict[Any, Any] = {}
    slices: Dict[Tuple[str, str], Tuple[Any, int, Tuple[int, ...]]] = {}
    for dt, items in by_dtype.items():
        total = sum(t.numel() for _p, _tag, t in items)
        pin = (cuda_ready()
               and total * items[0][2].element_size() <= _PIN_MAX_BYTES)
        buf = torch.empty(total, dtype=dt, pin_memory=pin)
        off = 0
        for path, tag, t in items:
            n = t.numel()
            buf[off:off + n].copy_(t.reshape(-1))
            slices[(path, tag)] = (dt, off, tuple(t.shape))
            off += n
        flat[dt] = buf
    index = {
        path: (slices[(path, "a")], slices[(path, "b")], float(alpha))
        for path, (_a, _b, alpha) in mapped.items()
    }
    ranks = {path: int(a.shape[0]) for path, (a, _b, _al) in mapped.items()}
    return {"ranks": ranks, "flat": flat, "index": index}


_WRAP_ATTR = "_cozy_lora_wrapped"


def _is_scaled_linear(mod: Any) -> bool:

    return isinstance(mod, fp8_scaled_linear_class())


def _install_branch_forward(mod: Any) -> None:
    if getattr(mod, _WRAP_ATTR, False):
        return
    orig = mod.forward

    def _forward_with_branch(x: Any, *args: Any, **kwargs: Any) -> Any:
        y = orig(x, *args, **kwargs)
        a = mod.__dict__.get("lora_a")
        b = mod.__dict__.get("lora_b")
        if a is None or b is None:
            return y
        if a.device != x.device:
            a2, b2 = a.to(x.device), b.to(x.device)
            if (mod.__dict__.get("lora_a") is a
                    and mod.__dict__.get("lora_b") is b):
                mod.lora_a, mod.lora_b = a2, b2
            a, b = a2, b2
        if a.dim() == 4:
            import torch.nn.functional as F

            x2 = x if x.dtype == a.dtype else x.to(a.dtype)
            h = F.conv2d(x2, a, stride=mod.stride, padding=mod.padding,
                         dilation=mod.dilation)
            return y + F.conv2d(h, b).to(y.dtype)
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != a.dtype:
            x2 = x2.to(a.dtype)
        addend = (x2 @ a.t()) @ b.t()
        return y + addend.reshape(*x.shape[:-1], b.shape[0]).to(y.dtype)

    mod._cozy_lora_orig_forward = orig
    mod.forward = _forward_with_branch
    setattr(mod, _WRAP_ATTR, True)


def _clear_branch_slots(mod: Any) -> None:
    if _is_scaled_linear(mod):
        mod._buffers["lora_a"] = None
        mod._buffers["lora_b"] = None
        return
    for name in ("lora_a", "lora_b"):
        mod._buffers.pop(name, None)
        mod.__dict__.pop(name, None)
    mod.lora_a = None
    mod.lora_b = None


def alloc_branch_buffers(mod: Any, bucket: int) -> None:
    """Zeroed A/B branch tensors on one branch-capable module."""
    import torch
    import torch.nn as nn

    dev = mod.weight.device
    dtype = adapter_fidelity.branch_compute_dtype(mod)
    _clear_branch_slots(mod)
    if isinstance(mod, nn.Conv2d):
        a = torch.zeros(bucket, mod.in_channels, *mod.kernel_size,
                        dtype=dtype, device=dev)
        b = torch.zeros(mod.out_channels, bucket, 1, 1,
                        dtype=dtype, device=dev)
        _install_branch_forward(mod)
        mod.lora_a = a
        mod.lora_b = b
        return
    a = torch.zeros(bucket, mod.in_features, dtype=dtype, device=dev)
    b = torch.zeros(mod.out_features, bucket, dtype=dtype, device=dev)
    if _is_scaled_linear(mod):
        mod.register_buffer("lora_a", a, persistent=False)
        mod.register_buffer("lora_b", b, persistent=False)
        return
    _install_branch_forward(mod)
    mod.lora_a = a
    mod.lora_b = b


def enable_lora_branches(model: Any, bucket: int) -> None:
    """Allocate branch buffers on EVERY branch-capable Linear (canonical placement — one traced graph over all coverage patterns; the compiled lane)."""
    if bucket not in RANK_BUCKETS:
        raise ValidationError(f"invalid lora rank bucket {bucket} (valid: {RANK_BUCKETS})")
    if branch_bucket(model) == bucket and not getattr(model, _SPARSE_ATTR, False):
        return
    for mod in branch_modules(model).values():
        alloc_branch_buffers(mod, bucket)
    setattr(model, _BUCKET_ATTR, int(bucket))
    setattr(model, _SPARSE_ATTR, False)
    setattr(model, _ACTIVE_ATTR, False)


def disable_lora_branches(model: Any) -> None:
    """Drop the branch buffers entirely (back to the branchless graph family)."""
    for mod in branch_modules(model).values():
        _clear_branch_slots(mod)
    if hasattr(model, _BUCKET_ATTR):
        delattr(model, _BUCKET_ATTR)
    setattr(model, _SPARSE_ATTR, False)
    setattr(model, _ACTIVE_ATTR, False)


def clear_branch_adapters(model: Any) -> None:
    """Deactivate."""
    if not branch_bucket(model):
        return
    if getattr(model, _SPARSE_ATTR, False):
        disable_lora_branches(model)
        return
    for mod in branch_modules(model).values():
        if getattr(mod, "lora_b", None) is not None:
            mod.lora_b.zero_()
    setattr(model, _ACTIVE_ATTR, False)


def branches_active(model: Any) -> bool:
    return bool(getattr(model, _ACTIVE_ATTR, False))


def enable_branch_execution_lanes(pipe: Any, bucket: int) -> Dict[str, Any]:
    """Allocate the rank-``bucket`` branch container on EVERY branch-capable denoiser (both experts of an MoE — the ``Compile(lora_bucket=)`` arming contract)."""
    targets = branch_targets(pipe)
    for model in targets.values():
        enable_lora_branches(model, bucket)
    return targets


def disable_branch_execution_lanes(pipe: Any) -> None:
    """Drop the branch containers from every denoiser (demote/teardown)."""
    for model in branch_targets(pipe).values():
        disable_lora_branches(model)


def clear_branch_execution_lanes(pipe: Any) -> None:
    """Deactivate every denoiser's branch (canonical: zero B; sparse: drop)."""
    for model in branch_targets(pipe).values():
        clear_branch_adapters(model)


def pipeline_branch_bucket(pipe: Any) -> int:
    """The pipeline's branch bucket — one value across the denoiser set (the set is always enabled/resized together), 0 when no branch is enabled."""
    return max(
        (branch_bucket(m) for m in branch_targets(pipe).values()), default=0)


def _stage_for(
    model: Any, adapters: Sequence[Tuple[Dict[str, Any], float, str]],
    *, request_id: str = "",
) -> List[Tuple[Dict[str, Any], float, str]]:
    cache: Dict[Any, Any] = getattr(model, _MAPCACHE_ATTR, None) or {}
    staged: List[Tuple[Dict[str, Any], float, str]] = []
    for sd, w, ref in adapters:
        key = (ref, id(sd), len(sd))
        entry = cache.get(key)
        fresh = entry is None
        if entry is None:
            mapped = map_adapter(sd, model, ref=ref)
            if not mapped:
                logger.error(
                    "adapter %s mapped ZERO modules on %s; the request will "
                    "render WITHOUT it", ref, type(model).__name__)
                activity_mod.emit_event(
                    activity_mod.KIND_LORA_FIDELITY,
                    f"ref={ref} model={type(model).__name__} "
                    f"keys={len(sd)}: the adapter mapped ZERO branch modules, "
                    f"so this request renders with NO adapter applied. The "
                    f"fidelity gate cannot see it — an empty mapping has "
                    f"nothing to measure — so it is reported here",
                    phase="not_applied",
                )
            entry = _stage_adapter(mapped)
            entry["survival"] = adapter_fidelity.evaluate_branch(
                mapped, branch_modules(model), ref=ref)
            cache[key] = entry
            while len(cache) > _MAPCACHE_MAX:
                cache.pop(next(iter(cache)))
        adapter_fidelity.gate(
            entry.get("survival"), request_id=request_id, announce=fresh)
        staged.append((entry, w, ref))
    setattr(model, _MAPCACHE_ATTR, cache)
    return staged


def _needed_rank(staged: Sequence[Tuple[Dict[str, Any], float, str]]) -> int:
    per_layer: Dict[str, int] = {}
    for entry, _w, _ref in staged:
        for path, r in entry["ranks"].items():
            per_layer[path] = per_layer.get(path, 0) + r
    return max(per_layer.values(), default=0)


def _settle_bucket(
    models: Sequence[Any], want: int, *, uniform: bool, allow_resize: bool,
) -> int:
    current = max((branch_bucket(m) for m in models), default=0)
    was_sparse = any(bool(getattr(m, _SPARSE_ATTR, False)) for m in models)
    if not uniform:
        return want
    if current >= want and current and not was_sparse:
        want = current
    if current != want or was_sparse:
        if not allow_resize:
            raise ValidationError(
                f"active LoRA set needs rank bucket {want} but the compiled "
                f"pipeline traced bucket {current or 'none'} — recompile at "
                "swap time is never allowed; publish a matching lora compiled graph"
            )
        for model in models:
            enable_lora_branches(model, want)
    return want


def _place_adapters(
    model: Any,
    staged: Sequence[Tuple[Dict[str, Any], float, str]],
    bucket: int,
    *,
    uniform: bool,
    current: int,
    t0: float,
) -> Dict[str, Any]:
    import torch

    mapped = list(staged)
    if not uniform:
        covered_paths: set[str] = set()
        for entry, _w, _ref in mapped:
            covered_paths.update(entry["ranks"])
        for path, mod in branch_modules(model).items():
            if path in covered_paths:
                if (getattr(mod, "lora_a", None) is None
                        or int(mod.lora_a.shape[0]) != bucket):
                    alloc_branch_buffers(mod, bucket)
            elif getattr(mod, "lora_a", None) is not None:
                _clear_branch_slots(mod)
        setattr(model, _BUCKET_ATTR, int(bucket))
        setattr(model, _SPARSE_ATTR, True)

    copied = 0
    covered = 0
    mods = branch_modules(model)
    with torch.no_grad():
        device = None
        for mod in mods.values():
            if getattr(mod, "lora_a", None) is not None:
                device = mod.lora_a.device
                break
        dev_flats: List[Dict[Any, Any]] = []
        for entry, _w, _ref in mapped:
            df = {dt: t.to(device, non_blocking=t.is_pinned())
                  for dt, t in entry["flat"].items()}
            dev_flats.append(df)
            copied += sum(t.numel() * t.element_size()
                          for t in entry["flat"].values())
        for path, mod in mods.items():
            if getattr(mod, "lora_a", None) is None:
                continue
            hit_any = False
            r0 = 0
            for (entry, w, _ref), df in zip(mapped, dev_flats):
                idx = entry["index"].get(path)
                if idx is None:
                    continue
                if not hit_any:
                    mod.lora_a.zero_()
                    mod.lora_b.zero_()
                    hit_any = True
                (dt_a, off_a, shp_a), (dt_b, off_b, shp_b), alpha_scale = idx
                r = shp_a[0]
                n_a = math.prod(shp_a)
                n_b = math.prod(shp_b)
                mod.lora_a[r0:r0 + r].copy_(
                    df[dt_a][off_a:off_a + n_a].view(shp_a))
                mod.lora_b[:, r0:r0 + r].copy_(
                    df[dt_b][off_b:off_b + n_b].view(shp_b))
                scale = alpha_scale * float(w)
                if scale != 1.0:
                    mod.lora_b[:, r0:r0 + r].mul_(scale)
                r0 += r
            if hit_any:
                covered += 1
            else:
                mod.lora_b.zero_()
        if cuda_ready():
            torch.cuda.synchronize()
    if mapped and not covered:
        raise RefCompatibilitySurprise(
            f"the adapter maps onto {len(mapped)} module set(s) of this "
            "denoiser but none of them carries a branch container — the "
            "adapter would apply to nothing; arm the lora bucket on every "
            "denoiser (compile_cache.apply_lora_lane) before attaching",
            axis="state_dict",
        )
    setattr(model, _ACTIVE_ATTR, True)
    return {
        "bucket": bucket, "resized": current != bucket,
        "sparse": not uniform, "covered": covered,
        "adapters": len(mapped),
        "modules": len(mods), "copied_bytes": copied,
        "swap_ms": int((time.monotonic() - t0) * 1000),
    }


def _idle_stats(model: Any) -> Dict[str, Any]:
    return {"bucket": branch_bucket(model), "resized": False, "covered": 0,
            "adapters": 0, "modules": 0, "copied_bytes": 0, "swap_ms": 0}


def apply_branch_adapters(
    model: Any,
    adapters: Sequence[Tuple[Dict[str, Any], float, str]],
    *,
    allow_resize: bool = True,
    uniform: bool = False,
    request_id: str = "",
    rank_floor: int = 0,
) -> Dict[str, Any]:
    """Make exactly ``adapters`` (state_dict, user weight, ref) ONE denoiser's active branch set."""
    t0 = time.monotonic()
    if not adapters:
        clear_branch_adapters(model)
        return _idle_stats(model)
    staged = _stage_for(model, adapters, request_id=request_id)
    current = branch_bucket(model)
    bucket = _settle_bucket(
        [model], rank_bucket(max(_needed_rank(staged), int(rank_floor), 1)),
        uniform=uniform, allow_resize=allow_resize)
    stats = _place_adapters(
        model, staged, bucket, uniform=uniform, current=current, t0=t0)
    logger.info(
        "[request_id=%s] w8a8 lora branch swap: adapters=%d bucket=%d "
        "covered=%d/%d copied_bytes=%d resized=%s swap_ms=%d",
        request_id, len(adapters), bucket, stats["covered"], stats["modules"],
        stats["copied_bytes"], stats["resized"], stats["swap_ms"],
    )
    return stats


def apply_branch_adapter_set(
    pipe: Any,
    routed: Mapping[str, Sequence[Tuple[Dict[str, Any], float, str]]],
    *,
    allow_resize: bool = True,
    uniform: bool = False,
    request_id: str = "",
) -> Dict[str, Any]:
    """Make exactly ``routed`` (component -> adapters) the PIPELINE's active branch set."""
    t0 = time.monotonic()
    targets = branch_targets(pipe)
    unknown = sorted(c for c in routed if c not in targets)
    if unknown:
        raise RefCompatibilitySurprise(
            f"adapter set targets component(s) {', '.join(unknown)} which "
            f"this pipeline does not carry (it has "
            f"{', '.join(targets) or 'no branch-capable denoiser'})",
            axis="component_missing",
        )
    staged: Dict[str, List[Tuple[Dict[str, Any], float, str]]] = {}
    for comp, model in targets.items():
        entries = list(routed.get(comp) or ())
        staged[comp] = (
            _stage_for(model, entries, request_id=request_id) if entries else [])
    want = max((_needed_rank(s) for s in staged.values()), default=0)
    pre = {comp: branch_bucket(model) for comp, model in targets.items()}
    current = max(pre.values(), default=0)
    bucket = current
    if want:
        bucket = _settle_bucket(
            list(targets.values()), rank_bucket(max(want, 1)),
            uniform=uniform, allow_resize=allow_resize)
    if uniform:
        bare = [c for c, s in staged.items() if s and not branch_bucket(targets[c])]
        if bare:
            raise RefCompatibilitySurprise(
                f"component(s) {', '.join(sorted(bare))} carry no branch "
                f"container at bucket {bucket} while the pipeline is "
                "compiled — every denoiser an adapter set targets must be "
                "armed (Compile(lora_bucket=...) arms them all; declare each "
                "expert as a compile target too)",
                axis="state_dict",
            )
    per: Dict[str, Dict[str, Any]] = {}
    for comp, model in targets.items():
        if not staged[comp]:
            clear_branch_adapters(model)
            per[comp] = _idle_stats(model)
            continue
        per[comp] = _place_adapters(
            model, staged[comp], bucket, uniform=uniform,
            current=pre[comp], t0=t0)
    stats: Dict[str, Any] = {
        "bucket": bucket, "resized": current != bucket,
        "sparse": not uniform,
        "adapters": sum(p["adapters"] for p in per.values()),
        "covered": sum(p["covered"] for p in per.values()),
        "modules": sum(p["modules"] for p in per.values()),
        "copied_bytes": sum(p["copied_bytes"] for p in per.values()),
        "swap_ms": int((time.monotonic() - t0) * 1000),
        "components": per,
    }
    logger.info(
        "[request_id=%s] w8a8 lora branch swap: adapters=%d bucket=%d "
        "covered=%d/%d copied_bytes=%d resized=%s swap_ms=%d components=%s",
        request_id, stats["adapters"], bucket, stats["covered"],
        stats["modules"], stats["copied_bytes"], stats["resized"],
        stats["swap_ms"],
        " ".join(f"{c}:{p['adapters']}a/{p['covered']}m"
                 for c, p in per.items()),
    )
    return stats


def effective_base_execution_lane(pipe: Any) -> str:
    """The branchless base weight lane COMPILED GRAPH IDENTITY rides on — the ONE resolution :func:`stamp_lane` memoizes: the memoized base, else the pipeline's stamped/probed lane, else the denoiser's ow..."""
    base = getattr(pipe, "_cozy_lora_base_lane", None)
    if base is not None:
        return str(base)

    execution_lane = pipeline_weight_lane(pipe)
    if execution_lane:
        return execution_lane
    for model in branch_targets(pipe).values():
        return branch_execution_lane(model)
    return ""


def stamp_execution_lane(pipe: Any, targets: Optional[Mapping[str, Any]] = None) -> None:
    """Keep the compile-cache graph key honest: branch-bearing pipelines are a different graph family per (base lane, bucket) — lane_drift guards both directions."""
    tg = branch_targets(pipe) if targets is None else dict(targets)
    models = list(tg.values())
    bucket = max((branch_bucket(m) for m in models), default=0)
    sparse = any(bool(getattr(m, _SPARSE_ATTR, False)) for m in models)
    base = getattr(pipe, "_cozy_lora_base_lane", None)
    if base is None:
        base = effective_base_execution_lane(pipe)
        try:
            pipe._cozy_lora_base_lane = base
        except Exception:
            return
    try:
        pipe._cozy_weight_lane = lora_execution_lane(bucket, sparse, base=base) if bucket else base
    except Exception:
        pass


def _accepts_unet_config(fn: Any) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "unet_config" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())


def _unresolved_count(pipe: Any, sd: Dict[str, Any]) -> Optional[int]:
    try:
        den, _rest = split_state_dict(sd)
        if not den:
            return None
        model = next(
            (m for m in (getattr(pipe, c, None) for c in _denoiser_components())
             if m is not None and hasattr(m, "named_modules")), None)
        if model is None:
            return None
        mods = branch_modules(model)
        if not mods:
            return None
        _groups, unresolved = _group_keys(den, mods)
        return len(unresolved)
    except Exception:  # noqa: BLE001 — a probe never breaks an overlay
        logger.debug("adapter resolvability probe failed", exc_info=True)
        return None


def _converted_resolves_at_least_as_well(
    pipe: Any, raw: Dict[str, Any], converted: Dict[str, Any], *, ref: str = "",
) -> bool:
    after = _unresolved_count(pipe, converted)
    if after in (None, 0):
        return True
    before = _unresolved_count(pipe, raw)
    if before is None or after <= before:
        return True
    logger.info(
        "lora_state_dict normalization for %s left %d unresolved denoiser "
        "key(s) against this model where the RAW keys leave %d; using raw "
        "keys (pgw#566)", ref, after, before,
    )
    return False


def normalize_adapter_state_dict(
    pipe: Any, sd: Dict[str, Any], *, ref: str = ""
) -> Dict[str, Any]:
    """Normalize a raw adapter through the pipeline class's own ``lora_state_dict`` converter (zero drift: byte-identical key handling with the boot-time ``load_lora_weights`` path)."""

    fn = getattr(type(pipe), "lora_state_dict", None)
    if fn is None:
        return sd
    kwargs: Dict[str, Any] = {}
    unet = getattr(pipe, "unet", None)
    if unet is not None and hasattr(unet, "config"):
        if _accepts_unet_config(fn):
            kwargs["unet_config"] = unet.config
    try:
        converted = fn(dict(sd), **kwargs)
    except Exception:
        logger.warning(
            "lora_state_dict normalization failed for %s; using raw keys",
            ref, exc_info=True,
        )
        return sd
    alphas: Dict[str, Any] = {}
    if isinstance(converted, tuple):
        if len(converted) != 2:
            logger.warning(
                "%s.lora_state_dict returned a %d-tuple; using raw keys",
                type(pipe).__name__, len(converted),
            )
            return sd
        converted, raw_alphas = converted
        alphas = dict(raw_alphas or {})
    if not _converted_resolves_at_least_as_well(pipe, sd, converted, ref=ref):
        return sd
    if any(".processor." in k for k in converted):
        logger.info(
            "lora_state_dict normalization for %s emitted legacy "
            "attn-processor names; using raw keys", ref,
        )
        return sd
    out = dict(converted)
    for k, v in alphas.items():
        key = k if k.endswith(".alpha") else f"{k}.alpha"
        out.setdefault(key, v)
    return out


def apply_lora_execution_lane(pipe: Any, bucket: int) -> bool:
    """Put the pipeline on the branch-bearing graph family for ``bucket`` (gw#561): canonical zeroed rank-``bucket`` branches on every branch-capable denoiser Linear (the gw#547 compiled-lane contract) + ..."""
    if not bucket:
        return False
    targets = enable_branch_execution_lanes(pipe, int(bucket))
    if not targets:
        raise RuntimeError(
            "a lora bucket was declared but the pipeline has no branch-capable "
            "denoiser (transformer/transformer_2/unet)"
        )
    stamp_execution_lane(pipe, targets)
    return True


def drop_lora_execution_lane(pipe: Any) -> None:
    """Undo :func:`apply_lora_execution_lane`: drop the branch buffers on every denoiser and restore the branchless lane stamp (the eager rollback — canonical zeroed branches cost +21-32% eager, gw#547)."""
    targets = branch_targets(pipe)
    if not targets:
        return
    disable_branch_execution_lanes(pipe)
    stamp_execution_lane(pipe, targets)


__all__ = [
    "apply_lora_execution_lane",
    "drop_lora_execution_lane",
    "RANK_BUCKETS",
    "apply_branch_adapter_set",
    "apply_branch_adapters",
    "branch_bucket",
    "branch_execution_lane",
    "branch_modules",
    "branch_targets",
    "branches_active",
    "clear_branch_adapters",
    "clear_branch_execution_lanes",
    "declared_component",
    "disable_branch_execution_lanes",
    "disable_lora_branches",
    "enable_branch_execution_lanes",
    "enable_lora_branches",
    "lora_execution_lane",
    "map_adapter",
    "normalize_adapter_state_dict",
    "pipeline_branch_bucket",
    "rank_bucket",
    "require_component_declaration",
    "route_denoiser_keys",
    "split_state_dict",
    "stamp_execution_lane",
]
