"""Runtime LoRA additive branches (gw#547 w8a8; gw#558 lane-general;
gw#627 Conv2d).

Denoiser adapter halves ride a compute-dtype SIDE-BRANCH — ``y += B(A @ x)``
reading the original activation and adding onto the output — never peft
module wrapping (ie#374: peft fights the layerwise-cast hooks) and never a
weight mutation. Three lanes are branch-capable:

- **w8a8 scaled_mm** (gw#547): Fp8ScaledLinear reads ``lora_a``/``lora_b``
  non-persistent buffers natively in its forward.
- **fp8-storage layerwise-cast** (gw#558): plain ``nn.Linear`` under
  diffusers cast hooks gets an idempotent instance-forward wrap; branch
  tensors live in the module ``__dict__`` (plain attrs) so ``.to(dtype)``
  cast hooks never round-trip them through fp8.
- **plain bf16/fp16 resident** (gw#558): same wrap; removal restores the
  original forward path bit-exactly.

Graph stability: every branch-capable Linear gets a branch under canonical
placement (zeroed slots for layers an adapter doesn't cover), and the
concatenated rank of the active adapter set is padded to a fixed bucket
(:data:`RANK_BUCKETS`). Every adapter set inside one bucket shares ONE traced
graph; hot-swap is a buffer copy. Multiple active adapters rank-concat into
one A/B pair (the gw#430 svdq trick); per-adapter scale (alpha/rank x user
weight) is folded into the B copy.

The branch-bearing pipeline stamps ``_cozy_weight_lane =
"<base-lane>-lora<bucket>"`` (``w8a8-lora32``, ``fp8-hooks-lora32``,
``lora32`` for plain bf16) so the SYMMETRIC ``compile_cache.lane_drift``
guard keeps LoRA-bearing pipelines and branchless compile cells apart in
both directions.
"""

from __future__ import annotations

import logging
import math
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
    """The pipeline's denoiser when it is branch-capable (gw#558: every
    lane — w8a8 scaled_mm, fp8-storage layerwise-cast, plain resident), else
    None (no discoverable denoiser module — such pipelines keep the whole
    peft path). Deliberately NO module scan here: this runs on every demote
    (residency.pre_demote -> detach) — adapters that map onto no Linear
    fail typed in :func:`map_adapter` (with the plain-lane peft fallback in
    AdapterResidency.activate)."""
    for name in ("transformer", "unet"):
        denoiser = getattr(pipe, name, None)
        if denoiser is not None and hasattr(denoiser, "named_modules"):
            return denoiser
    return None


def branch_lane(model: Any) -> str:
    """The denoiser's base weight lane for branch policy/stamping:
    ``"w8a8"`` | ``"fp8-hooks"`` | ``""`` (plain resident). Both fp8 GEMM
    dispatch branches (rowwise sm_90+, pertensor sm_89 — gw#564) are the
    w8a8 lane; the additive LoRA branch is orthogonal to the scaling mode."""
    if getattr(model, "_cozy_w8a8_mode", "") in ("rowwise", "pertensor"):
        return "w8a8"
    if getattr(model, "_cozy_fp8_storage_applied", False):
        return "fp8-hooks"
    return ""


def quantized_modules(model: Any) -> Dict[str, Any]:
    """name -> Fp8ScaledLinear for every quantized Linear in the denoiser."""
    from .w8a8 import fp8_scaled_linear_class

    cls = fp8_scaled_linear_class()
    return {n: m for n, m in model.named_modules() if isinstance(m, cls)}


def branch_modules(model: Any) -> Dict[str, Any]:
    """name -> branch-capable module for the denoiser: Fp8ScaledLinear,
    plain nn.Linear, or plain nn.Conv2d (gw#627 — the curated sdxl distill
    adapters carry conv pairs; convs are never quantized, so their branch
    is always the eager instance-forward wrap). Other module kinds are not
    branch targets — adapters that name them fail loud in
    :func:`map_adapter`."""
    import torch.nn as nn

    from .w8a8 import fp8_scaled_linear_class

    fp8_cls = fp8_scaled_linear_class()
    return {
        n: m for n, m in model.named_modules()
        if isinstance(m, fp8_cls) or type(m) in (nn.Linear, nn.Conv2d)
    }


def branch_bucket(model: Any) -> int:
    """The enabled bucket, 0 when branches are not enabled."""
    return int(getattr(model, _BUCKET_ATTR, 0) or 0)


def lora_lane(bucket: int, sparse: bool = False, base: str = "w8a8") -> str:
    # Sparse (eager-only) placement is a different graph per coverage
    # pattern — the "-sparse" suffix can never match a produced cell label.
    # ``base`` is the branchless lane the branch rides on ("w8a8",
    # "fp8-hooks", or "" for plain resident).
    prefix = f"{base}-" if base else ""
    return f"{prefix}lora{int(bucket)}" + ("-sparse" if sparse else "")


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
    """Shape-check one down/up pair against its branch module; conv pairs
    (gw#627) normalize the up half to [out, r, 1, 1]."""
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


# Flat staging buffers above this size stay pageable — pinned host memory is
# a shared, non-swappable resource and the cache holds up to _MAPCACHE_MAX.
_PIN_MAX_BYTES = 512 << 20


def _stage_adapter(mapped: Dict[str, Tuple[Any, Any, float]]) -> Dict[str, Any]:
    """One adapter's swap-ready form: per-dtype flat CPU staging tensors
    (pinned when small enough on CUDA hosts) + an index of every layer's
    (dtype, offset, shape) slices. Built once per resident adapter; hot-swaps
    then pay only one H2D transfer + device-side placement."""
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
        pin = (torch.cuda.is_available()
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
    from .w8a8 import fp8_scaled_linear_class

    return isinstance(mod, fp8_scaled_linear_class())


def _install_branch_forward(mod: Any) -> None:
    """Idempotent instance-forward wrap for a plain ``nn.Linear``
    (``y = orig(x) + (x @ A.T) @ B.T``) or plain ``nn.Conv2d``
    (``y = orig(x) + conv1x1(conv(x, A), B)`` — gw#627). Branch tensors are
    read from the module ``__dict__`` so layerwise-cast hooks
    (``.to(dtype)``) never see them (gw#558 / ie#374). With no branch
    installed the wrap is a pure pass-through — removal is bit-exact."""
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
            # Self-heal after a host-resident alloc (block-offload lane):
            # branch tensors are tiny; pin them to the execution device.
            # Rebind ONLY if the module still holds the exact pair we read —
            # a concurrent realloc must never be clobbered with stale copies
            # (the compute below uses the consistent local pair either way).
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


def alloc_branch_buffers(mod: Any, bucket: int) -> None:
    """Zeroed A/B branch tensors on one branch-capable module.

    Fp8ScaledLinear registers non-persistent buffers (they move with the
    module and its forward reads them natively; the w8a8 denoiser carries
    no cast hooks). Plain ``nn.Linear``/``nn.Conv2d`` get a forward wrap +
    plain ``__dict__`` attrs instead — registered buffers would be
    round-tripped bf16->fp8->bf16 by the layerwise-cast hooks on the
    fp8-storage lane. Conv branches (gw#627): A [bucket, in, kh, kw] runs
    the base conv's stride/padding, B [out, bucket, 1, 1] projects up."""
    import torch
    import torch.nn as nn

    dev = mod.weight.device
    # Branch tensors compute in the module's COMPUTE dtype — never its
    # storage dtype (on the fp8-storage lane weight AND bias rest in fp8).
    _compute = (torch.float16, torch.bfloat16, torch.float32)
    dtype = torch.bfloat16
    for cand in (mod.weight.dtype if not _is_scaled_linear(mod) else None,
                 mod.bias.dtype if mod.bias is not None else None):
        if cand in _compute:
            dtype = cand
            break
    for name in ("lora_a", "lora_b"):
        mod._buffers.pop(name, None)
        mod.__dict__.pop(name, None)
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
    """Allocate branch buffers on EVERY branch-capable Linear (canonical
    placement — one traced graph over all coverage patterns; the compiled
    lane). Idempotent at the same bucket; a different bucket reallocates
    (a new graph family)."""
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
    """Drop the branch buffers entirely (back to the branchless graph
    family). Used on demote/teardown, never between requests."""
    for mod in branch_modules(model).values():
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
    for mod in branch_modules(model).values():
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
    # Mapping and flat staging are pure in the state dict; repeat swaps of a
    # resident adapter (the AdapterCache serves the SAME dict object) skip
    # the key-mapping pass AND the CPU flatten — the flatten measured ~700ms
    # at SDXL scale, the actual H2D+device placement ~130ms.
    cache: Dict[Any, Any] = getattr(model, _MAPCACHE_ATTR, None) or {}
    mapped = []
    for sd, w, ref in adapters:
        key = (ref, id(sd), len(sd))
        entry = cache.get(key)
        if entry is None:
            entry = _stage_adapter(map_adapter(sd, model, ref=ref))
            cache[key] = entry
            while len(cache) > _MAPCACHE_MAX:
                cache.pop(next(iter(cache)))
        mapped.append((entry, w, ref))
    setattr(model, _MAPCACHE_ATTR, cache)
    per_layer: Dict[str, int] = {}
    for entry, _w, _ref in mapped:
        for path, r in entry["ranks"].items():
            per_layer[path] = per_layer.get(path, 0) + r
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
        for entry, _w, _ref in mapped:
            covered_paths.update(entry["ranks"])
        for path, mod in branch_modules(model).items():
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
    mods = branch_modules(model)
    with torch.no_grad():
        # One H2D transfer per adapter of its CACHED flat staging buffer
        # (pinned when small enough), then index-addressed device-side
        # cast/scale-fold/placement — the per-swap CPU flatten measured
        # ~700ms at SDXL scale; staged warm swaps pay only transfer+place.
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
                continue  # sparse placement: uncovered layer has no branch
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
                mod.lora_b.zero_()  # canonical zeroed slot (uniform)
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
    a different graph family per (base lane, bucket) — lane_drift guards
    both directions. The branchless base lane is remembered on first stamp
    so clearing the branch restores it exactly."""
    bucket = branch_bucket(model)
    sparse = bool(getattr(model, _SPARSE_ATTR, False))
    base = getattr(pipe, "_cozy_lora_base_lane", None)
    if base is None:
        from .loading import pipeline_weight_lane

        # The loader stamps _cozy_weight_lane on real pipelines; the
        # denoiser's own lane markers are the fallback authority.
        base = pipeline_weight_lane(pipe) or branch_lane(model)
        try:
            pipe._cozy_lora_base_lane = base
        except Exception:
            return
    try:
        pipe._cozy_weight_lane = lora_lane(bucket, sparse, base=base) if bucket else base
    except Exception:
        pass


def normalize_adapter_state_dict(
    pipe: Any, sd: Dict[str, Any], *, ref: str = ""
) -> Dict[str, Any]:
    """Normalize a raw adapter through the pipeline class's own
    ``lora_state_dict`` converter (te#81's zero-drift pattern: byte-identical
    key handling with the boot-time ``load_lora_weights`` path). Falls back
    to the raw dict when the class has no converter or it fails — the
    :func:`map_adapter` grammar (diffusers/peft/kohya) then applies as
    before. sdxl-class converters receive ``unet_config`` for SGM block
    remapping of kohya adapters. Returned ``network_alphas`` fold back in as
    ``<module>.alpha`` entries."""
    import inspect

    fn = getattr(type(pipe), "lora_state_dict", None)
    if fn is None:
        return sd
    kwargs: Dict[str, Any] = {}
    unet = getattr(pipe, "unet", None)
    if unet is not None and hasattr(unet, "config"):
        try:
            if "unet_config" in inspect.signature(fn).parameters:
                kwargs["unet_config"] = unet.config
        except (TypeError, ValueError):
            pass
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
    if any(".processor." in k for k in converted):
        # gw#627 live find: diffusers' non-diffusers converter emits LEGACY
        # attn-processor names for kohya sdxl attention keys
        # (…attn1.processor.to_q_lora.down.weight) — they match no real
        # module, so the whole adapter would fail typed. The raw kohya-flat
        # keys resolve directly against module paths in map_adapter.
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


__all__ = [
    "RANK_BUCKETS",
    "apply_branch_adapters",
    "branch_bucket",
    "branch_lane",
    "branch_modules",
    "branch_target",
    "branches_active",
    "clear_branch_adapters",
    "disable_lora_branches",
    "enable_lora_branches",
    "lora_lane",
    "map_adapter",
    "normalize_adapter_state_dict",
    "quantized_modules",
    "rank_bucket",
    "split_state_dict",
    "stamp_lane",
]
