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

**Branch targets are PER-COMPONENT (gw#679).** A pipeline's denoiser is a
SET, not a module: Wan 2.2 A14B is a dual-expert MoE (``transformer``
high-noise + ``transformer_2`` low-noise, handed off at ``boundary_ratio``)
and its distillation is correspondingly two adapters. Every branch
operation therefore runs over :func:`branch_targets` — the bucket container
is allocated on each expert, and an adapter set is ROUTED to the component
its own keys name (``transformer.`` / ``transformer_2.`` / ``unet.``).
Routing is data, never a wire field: diffusers already namespaces
multi-denoiser pipelines by component, so a per-expert adapter is mirrored
with its component prefix. On a multi-denoiser pipeline an adapter that
does NOT name its component is refused (``diffusers``' Wan converters
rewrite every non-diffusers key to the ``transformer.`` prefix whatever
expert the file was trained for — landing both halves on the high expert
with a clean log was the gw#679 defect). Single-denoiser pipelines route
every denoiser key to their one target exactly as before.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import activity as activity_mod
from ..component_vocab import denoiser_components
from ..api.errors import RefCompatibilitySurprise, ValidationError
# Flat staging buffers above this size stay pageable — pinned host memory
# is shared and non-swappable, and the cache holds up to _MAPCACHE_MAX.
# ONE owner for the worker's whole pinned budget (pgw#973 §4.24).
from ..media_transfer import PIN_MAX_BYTES as _PIN_MAX_BYTES
from . import adapter_fidelity
from .fp8_storage import structural_base
from .w8a8 import fp8_scaled_linear_class
import inspect
from .loading import pipeline_weight_lane

logger = logging.getLogger(__name__)

# Padded-rank buckets: one traced graph per bucket. 16/32 cover the bulk of
# the civitai catalog; 64/128 the high-rank tail (survey follow-up in gw#547).
RANK_BUCKETS = (16, 32, 64, 128)

_BUCKET_ATTR = "_cozy_lora_bucket"
_ACTIVE_ATTR = "_cozy_lora_active"
_SPARSE_ATTR = "_cozy_lora_sparse"
_MAPCACHE_ATTR = "_cozy_lora_mapcache"
_MAPCACHE_MAX = 8
# The kohya-flat prefixes are NOT component namespaces (see below) and are not
# vocabulary — sd-scripts emits them verbatim, so they stay literal here.
_KOHYA_FLAT_PREFIXES = ("lora_unet_", "lora_transformer_")
# gw#679: the denoiser components a pipeline can carry, in stamp order. A
# dual-expert MoE carries transformer (high noise) AND transformer_2 (low).
_denoiser_components = denoiser_components


def _denoiser_prefixes() -> tuple[str, ...]:
    return tuple(f"{c}." for c in _denoiser_components()) + _KOHYA_FLAT_PREFIXES


def _component_prefixes() -> tuple[tuple[str, str], ...]:
    """Key prefix -> the component it NAMES, longest first (``transformer_2.``
    must win over ``transformer.``). DOTTED forms only: that prefix IS the
    diffusers component namespace. The kohya-flat ``lora_unet_`` prefix is
    NOT a declaration — sd-scripts emits it for transformer denoisers too
    (flux/qwen adapters serve fine on the branch today) — so those keys name
    no component and follow the unprefixed rules below."""
    return tuple(
        (f"{c}.", c)
        for c in sorted(_denoiser_components(), key=len, reverse=True)
    )
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


def branch_targets(pipe: Any) -> Dict[str, Any]:
    """component name -> denoiser module for EVERY branch-capable denoiser
    the pipeline carries (gw#679), in stamp order.

    One entry for an ordinary pipeline (LTX/sdxl/qwen: ``transformer`` or
    ``unet``) — every operation below then degenerates to the pre-gw#679
    single-target behavior. TWO for a dual-expert MoE (Wan 2.2 A14B), and
    both are real branch targets: adapting only the high expert leaves the
    low one running undistilled weights on a distilled ladder.

    Branch-capability is per-lane-agnostic (gw#558: w8a8 scaled_mm,
    fp8-storage layerwise-cast, plain resident). Deliberately NO module scan
    here: this runs on every demote (residency.pre_demote -> detach) —
    adapters that map onto no Linear fail typed in :func:`map_adapter` (with
    the plain-lane peft fallback in AdapterResidency.activate)."""
    out: Dict[str, Any] = {}
    for name in _denoiser_components():
        denoiser = getattr(pipe, name, None)
        if denoiser is not None and hasattr(denoiser, "named_modules"):
            out[name] = denoiser
    return out


def declared_component(key: str) -> str:
    """The pipeline component one adapter key NAMES, or ``""`` when the key
    carries no component prefix (bare/kohya-flat module paths)."""
    for prefix, comp in _component_prefixes():
        if key.startswith(prefix):
            return comp
    return ""


def require_component_declaration(
    components: Iterable[str], raw_sd: Mapping[str, Any], *, ref: str = "",
) -> None:
    """gw#679 fail-closed, checked against the adapter's RAW keys.

    On a MULTI-denoiser pipeline an adapter must name the expert it adapts.
    This cannot be checked after normalization: diffusers' Wan converters
    rewrite every non-diffusers key (``diffusion_model.…``, ``lora_unet_…``)
    to the ``transformer.`` prefix regardless of which expert the file was
    trained for, so an unmirrored per-expert half would arrive looking like
    an explicit high-noise declaration and land there with a clean log —
    exactly the silent defect this issue is about. Mirror per-expert
    adapters with their component prefix instead."""
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
    """Partition one adapter's denoiser keys by the component they target
    (gw#679). Keys keep their prefixes — :func:`map_adapter` strips them.

    A key naming a component the pipeline does not carry is unroutable on
    ANY topology and refuses. Beyond that, a single-denoiser pipeline takes
    every remaining key, prefixed or not (unchanged pre-gw#679 behavior: an
    unresolvable key still fails typed in :func:`map_adapter`, which is what
    keeps the plain-lane peft fallback reachable). With two or more experts
    routing is explicit and total: a key naming no component is ambiguous
    and refuses too. Both refuse before anything is attached — never a
    partial application."""
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
        # One target: prefixed and unprefixed keys alike are its own.
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
    """The denoiser's base weight lane for branch policy/stamping:
    ``"w8a8"`` | ``"fp8-hooks"`` | ``""`` (plain resident). Both fp8 GEMM
    dispatch branches (rowwise sm_90+, pertensor sm_89 — gw#564) are the
    w8a8 lane; the additive LoRA branch is orthogonal to the scaling mode."""
    if getattr(model, "_cozy_w8a8_mode", "") in ("rowwise", "pertensor"):
        return "w8a8"
    if getattr(model, "_cozy_fp8_storage_applied", False):
        return "fp8-hooks"
    return ""


def branch_modules(model: Any) -> Dict[str, Any]:
    """name -> branch-capable module for the denoiser: Fp8ScaledLinear,
    plain nn.Linear, or plain nn.Conv2d (gw#627 — the curated sdxl distill
    adapters carry conv pairs; convs are never quantized, so their branch
    is always the eager instance-forward wrap). Other module kinds are not
    branch targets — adapters that name them fail loud in
    :func:`map_adapter`. Selection is by EXACT class over
    :func:`fp8_storage.structural_base`, so a pgw#727 fp8-storage leaf is
    targeted as the plain class it was restructured from (its branch reads
    the compute-dtype activation and adds onto the compute-dtype output —
    the fp8 storage is never touched, exactly as under the hook lane)."""
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
        (den if k.startswith(_denoiser_prefixes()) else rest)[k] = v
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
        # Longest first: "transformer_2." must not be read as "transformer.".
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


def _clear_branch_slots(mod: Any) -> None:
    """Drop one module's branch tensors.

    ``Fp8ScaledLinear`` keeps its slots DECLARED as ``None`` buffers
    (pgw#726) — popping them would leave plain attributes behind, which is
    both the shape ``register_buffer`` refuses and the shape export cannot
    see. Plain Linear/Conv keep the ``__dict__`` form their forward wrap
    reads."""
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
    # ONE definition, shared with pgw#794's fidelity gate: the grid the gate
    # judges the delta against must be the grid the buffers are allocated in,
    # by construction rather than by two copies agreeing.
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
        _clear_branch_slots(mod)
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


# ---------------------------------------------------------------------------
# Set-level operations (gw#679): every branch-capable denoiser the pipeline
# carries moves TOGETHER — one bucket, one lane stamp, one active set.
# ---------------------------------------------------------------------------


def enable_branch_execution_lanes(pipe: Any, bucket: int) -> Dict[str, Any]:
    """Allocate the rank-``bucket`` branch container on EVERY branch-capable
    denoiser (both experts of an MoE — the ``Compile(lora_bucket=)`` arming
    contract). Returns the targets it armed."""
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
    """The pipeline's branch bucket — one value across the denoiser set (the
    set is always enabled/resized together), 0 when no branch is enabled."""
    return max(
        (branch_bucket(m) for m in branch_targets(pipe).values()), default=0)


def _stage_for(
    model: Any, adapters: Sequence[Tuple[Dict[str, Any], float, str]],
    *, request_id: str = "",
) -> List[Tuple[Dict[str, Any], float, str]]:
    """Map + stage one component's adapters. PURE — no module is touched, so
    an unmappable adapter fails before anything is attached (gw#679's
    never-partially-attach rule). Repeat swaps of a resident adapter (the
    AdapterCache serves the SAME dict object) skip the key-mapping pass AND
    the CPU flatten — the flatten measured ~700ms at SDXL scale, the actual
    H2D+device placement ~130ms.

    pgw#794: also where the branch's FIDELITY gate runs. The delta is measured
    against the dtype the branch buffers are actually allocated in — read from
    the modules, so an fp8 branch would be judged as fp8 without an edit — and
    an adapter the branch would destroy is refused HERE, on the pure pass,
    before a buffer is touched. Survival is a property of (adapter, model), so
    it is computed once on the cold path and cached beside the staging entry
    (0.86 s for all 788 modules of sdxl lightning-4step on 4 CPU threads);
    warm swaps re-check the cached verdict for free."""
    cache: Dict[Any, Any] = getattr(model, _MAPCACHE_ATTR, None) or {}
    staged: List[Tuple[Dict[str, Any], float, str]] = []
    for sd, w, ref in adapters:
        key = (ref, id(sd), len(sd))
        entry = cache.get(key)
        fresh = entry is None
        if entry is None:
            mapped = map_adapter(sd, model, ref=ref)
            if not mapped:
                # pgw#824 VACUOUS GUARD. `evaluate_branch` returns None for an
                # empty mapping and `gate(None)` is a no-op, so an adapter that
                # maps to ZERO modules sails through the fidelity gate that
                # exists to catch exactly this. The request then renders with
                # NO adapter applied while reporting success — the user asked
                # for a LoRA and got an image without one, and nothing
                # anywhere says so.
                #
                # Reachable: `_normalize_lora_keys` falls back to RAW keys when
                # `lora_state_dict` normalization raises, and raw kohya keys
                # match no module path on this model.
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
    """The widest per-layer concatenated rank this staged set needs."""
    per_layer: Dict[str, int] = {}
    for entry, _w, _ref in staged:
        for path, r in entry["ranks"].items():
            per_layer[path] = per_layer.get(path, 0) + r
    return max(per_layer.values(), default=0)


def _settle_bucket(
    models: Sequence[Any], want: int, *, uniform: bool, allow_resize: bool,
) -> int:
    """The bucket the whole branch SET lands on, allocated under canonical
    placement. One value for every component: two experts on different
    buckets would be two graph families under one lane stamp."""
    current = max((branch_bucket(m) for m in models), default=0)
    was_sparse = any(bool(getattr(m, _SPARSE_ATTR, False)) for m in models)
    if not uniform:
        return want
    if current >= want and current and not was_sparse:
        want = current  # never shrink — stay on the already-traced graph
    if current != want or was_sparse:
        if not allow_resize:
            raise ValidationError(
                f"active LoRA set needs rank bucket {want} but the compiled "
                f"pipeline traced bucket {current or 'none'} — recompile at "
                "swap time is never allowed; publish a matching lora cell"
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
    """Copy one component's staged adapters into its branch buffers."""
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
    if mapped and not covered:
        # Every mapped key resolved to a branch-capable module, so zero
        # covered means this component carries no branch CONTAINER — the
        # adapter would be a silent no-op (gw#679: half-armed MoE).
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
    """Make exactly ``adapters`` (state_dict, user weight, ref) ONE denoiser's
    active branch set. Rank-concat across adapters, pad to the bucket, fold
    ``alpha/rank * weight`` into the B copy. Returns swap stats.

    ``uniform=True`` (compiled pipelines) keeps canonical placement — every
    quantized Linear carries a branch, zeroed slots for uncovered layers,
    never shrinking the bucket; ``allow_resize=False`` additionally refuses
    bucket changes (a resize is a new graph family, and prod never compiles
    at runtime). ``uniform=False`` (eager) allocates branches ONLY on
    covered layers and drops stale ones — eager pays a per-kernel launch
    tax even for zeroed slots, so sparse placement keeps uncovered layers
    at exactly branchless speed.

    ``rank_floor`` (gw#679) is the widest rank a SIBLING expert needs: the
    components of one pipeline always share a bucket. Prefer
    :func:`apply_branch_adapter_set` — it routes an adapter set across the
    whole denoiser set and computes the floor itself."""
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
    """Make exactly ``routed`` (component -> adapters) the PIPELINE's active
    branch set (gw#679).

    Every branch-capable denoiser is settled in one pass: components named
    in ``routed`` take their own adapters, components not named are cleared
    (a stale expert branch is exactly the silent-wrong-picture class this
    guards), and all of them share one bucket so the pipeline carries a
    single coherent graph family and lane stamp.

    Fail-closed: an unroutable component and any unmappable adapter key
    raise BEFORE a single buffer is touched — a half-attached MoE serves a
    distilled expert next to an undistilled one."""
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
    # Pure pass first: map + stage every component. Raises here leave the
    # pipeline exactly as it was.
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
        # A half-armed set (one expert carrying the container, the other not
        # — e.g. a family that declares only `transformer` as a compile
        # target) would place the sibling's adapter into nothing. Refuse
        # before touching either expert.
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
    """The branchless base weight lane CELL IDENTITY rides on — the ONE
    resolution :func:`stamp_lane` memoizes: the memoized base, else the
    pipeline's stamped/probed lane, else the denoiser's own lane markers
    (:func:`branch_lane`), which see the w8a8 GEMM mode
    (``_cozy_w8a8_mode``) that ``loading.pipeline_weight_lane`` cannot.

    pgw#686: the advertised requested cell key and the minted/published cell
    key MUST resolve the base lane identically — when they don't, the
    published cell is never requested by any worker, adoption is
    structurally impossible, and every cold pod re-mints (the ie#546 burst
    stampede: requested ``""``/``"fp8-hooks"`` vs published ``"w8a8"``,
    every other axis digest-identical)."""
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
    """Keep the compile-cache graph key honest: branch-bearing pipelines are
    a different graph family per (base lane, bucket) — lane_drift guards
    both directions. The branchless base lane is remembered on first stamp
    so clearing the branch restores it exactly.

    The stamp is per PIPELINE, over its whole denoiser set (gw#679): the
    components always carry the same bucket, and how many experts a family
    has is a property of the pipeline class, not of the lane — so the lane
    STRING (and therefore every published cell key) is unchanged by MoE
    support."""
    tg = branch_targets(pipe) if targets is None else dict(targets)
    models = list(tg.values())
    bucket = max((branch_bucket(m) for m in models), default=0)
    sparse = any(bool(getattr(m, _SPARSE_ATTR, False)) for m in models)
    base = getattr(pipe, "_cozy_lora_base_lane", None)
    if base is None:
        # One brain (pgw#686): the same resolution the advertised requested
        # key uses, so the stamped/published key can never diverge from it.
        base = effective_base_execution_lane(pipe)
        try:
            pipe._cozy_lora_base_lane = base
        except Exception:
            return
    try:
        pipe._cozy_weight_lane = lora_execution_lane(bucket, sparse, base=base) if bucket else base
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
