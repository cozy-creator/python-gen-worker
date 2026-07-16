"""Pinned host-RAM weight swapping (gw#551).

Tier swaps used to ride whole-object ``.to(device)``: pageable host memory
throttles H2D copies far below PCIe line rate, so promoting a ~38 GB
transformer took many seconds. This module keeps a pinned host copy of every
weight, cached on the module across swaps:

- demote (cuda -> cpu): weights are copied D2H into pinned staging ONCE and
  parameters re-pointed at it. Later demotes of an unchanged weight are pure
  pointer swaps (the host copy is already current — residency-managed weights
  are immutable; adapters detach via ``pre_demote`` before any move).
- promote (cpu -> cuda): one ``non_blocking`` H2D per tensor from pinned
  memory at full PCIe bandwidth.

Fail-soft everywhere: meta tensors, aliased partial-view storages, or a
failed pinned allocation fall back to the caller's ``.to()`` path (pageable,
slower, still correct).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Per-module swap cache: {qualified tensor name: _PinSlot}.
_CACHE_ATTR = "_cozy_pin_cache"


class _PinSlot:
    """One weight's pinned host copy + the data_ptr of the cuda tensor our
    last promote produced from it (0 = weight currently lives on the host).
    A cuda tensor whose ptr no longer matches was replaced out-of-band and
    gets a fresh D2H copy instead of a pointer swap."""

    __slots__ = ("host", "device_ptr")

    def __init__(self, host: Any, device_ptr: int = 0) -> None:
        self.host = host
        self.device_ptr = device_ptr


def _module_tensors(module: Any) -> List[Tuple[str, Any, str, Any, bool]]:
    """(qualified_name, owner_module, attr, tensor, is_param) for every
    parameter and buffer, including non-persistent buffers."""
    out: List[Tuple[str, Any, str, Any, bool]] = []
    for mname, mod in module.named_modules():
        prefix = f"{mname}." if mname else ""
        for pname, p in getattr(mod, "_parameters", {}).items():
            if p is not None:
                out.append((prefix + pname, mod, pname, p, True))
        for bname, b in getattr(mod, "_buffers", {}).items():
            if b is not None:
                out.append((prefix + bname, mod, bname, b, False))
    return out


def _slot_matches(slot: _PinSlot, t: Any) -> bool:
    h = slot.host
    return h is not None and h.shape == t.shape and h.dtype == t.dtype


def _assign(mod: Any, attr: str, new: Any, is_param: bool) -> None:
    if is_param:
        mod._parameters[attr].data = new
    else:
        mod._buffers[attr] = new


def swap_module(module: Any, device: str) -> bool:
    """Move every parameter/buffer of ``module`` to ``device`` through the
    pinned swap cache. True when fully handled; False when the caller should
    fall back to ``module.to(device)`` (a partial swap is safe to hand over —
    ``.to()`` finishes the move)."""
    try:
        import torch
    except Exception:
        return False
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        return False
    if target.type == "cuda" and not torch.cuda.is_available():
        return False

    tensors = _module_tensors(module)
    pending = [row for row in tensors if row[3].device.type != target.type]
    if not pending:
        return True
    # Structural pre-pass: shapes this mover cannot represent bail BEFORE any
    # mutation (meta tensors, partial-view aliases into a larger storage).
    for _, _, _, t, _ in pending:
        if t.is_meta or t.storage_offset() != 0:
            return False

    cache: Dict[str, _PinSlot] = getattr(module, _CACHE_ATTR, None) or {}
    moved: Dict[int, Any] = {}   # source data_ptr -> replacement (alias dedupe)
    keep_alive: List[Any] = []   # D2H sources stay alive until the sync
    try:
        with torch.inference_mode(False), torch.no_grad():
            for name, mod, attr, t, is_param in pending:
                key = t.data_ptr() if t.numel() else 0
                new = moved.get(key) if key else None
                if new is not None and (new.shape != t.shape or new.dtype != t.dtype):
                    return False  # aliased tensors with differing views
                if new is None:
                    new = _swap_one(torch, t, target, name, cache)
                    if key:
                        moved[key] = new
                    keep_alive.append(t)
                _assign(mod, attr, new, is_param)
    except Exception as exc:
        logger.warning(
            "pinned swap of %s to %s failed (%s: %s); falling back to .to()",
            type(module).__name__, device, type(exc).__name__, exc,
        )
        return False
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        del keep_alive
    if cache:
        try:
            object.__setattr__(module, _CACHE_ATTR, cache)
        except Exception:
            return True  # moved fine; cache just won't persist
    return True


def _swap_one(torch: Any, t: Any, target: Any, name: str, cache: Dict[str, _PinSlot]) -> Any:
    slot = cache.get(name)
    if target.type == "cpu":
        if (
            slot is not None
            and _slot_matches(slot, t)
            and slot.device_ptr == t.data_ptr()
        ):
            # Our own promote produced ``t`` from this host copy and nothing
            # replaced it since: the host bytes are current. Pointer swap.
            slot.device_ptr = 0
            return slot.host
        host = slot.host if (slot is not None and _slot_matches(slot, t)) else None
        if host is None:
            try:
                host = torch.empty_like(t, device="cpu", pin_memory=True)
            except Exception:
                host = torch.empty_like(t, device="cpu")  # pageable fallback
        host.copy_(t, non_blocking=host.is_pinned())
        cache[name] = _PinSlot(host, 0)
        return host
    # -> cuda. NB: ``p.data`` returns a fresh view per access, so "is this
    # our pinned staging?" must compare storage pointers, never identity.
    if (
        slot is not None
        and slot.device_ptr == 0
        and _slot_matches(slot, t)
        and slot.host.data_ptr() == t.data_ptr()
    ):
        dev = slot.host.to(target, non_blocking=slot.host.is_pinned())
        slot.device_ptr = dev.data_ptr()
        return dev
    # Foreign cpu tensor (first promote after a cold load): plain move; the
    # pinned cache is built on the first demote, not here (no double copy).
    return t.to(target)


def swap_object(obj: Any, device: str) -> bool:
    """Pinned swap for a residency object: a bare ``nn.Module`` (lane
    ModuleDicts) or a pipeline exposing ``components``. False = caller moves
    it with ``.to()``."""
    try:
        import torch.nn as nn
    except Exception:
        return False
    if isinstance(obj, nn.Module):
        return swap_module(obj, device)
    comps = getattr(obj, "components", None)
    if isinstance(comps, dict) and comps:
        mods = [m for m in comps.values() if isinstance(m, nn.Module)]
        if not mods:
            return False
        ok = True
        for m in mods:
            ok = swap_module(m, device) and ok
        return ok
    return False


def cached_swap_bytes(obj: Any) -> int:
    """Bytes already staged in pinned host caches under ``obj`` — a demote of
    this object needs that much LESS fresh host RAM."""
    try:
        import torch.nn as nn
    except Exception:
        return 0
    modules: List[Any] = []
    if isinstance(obj, nn.Module):
        modules = [obj]
    else:
        comps = getattr(obj, "components", None)
        if isinstance(comps, dict):
            modules = [m for m in comps.values() if isinstance(m, nn.Module)]
    total = 0
    for module in modules:
        cache = getattr(module, _CACHE_ATTR, None)
        if not isinstance(cache, dict):
            continue
        for slot in cache.values():
            host = getattr(slot, "host", None)
            if host is not None:
                total += host.numel() * host.element_size()
    return total


__all__ = ["swap_module", "swap_object", "cached_swap_bytes"]
