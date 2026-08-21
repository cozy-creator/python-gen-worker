"""Refuse host-RAM module moves that cannot fit the cgroup budget."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from .api.errors import HostRamMoveRefusedError
from .models.memory import probe_host_ram
from .models.residency import _effective_ram_floor_gb

logger = logging.getLogger(__name__)

_GIB = 1 << 30
_MIN_GUARDED_GIB = 1.0
_DISABLE_ENV = "GEN_WORKER_HOST_MOVE_GUARD"

_installed = False
_probe_root: Optional[Path] = None
_probe_self: Optional[Path] = None


def _enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip() not in ("0", "false", "no")


def _target_is_cpu(args: tuple, kwargs: dict) -> bool:
    try:
        import torch

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        return device is not None and device.type == "cpu"
    except Exception:
        raw = kwargs.get("device", args[0] if args else None)
        try:
            import torch

            return raw is not None and torch.device(raw).type == "cpu"
        except Exception:
            return False


class _Unmeasurable(RuntimeError):
    """This module's incoming byte count could not be computed."""


def _incoming_bytes(module: Any) -> int:
    total = 0
    try:
        for t in module.parameters(recurse=True):
            if t.device.type != "cpu":
                total += t.numel() * t.element_size()
        for t in module.buffers(recurse=True):
            if t.device.type != "cpu":
                total += t.numel() * t.element_size()
    except Exception as exc:
        raise _Unmeasurable(
            f"cannot size {type(module).__name__} for a host move: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return total


def _refuse_if_over_budget(incoming: int, **probe_kwargs: Any) -> None:
    ram = probe_host_ram(**probe_kwargs)
    available = int(ram.available_gb * _GIB)
    floor = int(_effective_ram_floor_gb() * _GIB)
    if incoming <= available - floor:
        return
    limit = int(ram.cgroup_limit_gb * _GIB) if ram.cgroup_limit_gb else None
    logger.error(
        "host-RAM move REFUSED (pgw#763): incoming=%.1fGiB available=%.1fGiB "
        "floor=%.1fGiB source=%s — refusing instead of dying to the cgroup "
        "OOM killer", incoming / _GIB, available / _GIB, floor / _GIB, ram.source,
    )
    raise HostRamMoveRefusedError(
        incoming_bytes=incoming, available_bytes=available,
        floor_bytes=floor, limit_bytes=limit,
    )


def check_host_ram_move(module: Any) -> None:
    """Raise :class:`HostRamMoveRefusedError` when moving ``module`` to CPU cannot fit ``available - floor``."""
    try:
        incoming = _incoming_bytes(module)
    except _Unmeasurable as exc:
        logger.warning("host-RAM move guard: %s — checking the floor anyway", exc)
        kwargs = {}
        if _probe_root is not None and _probe_self is not None:
            kwargs = {"root": _probe_root, "proc_self_cgroup": _probe_self}
        _refuse_if_over_budget(int(_MIN_GUARDED_GIB * _GIB), **kwargs)
        return
    if incoming < _MIN_GUARDED_GIB * _GIB:
        return
    kwargs = {}
    if _probe_root is not None and _probe_self is not None:
        kwargs = {"root": _probe_root, "proc_self_cgroup": _probe_self}
    _refuse_if_over_budget(incoming, **kwargs)


def install() -> bool:
    """Patch ``torch.nn.Module.to``/``.cpu``."""
    global _installed
    if _installed:
        return True
    if not _enabled():
        logger.info("host-RAM move guard disabled via %s", _DISABLE_ENV)
        return False
    try:
        import torch
    except Exception:
        return False

    module_cls = torch.nn.Module
    orig_to, orig_cpu = module_cls.to, module_cls.cpu

    def guarded_to(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _enabled() and _target_is_cpu(args, kwargs):
            check_host_ram_move(self)
        return orig_to(self, *args, **kwargs)

    def guarded_cpu(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _enabled():
            check_host_ram_move(self)
        return orig_cpu(self, *args, **kwargs)

    guarded_to._cozy_host_move_guard = True  # type: ignore[attr-defined]
    guarded_cpu._cozy_host_move_guard = True  # type: ignore[attr-defined]
    module_cls.to = guarded_to  # type: ignore[method-assign]
    module_cls.cpu = guarded_cpu  # type: ignore[method-assign]
    _installed = True
    logger.info(
        "host-RAM move guard armed (threshold %.0fGiB): oversized .to('cpu') "
        "fails typed instead of OOM-killing the worker (pgw#763)",
        _MIN_GUARDED_GIB,
    )
    return True


__all__ = ["check_host_ram_move", "install", "HostRamMoveRefusedError"]
