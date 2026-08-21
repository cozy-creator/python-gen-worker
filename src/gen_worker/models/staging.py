"""Pinned host staging + the dedicated H2D copy stream."""

from __future__ import annotations

import contextlib
import logging
import threading
import weakref
from typing import Any, Callable, Iterator, Optional
from ..hostfacts import cuda_ready

from .memory import (
    effective_ram_floor_gb,
    get_available_ram_gb,
    get_total_ram_gb,
)

logger = logging.getLogger(__name__)

_GiB = 1024 ** 3

_PINNED_TOTAL_FRACTION = 0.5


def _current_group() -> int:
    from ..topology import current_device_group

    try:
        return current_device_group()
    except Exception:
        return 0


def _floor_bytes() -> int:
    return int(effective_ram_floor_gb(get_total_ram_gb()) * _GiB)


class PinnedPool:
    """Bounded pinned host-RAM accounting."""

    def __init__(self, budget_fn: Optional[Callable[[], int]] = None) -> None:
        self._budget_fn = budget_fn
        self._lock = threading.Lock()
        self._reserved = 0
        self._groups = 1
        self._reserved_by_group: dict = {}

    def set_group_count(self, groups: int) -> None:
        with self._lock:
            self._groups = max(1, int(groups))

    @property
    def group_count(self) -> int:
        return self._groups

    def _group_cap(self) -> int:
        total = int(get_total_ram_gb() * _GiB)
        if total <= 0 or self._groups <= 1:
            return 0
        return int(total * _PINNED_TOTAL_FRACTION) // self._groups

    def budget_bytes(self) -> int:
        if self._budget_fn is not None:
            return max(0, int(self._budget_fn()))
        available = int(get_available_ram_gb() * _GiB)
        total = int(get_total_ram_gb() * _GiB)
        headroom = available - _floor_bytes()
        if total > 0:
            cap = int(total * _PINNED_TOTAL_FRACTION)
            remaining = cap - self.reserved_bytes()
            headroom = min(headroom, remaining)
        share = self._group_cap()
        if share:
            with self._lock:
                mine = int(self._reserved_by_group.get(_current_group(), 0))
            headroom = min(headroom, share - mine)
        return max(0, headroom)

    def reserved_bytes(self) -> int:
        with self._lock:
            return self._reserved

    def try_reserve(self, nbytes: int) -> bool:
        nbytes = int(nbytes)
        if nbytes <= 0:
            return True
        if nbytes > self.budget_bytes():
            return False
        group = _current_group()
        with self._lock:
            self._reserved += nbytes
            self._reserved_by_group[group] = int(
                self._reserved_by_group.get(group, 0)) + nbytes
        return True

    def release(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        if nbytes <= 0:
            return
        group = _current_group()
        with self._lock:
            self._reserved = max(0, self._reserved - nbytes)
            self._reserved_by_group[group] = max(
                0, int(self._reserved_by_group.get(group, 0)) - nbytes)


_pool = PinnedPool()


def pinned_pool() -> PinnedPool:
    return _pool


def set_pinned_pool(pool: PinnedPool) -> PinnedPool:
    """Swap the process pool (tests inject a deterministic budget)."""
    global _pool
    prev, _pool = _pool, pool
    return prev


def alloc_pinned_like(torch: Any, t: Any) -> Optional[Any]:
    """A pinned host tensor shaped like ``t``, budget-gated through the pool."""
    nbytes = int(t.numel()) * int(t.element_size())
    if not _pool.try_reserve(nbytes):
        logger.info(
            "pinned pool refused %.2fGiB (reserved %.2fGiB, budget %.2fGiB)",
            nbytes / _GiB, _pool.reserved_bytes() / _GiB,
            _pool.budget_bytes() / _GiB,
        )
        return None
    try:
        host = torch.empty_like(t, device="cpu", pin_memory=True)
    except Exception:
        _pool.release(nbytes)
        return None
    weakref.finalize(host, _pool.release, nbytes)
    return host


_stream_lock = threading.Lock()
_streams: dict = {}


def copy_stream(device: Optional[Any] = None) -> Optional[Any]:
    """The dedicated H2D copy stream FOR ONE DEVICE; ``None`` off-CUDA."""
    try:
        import torch
    except Exception:
        return None
    if not cuda_ready():
        return None
    if device is None:
        index = int(torch.cuda.current_device())
    else:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type != "cuda":
            return None
        index = int(dev.index) if dev.index is not None else int(torch.cuda.current_device())
    with _stream_lock:
        stream = _streams.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            _streams[index] = stream
        return stream


@contextlib.contextmanager
def copy_stream_ctx(device: Optional[Any] = None) -> Iterator[Optional[Any]]:
    """Run enclosed CUDA copies on the device's copy stream (no-op off-CUDA)."""
    stream = copy_stream(device)
    if stream is None:
        yield None
        return
    import torch

    with torch.cuda.stream(stream):
        yield stream


__all__ = [
    "PinnedPool",
    "pinned_pool",
    "set_pinned_pool",
    "alloc_pinned_like",
    "copy_stream",
    "copy_stream_ctx",
]
