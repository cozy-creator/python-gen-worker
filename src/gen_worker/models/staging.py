"""Pinned host staging + the dedicated H2D copy stream (pgw#674).

Rotating double-buffer serving (WORKER-RESIDENCY-DESIGN, Paul-ratified)
stages the NEXT checkpoint while the current job computes. Two shared
facilities live here:

- :class:`PinnedPool` — bounded accounting for pinned (page-locked) host
  RAM. Pinned memory is unswappable; an unbounded pinned tier can push the
  host into the gw#407 reclaim-thrash livelock exactly like an unbounded
  warm tier. The budget is MEASURED (available RAM minus the residency
  floor, hard-capped at half of total RAM) — no knobs, per the standing
  no-developer-facing-residency-knobs rule. Reservations release by
  weakref finalizer on the pinned tensor, so accounting tracks the actual
  allocation lifetime with no explicit free protocol.

- the process copy stream — a dedicated CUDA stream for weight H2D.
  Copy engines are separate hardware from the SMs, so a pinned-memory H2D
  on this stream runs concurrently with the serving job's compute instead
  of serializing behind it on the default stream (pgw#652 overlap #2).
  Interference is measured from ordinary production traffic
  (DESIGN-RULINGS §1.2), not from a bespoke harness — the
  ``benchmarks.swap_latency overlap`` case was deleted with its
  zero-adopter endpoint wrapper (pgw#883).

CPU-only hosts (and the CPU-only test suite) get honest no-ops: no CUDA
means no copy stream and pinned allocation falls back to pageable.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import weakref
from typing import Any, Callable, Iterator, Optional

from .memory import (
    effective_ram_floor_gb,
    get_available_ram_gb,
    get_total_ram_gb,
)

logger = logging.getLogger(__name__)

_GiB = 1024 ** 3

# Pinned memory is the harshest host allocation (unswappable): never let it
# claim more than half of physical RAM even on an idle host.
_PINNED_TOTAL_FRACTION = 0.5


def _current_group() -> int:
    """Which execution group is asking. Import-local: staging is imported by
    the residency/loading core that topology itself imports."""
    # CYCLE: topology -> residency -> staging; hoisting leaves REPLICATED
    # unbound while residency is still initializing.
    from ..topology import current_device_group

    try:
        return current_device_group()
    except Exception:
        return 0


def _floor_bytes() -> int:
    """The host-RAM floor the warm/pinned tiers must leave alone, in bytes.
    One owner (`memory.effective_ram_floor_gb`, pgw#973 §4.24) — this used to
    restate residency's two constants AND its derivation."""
    return int(effective_ram_floor_gb(get_total_ram_gb()) * _GiB)


class PinnedPool:
    """Bounded pinned host-RAM accounting. Thread-safe.

    ``try_reserve`` admits an allocation against the measured budget;
    :func:`alloc_pinned_like` attaches the matching release to the pinned
    tensor's lifetime. A refusal is not an error — callers fall back to
    pageable memory (slower H2D, still correct).
    """

    def __init__(self, budget_fn: Optional[Callable[[], int]] = None) -> None:
        self._budget_fn = budget_fn
        self._lock = threading.Lock()
        self._reserved = 0
        # pgw#748 phase 1: pinned host RAM is a POD budget, not a per-instance
        # one, and it is the harshest allocation class there is (unswappable).
        # With G execution groups the cap must be SHARED, or group 0 claims
        # the whole 50% and a G=4 degraded pod pages itself to death (§4.3
        # caveat 2). One process gets the honest total for free; this is the
        # fair-share half — each group may claim at most cap/G.
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
        # budget_bytes() measures AVAILABLE ram, which already reflects prior
        # pinned allocations that actually landed; only the total-RAM cap
        # needs the reserved counter (handled inside budget_bytes).
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
    """A pinned host tensor shaped like ``t``, budget-gated through the
    pool. ``None`` = refused or allocation failed; the caller uses pageable
    memory instead. The pool reservation releases when the returned tensor
    is garbage-collected (weakref finalizer), so accounting follows the
    real lifetime."""
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


# ---------------------------------------------------------------------------
# The H2D copy stream
# ---------------------------------------------------------------------------

_stream_lock = threading.Lock()
_streams: dict = {}


def copy_stream(device: Optional[Any] = None) -> Optional[Any]:
    """The dedicated H2D copy stream FOR ONE DEVICE; ``None`` off-CUDA.

    pgw#780 item 4: this used to be a process-wide singleton created on the
    first caller's device — device 0 in practice — so a promote onto
    ``cuda:3`` queued its copies on card 0's stream context (falling through
    to card 3's DEFAULT/compute stream) and then synchronized card 0: the
    overlap-with-compute property was silently lost for every group but 0,
    and the ``finally`` sync guarded the wrong card. One stream per device,
    keyed by the target.

    ``device`` may be a ``torch.device``, an index, or ``None`` (= the
    thread-current device, which handler threads pin per group).
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
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
    """Run enclosed CUDA copies on the device's copy stream (no-op off-CUDA).

    Yields the stream (or ``None``). Callers that need completion before
    returning tensors to compute streams call ``stream.synchronize()`` —
    NEVER ``torch.cuda.synchronize()``, which would also wait on the
    serving job's queued compute kernels."""
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
