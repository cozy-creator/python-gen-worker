"""Lane serve gate (gw#551): promote-on-use for LRU-swappable pipelines.

Multi-lane endpoints (gw#479) dispatch to a lane handler-side (pgw#509), so
the executor cannot know pre-dispatch which lane a request will run. When the
lanes overcommit VRAM, one sits demoted in host RAM — and nothing between
"demoted" and "the handler calls the pipeline" used to re-promote it: the
pipeline executed with its transformer on cpu and crashed mid-denoise (the
te#79 addmm / cuda-generator shapes).

The gate closes that hole at the shared machinery level: each lane pipeline's
``__call__`` is wrapped (dynamic subclass — object identity, isinstance and
attributes all preserved) to first pin the lane and promote it if demoted,
LRU-swapping the idle sibling out. Alternating t2i/edit traffic on one worker
becomes swap-per-alternation: degraded-but-correct, logged loudly with timing.
When VRAM truly cannot fit the lane it queues briefly, then raises the
executor-injected retryable error — never executes a cpu-resident lane.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Type

from .memory import get_available_vram_gb
from .residency import Residency, Tier, _obj_offload_hooked

logger = logging.getLogger(__name__)

_GATE_ATTR = "_cozy_lane_gate"
_GATED_FLAG = "_cozy_lane_gated"

_GiB = 1024 ** 3

# How long a call waits for VRAM headroom before failing retryable: the idle
# sibling's demote is seconds; anything longer means a genuinely stuck card.
_DEFAULT_WAIT_S = 45.0
_POLL_S = 0.25


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _inference_mode_off() -> Any:
    """Endpoints call pipelines under ``torch.inference_mode()``; tensors the
    swap creates in that scope would be inference tensors (poisonous outside
    it). Disable it around the promote."""
    try:
        import torch

        return torch.inference_mode(False)
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


class LaneGate:
    """Ensures one lane ref is execution-ready around each pipeline call."""

    def __init__(
        self,
        *,
        ref: str,
        residency: Residency,
        label: str = "",
        retry_exc: Type[Exception] = RuntimeError,
        wait_s: float = _DEFAULT_WAIT_S,
        on_swap: Optional[Callable[[str, int], None]] = None,
        offload_fallback: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.ref = ref
        self.residency = residency
        self.label = label or ref
        self.retry_exc = retry_exc
        self.wait_s = wait_s
        self.on_swap = on_swap  # (ref, promote_ms) — degraded-serve reporting
        # When promote truly cannot fit: arm a coherent offload rung instead
        # of failing (monolithic pipelines only — offload hooks on a shared
        # component would poison sibling lanes).
        self.offload_fallback = offload_fallback
        # Serializes swap decisions across threads for THIS lane; residency
        # itself is thread-safe, but two callers promoting the same demoted
        # lane should pay the wait once.
        self._lock = threading.Lock()

    @contextmanager
    def ensure_resident(self) -> Iterator[None]:
        """Pin the lane for the duration and promote it first if demoted."""
        res = self.residency
        with res.executing(self.ref):
            self._promote_if_needed()
            yield

    def _promote_if_needed(self) -> None:
        if not _cuda_available():
            return  # CPU-only host: everything already runs on cpu
        res = self.residency
        if not res.movable(self.ref):
            # Offload-hooked pipelines own their placement; object-less refs
            # have nothing to move.
            return
        obj = res.obj(self.ref)
        if obj is not None and _obj_offload_hooked(obj):
            return  # block-window offload (ie#468): hooks own placement
        with self._lock, _inference_mode_off():
            tier = res.tier(self.ref)
            if tier is Tier.VRAM:
                # Paranoid completeness walk (metadata-only when clean): a
                # crashed rollback must not serve a mixed-device pipeline.
                if res.promote(self.ref) or res.tier(self.ref) is not Tier.RAM:
                    return
                tier = Tier.RAM
            if tier is not Tier.RAM:
                return
            t0 = time.monotonic()
            deadline = t0 + self.wait_s
            while True:
                if res.promote(self.ref):
                    ms = int((time.monotonic() - t0) * 1000)
                    logger.warning(
                        "LANE_SWAP model=%s promote_ms=%d free_gb=%.1f: served "
                        "after RAM->VRAM swap (lanes overcommit VRAM; "
                        "alternating traffic swaps per alternation — degraded "
                        "but correct)",
                        self.label, ms, get_available_vram_gb(),
                    )
                    if self.on_swap is not None:
                        try:
                            self.on_swap(self.ref, ms)
                        except Exception:
                            logger.exception("lane-swap callback failed")
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(_POLL_S)
            if self.offload_fallback is not None:
                try:
                    if self.offload_fallback():
                        logger.warning(
                            "LANE_OFFLOAD model=%s: promote cannot fit (free "
                            "%.1f GiB); serving CPU-offloaded",
                            self.label, get_available_vram_gb(),
                        )
                        return
                except Exception:
                    logger.exception(
                        "offload fallback failed for %s", self.label)
            raise self.retry_exc(
                f"lane {self.label} cannot promote to VRAM (waited "
                f"{self.wait_s:.0f}s for headroom; free "
                f"{get_available_vram_gb():.1f} GiB); retrying"
            )


def arm_lane_gate(pipe: Any, gate: LaneGate) -> bool:
    """Wrap ``pipe.__call__`` with the gate. Idempotent: an already-gated
    pipeline just gets the fresh gate. Object identity and isinstance are
    preserved (dynamic subclass with the same class name)."""
    if pipe is None:
        return False
    if getattr(type(pipe), _GATED_FLAG, False):
        object.__setattr__(pipe, _GATE_ATTR, gate)
        return True
    cls = type(pipe)
    # Only wrap classes that define an INSTANCE ``__call__`` somewhere in the
    # MRO — plain ``getattr(cls, "__call__")`` on a call-less class resolves
    # to the metaclass constructor, which must never be captured.
    if not any("__call__" in vars(k) for k in cls.__mro__):
        return False
    base_call = cls.__call__

    def _gated_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        g = getattr(self, _GATE_ATTR, None)
        if g is None:
            return base_call(self, *args, **kwargs)
        with g.ensure_resident():
            return base_call(self, *args, **kwargs)

    try:
        gated = type(cls.__name__, (cls,), {
            "__call__": _gated_call,
            _GATED_FLAG: True,
            "__module__": cls.__module__,
        })
        pipe.__class__ = gated
        object.__setattr__(pipe, _GATE_ATTR, gate)
    except Exception as exc:
        logger.warning(
            "lane gate could not wrap %s (%s: %s); lane relies on eager "
            "promotion only", cls.__name__, type(exc).__name__, exc,
        )
        return False
    return True


__all__ = ["LaneGate", "arm_lane_gate"]
