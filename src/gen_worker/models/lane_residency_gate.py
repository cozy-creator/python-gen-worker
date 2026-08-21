"""Lane serve gate: promote-on-use for LRU-swappable pipelines."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Type

from .. import activity as activity_mod
from ..stall import SilenceWindow
from .memory import get_available_vram_gb
from .residency import Residency, Tier, _obj_offload_hooked
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

_GATE_ATTR = "_cozy_lane_gate"
_GATED_FLAG = "_cozy_lane_gated"

_GiB = 1024 ** 3

_HEADROOM_SILENCE_WINDOW_S = 45.0
_POLL_S = 0.25
_HEADROOM_STEP_GB = 1.0 / 16.0


def _cuda_available() -> bool:
    return cuda_ready()


def _inference_mode_off() -> Any:
    try:
        import torch

        return torch.inference_mode(False)
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


class LaneResidencyGate:
    """Ensures one lane ref is execution-ready around each pipeline call."""

    def __init__(
        self,
        *,
        ref: str,
        residency: Residency,
        label: str = "",
        retry_exc: Type[Exception] = RuntimeError,
        wait_s: float = _HEADROOM_SILENCE_WINDOW_S,
        on_swap: Optional[Callable[[str, int], None]] = None,
        offload_fallback: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.ref = ref
        self.residency = residency
        self.label = label or ref
        self.retry_exc = retry_exc
        self.wait_s = wait_s
        self.on_swap = on_swap
        self.offload_fallback = offload_fallback
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
            return
        res = self.residency
        if not res.movable(self.ref):
            return
        obj = res.obj(self.ref)
        if obj is not None and _obj_offload_hooked(obj):
            return
        with self._lock, _inference_mode_off():
            tier = res.tier(self.ref)
            if tier is Tier.VRAM:
                if res.promote(self.ref) or res.tier(self.ref) is not Tier.RAM:
                    return
                tier = Tier.RAM
            if tier is not Tier.RAM:
                return
            t0 = time.monotonic()
            headroom = SilenceWindow(self.wait_s)
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
                headroom.touch_if_changed(
                    round(get_available_vram_gb() / _HEADROOM_STEP_GB))
                if headroom.stalled():
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
                        activity_mod.emit_event(
                            activity_mod.KIND_SERVE_DEGRADE,
                            f"ref={self.ref} label={self.label} "
                            f"free_gb={get_available_vram_gb():.1f}: promote "
                            f"cannot fit and free VRAM stopped moving for "
                            f"{self.wait_s:.0f}s (waited "
                            f"{time.monotonic() - t0:.0f}s); serving "
                            "CPU-offloaded",
                            phase="lane_offload_engaged",
                        )
                        return
                except Exception:
                    logger.exception(
                        "offload fallback failed for %s", self.label)
            raise self.retry_exc(
                f"lane {self.label} cannot promote to VRAM (free VRAM "
                f"stopped moving for {self.wait_s:.0f}s after "
                f"{time.monotonic() - t0:.0f}s; free "
                f"{get_available_vram_gb():.1f} GiB); retrying"
            )


def arm_lane_residency_gate(pipe: Any, gate: LaneResidencyGate) -> bool:
    """Wrap ``pipe.__call__`` with the gate."""
    if pipe is None:
        return False
    if getattr(type(pipe), _GATED_FLAG, False):
        object.__setattr__(pipe, _GATE_ATTR, gate)
        return True
    cls = type(pipe)
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
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"ref={gate.ref} label={gate.label} cls={cls.__name__}: lane "
            f"gate could not wrap __call__; demoted-lane promote-on-use "
            f"protection is absent: {type(exc).__name__}: {exc}",
            phase="lane_gate_unarmed",
        )
        return False
    return True


__all__ = ["LaneResidencyGate", "arm_lane_residency_gate"]
