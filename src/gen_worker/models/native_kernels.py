"""Native-kernel implementations behind the format-1 baseline.

TCG owns the complete compiled-graph declaration and artifact schema. That
closed schema carries no worker kernel-lane verdict, so the worker must not
smuggle a second, unauthenticated execution policy beside it. Format 1 is
therefore explicitly ``baseline`` linears plus ``dense`` modulation.

The fused/packed implementations and their direct tests remain available for a
future TCG-owned format decision. The prebuilt ``_cozy_kernels`` extension is
also probed independently; extension presence never changes the format-1 lane.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

NATIVE_ENV = "GEN_WORKER_NATIVE_KERNELS"
NATIVE_LIB_ENV = "GEN_WORKER_NATIVE_KERNELS_LIB"

_EXT_DEFAULT = "/opt/cozy/native/libcozy_kernels.so"
_EXT_NAMESPACE = "cozy_kernels"

# Two independent decisions, each cached with the reason that produced it.
_EXECUTION_LANES: Dict[str, str] = {}
_REASONS: Dict[str, str] = {}


def native_kernels_requested() -> Optional[bool]:
    """Env tri-state: True (opt-in), False (kill), None (unset => off while
    the rollout is env-gated)."""
    raw = os.environ.get(NATIVE_ENV, "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return None


def _record(axis: str, value: str, reason: str) -> str:
    _EXECUTION_LANES[axis], _REASONS[axis] = value, reason
    return value


def _decide(
    axis: str, off: str,
) -> str:
    """Return the one format-1 lane, cached with an explicit reason."""
    if axis in _EXECUTION_LANES:
        return _EXECUTION_LANES[axis]

    requested = native_kernels_requested()
    if requested is False:
        logger.info("native kernels [%s]: %s (kill-switch)", axis, off)
        return _record(axis, off, f"{NATIVE_ENV}=0 (kill-switch)")
    if requested is None:
        reason = (f"dormant — rollout is env-gated (pgw#859 G0 / pgw#865), "
                  f"set {NATIVE_ENV}=1 to opt in")
        logger.info("native kernels [%s]: %s (%s)", axis, off, reason)
        return _record(axis, off, reason)
    reason = "TCG compiled-graph format 1 fixes the baseline execution lane"
    logger.info("native kernels [%s]: %s (%s)", axis, off, reason)
    return _record(axis, off, reason)


def svdq_linear_execution_lane() -> str:
    """``"fused"`` | ``"baseline"`` for the W4A4 linears in this process.
    Call at LOAD time — the first call compiles kernels and self-checks."""
    return _decide("linear", "baseline")


def svdq_modulation_execution_lane() -> str:
    """``"packed"`` | ``"dense"`` for the W4A16 AdaLN modulation. Independent
    of the linear lane: a card that wants the baseline linears can still want
    packed modulation, and sm_100 does."""
    return _decide("modulation", "dense")


def svdq_linear_execution_lane_reason() -> str:
    """The recorded reason for the linear-lane decision."""
    svdq_linear_execution_lane()
    return _REASONS["linear"]


def svdq_modulation_execution_lane_reason() -> str:
    """The recorded reason for the modulation-lane decision."""
    svdq_modulation_execution_lane()
    return _REASONS["modulation"]


def reset_native_kernels_arming() -> None:
    """Forget both lane decisions (tests only)."""
    _EXECUTION_LANES.clear()
    _REASONS.clear()


# ---------------------------------------------------------------------------
# The C++/CUDA extension (`csrc/`, prebuilt .so). Probed separately; no lane
# depends on it until a kernel actually ships there.
# ---------------------------------------------------------------------------


def extension_path() -> Optional[str]:
    """Where the prebuilt extension would be, or None. Env override first,
    then the base-image bake path."""
    override = os.environ.get(NATIVE_LIB_ENV, "").strip()
    if override:
        return override
    if os.path.exists(_EXT_DEFAULT):
        return _EXT_DEFAULT
    return None


def load_extension() -> Optional[str]:
    """Load the extension library. Returns None on success, else the typed
    reason it is unavailable."""
    path = extension_path()
    if path is None:
        return (f"no extension library (checked {NATIVE_LIB_ENV} and "
                f"{_EXT_DEFAULT})")
    if not os.path.exists(path):
        return f"extension library {path} does not exist"
    try:
        import torch

        torch.ops.load_library(path)
    except Exception as exc:  # noqa: BLE001 — surfaced, never fatal
        return f"extension library {path} failed to load: {exc}"
    return None


def extension_ops() -> Any:
    """The extension's op namespace (valid only after load_extension())."""
    import torch

    return getattr(torch.ops, _EXT_NAMESPACE)


def extension_available() -> bool:
    """Extension loaded + probe op present + (GPU) probe numerics.
    Never raises."""
    try:
        reason = load_extension()
        if reason is not None:
            logger.info("native extension: unavailable — %s", reason)
            return False
        import torch

        ns = extension_ops()
        if not hasattr(ns, "probe_add_one"):
            logger.warning("native extension: loaded but probe op missing")
            return False
        if cuda_ready():
            x = torch.arange(8, device="cuda", dtype=torch.float32)
            y = ns.probe_add_one(x)
            if not torch.equal(y, x + 1):
                logger.warning(
                    "native extension: probe op numerics FAILED — refusing")
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("native extension: probe raised (%s)", exc)
        return False


__all__ = [
    "NATIVE_ENV",
    "NATIVE_LIB_ENV",
    "extension_available",
    "extension_ops",
    "extension_path",
    "load_extension",
    "native_kernels_requested",
    "reset_native_kernels_arming",
    "svdq_linear_execution_lane",
    "svdq_linear_execution_lane_reason",
    "svdq_modulation_execution_lane",
    "svdq_modulation_execution_lane_reason",
]
