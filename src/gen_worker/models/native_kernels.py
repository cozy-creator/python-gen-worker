"""Native-kernel dispatch + capability probe (pgw#860).

One decision, made once per process at LOAD time: does the svdq serving path
run the FUSED native lane (pgw#862 triton kernels today; `_cozy_kernels` C++
ops if/when a lane needs them) or the baseline unfused lane. The decision is
typed, logged, and never silent-wrong:

- env kill-switch first: ``GEN_WORKER_NATIVE_KERNELS=0`` forces baseline.
  Rollout is env-GATED (Paul, pgw#859 G0): unset means OFF; ``=1`` opts in.
- capability probe: CUDA device + a supported SM + the fused kernels compile
  AND pass the numerics self-check (activation-quant BIT-IDENTITY vs the
  pgw#685 reference chain + GEMM tolerance) — any gap degrades to baseline
  with the reason logged, same artifact, no refusal.
- contract mismatches inside an armed lane raise typed errors; they are bugs,
  not degrade paths.

The prebuilt ``_cozy_kernels`` extension (.so baked into the shared cuda base
image; see csrc/) is probed independently — the fused-triton lane does not
need it, so extension absence never blocks the lane.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

NATIVE_ENV = "GEN_WORKER_NATIVE_KERNELS"
NATIVE_LIB_ENV = "GEN_WORKER_NATIVE_KERNELS_LIB"

# dot_scaled lowers to native block-scaled MMA on Blackwell (PTX census banked
# in pgw#862: sm_120a => kind::mxf4nvf4, sm_100a => tcgen05). 103/121 are
# family variants triton targets per-device; the self-check still gates them.
FUSED_SMS = (100, 103, 120, 121)

_EXT_DEFAULT = "/opt/cozy/native/libcozy_kernels.so"
_EXT_NAMESPACE = "cozy_kernels"

_LANE: Optional[str] = None
_LANE_REASON: str = ""


def native_kernels_requested() -> Optional[bool]:
    """Env tri-state: True (opt-in), False (kill), None (unset => off while
    the rollout is env-gated)."""
    raw = os.environ.get(NATIVE_ENV, "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return None


def _probe_fused_lane() -> Optional[str]:
    """Why the fused lane cannot arm HERE, or None when it can."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "fused svdq lane requires a CUDA GPU"
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm not in FUSED_SMS:
        return (f"fused svdq lane needs Blackwell block-scaled MMA "
                f"(sm_{'/'.join(str(s) for s in FUSED_SMS)}); this GPU is "
                f"sm_{sm}")
    from .svdq_awq_packed import awq_packed_self_check
    from .svdq_fused import fused_self_check

    reason = fused_self_check()
    if reason is not None:
        return reason
    reason = awq_packed_self_check()
    if reason is not None:
        return f"awq packed lane: {reason}"
    return None


def svdq_execution_lane() -> str:
    """``"fused"`` | ``"baseline"`` for this process. Call at LOAD time — the
    first call compiles kernels and runs the self-check."""
    global _LANE, _LANE_REASON

    if _LANE is not None:
        return _LANE
    requested = native_kernels_requested()
    if requested is False:
        _LANE, _LANE_REASON = "baseline", f"{NATIVE_ENV}=0 (kill-switch)"
        logger.info("native kernels: baseline lane (%s)", _LANE_REASON)
        return _LANE
    if requested is None:
        _LANE, _LANE_REASON = "baseline", (
            f"dormant — rollout is env-gated, set {NATIVE_ENV}=1 to opt in")
        logger.info("native kernels: baseline lane (%s)", _LANE_REASON)
        return _LANE
    try:
        reason = _probe_fused_lane()
    except Exception as exc:  # noqa: BLE001 — any probe gap => baseline
        reason = f"probe raised: {exc}"
    if reason is not None:
        _LANE, _LANE_REASON = "baseline", reason
        logger.warning("native kernels: requested but NOT armed — %s; "
                       "baseline lane serves the same artifact", reason)
    else:
        _LANE, _LANE_REASON = "fused", "probe + numerics self-check passed"
        logger.info("native kernels: FUSED lane armed (%s)", _LANE_REASON)
    return _LANE


def svdq_lane_reason() -> str:
    """The recorded reason for the current lane decision."""
    if _LANE is None:
        svdq_execution_lane()
    return _LANE_REASON


def reset_native_kernels_arming() -> None:
    """Forget the lane decision (tests only)."""
    global _LANE, _LANE_REASON

    _LANE, _LANE_REASON = None, ""


# ---------------------------------------------------------------------------
# The C++/CUDA extension (`csrc/`, prebuilt .so). Probed separately; no lane
# depends on it until a kernel actually ships there (pgw#863+).
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
        if torch.cuda.is_available():
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
    "FUSED_SMS",
    "NATIVE_ENV",
    "NATIVE_LIB_ENV",
    "extension_available",
    "extension_ops",
    "extension_path",
    "load_extension",
    "native_kernels_requested",
    "reset_native_kernels_arming",
    "svdq_execution_lane",
    "svdq_lane_reason",
]
