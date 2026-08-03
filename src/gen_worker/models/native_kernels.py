"""Native-kernel dispatch (pgw#860, pgw#946).

One decision, made once per process at LOAD time: does the svdq serving path
run the FUSED native lane (pgw#862 triton kernels today; `_cozy_kernels` C++
ops if/when a lane needs them) or the baseline unfused lane.

**The decision is not made here and is not derived from the SM (pgw#946).**
Which lane is faster is a per-card FACT — a custom op is opaque to inductor,
so our fusion beats inductor's own on sm_120 and loses to it on sm_100 — and
it used to live in a hand-maintained SM tuple informed by ~$12 benchmark
campaigns per card. It is now MEASURED at mint on the card the cell is being
minted for, recorded into the cell, and read back here: see
``gen_worker.kernel_lane``. This module's whole job is to apply the pin,
prove the kernels still self-check, and say loudly what it did.

Order of decision, each step typed and never silent-wrong:

- env kill-switch first: ``GEN_WORKER_NATIVE_KERNELS=0`` forces baseline.
  Rollout is still env-GATED (pgw#859 G0: unset means OFF, ``=1`` opts in)
  because flipping that gate belongs to pgw#865, not to this mechanism. The
  env is on th#1445's elimination list and MUST NOT grow new meanings — it
  gates the ROLLOUT, it never picks a lane.
- the recorded verdict: ``kernel_lane.pinned()``, set by the executor from
  the delivered cell before ``setup()`` runs. No pin (eager boot, pre-pgw#946
  cell, unreadable envelope) => the declared conservative default with the
  typed reason that says which of those it was.
- numerics self-check: a verdict of ``fused`` still has to pass the
  activation-quant BIT-IDENTITY check (vs the pgw#685 reference chain) and
  the awq-packed check on THIS box. A gap degrades to baseline with the
  reason logged — same artifact, no refusal.
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

from .. import kernel_lane

logger = logging.getLogger(__name__)

NATIVE_ENV = "GEN_WORKER_NATIVE_KERNELS"
NATIVE_LIB_ENV = "GEN_WORKER_NATIVE_KERNELS_LIB"

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


def _self_check_gap() -> Optional[str]:
    """Why an ARMED fused verdict cannot execute on this box, or None.

    Numerics only. The SM is not consulted: a cell that says ``fused`` was
    minted on this compute capability (``sm`` is a cell-key axis), so a
    capability question here would only ever re-derive what the verdict
    already proved by running.
    """
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "fused svdq lane requires a CUDA GPU"
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
    first call runs the numerics self-check."""
    global _LANE, _LANE_REASON

    if _LANE is not None:
        return _LANE
    baseline = kernel_lane.LANE_BASELINE
    requested = native_kernels_requested()
    if requested is False:
        _LANE, _LANE_REASON = baseline, f"{NATIVE_ENV}=0 (kill-switch)"
        logger.info("native kernels: baseline lane (%s)", _LANE_REASON)
        return _LANE
    if requested is None:
        _LANE, _LANE_REASON = baseline, (
            f"dormant — rollout is env-gated (pgw#859 G0 / pgw#865), set "
            f"{NATIVE_ENV}=1 to opt in")
        logger.info("native kernels: baseline lane (%s)", _LANE_REASON)
        return _LANE

    verdict_lane, verdict_reason = kernel_lane.pinned()
    if verdict_lane is None:
        # Nothing pinned a lane for this load: no cell was delivered, or the
        # executor never reached the adoption hook. The DECLARED default, and
        # it names itself.
        _LANE, _LANE_REASON = kernel_lane.DEFAULT_LANE, (
            f"{kernel_lane.REASON_ABSENT}: nothing recorded a measured "
            f"kernel-lane verdict for this load; serving the declared "
            f"default {kernel_lane.DEFAULT_LANE!r}")
        logger.warning("native kernels: %s", _LANE_REASON)
        return _LANE
    if verdict_lane == baseline:
        _LANE, _LANE_REASON = baseline, verdict_reason
        logger.info("native kernels: baseline lane (%s)", _LANE_REASON)
        return _LANE

    try:
        gap = _self_check_gap()
    except Exception as exc:  # noqa: BLE001 — any probe gap => baseline
        gap = f"self-check raised: {exc}"
    if gap is not None:
        _LANE, _LANE_REASON = baseline, (
            f"verdict said {verdict_lane!r} but the kernels do not "
            f"self-check here — {gap}")
        logger.warning("native kernels: NOT armed — %s; baseline lane serves "
                       "the same artifact", _LANE_REASON)
    else:
        _LANE, _LANE_REASON = kernel_lane.LANE_FUSED, (
            f"{verdict_reason}; numerics self-check passed")
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
