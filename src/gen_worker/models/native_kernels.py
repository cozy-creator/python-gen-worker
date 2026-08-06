"""Native-kernel dispatch (pgw#860, pgw#863, pgw#947).

TWO decisions, each made once per process at LOAD time and each INDEPENDENT
of the other (pgw#863 — binding them to one switch cost sm_100 either 9.5 GB
of residency or 19% of its step time, with no way to take both):

- ``svdq_linear_lane()`` — FUSED W4A4 linears (pgw#862 triton kernels;
  `_cozy_kernels` C++ ops if/when a lane needs them) or the baseline unfused
  chain. A throughput question.
- ``svdq_modulation_lane()`` — PACKED W4A16 AdaLN modulation (pgw#864) or
  dense bf16. A residency question.

**Neither decision is made here, and neither is derived from the SM
(pgw#947).** Which combination of the two wins on a card is a per-card FACT,
not a derivable one — a custom op is opaque to inductor, so our fusion beats
inductor's own on sm_120 and loses to it on sm_100 — and it used to live in
hand-maintained SM tuples informed by ~$12 benchmark campaigns per card. It
is now MEASURED at mint on the card the cell is being minted for, ranked by
fit-constrained speed (which prices the residency axis and the throughput
axis in one comparison instead of hard-coding either), recorded into the cell
as ONE combined lane, and read back here: see ``gen_worker.kernel_lane``.
This module's whole job is to project the pin onto its axis, prove that
axis's kernels still self-check, and say loudly what it did.

Order of decision, PER AXIS, each step typed and never silent-wrong:

- env kill-switch first: ``GEN_WORKER_NATIVE_KERNELS=0`` forces baseline.
  Rollout is still env-GATED (pgw#859 G0: unset means OFF, ``=1`` opts in)
  because flipping that gate belongs to pgw#865, not to this mechanism. The
  env is on th#1445's elimination list and MUST NOT grow new meanings — it
  gates the ROLLOUT, it never picks a lane.
- the recorded verdict: ``kernel_lane.pinned()``, set by the executor from
  the delivered cell before ``setup()`` runs. No pin (eager boot, pre-pgw#947
  cell, unreadable envelope) => the declared conservative default with the
  typed reason that says which of those it was.
- numerics self-check, per axis: an armed value still has to pass THIS
  axis's check on THIS box (fused: activation-quant BIT-IDENTITY vs the
  pgw#685 reference chain; packed: output vs the decoded bf16 Linear). A gap
  degrades that axis ALONE — the other keeps its verdict, which is the whole
  point of the pgw#863 split.
- contract mismatches inside an armed lane raise typed errors; they are bugs,
  not degrade paths.

The prebuilt ``_cozy_kernels`` extension (.so baked into the shared cuda base
image; see csrc/) is probed independently — no lane needs it, so extension
absence never blocks one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from .. import kernel_path
from .svdq_fused import fused_self_check
from .svdq_awq_packed import awq_packed_self_check

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


def _cuda_gap() -> Optional[str]:
    """Why no native lane can run here at all, or None."""
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "native svdq kernels require a CUDA GPU"
    return None


def _fused_linear_self_check() -> Optional[str]:
    """Numerics only. The SM is deliberately NOT consulted: a cell that names
    the fused linear was minted on this compute capability (``sm`` is a
    cell-key axis), so a capability question here would only re-derive what
    the verdict already proved by running."""
    gap = _cuda_gap()
    if gap is not None:
        return gap

    return fused_self_check()


def _packed_modulation_self_check() -> Optional[str]:
    """Same contract as the linear check, for the W4A16 modulation kernel."""
    gap = _cuda_gap()
    if gap is not None:
        return gap

    return awq_packed_self_check()


def _record(axis: str, value: str, reason: str) -> str:
    _EXECUTION_LANES[axis], _REASONS[axis] = value, reason
    return value


def _decide(
    axis: str,
    armed: str,
    off: str,
    project: Callable[[str], str],
    self_check: Callable[[], Optional[str]],
) -> str:
    """One axis's arming decision: env, then the recorded verdict, then this
    axis's numerics. Cached per process with the reason that produced it."""
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

    execution_lane, execution_lane_reason = kernel_path.pinned()
    if execution_lane is None:
        # Nothing pinned a lane for this load: no cell was delivered, or the
        # executor never reached the adoption hook. The DECLARED default, and
        # it names itself — there is no SM allowlist left to fall back on.
        reason = (f"{kernel_path.REASON_ABSENT}: nothing recorded a measured "
                  f"kernel-lane verdict for this load; serving the declared "
                  f"default {kernel_path.DEFAULT_EXECUTION_LANE!r}")
        logger.warning("native kernels [%s]: %s", axis, reason)
        return _record(axis, project(kernel_path.DEFAULT_EXECUTION_LANE), reason)

    want = project(execution_lane)
    if want != armed:
        logger.info("native kernels [%s]: %s (%s)", axis, want, execution_lane_reason)
        return _record(axis, want, execution_lane_reason)
    try:
        gap = self_check()
    except Exception as exc:  # noqa: BLE001 — any self-check gap => degrade
        gap = f"self-check raised: {type(exc).__name__}: {exc}"
    if gap:
        reason = (f"verdict said {want!r} but the {axis} kernels do not "
                  f"self-check here — {gap}")
        logger.warning("native kernels [%s]: NOT armed — %s; %s serves the "
                       "same artifact", axis, reason, off)
        return _record(axis, off, reason)
    reason = f"{execution_lane_reason}; {axis} numerics self-check passed"
    logger.info("native kernels [%s]: %s armed (%s)", axis, armed, reason)
    return _record(axis, armed, reason)


def svdq_linear_execution_lane() -> str:
    """``"fused"`` | ``"baseline"`` for the W4A4 linears in this process.
    Call at LOAD time — the first call compiles kernels and self-checks."""
    return _decide(
        kernel_path.AXIS_LINEAR,
        kernel_path.LINEAR_FUSED, kernel_path.LINEAR_BASELINE,
        kernel_path.linear_of, _fused_linear_self_check)


def svdq_modulation_execution_lane() -> str:
    """``"packed"`` | ``"dense"`` for the W4A16 AdaLN modulation. Independent
    of the linear lane: a card that wants the baseline linears can still want
    packed modulation, and sm_100 does."""
    return _decide(
        kernel_path.AXIS_MODULATION,
        kernel_path.MOD_PACKED, kernel_path.MOD_DENSE,
        kernel_path.modulation_of, _packed_modulation_self_check)


def svdq_linear_execution_lane_reason() -> str:
    """The recorded reason for the linear-lane decision."""
    svdq_linear_execution_lane()
    return _REASONS[kernel_path.AXIS_LINEAR]


def svdq_modulation_execution_lane_reason() -> str:
    """The recorded reason for the modulation-lane decision."""
    svdq_modulation_execution_lane()
    return _REASONS[kernel_path.AXIS_MODULATION]


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
    "svdq_linear_execution_lane",
    "svdq_linear_execution_lane_reason",
    "svdq_modulation_execution_lane",
    "svdq_modulation_execution_lane_reason",
]
