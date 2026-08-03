"""Native-kernel dispatch + capability probe (pgw#860).

TWO decisions, each made once per process at LOAD time and each INDEPENDENT of
the other (pgw#863 — binding them to one switch cost sm_100 either 9.5 GB of
residency or 19% of its step time, with no way to take both):

- ``svdq_linear_lane()`` — FUSED W4A4 linears (pgw#862 triton kernels;
  `_cozy_kernels` C++ ops if/when a lane needs them) or the baseline unfused
  chain. Armed only where the fused path is measurably faster in the
  PRODUCTION (compiled) posture.
- ``svdq_modulation_lane()`` — PACKED W4A16 AdaLN modulation (pgw#864) or
  dense bf16. A residency win rather than a throughput one, so it arms on
  every Blackwell card.

Both decisions are typed, logged, and never silent-wrong:

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

# Blackwell block-scaled MMA silicon (PTX census banked in pgw#862: sm_120a =>
# kind::mxf4nvf4, sm_100a => tcgen05). 103/121 are family variants triton
# targets per-device; the self-checks still gate them.
BLACKWELL_SMS = (100, 103, 120, 121)

# Where the FUSED W4A4 LINEAR is the faster serving path — measured, per card,
# in the PRODUCTION (torch.compile) posture, which is the only posture that
# decides anything:
#   sm_120 (RTX 5090): fused+compiled 785 ms/step beats the baseline lane and
#     nunchaku's 843 (pgw#862 final table).
#   sm_100 (B200): fused+compiled 385 ms/step LOSES to the SAME artifact on the
#     baseline lane compiled, 312 ms/step (-19% at 1328^2, -34% at 1024^2;
#     pgw#863 run b200-r3, pod aokw0wficrhwkb). Inductor fuses the baseline's
#     open elementwise chain better than our custom ops can, and our ops are
#     opaque to it — so on this card the fusion is a pessimisation.
# sm_100/103 are therefore deliberately ABSENT. This is a measurement, not a
# capability gap: the kernels arm, they are bit-identical, they are simply
# slower there.
FUSED_LINEAR_SMS = (120, 121)

# Where the PACKED W4A16 modulation is worth arming. It is a pure residency
# win — 22.8 -> 13.3 GB transformer-resident on B200, speed-neutral — so it
# arms on every Blackwell card, INCLUDING the ones whose linear lane stays on
# the baseline. Binding these two to one switch was the pgw#862 shape and it
# cost sm_100 either 9.5 GB or 19% of its step time with no way to take both.
PACKED_MODULATION_SMS = (100, 103, 120, 121)

_EXT_DEFAULT = "/opt/cozy/native/libcozy_kernels.so"
_EXT_NAMESPACE = "cozy_kernels"

# Two independent decisions, each cached with its own recorded reason.
_LANES: dict[str, str] = {}
_REASONS: dict[str, str] = {}


def native_kernels_requested() -> Optional[bool]:
    """Env tri-state: True (opt-in), False (kill), None (unset => off while
    the rollout is env-gated)."""
    raw = os.environ.get(NATIVE_ENV, "").strip().lower()
    if raw in ("1", "true", "on", "yes"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False
    return None


def _gpu_sm() -> tuple[int, Optional[str]]:
    """``(sm, None)`` or ``(0, why-not)``."""
    try:
        import torch
    except ImportError:
        return 0, "torch is not installed"
    if not torch.cuda.is_available():
        return 0, "native svdq kernels require a CUDA GPU"
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor, None


def _probe_fused_linear(sm: int) -> Optional[str]:
    """Why the fused W4A4 linear cannot arm HERE, or None when it can."""
    if sm not in FUSED_LINEAR_SMS:
        if sm in BLACKWELL_SMS:
            return (f"sm_{sm} has the silicon but the fused linear is SLOWER "
                    f"than the baseline lane under torch.compile there "
                    f"(pgw#863); armed on "
                    f"sm_{'/'.join(str(s) for s in FUSED_LINEAR_SMS)}")
        return (f"fused svdq linear needs Blackwell block-scaled MMA "
                f"(sm_{'/'.join(str(s) for s in FUSED_LINEAR_SMS)}); this GPU "
                f"is sm_{sm}")
    from .svdq_fused import fused_self_check

    return fused_self_check()


def _probe_packed_modulation(sm: int) -> Optional[str]:
    """Why the packed W4A16 modulation cannot arm HERE, or None."""
    if sm not in PACKED_MODULATION_SMS:
        return (f"packed modulation needs Blackwell "
                f"(sm_{'/'.join(str(s) for s in PACKED_MODULATION_SMS)}); "
                f"this GPU is sm_{sm}")
    from .svdq_awq_packed import awq_packed_self_check

    return awq_packed_self_check()


def _decide(kind: str, on: str, off: str, probe) -> str:
    """One arming decision: env, then silicon, then numerics. Cached per
    process with the reason that produced it."""
    if kind in _LANES:
        return _LANES[kind]
    requested = native_kernels_requested()
    if requested is False:
        _LANES[kind], _REASONS[kind] = off, f"{NATIVE_ENV}=0 (kill-switch)"
    elif requested is None:
        _LANES[kind], _REASONS[kind] = off, (
            f"dormant — rollout is env-gated, set {NATIVE_ENV}=1 to opt in")
    else:
        sm, why = _gpu_sm()
        try:
            reason = why or probe(sm)
        except Exception as exc:  # noqa: BLE001 — any probe gap => degrade
            reason = f"probe raised: {exc}"
        if reason is not None:
            _LANES[kind], _REASONS[kind] = off, reason
            logger.warning("native kernels [%s]: requested but NOT armed — "
                           "%s; %s serves the same artifact",
                           kind, reason, off)
            return _LANES[kind]
        _LANES[kind], _REASONS[kind] = on, "probe + numerics self-check passed"
        logger.info("native kernels [%s]: %s armed", kind, on)
        return _LANES[kind]
    logger.info("native kernels [%s]: %s (%s)", kind, off, _REASONS[kind])
    return _LANES[kind]


def svdq_linear_lane() -> str:
    """``"fused"`` | ``"baseline"`` for the W4A4 linears in this process.
    Call at LOAD time — the first call compiles kernels and self-checks."""
    return _decide("linear", "fused", "baseline", _probe_fused_linear)


def svdq_modulation_lane() -> str:
    """``"packed"`` | ``"dense"`` for the W4A16 AdaLN modulation. Independent
    of the linear lane: it is a residency win, not a throughput one, so a card
    that wants the baseline linears still wants this."""
    return _decide("modulation", "packed", "dense", _probe_packed_modulation)


def svdq_linear_lane_reason() -> str:
    """The recorded reason for the linear-lane decision."""
    svdq_linear_lane()
    return _REASONS["linear"]


def svdq_modulation_lane_reason() -> str:
    """The recorded reason for the modulation-lane decision."""
    svdq_modulation_lane()
    return _REASONS["modulation"]


def reset_native_kernels_arming() -> None:
    """Forget both lane decisions (tests only)."""
    _LANES.clear()
    _REASONS.clear()


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
    "BLACKWELL_SMS",
    "FUSED_LINEAR_SMS",
    "PACKED_MODULATION_SMS",
    "NATIVE_ENV",
    "NATIVE_LIB_ENV",
    "extension_available",
    "extension_ops",
    "extension_path",
    "load_extension",
    "native_kernels_requested",
    "reset_native_kernels_arming",
    "svdq_linear_lane",
    "svdq_linear_lane_reason",
    "svdq_modulation_lane",
    "svdq_modulation_lane_reason",
]
