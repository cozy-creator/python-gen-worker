"""Serve-time adaptive fit (th#683 P3).

The worker NEVER hard-refuses a function on the recommended-VRAM hint. On
whatever card it is actually on, it serves the function by the best available
means and is HONEST about the trade:

  bf16/native  -> highest quality, fastest (full VRAM residency)
  emergency nf4-> 4-bit, below-platform quality, still on-GPU
  offload      -> weights spill to CPU/disk, slower but valid (the PRIMARY
                  lever at the low end where weights exceed VRAM even quantized)
  cpu          -> no GPU at all: very slow, offered behind a loud warning
                  rather than refused

A function is UNSERVEABLE only when a genuine incompatibility bars it (compute
capability / required quant library) OR the only way to run it here is a
CPU-touching placement that this box forbids (GEN_WORKER_FORBID_CPU_OFFLOAD=1 —
those runs belong on the GPU lane / the right rented card). It is never refused
on hardware inadequacy alone.

Flavor selection across a card is realized by the pre-expanded per-flavor
functions (registry expands ``variants={}`` into separate routable functions):
this planner marks each flavor-function serveable/unserveable + how-it-runs, and
the hub's routing (th#597 ranking: bigger declared vram / svdq-fp4 first) picks
the highest-quality fitting flavor. bf16 -> fp8 -> nvfp4 -> int4 falls out of
that ranking over the serveable set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .hub_policy import (
    FIT_EMERGENCY,
    FIT_FITS,
    FIT_INCOMPATIBLE,
    FIT_SVDQ_FP4,
    FIT_SVDQ_INT4,
    TensorhubWorkerCapabilities,
    variant_fit,
)
from .memory import cpu_offload_forbidden

# Run modes, cheapest(fastest)-first. Mirrors the profiling.RunMode vocabulary
# on the hub side so the two speak the same language.
RUN_NATIVE = "native"
RUN_EMERGENCY = "emergency_quant"
RUN_OFFLOAD = "offload"
RUN_CPU = "cpu"

# Coarse latency multipliers vs a native GPU run, for honest-guidance. These are
# order-of-magnitude guides (the hub's measured fit-matrix latency is the
# authoritative source when available); they exist so the worker never stays
# silent about a slow trade.
_LATENCY_MULTIPLIER = {
    RUN_NATIVE: 1.0,
    RUN_EMERGENCY: 1.1,   # quality hit, not much slower
    RUN_OFFLOAD: 2.5,     # weights stream from CPU/disk
    RUN_CPU: 40.0,        # no GPU: dramatically slower
}


@dataclass(frozen=True)
class ServePlan:
    """How a single (already flavor-resolved) function will run on this card."""

    serveable: bool
    run_mode: str
    fit: str                      # the underlying variant_fit verdict
    reason: str = ""              # why unserveable, when !serveable
    warning: str = ""             # honest-guidance warning for a slow/degraded run
    est_latency_multiplier: float = 1.0
    recommended_vram_gb: Optional[float] = None  # the ideal card for this fn

    @property
    def degraded(self) -> bool:
        """True when it runs, but not natively (slower and/or lower quality)."""
        return self.serveable and self.run_mode != RUN_NATIVE


def plan_serve(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
    forbid_cpu_offload: Optional[bool] = None,
) -> ServePlan:
    """Decide how one function serves on the actual card. Never refuses on the
    recommended-VRAM hint alone.

    ``forbid_cpu_offload`` defaults to the live env predicate; pass explicitly
    in tests.
    """
    if forbid_cpu_offload is None:
        forbid_cpu_offload = cpu_offload_forbidden()

    recommended = getattr(resources, "vram_gb", None)
    needs_gpu = bool(getattr(resources, "gpu", False))

    verdict, detail = variant_fit(resources, caps, free_vram_gb, binding=binding)

    # No CUDA GPU present. variant_fit calls this incompatible; P3 turns it into
    # a CPU-only rung (offered behind the forbid guard + a loud warning).
    if verdict == FIT_INCOMPATIBLE and needs_gpu and caps.gpu_sm <= 0:
        if forbid_cpu_offload:
            return ServePlan(
                serveable=False,
                run_mode=RUN_CPU,
                fit=FIT_INCOMPATIBLE,
                reason=(
                    "no GPU and CPU inference is forbidden here "
                    "(GEN_WORKER_FORBID_CPU_OFFLOAD=1); run on a GPU host"
                ),
                recommended_vram_gb=recommended,
            )
        return ServePlan(
            serveable=True,
            run_mode=RUN_CPU,
            fit=FIT_INCOMPATIBLE,
            warning=_honest_warning(RUN_CPU, recommended),
            est_latency_multiplier=_LATENCY_MULTIPLIER[RUN_CPU],
            recommended_vram_gb=recommended,
        )

    # Genuine incompatibility (compute capability / missing quant library / svdq
    # SM window): no lever helps — this really cannot run here.
    if verdict == FIT_INCOMPATIBLE:
        return ServePlan(
            serveable=False,
            run_mode=RUN_NATIVE,
            fit=FIT_INCOMPATIBLE,
            reason=detail or "incompatible with this GPU",
            recommended_vram_gb=recommended,
        )

    # Fits natively (incl. the svdq flavor rungs, which are native on their
    # supported silicon).
    if verdict in (FIT_FITS, FIT_SVDQ_FP4, FIT_SVDQ_INT4):
        return ServePlan(
            serveable=True,
            run_mode=RUN_NATIVE,
            fit=verdict,
            recommended_vram_gb=recommended,
        )

    # Runs, but degraded: emergency 4-bit or the offload ladder. At the low end
    # offload is the PRIMARY lever (weights exceed VRAM even quantized) — fit
    # over speed. Both are CPU-touching only for the offload case.
    run_mode = RUN_EMERGENCY if verdict == FIT_EMERGENCY else RUN_OFFLOAD
    if run_mode == RUN_OFFLOAD and forbid_cpu_offload:
        return ServePlan(
            serveable=False,
            run_mode=RUN_OFFLOAD,
            fit=verdict,
            reason=(
                "only runs via CPU/disk offload here, which is forbidden "
                "(GEN_WORKER_FORBID_CPU_OFFLOAD=1); run on a larger card / GPU lane"
            ),
            recommended_vram_gb=recommended,
        )
    return ServePlan(
        serveable=True,
        run_mode=run_mode,
        fit=verdict,
        warning=_honest_warning(run_mode, recommended, detail),
        est_latency_multiplier=_LATENCY_MULTIPLIER[run_mode],
        recommended_vram_gb=recommended,
    )


def _honest_warning(run_mode: str, recommended_vram_gb: Optional[float], detail: str = "") -> str:
    ideal = (
        f" For full speed/quality use a ~{recommended_vram_gb:.0f} GB card."
        if recommended_vram_gb
        else ""
    )
    mult = _LATENCY_MULTIPLIER.get(run_mode, 1.0)
    if run_mode == RUN_CPU:
        return (
            "running on CPU (no GPU detected): expect dramatically slower "
            f"generation (~{mult:.0f}x)." + ideal
        )
    if run_mode == RUN_OFFLOAD:
        return (
            "weights do not fit VRAM; streaming from CPU/disk (offload): slower "
            f"(~{mult:.1f}x) but valid." + ideal
        )
    if run_mode == RUN_EMERGENCY:
        return (
            "does not fit at full precision; running 4-bit emergency "
            "quantization: below-platform quality." + ideal
        )
    return detail
