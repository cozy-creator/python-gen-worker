"""Serve-time adaptive fit (th#683 P3).

The worker NEVER hard-refuses a function on the recommended-VRAM hint. On
whatever card it is actually on, it serves the function by the best available
means and is HONEST about the trade. The full ladder, best-first:

  stored, native-> the binding's own precision at full VRAM residency:
                  bf16/fp16, #fp8 (Ada/Hopper+), #nvfp4 (Blackwell),
                  #svdq-* (their SM windows) — each HW-window-gated in
                  hub_policy.variant_fit; wrong silicon is a refusal
  fp8 storage   -> runtime fp8-E4M3 weight storage + bf16 compute
                  (loading.apply_fp8_storage; no fp8 silicon required):
                  near-native quality, weights ~halve
  emergency nf4 -> runtime 4-bit, below-platform quality, still on-GPU
  offload       -> weights spill to CPU/disk, slower but valid (the PRIMARY
                  lever at the low end where weights exceed VRAM even quantized)
  cpu           -> no GPU at all: very slow, offered behind a loud warning
                  rather than refused

A function is UNSERVEABLE only when a genuine incompatibility bars it (compute
capability / required quant library / a stored flavor outside its SM window)
OR the author opted out of the CPU-touching rungs with
``Resources(strict_vram=True)`` (a binding that cannot tolerate CPU-resident
weights — compiled fixed-shape graphs, TRT engines — and would rather refuse
than serve slowly). It is never refused on hardware inadequacy alone: gen
workers don't offload to CPU because we want them to, they do it out of
necessity — better to run degraded than not run at all (Paul's ruling,
2026-07-10). The orchestrator hears about every degraded serve (FnDegraded)
and owns moving the workload to a bigger card.

Selection ACROSS stored flavors stays upstream: the registry pre-expands
``variants={}`` into separate routable per-flavor functions, this planner
marks each one serveable/unserveable + how-it-runs, and the hub's routing
(th#597 ranking) — or ``hub_policy.select_variant`` for cozy-local
``--variant auto`` — picks the highest-quality fitting flavor. bf16 -> fp8 ->
nvfp4 -> int4 falls out of that ranking over the serveable set; this planner
adds the RUNTIME rungs (fp8 storage / nf4 / offload / cpu) for the one
function it was given, plus an honest hint when a stored flavor would have
served natively.

Every degraded plan carries ``wanted`` (what the function declares) and
``ran`` (what actually runs) so the worker can report the degradation
STRUCTURALLY to the orchestrator (FnDegraded) as a placement signal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from .hub_policy import (
    FIT_EMERGENCY,
    FIT_EMERGENCY_FP8,
    FIT_FITS,
    FIT_FP8,
    FIT_INCOMPATIBLE,
    FIT_NVFP4,
    FIT_SVDQ_FP4,
    FIT_SVDQ_INT4,
    TensorhubWorkerCapabilities,
    variant_fit,
)

# Run modes, cheapest(fastest)-first. Mirrors the profiling.RunMode vocabulary
# on the hub side so the two speak the same language.
RUN_NATIVE = "native"
RUN_FP8_STORAGE = "fp8_storage"
RUN_EMERGENCY = "emergency_quant"
RUN_OFFLOAD = "offload"
RUN_CPU = "cpu"

# Coarse latency multipliers vs a native GPU run, for honest-guidance. These are
# order-of-magnitude guides (the hub's measured fit-matrix latency is the
# authoritative source when available); they exist so the worker never stays
# silent about a slow trade.
_LATENCY_MULTIPLIER = {
    RUN_NATIVE: 1.0,
    RUN_FP8_STORAGE: 1.05,  # per-layer upcast overhead; near-native quality
    RUN_EMERGENCY: 1.1,     # quality hit, not much slower
    RUN_OFFLOAD: 2.5,       # weights stream from CPU/disk
    RUN_CPU: 40.0,          # no GPU: dramatically slower
}

# The FIT verdicts that run natively: full residency at the binding's own
# stored precision on supported silicon.
_NATIVE_FITS = (FIT_FITS, FIT_FP8, FIT_NVFP4, FIT_SVDQ_FP4, FIT_SVDQ_INT4)


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
    wanted: str = ""              # what the function declares (flavor, or "bf16")
    ran: str = ""                 # what actually runs (flavor when native, else run_mode)

    @property
    def degraded(self) -> bool:
        """True when it runs, but not natively (slower and/or lower quality)."""
        return self.serveable and self.run_mode != RUN_NATIVE


def _wanted(binding: Any) -> str:
    flavor = str(getattr(binding, "flavor", "") or "").strip().lower()
    return flavor or "bf16"


def plan_serve(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
) -> ServePlan:
    """Decide how one function serves on the actual card. Never refuses on the
    recommended-VRAM hint alone; ``Resources(strict_vram=True)`` is the sole
    author opt-out of the CPU-touching rungs (offload / cpu).
    """
    recommended = getattr(resources, "vram_gb", None)
    needs_gpu = bool(getattr(resources, "gpu", False))
    strict_vram = bool(getattr(resources, "strict_vram", False))
    wanted = _wanted(binding)

    verdict, detail = variant_fit(resources, caps, free_vram_gb, binding=binding)

    # No CUDA GPU present. variant_fit calls this incompatible; P3 turns it
    # into a CPU-only rung (behind a loud warning) unless the author opted
    # out of CPU-resident weights entirely.
    if verdict == FIT_INCOMPATIBLE and needs_gpu and caps.gpu_sm <= 0:
        if strict_vram:
            return ServePlan(
                serveable=False,
                run_mode=RUN_CPU,
                fit=FIT_INCOMPATIBLE,
                reason=(
                    "no GPU here and the author requires full VRAM residency "
                    "(strict_vram=True); run on a GPU host"
                ),
                recommended_vram_gb=recommended,
                wanted=wanted,
            )
        return ServePlan(
            serveable=True,
            run_mode=RUN_CPU,
            fit=FIT_INCOMPATIBLE,
            warning=_honest_warning(RUN_CPU, recommended),
            est_latency_multiplier=_LATENCY_MULTIPLIER[RUN_CPU],
            recommended_vram_gb=recommended,
            wanted=wanted,
            ran=RUN_CPU,
        )

    # Genuine incompatibility (compute capability / missing quant library /
    # a stored flavor outside its SM window): no lever helps — this really
    # cannot run here.
    if verdict == FIT_INCOMPATIBLE:
        return ServePlan(
            serveable=False,
            run_mode=RUN_NATIVE,
            fit=FIT_INCOMPATIBLE,
            reason=detail or "incompatible with this GPU",
            recommended_vram_gb=recommended,
            wanted=wanted,
        )

    # Fits natively at its own stored precision (incl. the fp8/nvfp4/svdq
    # flavor rungs, which are native on their supported silicon).
    if verdict in _NATIVE_FITS:
        return ServePlan(
            serveable=True,
            run_mode=RUN_NATIVE,
            fit=verdict,
            recommended_vram_gb=recommended,
            wanted=wanted,
            ran=wanted,
        )

    # Runs, but degraded: runtime fp8 storage, emergency 4-bit, or the offload
    # ladder. At the low end offload is the PRIMARY lever (weights exceed VRAM
    # even quantized) — fit over speed. Only offload is CPU-touching.
    if verdict == FIT_EMERGENCY_FP8:
        run_mode = RUN_FP8_STORAGE
    elif verdict == FIT_EMERGENCY:
        run_mode = RUN_EMERGENCY
    else:
        run_mode = RUN_OFFLOAD
    if run_mode == RUN_OFFLOAD and strict_vram:
        return ServePlan(
            serveable=False,
            run_mode=RUN_OFFLOAD,
            fit=verdict,
            reason=(
                "only runs via CPU/disk offload here and the author requires "
                "full VRAM residency (strict_vram=True); run on a card with "
                + (f"~{recommended:.0f} GB" if recommended else "more VRAM")
            ),
            recommended_vram_gb=recommended,
            wanted=wanted,
        )
    return ServePlan(
        serveable=True,
        run_mode=run_mode,
        fit=verdict,
        warning=_honest_warning(run_mode, recommended, detail),
        est_latency_multiplier=_LATENCY_MULTIPLIER[run_mode],
        recommended_vram_gb=recommended,
        wanted=wanted,
        ran=run_mode,
    )


def demoted(
    plan: Optional[ServePlan],
    *,
    detail: str,
    placement_mode: str = "",
) -> ServePlan:
    """The reactive ladder transition (gw#463): a runtime CUDA OOM demoted this
    function to the offload rung. Produces the updated ServePlan so plan-time
    and reactive degradation share one vocabulary, warning, and FnDegraded
    shape. ``placement_mode`` (model_offload/group_offload/sequential) rides
    ``ran`` as ``offload:<mode>`` so each deeper sub-rung re-reports."""
    base = plan if plan is not None else ServePlan(
        serveable=True, run_mode=RUN_NATIVE, fit="", wanted="bf16", ran="bf16",
    )
    ran = f"{RUN_OFFLOAD}:{placement_mode}" if placement_mode else RUN_OFFLOAD
    return replace(
        base,
        serveable=True,
        run_mode=RUN_OFFLOAD,
        warning=detail,
        est_latency_multiplier=_LATENCY_MULTIPLIER[RUN_OFFLOAD],
        ran=ran,
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
    if run_mode == RUN_FP8_STORAGE:
        return (
            "does not fit at full precision; running fp8-E4M3 weight storage: "
            "near-native quality. A stored #fp8 flavor of this model would "
            "serve natively here." + ideal
        )
    if run_mode == RUN_EMERGENCY:
        return (
            "does not fit at full precision; running 4-bit emergency "
            "quantization: below-platform quality. A stored 4-bit flavor "
            "(#nvfp4 / #svdq-int4) would serve natively here." + ideal
        )
    return detail
