"""Serve-time adaptive fit.

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
  offload       -> weights spill to CPU/disk, slower but valid (the PRIMARY
                  lever at the low end where weights exceed VRAM even quantized)
  cpu           -> no GPU at all: very slow, offered behind a loud warning
                  rather than refused

A function is UNSERVEABLE only when a genuine incompatibility bars it (compute
capability / required quant library / a stored flavor outside its SM window)
OR the author opted out of the CPU-touching rungs with
``Resources(strict_vram=True)`` (a binding that cannot tolerate CPU-resident
weights — compiled fixed-shape graphs — and would rather refuse than serve
slowly). It is never refused on hardware inadequacy alone: better to run
degraded than not run at all. The orchestrator hears about every degraded serve
(FnDegraded) and owns moving the workload to a bigger card.

Selection ACROSS stored flavors stays upstream: this planner marks each
function serveable/unserveable + how-it-runs, and the hub's routing ranking
picks the highest-quality fitting flavor. bf16 -> fp8 -> nvfp4 -> int4 falls out
of that ranking over the serveable set; this planner adds the RUNTIME rungs (fp8
storage / offload / cpu) for the one function it was given, plus an honest hint
when a stored flavor would have served natively.

Every degraded plan carries ``wanted`` (what the function declares) and
``ran`` (what actually runs) so the worker can report the degradation
STRUCTURALLY to the orchestrator (FnDegraded) as a placement signal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from .hub_policy import (
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

# Run modes and prices are the One Rung ladder's; re-exported here because
# this module is the hub-vocabulary projection.
from .rung import (
    RUN_CPU as RUN_CPU,
    RUN_FP8_STORAGE as RUN_FP8_STORAGE,
    RUN_NATIVE as RUN_NATIVE,
    RUN_OFFLOAD as RUN_OFFLOAD,
    price as price,
)

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
        """True when it runs, but not as planned: non-native placement,
        or a precision pick that could not be applied
        (``ran`` != ``wanted``, e.g. a dropped fp8 cast serving bf16)."""
        if not self.serveable:
            return False
        if self.run_mode != RUN_NATIVE:
            return True
        return bool(self.wanted) and bool(self.ran) and self.ran != self.wanted


def _wanted(binding: Any) -> str:
    """The precision the (post-resolution) binding plans to run: its cast
    directive (storage_dtype — a hub pick folds in as one), else base
    bf16. Counting the cast here makes a SUCCESSFUL cast visible (wanted=fp8
    ran=fp8) instead of masquerading as bf16.

    What the stored bytes are is the checkpoint's tensor-layout contract, not a
    token on the binding (§1.32(d))."""
    storage = str(getattr(binding, "storage_dtype", "") or "").strip().lower()
    return storage or "bf16"


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
    recommended = getattr(resources, "vram_gb_hint", None)
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
            est_latency_multiplier=price(RUN_CPU),
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

    # Runs, but degraded: runtime fp8 storage or the offload ladder. Offload
    # is the PRIMARY lever whenever the weights exceed VRAM — fit over speed.
    # Only offload is CPU-touching. Nothing quantizes at runtime.
    run_mode = RUN_FP8_STORAGE if verdict == FIT_EMERGENCY_FP8 else RUN_OFFLOAD
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
        est_latency_multiplier=price(run_mode),
        recommended_vram_gb=recommended,
        wanted=wanted,
        ran=run_mode,
    )


def replan(
    plan: Optional[ServePlan],
    *,
    run_mode: str = "",
    wanted: str = "",
    ran: str = "",
    detail: str,
) -> ServePlan:
    """The ONE runtime re-projection: demotion, load-rung engagement and
    cast-drop all fold into this seam.

    A runtime ladder transition re-prices the plan at ``run_mode`` and reports
    it structurally (FnDegraded) with the SAME vocabulary as plan-time.
    ``ran`` stays inside the hub's exact-match RunMode vocabulary — placement
    detail travels in ``detail``/``warning``, never decorates the token
    tensorhub switches on (degradation_reschedule.go). A cast that could not
    apply passes ``wanted``/``ran`` dtype tokens with no ``run_mode`` change.
    """
    base = plan if plan is not None else ServePlan(
        serveable=True, run_mode=RUN_NATIVE, fit="", wanted="bf16", ran="bf16",
    )
    mode = run_mode or base.run_mode
    return replace(
        base,
        serveable=True,
        run_mode=mode,
        wanted=(wanted or base.wanted),
        ran=(ran or (mode if run_mode else base.ran)),
        warning=detail,
        est_latency_multiplier=price(mode),
    )


def _honest_warning(run_mode: str, recommended_vram_gb: Optional[float], detail: str = "") -> str:
    ideal = (
        f" For full speed/quality use a ~{recommended_vram_gb:.0f} GB card."
        if recommended_vram_gb
        else ""
    )
    mult = price(run_mode)
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
    return detail
