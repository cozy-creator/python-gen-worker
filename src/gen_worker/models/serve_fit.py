"""Serve-time adaptive fit (th#683 P3), after th#1867 took the estimate out.

The worker NEVER refuses a function because a card is small. On whatever card
it is actually on it serves by the best available means and is HONEST about the
trade — and WHICH means is a decision this module no longer makes. th#1867
(§1.35) deleted `Resources(vram_gb_hint / min_vram_gb / strict_vram)`, which
were the only inputs to the plan-time ladder here, so the rung is now chosen at
LOAD time by `models/memory.select_auto_mode` against the pipeline's real size
and the card's real free VRAM, and reported as it happens through
:func:`replan` -> FnDegraded. The prediction is gone; the ladder is not.

WHAT STILL REFUSES. Exactly ONE verdict, and it names our IMAGE: a quant
library this build does not carry. The executor's own gate reports that as
`missing_cuda_library` before this planner is reached, so in practice this
module refuses nothing at all — every card serves, and the only question left
is which rung.

NO CUDA IS NOT A REFUSAL, AND THAT WAS TESTED RATHER THAN ASSUMED. th#1867
drafted a `cuda_unavailable` refusal here on the reasoning that pgw#1212 records
the CPU serve path as never executed. The repo's own boot tests refuted it:
every harness endpoint declares `Resources(gpu=True)` and serves on CPU-only CI,
and `test_boot_span_ladder_pgw797` boots one and runs a warmup forward through
it. Withdrawing a function that demonstrably works is the opposite of §1.35
amendment 2's bar ("the worker either serves, however slowly, or emits a typed
defect naming what in OUR code failed"). pgw#1212's protection stays where it
already was and where it is true: the REACTIVE walk in `models/rung` stops one
rung short of CPU (`FLOOR_CPU_RUNG_UNEXECUTABLE`), so a pod that OOMs its way
down refuses and names our code. Plan time is not that situation.

Every degraded plan carries ``wanted`` (what the function declares) and ``ran``
(what actually runs) so the worker can report the degradation STRUCTURALLY to
the orchestrator (FnDegraded) as OPERATOR EVIDENCE. It carries no card size:
th#1867 deleted `FnDegraded.recommended_vram_gb` because its only source was
the author's own guess, which §1.2 measured wrong in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from .hub_policy import (
    FIT_INCOMPATIBLE,
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


@dataclass(frozen=True)
class ServePlan:
    """How a single (already flavor-resolved) function will run on this card."""

    serveable: bool
    run_mode: str
    fit: str                      # the underlying variant_fit verdict
    reason: str = ""              # why unserveable, when !serveable
    warning: str = ""             # honest-guidance warning for a slow/degraded run
    est_latency_multiplier: float = 1.0
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
    """Decide how one function serves on the actual card.

    Never refuses on card SIZE — there is no size input left to refuse on
    (th#1867). ``free_vram_gb`` is accepted and unused for the same reason
    :func:`variant_fit` keeps it: it is the number the deleted comparison was
    made against, and a caller that stops passing it would hide the fact that
    reintroducing the comparison is a one-line edit.
    """
    needs_gpu = bool(getattr(resources, "gpu", False))
    wanted = _wanted(binding)

    verdict, detail = variant_fit(resources, caps, free_vram_gb, binding=binding)

    # No CUDA device: the CPU rung, SERVED, behind a loud warning.
    #
    # This branch was very nearly a refusal (th#1867 drafted one citing
    # pgw#1212's unproven CPU path). The repo's own boot tests refuted it:
    # every harness endpoint declares `Resources(gpu=True)` and serves on
    # CPU-only CI — `test_boot_span_ladder_pgw797` boots one and runs a warmup
    # forward. Refusing here would have withdrawn a function that demonstrably
    # WORKS, which is the opposite of §1.35 amendment 2's robustness bar.
    #
    # pgw#1212's protection is real and stays exactly where it already was: the
    # REACTIVE walk in `models/rung` stops one rung short of CPU
    # (FLOOR_CPU_RUNG_UNEXECUTABLE), so a pod that OOMs its way down still
    # refuses, naming our code. Plan time is not that situation — nothing has
    # failed here.
    if verdict == FIT_INCOMPATIBLE and needs_gpu and caps.gpu_sm <= 0:
        return ServePlan(
            serveable=True,
            run_mode=RUN_CPU,
            fit=FIT_INCOMPATIBLE,
            warning=_honest_warning(RUN_CPU, "no CUDA device detected on this pod"),
            est_latency_multiplier=price(RUN_CPU),
            wanted=wanted,
            ran=RUN_CPU,
        )

    # A quant library this build does not carry. Names our IMAGE; the
    # executor's own gate reports it as `missing_cuda_library`.
    if verdict == FIT_INCOMPATIBLE:
        return ServePlan(
            serveable=False,
            run_mode=RUN_NATIVE,
            fit=FIT_INCOMPATIBLE,
            reason=detail or "this build is missing a library this function needs",
            wanted=wanted,
        )

    # Everything else SERVES. Which rung it lands on is measured at load time
    # (models/memory.select_auto_mode) and reported through `replan`, never
    # predicted here.
    return ServePlan(
        serveable=True,
        run_mode=RUN_NATIVE,
        fit=verdict,
        wanted=wanted,
        ran=wanted,
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
    detail travels in ``detail``/``warning``, never decorates the token the
    hub switches on. A cast that could not apply passes ``wanted``/``ran``
    dtype tokens with no ``run_mode`` change.

    th#1867: this is now the ONLY producer of the honest degradation warning.
    It used to be written at plan time from the author's declared card size;
    the transition it describes here is one that ACTUALLY HAPPENED, which is
    the loud-and-quantified evidence §1.36's amendment asks for and the
    declared number never was.
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
        warning=_honest_warning(mode, detail),
        est_latency_multiplier=price(mode),
    )


def _honest_warning(run_mode: str, detail: str = "") -> str:
    """The observed transition, priced. ``detail`` is the caller's own words
    and is the whole answer for a mode with no standard story."""
    # th#1867: no " For full speed/quality use a ~N GB card." tail. That number
    # was the author's declaration, and §1.2 measured it wrong in both
    # directions on live releases — advice derived from it was confident and
    # unfounded. What the warning says now is only what was observed.
    mult = price(run_mode)
    if run_mode == RUN_CPU:
        head = ("running on CPU (no GPU detected): expect dramatically slower "
                f"generation (~{mult:.0f}x).")
    elif run_mode == RUN_OFFLOAD:
        head = ("weights do not fit VRAM; streaming from CPU/disk (offload): "
                f"slower (~{mult:.1f}x) but valid.")
    elif run_mode == RUN_FP8_STORAGE:
        head = ("does not fit at full precision; running fp8-E4M3 weight "
                "storage: near-native quality. A stored #fp8 flavor of this "
                "model would serve natively here.")
    else:
        return detail
    # The caller's own words carry the MEASURED numbers behind the transition
    # (the size that did not fit, the VRAM that was free). Dropping them to
    # print a canned sentence is how a quantified degradation becomes a
    # qualitative one — the failure mode this whole issue is about.
    return f"{head} {detail}".strip() if detail else head
