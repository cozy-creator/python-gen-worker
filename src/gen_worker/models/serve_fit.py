from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple

from .. import hostfacts
from . import machine_fit
from .hub_policy import (
    FIT_INCOMPATIBLE,
    TensorhubWorkerCapabilities,
    variant_fit,
)
from .memory import report_under_minimum

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
    fit: str
    reason: str = ""
    warning: str = ""
    est_latency_multiplier: float = 1.0
    wanted: str = ""
    ran: str = ""
    lane: str = ""
    under_minimum: Tuple[machine_fit.Shortfall, ...] = ()

    @property
    def degraded(self) -> bool:
        """True when it runs, but not as planned: non-native placement, a precision pick that could not be applied (``ran`` != ``wanted``, e.g."""
        if not self.serveable:
            return False
        if self.under_minimum:
            return True
        if self.run_mode != RUN_NATIVE:
            return True
        return bool(self.wanted) and bool(self.ran) and self.ran != self.wanted


def _wanted(binding: Any) -> str:
    storage = str(getattr(binding, "storage_dtype", "") or "").strip().lower()
    return storage or "bf16"


def plan_serve(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
    lanes: Sequence[machine_fit.LaneCandidate] = (),
    facts: Optional[machine_fit.MachineFacts] = None,
    scope: str = "",
) -> ServePlan:
    """Decide how one function serves on the actual card."""
    needs_gpu = bool(getattr(resources, "gpu", False))
    wanted = _wanted(binding)
    if facts is None:
        facts = machine_fit.measure_machine_facts(caps)
    choice = machine_fit.select_lane(tuple(lanes), facts)

    verdict, detail = variant_fit(resources, caps, free_vram_gb, binding=binding)

    if verdict == FIT_INCOMPATIBLE and needs_gpu and caps.gpu_sm <= 0:
        state = hostfacts.cuda_state()
        why = (
            f"this pod HAS a CUDA device and it will not answer "
            f"({state.probe_class}: {state.detail})"
            if state.unreadable else "no CUDA device detected on this pod"
        )
        return _confess_under_minimum(ServePlan(
            serveable=True,
            run_mode=RUN_CPU,
            fit=FIT_INCOMPATIBLE,
            warning=_honest_warning(RUN_CPU, why),
            est_latency_multiplier=price(RUN_CPU),
            wanted=wanted,
            ran=RUN_CPU,
            lane=choice.lane,
        ), resources=resources, choice=choice, facts=facts, scope=scope)

    if verdict == FIT_INCOMPATIBLE:
        return ServePlan(
            serveable=False,
            run_mode=RUN_NATIVE,
            fit=FIT_INCOMPATIBLE,
            reason=detail or "this build is missing a library this function needs",
            wanted=wanted,
        )

    return _confess_under_minimum(ServePlan(
        serveable=True,
        run_mode=RUN_NATIVE,
        fit=verdict,
        wanted=wanted,
        ran=wanted,
        lane=choice.lane,
    ), resources=resources, choice=choice, facts=facts, scope=scope)


def _confess_under_minimum(
    plan: ServePlan,
    *,
    resources: Any,
    choice: machine_fit.LaneChoice,
    facts: machine_fit.MachineFacts,
    scope: str,
) -> ServePlan:
    declared = (resources.requirement()
                if hasattr(resources, "requirement") else None)
    shortfalls = (
        machine_fit.under_minimum(declared, facts).shortfalls
        + choice.under_minimum
    )
    if not shortfalls:
        return plan
    warning = report_under_minimum(
        shortfalls, scope=scope or "this function", lane=choice.lane,
        posture=plan.run_mode)
    return replace(
        plan,
        under_minimum=tuple(shortfalls),
        warning=f"{plan.warning} {warning}".strip(),
    )


def replan(
    plan: Optional[ServePlan],
    *,
    run_mode: str = "",
    wanted: str = "",
    ran: str = "",
    detail: str,
) -> ServePlan:
    """The ONE runtime re-projection: demotion, load-rung engagement and cast-drop all fold into this seam."""
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
    return f"{head} {detail}".strip() if detail else head
