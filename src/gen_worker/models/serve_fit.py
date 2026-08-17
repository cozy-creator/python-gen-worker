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
defect naming what in OUR code failed"). pgw#1315 closed the other half: the
REACTIVE walk used to stop one rung short of CPU, so a pod that OOM'd its way
down refused where plan time served. Both ends now reach the same bottom rung
and it EXECUTES (`memory.apply_low_vram_config` mode `cpu`).

WHAT IT DOES DECIDE, AS OF pgw#1315: which LANE. Among the lanes a release
binds, the machine's measured facts pick the one whose declared requirements
they best satisfy (`models/machine_fit.select_lane`), and the declared
MINIMUMS this machine misses become a loud typed warning through pgw#1312's
one confession home. Both are question 2 in
`research/machine-compatibility-design.md`, and neither can refuse: a minimum
gates a config-WRITE hub-side and NOTHING at execution, so an under-minimum
machine is the normal input to a degraded run rather than an error.

Every degraded plan carries ``wanted`` (what the function declares) and ``ran``
(what actually runs) so the worker can report the degradation STRUCTURALLY to
the orchestrator (FnDegraded) as OPERATOR EVIDENCE. It carries no card size:
th#1867 deleted `FnDegraded.recommended_vram_gb` because its only source was
the author's own guess, which §1.2 measured wrong in both directions.
"""

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
    # pgw#1315 deliverable 4: the lane these machine facts picked, among the
    # ones the release binds. "" when the function binds no lane that declares
    # a requirement — there is nothing to rank by, and ranking undeclared
    # lanes would be the guess this program deletes.
    lane: str = ""
    # pgw#1315 deliverable 3: the declared MINIMUMS this machine does not meet.
    # Non-empty means it runs anyway, degraded, having said so.
    under_minimum: Tuple[machine_fit.Shortfall, ...] = ()

    @property
    def degraded(self) -> bool:
        """True when it runs, but not as planned: non-native placement,
        a precision pick that could not be applied (``ran`` != ``wanted``,
        e.g. a dropped fp8 cast serving bf16), or a machine under a declared
        minimum.

        The last one is why an under-minimum run reaches the orchestrator at
        all: `_emit_degraded` sends `FnDegraded` for exactly the plans this
        says yes to, carrying ``warning`` as the reason. It stays a WARNING —
        `ran` keeps its own vocabulary, so no hub arm that switches on a
        RunMode can read a requirement shortfall as a placement demotion.
        """
        if not self.serveable:
            return False
        if self.under_minimum:
            return True
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
    lanes: Sequence[machine_fit.LaneCandidate] = (),
    facts: Optional[machine_fit.MachineFacts] = None,
    scope: str = "",
) -> ServePlan:
    """Decide how one function serves on the actual card.

    Never refuses on card SIZE — there is no size input left to refuse on
    (th#1867). ``free_vram_gb`` is accepted and unused for the same reason
    :func:`variant_fit` keeps it: it is the number the deleted comparison was
    made against, and a caller that stops passing it would hide the fact that
    reintroducing the comparison is a one-line edit.

    pgw#1315 adds question 2's other half. ``lanes`` are the lanes this
    release binds for the function (``machine_fit.lane_candidates`` builds
    them from the slot's accepted handles and their declared requirements);
    the machine's own FACTS pick among them, and the declared MINIMUMS this
    machine misses become a loud typed warning — never a refusal, because
    "it always runs" is the whole answer to question 2.

    ``facts`` is measured once by the caller (the gate loop plans every
    function against ONE machine); ``None`` measures here.
    """
    needs_gpu = bool(getattr(resources, "gpu", False))
    wanted = _wanted(binding)
    if facts is None:
        facts = machine_fit.measure_machine_facts(caps)
    choice = machine_fit.select_lane(tuple(lanes), facts)

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
    # pgw#1315: the reactive walk reaches this same rung now, so plan time and
    # a descent that exhausts every offload rung land on ONE answer.
    #
    # pgw#896: "no CUDA device detected" was said for BOTH a genuinely cardless
    # pod and a pod whose card would not answer, because the predicate behind
    # it (`torch.cuda.is_available()`) cannot tell them apart. The operator
    # reading the warning then goes looking for a provisioning mistake on a box
    # that has an H100 in it. The strong probe's own class discriminates.
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
    """Serve, and SAY that a declared floor is not being met.

    TWO scopes reach a worker and both are evaluated here: the FUNCTION scope
    (`Resources(requires=)`, for code with no contract to hang a requirement
    on) and the (slot, handle) scope of the lane the facts just picked. The
    hub owns the third (the contract's intrinsic floor) and it can only be
    looser, so nothing here re-derives it.

    The plan is returned SERVEABLE either way — this function has no arm that
    can refuse. The warning rides `ServePlan.warning` to `FnDegraded.reason`,
    and the same sentence rides the typed `serve_degrade` event out of
    pgw#1312's one confession home, so the operator's line and the hub's row
    are one derivation.
    """
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
