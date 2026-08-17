"""pgw#1315 deliverable 3 — an under-minimum machine SERVES, and CONFESSES.

`research/machine-compatibility-design.md`, verbatim: *"**Under-minimum is a
WARNING, never a refusal.** A machine below a declared minimum is the *normal*
input to a degraded run — that is what the minimum being advisory-at-execution
means. The typed warning names the term, the declared floor, this machine's
fact, and the posture taken."*

So there are exactly two ways to fail this file, and they are opposite:

1. **REFUSING.** A declared minimum that gates EXECUTION breaks always-runs.
   The minimum gates one thing and it is hub-side: a config-WRITE.
2. **SERVING SILENTLY.** The warning is LOAD-BEARING — it is the whole
   deliverable. An under-minimum run with no confession is the defect, which
   is why the emitter is disarmed to red-verify these: with
   `memory._confess_serve_degrade` a no-op, every assertion below that reads
   the sink fails and the always-runs assertions stay green.

The confession rides pgw#1312's ONE home (`serve_degrade`, its own phase
token) — extended, never twinned — and the same sentence rides
`ServePlan.warning` to `FnDegraded.reason`, so the operator's line and the
hub's row are one derivation.

Torch-free by construction: the seam is the requirement EVALUATOR and the
report, not a card. Bounded per the local-inference rule — logic only, no
compile, no mint.
"""

from __future__ import annotations

import asyncio
from typing import Any, List
from gen_worker import activity as activity_mod
from gen_worker import measured_posture as posture_mod
from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.api.slot import Slot
from gen_worker.executor import Executor
from gen_worker.hostfacts import HostFacts
from gen_worker.models import machine_fit
from gen_worker.models import memory as memory_mod
from gen_worker.models.hub_policy import TensorhubWorkerCapabilities
from gen_worker.models.serve_fit import RUN_NATIVE, plan_serve
from gen_worker.models.tensor_layout_contract import (
    CONTRACT_PLAIN_BF16,
    LayoutRequirements,
)
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec

import msgspec


class _Events:
    """The REAL activity sink the worker transport installs, drained after the
    plan — so these assertions read exactly the ActivityUpdates a hub would
    bank, not an in-process spy."""

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.loop = asyncio.new_event_loop()

    def __enter__(self) -> "_Events":
        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        activity_mod.bind_sink(_send, self.loop)
        return self

    def __exit__(self, *exc: object) -> None:
        self.loop.run_until_complete(asyncio.sleep(0.02))
        activity_mod.reset_for_tests()
        self.loop.close()

    def under_minimum(self) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == activity_mod.KIND_SERVE_DEGRADE
            and m.activity_update.phase == memory_mod.UNDER_MINIMUM_PHASE
        ]


_CAPS = TensorhubWorkerCapabilities(
    cuda_version="12.8", gpu_sm=86, torch_version="2.9.0+cu128",
    installed_libs=[],
)

#: An RTX A6000-shaped machine: sm86, 48 GiB, a roomy host.
_MACHINE = machine_fit.MachineFacts(
    sm=86, vram_gb=48.0, host_ram_gb=128.0, cuda="12.8", torch="2.9")


def _plan(requires: Any, *, facts: machine_fit.MachineFacts = _MACHINE) -> Any:
    return plan_serve(
        Resources(gpu=True, requires=requires), _CAPS, 40.0,
        facts=facts, scope="generate")


# ---------------------------------------------------------------------------
# 1. it runs — the whole point of question 2
# ---------------------------------------------------------------------------


def test_a_machine_far_under_every_declared_minimum_still_SERVES() -> None:
    """§1.35 amendment 2: *"'This model does not run on this card' is never an
    acceptable terminal state."* The minimum gates a config-WRITE hub-side and
    nothing at execution, so there is no arm here that can decline."""
    with _Events():
        plan = _plan("sm100+, vram80g, cuda13.0+, torch2.13+")

    assert plan.serveable is True
    assert plan.reason == ""
    assert plan.run_mode == RUN_NATIVE, (
        "an under-minimum machine is the NORMAL input to this planner; it "
        "must not be demoted a rung for missing a declared floor — which rung "
        "it lands on is measured at load time, not predicted from a "
        "declaration"
    )


# ---------------------------------------------------------------------------
# 2. and it confesses — the load-bearing half
# ---------------------------------------------------------------------------


def test_the_under_minimum_warning_names_TERM_FLOOR_and_FACT() -> None:
    """The three parts §1.36's amendment requires, on the typed carrier.

    A warning that says "degraded" and nothing else is the qualitative
    degradation this whole program exists to replace.
    """
    with _Events() as events:
        _plan("sm100+, vram80g")

    rows = events.under_minimum()
    assert len(rows) == 1, (
        "ONE confession per plan, through pgw#1312's one home — a second "
        "emitter is a second answer, and the one the hub banks is never the "
        "one the operator read"
    )
    detail = rows[0].detail
    for fragment in ("min_sm", "100", "sm=86", "min_vram_gb", "80", "vram_gb=48"):
        assert fragment in detail, (fragment, detail)
    assert rows[0].phase == posture_mod.REASON_BELOW_DECLARED_MINIMUM, (
        "the phase token IS the machine-readable cause, so the reason has one "
        "spelling on both carriers"
    )


def test_the_warning_rides_ServePlan_warning_to_FnDegraded() -> None:
    """`lifecycle._emit_degraded` sends `FnDegraded(reason=plan.warning)` for
    exactly the plans `plan.degraded` says yes to. An under-minimum plan that
    is not `degraded` never reaches the orchestrator at all."""
    with _Events() as events:
        plan = _plan("sm100+, vram80g")

    assert plan.degraded is True
    assert plan.warning
    assert plan.warning == events.under_minimum()[0].detail, (
        "one derivation, two carriers: the operator's line and the hub's row "
        "cannot be allowed to disagree"
    )
    assert len(plan.under_minimum) == 2


def test_the_warning_suggests_NO_CARD() -> None:
    """th#1867 deleted `FnDegraded.recommended_vram_gb` because the worker's
    suggestion was the author's own guess handed back, and §1.2 measured that
    guess wrong in BOTH directions on live releases. Only the hub knows the
    catalog (th#2075 clause 3)."""
    with _Events():
        plan = _plan("sm100+, vram80g")

    lowered = plan.warning.lower()
    assert "recommend" not in lowered
    assert " use a " not in lowered and "upgrade" not in lowered


def test_the_ran_token_stays_out_of_the_hubs_RunMode_vocabulary() -> None:
    """tensorhub matches `FnDegraded.ran` EXACTLY (`degradation_reschedule.go`
    switches on `offload`/`cpu`). A requirement shortfall is not a placement
    demotion, so it must not arrive spelled as one — an under-minimum warning
    that reported `ran="cpu"` would make the hub reschedule a pod that is
    running natively and fine."""
    with _Events():
        plan = _plan("sm100+, vram80g")

    assert plan.ran not in ("offload", "cpu")
    assert plan.ran == plan.wanted == "bf16"


# ---------------------------------------------------------------------------
# 3. what must NOT warn
# ---------------------------------------------------------------------------


def test_a_machine_that_MEETS_the_minimum_says_nothing() -> None:
    with _Events() as events:
        plan = _plan("sm80+, vram24g, cuda12.0+, torch2.5+")

    assert events.under_minimum() == []
    assert plan.warning == ""
    assert plan.degraded is False
    assert plan.under_minimum == ()


def test_recommended_gates_NOTHING_and_warns_about_nothing() -> None:
    """*"A recommended requirement gates nothing at all, ever."*
    `recommended_vram_gb` was deleted for cause (th#1720: the hub learned a
    monotone buy floor from it), so a machine below a RECOMMENDED level is not
    even a warning — it only ranks lanes."""
    requirement = LayoutRequirements(
        minimum="sm80+, vram24g", recommended="sm90+, vram80g")
    with _Events() as events:
        plan = _plan(requirement)

    assert events.under_minimum() == []
    assert plan.warning == ""
    assert plan.degraded is False


def test_an_undeclared_axis_is_NOT_EVALUATED() -> None:
    """NO DEFAULTS. A requirement that names only `min_sm` says nothing about
    VRAM, and a planner that read the absence as a floor of zero — or as a
    floor of anything — would be inventing the author's declaration."""
    with _Events() as events:
        plan = _plan("sm80+")

    assert events.under_minimum() == []
    assert plan.under_minimum == ()


def test_an_UNMEASURED_fact_is_unevaluated_rather_than_a_shortfall() -> None:
    """A cardless pod measures no `sm`. Reading that 0 as "meets no floor"
    would turn every declaration into a shortfall report about the wrong
    thing — the pod already knows it has no card, and the CPU rung already
    says so, loudly, through the same confession home."""
    cardless = machine_fit.MachineFacts(host_ram_gb=64.0)
    verdict = machine_fit.under_minimum(
        Resources(gpu=True, requires="sm100+, vram80g").requirement(), cardless)

    assert verdict.shortfalls == ()
    assert verdict.unevaluated == ("min_sm", "min_vram_gb")


# ---------------------------------------------------------------------------
# 4. through the PRODUCTION gate, not the planner in isolation
# ---------------------------------------------------------------------------


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def generate(self, ctx: Any, payload: _In) -> None:  # pragma: no cover
        return None


def _spec(requires: Any) -> EndpointSpec:
    return EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/sdxl")},
        slots={"pipeline": Slot(
            _Fake, selected_by="model",
            layouts={"*": (CONTRACT_PLAIN_BF16,)},
            layout_requirements={CONTRACT_PLAIN_BF16: "sm100+, vram80g"},
        )},
        resources=Resources(gpu=True, requires=requires),
    )


def test_the_EXECUTOR_GATE_confesses_and_keeps_the_function_available(
) -> None:
    """ASSERT EXECUTION, NOT REGISTRATION. `gate_functions` is the seam that
    decides what this pod advertises; it is where an under-minimum machine
    either serves loudly or quietly disappears from the fleet."""

    async def _send(_msg: pb.WorkerMessage) -> None:  # pragma: no cover
        return None

    with _Events() as events:
        ex = Executor([_spec("sm100+")], _send)
        ex.gate_functions(HostFacts(
            vram_total_bytes=48 * 1024 ** 3,
            vram_free_bytes=40 * 1024 ** 3,
            gpu_sm="86", cuda_version="12.8", torch_version="2.9.0",
        ))

    assert "generate" not in ex.unavailable, (
        "a declared minimum may never withdraw a function: it gates a "
        "config-WRITE hub-side and nothing at execution"
    )
    plan = ex.serve_plans["generate"]
    assert plan.serveable is True and plan.degraded is True
    rows = events.under_minimum()
    assert len(rows) == 1
    # BOTH scopes reach the worker and both are evaluated: the function scope
    # (`Resources(requires=)`) and the (slot, handle) scope of the picked lane.
    assert "generate" in rows[0].detail
    assert CONTRACT_PLAIN_BF16 in rows[0].detail
    assert len(plan.under_minimum) == 3, (
        "min_sm from the function scope, min_sm + min_vram_gb from the lane "
        "scope — a scope that is silently dropped is a floor that silently "
        "does not hold"
    )


def test_the_gate_leaves_a_MEETING_machine_silent() -> None:
    """The negative control for the arm above: same seam, same spec shape, a
    machine that clears both scopes."""

    async def _send(_msg: pb.WorkerMessage) -> None:  # pragma: no cover
        return None

    spec = EndpointSpec(
        name="generate", method=_Fake.generate, kind="inference",
        payload_type=_In, output_mode="single", cls=_Fake,
        models={"pipeline": Hub("acme/sdxl")},
        slots={"pipeline": Slot(
            _Fake, selected_by="model",
            layouts={"*": (CONTRACT_PLAIN_BF16,)},
            layout_requirements={CONTRACT_PLAIN_BF16: "sm80+, vram24g"},
        )},
        resources=Resources(gpu=True, requires="sm80+"),
    )
    with _Events() as events:
        ex = Executor([spec], _send)
        ex.gate_functions(HostFacts(
            vram_total_bytes=48 * 1024 ** 3,
            vram_free_bytes=40 * 1024 ** 3,
            gpu_sm="86", cuda_version="12.8", torch_version="2.9.0",
        ))

    assert events.under_minimum() == []
    assert ex.serve_plans["generate"].degraded is False


# ---------------------------------------------------------------------------
# 5. the vocabulary is shared, so the evaluator cannot drift
# ---------------------------------------------------------------------------


def test_every_requirement_term_has_a_measured_fact_of_the_same_name() -> None:
    """*"A requirement term and the machine fact it is compared against have
    the SAME NAME, so evaluation is a name lookup, never a bespoke comparator
    per term."* Growing `KNOWN_REQUIREMENT_TERMS` must not need an edit here —
    but a term with no fact behind it is a floor that silently does not hold,
    which is exactly why `kernels` is refused at declaration."""
    fields = set(machine_fit.MachineFacts.__struct_fields__)
    assert set(machine_fit.FACT_OF_TERM.values()) == fields


def test_the_comparator_is_the_declaration_sites_own() -> None:
    """One evaluator. `recommended >= minimum` at declaration and `fact >=
    floor` at runtime are the same question, and two implementations of it
    drift into disagreeing about what "meets" means."""
    from gen_worker.models import tensor_layout_contract as tlc

    assert machine_fit.term_meets is tlc.term_meets
    # And it is version-aware rather than lexicographic, which is the whole
    # reason a per-KIND comparator exists: "2.10" is above "2.9".
    assert tlc.term_meets("min_torch", "2.10", "2.9") is True
    assert tlc.term_meets("min_torch", "2.9", "2.10") is False
