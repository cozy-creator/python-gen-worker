"""Compiled admission is PROBED and per-target — never a declared refusal.

pgw#1587. Paul, 2026-08-20, on the ``vram12g`` lane declaration that made a
7.3 GiB card refuse to mint or arm SDXL at all:

    *"Every card should be able to run any job. There is a minimum floor,
    below which running compiled does not work; we keep eager. The advantage
    of eager is that it's more flexible. For SDXL in particular, we need to
    offload the text encoders to free up room for the Unet, and then it works,
    during inference. This doesn't conflict with compilation however because
    [we] are only running the compiled UNet. Remove this '12gb floor'."*

    *"having some memory requirement per tensor-layout-contract makes sense.
    But yeah, this limit is too high. And we should be able to serve
    no-compiled below this limit."*

So the declaration stays and the REFUSAL is gone. What decides compiled
admission is the compiled graph's own working set — its weights held by
reference, its activation peak — measured on the card, with every component
outside the graph offloadable to make room for it. Three causes send a load to
eager and all three are LOUD and NAMED: the operator ordered it, the probe
refused, or the target's own weights move. Nothing refuses the JOB.

The two directions are both armed here, because the failure this replaces was
symmetric: a card that fits must ARM (the old gate said no on a declaration),
and a card that cannot must fall to EAGER by name (never an OOM, never a
crash).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, cast

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers import DiffusionPipeline, ModelMixin  # noqa: E402
from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402

from gen_worker import serve_posture  # noqa: E402
from gen_worker.models import provision  # noqa: E402
from gen_worker.models.partial_resident import (  # noqa: E402
    apply_component_residency,
    parks_module,
    plan_for_pipeline,
)
from gen_worker.models.rung import moves_every_component, touches_host_ram  # noqa: E402
from gen_worker.serving.context import DeployBinding, LoadContext  # noqa: E402

_MIB = 1 << 20


class _Block(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 8):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.lin(x)


class _Pipeline(DiffusionPipeline):
    """SDXL's shape in miniature: two things before the denoiser, one after."""

    model_cpu_offload_seq = "text_encoder->unet->vae"
    text_encoder: Any
    unet: Any
    vae: Any

    def __init__(self, text_encoder: Any, unet: Any, vae: Any) -> None:
        super().__init__()
        cast(Any, self).register_modules(
            text_encoder=text_encoder, unet=unet, vae=vae
        )


def _armed_pipeline() -> Any:
    """A pipeline with the rung ARMED: encoder parked, denoiser resident.

    Same shape the rung takes on the campaign card — the components before the
    denoiser go to pinned host RAM so the denoiser can stay on the card.
    """
    pipe = _Pipeline(_Block(8), _Block(16), _Block(4))
    plan = plan_for_pipeline(
        pipe,
        budget_bytes=1200,
        free_bytes=64 * _MIB,
        transient_reserve_bytes=0,
        forced_resident=("vae",),
        sizer=lambda m: sum(p.numel() * p.element_size() for p in m.parameters()),
    )
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("text_encoder",) and "unet" in plan.resident
    assert apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t")
    )
    return pipe


def _ctx(sink_calls: List[Any]) -> "LoadContext[Any]":
    """A context whose compile sink RECORDS and answers ``"ARMED"``.

    The sink is torchcg's adoption seam; what it returns is irrelevant here and
    what it was CALLED WITH is the whole fact under test.
    """

    def sink(target: Any) -> Any:
        sink_calls.append(target)
        return "ARMED"

    return LoadContext(
        binding=DeployBinding(checkpoint_ref="r", checkpoint_dir=Path(".")),
        compile_sink=sink,
    )


# --------------------------------------------------------------------------
# The rung vocabulary: two different facts, no longer one
# --------------------------------------------------------------------------


def test_the_rung_that_parks_named_components_is_not_the_rung_that_moves_all() -> None:
    """`touches_host_ram` is ACCOUNTING; `moves_every_component` is STABILITY.

    Reading the first as the second is the whole defect: `partial_resident`
    charges host RAM (it really does hold weights there) and leaves every
    component it did not name device-resident for the life of the load. One
    boolean answering both questions is what made "small card" mean "no
    compiled graph, ever".
    """
    assert touches_host_ram("partial_resident"), "it does hold weights on the host"
    assert not moves_every_component("partial_resident"), (
        "it moves only what it named — that is the point of the rung"
    )
    for every in ("model_offload", "group_offload", "sequential", "cpu"):
        assert moves_every_component(every), every
    assert not moves_every_component(""), "no rung engaged moves nothing"
    assert not moves_every_component("native")


# --------------------------------------------------------------------------
# Direction 1 — A CARD THAT FITS MUST ADMIT COMPILED
# --------------------------------------------------------------------------


def test_the_resident_denoiser_compiles_under_the_offload_rung() -> None:
    """PAUL'S CASE, as a test. The encoders come off the card, the compiled
    UNet stays on it, and the two do not conflict.

    RED ARM: before pgw#1587 `ctx.compile` refused every target whenever any
    host-RAM-touching rung was engaged, so this assertion failed and SDXL on a
    7.3 GiB card served eager no matter what was minted for it.
    """
    pipe = _armed_pipeline()
    sink_calls: List[Any] = []
    ctx = _ctx(sink_calls)
    ctx._engaged_rung = "partial_resident"

    assert ctx.compile(pipe.unet) == "ARMED", (
        "the denoiser is in the plan's RESIDENT set — its device pointers are "
        "stable for the life of the load, which is exactly the precondition a "
        "compiled graph's by-reference constants need"
    )
    assert sink_calls == [pipe.unet]


def test_the_parked_component_serves_eager_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other half of per-target: what the rung DID park still cannot
    compile, and the refusal names the rung and the reason."""
    pipe = _armed_pipeline()
    sink_calls: List[Any] = []
    ctx = _ctx(sink_calls)
    ctx._engaged_rung = "partial_resident"

    assert parks_module(pipe.text_encoder), "the fixture must have parked it"
    with caplog.at_level(logging.WARNING):
        assert ctx.compile(pipe.text_encoder) is pipe.text_encoder
    assert sink_calls == [], "a parked target must never reach the sink"
    assert any("EAGER" in r.message and "pgw#1587" in r.message
               for r in caplog.records), (
        "falling to eager is a real degradation and must be said out loud"
    )


def test_a_rung_that_moves_everything_still_refuses_every_target(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """pgw#1486 IS NOT WEAKENED. Under accelerate's hooks every component's
    weights are onloaded per forward and freed after, so a bound constant is a
    dangling pointer — an uncatchable SIGSEGV, not an OOM anyone retries."""
    module = torch.nn.Linear(2, 2)
    sink_calls: List[Any] = []
    ctx = _ctx(sink_calls)

    for rung in ("model_offload", "group_offload", "sequential", "cpu"):
        ctx._engaged_rung = rung
        with caplog.at_level(logging.WARNING):
            assert ctx.compile(module) is module, rung
    assert sink_calls == []


def test_no_rung_engaged_is_unchanged() -> None:
    module = torch.nn.Linear(2, 2)
    sink_calls: List[Any] = []
    ctx = _ctx(sink_calls)
    ctx._engaged_rung = ""
    assert ctx.compile(module) == "ARMED"
    assert sink_calls == [module]


# --------------------------------------------------------------------------
# Direction 2 — A CARD THAT CANNOT HOLD IT FALLS TO EAGER, BY NAME
# --------------------------------------------------------------------------


def test_the_probe_refuses_a_plan_the_arithmetic_admitted() -> None:
    """The admission is asked of the CARD, not of a constant (pgw#1577).

    Arithmetic over component sizes and a free-VRAM read cannot see allocator
    fragmentation or a co-tenant's share; on the campaign card a plan those
    numbers admitted then died 5 MiB short. The probe does the worst onload
    once and reads what is left, and a refusal leaves NOTHING armed — the
    caller falls to the next rung rather than inheriting half an arrangement.
    """
    pipe = _Pipeline(_Block(8), _Block(16), _Block(4))
    plan = plan_for_pipeline(
        pipe,
        budget_bytes=1200,
        free_bytes=64 * _MIB,
        transient_reserve_bytes=0,
        forced_resident=("vae",),
        sizer=lambda m: sum(p.numel() * p.element_size() for p in m.parameters()),
    )
    assert plan.fits

    armed = apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t"),
        # The card answers with almost nothing free once the worst evicted
        # component is on it. Same shape as the measured 5 MiB shortfall.
        free_bytes_now=lambda: 0,
    )
    assert not armed, "the card disagreed with the plan, and the card is right"
    assert not parks_module(pipe.text_encoder), (
        "a refused probe must leave no parked component behind"
    )


def test_an_operator_eager_only_order_is_observed_by_the_dispatch_seam() -> None:
    """The third entry into loud eager, RE-AIMED at the seam that runs.

    pgw#1587 filed this correctly and pointed it one layer off. Its reader was
    `compile_cache.arming_block` (the v1 precondition authority), so this row
    moved it to `provision.arm_aot` and called that "the v2 arm" — but
    `arm_aot` imports `aot_serve`, i.e. it is the SAME v1 tier, which pgw#1573
    measured as having no production caller. Both readers were in dead code, so
    an operator could issue the order, get an ack, and watch the pod keep
    serving from its compiled graphs (filed as pgw#1589).

    The v2 arm is `AdoptSession` installing a dispatcher; the order arrives
    over a live control channel long AFTER that and is RELEASABLE, so the
    honest altitude is the dispatch itself. `serving.adapter_guard` reads it
    per call — which makes both directions work with no re-arm and no de-arm,
    exactly the reversibility `apply_command` promises.

    The behavioural proof lives beside the seam, in
    `tests/test_adapter_on_compiled.py`
    (`test_an_operator_eager_only_order_suppresses_compiled_dispatch`, red-armed).
    What this row keeps is pgw#1587's own claim: the order is a REFUSAL WITH A
    NAME, and that name is the token the enum reserved for it.
    """
    from gen_worker.serving import adapter_guard

    serve_posture.reset()
    try:
        assert adapter_guard._eager_only_reason() == "", (
            "a worker under no order must not report one")

        assert serve_posture.apply_command(
            True, actor="operator@test", reason="a card under investigation")
        why = adapter_guard._eager_only_reason()
        assert why, "the dispatch seam cannot see a standing order"
        assert "operator@test" in why and "a card under investigation" in why
        assert serve_posture.REASON == "operator_eager_only", (
            "the order's token must stay the one EagerPhase reserved for it — "
            "never counted with the failure classes, never with "
            "`hub_ordered_eager` (one PLAN's backend, not a standing order)")

        # RELEASED: policy stops being the cause, with nothing re-armed.
        assert serve_posture.apply_command(False, actor="operator@test")
        assert adapter_guard._eager_only_reason() == "", (
            "releasing the order must let dispatch run normally again")
    finally:
        serve_posture.reset()


def test_the_declared_requirement_never_refuses_a_load() -> None:
    """A lane requirement INFORMS; it does not permit (Paul, 2026-08-18, and
    again 2026-08-20). The worker's reader warns on every request and loads.

    This is the fence on the deleted gate: if a declared number ever regains
    the power to stop a load, it fails here rather than on a card.
    """
    from gen_worker.serving.placement import DeviceFacts, Shortfall, shortfalls

    facts = DeviceFacts(name="tiny", vram_gib=4.0, sm=89)

    class _NoLanes:
        pass

    assert shortfalls(_NoLanes, None, facts=facts) == (), (
        "an eager-permanent model declares no lane and so no requirement"
    )
    message = Shortfall("min_vram_gb", "sdxl.diffusers@1+plain.bf16@1", 7.0, 4.0, "tiny").message
    assert "Running anyway" in message
    assert "never permission" in message
