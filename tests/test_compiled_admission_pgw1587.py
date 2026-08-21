"""Compiled admission is PROBED and per-target — never a declared refusal."""

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

    def sink(target: Any) -> Any:
        sink_calls.append(target)
        return "ARMED"

    return LoadContext(
        binding=DeployBinding(checkpoint_ref="r", checkpoint_dir=Path(".")),
        compile_sink=sink,
    )


def test_the_rung_that_parks_named_components_is_not_the_rung_that_moves_all() -> None:
    """`touches_host_ram` is ACCOUNTING; `moves_every_component` is STABILITY."""
    assert touches_host_ram("partial_resident"), "it does hold weights on the host"
    assert not moves_every_component("partial_resident"), (
        "it moves only what it named — that is the point of the rung"
    )
    for every in ("model_offload", "group_offload", "sequential", "cpu"):
        assert moves_every_component(every), every
    assert not moves_every_component(""), "no rung engaged moves nothing"
    assert not moves_every_component("native")


def test_the_resident_denoiser_compiles_under_the_offload_rung() -> None:
    """PAUL'S CASE, as a test."""
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
    """The other half of per-target: what the rung DID park still cannot compile, and the refusal names the rung and the reason."""
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


def test_the_probe_refuses_a_plan_the_arithmetic_admitted() -> None:
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
        free_bytes_now=lambda: 0,
    )
    assert not armed, "the card disagreed with the plan, and the card is right"
    assert not parks_module(pipe.text_encoder), (
        "a refused probe must leave no parked component behind"
    )


def test_an_operator_eager_only_order_is_observed_by_the_dispatch_seam() -> None:
    """The third entry into loud eager, RE-AIMED at the seam that runs."""
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

        assert serve_posture.apply_command(False, actor="operator@test")
        assert adapter_guard._eager_only_reason() == "", (
            "releasing the order must let dispatch run normally again")
    finally:
        serve_posture.reset()


def test_the_declared_requirement_never_refuses_a_load() -> None:
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
