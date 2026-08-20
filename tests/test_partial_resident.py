"""Component-granular residency: the admission arithmetic and the mechanism.

Lineage: pgw#1577. The mechanism tests drive a REAL ``diffusers.DiffusionPipeline``
through ``diffusers``' own offload code with ``torch.nn.Module.to`` counted — the
same instrument the issue's decomposition used, just on the CPU execution device
so it runs without a card. The counter is what makes the defect visible: the
stock rung moves every component every request, and the first test here asserts
that it does. If that test ever goes green without the second changing, the
instrument has gone blind.
"""

from __future__ import annotations

import logging
from typing import Any, List, cast

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers import DiffusionPipeline, ModelMixin  # noqa: E402
from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402

from gen_worker.models.partial_resident import (  # noqa: E402
    PARKED_COMPONENTS_ATTR,
    PARTIAL_RESIDENT_RESERVE_GB,
    apply_component_residency,
    plan_component_residency,
    COMPONENT_RESIDENCY_ATTR,
    plan_for_pipeline,
)

_MIB = 1 << 20
_GIB = 1 << 30

# SDXL bf16, measured on the campaign card (pgw#1577).
_SDXL = {
    "text_encoder": int(0.25 * _GIB),
    "text_encoder_2": int(1.39 * _GIB),
    "unet": int(5.14 * _GIB),
    "vae": int(0.17 * _GIB),
}
_SDXL_ORDER = ("text_encoder", "text_encoder_2", "unet", "vae")


def _plan(sizes, *, free_gb, budget_gb=None, forced=(), denoiser="unet", order=None):
    if budget_gb is None:
        budget_gb = free_gb - PARTIAL_RESIDENT_RESERVE_GB
    return plan_component_residency(
        sizes=sizes,
        order=order or tuple(sizes),
        denoiser=denoiser,
        forced_resident=forced,
        budget_bytes=int(budget_gb * _GIB),
        free_bytes=int(free_gb * _GIB),
    )


# --------------------------------------------------------------------------
# Admission arithmetic
# --------------------------------------------------------------------------


def test_sdxl_on_a_7_3_gib_card_keeps_the_denoiser_and_evicts_the_encoders():
    # free 7.3, reserve 1.25 -> budget 6.05. Weights 6.95; must free 0.90.
    # {text_encoder_2} alone (1.39) clears the budget and is the fewest bytes,
    # but its transient peak is 6.95 of 7.3 free — 96% — so the reserve rejects
    # it. `vae` is forced resident on this family — it is the `force_upcast`
    # one — so the arithmetic takes both encoders at 1.64 and a 6.70 peak.
    plan = _plan(_SDXL, free_gb=7.3, forced=("vae",), order=_SDXL_ORDER)
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("text_encoder", "text_encoder_2")
    assert plan.resident == ("unet", "vae")
    # The whole point: per-request traffic collapses from the pipeline's full
    # weight set, twice over, to the encoders once.
    assert plan.offloaded_bytes < sum(_SDXL.values()) / 4


def test_a_roomier_card_evicts_only_the_one_component_it_has_to():
    # Same pipeline, 7.6 GiB free: the tighter plan the reserve rejected above is
    # admissible here, and the search takes it. The policy is the arithmetic,
    # not a hardcoded component list.
    plan = _plan(_SDXL, free_gb=7.6, order=_SDXL_ORDER)
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("text_encoder_2",)


def test_fewest_bytes_wins_over_fewest_components():
    # Freeing 1.0 GiB: one 1.5 GiB component fits and so do two 0.6 GiB ones.
    # PCIe charges bytes, so the pair is the cheaper plan, and a search that
    # stopped at the first admitting subset size would pick the single.
    sizes = {"unet": 4 * _GIB, "big": int(1.5 * _GIB), "a": int(0.6 * _GIB),
             "b": int(0.6 * _GIB)}
    plan = _plan(sizes, budget_gb=5.7, free_gb=16.0)
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("a", "b")
    assert plan.offloaded_bytes == int(1.2 * _GIB)


def test_the_denoiser_is_never_a_candidate_for_eviction():
    plan = _plan(_SDXL, free_gb=7.3, forced=("vae",), order=_SDXL_ORDER)
    assert "unet" not in plan.offloaded


def test_a_denoiser_larger_than_the_budget_refuses_rather_than_evicting_it():
    # This is where `model_offload` remains the right rung, and the plan must
    # say so instead of quietly producing one that reproduces it.
    plan = _plan(_SDXL, budget_gb=4.0, free_gb=6.0, order=_SDXL_ORDER)
    assert not plan.fits
    assert "forced-resident set" in plan.refusal


def test_forced_resident_components_are_never_evicted():
    # SDXL's force_upcast VAE (gw#441/gw#469) must stay on the card. Diffusers'
    # own `_exclude_from_cpu_offload` does NOT achieve this — it is consulted
    # only for components absent from `model_cpu_offload_seq`, and `vae` is in
    # SDXL's — so this rung enforces it itself.
    plan = _plan(_SDXL, free_gb=7.3, forced=("vae",), order=_SDXL_ORDER)
    assert plan.fits, plan.refusal
    assert "vae" not in plan.offloaded


def test_the_transient_ceiling_rejects_a_plan_the_budget_alone_admits():
    # Budget says the resident set fits; onloading the evicted component during
    # the request would still exceed free VRAM. Admission has to see both, or
    # the rung OOMs mid-encode on a plan that looked fine at load.
    sizes = {"unet": 5 * _GIB, "encoder": 3 * _GIB}
    roomy = _plan(sizes, budget_gb=5.0, free_gb=9.0)
    assert roomy.fits and roomy.offloaded == ("encoder",)
    tight = _plan(sizes, budget_gb=5.0, free_gb=8.4)
    assert not tight.fits
    assert "transient ceiling" in tight.refusal


def test_a_plan_that_fits_reports_the_arithmetic_it_used():
    plan = _plan(_SDXL, free_gb=7.3, forced=("vae",), order=_SDXL_ORDER)
    assert plan.resident_bytes + plan.offloaded_bytes == sum(_SDXL.values())
    assert plan.resident_bytes <= plan.budget_bytes
    assert plan.transient_peak_bytes == plan.resident_bytes + _SDXL["text_encoder_2"]
    assert plan.refusal == ""
    assert "text_encoder_2" in plan.summary()


def test_a_refusal_always_names_a_reason():
    plan = _plan({"vae": _GIB}, budget_gb=8.0, free_gb=12.0, denoiser="unet")
    assert not plan.fits
    assert plan.refusal


# --------------------------------------------------------------------------
# Mechanism, against real diffusers code
# --------------------------------------------------------------------------


class _Block(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 8):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.lin(x)


class _ThreeStagePipeline(DiffusionPipeline):
    """A real ``DiffusionPipeline``: diffusers' own offload code runs on it.

    The component attributes are created by ``register_modules`` at runtime and
    carry no annotations upstream, so they are declared here rather than
    silenced per line."""

    model_cpu_offload_seq = "text_encoder->unet->vae"
    text_encoder: Any
    unet: Any
    vae: Any

    def __init__(self, text_encoder: Any, unet: Any, vae: Any) -> None:
        super().__init__()
        cast(Any, self).register_modules(
            text_encoder=text_encoder, unet=unet, vae=vae
        )

    def run_once(self) -> None:
        self.text_encoder(torch.randn(2, 8))
        for _ in range(3):  # a denoise loop, in miniature
            self.unet(torch.randn(2, 16))
        self.vae(torch.randn(2, 4))
        cast(Any, self).maybe_free_model_hooks()


def _pipeline():
    return _ThreeStagePipeline(_Block(8), _Block(16), _Block(4))


class _MoveCounter:
    """Counts ``Module.to`` by component. This is the PCIe bill, on a device
    where the copies are free — the count is the fact under test."""

    def __init__(self, pipe):
        self.moved: List[str] = []
        self._names = {
            id(m): n for n, m in pipe.components.items()
            if isinstance(m, torch.nn.Module)
        }
        self._orig = torch.nn.Module.to

    def __enter__(self):
        names, moved, orig = self._names, self.moved, self._orig

        def counted(module, *args, **kwargs):
            name = names.get(id(module))
            if name is not None:
                moved.append(name)
            return orig(module, *args, **kwargs)

        torch.nn.Module.to = counted
        return self

    def __exit__(self, *exc):
        torch.nn.Module.to = self._orig
        return False


def test_stock_model_offload_moves_every_component_every_request():
    """THE DEFECT, stated as a passing test. `model_offload` evicts the whole
    pipeline after each call and re-onloads it before the next — on SDXL that
    is 13 GiB of PCIe per request to reclaim 1.2 GiB (pgw#1577)."""
    pipe = _pipeline()
    pipe.enable_model_cpu_offload(device="cpu")
    with _MoveCounter(pipe) as counter:
        pipe.run_once()
    assert "unet" in counter.moved, (
        "the stock rung is supposed to move the denoiser every request; if it "
        "no longer does, this counter is measuring the wrong thing"
    )
    assert set(counter.moved) == {"text_encoder", "unet", "vae"}


def _arm(pipe, **kw):
    plan = plan_for_pipeline(
        pipe,
        budget_bytes=kw.pop("budget_bytes", 1200),
        free_bytes=kw.pop("free_bytes", 64 * _MIB),
        transient_reserve_bytes=kw.pop("transient_reserve_bytes", 0),
        sizer=lambda m: sum(p.numel() * p.element_size() for p in m.parameters()),
        **kw,
    )
    armed = apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t")
    )
    return plan, armed


def test_partial_residency_never_moves_the_components_it_kept():
    pipe = _pipeline()
    plan, armed = _arm(pipe, forced_resident=("vae",))
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("text_encoder",)
    assert armed
    with _MoveCounter(pipe) as counter:
        pipe.run_once()
        pipe.run_once()
    assert "unet" not in counter.moved, (
        "the denoiser left the card — that is the whole cost this rung deletes"
    )
    assert "vae" not in counter.moved
    assert getattr(pipe, COMPONENT_RESIDENCY_ATTR) is plan


def test_an_evicted_component_lives_in_its_host_mirror_between_requests():
    pipe = _pipeline()
    _, armed = _arm(pipe)
    assert armed
    parked = getattr(pipe, PARKED_COMPONENTS_ATTR)["text_encoder"]
    assert not parked.on_device
    pipe.text_encoder(torch.randn(2, 8))
    assert parked.on_device
    pipe.maybe_free_model_hooks()
    assert not parked.on_device, (
        "the request ended with an evicted component still holding VRAM"
    )


def test_at_most_one_evicted_component_holds_the_card_at_a_time():
    """The transient ceiling the plan admitted assumes exactly this. If two can
    be resident together the admission arithmetic is a fiction."""
    pipe = _pipeline()
    plan = plan_component_residency(
        sizes={"text_encoder": 400, "unet": 1000, "vae": 300},
        order=("text_encoder", "unet", "vae"),
        denoiser="unet",
        budget_bytes=1000,
        free_bytes=2000,
        transient_reserve_bytes=0,
    )
    assert plan.fits and set(plan.offloaded) == {"text_encoder", "vae"}
    assert apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t")
    )
    parked = getattr(pipe, PARKED_COMPONENTS_ATTR)
    pipe.text_encoder(torch.randn(2, 8))
    pipe.vae(torch.randn(2, 4))
    assert sum(p.on_device for p in parked.values()) == 1


def test_the_evicted_component_is_released_before_the_denoiser_runs():
    """It must be off the card BEFORE the denoise loop, not at the end of the
    call — otherwise the peak the plan admitted is a floor the request never
    comes back under."""
    pipe = _pipeline()
    _, armed = _arm(pipe)
    assert armed
    parked = getattr(pipe, PARKED_COMPONENTS_ATTR)["text_encoder"]
    pipe.text_encoder(torch.randn(2, 8))
    assert parked.on_device
    pipe.unet(torch.randn(2, 16))
    assert not parked.on_device, (
        "the encoder was still holding VRAM during the denoise loop"
    )


def test_arming_does_not_change_what_the_pipeline_computes():
    pipe = _pipeline()
    torch.manual_seed(0)
    probe = torch.randn(2, 8)
    before = pipe.text_encoder(probe).detach().clone()
    _, armed = _arm(pipe)
    assert armed
    after = pipe.text_encoder(probe)
    assert torch.equal(before, after)
    pipe.maybe_free_model_hooks()
    assert torch.equal(before, pipe.text_encoder(probe))


def test_the_free_hooks_override_does_not_re_arm_the_stock_rung():
    """RED ARM for the trap that makes this rung worse than useless without it.

    ``DiffusionPipeline.maybe_free_model_hooks`` ends by calling
    ``enable_model_cpu_offload``, whose FIRST statement is ``self.to("cpu")``.
    Left in place it drags the resident denoiser to the host and back on every
    call. Delete the override in ``_install_residency_hooks`` and this fails.
    """
    pipe = _pipeline()
    _arm(pipe)

    rearmed: List[str] = []
    stock = type(pipe).enable_model_cpu_offload

    def watched(self, *args, **kwargs):
        rearmed.append("re-armed")
        return stock(self, *args, **kwargs)

    type(pipe).enable_model_cpu_offload = watched
    try:
        pipe.maybe_free_model_hooks()
    finally:
        type(pipe).enable_model_cpu_offload = stock
    assert rearmed == [], (
        "the stock maybe_free_model_hooks re-ran enable_model_cpu_offload, "
        "which starts with self.to('cpu') — the resident set is gone"
    )


def test_arming_refuses_a_plan_that_does_not_fit_rather_than_half_applying_it():
    pipe = _pipeline()
    plan = plan_for_pipeline(
        pipe, budget_bytes=1, free_bytes=1,
        sizer=lambda m: sum(p.numel() * p.element_size() for p in m.parameters()),
    )
    assert not plan.fits
    assert not apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t")
    )
    assert getattr(pipe, COMPONENT_RESIDENCY_ATTR, None) is None


def test_the_placement_census_reads_live_devices_not_the_flag_that_set_them():
    """pgw#1577's red arm for the confession itself. `_pin_unhookable_components`
    has reported `vae_resident` since gw#441 by setting diffusers'
    `_exclude_from_cpu_offload` — a list diffusers consults ONLY for components
    absent from `model_cpu_offload_seq`, where SDXL's `vae` is not. The claim
    was decorative and nothing could see that. A census can, because it reads
    the tensors."""
    from gen_worker.models.memory import component_placement_census

    pipe = _pipeline()
    census = component_placement_census(pipe)
    assert "unet@cpu" in census and "vae@cpu" in census
    pipe.unet.to("meta")
    moved = component_placement_census(pipe)
    assert "unet@meta" in moved, (
        "the census reported a device the component is not on — it is reading "
        "a flag, not the tensors"
    )
    assert "vae@cpu" in moved
