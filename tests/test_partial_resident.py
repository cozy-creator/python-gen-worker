"""Component-granular residency: the admission arithmetic and the mechanism."""

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


def test_sdxl_on_a_7_45_gib_card_keeps_the_denoiser_and_evicts_the_encoders():
    plan = _plan(_SDXL, free_gb=7.45, forced=("vae",), order=_SDXL_ORDER)
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("text_encoder", "text_encoder_2")
    assert plan.resident == ("unet", "vae")
    assert plan.offloaded_bytes < sum(_SDXL.values()) / 4


def test_a_busier_card_refuses_the_rung_rather_than_admitting_a_plan_that_ooms():
    plan = _plan(_SDXL, free_gb=7.1, forced=("vae",), order=_SDXL_ORDER)
    assert not plan.fits
    assert "forced-resident set alone" in plan.refusal


def test_fewest_bytes_wins_over_fewest_components():
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
    plan = _plan(_SDXL, budget_gb=4.0, free_gb=6.0, order=_SDXL_ORDER)
    assert not plan.fits
    assert "forced-resident set" in plan.refusal


def test_forced_resident_components_are_never_evicted():
    plan = _plan(_SDXL, free_gb=7.45, forced=("vae",), order=_SDXL_ORDER)
    assert plan.fits, plan.refusal
    assert "vae" not in plan.offloaded


def test_the_transient_ceiling_rejects_a_plan_the_budget_alone_admits():
    sizes = {"unet": 5 * _GIB, "encoder": 3 * _GIB}
    roomy = _plan(sizes, budget_gb=5.0, free_gb=9.0)
    assert roomy.fits and roomy.offloaded == ("encoder",)
    tight = _plan(sizes, budget_gb=5.0, free_gb=8.4)
    assert not tight.fits
    assert "transient ceiling" in tight.refusal


def test_a_plan_that_fits_reports_the_arithmetic_it_used():
    plan = _plan(_SDXL, free_gb=7.45, forced=("vae",), order=_SDXL_ORDER)
    assert plan.resident_bytes + plan.offloaded_bytes == sum(_SDXL.values())
    assert plan.resident_bytes <= plan.budget_bytes
    assert plan.transient_peak_bytes == plan.resident_bytes + _SDXL["text_encoder_2"]
    assert plan.refusal == ""
    assert "text_encoder_2" in plan.summary()


def test_a_refusal_always_names_a_reason():
    plan = _plan({"vae": _GIB}, budget_gb=8.0, free_gb=12.0, denoiser="unet")
    assert not plan.fits
    assert plan.refusal


class _Block(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 8):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return self.lin(x)


class _ThreeStagePipeline(DiffusionPipeline):

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
        for _ in range(3):
            self.unet(torch.randn(2, 16))
        self.vae(torch.randn(2, 4))
        cast(Any, self).maybe_free_model_hooks()


def _pipeline():
    return _ThreeStagePipeline(_Block(8), _Block(16), _Block(4))


class _MoveCounter:

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
    """THE DEFECT, stated as a passing test."""
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
    """The transient ceiling the plan admitted assumes exactly this."""
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
    """It must be off the card BEFORE the denoise loop, not at the end of the call — otherwise the peak the plan admitted is a floor the request never comes back under."""
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
    """RED ARM for the trap that makes this rung worse than useless without it."""
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


def test_the_applied_summary_separates_techniques_from_their_numbers():
    from gen_worker.models.memory import _applied_summary

    line = _applied_summary({
        "mode": "partial_resident",
        "vae_slicing": True,
        "partial_resident": True,
        "vae_tiling": False,
        "partial_resident_offloaded": ["text_encoder_2"],
        "partial_resident_bytes": 1492501790,
        "plan_budget_gb": 6.05,
    })
    techniques = [p for p in line.split(",") if "=" not in p]
    assert techniques == ["vae_slicing", "partial_resident"], (
        "a data entry is being counted as an engaged technique"
    )
    assert "partial_resident_offloaded=text_encoder_2" in line
    assert "plan_budget_gb=6.05" in line


def test_the_confession_reports_the_free_vram_the_DECISION_saw():
    import gen_worker.models.memory as m

    seen = {}

    def fake_line(**kw):
        seen.update(kw)
        return "line"

    real_line, real_free, real_size = (
        m.transition_line, m.get_available_vram_gb, m.estimate_pipeline_size_gb)
    m.transition_line = fake_line
    m.get_available_vram_gb = lambda *a, **k: 0.4
    m.estimate_pipeline_size_gb = lambda *a, **k: 6.5
    try:
        m._report_offload_engaged(
            _pipeline(), "partial_resident", {"partial_resident": True},
            logging.getLogger("t"), plan_free_gb=7.3,
        )
    finally:
        m.transition_line, m.get_available_vram_gb, m.estimate_pipeline_size_gb = (
            real_line, real_free, real_size)

    assert seen["free_gb"] == 7.3, (
        "the confession printed the post-placement re-read as the decision's "
        "input — this is the bug that cost pgw#1595 a root cause"
    )
    assert "free_after_gb=0.4" in seen["detail"], (
        "the post-placement figure is a real fact and must still be reported"
    )


def test_a_declared_per_request_peak_raises_the_reserve_above_the_constant():
    import gen_worker.models.memory as m
    from gen_worker.models.partial_resident import PARTIAL_RESIDENT_RESERVE_GB

    budgets = []
    real_free, real_unhook = m.get_available_vram_gb, m.unhookable_components

    import gen_worker.models.partial_resident as pr
    real_pfp = pr.plan_for_pipeline

    def spy(pipeline, *, budget_bytes, **kw):
        budgets.append(budget_bytes / (1 << 30))
        return real_pfp(pipeline, budget_bytes=budget_bytes, **kw)

    pr.plan_for_pipeline = spy
    m.get_available_vram_gb = lambda *a, **k: 8.0
    m.unhookable_components = lambda *a, **k: []
    try:
        pipe = _pipeline()
        m._plan_partial_resident(pipe, logging.getLogger("t"))
        m._plan_partial_resident(
            pipe, logging.getLogger("t"), peak_vram_gb=9.0, model_size_gb=6.5)
    finally:
        pr.plan_for_pipeline = real_pfp
        m.get_available_vram_gb, m.unhookable_components = real_free, real_unhook

    assert len(budgets) == 2
    assert abs(budgets[0] - (8.0 - PARTIAL_RESIDENT_RESERVE_GB)) < 0.01, (
        "the constant is no longer the floor when nothing is declared")
    assert abs(budgets[1] - (8.0 - 2.5)) < 0.01, (
        "the declared per-request peak was ignored — the defect pgw#1595 found")


def test_the_probe_reports_its_measurement_even_when_it_passes():
    pipe = _pipeline()
    plan, _ = _arm(pipe)
    facts: dict = {}
    apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t"),
        free_bytes_now=lambda: 64 * _MIB, facts=facts,
    )
    assert facts.get("probe_free_bytes") == 64 * _MIB, (
        "a passing probe left no measurement behind"
    )


def test_the_production_planner_leaves_room_for_the_reserve_it_planned_against():
    """THE INVARIANT THE RESERVE EXISTS FOR — asserted through the PRODUCTION path, `_plan_partial_resident`, because that is where the budget formula lives."""
    import gen_worker.models.memory as m

    pipe = _pipeline()
    total = sum(
        p.numel() * p.element_size()
        for c in pipe.components.values()
        if isinstance(c, torch.nn.Module) for p in c.parameters()
    )
    free_bytes = int(PARTIAL_RESIDENT_RESERVE_GB * _GIB) + 1200
    real_free, real_unhook = m.get_available_vram_gb, m.unhookable_components
    m.get_available_vram_gb = lambda *a, **k: free_bytes / _GIB
    m.unhookable_components = lambda *a, **k: []
    try:
        plan = m._plan_partial_resident(pipe, logging.getLogger("t"))
    finally:
        m.get_available_vram_gb, m.unhookable_components = real_free, real_unhook

    assert plan is not None, (
        "the planner admitted nothing where eviction was required — the budget "
        "is no longer being reduced by the reserve"
    )
    assert plan.offloaded, "a plan that evicts nothing cannot be over budget"
    assert plan.resident_bytes + total - total == plan.resident_bytes
    headroom = free_bytes - plan.resident_bytes
    assert headroom >= PARTIAL_RESIDENT_RESERVE_GB * _GIB, (
        f"the admitted plan leaves {headroom} bytes for activations, under the "
        f"{PARTIAL_RESIDENT_RESERVE_GB} GiB reserve it was planned against"
    )


def test_the_probe_counts_the_reusable_allocator_pool_as_available():
    import gen_worker.models.partial_resident as pr

    pipe = _pipeline()
    plan, armed = _arm(pipe)
    assert armed
    parked = getattr(pipe, PARKED_COMPONENTS_ATTR)

    floor = 256 * _MIB
    driver_free = 100 * _MIB
    cache = 400 * _MIB

    real_attr = pr._placement_attribution
    pr._placement_attribution = lambda torch_mod: {"attr_cache_bytes": cache}
    try:
        ok, reported, basis = pr.probe_plan(
            parked, free_bytes_now=lambda: driver_free, floor_bytes=floor)
    finally:
        pr._placement_attribution = real_attr

    assert ok, (
        "the probe refused a plan with 500 MiB genuinely available to "
        "activations because only 100 MiB of it was visible to driver-free"
    )
    assert reported == driver_free, (
        "the reported number must stay the DRIVER-free figure — the soft cache "
        "term belongs in the decision, not in what the log claims is free"
    )
    assert basis == "free+cache", (
        "an eager admit must confess the free+cache basis (pgw#1627)"
    )


# --------------------------------------------------------------------------
# pgw#1627 — the headroom split: allocator cache is EAGER-ONLY money
# --------------------------------------------------------------------------

# The 8 GiB death, as measured (arm-static-c51ba51f/up.log): driver_free at
# the compiled first call and the dead cache parking stranded in the allocator.
_DEATH_FREE = int(1.18 * _GIB)
_DEATH_CACHE = int(1.02 * _GIB)
# A HYPOTHETICAL stamp value exercising the predicate's arithmetic — NOT a
# measured demand. The on-card discriminator FALSIFIED the original
# "+1154 MiB demand" reading of the death log: with 1326 MiB more freed the
# first call consumed ~2474 of 2506 available and died identically, so a death
# only ever reports the free memory it consumed (greedy or weight-scaled). The
# real sdxl sm_89 demand is UNKNOWN (> 2501 MiB) and a stamp may only come
# from a SUCCESSFUL run (pgw#1601).
_HYPOTHETICAL_STAMP = int(1.15 * _GIB)


def test_the_death_shape_numbers_refuse_compiled_and_admit_eager():
    """pgw#1627, the RED arm of the regime split, on the exact numbers that
    killed the process. Counting cache for a compiled admit said 2.2 GiB
    against a real budget of 1.18 — AOTI allocates outside the torch
    allocator, so the cache was money the compiled call could not spend, and
    the process died at step 0 with no traceback."""
    from gen_worker.models.partial_resident import headroom_admits

    ok, basis = headroom_admits(
        regime="compiled", free_bytes=_DEATH_FREE, cache_bytes=_DEATH_CACHE,
        demand_bytes=_HYPOTHETICAL_STAMP,
    )
    assert not ok, (
        "the compiled predicate admitted the death shape — 1.18 GiB of "
        "driver_free cannot cover a 1.15 GiB out-of-allocator demand plus "
        "the floor, and the 1.02 GiB of cache is not compiled money"
    )
    assert basis == "driver_free"

    ok, basis = headroom_admits(
        regime="eager", free_bytes=_DEATH_FREE, cache_bytes=_DEATH_CACHE,
    )
    assert ok, (
        "the same numbers must ADMIT eager — the cache is spendable by the "
        "torch allocator, and refusing here would be the pgw#1586 spurious "
        "refusal all over again"
    )
    assert basis == "free+cache"


def test_post_release_driver_free_admits_compiled_when_a_stamp_fits():
    """Predicate arithmetic only: once `release_cached_vram()` returns the
    parked cache to the driver, driver_free covering stamp+floor ADMITS.
    (The on-card run showed the REAL sm_89 first call takes more than even
    post-release free on this card — >2501 MiB — so with a real stamp the
    same card correctly REFUSES: 8 GiB is a measured NO for compiled SDXL
    UNet-only. A split that refused a fitting stamp would be a cliff.)"""
    from gen_worker.models.partial_resident import headroom_admits

    ok, basis = headroom_admits(
        regime="compiled", free_bytes=_DEATH_FREE + _DEATH_CACHE,
        cache_bytes=0, demand_bytes=_HYPOTHETICAL_STAMP,
    )
    assert ok, (
        "post-release driver_free (2.20 GiB) covers stamp+floor (1.40 GiB) "
        "and must admit — a split that refuses this is a cliff, not a guard"
    )
    assert basis == "driver_free"
    # And a stamp at the on-card LOWER BOUND refuses this card outright.
    ok, _ = headroom_admits(
        regime="compiled", free_bytes=_DEATH_FREE + _DEATH_CACHE,
        cache_bytes=0, demand_bytes=int(2.45 * _GIB),
    )
    assert not ok, "a >2.4 GiB stamp must refuse 2.20 GiB of driver_free"


def test_probe_plan_itself_refuses_a_compiled_leg_the_cache_would_admit():
    """The split through `probe_plan`'s real code path — the mirror image of
    `test_the_probe_counts_the_reusable_allocator_pool_as_available` above:
    the SAME reusable-pool term that rightly rescues an eager admit must be
    invisible to a compiled one."""
    import gen_worker.models.partial_resident as pr

    pipe = _pipeline()
    _, armed = _arm(pipe)
    assert armed
    parked = getattr(pipe, PARKED_COMPONENTS_ATTR)

    real_attr = pr._placement_attribution
    pr._placement_attribution = (
        lambda torch_mod: {"attr_cache_bytes": _DEATH_CACHE})
    try:
        ok, reported, basis = pr.probe_plan(
            parked, free_bytes_now=lambda: _DEATH_FREE,
            regime="compiled", demand_bytes=_HYPOTHETICAL_STAMP,
        )
        eager_ok, _, _ = pr.probe_plan(
            parked, free_bytes_now=lambda: _DEATH_FREE,
        )
    finally:
        pr._placement_attribution = real_attr

    assert not ok and basis == "driver_free", (
        "probe_plan admitted a compiled leg on cache the compiled call "
        "cannot spend — the exact death arithmetic"
    )
    assert reported == _DEATH_FREE
    assert eager_ok, "the eager reading of the same card must stay admitted"


def _admit_through_production_path(*, regime, free_bytes, cache_bytes,
                                   demand_bytes=0):
    """Arm through `apply_component_residency` — the ONLY production caller of
    `probe_plan`, and the call the first #1627 PR left unpassed (dead code on
    the exact path that produced the death log's probe line)."""
    import gen_worker.models.partial_resident as pr

    pipe = _pipeline()
    plan = plan_for_pipeline(
        pipe, budget_bytes=1200, free_bytes=64 * _MIB,
        transient_reserve_bytes=0,
        sizer=lambda m: sum(p.numel() * p.element_size() for p in m.parameters()),
    )
    facts: dict = {}
    real_attr = pr._placement_attribution
    pr._placement_attribution = (
        lambda torch_mod: {"attr_cache_bytes": cache_bytes})
    try:
        armed = apply_component_residency(
            pipe, plan, device="cpu", log=logging.getLogger("t"),
            free_bytes_now=lambda: free_bytes, facts=facts,
            regime=regime, demand_bytes=demand_bytes,
        )
    finally:
        pr._placement_attribution = real_attr
    return armed, facts


def test_an_eager_load_admits_and_confesses_free_plus_cache():
    """Acceptance (c), eager half — through the production path."""
    armed, facts = _admit_through_production_path(
        regime="eager", free_bytes=_DEATH_FREE, cache_bytes=_DEATH_CACHE)
    assert armed
    assert facts.get("headroom_basis") == "free+cache", (
        "the probe admitted and did not say which budget arithmetic it used"
    )


def test_a_compiled_load_refuses_the_death_shape_THROUGH_the_production_path():
    """The wiring test the first #1627 PR was missing: `regime` existed on
    `probe_plan` while its only production caller never passed it, so the
    compiled branch was UNREACHABLE in production and `headroom_basis` was a
    constant "free+cache" — a confession that could not go red. This drives
    the death-shape numbers through `apply_component_residency` itself and
    demands the driver_free basis AND the refusal."""
    armed, facts = _admit_through_production_path(
        regime="compiled", free_bytes=_DEATH_FREE, cache_bytes=_DEATH_CACHE,
        demand_bytes=_HYPOTHETICAL_STAMP)
    assert facts.get("headroom_basis") == "driver_free", (
        "a compiled load's probe still confesses the eager basis — the "
        "regime never reached the predicate (the dead-plumbing finding)"
    )
    assert facts.get("headroom_demand_bytes") == _HYPOTHETICAL_STAMP, (
        "the demand the admit was checked against must be in the confession, "
        "or demand=0 inertness is indistinguishable from a real guard"
    )
    assert not armed, (
        "the death shape was admitted compiled through the production path — "
        "1.02 GiB of allocator cache is not money AOTI can spend"
    )


def test_a_compiled_load_with_no_stamp_is_wired_but_inert():
    """Honesty pin: until pgw#1601's mint-time demand stamp lands, production
    passes demand_bytes=0 and the compiled refusal DOES NOT bite (1.18 GiB >=
    0 + floor admits). The split is WIRED — the basis reads driver_free — and
    the seam release is the only active protection. If this test fails on
    `armed`, a demand source was wired in: move the death-shape refusal
    expectation to that source's tests and delete this pin."""
    armed, facts = _admit_through_production_path(
        regime="compiled", free_bytes=_DEATH_FREE, cache_bytes=_DEATH_CACHE)
    assert armed, "demand=0 admits — the refusal is inert until the stamp"
    assert facts.get("headroom_basis") == "driver_free"
    assert facts.get("headroom_demand_bytes") == 0


def test_the_load_path_threads_compile_intent_into_the_probe():
    """The chain above `apply_component_residency`, pinned at the source level
    so the dead-plumbing class cannot re-open silently: `ctx.load`'s fit-rung
    call derives `regime` from its compile sink (intent — adopt arms AFTER
    load, so `compiled_dispatch_armed` is unusable at admission), and
    `apply_low_vram_config` hands regime + demand to the probe call."""
    import inspect

    import gen_worker.models.memory as m
    import gen_worker.serving.context as c

    fit = inspect.getsource(c.LoadContext._fit_rung)
    assert "_compile_sink" in fit and "regime=" in fit, (
        "ctx.load no longer derives the probe regime from its compile intent"
    )
    low_vram = inspect.getsource(m.apply_low_vram_config)
    idx = low_vram.index("apply_component_residency(")
    call = low_vram[idx:idx + 600]  # the call site plus its kwargs window
    assert "regime=regime" in call and "demand_bytes=" in call, (
        "apply_low_vram_config's probe call dropped regime/demand — the "
        "regime split is dead code in production again"
    )


# --------------------------------------------------------------------------
# pgw#1627 — the park→compiled seam releases the cache, and ONLY for compiled
# --------------------------------------------------------------------------


class _FakeDispatcher:
    """torchcg's adopt shape (adopt.py:551): an INSTANCE `forward` shadowing
    the class one, exposing `armed_graphs()`. Duck-typed exactly like the real
    `_ForwardDispatcher` so the seam's gate reads it the same way."""

    def __init__(self, module: Any, events: List[str]) -> None:
        self.eager_forward = module.forward
        self._events = events

    def armed_graphs(self) -> tuple:
        return ("graph-1",)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._events.append("compiled_call")
        return self.eager_forward(*args, **kwargs)


def _seam_events(adopt: bool, guard: bool = False) -> List[str]:
    """Arm the rung, THEN (optionally) adopt, then serve one request —
    recording park / release / compiled-call order.

    The adopt-AFTER-hook-install ordering is the death log's own
    (residency confession before `adopt: … armed`), and it is load-bearing:
    a gate evaluated at hook-install time reads "not compiled" forever and
    this test's `released` event never appears.

    ``guard=True`` is THE PRODUCTION SHAPE: `host.py` runs
    `adapter_guard.install()` after adopt on the same module, which rebinds
    `module.forward` to its `guarded` closure. The first gate duck-typed
    `forward.armed_graphs` directly and was proven INERT on-card against
    exactly this shape (red_stub_calls=0 with 2 graphs armed) — a seam test
    that arms a dispatcher without the guard stays green forever."""
    import gen_worker.models.memory as m
    import gen_worker.models.partial_resident as pr
    import gen_worker.serving.adapter_guard as ag

    events: List[str] = []
    pipe = _pipeline()
    _, armed = _arm(pipe)
    assert armed

    if adopt:
        pipe.unet.forward = _FakeDispatcher(pipe.unet, events)
        if guard:
            assert ag.install(pipe.unet), (
                "the adapter guard refused the fake dispatcher — the fixture "
                "no longer matches dispatcher_of's duck-type and this test "
                "is not exercising the production forward shape"
            )

    real_park = pr.ParkedComponent.park
    real_release = m.release_cached_vram

    def counted_park(self: Any) -> None:
        if self.on_device:
            events.append("parked")
        real_park(self)

    pr.ParkedComponent.park = counted_park  # type: ignore[method-assign]
    m.release_cached_vram = lambda: events.append("released")
    try:
        # One request, the pipeline's own call order: the parked encoder
        # onloads via its pre-hook; the resident unet's pre-hook parks it and
        # sits exactly at the park→compiled seam.
        pipe.text_encoder(torch.randn(2, 8))
        pipe.unet(torch.randn(2, 16))
    finally:
        pr.ParkedComponent.park = real_park  # type: ignore[method-assign]
        m.release_cached_vram = real_release
    return events


def test_the_seam_releases_the_cache_THROUGH_the_adapter_guard():
    """THE PRODUCTION SHAPE, and the red test the first two PRs lacked:
    dispatcher adopted, then `adapter_guard.install()` rebinds forward to its
    guard closure — the gate must see the dispatcher THROUGH the wrapper
    (via adapter_guard's own accessor) or it is inert on every compiled
    endpoint, which is exactly what the on-card run measured against
    790d0290. Order: parked → released → first compiled call."""
    events = _seam_events(adopt=True, guard=True)
    assert "parked" in events, "the seam never parked — the instrument is blind"
    assert "released" in events, (
        "the park→compiled seam never released the allocator cache WITH THE "
        "ADAPTER GUARD INSTALLED — the gate is reading module.forward "
        "directly instead of asking adapter_guard.dispatcher_of/armed_graphs "
        "(the third gate-keyed-on-a-wrapped-signal instance)"
    )
    assert "compiled_call" in events
    assert events.index("parked") < events.index("released") < events.index(
        "compiled_call"
    ), f"wrong order at the seam: {events}"


def test_the_seam_releases_for_a_bare_dispatcher_too():
    """The pre-guard shape (local runs, any host that skips the guard): a
    dispatcher fronting forward directly must also release. The dispatcher
    is installed AFTER the hooks (adopt runs after the rung arms, per the
    death log), so a release gated at install time — the pgw#1587
    wrong-moment shape — fails here by never firing."""
    events = _seam_events(adopt=True, guard=False)
    assert "parked" in events
    assert "released" in events, (
        "the seam released only through the guard wrapper — a bare armed "
        "dispatcher must gate the release too"
    )
    assert events.index("parked") < events.index("released") < events.index(
        "compiled_call"
    ), f"wrong order at the seam: {events}"


def test_the_seam_keeps_the_cache_for_an_eager_denoiser():
    """The split's other half: no armed dispatcher means the cache is the
    eager regime's activation money (pgw#1586) and must NOT be released."""
    events = _seam_events(adopt=False)
    assert "parked" in events
    assert "released" not in events, (
        "the seam released the allocator cache under an EAGER denoiser — "
        "that cache is eager's activation money, and this empty_cache is a "
        "per-request tax the regime split exists to avoid"
    )


class _MidStagePipeline(DiffusionPipeline):
    """A family whose first resident module AFTER the parked chain is NOT the
    denoiser. On SDXL the two coincide (encoders park, the unet is next), so
    a gate that interrogates the hook's own module is right there by
    execution-order coincidence only — this shape is where that gate goes
    silently blind."""

    model_cpu_offload_seq = "text_encoder->mid->unet->vae"
    text_encoder: Any
    mid: Any
    unet: Any
    vae: Any

    def __init__(self, text_encoder: Any, mid: Any, unet: Any, vae: Any) -> None:
        super().__init__()
        cast(Any, self).register_modules(
            text_encoder=text_encoder, mid=mid, unet=unet, vae=vae
        )


def test_the_seam_gate_reads_the_DENOISER_not_whichever_module_holds_the_hook():
    """pgw#1627 follow-up, finding 3: the release hook rides the first
    resident module after the parked chain, and the gate must still ask the
    DENOISER — the compile target — whether a compiled call is coming. Here
    that first module is `mid` (never compiled) while the dispatcher sits on
    `unet`: a gate keyed to the hook's own module never releases, and this
    test goes red."""
    import gen_worker.models.memory as m
    import gen_worker.models.partial_resident as pr

    events: List[str] = []
    pipe = _MidStagePipeline(_Block(8), _Block(12), _Block(16), _Block(4))
    sizer = lambda mod: sum(  # noqa: E731
        p.numel() * p.element_size() for p in mod.parameters())
    total = sum(sizer(c) for c in cast(Any, pipe).components.values())
    plan = plan_for_pipeline(
        pipe,
        # Room for everything but the text encoder: the cheapest legal plan
        # parks exactly `text_encoder`, so `mid` is the first post-parked
        # resident and carries the release hook.
        budget_bytes=total - sizer(pipe.text_encoder) + 8,
        free_bytes=64 * _MIB,
        transient_reserve_bytes=0,
        sizer=sizer,
    )
    assert plan.fits and plan.offloaded == ("text_encoder",)
    armed = apply_component_residency(
        pipe, plan, device="cpu", log=logging.getLogger("t"))
    assert armed

    pipe.unet.forward = _FakeDispatcher(pipe.unet, events)
    import gen_worker.serving.adapter_guard as ag

    assert ag.install(pipe.unet)  # production shape: guard over the dispatcher
    real_release = m.release_cached_vram
    m.release_cached_vram = lambda: events.append("released")
    try:
        pipe.text_encoder(torch.randn(2, 8))
        pipe.mid(torch.randn(2, 12))
        pipe.unet(torch.randn(2, 16))
    finally:
        m.release_cached_vram = real_release

    assert "released" in events, (
        "no release before the compiled denoiser call — the gate asked the "
        "hook's own module (mid) instead of resolving the denoiser"
    )
    assert events.index("released") < events.index("compiled_call"), (
        f"the release must precede the compiled call: {events}"
    )


def test_EVERY_rung_confesses_the_decision_time_free_vram_not_a_re_read():
    import inspect
    import re

    import gen_worker.models.memory as m

    src = inspect.getsource(m.apply_low_vram_config)
    calls = re.findall(r"_report_offload_engaged\((.*?)\)", src, re.S)
    assert calls, "no confession call sites found — this test has gone blind"
    missing = [c for c in calls if "plan_free_gb" not in c]
    assert missing == [], (
        f"{len(missing)} rung(s) still confess a post-placement re-read as the "
        f"decision's input: {missing}"
    )


class _MethodDrivenBlock(_Block):

    def decode(self, x: Any) -> Any:
        return self.lin(x)

    def encode(self, x: Any) -> Any:
        return self.lin(x)


class _PipelineWithMethodDrivenComponent(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->unet->vae"
    text_encoder: Any
    unet: Any
    vae: Any

    def __init__(self, text_encoder: Any, unet: Any, vae: Any) -> None:
        super().__init__()
        cast(Any, self).register_modules(
            text_encoder=text_encoder, unet=unet, vae=vae
        )


def test_a_forward_pre_hook_does_NOT_fire_on_a_named_method():
    """THE DEFECT ITSELF, asserted so it cannot silently stop being true."""
    m = _MethodDrivenBlock(8)
    fired: List[str] = []
    m.register_forward_pre_hook(lambda mod, a: fired.append("forward"))
    m(torch.zeros(1, 8))
    assert fired == ["forward"], "the hook does not even fire on __call__"
    fired.clear()
    m.decode(torch.zeros(1, 8))
    assert fired == [], (
        "a forward pre-hook fired on a named method — if torch changed this, "
        "pgw#1619's refusal can be revisited"
    )


def test_a_component_this_rung_cannot_enter_is_never_parked():
    from gen_worker.models.partial_resident import method_driven_components

    pipe = _PipelineWithMethodDrivenComponent(_Block(8), _Block(16), _MethodDrivenBlock(4))
    assert method_driven_components(pipe) == ["vae"], (
        "the structural check did not spot the method-driven component"
    )

    sizer = lambda m: sum(p.numel() * p.element_size() for p in m.parameters())
    plan = plan_for_pipeline(
        pipe, budget_bytes=1200, free_bytes=64 * _MIB,
        sizer=sizer, transient_reserve_bytes=0,
        forced_resident=method_driven_components(pipe),
    )
    assert plan.fits, plan.refusal
    assert "vae" not in plan.offloaded, (
        "a component whose onload hook can never fire was selected for parking"
    )


def test_the_PRODUCTION_planner_refuses_to_park_what_it_cannot_enter():
    """THE ONE THAT GUARDS THE FIX, through `_plan_partial_resident`."""
    import gen_worker.models.memory as m

    pipe = _PipelineWithMethodDrivenComponent(
        _Block(20), _Block(16), _MethodDrivenBlock(9)
    )
    total = 1680 + 1088 + 360
    free_bytes = int(PARTIAL_RESIDENT_RESERVE_GB * _GIB) + (total - 328)

    real_free, real_unhook = m.get_available_vram_gb, m.unhookable_components
    m.get_available_vram_gb = lambda *a, **k: free_bytes / _GIB
    m.unhookable_components = lambda *a, **k: []
    try:
        plan = m._plan_partial_resident(pipe, logging.getLogger("t"))
    finally:
        m.get_available_vram_gb, m.unhookable_components = real_free, real_unhook

    assert plan is not None and plan.fits, "the planner admitted nothing"
    assert "vae" not in plan.offloaded, (
        "the production planner parked a component whose onload hook can NEVER "
        "fire — this is the pgw#1619 decode-against-host-weights death"
    )
    assert "text_encoder" in plan.offloaded, (
        "the guard did not merely refuse the vae, it broke the plan"
    )


def test_the_structural_check_does_not_over_refuse_the_denoiser_or_encoders():
    """The guard must cost only what it has to."""
    from gen_worker.models.partial_resident import method_driven_components

    pipe = _PipelineWithMethodDrivenComponent(_Block(8), _Block(16), _MethodDrivenBlock(4))
    refused = method_driven_components(pipe)
    assert "unet" not in refused and "text_encoder" not in refused, (
        f"the guard over-refused: {refused}"
    )
