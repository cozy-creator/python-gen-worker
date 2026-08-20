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


def test_sdxl_on_a_7_45_gib_card_keeps_the_denoiser_and_evicts_the_encoders():
    # free 7.45, reserve 2.00 -> budget 5.45. `vae` is forced resident on this
    # family (the `force_upcast` one) and unet+vae alone is 5.31, so the budget
    # admits only the plan that evicts BOTH encoders: resident 5.31, peak 6.70.
    # The single-encoder plan (resident 5.56) is excluded because 5.56 + the
    # 2.00 reserve exceeds 7.45 — which is exactly what the reserve is for.
    plan = _plan(_SDXL, free_gb=7.45, forced=("vae",), order=_SDXL_ORDER)
    assert plan.fits, plan.refusal
    assert plan.offloaded == ("text_encoder", "text_encoder_2")
    assert plan.resident == ("unet", "vae")
    # The whole point: per-request traffic collapses from the pipeline's full
    # weight set, twice over, to one encoder once.
    assert plan.offloaded_bytes < sum(_SDXL.values()) / 4


def test_a_busier_card_refuses_the_rung_rather_than_admitting_a_plan_that_ooms():
    # 7.1 GiB free — a co-tenant took 200 MiB. Nothing clears the transient
    # ceiling any more, and the honest answer is to REFUSE and let the load fall
    # to `model_offload`: slow, correct, loud. This IS a performance cliff and it
    # is owed work (see `_TRANSIENT_RESERVE_BYTES`) — but the alternative was
    # measured on this card and it is an OOM inside `ParkedComponent.onload`.
    plan = _plan(_SDXL, free_gb=7.1, forced=("vae",), order=_SDXL_ORDER)
    assert not plan.fits
    # At the truthful 2.00 reserve the denoiser plus its activations no longer
    # fit at all on this card, so the refusal comes from the BUDGET rather than
    # the transient ceiling. Falling to `model_offload` is the honest answer.
    assert "forced-resident set alone" in plan.refusal


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
    plan = _plan(_SDXL, free_gb=7.45, forced=("vae",), order=_SDXL_ORDER)
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


# --------------------------------------------------------------------------
# pgw#1595 / pgw#1586 — the confession states the DECISION, and the reserve
# comes from the REQUEST
# --------------------------------------------------------------------------


def test_the_applied_summary_separates_techniques_from_their_numbers():
    """pgw#1586 item 3. It used to print the KEY of anything truthy, so two data
    entries rendered as savers that engaged — four names for two techniques. The
    split is by TYPE so the next data entry cannot masquerade either."""
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
    """RED ARM for pgw#1595, which was filed against the wrong cause because of
    exactly this. The rung is chosen against free VRAM BEFORE placement; the old
    line re-read it AFTER, so a plan made at 7.3 GiB printed `free_gb=0.4`
    beside its own name."""
    import gen_worker.models.memory as m

    seen = {}

    def fake_line(**kw):
        seen.update(kw)
        return "line"

    real_line, real_free, real_size = (
        m.transition_line, m.get_available_vram_gb, m.estimate_pipeline_size_gb)
    m.transition_line = fake_line
    m.get_available_vram_gb = lambda *a, **k: 0.4       # post-placement truth
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
    """pgw#1595. The reserve was a constant from ONE workload shape, and a
    28-step job overran it. The endpoint's declared peak was already in the
    caller and was being dropped."""
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
    # declared 9.0 total - 6.5 weights = 2.5 GiB of activations, above the 1.25
    # constant, so the budget must shrink by the declared figure instead.
    assert abs(budgets[1] - (8.0 - 2.5)) < 0.01, (
        "the declared per-request peak was ignored — the defect pgw#1595 found")


def test_the_probe_reports_its_measurement_even_when_it_passes():
    """pgw#1559 class, in this rung's own code: success was INFO and inaudible
    at the endpoint's WARNING level, so a passing probe and a probe that never
    ran looked identical."""
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
    """THE INVARIANT THE RESERVE EXISTS FOR — asserted through the PRODUCTION
    path, `_plan_partial_resident`, because that is where the budget formula
    lives.

    An earlier version of this test called `plan_component_residency` directly
    and computed the budget itself, which made it TAUTOLOGICAL: it passed with
    the reserve reverted AND with the budget formula stripped of its reserve
    subtraction. A test that cannot fail is not a test. This one goes through
    the real planner, so breaking `budget = free - reserve` turns it red.

    The denoise phase holds resident weights and activations at once, so
    `resident + reserve <= free` must hold for every admitted plan — otherwise a
    plan fits its own budget and still OOMs mid-denoise, which is what pgw#1595
    found on the card.
    """
    import gen_worker.models.memory as m

    pipe = _pipeline()
    total = sum(
        p.numel() * p.element_size()
        for c in pipe.components.values()
        if isinstance(c, torch.nn.Module) for p in c.parameters()
    )
    # Sit free VRAM just above the reserve so this toy tree is genuinely over
    # budget and eviction is forced, exactly as SDXL is on the real card.
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
    """pgw#1586. A parked component's blocks stay in the caching allocator's
    pool — `park()` drops the reference, the allocator keeps the block, and
    `mem_get_info` never sees it return. The plan counted those bytes as freed
    while this probe counted them as used, so the SAME bytes were both.

    Measured on the card: driver_free 0.45 + reusable cache 1.56 = 2.01 GiB
    against a 2.00 GiB reserve. Reading driver-free alone made a workable plan
    look 1.56 GiB short, which is a SPURIOUS REFUSAL — the conservative
    direction, which is why nothing had failed from it yet.
    """
    import gen_worker.models.partial_resident as pr

    pipe = _pipeline()
    plan, armed = _arm(pipe)
    assert armed
    parked = getattr(pipe, PARKED_COMPONENTS_ATTR)

    floor = 256 * _MIB
    driver_free = 100 * _MIB          # on its own, below the floor
    cache = 400 * _MIB                # reusable pool the allocator still holds

    real_attr = pr._placement_attribution
    pr._placement_attribution = lambda torch_mod: {"attr_cache_bytes": cache}
    try:
        ok, reported = pr.probe_plan(
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


def test_EVERY_rung_confesses_the_decision_time_free_vram_not_a_re_read():
    """pgw#1586 closing the class pgw#1595 opened.

    pgw#1595's fix threaded the plan-time figure into the `partial_resident`
    confession ONLY. Six siblings — `model_offload`, `sequential`,
    `partial_stream`, both `cpu` arms and the fall-through — kept re-reading
    free VRAM AT REPORT TIME, after placement. Within hours the pgw#1548 lane
    read `free_gb=0.4` off a `model_offload` line on a card with 7.9 GiB free at
    boot and reached for a boot-ordering cause — the SAME wrong conclusion
    pgw#1595 was filed on, from the same artefact, on a rung the fix had not
    covered.

    So this asserts the CLASS, by reading the source: no `_report_offload_engaged`
    call may omit `plan_free_gb`. A per-rung test would have passed for
    `partial_resident` and missed the other six, which is exactly how the first
    fix shipped incomplete.
    """
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
