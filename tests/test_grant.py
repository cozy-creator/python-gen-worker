"""The grant seam's admission rule, on a cardless box.

Every number in this file is a BANKED MEASUREMENT with a tracker citation, not a fixture
invented to make an assertion pass. The decision is the part of the memory system that has
been wrong, so it is the part that has to be falsifiable without a GPU window.
"""

from __future__ import annotations

import pytest

from gen_worker.models.grant import (
    COMPILED,
    EAGER,
    RESIDENT,
    STREAMED,
    ComponentDecl,
    Grant,
    RequestArena,
    Spendable,
    plan_grant,
)

MIB = 1 << 20
GIB = 1 << 30

# --- the banked substrate ------------------------------------------------------------------
# RTX 4070 Laptop, 8188 MiB total / 7803 MiB usable (7.62 GiB). varena#3, 2026-08-21.
#
# anima head-to-head: ComfyUI served the identical 1024x1024/20-step/CFG request FULLY
# RESIDENT at a 6778-7226 MiB peak across four legs with no OOM, and ran 20-37% faster than
# our budget-fed arm, whose peak pinned at 6102 MiB in every leg.
ANIMA_FULLY_RESIDENT_PEAK = 7226 * MIB
ANIMA_BUDGET_FED_PEAK = 6102 * MIB

# The card's free bytes are taken FROM THE DEMONSTRATION rather than from a baseline reading:
# ComfyUI reached a 7226 MiB peak on this card without an OOM, so the card had at least that
# much to give. Nothing is inferred and no baseline is assumed.
CARD_FREE = ANIMA_FULLY_RESIDENT_PEAK

# pgw#1586's GREEN arm: SDXL, weights pinned for the process's life, 7540 MiB peak over
# 5693 MiB of resident weights => 1847 MiB of activations under the fully-resident allocator
# regime (fragmentation the offloaded rung never pays).
SDXL_WEIGHTS = 5693 * MIB
SDXL_ACTIVATIONS = 1847 * MIB

# ⚠️ NOT A MEASUREMENT. pgw#1627's second re-open FALSIFIED the "+1154 MiB, 4/4 runs,
# batch-invariant" figure on-card 2026-08-21: it was the RED run's consumption, and a death
# only ever reports the free memory it consumed. Given 1326 MiB more, the same first call
# consumed ~2474 of 2506 and died identically. sdxl sm_89's compiled demand is UNKNOWN,
# lower-bounded >2501 MiB; 8 GiB is a MEASURED NO for compiled SDXL UNet-only.
#
# The tests below need SOME stamp value to exercise the regime split, so this is a declared
# HYPOTHETICAL and is named as one. Nothing here asserts it is the real demand — the point
# under test is that a stamp's PRESENCE gates the compiled admit and its ABSENCE forces
# eager, which is the rule the falsification above exists to justify.
_HYPOTHETICAL_STAMP = 1154 * MIB


def anima_components(weights: int) -> list[ComponentDecl]:
    """A three-component pipeline shaped like anima: denoiser, text encoder, VAE."""
    te = int(weights * 0.22)
    vae = int(weights * 0.06)
    # The denoiser takes the remainder so the split is EXACT — a test that loses two bytes to
    # truncation cannot assert against a measured peak.
    return [
        ComponentDecl("transformer", weights - te - vae, phase=1),
        ComponentDecl("text_encoder", te, phase=0),
        ComponentDecl("vae", vae, phase=2, pinned=True),
    ]


def cheapest_streamed(components, *, budget_bytes):
    """A stand-in for `partial_resident.plan_component_residency`: page out the smallest
    unpinned non-denoiser components until the resident set fits. The real search is that
    function; this one only has to be honest enough to exercise the seam."""
    movable = sorted(
        (c for c in components if not c.pinned and c.phase != 1),
        key=lambda c: c.weight_bytes,
    )
    resident = sum(c.weight_bytes for c in components)
    out = []
    for c in movable:
        if resident <= budget_bytes:
            break
        out.append(c.name)
        resident -= c.weight_bytes
    return out


# --- the case the whole redesign exists for ------------------------------------------------


def test_full_residency_is_granted_where_comfyui_demonstrated_it_fits():
    """THE anima self-harm case, in one assertion.

    ComfyUI ran this exact request fully resident at a 7226 MiB peak on this card, four
    times, without an OOM, 20-37% faster than we did. Our decider refused full
    residency and pinned at 6102 MiB. A grant that does not come back RESIDENT here has
    reproduced the defect.
    """
    # Weights are sized so that weights + the cold request arena is EXACTLY the peak ComfyUI
    # demonstrated. No number is invented: the demand under test is the measured peak.
    weights = ANIMA_FULLY_RESIDENT_PEAK - RequestArena.cold().bytes
    g = plan_grant(
        anima_components(weights),
        spendable=Spendable(driver_free_bytes=CARD_FREE),
        request=RequestArena.cold(),
        stream_selector=cheapest_streamed,
    )
    assert g.fully_resident, g.line()
    assert g.regime == EAGER
    assert set(g.residency.values()) == {RESIDENT}
    assert g.streamed_bytes == 0
    # And the grant spends the card it was given, rather than a number below it: the granted
    # occupancy is the demonstrated peak, which is 1124 MiB more of the card than our own
    # budget-fed arm used — and that arm was the slower one.
    assert g.resident_bytes + g.request_bytes == ANIMA_FULLY_RESIDENT_PEAK
    assert g.resident_bytes + g.request_bytes > ANIMA_BUDGET_FED_PEAK


def test_the_old_two_gib_reserve_is_what_refused_it():
    """The falsifier for the claim above: reintroduce the deleted guess and the same card,
    the same weights and the same measurement stop granting full residency.

    This is not testing a code path — it is pinning WHY the constant had to go, so a later
    reader cannot restore it as a safety improvement without seeing what it costs.
    """
    weights = ANIMA_FULLY_RESIDENT_PEAK - RequestArena.cold().bytes
    two_gib_guess = RequestArena(bytes=2 * GIB, basis="declared")
    g = plan_grant(
        anima_components(weights),
        spendable=Spendable(driver_free_bytes=CARD_FREE),
        request=two_gib_guess,
        stream_selector=cheapest_streamed,
    )
    assert not g.fully_resident
    assert g.streamed, "the 2 GiB guess pages components the card had room for"


# --- the admission rule ---------------------------------------------------------------------


def test_compiled_requires_full_residency_and_driver_free_alone():
    """COMPILED IFF FULLY RESIDENT — and the AOTI pool must fit in driver_free, never cache.

    Cache is eager-spendable money. The same demand admits compiled when driver_free covers
    it and does NOT when only the cache makes up the difference. The stamp value here is a
    declared HYPOTHETICAL (see `_HYPOTHETICAL_STAMP`) — what is under test is the SPLIT, not
    the number.
    """
    comps = [ComponentDecl("unet", SDXL_WEIGHTS, phase=1)]
    req = RequestArena(
        bytes=SDXL_ACTIVATIONS, basis="measured", compiled_extra_bytes=_HYPOTHETICAL_STAMP
    )
    need = SDXL_WEIGHTS + SDXL_ACTIVATIONS + _HYPOTHETICAL_STAMP

    ok = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=need),
        request=req,
        compile_intent=True,
        stream_selector=cheapest_streamed,
    )
    assert ok.regime == COMPILED
    assert ok.headroom_basis == "driver_free"
    assert ok.fully_resident

    # Same total money, but half of it is allocator cache. AOTI cannot spend it.
    split = plan_grant(
        comps,
        spendable=Spendable(
            driver_free_bytes=need - _HYPOTHETICAL_STAMP,
            allocator_cache_bytes=_HYPOTHETICAL_STAMP,
        ),
        request=req,
        compile_intent=True,
        stream_selector=cheapest_streamed,
    )
    assert split.regime == EAGER, split.line()
    assert split.headroom_basis == "free+cache"
    # The eager arm may spend the cache, so it still gets full residency here.
    assert split.fully_resident


def test_an_unmeasured_compile_intent_is_eager_and_says_so():
    """No stamp, no compiled admit — structurally, and NAMED.

    A mid-graph OOM in a compiled artifact is process death, so "we did not measure it" can
    never be spent as "it probably fits". The note is the point: pgw#1627's lesson is that a
    branch nobody reaches reports as a branch that passed.
    """
    comps = [ComponentDecl("unet", 1 * GIB, phase=1)]
    g = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=64 * GIB),
        request=RequestArena.cold(),
        compile_intent=True,
        stream_selector=cheapest_streamed,
    )
    assert g.regime == EAGER
    assert g.fully_resident
    assert any("compiled not admitted" in n for n in g.notes), g.notes

    # A measured request peak alone is still not enough: the mint stamp is the other half.
    half = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=64 * GIB),
        request=RequestArena(bytes=1 * GIB, basis="measured"),
        compile_intent=True,
        stream_selector=cheapest_streamed,
    )
    assert half.regime == EAGER
    assert any("mint demand stamp" in n for n in half.notes), half.notes


def test_compiled_is_refused_when_anything_would_be_streamed():
    """Full residency is NECESSARY. A grant that pages one component is eager, whatever the
    lane wanted and however good the stamp is."""
    comps = anima_components(6 * GIB)
    g = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=5 * GIB),
        request=RequestArena(bytes=512 * MIB, basis="measured", compiled_extra_bytes=_HYPOTHETICAL_STAMP),
        compile_intent=True,
        stream_selector=cheapest_streamed,
    )
    assert g.regime == EAGER
    assert not g.fully_resident
    assert STREAMED in g.residency.values()


# --- varena always says yes -------------------------------------------------------------


def test_a_demand_the_card_cannot_hold_still_gets_a_grant():
    """varena always says yes. There is no `fits` field and no refusal to branch on.

    What the caller gets instead is a grant with `over_card` set — an honest statement that
    the reactive net (oom_ladder's tile and attention ladders) is what stands between this
    and an OOM. Endpoint code never sees the difference.
    """
    comps = anima_components(40 * GIB)
    g = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=4 * GIB),
        request=RequestArena.cold(),
        stream_selector=cheapest_streamed,
    )
    assert isinstance(g, Grant)
    assert not hasattr(g, "refusal")
    assert g.over_card, g.line()
    assert g.line()  # the confession renders rather than raising


def test_no_selector_still_yields_a_grant():
    """A caller with nothing to page — one component, or no search wired — is not an error."""
    g = plan_grant(
        [ComponentDecl("unet", 40 * GIB, phase=1)],
        spendable=Spendable(driver_free_bytes=4 * GIB),
        request=RequestArena.cold(),
    )
    assert g.regime == EAGER
    assert g.streamed_bytes == 0
    assert g.over_card


def test_a_pinned_component_is_never_streamed_even_if_the_selector_says_so():
    """pgw#1619 ruled a method-driven component a REFUSAL rather than a second hook. Paging a
    dtype-fragile VAE does not fail loudly — it produces black images — so a selector that
    names one is overruled and the override is recorded."""
    comps = anima_components(40 * GIB)

    def bad_selector(components, *, budget_bytes):
        return [c.name for c in components]

    g = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=1 * GIB),
        request=RequestArena.cold(),
        stream_selector=bad_selector,
    )
    assert "vae" not in g.streamed
    assert g.residency["vae"] == RESIDENT
    assert any("refused" in n for n in g.notes), g.notes


# --- the confession ------------------------------------------------------------------------


def test_the_headroom_basis_can_go_red():
    """`headroom_basis` shipped as a constant "free+cache" and could not report the bug it was
    added to expose (pgw#1627). Both values must be reachable from the production entry
    point, or the field is decoration."""
    comps = [ComponentDecl("unet", 1 * GIB, phase=1)]
    stamped = RequestArena(bytes=1 * GIB, basis="measured", compiled_extra_bytes=_HYPOTHETICAL_STAMP)
    compiled = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=64 * GIB),
        request=stamped,
        compile_intent=True,
    )
    eager = plan_grant(
        comps, spendable=Spendable(driver_free_bytes=64 * GIB), request=RequestArena.cold()
    )
    assert {compiled.headroom_basis, eager.headroom_basis} == {"driver_free", "free+cache"}


def test_the_line_names_every_input_not_the_verdict():
    comps = anima_components(6 * GIB)
    g = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=5 * GIB, allocator_cache_bytes=256 * MIB),
        request=RequestArena.cold(),
        stream_selector=cheapest_streamed,
    )
    line = g.line()
    for token in ("regime=", "weights=", "request=", "spendable=", "probe", "free+cache"):
        assert token in line, line


@pytest.mark.parametrize("regime,extra", [(EAGER, 0), (COMPILED, _HYPOTHETICAL_STAMP)])
def test_the_request_arena_demand_is_regime_split(regime, extra):
    req = RequestArena(bytes=512 * MIB, basis="measured", compiled_extra_bytes=_HYPOTHETICAL_STAMP)
    assert req.demand(regime) == 512 * MIB + extra


# --- THE WIRING, and it is the half that shipped dead last time -----------------------------
#
# pgw#1627's lesson, verbatim from the tracker: "'Both guards proven red-able' was true of the
# FUNCTION and false of the WIRING — the green-test-on-an-unreached-seam class." Every test
# above exercises `plan_grant` directly. On a cardless box `apply_low_vram_config` never
# reaches it (the grant returns None when free VRAM reads 0 and the old walk takes over), so
# without the tests below this whole module would be green and unreached in production.

import logging  # noqa: E402
from typing import Any, cast  # noqa: E402

import torch  # noqa: E402
from diffusers import DiffusionPipeline, ModelMixin  # noqa: E402
from diffusers.configuration_utils import ConfigMixin, register_to_config  # noqa: E402


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
        cast(Any, self).register_modules(text_encoder=text_encoder, unet=unet, vae=vae)


def _pipe():
    return _ThreeStagePipeline(_Block(8), _Block(16), _Block(4))


def _with_card(monkeypatch: Any, free_gb: float, total_gb: float = 24.0) -> Any:
    """Present a readable card to `memory`. Without this every test in this file that goes
    through the production entry point silently exercises the fallback instead."""
    import gen_worker.models.memory as m

    monkeypatch.setattr(m, "get_available_vram_gb", lambda *a, **k: free_gb)
    monkeypatch.setattr(m, "get_total_vram_gb", lambda *a, **k: total_gb)
    monkeypatch.setattr(m, "unhookable_components", lambda *a, **k: [])
    return m


def test_the_production_entry_point_actually_reaches_the_grant(monkeypatch):
    """The wiring guard. A readable card must produce a GRANT, not the free-VRAM walk."""
    m = _with_card(monkeypatch, free_gb=24.0)
    seen = {}
    real = m._grant_for_pipeline

    def spy(*a: Any, **k: Any) -> Any:
        g, p = real(*a, **k)
        seen["grant"] = g
        return g, p

    monkeypatch.setattr(m, "_grant_for_pipeline", spy)
    monkeypatch.setattr(
        m, "select_auto_mode",
        lambda **k: pytest.fail("the free-VRAM walk ran; the grant seam was not reached"),
    )
    applied = m.apply_low_vram_config(_pipe(), mode="auto", logger=logging.getLogger("t"))
    assert seen.get("grant") is not None
    assert seen["grant"].fully_resident
    assert applied["mode"] in ("off", "vae_only")


def test_the_wiring_guard_can_go_red(monkeypatch):
    """Falsification, in the file rather than in a scratch edit: an UNREADABLE card must take
    the fallback. If this passes while the test above also passes, both are measuring
    something real; if the seam were unreachable, this would be the only reachable path."""
    m = _with_card(monkeypatch, free_gb=0.0)
    walked: list = []

    def _walk(**k: Any) -> str:
        walked.append(1)
        return "group_offload"

    monkeypatch.setattr(m, "select_auto_mode", _walk)
    m.apply_low_vram_config(_pipe(), mode="auto", logger=logging.getLogger("t"))
    assert walked, "an unreadable card must fall through to the free-VRAM walk"


def test_a_tiny_card_streams_through_the_grant_and_not_the_upgrade_branch(monkeypatch):
    """The other side of the seam: when the grant cannot give full residency it selects the
    streamed set, and the lease-driven UPGRADE branches must NOT also fire. Two deciders for
    one question is the defect this issue exists to remove."""
    m = _with_card(monkeypatch, free_gb=0.000_5, total_gb=24.0)
    monkeypatch.setattr(
        m, "_plan_partial_resident",
        lambda *a, **k: pytest.fail("the legacy partial_resident upgrade ran beside the grant"),
    )
    applied = m.apply_low_vram_config(
        _pipe(), mode="auto", logger=logging.getLogger("t"), stream_budget_bytes=1 << 20
    )
    # Whatever it lands on, it is NOT the lease upgrade — that branch is gated on `grant is
    # None` now, and the fixture above would have failed the test if it had run.
    assert applied["mode"] != "partial_stream"


def test_the_declaration_marks_a_dtype_fragile_vae_pinned(monkeypatch):
    """`pinned` is the author saying *never page this*. Today it is derived, and the derivation
    has to keep carrying pgw#1619's refusal: diffusers drives the VAE by `.decode(...)`, so a
    parked one never onloads."""
    import gen_worker.models.memory as m

    monkeypatch.setattr(m, "unhookable_components", lambda *a, **k: ["vae"])
    decls = m._component_declaration(_pipe())
    by_name = {d.name: d for d in decls}
    assert by_name["vae"].pinned
    assert not by_name["unet"].pinned
    # And the phase order is diffusers' own published sequence, not an invention.
    assert by_name["text_encoder"].phase < by_name["unet"].phase < by_name["vae"].phase


# --- the SDXL case, measured independently by pgw#1604 --------------------------------------

# pgw#1604 finding 1, the VRAM-limbo curve on the same 4070: SDXL's confessed `needed_gb` is
# 6.5, so `select_auto_mode`'s `needed <= avail - 2.0` wants 8.5 GiB free on a 7.62 GiB card.
# There is no `off`/`native` row AT ANY BUDGET. "The ceiling is already a degraded rung."
SDXL_NEEDED = int(6.5 * GIB)
CARD_USABLE = 7803 * MIB

# pgw#1604 finding 5: the SAME request peaks at 2603 MiB under a 6.0 GiB cap, 2218 at 2.5,
# 1962 at 2.0, and 494 once tiling engages. The allocator hands cached blocks back under
# pressure without being asked, so a reserve fitted to a roomy-card high-water mark measures
# the allocator's generosity, not the request's need.
SDXL_PEAK_AT_6GIB_CAP = 2603 * MIB
SDXL_PEAK_AT_2GIB_CAP = 1962 * MIB


def test_sdxl_gets_a_resident_row_the_old_ladder_could_not_produce():
    """pgw#1604's finding 1, inverted into the fix.

    The card cannot hold 6.5 GiB of weights AND a 2.0 GiB guess — that is the arithmetic that
    produced "no `off` row at any budget". It CAN hold 6.5 GiB of weights and the probe floor,
    and whether that actually serves is then a question for the card rather than for a
    constant.
    """
    # SDXL's denoiser is never a paging candidate, so this declaration has nothing to stream
    # — which is exactly the position the old ladder was in. The predicate that matters here
    # is therefore `over_card`: did the demand FIT, not was anything paged.
    comps = [ComponentDecl("unet", SDXL_NEEDED, phase=1)]
    g = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=CARD_USABLE),
        request=RequestArena.cold(),
        stream_selector=cheapest_streamed,
    )
    assert g.fully_resident and not g.over_card, g.line()

    old = plan_grant(
        comps,
        spendable=Spendable(driver_free_bytes=CARD_USABLE),
        request=RequestArena(bytes=2 * GIB, basis="declared"),
        stream_selector=cheapest_streamed,
    )
    assert old.over_card, "the 2 GiB guess is what removed SDXL's resident row"
    # 6.5 + 2.0 = 8.5 GiB wanted against 7.62 GiB of card. pgw#1604's arithmetic, exactly.
    assert old.resident_bytes + old.request_bytes > CARD_USABLE


def test_a_reserve_read_off_a_roomy_card_overstates_the_requirement():
    """pgw#1604 finding 5, as an admission consequence rather than an observation.

    Sizing the request arena from the 6.0 GiB-cap high-water mark costs SDXL its resident row;
    sizing it from what the same request actually needed under pressure does not. The measured
    spread between the two is the cost of measuring a peak on a roomy card.
    """
    # pgw#1586's measured resident weights, so both sides of the comparison are banked
    # numbers and nothing is fitted to make the assertion land.
    comps = [ComponentDecl("unet", SDXL_WEIGHTS, phase=1)]
    spend = Spendable(driver_free_bytes=CARD_USABLE)

    roomy = plan_grant(
        comps, spendable=spend,
        request=RequestArena(bytes=SDXL_PEAK_AT_6GIB_CAP, basis="measured"),
        stream_selector=cheapest_streamed,
    )
    pressured = plan_grant(
        comps, spendable=spend,
        request=RequestArena(bytes=SDXL_PEAK_AT_2GIB_CAP, basis="measured"),
        stream_selector=cheapest_streamed,
    )
    assert roomy.over_card, "a roomy-card peak overstates the requirement past the card"
    assert not pressured.over_card, pressured.line()


def test_no_stamp_is_the_only_honest_answer_when_the_demand_is_unknown():
    """pgw#1627's stamp-source rule, as an admission consequence.

    sdxl sm_89's compiled first-call demand is UNKNOWN — the only figure anyone had was a
    death trace's consumption, and giving the same call 1326 MiB more room made it consume
    that too. "8 GiB is a MEASURED NO for compiled SDXL UNet-only."

    The grant must reach that verdict from the ABSENCE of a stamp, on a card of any size. A
    design that only refused compiled when the arithmetic came out short would have admitted
    it here on a big card, on a demand nobody has ever measured.
    """
    comps = [ComponentDecl("unet", SDXL_WEIGHTS, phase=1)]
    unknown = RequestArena(bytes=SDXL_ACTIVATIONS, basis="measured")  # no compiled stamp
    for card in (8 * GIB, 24 * GIB, 80 * GIB):
        g = plan_grant(
            comps,
            spendable=Spendable(driver_free_bytes=card),
            request=unknown,
            compile_intent=True,
            stream_selector=cheapest_streamed,
        )
        assert g.regime == EAGER, f"{card / GIB:.0f} GiB card: {g.line()}"
        assert any("mint demand stamp" in n for n in g.notes), g.notes


def test_an_over_card_grant_is_never_placed_resident(monkeypatch):
    """The regression this seam could have shipped, caught in self-review.

    `Grant.fully_resident` means NOTHING WAS PAGED. It does not mean the demand fit — a
    declaration whose only movable component is the denoiser (which is never a paging
    candidate: SDXL's shape exactly) comes back unpaged AND `over_card`. Routing that to the
    resident rung places 6.5 GiB of weights on a card that cannot hold them and OOMs at LOAD,
    which is strictly worse than the coarse rung the old ladder chose.

    The predicate for "may I place this on the card" is BOTH properties, and this pins it
    through the production entry point rather than at the arithmetic layer where the bug was
    not visible.
    """
    import gen_worker.models.memory as m

    # A card far too small for the fixture pipeline, but readable — so the grant path runs.
    _with_card(monkeypatch, free_gb=0.000_5, total_gb=24.0)
    seen = {}
    real = m._grant_for_pipeline

    def spy(*a: Any, **k: Any) -> Any:
        g, p = real(*a, **k)
        seen["grant"] = g
        return g, p

    monkeypatch.setattr(m, "_grant_for_pipeline", spy)
    applied = m.apply_low_vram_config(_pipe(), mode="auto", logger=logging.getLogger("t"))
    g = seen.get("grant")
    assert g is not None and g.over_card, g.line() if g else "no grant"
    assert applied["mode"] not in ("off", "vae_only"), (
        f"an over-card grant was placed resident as {applied['mode']!r} — {g.line()}"
    )


def test_the_declaration_totals_what_the_sizer_totals():
    """The grant is sized from the declaration; every confession beside it quotes
    `estimate_pipeline_size_gb`. If those two walks ever disagree, the grant admits against one
    number and reports another, and nothing in either output would say so.

    They agree today because both bottom out in `module_storage_bytes` over the pipeline's
    component vocabulary. Pinned so a change to either walk has to notice the other.
    """
    import gen_worker.models.memory as m

    pipe = _pipe()
    declared = sum(d.weight_bytes for d in m._component_declaration(pipe))
    estimated = int(m.estimate_pipeline_size_gb(pipe) * (1 << 30))
    assert declared == estimated, f"declaration {declared} vs sizer {estimated}"


# --- the group-offload threshold, and what the grant path does to it ------------------------


def test_the_group_offload_threshold_is_never_READ_on_the_grant_path(monkeypatch):
    """`_DEFAULT_GROUP_OFFLOAD_THRESHOLD_GB = 6.0` is a hard boundary that drops TWO rungs at
    once, and the ComfyUI floor ladder measured the cost: at a 6144 MiB budget our run peaks
    at 3494 MiB, leaves 5.8 GiB unspent, and takes 37.4-41.4 s against ComfyUI's 24.2 s
    (1.55-1.71x) — ComfyUI spends 5553 MiB and never leaves its normal path.

    All three uses of the two threshold constants are inside `select_auto_mode`. This asserts
    the consequence across the whole budget range the ladder covered: with a readable card, a
    grant is produced and `select_auto_mode` is never CALLED, so the cliff cannot occur.

    The assertion is deliberately "the walk did not run" rather than "the mode was not
    group_offload". On a cardless test box every offload rung ends as the `cpu` rung
    downstream (`cuda_ok` is False), so a mode assertion here would pass for the wrong reason
    — vacuously green, which is the failure class this file exists to avoid. Each iteration
    also asserts a grant was actually produced, so the guard cannot pass by never reaching
    the seam either.
    """
    import gen_worker.models.memory as m

    # Captured ONCE. Re-reading the attribute inside the loop picks up the previous
    # iteration's spy — monkeypatch does not undo between iterations — and recurses.
    real = m._grant_for_pipeline
    seen: dict = {}

    def spy(*a: Any, **k: Any) -> Any:
        g, p = real(*a, **k)
        seen["grant"] = g
        return g, p

    for free_gb in (7.0, 6.5, 6.0, 5.9, 4.0, 2.5, 1.2, 0.75):
        seen.clear()
        _with_card(monkeypatch, free_gb=free_gb, total_gb=8.0)
        monkeypatch.setattr(m, "_grant_for_pipeline", spy)
        monkeypatch.setattr(
            m, "select_auto_mode",
            lambda **k: pytest.fail(f"the threshold walk ran at {free_gb} GiB free"),
        )
        m.apply_low_vram_config(_pipe(), mode="auto", logger=logging.getLogger("t"))
        assert seen.get("grant") is not None, f"no grant at {free_gb} GiB — guard is vacuous"


def test_group_offload_becoming_unreachable_is_an_UNMEASURED_change():
    """The honest other half, pinned so the test above is not read as a win.

    `group_offload` is the aggressive per-block rung and the grant path can no longer select
    it: the streamed-set search fall-through lands on `model_offload`. pgw#1604 measured the
    6.0 cliff costing 2.1x, so removing it SHOULD help mid-band — but the bottom of the curve
    now serves on a rung it did not use before, and **nothing here has run on a card**.

    There is no assertion to make about speed from a cardless box. What IS assertable is that
    the grant vocabulary contains no rung at all, which is why the rung question moved out of
    the decider and into the projection — and why it is the floor-preservation leg, not this
    file, that decides whether the bottom held.
    """
    from gen_worker.models import grant as G

    assert set(G.__all__) & {"RESIDENT", "STREAMED"}
    assert not any("offload" in n.lower() for n in G.__all__), G.__all__
    assert not any("rung" in n.lower() for n in G.__all__), G.__all__
