"""pgw#992: "free right now" is not a budget for K children's simultaneous peak.

The run this file is written from — the FIRST AOT mint ever to reach the
compile phase on the real path, dead at entry 2 of 36, deterministically::

    width_reason  'K=2 (vram-bound, goals=mint): 29.5 GiB VRAM (sampled)
                   / 9.9 GiB per entry (estimated) -> 2'
    peak_child_device_bytes  6461325312     (6.02 GiB, MEASURED)
    pool_workers 2 -> 4      peak_concurrency 4

    OutOfMemoryError: tried to allocate 14.00 MiB; 2.69 MiB free of 44.39 GiB
       9.54 GiB  eager-serving parent      (resident, pgw#784's contract)
      16.20 GiB  mint child's pipeline     (resident, this process)
      18.61 GiB  four entry children

A4's premise was right — the 9.9 GiB estimate really was ~56 % unobserved, and
6.02 GiB really is the truth. Its arithmetic divided a MOMENTARY free sample by
that truth and called the quotient a simultaneous budget. It is not one: the
sample was taken before the widened children existed, and the two resident
consumers grew from 14.9 GiB to 25.7 GiB against the same card while it aged.

Every number below is that pod's. The card is the variable; A4's divisor is
held constant, which is the exact inverse of ``test_pool_rewiden_pgw868_a4``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import mint_budget, worker_goals

_GIB = 1024 ** 3

# --- the pod, in bytes ------------------------------------------------------
CARD_TOTAL = 47661043712          # 44.39 GiB, the L40S as the driver reports it
FREE_AT_OPEN = 31664532480        # 29.49 GiB — the sample A4 divided
SERVING_PARENT = 10243173417      # 9.54 GiB, resident throughout by design
OWN_AT_OPEN = CARD_TOTAL - FREE_AT_OPEN - SERVING_PARENT   # 5.36 GiB
OWN_PEAK = 17394617548            # 16.20 GiB — the mint child's pipeline
MEASURED_ENTRY_PEAK = 6461325312  # 6.02 GiB, from a real EntryReport


def _incident_census() -> pool.CardCensus:
    return pool.CardCensus(CARD_TOTAL, FREE_AT_OPEN, OWN_AT_OPEN, "sampled")


def _incident_width() -> pool.PoolWidth:
    """K=2 from the REAL policy on the REAL estimate — reproduced, not typed."""
    return pool.entry_workers(
        36, vcpus=256, available_bytes=116 * _GIB, peak_rss_bytes=3 * _GIB,
        free_vram_bytes=FREE_AT_OPEN, device_bytes=9.9 * _GIB and int(9.9 * _GIB),
        device_basis="estimated", device_lock=True,
        goals=worker_goals.MINT_ONLY)


def _pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
          census: pool.CardCensus, own_peak: int) -> pool.EntryCompilePool:
    monkeypatch.setattr(pool, "card_census", lambda device=-1: census)
    monkeypatch.setattr(pool, "own_device_high_water", lambda device=-1: own_peak)
    return pool.EntryCompilePool(tmp_path / "pool", width=_incident_width())


def _observe(box: pool.EntryCompilePool, reserved: int, n: int = 2) -> None:
    for i in range(n):
        box.observe_entry_device(pool.EntryReport(
            entry=f"unet/dim={i}", status=pool.COMPILED,
            peak_device_reserved_bytes=reserved))
        box._rewiden()


# ---------------------------------------------------------------------------
# RED: the incident, reproduced and then refused
# ---------------------------------------------------------------------------

def test_the_free_sample_alone_still_says_K4_which_is_what_went_wrong() -> None:
    """The defect, isolated: A4's own question, unchanged, still answers 4.

    This is not a test of the fix — it is the control that proves the fix is
    changing the ANSWER and not the question. If this row ever stops saying 4,
    the incident is no longer being reproduced and the next row proves nothing.
    """
    ask = mint_budget.entry_device_ask(MEASURED_ENTRY_PEAK)
    a4 = pool.entry_workers(
        36, vcpus=256, available_bytes=116 * _GIB, peak_rss_bytes=3 * _GIB,
        free_vram_bytes=FREE_AT_OPEN, device_bytes=ask,
        device_basis="measured", device_lock=True,
        goals=worker_goals.MINT_ONLY)
    assert a4.workers == 4, a4.reason
    # ...and four of them do not fit beside the residents. This is the OOM.
    assert 4 * ask + SERVING_PARENT + OWN_PEAK > CARD_TOTAL


def test_the_simultaneity_bound_holds_K_at_two_on_the_incident_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix. Same pod, same measured peak, same A4 arithmetic — and the
    pool does not widen, because the CARD cannot hold the widened set."""
    box = _pool(tmp_path, monkeypatch,
                census=_incident_census(), own_peak=OWN_PEAK)
    # The pool OPENS at K=1, not the K=2 the free sample licensed: 18.65 GiB of
    # card room cannot hold two children at the 9.9 GiB the pod was still
    # ASKING for. That the real pod survived at K=2 is a fact about the
    # estimate being wrong, not about two 9.9 GiB children fitting.
    assert box.width.workers == 1, box.width.reason
    assert box.width.binding == "simultaneity"

    _observe(box, MEASURED_ENTRY_PEAK)

    # ...and the measurement then buys the width back, to the largest K the
    # card can actually hold. This is the convergence the bound is for: A4's
    # observation still moves K, it just cannot move it past the card.
    assert box.width.workers == 2, (
        "A4 asked for K=4 on a 29.49 GiB free SAMPLE; the card holds 44.39 "
        "GiB, of which 25.74 GiB is two resident consumers. Widening past 2 "
        "here is the OOM.")
    ask = mint_budget.entry_device_ask(MEASURED_ENTRY_PEAK)
    terms = box.simultaneity
    assert terms["simultaneity_budget_bytes"] == (
        CARD_TOTAL - SERVING_PARENT - OWN_PEAK)
    assert terms["simultaneity_k_cap"] == 2
    # The granted set actually fits, which is the property that matters.
    assert (box.width.workers * ask + SERVING_PARENT + OWN_PEAK) <= CARD_TOTAL


def test_the_same_measurement_on_a_card_that_can_hold_it_still_widens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound must not be a revert. Give the identical pool a card with no
    co-tenant and a small resident set, and A4's widen goes through."""
    roomy = pool.CardCensus(CARD_TOTAL, CARD_TOTAL - 2 * _GIB, 2 * _GIB,
                            "sampled")
    box = _pool(tmp_path, monkeypatch, census=roomy, own_peak=2 * _GIB)
    _observe(box, MEASURED_ENTRY_PEAK)
    assert box.width.workers == 4, box.width.reason
    assert box.width.per_entry_device_basis == "measured"


# ---------------------------------------------------------------------------
# the terms are named, and every unreadable one refuses
# ---------------------------------------------------------------------------

def test_the_bound_names_every_term_it_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance row 2: a future OOM must say WHICH term was wrong.

    An unnamed bound is the state this incident was debugged out of — the
    width record said ``29.5 GiB / 9.9 GiB -> 2`` and nothing anywhere said
    what else was on the card."""
    box = _pool(tmp_path, monkeypatch,
                census=_incident_census(), own_peak=OWN_PEAK)
    _observe(box, MEASURED_ENTRY_PEAK)

    terms = box.simultaneity
    for key in ("simultaneity_ask_bytes", "simultaneity_own_peak_bytes",
                "simultaneity_tenant_reserve_bytes",
                "simultaneity_budget_bytes", "simultaneity_k_cap",
                "simultaneity_basis"):
        assert key in terms, key
    assert terms["simultaneity_own_peak_bytes"] == OWN_PEAK
    assert terms["simultaneity_ask_bytes"] == mint_budget.entry_device_ask(
        MEASURED_ENTRY_PEAK)
    # ...and the census itself rides the emitted width row.
    census = _incident_census().facts()
    assert census["card_resident_other_bytes"] == SERVING_PARENT
    assert census["card_total_bytes"] == CARD_TOTAL


@pytest.mark.parametrize("basis", ["absent", "unreadable"])
def test_an_unreadable_card_refuses_the_widen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, basis: str,
) -> None:
    """Fail-closed. A pool that cannot price simultaneity does not get to
    assume an empty card — the failure mode of guessing here is a 13-minute
    mint that dies at entry 2 of 36."""
    box = _pool(tmp_path, monkeypatch,
                census=pool.CardCensus(0, 0, 0, basis), own_peak=0)
    _observe(box, MEASURED_ENTRY_PEAK)
    assert box.width.workers == 2
    assert box.simultaneity["simultaneity_verdict"] == "unreadable — no widen"


def test_a_serving_pod_pays_the_tenant_reserve_inside_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`serve_goal` is already in the width record, so the pool knows a tenant
    forward is coming. The reserve is a TERM of the budget, not a pad bolted
    on after the division (§4.24: the bound models the threat)."""
    monkeypatch.setattr(pool, "card_census",
                        lambda device=-1: _incident_census())
    monkeypatch.setattr(pool, "own_device_high_water", lambda device=-1: 0)
    serving = pool.entry_workers(
        36, vcpus=256, available_bytes=116 * _GIB, peak_rss_bytes=3 * _GIB,
        free_vram_bytes=FREE_AT_OPEN, device_bytes=int(9.9 * _GIB),
        device_basis="estimated", device_lock=True,
        goals=worker_goals.WorkerGoals(serve=True, mint=True))
    box = pool.EntryCompilePool(tmp_path / "pool", width=serving)
    _observe(box, MEASURED_ENTRY_PEAK)
    assert box.simultaneity["simultaneity_tenant_reserve_bytes"] == \
        pool.DEVICE_RESERVE_BYTES
    assert box.simultaneity["simultaneity_budget_bytes"] == (
        CARD_TOTAL - SERVING_PARENT - OWN_AT_OPEN - pool.DEVICE_RESERVE_BYTES)


def test_the_census_is_taken_before_any_child_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why the census is a CONSTRUCTION-time reading and can never be retaken.

    The subtraction ``total - free - own`` names the co-tenant only while the
    pool's own children are absent from the card. Taken mid-run it would price
    the pool's own children as co-tenants and narrow forever.
    """
    calls: list[int] = []

    def _census(device: int = -1) -> pool.CardCensus:
        calls.append(device)
        return _incident_census()

    monkeypatch.setattr(pool, "card_census", _census)
    monkeypatch.setattr(pool, "own_device_high_water", lambda device=-1: OWN_PEAK)
    box = pool.EntryCompilePool(tmp_path / "pool", width=_incident_width())
    _observe(box, MEASURED_ENTRY_PEAK, n=4)
    assert len(calls) == 1, "the census is read once, at construction"
    assert box.census.resident_other_bytes == SERVING_PARENT


def test_a_driver_reading_that_does_not_add_up_yields_no_free_capacity() -> None:
    """`free + own > total` is nonsense, and nonsense must not become room."""
    bad = pool.CardCensus(10 * _GIB, 9 * _GIB, 8 * _GIB, "sampled")
    assert bad.resident_other_bytes == 0


# ---------------------------------------------------------------------------
# the z-image contrast specimen: the bound is not about the DIVISOR
# ---------------------------------------------------------------------------

# The same `_rewiden` code on a different pod, from the pgw#992 filing:
#   free_device 16.2 GiB / per_entry 25.0 GiB (ESTIMATED) -> K=1, underwidth=3
# and 16.2 GiB free on an 80 GB card whose static slot sum is 53.3 GiB — ~9 GiB
# of CUDA context, allocator fragmentation and child overhead that no catalog
# arithmetic can see.
ZI_CARD_TOTAL = 85899345920           # 80 GB
ZI_FREE = int(16.2 * _GIB)
ZI_PER_ENTRY_ESTIMATE = 25 * _GIB
ZI_STATIC_SLOT_SUM = int(53.3 * _GIB)


def test_the_estimate_only_LOOKED_safe_and_the_bound_does_not_rely_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant, stated as the thing it must NOT be.

    z-image survived because its per-entry ESTIMATE (25.0 GiB) happened to be
    larger than the truth; the L40S died because its MEASURED peak (6.02 GiB)
    was smaller. A fix that said "prefer the estimate" or "distrust the
    measurement" would be a statement about the DIVISOR — and the divisor is
    not what was wrong. The bound must bite on the card, whichever basis
    supplies the ask.
    """
    census = pool.CardCensus(
        ZI_CARD_TOTAL, ZI_FREE, ZI_STATIC_SLOT_SUM, "sampled")
    monkeypatch.setattr(pool, "card_census", lambda device=-1: census)
    monkeypatch.setattr(
        pool, "own_device_high_water", lambda device=-1: ZI_STATIC_SLOT_SUM)

    # The card's own reading is 16.2 GiB free where the slot sum says 26.7 GiB
    # should be: ~9 GiB is context/fragmentation/child overhead. A bound that
    # summed checkpoints instead of reading the device would over-grant by it.
    invisible = (ZI_CARD_TOTAL - ZI_STATIC_SLOT_SUM) - ZI_FREE
    assert invisible > 9 * _GIB * 0.9

    room = ZI_CARD_TOTAL - census.resident_other_bytes - ZI_STATIC_SLOT_SUM
    for basis, ask in (("estimated", ZI_PER_ENTRY_ESTIMATE),
                       ("measured", MEASURED_ENTRY_PEAK)):
        width = pool.entry_workers(
            4, vcpus=128, available_bytes=256 * _GIB, peak_rss_bytes=3 * _GIB,
            free_vram_bytes=ZI_FREE, device_bytes=ask, device_basis=basis,
            device_lock=True, goals=worker_goals.MINT_ONLY)
        box = pool.EntryCompilePool(tmp_path / f"pool-{basis}", width=width)
        granted = box.width.workers
        run_ask = int(box.width.per_entry_device_bytes or ask)
        # The property, stated so it holds for BOTH bases: the pool never
        # grants a second child the card cannot hold. K=1 is exempt because it
        # is the floor — and on this pod at the ESTIMATE even one child does
        # not fit (25.0 GiB into 15.4 GiB of room), which is the whole point:
        # that pod was not protected by a safe policy, it was at the floor.
        assert granted >= 1
        if granted > 1:
            assert granted * run_ask <= room, (
                f"basis={basis}: granted {granted} children of "
                f"{run_ask / _GIB:.2f} GiB into {room / _GIB:.2f} GiB")
    assert ZI_PER_ENTRY_ESTIMATE > room, (
        "the z-image estimate does not fit even ONCE — 'the estimate kept it "
        "safe' is the floor doing the work, not the policy")


def test_the_constructed_width_is_bounded_too_not_only_the_widen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_rewiden` is not the only way a pool gets a K it cannot hold.

    `entry_workers` divides a free SAMPLE at construction with exactly the same
    blind spot, so the bound belongs on every width the pool runs — not on the
    one path that happens to widen.
    """
    census = pool.CardCensus(CARD_TOTAL, FREE_AT_OPEN, OWN_AT_OPEN, "sampled")
    monkeypatch.setattr(pool, "card_census", lambda device=-1: census)
    monkeypatch.setattr(pool, "own_device_high_water", lambda device=-1: OWN_PEAK)

    # A width the free sample licenses (29.49 / 4 GiB -> 7) but the card cannot
    # hold beside 25.74 GiB of residents.
    optimistic = pool.entry_workers(
        36, vcpus=256, available_bytes=116 * _GIB, peak_rss_bytes=3 * _GIB,
        free_vram_bytes=FREE_AT_OPEN, device_bytes=4 * _GIB,
        device_basis="measured", device_lock=True,
        goals=worker_goals.MINT_ONLY)
    assert optimistic.workers >= 5, optimistic.reason

    box = pool.EntryCompilePool(tmp_path / "pool", width=optimistic)
    assert box.width.workers == (CARD_TOTAL - SERVING_PARENT - OWN_PEAK) // (4 * _GIB)
    assert box.width.binding == "simultaneity"
    # `_rewiden` re-derives against the width the pool ACTUALLY ran.
    assert box.width_initial.workers == box.width.workers


def test_the_bound_never_narrows_below_the_serial_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K=1 is what the pool degrades TO. A bound that could forbid it would
    forbid minting, which is not a safety property."""
    starved = pool.CardCensus(CARD_TOTAL, 1 * _GIB, CARD_TOTAL - 2 * _GIB,
                              "sampled")
    monkeypatch.setattr(pool, "card_census", lambda device=-1: starved)
    monkeypatch.setattr(
        pool, "own_device_high_water", lambda device=-1: CARD_TOTAL)
    box = pool.EntryCompilePool(tmp_path / "pool", width=_incident_width())
    assert box.width.workers == 1
