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
    assert box.width.workers == 2, box.width.reason

    _observe(box, MEASURED_ENTRY_PEAK)

    assert box.width.workers == 2, (
        "A4 asked for K=4 on a 29.49 GiB free SAMPLE; the card holds 44.39 "
        "GiB, of which 25.74 GiB is two resident consumers. Widening here is "
        "the OOM.")
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
