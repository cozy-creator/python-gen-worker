"""pgw#848: the mint's VRAM cap was the ESTIMATE, not the card.

Two pods, fifteen attempts into the whole-graph proof, died the same way:

    pod   card total   free at OOM   cap imposed   entries exported
    4090   23.52 GiB      660 MiB     11.09 GiB     1 of 36
    L40S   44.39 GiB    21.48 GiB     11.08 GiB     5 of 36

**21.48 GiB free, and the mint died for 30 MiB.** The cap did not move across a
2x card change or a ``vram_gb`` 12->20 change, because it was a property of
neither: ``mint_budget.co_residency().need_bytes`` was handed to the child as a
hard ``set_per_process_memory_fraction``, and for sdxl that is
``4.87 x 1.25 + 4 + 1 = 11.09 GiB`` — a number derived from
``_UNMEASURED_ACTIVATION_FRACTION``, which nobody ever measured.

The estimate answers *should this start*. The ceiling answers *how far may it
go*. They are different questions and this file pins them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gen_worker import aot_mint, mint_budget, mint_child, mint_process

import inspect

from gen_worker.worker_goals import WorkerGoals

# pgw#930: these were `forge=True` / `forge=False`. The pool no longer takes a
# mode boolean — it takes the GOAL SET, and derives its three tenant reserves from
# whether a serve goal is held. Naming the goals makes the test say which fact it
# is exercising instead of which branch of a two-valued ternary.
_SERVE_ONLY = WorkerGoals(serve=True, mint=False, declared="serve")
_MINT_ONLY = WorkerGoals(serve=False, mint=True, declared="forge")

_GIB = 1 << 30

#: sdxl's UNet as both pods measured it, and the fraction the old ceiling
#: applied to it. Kept as data so the arithmetic below is the pods' own.
_SDXL_RESIDENT = int(4.87 * _GIB)


def _budget(*, free: float, allocated: int = _SDXL_RESIDENT) -> Any:
    """``co_residency``'s arithmetic on a stated card, without a card.

    Deliberately re-derived from the module's OWN constants rather than
    hardcoded, so a change to them fails this test instead of drifting past it.
    """
    activation = int(allocated * mint_budget._UNMEASURED_ACTIVATION_FRACTION)
    need = (allocated + activation
            + mint_budget._COMPILE_WORKSPACE_BYTES
            + mint_budget._CUDA_CONTEXT_FLOOR_BYTES)
    free_bytes = int(free * _GIB)
    return mint_budget.MintBudget(
        fits=free_bytes >= need, probed=True, free_bytes=free_bytes,
        need_bytes=need, resident_bytes=allocated,
        activation_bytes=activation,
        cap_bytes=max(need, free_bytes - activation))


def test_the_two_pods_that_died_are_reproduced_by_the_old_ceiling() -> None:
    """Before anything is fixed: show the failure is arithmetic, not luck.

    Both pods printed 11.09/11.08 GiB. If this module's own constants do not
    reproduce that number from sdxl's resident set, the diagnosis is wrong and
    nothing below it is worth landing.
    """
    l40s = _budget(free=21.48)
    assert 11.0 < l40s.need_bytes / _GIB < 11.2, (
        f"the estimate does not reproduce the pods' printed cap: "
        f"{l40s.need_bytes / _GIB:.2f} GiB")
    # ...and the card it died on had twice that free.
    assert l40s.free_bytes / _GIB == pytest.approx(21.48, abs=0.01)
    assert l40s.fits, "it was admitted, then capped below what it was admitted for"


def test_the_ceiling_is_the_card_and_not_the_estimate() -> None:
    """The fix, stated as the property that was violated: on a card with room
    to spare, the child gets the room."""
    l40s = _budget(free=21.48)
    assert l40s.cap_bytes > l40s.need_bytes
    assert l40s.cap_bytes / _GIB == pytest.approx(20.26, abs=0.05), (
        f"cap {l40s.cap_bytes / _GIB:.2f} GiB")
    # The tenant's next forward is still reserved, exactly and by construction.
    assert l40s.free_bytes - l40s.cap_bytes == l40s.activation_bytes


def test_a_tight_card_is_not_widened_and_pgw784_still_holds() -> None:
    """Do NOT weaken the cap into nonexistence.

    On a card where the estimate already eats the free bytes, the ceiling
    falls back to the estimate — and the tenant's activation set is reserved
    on every card, roomy or tight. A child that could take `free` outright is
    a child that can evict the tenant, which is the whole thing pgw#784
    forbids.
    """
    tight = _budget(free=11.5)
    assert tight.fits
    assert tight.cap_bytes == tight.need_bytes, (
        "a tight card must not be widened past the estimate")
    for free in (11.5, 14.0, 21.48, 44.0):
        b = _budget(free=free)
        assert b.cap_bytes <= b.free_bytes, "never license the whole card"
        assert b.cap_bytes >= b.need_bytes, "never cap below what was admitted"


def test_the_cap_that_reaches_the_child_is_the_ceiling_not_the_estimate(
    tmp_path: Path,
) -> None:
    """The wiring, not the arithmetic: `build_request` must hand down
    `cap_bytes`. This is the line that made the arithmetic irrelevant."""
    import inspect

    from gen_worker import mint_delegate

    source = inspect.getsource(mint_delegate.build_cell)
    assert "cap_bytes=budget.cap_bytes" in source, (
        "build_cell still hands the child the ESTIMATE as its hard ceiling")
    assert "cap_bytes=budget.need_bytes" not in source


def test_a_device_oom_during_export_is_a_shortfall_not_a_refusal() -> None:
    """`export_program`'s broad `except Exception` laundered a CUDA OOM into
    `MintRefused` -> EXIT_REFUSED -> never retried: the mint a bigger cap
    would fix was the one mint that could never be given one."""

    class _FakeOOM(RuntimeError):
        pass

    oom = _FakeOOM("CUDA out of memory. Tried to allocate 30.00 MiB")
    with pytest.raises(aot_mint.MintResourceExhausted) as raised:
        aot_mint.raise_if_device_oom(oom, "torch.export(strict=False)")
    assert "OUT OF DEVICE MEMORY" in str(raised.value)
    assert "NOT a deterministic" in str(raised.value)
    assert not isinstance(raised.value, aot_mint.MintRefused)
    assert mint_child._is_resource_error(raised.value) is True
    # ...and an ordinary export failure is still a refusal.
    aot_mint.raise_if_device_oom(ValueError("bad dynamic dim"), "export")


def test_a_failed_attempt_banks_its_device_peak_so_the_next_ask_widens() -> None:
    """The widen-on-OOM path.

    `record_child_peak` was gated on a truthy `peak_vram_bytes`, and the crash
    and refusal report paths carried none — so the attempt that hit the cap
    taught the retry nothing and attempt N+1 re-asked identically, forever.
    """
    fam, lane = "pgw848-cap", "w8a8"
    assert mint_budget.child_peak(fam, lane) == 0
    # A child that died AT its cap reports that cap as its peak.
    mint_budget.record_child_peak(fam, lane, 11 * _GIB)
    assert mint_budget.child_peak(fam, lane) == 11 * _GIB
    # The next ask is built from the fact, not the fraction, and is BIGGER.
    banked = mint_budget.child_peak(fam, lane)
    widened = banked + mint_budget._CUDA_CONTEXT_FLOOR_BYTES
    estimate = (_SDXL_RESIDENT
                + int(_SDXL_RESIDENT * mint_budget._UNMEASURED_ACTIVATION_FRACTION)
                + mint_budget._COMPILE_WORKSPACE_BYTES
                + mint_budget._CUDA_CONTEXT_FLOOR_BYTES)
    assert max(widened, estimate) > estimate, (
        "banking the failed attempt's peak must WIDEN the next ask")


def test_every_terminus_reports_the_device_peak() -> None:
    """A mint that died against its cap is exactly the mint whose peak the
    next attempt needs. All three report paths must carry it."""
    import inspect

    source = inspect.getsource(mint_child.main)
    assert source.count("peak_vram_bytes=_peak_vram()") == 2, (
        "the refusal and crash reports must both carry the device peak")
    assert mint_process.MintReport(status="x").peak_vram_bytes == 0


# ---------------------------------------------------------------------------
# th#1359: on a FORGE pod every tenant reserve in this file protects nobody
# ---------------------------------------------------------------------------


def test_no_serve_goal_means_no_tenant_to_reserve_for(monkeypatch) -> None:
    """The premise of this whole module is a co-resident serving process.

    A pod holding no serve goal receives no tenant dispatch and holds no
    resident serving model, so the reserve is not "small" — it is zero, and the
    mint gets the card. Stated explicitly rather than left to emerge from
    `allocated -> 0`: "mostly falls out on its own" is how the 11.09 GiB ceiling
    survived fifteen attempts.

    pgw#930 rewrote this from a source-text assertion (`"if forge:" in
    inspect.getsource(...)`) into a BEHAVIOURAL one. The old form could only
    ever check that a particular branch was still spelled a particular way, so
    it would have passed unchanged if the branch had been keyed on the wrong
    fact — and it would fail on any refactor that kept the behaviour. It also
    could not express the case that matters now: a pod holding BOTH goals keeps
    its reserve, which is a third answer the two-valued check had no room for.
    """
    reading = mint_budget._DeviceRead(
        free_bytes=40 * _GIB, allocated=5 * _GIB,
        measured_activation=0, activation=3 * _GIB, cache_slack=0)
    monkeypatch.setattr(mint_budget, "_read_device", lambda device: reading)

    serve_only = mint_budget.co_residency(0, goals=_SERVE_ONLY)
    mint_only = mint_budget.co_residency(0, goals=_MINT_ONLY)
    both = mint_budget.co_residency(
        0, goals=WorkerGoals(serve=True, mint=True))

    assert serve_only.activation_bytes == 3 * _GIB, (
        "a pod holding a serve goal must keep its tenant activation reserve")
    assert mint_only.activation_bytes == 0, (
        "a pod holding no serve goal has no tenant to reserve for")
    assert both.activation_bytes == serve_only.activation_bytes, (
        "a pod serving AND minting kept no tenant reserve — this is the case "
        "the deleted `forge` boolean could not express, and getting it wrong "
        "is pgw#846's 11.09 GiB ceiling pointed at a live tenant")

    # The signature takes the GOAL SET, not a mode, and defaults to reading the
    # goals the process entry published.
    params = inspect.signature(mint_budget.co_residency).parameters
    assert "goals" in params and "forge" not in params


def test_a_forge_pool_drops_all_three_tenant_reserves() -> None:
    """CPU headroom, VRAM reserve and the host-RAM reserve are all tenant
    reserves. On pgw#846's attempts 14/15 the VRAM one alone held the pool at
    K=1 on a host that could have run it 127 CPU-side."""
    from gen_worker import aot_compile_pool as pool

    hw: Any = dict(
        vcpus=127, available_bytes=116 * _GIB, device_bytes=int(11.09 * _GIB),
        device_lock=True)
    serving = pool.entry_workers(
        36, free_vram_bytes=int(21.48 * _GIB), goals=_SERVE_ONLY, **hw)
    forge = pool.entry_workers(
        36, free_vram_bytes=int(44.39 * _GIB), goals=_MINT_ONLY, **hw)

    assert serving.workers == 1 and serving.binding == "vram", serving.reason
    assert forge.workers > serving.workers, (
        f"a pod holding no serve goal must not inherit the tenant reserves: "
        f"{serving.reason!r} vs {forge.reason!r}")
    # Every reserve, individually, on identical hardware.
    same = dict(hw, free_vram_bytes=int(44.39 * _GIB))
    a = pool.entry_workers(36, goals=_SERVE_ONLY, **same)
    b = pool.entry_workers(36, goals=_MINT_ONLY, **same)
    assert b.cpu_workers > a.cpu_workers, "CPU headroom not dropped"
    assert b.mem_workers > a.mem_workers, "host-RAM reserve not dropped"
    assert b.device_workers > a.device_workers, "VRAM reserve not dropped"


def test_the_width_row_says_which_regime_it_was() -> None:
    """A K of 1 with a serve goal and a K of 1 without one are different
    defects. The row has to say which — and pgw#930 makes that TWO facts
    rather than one boolean, because a pod can hold both goals at once and a
    single `forge` flag could not describe it."""
    from gen_worker import aot_compile_pool as pool

    mint_only = pool.entry_workers(
        36, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, goals=_MINT_ONLY)
    assert mint_only.facts()["serve_goal"] is False
    assert mint_only.facts()["mint_goal"] is True
    assert "goals=mint" in mint_only.reason
    serving = pool.entry_workers(
        36, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, goals=_SERVE_ONLY)
    assert serving.facts()["serve_goal"] is True
    assert serving.facts()["mint_goal"] is False
    assert "goals=serve" in serving.reason

    # The combination the deleted boolean could not spell.
    both = pool.entry_workers(
        36, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, goals=WorkerGoals(serve=True, mint=True))
    assert (both.facts()["serve_goal"], both.facts()["mint_goal"]) == (True, True)
    assert "goals=serve+mint" in both.reason


def test_serving_mode_is_completely_unchanged() -> None:
    """The forge branch must be additive. A serving pod's width is a number
    two lanes are currently measuring against; it must not move."""
    from gen_worker import aot_compile_pool as pool

    w = pool.entry_workers(
        18, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, goals=_SERVE_ONLY)
    assert w.cpu_workers == (16 - pool.SERVING_HEADROOM_CPUS) // 2 == 7
    assert w.mem_workers == (64 * _GIB - pool.ENTRY_RSS_RESERVE_BYTES) // (
        pool.DEFAULT_ENTRY_PEAK_RSS_BYTES)
