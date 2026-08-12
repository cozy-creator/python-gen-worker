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

# §4.28 / pgw#1092: the reserves used to be relaxed to zero on a pod holding no
# serve goal — a forge pod. That pod class is DELETED, so neither `co_residency`
# nor `entry_workers` takes a goal set any more and every reserve is
# unconditional. The tests below are the inverse of what they used to assert.

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
    fam, execution_lane = "pgw848-cap", "w8a8"
    assert mint_budget.child_peak(fam, execution_lane) == 0
    # A child that died AT its cap reports that cap as its peak.
    mint_budget.record_child_peak(fam, execution_lane, 11 * _GIB)
    assert mint_budget.child_peak(fam, execution_lane) == 11 * _GIB
    # The next ask is built from the fact, not the fraction, and is BIGGER.
    banked = mint_budget.child_peak(fam, execution_lane)
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
# §4.28 / pgw#1092: there is no pod class the tenant reserve does not protect
# ---------------------------------------------------------------------------


def test_the_tenant_reserve_is_unconditional(monkeypatch) -> None:
    """The premise of this whole module is a co-resident serving process, and
    after §4.28 that premise is UNIVERSAL.

    The forge — the mint-only pod that held no serve goal and therefore no
    tenant — is deleted (th#1751 W4 / pgw#1092). The only mint left is the one
    a SERVING pod runs in the background on a cell miss (pgw#784), so there is
    always a tenant and the reserve is never zeroed.

    RED before this change: `co_residency` took `goals=` and returned
    `activation_bytes == 0` for `WorkerGoals(serve=False, mint=True)`.
    """
    reading = mint_budget._DeviceRead(
        free_bytes=40 * _GIB, allocated=5 * _GIB,
        measured_activation=0, activation=3 * _GIB, cache_slack=0)
    monkeypatch.setattr(mint_budget, "_read_device", lambda device: reading)

    assert mint_budget.co_residency(0).activation_bytes == 3 * _GIB, (
        "the tenant activation reserve is unconditional after §4.28")

    # No posture argument survives on the signature: there is nothing left to
    # relax it with, and a knob nobody can set is a knob a lane will re-key.
    params = inspect.signature(mint_budget.co_residency).parameters
    assert "goals" not in params and "forge" not in params


def test_the_pool_keeps_all_three_tenant_reserves_for_every_pod() -> None:
    """CPU headroom, VRAM reserve and the host-RAM reserve are all tenant
    reserves, and none of them can be dropped any more.

    RED before this change: `entry_workers(goals=WorkerGoals(serve=False,
    mint=True))` widened on all three axes against the identical hardware.
    """
    from gen_worker import aot_compile_pool as pool

    hw: Any = dict(
        vcpus=127, available_bytes=116 * _GIB, device_bytes=int(11.09 * _GIB),
        device_lock=True)
    tight = pool.entry_workers(36, free_vram_bytes=int(21.48 * _GIB), **hw)
    assert tight.workers == 1 and tight.binding == "vram", tight.reason

    # The reserves are visible in the arithmetic: the width sized against the
    # same card MINUS each reserve, not against the raw figure.
    wide = pool.entry_workers(36, free_vram_bytes=int(44.39 * _GIB), **hw)
    no_reserve = pool.entry_workers(
        36, free_vram_bytes=int(44.39 * _GIB) + pool.DEVICE_RESERVE_BYTES,
        **hw)
    assert no_reserve.device_workers > wide.device_workers, (
        "the VRAM reserve is not being subtracted at all")

    # And the goal set is gone from the signature entirely.
    params = inspect.signature(pool.entry_workers).parameters
    assert "goals" not in params and "forge" not in params


def test_the_width_row_no_longer_reports_a_goal_set() -> None:
    """A K of 1 is a K of 1: there is one pod class left, so the row has no
    regime to name.

    RED before this change: `facts()` carried `serve_goal`/`mint_goal` and the
    reason string carried `goals=mint` / `goals=serve` / `goals=serve+mint`.
    """
    from gen_worker import aot_compile_pool as pool

    w = pool.entry_workers(
        36, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True)
    facts = w.facts()
    assert "serve_goal" not in facts and "mint_goal" not in facts
    assert "goals=" not in w.reason, w.reason


def test_serving_mode_is_completely_unchanged() -> None:
    """A serving pod's width is a number two lanes are currently measuring
    against; deleting the forge relaxation must not move it."""
    from gen_worker import aot_compile_pool as pool

    w = pool.entry_workers(
        18, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True)
    assert w.cpu_workers == (16 - pool.SERVING_HEADROOM_CPUS) // 2 == 7
    assert w.mem_workers == (64 * _GIB - pool.ENTRY_RSS_RESERVE_BYTES) // (
        pool.DEFAULT_ENTRY_PEAK_RSS_BYTES)
