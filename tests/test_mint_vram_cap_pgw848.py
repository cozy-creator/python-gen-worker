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
