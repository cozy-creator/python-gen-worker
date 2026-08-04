"""pgw#877 (#1 + #2 + #4): the entry-child DEVICE measurement must reach the
decision it exists for, and a measurement must be allowed to NARROW.

One defect in three parts, which is why they are one test file:

1. **The device bank was read in a process that can never have written it.**
   ``mint_budget._CHILD_PEAKS`` is module-local and its only writer is
   ``mint_delegate.record_child_peak``, in the SERVING PARENT. The other
   reader of ``co_residency`` is ``aot_mint._entry_device_bytes``, which runs
   INSIDE THE MINT CHILD, where the dict is empty by construction. So
   ``banked`` was 0 on every mint on every pod, and the per-entry device ask
   was always the estimate — which is why ``per_entry_device_bytes`` printed
   exactly ``allocated * 1.25 + 5 GiB`` (11.09 GiB) on a 4090 AND on an L40S,
   unmoved by a 2x card change or a ``vram_gb`` 12->20 change. The HOST half
   of the identical loop does not have this hole, because it rides
   ``MintRequest.entry_peak_rss_bytes``. The asymmetry is the proof, and the
   mirror field is the fix.
2. **``EntryReport.peak_device_bytes`` had no reader.** pgw#868 A4 measured
   the real thing and left it telemetry-only; ``_collect`` decoded it and
   dropped it. That is the measurement that ends the estimate.
3. **The bank could only ever RAISE the ask** — ``max(banked, estimate)``, so
   *an estimate acting as a floor a measurement isn't allowed to correct*.
   Wire the other two and a measurement still narrows nothing.

The observable that matters is on-pod: ``per_entry_device_bytes`` MOVES on a
real mint, off the second mint's own measurement. These tests prove every
link of the relay off-pod so that the on-pod run is a confirmation and not a
first attempt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import mint_budget

from gen_worker.worker_goals import WorkerGoals

# pgw#930: these were `forge=True` / `forge=False`. The pool no longer takes a
# mode boolean — it takes the GOAL SET, and derives its three tenant reserves from
# whether a serve goal is held. Naming the goals makes the test say which fact it
# is exercising instead of which branch of a two-valued ternary.
_SERVE_ONLY = WorkerGoals(serve=True, mint=False, declared="serve")
_MINT_ONLY = WorkerGoals(serve=False, mint=True, declared="forge")

_GIB = 1 << 30


# --------------------------------------------------------------- part 3 (#4)

def _fake_card(
    monkeypatch: pytest.MonkeyPatch, *, total_gib: float,
    resident_gib: float, peak_gib: float,
) -> None:
    import torch

    total = int(total_gib * _GIB)
    resident = int(resident_gib * _GIB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev=0: (total - resident, total))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev=0: resident)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=0: resident)
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda dev=0: int(peak_gib * _GIB))


def test_a_measured_child_peak_BELOW_the_estimate_narrows_the_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#877 #4 — the disease of this subsystem in one sentence: *an estimate
    acting as a floor a measurement isn't allowed to correct.*

    ``need`` was ``max(banked + ctx, allocated + activation + workspace +
    ctx)``. Banking is ALREADY monotone at the write (``record_child_peak``
    keeps the high-water), so the ``max`` at the READ is a second ratchet that
    can only pin the ask to the estimate. A child that really peaked at 7 GiB
    against an 11 GiB guess re-asked for 11 GiB forever.

    What a measurement may correct is exactly the UNMEASURED terms — the 0.25
    activation fraction and the flat compile workspace. It may never claim the
    child needs less than a weight copy plus a context, because those two the
    child provably holds.
    """
    monkeypatch.setattr(mint_budget, "_CHILD_PEAKS", {})
    fam, execution_lane = "pgw877-narrow", "w8a8"
    # 8 GiB resident, a 9 GiB high-water -> activation floor is the 0.25 guess
    # (2 GiB) since measured (1 GiB) is smaller. Estimate = 8 + 2 + 4 + 1 = 15.
    _fake_card(monkeypatch, total_gib=80, resident_gib=8, peak_gib=9)
    estimated = mint_budget.co_residency(0, family=fam, weight_lane=execution_lane)
    assert estimated.need_bytes == pytest.approx(15 * _GIB, rel=0.01), (
        f"{estimated.need_bytes / _GIB:.2f} GiB")

    # The child then actually ran and peaked at 10 GiB — BELOW the estimate.
    mint_budget.record_child_peak(fam, execution_lane, 10 * _GIB)
    measured = mint_budget.co_residency(0, family=fam, weight_lane=execution_lane)
    assert measured.need_bytes == pytest.approx(11 * _GIB, rel=0.01), (
        f"the measurement must replace the guesses it measured, not be maxed "
        f"against them: {measured.need_bytes / _GIB:.2f} GiB")
    assert measured.need_bytes < estimated.need_bytes
    assert measured.measured


def test_a_measurement_may_never_narrow_below_a_weight_copy_and_a_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor that makes the narrowing safe.

    ``record_child_peak`` banks on EVERY outcome including failures (pgw#848,
    deliberately — the attempt that died is the one whose measurement the next
    needs). So a child that OOMed during ``load`` banks a tiny peak, and a
    narrowing that trusted it blindly would admit a mint onto a card that
    cannot hold a weight copy. ``allocated`` is MEASURED; the activation
    fraction and the workspace are the guesses. Only the guesses may go.
    """
    monkeypatch.setattr(mint_budget, "_CHILD_PEAKS", {})
    fam, execution_lane = "pgw877-floor", "w8a8"
    _fake_card(monkeypatch, total_gib=80, resident_gib=8, peak_gib=9)
    mint_budget.record_child_peak(fam, execution_lane, 1 * _GIB)   # died at load
    budget = mint_budget.co_residency(0, family=fam, weight_lane=execution_lane)
    assert budget.need_bytes == pytest.approx(9 * _GIB, rel=0.01), (
        f"floor is resident + context, not the dead child's peak: "
        f"{budget.need_bytes / _GIB:.2f} GiB")


def test_the_write_side_ratchet_is_the_one_that_keeps_the_ask_honest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narrowing at the READ must not weaken monotonicity at the WRITE: a
    lucky run still cannot talk the ask down."""
    monkeypatch.setattr(mint_budget, "_CHILD_PEAKS", {})
    fam, execution_lane = "pgw877-monotone", "w8a8"
    mint_budget.record_child_peak(fam, execution_lane, 30 * _GIB)
    mint_budget.record_child_peak(fam, execution_lane, 2 * _GIB)
    assert mint_budget.child_peak(fam, execution_lane) == 30 * _GIB


# --------------------------------------------------------------- part 2 (#2)

def test_the_pool_reads_the_entry_childs_device_high_water(
    tmp_path: Path,
) -> None:
    """``EntryReport.peak_device_bytes`` must reach the pool, and the pool must
    keep the RESERVED figure — the child's own docstring says why: allocated
    is what the compile needed, reserved is what the caching allocator held
    and therefore what a CONCURRENT SIBLING actually cannot have. K is a
    question about siblings."""
    width = pool.entry_workers(4, vcpus=8, available_bytes=32 * _GIB,
                               free_vram_bytes=0, device_lock=True)
    box = pool.EntryCompilePool(tmp_path / "pool", width=width)
    assert box.peak_device_bytes == 0
    box.observe_entry_device(
        pool.EntryReport(entry="unet/dim=0", status=pool.COMPILED,
                         peak_device_bytes=3 * _GIB,
                         peak_device_reserved_bytes=5 * _GIB))
    assert box.peak_device_bytes == 5 * _GIB
    # Monotone across entries, like every other high-water here.
    box.observe_entry_device(
        pool.EntryReport(entry="unet/dim=1", status=pool.COMPILED,
                         peak_device_bytes=1 * _GIB,
                         peak_device_reserved_bytes=2 * _GIB))
    assert box.peak_device_bytes == 5 * _GIB
    # A child too old to report reserved still contributes its allocated.
    box.observe_entry_device(
        pool.EntryReport(entry="unet/dim=2", status=pool.COMPILED,
                         peak_device_bytes=9 * _GIB))
    assert box.peak_device_bytes == 9 * _GIB


def test_the_pool_facts_carry_the_device_peak_beside_the_rss_one() -> None:
    """It has to ride the phase table, because the phase table is what
    survives the mint child — which is the process that dies."""
    from gen_worker import aot_mint

    width = pool.entry_workers(4, vcpus=8, available_bytes=32 * _GIB,
                               free_vram_bytes=0, device_lock=True)
    box = pool.EntryCompilePool(Path("/tmp"), width=width)
    box.peak_rss_bytes = 3 * _GIB
    box.peak_device_bytes = 7 * _GIB
    facts = aot_mint._pool_facts(box)
    assert facts["peak_child_rss_bytes"] == 3 * _GIB
    assert facts["peak_child_device_bytes"] == 7 * _GIB


# --------------------------------------------------------------- part 1 (#1)

def test_the_entry_device_bank_is_keyed_monotone_and_narrows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mint_budget, "_ENTRY_DEVICE_PEAKS", {})
    fam, execution_lane = "pgw877-entry", "w8a8"
    assert mint_budget.entry_device_peak(fam, execution_lane) == 0
    mint_budget.record_entry_device_peak(fam, execution_lane, 5 * _GIB)
    assert mint_budget.entry_device_peak(fam, execution_lane) == 5 * _GIB
    mint_budget.record_entry_device_peak(fam, execution_lane, 2 * _GIB)
    assert mint_budget.entry_device_peak(fam, execution_lane) == 5 * _GIB
    assert mint_budget.entry_device_peak(fam, "plain") == 0
    # The ask adds a CUDA context: the peak is the ALLOCATOR's high-water and
    # a context lives outside the allocator.
    assert mint_budget.entry_device_ask(5 * _GIB) == 6 * _GIB
    assert mint_budget.entry_device_ask(0) == 0


def test_the_measured_entry_ask_reaches_the_width_and_says_it_is_measured(
) -> None:
    """The end of the relay. A width sized off a real entry-child peak must
    both MOVE and be LABELLED differently from one sized off the estimate."""
    common: Dict[str, Any] = dict(
        entries=36, vcpus=127, available_bytes=116 * _GIB,
        free_vram_bytes=int(21.48 * _GIB), device_lock=True, goals=_MINT_ONLY)
    estimated = pool.entry_workers(
        device_bytes=int(11.09 * _GIB), device_basis="estimated", **common)
    measured = pool.entry_workers(
        device_bytes=int(3.2 * _GIB), device_basis="measured", **common)
    assert estimated.per_entry_device_basis == "estimated"
    assert measured.per_entry_device_basis == "measured"
    assert estimated.device_workers == 1, estimated.reason
    assert measured.device_workers == 6, measured.reason
    assert measured.workers > estimated.workers, (
        f"the whole point: {estimated.reason!r} vs {measured.reason!r}")


def test_a_readable_card_with_no_footprint_refuses_to_widen() -> None:
    """pgw#877 #5. ``DEFAULT_ENTRY_DEVICE_BYTES = 8 GiB`` was UNREACHABLE:
    ``_entry_device_bytes`` returns 0 only when ``co_residency`` is unprobed,
    and ``_probe_free_device_bytes`` returns 0 in exactly those conditions, so
    the fallback was never consulted. Deleted rather than left as a constant
    nothing can reach — and the branch now does what the other branch's
    comment always CLAIMED: a card we can see but cannot size against does not
    license concurrency on itself. Same rule ``aot_export_parallel.width_for``
    already states for the export footprint."""
    assert not hasattr(pool, "DEFAULT_ENTRY_DEVICE_BYTES")
    w = pool.entry_workers(
        36, vcpus=127, available_bytes=116 * _GIB,
        free_vram_bytes=int(21.48 * _GIB), device_lock=True, goals=_MINT_ONLY)
    assert w.device_workers == 1, w.reason
    assert w.per_entry_device_basis == "unmeasured", w.reason
    # A CPU-only cell is a different case and keeps its unbounded device term.
    cpu_only = pool.entry_workers(
        36, vcpus=127, available_bytes=116 * _GIB, free_vram_bytes=0,
        device_lock=True, goals=_MINT_ONLY)
    assert cpu_only.device_workers == pool.MAX_ENTRY_WORKERS, cpu_only.reason


def test_the_wire_carries_the_entry_device_peak_like_it_carries_the_rss_one(
) -> None:
    """#1's fix, stated as the asymmetry that proved the defect: the HOST ask
    travels on the request, so the child can act on it; the DEVICE ask did
    not, so the child read an empty module-global instead."""
    from gen_worker import mint_process

    fields = set(mint_process.MintRequest.__struct_fields__)
    assert "entry_peak_rss_bytes" in fields
    assert "entry_device_peak_bytes" in fields, (
        "the device measurement has no way to reach the process that sizes K")


def test_the_child_hands_the_banked_device_peak_to_the_mint() -> None:
    """The child must READ the request field. A wire field nothing unpacks is
    the same defect one layer down."""
    import inspect

    from gen_worker import aot_mint, mint_child

    assert "entry_device_peak_bytes" in inspect.getsource(mint_child._mint_aot)
    for fn in (aot_mint.mint, aot_mint._mint_cell):
        assert "entry_device_peak_bytes" in inspect.signature(fn).parameters, (
            f"{fn.__name__} cannot be told what the last mint measured")


def test_the_parent_banks_the_pools_device_peak_from_both_termini() -> None:
    """Banked on EVERY outcome, from the report AND from the live snapshot a
    killed mint leaves behind — the same shape pgw#848 gave the host half,
    for the same reason: the attempt that FAILED is the one whose measurement
    the next attempt needs."""
    import inspect

    from gen_worker import mint_delegate

    source = inspect.getsource(mint_delegate.build_cell)
    assert "record_entry_device_peak" in source
    assert source.count("peak_child_device_bytes") >= 2, (
        "a killed mint writes no report; the snapshot is the only measurement "
        "that survives it")
    assert "entry_device_peak_bytes=" in source


# --------------------------------------------------------------- part 6 (#6)

def test_an_unnamed_device_on_a_multi_gpu_pod_refuses_to_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#877 #6. ``mint_process.child_env`` pins ``CUDA_VISIBLE_DEVICES``
    only when ``request.device >= 0``, so a request carrying -1 leaves the
    child seeing every card — and ``cap_vram`` capped ordinal 0 regardless.

    A cap on the wrong card neither bounds the child nor protects the tenant,
    and reports a note claiming it did both. Refusing, loudly, is the honest
    answer; the pgw#737 admission gate still stands in front of it.
    """
    import torch

    from gen_worker import mint_child

    seen: Dict[str, Any] = {}
    _fake_card(monkeypatch, total_gib=24, resident_gib=0, peak_gib=0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda dev: None)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction",
        lambda frac, dev=0: seen.update(frac=frac, dev=dev))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    note = mint_child.cap_vram(-1, 12 * _GIB)
    assert not seen, "capped a card nobody named"
    assert "NOT applied" in note and "4 cards" in note

    # One visible card is unambiguous even without a pin.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    note = mint_child.cap_vram(-1, 12 * _GIB)
    assert seen["frac"] == pytest.approx(0.5, rel=0.01)
    assert "12.00GiB" in note
