"""pgw#1099: ``reap_lag_s`` is a MEASURED span, never the outer partition's
catch-all.

The defect
----------
``EntryCompilePool._close_entry_partition`` closed ``compile_s`` with three
members. When a child reported no ``report_epoch`` — pgw#840's case: a child
too old for the span table, or one that died between writing its report and
being reaped — the parent computed

    reap_lag_s = compile_s - child_boot_s - child_wall_s

and wrote the whole unattributable remainder under a name that means "the
child's exit plus the parent's poll granularity". ``aot_compile_pool``'s own
``EntryReport.code_digest`` comment already said this happened ("the parent
absorbed its whole compile into ``reap_lag_s``"), and ``RESIDUALS`` never
listed ``reap_lag_s``, so ``dark_fraction`` reported those entries as **fully
attributed**.

What it cost, measured
----------------------
pgw#1085 §5c's 36-entry sdxl-on-L40S mint recorded a ``reap_lag_s`` median of
259.6 s (max 403.4 s) summing to 164.5 min. pgw#1099 was filed on that number
as "the single largest unclaimed block of time in the run, and no pgw#1051
lever addresses it". Both halves of the reading were wrong, and this file pins
both corrections:

1. ``reap_lag_s`` is a SUB-SPAN of ``compile_s``, so summing it over entries
   compiled at K=3 and comparing that to the pool's wall is a category error —
   ``compile_s`` itself sums to 3.35x the same wall.
2. A 259.6 s median with a 403.4 s max is the signature of the residual branch
   firing, not of poll granularity.

The fix is a declared residual of its own, ``parent_other_s``, recorded on
every entry so ``check`` covers it and ``dark_fraction`` counts it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from gen_worker import aot_compile_spans as spans
from gen_worker.aot_compile_pool import EntryCompilePool, EntryReport, _Running


def _running(entry: str = "unet", *, spawn_epoch: float = 1000.0) -> _Running:
    row = _Running(
        entry=entry,
        proc=None,  # type: ignore[arg-type]  # never touched by this seam
        job=None,  # type: ignore[arg-type]
        program_path=Path("/nonexistent/program.pt2"),
        started=0.0,
        stderr_path=Path("/nonexistent/stderr.log"),
    )
    row.spawn_epoch = spawn_epoch
    return row


def _close(
    report: EntryReport, *, elapsed: float, reap_epoch: float,
) -> Dict[str, float]:
    """Drive the REAL `_close_entry_partition`. Nothing about the partition
    arithmetic touches the pool's construction, so the seam is entered
    directly rather than through a pool double that could disagree with it."""
    pool = EntryCompilePool.__new__(EntryCompilePool)
    pool.entry_stage_seconds = {}
    pool.entry_spawn_seconds = {}
    return pool._close_entry_partition(
        _running(), report, elapsed=elapsed, reap_epoch=reap_epoch)


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_the_outer_partition_declares_its_OWN_residual() -> None:
    """RED before pgw#1099: ``compile_s``'s members were the three measured
    spans, so the level had no residual and the arithmetic had to go
    somewhere. It went into a measured member."""
    assert spans.PARTITIONS["compile_s"] == (
        "child_boot_s", "child_wall_s", "reap_lag_s", "parent_other_s")
    assert "parent_other_s" in spans.RESIDUALS


def test_no_MEASURED_span_is_any_levels_residual() -> None:
    """The invariant that makes the table readable: every level's residual is
    a name that means *only* "unclaimed". If a measured span is also a
    residual, one number means two things and no reader can tell which."""
    measured = {
        member
        for members in spans.PARTITIONS.values()
        for member in members
        if member not in spans.RESIDUALS
    }
    assert measured.isdisjoint(spans.RESIDUALS)
    # exactly one residual per level, and every level has one
    for total, members in spans.PARTITIONS.items():
        got = [m for m in members if m in spans.RESIDUALS]
        assert len(got) == 1, f"{total} has residual members {got!r}"


def test_the_ledger_version_moved_with_the_partition_shape() -> None:
    """A reader must never mix a v1 table (where ``reap_lag_s`` could mean
    'unattributed') with a v2 one."""
    assert spans.SPANS_V >= 2


# ---------------------------------------------------------------------------
# The pool, on the two real branches
# ---------------------------------------------------------------------------


def test_a_reporting_child_measures_reap_lag_and_leaves_the_residual_empty(
) -> None:
    """The good branch is unchanged: an epoch-reporting child's poll lag is
    still measured and named, and nothing lands in the residual."""
    report = EntryReport(
        entry="unet",
        spans={"child_wall_s": 8.6},
        run_start_epoch=1001.0,
        module_import_epoch=1000.5,
        report_epoch=1009.6,
    )
    table = _close(report, elapsed=10.0, reap_epoch=1010.0)

    assert table["child_boot_s"] == pytest.approx(1.0)
    assert table["reap_lag_s"] == pytest.approx(0.4)
    assert table["parent_other_s"] == pytest.approx(0.0)
    assert spans.check(table) == []


def test_a_SILENT_child_puts_the_gap_in_the_residual_not_in_reap_lag() -> None:
    """THE ROW THIS ISSUE EXISTS FOR. A child that reports no epochs — the
    pgw#840 case ``EntryReport.code_digest`` documents — must not have its
    unattributable time written under a measured span's name.

    RED on master: ``reap_lag_s`` came back as the full 259.6 s remainder,
    exactly the shape pgw#1085 §5c misread as poll lag.
    """
    report = EntryReport(entry="unet", spans={}, run_start_epoch=0.0, report_epoch=0.0)
    table = _close(report, elapsed=259.6, reap_epoch=1259.6)

    assert table["reap_lag_s"] == 0.0, (
        "the parent cannot know the poll lag of a child that reported no "
        "epoch, and must not claim a number for it")
    assert table["parent_other_s"] == pytest.approx(259.6)

    # And it stays LOUD. A child this silent recorded no `child_wall_s`
    # either, and `check` must keep naming that — the residual holding the
    # seconds is the honest bookkeeping, not an excuse to stop complaining.
    problems = spans.check(table)
    assert any("child_wall_s" in p for p in problems), problems


def test_the_silent_child_is_now_visible_to_dark_fraction() -> None:
    """The second half of the defect: because ``reap_lag_s`` was never in
    ``RESIDUALS``, an entry whose ENTIRE compile was unattributed reported
    ``dark_fraction == 0.0`` — fully attributed. pgw#830's whole contract is
    that unnamed time is loud."""
    report = EntryReport(entry="unet", spans={}, run_start_epoch=0.0, report_epoch=0.0)
    table = _close(report, elapsed=259.6, reap_epoch=1259.6)

    assert spans.dark_fraction(table) == pytest.approx(1.0)


def test_a_partial_report_keeps_what_it_measured_and_banks_only_the_gap(
) -> None:
    """A child that wrote its report epoch but whose inner ledger is missing
    must keep its real poll lag; only the genuinely unclaimed seconds move."""
    report = EntryReport(
        entry="unet",
        spans={"child_wall_s": 200.0},
        run_start_epoch=1002.0,
        report_epoch=1250.0,
    )
    table = _close(report, elapsed=259.6, reap_epoch=1250.5)

    assert table["child_boot_s"] == pytest.approx(2.0)
    assert table["reap_lag_s"] == pytest.approx(0.5)
    assert table["parent_other_s"] == pytest.approx(57.1)
    assert spans.check(table) == []


def test_every_entry_records_the_residual_so_check_can_see_it() -> None:
    """``check`` reports a partition member that was never recorded. If
    ``parent_other_s`` were written only on the silent branch, every healthy
    entry would trip that rule — which is why it is recorded always, as 0.0
    when the named members already close the level."""
    report = EntryReport(
        entry="unet",
        spans={"child_wall_s": 9.0},
        run_start_epoch=1000.5,
        report_epoch=1009.9,
    )
    table = _close(report, elapsed=10.0, reap_epoch=1010.0)

    assert "parent_other_s" in table
    assert spans.check(table) == []


# ---------------------------------------------------------------------------
# The arithmetic that refutes the filing
# ---------------------------------------------------------------------------


def test_a_subspan_sum_across_entries_is_not_comparable_to_the_pool_wall(
) -> None:
    """pgw#1099's headline — "``reap_lag_s`` sums to 164.5 min on a 92-min
    wall" — is not evidence of anything. ``reap_lag_s`` is INSIDE
    ``compile_s``, and at K entries in flight the per-entry totals sum to
    roughly K times the wall by construction. On the same row-7 table
    ``compile_s`` sums to 308.1 min against the same 92.0-minute wall.
    """
    wall_min = 92.0
    compile_s_sum_min = 308.1          # pgw#1085 §5c, row 7, 36 entries, K=3
    reap_lag_sum_min = 164.5

    assert "reap_lag_s" in spans.PARTITIONS["compile_s"]
    assert compile_s_sum_min > wall_min, (
        "the containing span already exceeds the wall, so its member doing so "
        "carries no information")
    assert reap_lag_sum_min < compile_s_sum_min
