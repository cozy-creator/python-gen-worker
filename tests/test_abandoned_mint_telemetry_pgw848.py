"""pgw#848: an ABANDONED mint discarded 29 minutes of measurement.

Attempt sixteen ran on the `8559140` cap fix and proved it — 29 minutes with
no memory error, where attempts 14 and 15 died at +11.6 and +11.0 min against
the 11.09 GiB ceiling. Then the worker's endpoint instances were torn down
under the drain path and the mint was abandoned, and the ENTIRE phase table
for those 29 minutes was one row::

    status=abandoned total_s=1741.33 — no cell produced

**Zero `entry:` rows. No `pool` row.** K, its binding constraint, every
per-entry timing and every peak were measured and thrown away, and the
K-and-binding answer had to be re-bought with another pod.

The mechanism: `report.json` is written ONCE, at a terminus the child reaches
under its own power. A child that is group-killed reaches no terminus, raises
nothing, and writes nothing — so `f9c1b2d`'s work on the *aborted* path (the
failed attempt teaches the retry instead of discarding what it measured) never
applied here. Same code, different exit.

The fix is a snapshot on disk, rewritten atomically on every beat, because a
file is the only thing that survives a signal — the same principle the pgw#848
resume design keys on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from gen_worker import aot_compile_pool as pool
from gen_worker import aot_mint, mint_delegate, mint_process

_GIB = 1 << 30


def _progress_midflight(tmp_path: Path) -> aot_mint.MintProgress:
    """A mint that has finished some entries and is inside another — exactly
    the state attempt sixteen was killed in."""
    width = pool.entry_workers(
        36, vcpus=16, available_bytes=64 * _GIB, free_vram_bytes=0,
        device_lock=True, limit=4)
    progress = aot_mint.MintProgress()
    progress.t_mint = 0.0
    progress.width = width
    progress.timings.update({"export_all_s": 61.2})
    progress.pool_ledger = {
        "pool_workers": width.workers, "pool_efficiency": 0.97,
        "peak_child_rss_bytes": 3 * _GIB, "peak_concurrency": width.workers,
    }
    progress.at = {
        "phase": aot_mint.PHASE_INDUCTOR_COMPILE, "step": 30, "total": 36,
        "note": "unet/adapter=true/1024x1024"}
    return progress


def test_a_killed_mints_measurements_are_on_disk_before_it_dies(
    tmp_path: Path,
) -> None:
    """The snapshot exists, is complete, and is written atomically.

    Atomicity is not decoration: the parent reads this file the instant after
    it kills the child, and a half-written table is a table nobody can use.
    """
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    progress = _progress_midflight(tmp_path)

    aot_mint.write_phase_snapshot(snapshot, progress)
    table = json.loads(snapshot.read_text())

    assert table["pool"]["entry_workers"] == progress.width.workers
    assert table["pool"]["binding"] == progress.width.binding
    assert table["pool"]["peak_child_rss_bytes"] == 3 * _GIB
    assert table["at"]["step"] == 30, (
        "the entry a mint DIES ON is the one a reader most needs named")
    assert table["terminus"] == "in_flight"
    assert not list(tmp_path.glob("*.tmp")), (
        "the atomic write must leave no temp file behind")


def test_nothing_measured_writes_nothing(tmp_path: Path) -> None:
    """"No measurement" and "zero" must not read the same."""
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    aot_mint.write_phase_snapshot(snapshot, aot_mint.MintProgress())
    assert not snapshot.exists()
    assert aot_mint.partial_phase_table(aot_mint.MintProgress()) == {}


def test_an_abandoned_outcome_emits_the_rows_it_measured(
    tmp_path: Path,
) -> None:
    """THE REGRESSION, over the real parent-side relay.

    A child that wrote no report at all — which is every abandoned and every
    killed mint — must still put its entry rows and its pool row on the wire.
    """
    snapshot = tmp_path / mint_process.PHASES_SNAPSHOT_NAME
    progress = _progress_midflight(tmp_path)
    aot_mint.write_phase_snapshot(snapshot, progress)

    request = mint_process.MintRequest(
        function="f", modules=(), family="sdxl", cell_key="k",
        target=str(tmp_path / "cell.tar.gz"), capture=str(tmp_path),
        report=str(tmp_path / "report.json"),
        cfg=mint_process.CompileCellSpec(),
        phases_snapshot=str(snapshot))
    recovered = mint_process._read_phase_snapshot(request.phases_snapshot)
    assert recovered, "the parent could not read what the child wrote"

    outcome = mint_process.MintOutcome(
        status=mint_process.ABANDONED,
        detail="background mint abandoned (shutdown: worker shutdown)",
        report=None, elapsed_s=1741.33, partial_phases=recovered)

    emitted: list[Dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        emitted.append(kwargs)

    original = aot_mint.emit_phase_events
    aot_mint.emit_phase_events = _capture  # type: ignore[assignment]
    try:
        mint_delegate._emit_aot_phases(outcome, family="sdxl", lane="w8a8")
    finally:
        aot_mint.emit_phase_events = original  # type: ignore[assignment]

    assert emitted, (
        "an abandoned mint emitted NO phase table — this is attempt sixteen, "
        "29 minutes reported as one row")
    table = emitted[0]["table"]
    assert table["pool"]["entry_workers"] == progress.width.workers, (
        "the K-and-binding answer is the one the coordinator had to re-buy "
        "with another pod")
    assert table["pool"]["binding"] == progress.width.binding
    assert emitted[0]["terminus"] == "abandoned", (
        "an abandoned mint must not be relabelled as an ordinary abort — the "
        "cause is a co-tenancy decision, not a mint failure")
    assert table["recovered_from"] == "phase_snapshot", (
        "a recovered table must say it was recovered; a reader must never "
        "mistake it for one the child wrote at its own terminus")


def test_a_report_beats_a_snapshot_when_both_exist(tmp_path: Path) -> None:
    """The child reaching its own terminus is better evidence than the last
    beat before it got there. The snapshot is a fallback, never an override."""
    outcome = mint_process.MintOutcome(
        status=mint_process.REFUSED, elapsed_s=10.0,
        report=mint_process.MintReport(
            status="refused", elapsed_s=10.0,
            mint_phases={"v": 1, "terminus": "aborted",
                         "pool": {"entry_workers": 7}}),
        partial_phases={"v": 1, "pool": {"entry_workers": 99}})

    emitted: list[Dict[str, Any]] = []
    original = aot_mint.emit_phase_events
    aot_mint.emit_phase_events = (
        lambda **kw: emitted.append(kw))  # type: ignore[assignment]
    try:
        mint_delegate._emit_aot_phases(outcome, family="sdxl", lane="w8a8")
    finally:
        aot_mint.emit_phase_events = original  # type: ignore[assignment]

    assert emitted[0]["table"]["pool"]["entry_workers"] == 7
    assert "recovered_from" not in emitted[0]["table"]


def test_the_snapshot_path_reaches_the_child(tmp_path: Path) -> None:
    """The wiring. Every measurement above is worthless if the child is never
    told where to write."""
    import inspect

    assert "phases_snapshot=str(" in inspect.getsource(
        mint_delegate.build_request)
    from gen_worker import mint_child

    assert "phase_snapshot=(" in inspect.getsource(mint_child._mint_aot)


def test_an_unreadable_snapshot_never_changes_an_outcome(
    tmp_path: Path,
) -> None:
    """Telemetry must not be able to fail a mint, in either direction."""
    assert mint_process._read_phase_snapshot("") == {}
    assert mint_process._read_phase_snapshot(str(tmp_path / "nope")) == {}
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    assert mint_process._read_phase_snapshot(str(junk)) == {}
    listy = tmp_path / "listy.json"
    listy.write_text("[1, 2, 3]")
    assert mint_process._read_phase_snapshot(str(listy)) == {}


def test_the_retry_decision_is_untouched_by_recovered_telemetry() -> None:
    """`retryable` branches on ``report is None``. Carrying the recovered
    table in a SEPARATE field rather than a synthesized report is what keeps
    a telemetry fix from silently changing a retry policy."""
    crashed = mint_process.MintOutcome(
        status=mint_process.CRASHED, partial_phases={"v": 1})
    assert crashed.report is None
    assert crashed.retryable is True
    abandoned = mint_process.MintOutcome(
        status=mint_process.ABANDONED, partial_phases={"v": 1})
    assert abandoned.retryable is False, (
        "abandonment is not a failure and must never be retried into a "
        "second billed compile")


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_pool_ledger_is_live_not_end_of_run() -> None:
    """A ledger written only at the end is a ledger an abandoned mint never
    gets. `_compile_entries_parallel` must refresh it per completed entry."""
    import inspect

    source = inspect.getsource(aot_mint._compile_entries_parallel)
    assert "progress.pool_ledger = _pool_facts(pool)" in source
    assert "on_entry=_tick" in source, (
        "the refresh must be on the per-entry callback, not after the pool "
        "returns — a mint killed at entry 30 of 36 never reaches after")


# ---------------------------------------------------------------------------
# pgw#848 long-fuse sweep: the pod-side reaper's progress signal had no producer
# ---------------------------------------------------------------------------


def test_the_mint_feeds_the_pod_side_reapers_progress_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """podguard's own docstring: both its layers "kill on liveness +
    progress-staleness" — Paul's rule, implemented. The pod-side layer reads a
    token file that `podguard-progress` writes, and **nothing in the SDK has
    ever written it** (zero references to podguard in gen_worker).

    So the pod-side progress path had no producer, and the only thing keeping
    a minting pod alive was podguard's renewal thread — the thread that was
    never started. Two independent failures were required to reap attempt
    sixteen and only one was visible.
    """
    state = tmp_path / "podguard"
    monkeypatch.setenv(aot_mint.PODGUARD_STATE_ENV, str(state))

    aot_mint._touch_pod_progress("aot_mint inductor_compile 3/36 unet/x")
    token_a = (state / "progress").read_text()
    aot_mint._touch_pod_progress("aot_mint inductor_compile 4/36 unet/y")
    token_b = (state / "progress").read_text()

    assert "3/36" in token_a and "4/36" in token_b
    assert token_a != token_b, (
        "the watchdog compares the token's CONTENT, so a value that does not "
        "change reads as NO progress however often the file is rewritten")


def test_the_progress_signal_is_inert_off_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset everywhere but a podguard-rented pod, and a mint must never fail
    because a telemetry file could not be written."""
    monkeypatch.delenv(aot_mint.PODGUARD_STATE_ENV, raising=False)
    aot_mint._touch_pod_progress("nothing should happen")
    assert not list(tmp_path.iterdir())

    # An unwritable state dir is survivable, not fatal.
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    monkeypatch.setenv(aot_mint.PODGUARD_STATE_ENV, str(blocked))
    aot_mint._touch_pod_progress("still must not raise")


def test_every_mint_beat_feeds_both_survivors(tmp_path: Path) -> None:
    """The two things that must outlive a killed mint are fed by the SAME
    beat: the phase snapshot (what it measured) and the pod-side progress
    token (that it was working). Neither may depend on the other running."""
    import inspect

    source = inspect.getsource(aot_mint.mint)
    assert "write_phase_snapshot(snap, progress)" in source
    assert "_touch_pod_progress(" in source
