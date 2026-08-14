"""pgw#1189 — an ABANDONED mint's per-graph-class partition was dropped, so the
one question the compile loop kept re-opening was structurally unanswerable.

pgw#830 measures the child's whole wall (`child_seal_s`, `child_setup_s`,
`child_interp_s`, `child_boot_s`, `reap_lag_s`, `parent_stage_s`…) and pgw#832's
seal split rides the overlays. Both used to reach the phase table through
``aot_mint._fold_pool_results`` — which ran ONLY after the pool RETURNED. The
``finally`` above it refreshed only ``progress.pool_ledger``.

So a mint killed mid-pool kept its LEDGER and lost its per-class SPANS. Verified
against the standing hub, 2026-08-12: all three recorded gen-worker 0.112.0
mints carry the six inductor leaf timers and `overlays={}` — not one child span
— and the two sdxl mints were abandoned at 1/36 and 16/36. **The 16/36 one is
the run th#1834's P0-E read its figures from**, which is why "what is the ~39 s
residual" was answered by inference: the split was never on the wire.

pgw#848 had already written the rule this violates, for the ledger, one line
away: *"A mint killed at entry 30 of 36 then leaves 30 entries' worth of
measurement on disk instead of one bare 'no cell produced' row."*

**pgw#1215 moved WHERE the property lives, and it is stronger there.** There is
no fold any more, because there is nothing to fold onto: a compile child traces,
compiles and packs its own share, so it MEASURES per graph class and the parent
records what it reported the moment the share is collected — inside
``EntryCompilePool._collect``, before any gate that can raise and long before
``compile()`` returns. The rows below pin that, by making the run FAIL after
every share has landed and asserting the measurements survived it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from torch_compiled_graphs import spans

from gen_worker import aot_compile_pool as pool
from harness import fake_compile_child

_DECLARED = 6


def _pool(tmp_path: Path, workers: int = 2) -> pool.EntryCompilePool:
    width = pool.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == workers
    return pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path))


def _template(tmp_path: Path) -> pool.EntryJob:
    return pool.EntryJob(
        function="generate", modules=("harness.toy_endpoints",),
        out_dir=str(tmp_path / "artifacts"))


def test_a_run_that_dies_after_collection_keeps_every_landed_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#1189: the per-class numbers were assembled only on the
    RETURN path, so any terminus that was not a clean return threw them away.

    The failure is injected AFTER both shares have been collected — the class
    set does not cover the declaration, which ``_assert_shares_whole`` raises
    on once every child has reported — so this asserts durability and not
    merely "the happy path records something".
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "collide")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path)

    with pytest.raises(pool.EntryCompileFailed):
        box.compile(_template(tmp_path))

    assert set(box.entry_seconds) == {"share-000", "share-001"}, (
        "a run that raised kept none of the seconds it really spent")
    for share in ("share-000", "share-001"):
        table = box.entry_phases[share]
        assert not spans.check(table), table
        # The child spans, not just the inductor leaves. This is the exact
        # half that was missing from every recorded fleet mint.
        for member in spans.PARTITIONS["child_wall_s"]:
            assert member in table, (
                f"{share}: {member} did not survive the abandoned run — "
                f"pgw#1189's whole finding is that the child half never "
                f"reached a reader")
        assert box.entry_overlays[share].get("seal_libhash_s") is not None


def test_the_measurement_granularity_is_the_GRAPH_CLASS(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One share is several classes, so a per-share number answers nothing.

    th#1834's P0-E read a ~39 s residual off per-entry rows and attributed it
    by inference; the fix for that class of reading is a number at the
    granularity of the thing being asked about. ``class_spans`` is that number
    and it is recorded per class, from the child that traced it.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "ok")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path)

    out = box.compile(_template(tmp_path))

    assert set(box.class_spans) == set(out) == {
        f"cls/dim={i}" for i in range(_DECLARED)}
    for name, row in box.class_spans.items():
        assert row["export_s"] > 0.0 and row["compile_s"] > 0.0, (name, row)
    # ...and the same rows ride the packed result, so a reader who has the
    # artifact has the measurement.
    for name, packed in out.items():
        assert packed.spans == box.class_spans[name]


def test_an_empty_share_records_its_seconds_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A share that packed nothing still spent real seconds, and an absent row
    is how "this child did nothing" and "nobody measured this child" become
    indistinguishable."""
    monkeypatch.setenv("PGW_FAKE_CHILD", "short")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path)

    with pytest.raises(pool.EntryCompileFailed):
        box.compile(_template(tmp_path))

    assert "share-000" in box.entry_seconds
    assert "share-000" in box.entry_phases
    assert box.entry_declared["share-000"] == _DECLARED
