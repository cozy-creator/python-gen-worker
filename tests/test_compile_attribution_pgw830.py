"""pgw#830: every second of a pooled compile has a name, and stays named.

Attempt nine recorded ``compile_s=1331.72`` with five named phases summing to
742.6 s. The other **44 % had no name** — not because a metric was missing,
but because ``compile_s`` measured a CHILD PROCESS while ``phases`` measured
the inside of ``aot_compile``, and nothing ever reconciled the two.

The regression this file exists to prevent is attribution ROT: a phase that
stops being measured, a span that stops being recorded, or a new step added
to the child that nobody names. All three look identical from the outside —
the table still renders, the numbers still add up to something, and the gap
grows silently. So the assertion is not "these labels exist"; it is the
INVARIANT that each partition equals the sum of its members, run against a
real pool driving real children.

⚠️ **What this file can and cannot drive since pgw#1215.** A compile child can
no longer be handed an already-exported program: it builds its own weight-free
pipeline from a declaration and compiles what it traced, so the cheap
four-linear graph that used to make a real pool run cost seconds does not
exist as an input any more. The children below are therefore driven REAL and
REFUSING — their job names a module this tree does not have — which exercises
the whole parent/child span boundary (spawn, boot, seal, torch import, device
lock, report, reap) and closes the partition on it. The three spans only a
compiled share pays (``child_trace_s`` / ``compile_wall_s`` / ``child_pack_s``)
are covered here by driving the child's own ledger sealer directly, and
end-to-end by pgw#1215 step 3's POD leg, which is where a real compile is
allowed to run at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import aot_compile_child as child
from gen_worker import aot_compile_pool as pool
from torch_compiled_graphs import spans

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _pool(tmp_path: Path, *, shares: int) -> pool.EntryCompilePool:
    width = pool.entry_workers(
        shares, limit=shares, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == shares
    return pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"))


def _refusing(tmp_path: Path) -> pool.EntryJob:
    """A recipe every child will refuse, AFTER paying its whole boot.

    The refusal happens in ``build_pipeline``, which runs after the seal, the
    torch import and the device-lock install — so every span this file is
    about is recorded and complete by the time the child writes its report.
    """
    return pool.EntryJob(
        function="nope", modules=("gen_worker_no_such_endpoint_module",),
        out_dir=str(tmp_path / "artifacts"))


def _reports(tmp_path: Path) -> list:
    import msgspec

    out = []
    for path in sorted((tmp_path / "pool").glob("share-*/report.json")):
        out.append(msgspec.json.decode(
            path.read_bytes(), type=pool.EntryReport))
    return out


# ---------------------------------------------------------------------------
# The invariant, on a real pool
# ---------------------------------------------------------------------------


def test_every_second_of_a_real_childs_wall_is_attributed(
    tmp_path: Path,
) -> None:
    """The child's wall partitions itself, on a real pool run.

    If someone adds a step to the compile child and does not name it,
    ``child_other_s`` grows and the residual assertion fires; if someone
    removes a span, the partition stops summing and
    :func:`torch_compiled_graphs.spans.check` fires.
    """
    box = _pool(tmp_path, shares=2)
    with pytest.raises(pool.EntryCompileFailed):
        box.compile(_refusing(tmp_path))

    reports = _reports(tmp_path)
    assert reports, "no child wrote a report — nothing was measured"
    for report in reports:
        table = dict(report.spans)
        violations = spans.check(table)
        assert not violations, (
            f"{report.entry}: the child's wall no longer partitions itself — "
            f"{violations}. Every interval between the recorded marks must "
            f"have a name (pgw#830); a table that does not add up is how "
            f"44 % went dark on attempt nine.\ntable={table}")

        # Named, not merely present: the spans that were literally invisible
        # before this issue. Each is real work paid once PER SHARE because the
        # pool's unit of parallelism is a process that exits.
        for label in ("child_seal_s", "child_setup_s"):
            assert table.get(label, 0.0) > 0.0, (
                f"{report.entry}: {label} is not being measured. It is inside "
                f"the recorded compile_s whether or not anyone names it")

        # The single largest named non-compile span on attempt nine's shape,
        # and the reason this issue found something: `env_seal.establish()`
        # re-hashes every toolchain .so in the image, per child. It must stay
        # broken out — folded back into an anonymous `child_seal_s` it would
        # be indistinguishable from unavoidable process setup.
        seal = dict(report.overlays)
        assert "seal_libhash_s" in seal and "seal_config_s" in seal, (
            f"{report.entry}: the seal's own split is gone ({seal}) — "
            f"`child_seal_s` on its own cannot tell a 4 GB SHA-256 pass "
            f"apart from an unavoidable setup cost")
        assert sum(
            v for k, v in seal.items()
            if k.startswith("seal_") and k != "seal_libhash_s"
        ) == pytest.approx(table["child_seal_s"], abs=0.25), (
            f"{report.entry}: the seal split no longer covers child_seal_s "
            f"({seal} vs {table['child_seal_s']})")

    # And the PARENT's outer partition, which spans the process boundary and
    # therefore could never be closed inside the child.
    for share, table in box.entry_phases.items():
        assert not spans.check(table), table
        assert table["compile_s"] == pytest.approx(
            box.entry_seconds[share], abs=0.05)
        assert table.get("child_boot_s", 0.0) > 0.0


def test_the_compile_halfs_spans_are_recorded_even_when_they_are_zero(
) -> None:
    """``child_trace_s`` / ``compile_wall_s`` / ``child_pack_s`` are PARTITION
    MEMBERS, so a share that never reached them must still record them.

    pgw#830's own failure mode, one level down: a member the ledger simply
    never touched is not "0" to :func:`check` — it is MISSING, and the
    residual silently absorbs it. This drives the child's real ledger sealer,
    which is the code that seeds them; a compile is not needed to prove that
    property and is not permitted on this box to prove any other.
    """
    ledger = spans.SpanLedger()
    with ledger.span("child_seal_s"):
        pass
    fields = child._span_fields(ledger, {}, {}, {"seal_config_s": 0.0})
    table = dict(fields["spans"])
    for member in spans.PARTITIONS["child_wall_s"]:
        assert member in table, (
            f"{member} is a declared member of child_wall_s and the child did "
            f"not record it — check() reads that as 'the residual silently "
            f"absorbed it', which is exactly the defect pgw#830 closed")
    assert not spans.check(table), table
    # And the round trip a report takes, so a member that decodes to a default
    # cannot look recorded.
    assert set(spans.PARTITIONS["child_wall_s"]) <= set(table)


# ---------------------------------------------------------------------------
# Pool idle is a SEPARATE line item from serial dark time
# ---------------------------------------------------------------------------


def test_pool_idle_is_accounted_separately_from_child_seconds(
    tmp_path: Path,
) -> None:
    """``capacity = busy + idle``, and idle splits into named causes.

    These two numbers have OPPOSITE fixes. Dark time inside ``compile_s`` is
    work on the critical path — the target of instrumentation and then
    optimization. Pool idle is workers with nothing to run; it shrinks when
    the class count or the straggler spread changes (pgw#829) and not at all
    when the compiler gets faster. A table that adds them into one "dark"
    figure aims the next lane at the wrong term.
    """
    box = _pool(tmp_path, shares=2)
    with pytest.raises(pool.EntryCompileFailed):
        box.compile(_refusing(tmp_path))

    led = box.ledger
    assert led.workers == 2 and led.entries == 2
    assert led.capacity_s == pytest.approx(led.wall_s * led.workers, abs=0.01)
    # The identity that makes "75 % pool efficiency" a measured statement
    # rather than a ratio someone divided out afterwards. Stated as `<=` on
    # THIS path and only this one: the pool tears its siblings down group-wide
    # the moment one share fails, so a torn-down child's seconds are never
    # charged to `busy_s` — a run that claimed the identity here would be
    # claiming a measurement of a process it killed. The equality holds on a
    # run where every share completes, which needs a real compile (pgw#1215
    # step 3's pod leg).
    assert led.busy_s + led.idle_s <= led.capacity_s * 1.05, (
        f"pool capacity {led.capacity_s:.2f}s < busy {led.busy_s:.2f}s + "
        f"idle {led.idle_s:.2f}s — the idle split is charging seconds nobody "
        f"spent")
    assert led.busy_s > 0.0

    facts = led.facts()
    assert facts["pool_idle_s"] == pytest.approx(
        facts["idle_staging_s"] + facts["idle_drain_s"]
        + facts["idle_spawn_s"] + facts["idle_other_s"], abs=0.01)
    # Spawning is parent-serial and really does hold a freed slot.
    assert facts["spawn_total_s"] > 0.0


# ---------------------------------------------------------------------------
# The checker itself
# ---------------------------------------------------------------------------


def test_check_names_the_partition_that_broke() -> None:
    """A silent gap must be impossible: the checker says WHICH total stopped
    adding up and by how much, because "attribution is off" with no name is
    the same defect this issue is closing."""
    good = {
        "compile_s": 10.0, "child_boot_s": 1.0, "child_wall_s": 8.5,
        # pgw#1099: the outer partition has its own residual now, so a
        # measured span never doubles as the catch-all. Zero here because the
        # named members already close the level — which is the normal case.
        "reap_lag_s": 0.5, "parent_other_s": 0.0,
        "child_seal_s": 0.1, "child_torch_import_s": 1.5,
        "child_devlock_s": 0.0, "child_setup_s": 0.4,
        "child_trace_s": 0.0, "child_pack_s": 0.0,
        "compile_wall_s": 6.0, "child_other_s": 0.5,
        "lowering_s": 0.5, "codegen_s": 1.0, "graph_passes_s": 0.3,
        "host_compile_s": 3.7, "compile_other_s": 0.5,
    }
    assert spans.check(good) == []
    assert spans.dark_fraction(good) == pytest.approx(0.10)

    broken = dict(good, host_compile_s=1.0)
    problems = spans.check(broken)
    assert problems and "compile_wall_s" in problems[0]
    assert "-2.700" in problems[0] or "+" in problems[0] or "-2.7" in problems[0]

    # A dropped span must not be absorbed in silence.
    dropped = {k: v for k, v in good.items() if k != "child_torch_import_s"}
    problems = spans.check(dropped)
    assert any("child_torch_import_s" in p for p in problems), problems

    # An overlay summed as a member drives the residual negative, and that is
    # reported as double-counting rather than as a small anomaly.
    doubled = dict(good, compile_other_s=-1.0)
    assert any("double-counted" in p for p in spans.check(doubled))
