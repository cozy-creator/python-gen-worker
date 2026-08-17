"""pgw#1355: a cold boot states its own decomposition, once, on the channel
everything else is read from.

Integration-style: the ladder tests drive the REAL
:mod:`gen_worker.boot_phases` recorder — opening real spans and marking real
milestones — and then assert the stage table `collect()` folds out of it, so
nothing here passes against a boot shape production cannot produce. The
emission tests drain the REAL activity sink the worker transport installs, so
the assertions read exactly the ActivityUpdates a hub would receive.

The load-bearing property, and the one every red proof is aimed at: **the
total is a UNION and never a sum.** A boot whose stages overlap must not report
more time than it took.
"""

from __future__ import annotations

import time
from typing import Iterator, List

import pytest

from gen_worker import activity as activity_mod
from gen_worker import boot_phases
from gen_worker import boot_stages
from gen_worker.boot_stages import (
    BootStageTable,
    Stage,
    StageSpan,
    UnknownStageError,
)
from gen_worker.pb import worker_scheduler_pb2 as pb


@pytest.fixture(autouse=True)
def _clean_recorders() -> Iterator[None]:
    boot_phases.reset_for_tests()
    boot_stages.reset_for_tests()
    activity_mod.reset_for_tests()
    yield
    boot_phases.reset_for_tests()
    boot_stages.reset_for_tests()
    activity_mod.reset_for_tests()


class _Events:
    """The real sink the transport binds, drained after the fact."""

    def __init__(self) -> None:
        self.updates: List[pb.ActivityUpdate] = []

    def install(self) -> None:
        activity_mod._sink = self.updates.append  # the bound-sink contract

    def of_kind(self, kind: str) -> List[pb.ActivityUpdate]:
        return [u for u in self.updates if u.kind == kind]


def _table(*spans: StageSpan, wall_ms: int, servable_ms: int = 0) -> BootStageTable:
    return BootStageTable(
        spans=tuple(spans), wall_ms=wall_ms, servable_ms=servable_ms)


# ---------------------------------------------------------------------------
# The closed vocabulary
# ---------------------------------------------------------------------------


def test_every_boot_phase_has_a_stage_or_a_documented_exemption() -> None:
    """A boot phase with no home here silently drops its seconds out of the
    table. Adding one to `boot_phases` without deciding where it belongs is
    red HERE rather than invisible in production."""
    assert boot_stages.unmapped_phases() == ()


def test_an_unknown_stage_refuses_at_the_call_site() -> None:
    """The vocabulary is closed because a renderer in another repository binds
    to these tokens. A stage invented at a call site is a column that appears
    in production and in no reader."""
    with pytest.raises(UnknownStageError):
        StageSpan(stage="teleportation", t0_ms=0, t1_ms=1)  # type: ignore[arg-type]
    with pytest.raises(UnknownStageError):
        boot_stages.record("teleportation", t0_ms=0, t1_ms=1)  # type: ignore[arg-type]


def test_a_reader_refuses_a_fleet_it_does_not_understand() -> None:
    """A packed run naming an unknown stage RAISES rather than being skipped.

    Skipping would under-report a boot — quietly, and in the direction that
    looks healthy. Refusing says "this reader does not understand the fleet
    that emitted this", which is the sentence an operator can act on."""
    with pytest.raises(UnknownStageError):
        boot_stages.parse_runs("process_boot:0-10,warp_core:10-20")


def test_a_span_that_ends_before_it_starts_refuses() -> None:
    with pytest.raises(ValueError):
        StageSpan(stage=Stage.MODEL_LOAD, t0_ms=900, t1_ms=100)


# ---------------------------------------------------------------------------
# THE red proof: overlap is honest, and stages never sum past wall
# ---------------------------------------------------------------------------


def test_concurrent_stages_do_not_sum_past_wall() -> None:
    """A snapshot pull overlapping a model load must not report more time than
    the boot took.

    This is the defect `boot_phases`' own header records paying for: a summing
    reconciliation once "explained" 3,338 ms of a 909 ms fetch. A sum that
    exceeds wall is not a rounding problem — it is a report that has stopped
    describing time.

    RED PROOF: make `critical_path_ms` return `span_sum_ms` and this fails on
    both assertions below.
    """
    table = _table(
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=1_000, t1_ms=5_000),
        StageSpan(stage=Stage.MODEL_LOAD, t0_ms=3_000, t1_ms=7_000),
        wall_ms=8_000,
    )
    # The two spans are 4 s each and share 2 s of wall.
    assert table.span_sum_ms == 8_000
    assert table.critical_path_ms == 6_000, "1000..7000 covered, overlap once"
    assert table.critical_path_ms < table.span_sum_ms, (
        "the sum must exceed the union when anything overlapped — a table "
        "where they are equal has silently serialized concurrent work")
    assert table.critical_path_ms <= table.wall_ms, (
        "no arrangement of stages may account for more wall than the boot had")
    assert table.overlap_ms == 2_000
    assert table.unmeasured_ms == 2_000


def test_overlap_is_reported_and_not_smeared_away() -> None:
    """Two stages that ran ENTIRELY concurrently cost one stage's wall.

    Shortening a stage that ran inside another stage's window buys nothing,
    and only a union can say so — which is the whole reason a cold-start
    budget is spent against `critical_path_ms` and not against a sum."""
    table = _table(
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=0, t1_ms=10_000),
        StageSpan(stage=Stage.MODEL_LOAD, t0_ms=2_000, t1_ms=8_000),
        wall_ms=10_000,
    )
    assert table.critical_path_ms == 10_000
    assert table.overlap_ms == 6_000
    assert table.unmeasured_ms == 0
    assert table.accounted_pct == 100


def test_sequential_stages_have_no_overlap() -> None:
    """The control: the arithmetic must not manufacture concurrency either."""
    table = _table(
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=0, t1_ms=3_000),
        StageSpan(stage=Stage.MODEL_LOAD, t0_ms=3_000, t1_ms=5_000),
        wall_ms=5_000,
    )
    assert table.overlap_ms == 0
    assert table.critical_path_ms == table.span_sum_ms == 5_000


def test_a_stage_that_ran_twice_keeps_its_gap() -> None:
    """Merged per stage, not globally: a stage that ran twice with a gap is a
    different fact from one that ran once across both."""
    table = _table(
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=0, t1_ms=1_000),
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=4_000, t1_ms=5_000),
        wall_ms=5_000,
    )
    runs = table.runs()
    assert runs == [
        (Stage.SNAPSHOT_PULL, 0, 1_000),
        (Stage.SNAPSHOT_PULL, 4_000, 5_000),
    ]
    assert table.busy_ms(Stage.SNAPSHOT_PULL) == 2_000
    assert table.unmeasured_ms == 3_000, "the gap is a HOLE, and it is named"


def test_unmeasured_is_named_never_smeared_across_the_stages() -> None:
    """"unmeasured" and "zero" are different answers, and the hole is the hint
    about where the next instrument belongs."""
    table = _table(
        StageSpan(stage=Stage.MODEL_LOAD, t0_ms=0, t1_ms=1_000),
        wall_ms=100_000,
    )
    assert table.unmeasured_ms == 99_000
    assert table.busy_ms(Stage.MODEL_LOAD) == 1_000, (
        "the measured stage keeps its own honest 1 s — the hole is NOT "
        "distributed into it")
    assert table.accounted_pct == 1


# ---------------------------------------------------------------------------
# The wire codec
# ---------------------------------------------------------------------------


def test_the_packed_table_round_trips() -> None:
    """The renderer in another repo parses this token. The round trip is the
    contract, asserted here rather than assumed."""
    table = _table(
        StageSpan(stage=Stage.PROCESS_BOOT, t0_ms=0, t1_ms=800),
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=1_000, t1_ms=5_000),
        StageSpan(stage=Stage.MODEL_LOAD, t0_ms=3_000, t1_ms=7_000),
        wall_ms=8_000,
    )
    packed, truncated = boot_stages.pack_runs(table)
    assert not truncated
    assert boot_stages.parse_runs(packed) == (
        (Stage.PROCESS_BOOT, 0, 800),
        (Stage.SNAPSHOT_PULL, 1_000, 5_000),
        (Stage.MODEL_LOAD, 3_000, 7_000),
    )


def test_the_detail_grammar_is_the_one_the_renderer_already_parses() -> None:
    """Space-separated `k=v`, values never containing whitespace — the grammar
    `(\\w+)=(\\S+)` parses entirely, which e2e's `detailKV` already implements.
    A value with a space splits one pair into two and means something else."""
    table = _table(
        StageSpan(stage=Stage.KEYSET, t0_ms=100, t1_ms=805_000,
                  label="boot_adopt.key_set",
                  attrs={"keys_from": "traced", "classes": "36"}),
        wall_ms=830_000,
    )
    detail = boot_stages.rollup_detail(table)
    pairs = dict(tok.split("=", 1) for tok in detail.split(" ") if "=" in tok)
    assert pairs["v"] == "1"
    assert pairs["wall_ms"] == "830000"
    assert pairs["keys_from"] == "traced"
    assert pairs["unmeasured_ms"] == str(830_000 - 804_900)
    for token in detail.split(" "):
        assert token.count("=") >= 1, f"{token!r} is not a k=v pair"

    stage_detail = boot_stages.stage_detail(table.spans[0])
    kv = dict(tok.split("=", 1) for tok in stage_detail.split(" "))
    assert kv["stage"] == "keyset"
    assert kv["t0_ms"] == "100"
    assert kv["t1_ms"] == "805000"
    assert kv["keys_from"] == "traced"


def test_an_empty_value_can_never_reach_the_wire() -> None:
    """An empty value ends the token at the `=` and silently merges the next
    pair into it. Every attribute goes through the same guard."""
    boot_stages.record(
        Stage.KEYSET, t0_ms=0, t1_ms=10, keys_from="", family="sd xl")
    span = boot_stages.recorded()[0]
    assert "keys_from" not in span.attrs, "an empty attr is DROPPED, not blanked"
    assert span.attrs["family"] == "sdxl", "whitespace is stripped, never shipped"


def test_a_truncated_pack_says_so() -> None:
    """A table that dropped runs is a table whose union no longer closes, and a
    reader must be able to tell that from a boot that had fewer stages."""
    spans = [
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=i * 100, t1_ms=i * 100 + 10)
        for i in range(boot_stages.MAX_PACKED_RUNS + 5)
    ]
    table = _table(*spans, wall_ms=10_000)
    packed, truncated = boot_stages.pack_runs(table)
    assert truncated
    assert len(boot_stages.parse_runs(packed)) == boot_stages.MAX_PACKED_RUNS
    assert "runs_truncated=1" in boot_stages.rollup_detail(table)


# ---------------------------------------------------------------------------
# Folding a REAL boot_phases ladder
# ---------------------------------------------------------------------------


def _drive_a_real_boot() -> None:
    """Drive the real recorder through the shape a cold pod boots in.

    Concurrency is real here — the two component fetches are open at the same
    time — so the folded table has to survive genuine overlap rather than a
    fixture's arithmetic.
    """
    boot_phases.mark_once(boot_phases.PHASE_SDK_READY, detail="endpoints=1")
    boot_phases.mark_once(boot_phases.PHASE_HELLO, since_process_start=True)
    weights = boot_phases.open_span(boot_phases.PHASE_WEIGHTS_FETCH, ref="r")
    with boot_phases.parent_scope(weights.ordinal):
        unet = boot_phases.open_span(
            boot_phases.PHASE_COMPONENT_FETCH, function="unet")
        vae = boot_phases.open_span(
            boot_phases.PHASE_COMPONENT_FETCH, function="vae")
        time.sleep(0.02)
        vae.bytes_moved(4096, boot_phases.SOURCE_R2)
        vae.close()
        unet.bytes_moved(8192, boot_phases.SOURCE_R2)
        unet.close()
    weights.close()
    with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD, function="generate"):
        time.sleep(0.01)
    boot_phases.mark_once(boot_phases.PHASE_EAGER_READY, function="generate")
    boot_phases.mark_once(
        boot_phases.PHASE_FIRST_REQUEST_SERVABLE, since_process_start=True)


def test_a_real_ladder_folds_into_the_closed_vocabulary() -> None:
    _drive_a_real_boot()
    table = boot_stages.collect()

    seen = {span.stage for span in table.spans}
    assert Stage.PROCESS_BOOT in seen
    assert Stage.IMPORTS in seen
    assert Stage.SNAPSHOT_PULL in seen, "weights_fetch + component_fetch fold here"
    assert Stage.MODEL_LOAD in seen, "pipeline_load folds here"
    assert Stage.READY in seen
    assert seen <= set(Stage), "nothing outside the closed vocabulary"

    assert table.wall_ms > 0, "eager_ready is the wall"
    assert table.critical_path_ms <= table.wall_ms, (
        "a real ladder with two concurrent component fetches must still not "
        "account for more wall than the boot had")


def test_the_two_concurrent_component_fetches_are_visible_as_overlap() -> None:
    """Four components inside one pull that each measure 180 s were OVERLAPPED,
    and that is a different finding from four sequential 50 s ones. Only an
    interval can tell them apart, which is why spans carry offsets."""
    _drive_a_real_boot()
    table = boot_stages.collect()
    pull_spans = [s for s in table.spans if s.stage is Stage.SNAPSHOT_PULL]
    assert len(pull_spans) >= 3, "the parent fetch plus its two components"
    assert table.overlap_ms > 0, (
        "the components ran inside the parent span and beside each other — a "
        "table reporting zero overlap has serialized work that was concurrent")
    assert table.busy_ms(Stage.SNAPSHOT_PULL) < sum(
        s.duration_ms for s in pull_spans), (
        "the stage's own wall is the union of its spans, not their sum")


def test_a_boot_that_never_reached_ready_still_reports_what_happened() -> None:
    """An operator looking at a STUCK pod wants exactly this table. The wall
    falls back to the furthest span end, so the reconciliation describes what
    has happened rather than dividing by zero."""
    boot_phases.mark_once(boot_phases.PHASE_SDK_READY)
    with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD, function="generate"):
        time.sleep(0.01)
    table = boot_stages.collect()
    assert table.wall_ms > 0
    assert Stage.MODEL_LOAD in {s.stage for s in table.spans}


def test_the_keyset_stage_carries_where_the_keys_CAME_FROM() -> None:
    """pgw#1353's whole finding is that every production pod reported
    `keys_from=traced` — the FALLBACK — and nothing made that visible while it
    was happening. It is the axis separating an 805 s boot from a 5 ms one."""
    _drive_a_real_boot()
    boot_stages.record(
        Stage.KEYSET, t0_ms=1, t1_ms=804_701, label="boot_adopt.key_set",
        keys_from="traced", classes=36, family="sdxl")
    table = boot_stages.collect()
    assert table.attr("keys_from") == "traced"
    assert "keys_from=traced" in boot_stages.rollup_detail(table)


def test_the_derive_stage_covers_a_window_the_ladder_structurally_cannot() -> None:
    """pgw#1353: the derive runs during a REQUEST, after the boot window
    closed, so `boot_phases.in_boot()` is already False and no ladder row can
    cover it. `record_ending_now` is the minimal instrument that joins it."""
    _drive_a_real_boot()
    assert not boot_phases.in_boot(), "the boot window is closed"
    boot_stages.record_ending_now(
        Stage.KEYSET, duration_ms=10, label="boot_adopt.key_set",
        keys_from="traced")
    table = boot_stages.collect()
    keyset = [s for s in table.spans if s.stage is Stage.KEYSET]
    assert len(keyset) == 1
    assert keyset[0].duration_ms == 10
    assert keyset[0].t0_ms >= 0
    assert "clamped_ms" not in keyset[0].attrs


def test_a_span_longer_than_the_process_CONFESSES_instead_of_shrinking() -> None:
    """A duration longer than the process has existed came from a different
    clock. The span is clamped — a negative offset is not representable — but
    the clamp is stated, because a silently shortened span is a wrong
    measurement that reads as a right one.

    RED PROOF: drop the `clamped_ms` attr and this passes while production
    quietly under-reports its dominant stage.
    """
    uptime = boot_phases.process_uptime_ms()
    boot_stages.record_ending_now(
        Stage.KEYSET, duration_ms=uptime + 900_000, label="impossible")
    span = boot_stages.recorded()[0]
    assert span.t0_ms == 0
    assert int(span.attrs["clamped_ms"]) >= 900_000 - 1_000, (
        "the seconds that could not be represented are NAMED, not dropped")


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def test_emission_is_a_series_with_the_rollup_LAST() -> None:
    """A reader that sees the roll-up knows the series is complete."""
    events = _Events()
    events.install()
    _drive_a_real_boot()
    assert boot_stages.emit() is True

    rows = events.of_kind(boot_stages.KIND)
    assert rows, "the boot reported nothing at all"
    assert rows[-1].phase == boot_stages.PHASE_READY, (
        "the roll-up is emitted last, so a reader that has it has everything")
    assert all(
        r.phase.startswith(boot_stages.PHASE_STAGE_PREFIX) for r in rows[:-1])
    assert sum(
        1 for r in rows if r.phase == boot_stages.PHASE_READY) == 1, (
        "exactly one roll-up, or a reader grouping on it double-counts")


def test_the_rollup_duration_is_wall_to_ready() -> None:
    """`duration_ms` lands in a numeric hub column, so the cold-boot number can
    be grouped and percentiled — which a number interpolated into `detail`
    cannot."""
    events = _Events()
    events.install()
    _drive_a_real_boot()
    boot_stages.emit()
    rollup = [
        r for r in events.of_kind(boot_stages.KIND)
        if r.phase == boot_stages.PHASE_READY][0]
    assert rollup.duration_ms == boot_stages.collect().wall_ms
    assert rollup.duration_ms > 0


def test_emit_is_once_per_process() -> None:
    """The caller sits on a `mark_once` boundary; a double report would put two
    walls for one boot in the table."""
    events = _Events()
    events.install()
    _drive_a_real_boot()
    assert boot_stages.emit() is True
    before = len(events.of_kind(boot_stages.KIND))
    assert boot_stages.emit() is False
    assert len(events.of_kind(boot_stages.KIND)) == before


def test_emission_never_breaks_the_boot_it_measures() -> None:
    """A boot that reached ready is never failed by the report of it."""
    def explode(_update: pb.ActivityUpdate) -> None:
        raise RuntimeError("the hub stream is gone")

    activity_mod._sink = explode
    _drive_a_real_boot()
    boot_stages.emit()  # must not raise


def test_the_emitted_rows_reconstruct_the_table() -> None:
    """The whole point: reading a boot's shape is a ONE-ROW query, and the row
    that answers it must agree with the series beside it."""
    events = _Events()
    events.install()
    _drive_a_real_boot()
    boot_stages.emit()
    rows = events.of_kind(boot_stages.KIND)
    rollup = [r for r in rows if r.phase == boot_stages.PHASE_READY][0]
    kv = dict(
        tok.split("=", 1) for tok in rollup.detail.split(" ") if "=" in tok)

    runs = boot_stages.parse_runs(kv["runs"])
    assert runs, "the packed table is the renderer's whole input"
    union = sum(end - start for _stage, start, end in runs)
    assert union == int(kv["critical_path_ms"]), (
        "the packed runs and the stated union must be the same measurement — "
        "a renderer recomputing from the runs must reach the stated total")
    assert int(kv["critical_path_ms"]) <= int(kv["wall_ms"])
    assert (int(kv["span_sum_ms"]) - int(kv["critical_path_ms"])
            == int(kv["overlap_ms"]))
    assert (int(kv["wall_ms"]) - int(kv["critical_path_ms"])
            == int(kv["unmeasured_ms"]))


def test_the_render_is_pasteable_and_states_its_own_totals() -> None:
    _drive_a_real_boot()
    text = boot_stages.render(boot_stages.collect())
    assert "stage" in text and "busy_ms" in text
    assert "critical_path_ms=" in text
    assert "span_sum_ms=" in text, (
        "the sum is shown BESIDE the union, never instead of it")


def test_the_report_can_never_cost_more_than_the_boot_it_reports_on() -> None:
    """A pathological boot must not become a two-thousand-message burst on the
    worker->hub stream at the exact moment the pod is trying to start serving.

    `boot_phases` buffers up to 2048 rows, so this bound is a real one, not a
    stylistic cap. The ROLL-UP survives truncation — it carries the packed
    table and is what the question is actually asked of — and the drop is
    stated rather than left for a reader to infer from a short series.

    RED PROOF: emit every span unconditionally and the row count blows past the
    bound.
    """
    events = _Events()
    events.install()
    spans = [
        StageSpan(stage=Stage.SNAPSHOT_PULL, t0_ms=i, t1_ms=i + 1,
                  attrs={"component": f"c{i}"})
        for i in range(boot_stages.MAX_STAGE_ROWS + 40)
    ]
    boot_stages.emit(_table(*spans, wall_ms=9_000))

    rows = events.of_kind(boot_stages.KIND)
    stage_rows = [
        r for r in rows if r.phase.startswith(boot_stages.PHASE_STAGE_PREFIX)]
    assert len(stage_rows) == boot_stages.MAX_STAGE_ROWS

    rollup = [r for r in rows if r.phase == boot_stages.PHASE_READY]
    assert len(rollup) == 1, "the roll-up is never the row that gets dropped"
    assert "rows_truncated=40" in rollup[0].detail, (
        "a truncated series must SAY it was truncated — otherwise a stage with "
        "fewer rows than spans reads as a stage that ran fewer times")


def test_truncation_keeps_the_spans_worth_reading() -> None:
    """Longest first. Dropping an arbitrary tail would be as likely to discard
    an 805 s derive as a 2 ms fold, and the whole point of the report is that
    the expensive stage is the one you can see."""
    events = _Events()
    events.install()
    spans = [
        StageSpan(stage=Stage.ARM, t0_ms=i, t1_ms=i + 1)
        for i in range(boot_stages.MAX_STAGE_ROWS + 10)
    ]
    spans.append(StageSpan(
        stage=Stage.KEYSET, t0_ms=0, t1_ms=805_000, label="the whole finding"))
    boot_stages.emit(_table(*spans, wall_ms=830_000))

    kept = [
        r for r in events.of_kind(boot_stages.KIND)
        if r.phase == boot_stages.PHASE_STAGE_PREFIX + Stage.KEYSET.value]
    assert kept, (
        "the 805 s stage was dropped in favour of sixty-odd 1 ms ones — a "
        "truncation rule that can discard the finding is worse than no rows")
    assert kept[0].duration_ms == 805_000


def test_the_emitter_proves_its_own_packing_parses_back() -> None:
    """The renderer that consumes `runs=` lives in another repository, on a
    release cadence this worker knows nothing about.

    If a packing bug ships, the symptom over there is a parse error on an
    otherwise healthy pod — days later, in somebody else's lane, with no way to
    tell a broken emitter from a broken reader. So the emitter checks its own
    output against its own parser before shipping it, and a token that does not
    round-trip is DROPPED and confessed rather than shipped hopefully.

    RED PROOF: skip the round-trip check and a corrupt token ships silently.
    """
    table = _table(
        StageSpan(stage=Stage.KEYSET, t0_ms=0, t1_ms=805_000),
        wall_ms=830_000)
    good = boot_stages.rollup_detail(table)
    assert "runs_unpackable" not in good
    assert "runs=keyset:0-805000" in good

    # A packer that emits a stage token this vocabulary does not admit.
    original = boot_stages.pack_runs
    try:
        boot_stages.pack_runs = lambda _t: ("warp_core:0-1", False)  # type: ignore[assignment]
        degraded = boot_stages.rollup_detail(table)
    finally:
        boot_stages.pack_runs = original  # type: ignore[assignment]

    assert "runs_unpackable=1" in degraded, (
        "a table that does not parse back was shipped anyway — the reader in "
        "the other repo would fail on it with no way to place the blame")
    assert "runs=-" in degraded, "the corrupt token must be DROPPED, not shipped"
    # And the report still serves: every scalar total is stated independently
    # of the packed table, so losing it costs detail and never the answer.
    for key in ("wall_ms", "critical_path_ms", "overlap_ms", "unmeasured_ms"):
        assert f"{key}=" in degraded, (
            f"{key} went missing with the packed table — the degradation took "
            f"the report with it instead of just the detail")
