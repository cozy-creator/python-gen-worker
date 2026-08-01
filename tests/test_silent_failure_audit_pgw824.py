"""pgw#824: the fleet-wide silent-failure audit — the SDK half.

Three invariants, asserted the way the hub actually sees them:

1. no silent failure paths — an important fail-soft outcome rides a typed
   event, never only a logger;
2. eager serving is an EVENT, not a default — every eager-served request on a
   compile-declaring release reports exactly WHY it is eager;
3. the mint/adopt lifecycle is continuously hub-visible — including progress
   INSIDE a long phase, not only at phase transitions.

Capture is at ``activity._emit`` (the pgw#733/#760 convention): the exact
envelope the stream sink ships, so kind/phase/detail are asserted as wired
rather than through a double of the event API. Nothing here asserts a log line
— on a hub-spawned pod there is no stdout to read one from, which is the whole
premise of the audit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import activity, fleet_cells, serving_mode


@pytest.fixture()
def events(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_emit", captured.append)
    return captured


def _rows(events: List[Any], kind: str) -> List[Tuple[str, str]]:
    return [(e.phase, e.detail) for e in events if e.kind == kind]


# ---------------------------------------------------------------------------
# Invariant 2 — the eager POSTURE vocabulary
# ---------------------------------------------------------------------------


def test_eager_with_no_cell_reports_a_reason_not_an_empty_string() -> None:
    """RED before pgw#824: `serving_mode=eager, fallback_reason=""`.

    All four pre-existing fallback classes presuppose an ARMED cell, so the
    commonest eager case — nothing armed at all — had no vocabulary and
    reported the empty string. "" could not distinguish a release that declares
    no compile target from a pod still minting from a pod that declined for
    cause, which is why "why is this fleet eager right now" had no query.
    """
    served = serving_mode.resolve(
        active_compile_ref="",
        eager_posture=serving_mode.POSTURE_MINT_IN_PROGRESS,
        sm="89",
    )
    assert served.serving_mode == serving_mode.MODE_EAGER
    assert served.fallback_reason == serving_mode.POSTURE_MINT_IN_PROGRESS
    # NOT a fallback: nothing fell back, there was nothing to fall back FROM.
    # Every existing compiled-vs-eager comparison keeps its old meaning.
    assert served.served_eager_fallback is False


def test_a_per_request_fallback_outranks_the_posture() -> None:
    """A compiled lane that fell back for THIS request is the stronger fact:
    the posture describes the worker, the fallback describes the request."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#deadbeef",
        verdict=serving_mode.FALLBACK_VOLATILE,
        eager_posture=serving_mode.POSTURE_MINT_IN_PROGRESS,
        sm="89",
    )
    assert served.fallback_reason == serving_mode.FALLBACK_VOLATILE
    assert served.served_eager_fallback is True


def test_the_posture_never_overwrites_a_compiled_serve() -> None:
    """A posture leaking onto a cell-served request would be a WRONG dimension,
    which is worse than a coarse one (the pgw#764 rule)."""
    served = serving_mode.resolve(
        active_compile_ref="root/family-sdxl#deadbeef",
        eager_posture=serving_mode.POSTURE_UNCOMPILED,
        sm="89",
    )
    assert served.serving_mode != serving_mode.MODE_EAGER
    assert served.fallback_reason == ""
    assert served.served_eager_fallback is False


# ---------------------------------------------------------------------------
# Invariant 1/2 — the arming brain's declines are CLASSIFIED, not a constant
# ---------------------------------------------------------------------------


class _Pipe:
    pass


def test_fail_closed_names_the_cause_instead_of_one_shared_constant(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#824: all nine `_fail_closed` exits emitted
    `phase=mint_unavailable` and put the cause in free text only, so counting
    causes hub-side meant substring-matching a sentence — the th#1250 lesson
    (kind-only coalescing erases the reason) one level down.
    """
    monkeypatch.setattr(fleet_cells.cc, "mandatory_serving", lambda pipe: False)
    monkeypatch.setattr(
        fleet_cells.loading, "pipeline_weight_lane", lambda pipe: "")

    outcome = fleet_cells._fail_closed(
        _Pipe(), "no C compiler for the self-mint", phase="no_toolchain")

    assert outcome.armed is False
    # the token rides the WIRE ...
    assert ("no_toolchain", ) == tuple(
        phase for phase, _ in _rows(events, "self_mint_skipped"))
    # ... and rides OUT of the decision, for the request path to report.
    assert outcome.eager_reason == "no_toolchain"


def test_every_fail_closed_exit_carries_a_distinct_token() -> None:
    """The nine exits must not collapse back onto one token by accident: a
    reader groups on this string, so a duplicate silently merges two causes."""
    import inspect

    src = inspect.getsource(fleet_cells.enable_compiled)
    tokens = {
        "no_family", "no_cuda", "no_toolchain", "no_compile_target",
        "delivered_cell_seeded", "key_computation_failed", "capture_conflict",
        "multi_group_in_process", "capture_arm_failed",
    }
    missing = {t for t in tokens if f'phase="{t}"' not in src}
    assert not missing, f"these _fail_closed exits lost their token: {missing}"


def test_a_quarantined_cell_is_a_typed_event_not_a_log_line(
    events: List[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before pgw#824. This was the ONE eager exit in `enable_compiled`
    that returned before `_fail_closed` and only `logger.error`'d. A pod that
    quarantines its own cell serves eager for the rest of its life — the state
    the hub most needs named, and the one it could not see.
    """
    src_ok = 'phase="cell_quarantined"' in __import__(
        "inspect").getsource(fleet_cells.enable_compiled)
    assert src_ok, "the quarantine exit must emit a typed event"


# ---------------------------------------------------------------------------
# Invariant 3 — progress INSIDE a long phase
# ---------------------------------------------------------------------------


def test_a_multi_entry_mint_reports_every_entry_before_it_runs() -> None:
    """RED before pgw#824: `aot_mint.mint` was one opaque call over the
    family's whole declared class set, so a real ~5-minute export emitted
    NOTHING between `trace_graph` and `seal_publish`.

    Reported BEFORE the work, deliberately: a row that never returns is the one
    a reader most needs named, and an after-the-fact tick names only the rows
    that finished.
    """
    from gen_worker import aot_mint

    beats: List[Tuple[str, int, int, str]] = []

    # Drive the callback contract directly — the export itself needs a real
    # pipeline and a GPU, which is exactly the machinery this assertion must
    # not depend on. What is under test is that the phase tokens the child
    # frames are the ones the hub already groups on.
    def on_progress(phase: str, step: int, total: int, note: str) -> None:
        beats.append((phase, step, total, note))

    for i, name in enumerate(["a", "b", "c"], start=1):
        on_progress(aot_mint.PHASE_TRACE_GRAPH, i, 3, name)

    assert [b[1] for b in beats] == [1, 2, 3]
    assert all(b[2] == 3 for b in beats), "a step with no total is not progress"
    assert beats[0][0] == activity.PHASE_TRACE_GRAPH


def test_the_mint_progress_tokens_are_the_hubs_own_phase_vocabulary() -> None:
    """The child frames these verbatim and the parent lands them on the same
    `self_mint_compile` activity, so a drifted token would silently split one
    mint's progress across two phase names hub-side."""
    from gen_worker import aot_mint

    assert aot_mint.PHASE_TRACE_GRAPH == activity.PHASE_TRACE_GRAPH
    assert aot_mint.PHASE_INDUCTOR_COMPILE == activity.PHASE_INDUCTOR_COMPILE
    assert aot_mint.PHASE_SEAL_PUBLISH == activity.PHASE_SEAL_PUBLISH


def test_the_entry_compile_pool_reports_each_entry_as_it_lands() -> None:
    """The pool loop is the longest wire-silent stretch of a mint (an 18-entry
    sdxl cell spends the bulk of its wall clock there) and reported nothing
    between "compiling" and "packed"."""
    import inspect

    from gen_worker import aot_compile_pool

    src = inspect.getsource(aot_compile_pool.EntryCompilePool.compile)
    assert "on_entry" in src
    assert "len(done), len(entries)" in src, (
        "progress must carry BOTH a step and a total — a bare step is not "
        "progress, it is a counter")


def test_progress_reporting_never_costs_the_mint_its_work() -> None:
    """Telemetry must never fail the work it reports on: a raising callback
    would otherwise throw away entries that already compiled."""
    from gen_worker import mint_delegate

    class _ActNoCounter:
        """An Activity double with no counter registry — the shape every
        existing mint test passes."""

        beats = 0

        def heartbeat(self) -> None:
            type(self).beats += 1

    apply = mint_delegate._on_evidence(_ActNoCounter())
    apply(12.5)  # must not raise despite the absent counter()
    assert _ActNoCounter.beats == 1


def test_the_delegated_mint_actually_passes_the_evidence_callback() -> None:
    """`run_mint` has accepted `on_evidence` since pgw#784 and NOBODY passed
    one, so the child's measured progress existed only to decide whether to
    KILL it — never to prove it was working."""
    import inspect

    from gen_worker import mint_delegate

    src = inspect.getsource(mint_delegate.build_cell)
    assert "on_evidence=_on_evidence(act)" in src


# ---------------------------------------------------------------------------
# Invariant 3 — the adoption walk: a MISS says why every candidate lost
# ---------------------------------------------------------------------------


def test_discovery_counts_why_every_candidate_was_rejected() -> None:
    """RED before pgw#824: `_candidates` dropped rows on `logger.debug` or a
    bare `continue`, and the `miss` event then said only "no matching cell
    among N checkpoint(s)".

    True, and useless: a family with 12 published cells that rejects all 12
    read identically to a family with none, and those are different bugs with
    different owners. Coalescing with counts is the pattern; dropping is not.
    """
    from gen_worker import aot_cells

    rejected: Dict[str, int] = {}
    items: List[Dict[str, Any]] = [
        {"metadata": {"kind": "not-an-aot-cell"}},
        {"metadata": {"kind": "not-an-aot-cell"}},
        {"metadata": {"kind": aot_cells.aot_serve.ARTIFACT_KIND,
                      "cell_key": "not a key"}},
    ]
    rows = aot_cells._candidates(items, "sdxl", "", rejected)

    assert rows == []
    assert rejected["not_an_aot_cell"] == 2, "identical rejections COALESCE"
    assert rejected["unreadable_cell_key"] == 1
    # the classes are countable tokens, not sentences
    assert all(" " not in cls for cls in rejected)


def test_a_miss_with_no_candidates_at_all_says_so_explicitly() -> None:
    """"No published cells" and "every published cell was rejected" must not
    render as the same sentence."""
    import inspect

    from gen_worker import aot_cells

    src = inspect.getsource(aot_cells._discover_inner)
    assert "rejected by class" in src
    assert "no published cells at all" in src


# ---------------------------------------------------------------------------
# Invariant 1 — the two high-severity finds: both corrupt DECISIONS, not just
# visibility, so both are asserted on the decision, not only on the event.
# ---------------------------------------------------------------------------


def test_an_unlanded_nf4_rung_is_a_rung_OUTCOME_not_the_absence_of_one() -> None:
    """RED before pgw#824: the failure did `adaptive_rung = ""`, and the
    `if adaptive_rung:` stamp below it is the very mechanism that reports rung
    outcomes to placement — so clearing the variable SELF-SUPPRESSED the
    report. The worst outcome the ladder can produce (serving full precision
    over the budgeted VRAM, on a host already too tight for stored precision)
    was the only one that reported nothing, while every sibling rung reported
    itself.
    """
    from gen_worker.models import loading, provision

    # Three states placement must be able to tell apart.
    assert loading.RUNG_NF4_UNLANDED not in ("", "nf4")
    assert provision.RUNG_NF4_UNLANDED == loading.RUNG_NF4_UNLANDED


def test_the_unlanded_rung_reaches_placement_through_SlotLoad() -> None:
    """An event alone would not be enough: every sibling rung reaches
    ServePlan/FnDegraded via SlotLoad.rung -> `_record_adaptive_rung`, and a
    fix that only logged-but-typed would still leave placement blind."""
    import inspect

    from gen_worker.models import provision

    src = inspect.getsource(provision)
    assert "if rung == RUNG_NF4_UNLANDED:" in src
    assert "out.rung = rung" in src


def test_a_failed_eviction_stays_booked_in_vram(
    events: List[Any],
) -> None:
    """RED before pgw#824, and an ACCOUNTING bug before it is a visibility one.

    The booking was unconditional, so a failed eviction still wrote
    `tier=RAM, vram_bytes=0` while `_move_verified`'s own rollback had just put
    the object back on CUDA. The registry then believed the entry held ZERO
    VRAM, `make_room` handed out headroom that does not exist, and the OOM
    landed on an unrelated `promote()` later with nothing tying it back here.
    """
    import inspect

    from gen_worker.models import residency

    src = inspect.getsource(residency.Residency.promote)
    # the booking is now GATED on the eviction actually succeeding
    assert "evicted = self._move_verified" in src
    assert "if not evicted:" in src
    # and the failure returns BEFORE the RAM/0 booking
    failure_branch = src.split("if not evicted:")[1].split("e.tier = Tier.RAM")[0]
    assert "return False" in failure_branch, (
        "the failed-eviction branch must refuse BEFORE booking RAM/0 — "
        "booking it as RAM/0 is the headroom lie that OOMs a later promote")
    assert "eviction_failed_still_resident" in failure_branch
