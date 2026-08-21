"""pgw#1630: the watchdog verdict is KERNEL-EVIDENCE-ONLY.

## The fact this file was written around

In the pgw#1613 kills the child **passed** the kernel-evidence rung. The CAS
fill kept `tree_evidence` advancing, `no_work_accrued` did not fire, and the
kill came from the rung below it — `loop_wedged_no_activity`. The parent looked
at kernel-accounted proof of life and killed anyway, because a LABEL was
missing. Two H3 pods, two pins, byte-identical death, exit 137, rental burned.

pgw#1613's fix opened an activity at the ONE site that had been killed. That
closed one site and left the class: every future long-running path owed a ritual
or died. So the rung is gone, and this file is the fence that keeps it gone.

Everything here drives the PRODUCTION decision objects — `liveness.EvidenceTrack`
and `_ChildSlot._walk_liveness_ladder` — never a copy of their logic.

## No clocks were harmed

`now` is a parameter everywhere, because the thing under test IS a
progress-versus-time relationship and a test that slept would be measuring the
scheduler. The window each assertion uses is computed from the track's own
`window_s`, so none of these numbers is a magic constant either.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest

from gen_worker import activity
from gen_worker.procsplit import liveness, procdiag
from gen_worker.procsplit.liveness import (
    DEFAULT_FLATNESS_FLOOR_S,
    RUNG_ALIVE,
    RUNG_DIAGNOSE,
    RUNG_KILL,
    RUNG_ORDER,
    RUNG_REPORT,
    RUNG_TERM,
    RUNG_UNMEASURABLE,
    EvidenceTrack,
)

#: The KIND every long-running site declares — the four the design names, plus
#: the empty label, which is what a site that forgets looks like. Under
#: pgw#1630 the ladder must behave IDENTICALLY for all of them: that is the
#: whole verdict, expressed as a parametrization.
SITE_LABELS = (
    activity.KIND_BOOT_MATERIALIZE,
    activity.KIND_WARMUP,
    activity.KIND_SELF_MINT_COMPILE,
    activity.KIND_BOOT_ADOPT,
    "",  # the site that forgot the ritual — pgw#1613's actual victim
)


def _track(floor_s: float = DEFAULT_FLATNESS_FLOOR_S) -> EvidenceTrack:
    return EvidenceTrack(floor_s=floor_s)


def _burning(track: EvidenceTrack, *, until: float, step: float) -> float:
    """Feed advancing evidence up to ``until``. Returns the last `now`."""
    now = 0.0
    value = 0.0
    while now < until:
        now += step
        value += 1.0
        track.observe(value, now)
    return now


# ---------------------------------------------------------------------------
# 1. THE SHARPEST FACT — evidence advancing means HELD, unconditionally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", SITE_LABELS, ids=lambda s: s or "no-label")
def test_a_cpu_burning_child_is_ALIVE_whatever_it_did_or_did_not_declare(
    label: str,
) -> None:
    """pgw#1613's death, inverted — and the CLASS closed, not one site.

    The child is burning CPU with a starved event loop. Under the old ladder,
    the one with the label lived and the one without it was SIGKILLed. Under
    pgw#1630 the label is not an input at all, so all five of these are the
    same child.
    """

    track = _track()
    now = _burning(track, until=10 * track.window_s, step=1.0)

    assert track.verdict(now) == RUNG_ALIVE, (
        f"a child accruing kernel-accounted work was not ALIVE with "
        f"label={label!r} — the label is back on the kill path"
    )


def test_the_activity_rung_is_GONE_from_the_parent() -> None:
    """STRUCTURAL. The cooperative rung must not be reintroducible by accident.

    `_hang_verdict` and its `loop_wedged_no_activity` / `no_evidence_source`
    strings were the whole defect. Their absence is asserted here because a
    behavioural test can only prove the paths it thinks to drive, and this one
    proves the code is not there to be reached.
    """

    import ast

    from gen_worker.procsplit import parent as parent_mod

    # STRING LITERALS ONLY. A comment naming the deleted rung is documentation
    # of why it is deleted — which this file wants to survive — while a live
    # string literal is the verdict being produced again.
    module = ast.parse(Path(parent_mod.__file__).read_text())
    literals = {
        node.value for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    gone = {"loop_wedged_no_activity", "no_evidence_source", "no_work_accrued"}
    revived = {v for v in literals if any(g in v for g in gone)}
    assert not revived, (
        f"a deleted verdict is being produced again as a value: {revived}"
    )
    assert not hasattr(parent_mod._ChildSlot, "_hang_verdict"), (
        "the cooperative verdict function is back"
    )
    assert hasattr(parent_mod._ChildSlot, "_walk_liveness_ladder")


# ---------------------------------------------------------------------------
# 2. The ladder, in order, each rung requiring CONTINUED flatness
# ---------------------------------------------------------------------------


def test_a_flat_child_walks_report_diagnose_term_kill_in_order() -> None:
    """Escalation, not a cliff. And the ORDER is the point: a diagnosis is
    captured while the child is still alive, which is the only moment any of it
    is readable."""

    track = _track()
    track.observe(100.0, 0.0)
    w = track.window_s

    seen = [track.verdict(t) for t in (
        w * 0.5, w * 1.5, w * 2.5, w * 3.5, w * 4.5,
    )]
    assert seen == [RUNG_ALIVE, RUNG_REPORT, RUNG_DIAGNOSE, RUNG_TERM, RUNG_KILL]
    assert list(RUNG_ORDER) == seen, "the ladder's declared order and its behaviour disagree"


@pytest.mark.parametrize("rung_at", (1.5, 2.5, 3.5, 4.5))
def test_any_evidence_advance_resets_the_ladder_to_zero(rung_at: float) -> None:
    """A wedge that un-wedges mid-ladder was never a wedge.

    Asserted from EVERY rung including the one immediately before SIGKILL,
    because the rung where this matters most is the last one.
    """

    track = _track()
    track.observe(100.0, 0.0)
    w = track.window_s
    at = w * rung_at
    assert track.verdict(at) != RUNG_ALIVE, "the scenario must start off-ladder"

    # One byte of I/O, one CPU tick — anything at all.
    # Comfortably past `eps`; `100.0 + eps` does not survive float rounding,
    # and this assertion is about the RESET, not about the epsilon.
    assert track.observe(100.0 + track.eps * 10, at) is True
    assert track.verdict(at) == RUNG_ALIVE
    assert track.verdict(at + w * 0.9) == RUNG_ALIVE, (
        "the ladder did not restart from zero; the child is being charged for "
        "flatness it has already disproved"
    )
    # And re-wedging starts again from the FIRST rung, not from where it left
    # off — but against the child's NEW window, because closing that gap was
    # itself a demonstration that this child can legitimately go that long
    # between ticks. The leash grew, which is the observed-gap rule working.
    grown = track.window_s
    assert grown > w, (
        "the child just demonstrated a longer inter-progress gap than any "
        "before it and did not earn a longer window"
    )
    assert track.verdict(at + grown * 0.9) == RUNG_ALIVE
    assert track.verdict(at + grown * 1.5) == RUNG_REPORT


@pytest.mark.parametrize("label", SITE_LABELS, ids=lambda s: s or "no-label")
def test_the_ladder_is_identical_for_every_long_running_site(label: str) -> None:
    """The parameterized reprieve test the design asks for, in its post-Verdict-1
    form: there is no per-site reprieve to prove reachable, so what is proven
    instead is that the site's LABEL changes nothing.

    (a) evidence advancing -> held; (b) evidence forced flat -> report ->
    diagnose -> TERM -> KILL in order. Identical for all five labels, which is
    what makes "a path that forgets loses a label, not a process" true.
    """

    track = _track()
    now = _burning(track, until=3 * track.window_s, step=1.0)
    assert track.verdict(now) == RUNG_ALIVE

    w = track.window_s
    base = now
    assert [track.verdict(base + w * k) for k in (0.5, 1.5, 2.5, 3.5, 4.5)] == [
        RUNG_ALIVE, RUNG_REPORT, RUNG_DIAGNOSE, RUNG_TERM, RUNG_KILL
    ]


# ---------------------------------------------------------------------------
# 3. The window is DERIVED FROM OBSERVATION
# ---------------------------------------------------------------------------


def test_a_child_that_demonstrates_long_gaps_earns_a_longer_leash() -> None:
    """W = max(floor, k x the largest gap this child has ACTUALLY shown).

    The whole no-magic-timeouts point: the only clock left measures absence of
    progress, and its SCALE comes from the child's own progress. An inductor
    compile that legitimately ticks every 100 s is not held to the same window
    as a loop that ticks every 200 ms.
    """

    fast = _track()
    _burning(fast, until=60.0, step=0.2)
    assert fast.largest_gap_s == pytest.approx(0.2, abs=0.01)
    assert fast.window_s == pytest.approx(DEFAULT_FLATNESS_FLOOR_S), (
        "a child with tiny gaps is held to the FLOOR, not to 3x0.2s — the "
        "floor is what stops the derived window collapsing"
    )

    slow = _track()
    slow.observe(1.0, 0.0)
    slow.observe(2.0, 100.0)   # a demonstrated 100 s inter-progress gap
    assert slow.largest_gap_s == pytest.approx(100.0)
    assert slow.window_s == pytest.approx(300.0)
    assert slow.window_s > fast.window_s, (
        "the child that demonstrated it needs longer did not get longer"
    )
    # ...and the kill is four of those windows away, not four of the floor's.
    assert slow.verdict(100.0 + 4 * 300.0) == RUNG_TERM
    assert slow.verdict(100.0 + 4 * 300.0 + 1.0) == RUNG_KILL


def test_the_floor_is_a_lower_bound_and_never_shortens_a_window() -> None:
    """The operator lever pgw#1613 proved must exist ("NO KNOB"), and the
    direction it may move things.

    A smaller floor must not be able to shorten a window the child's own
    observed gaps have already justified — otherwise the knob becomes a way to
    reintroduce the cliff.
    """

    tiny_floor = EvidenceTrack(floor_s=1.0)
    tiny_floor.observe(1.0, 0.0)
    tiny_floor.observe(2.0, 100.0)
    assert tiny_floor.window_s == pytest.approx(300.0), (
        "the observed-gap term must win over a small floor"
    )

    big_floor = EvidenceTrack(floor_s=10_000.0)
    big_floor.observe(1.0, 0.0)
    big_floor.observe(2.0, 100.0)
    assert big_floor.window_s == pytest.approx(10_000.0)


def test_the_default_floor_puts_the_earliest_kill_four_windows_out() -> None:
    """The default is a DERIVED number and its consequence is stated.

    A number in a supervisor has to earn its place: the earliest possible kill
    is 4 x floor of provably-zero kernel work, with a report at 2x and a
    captured diagnosis at 3x. If someone shortens the default, this fails and
    they have to say why.
    """

    track = _track()
    track.observe(1.0, 0.0)
    assert track.window_s == DEFAULT_FLATNESS_FLOOR_S
    assert track.verdict(4 * DEFAULT_FLATNESS_FLOOR_S) == RUNG_TERM
    assert track.verdict(4 * DEFAULT_FLATNESS_FLOOR_S + 1) == RUNG_KILL
    assert DEFAULT_FLATNESS_FLOOR_S * 4 >= 480.0, (
        "the earliest kill is under eight minutes of provably-zero kernel "
        "work; that is a cliff wearing a ladder's clothes"
    )


def test_the_settings_knob_reaches_the_parent() -> None:
    """A lever nothing reads is not a lever. pgw#1620's whole lesson."""

    from gen_worker.config.loader import _ENV_TO_FIELD
    from gen_worker.config.settings import Settings

    assert _ENV_TO_FIELD["GEN_WORKER_WATCHDOG_FLATNESS_FLOOR_S"] == (
        "watchdog_flatness_floor_s"
    )
    assert Settings().watchdog_flatness_floor_s == 0.0, (
        "0 must mean 'use the derived default', not 'kill immediately'"
    )


# ---------------------------------------------------------------------------
# 4. Failure to measure HOLDS
# ---------------------------------------------------------------------------


def test_an_unreadable_proc_holds_and_reports_forever() -> None:
    """`no_evidence_source -> KILL` inverted the burden of proof.

    An unreadable `/proc` — psutil import failure, a race on exit, a container
    that lost the capability — is ABSENCE OF INSTRUMENT, not guilt. Both halves
    are asserted: before any successful read, and (the one that is easy to
    miss) AFTER one, where elapsed time keeps running while nothing is being
    measured.
    """

    never = _track()
    assert never.verdict(0.0) == RUNG_UNMEASURABLE
    assert never.observe(None, 10_000.0) is False
    assert never.verdict(10_000.0) == RUNG_UNMEASURABLE, (
        "an instrument that has never worked must never produce a kill"
    )

    lost = _track()
    lost.observe(100.0, 0.0)
    lost.observe(None, 1.0)  # the instrument goes away
    w = lost.window_s
    assert lost.verdict(w * 99) == RUNG_UNMEASURABLE, (
        "the ladder walked on ELAPSED TIME while nothing was being measured — "
        "an instrument outage must not become a kill"
    )
    # And it recovers the instant the instrument returns.
    lost.observe(100.0, w * 99)
    assert lost.verdict(w * 99) in (RUNG_ALIVE, RUNG_REPORT, RUNG_DIAGNOSE,
                                    RUNG_TERM, RUNG_KILL)
    assert lost.verdict(w * 99) != RUNG_UNMEASURABLE


def test_an_unreadable_sample_does_not_accrue_flatness() -> None:
    """A `None` is not a flat reading, and must not be recorded as one."""

    track = _track()
    track.observe(100.0, 0.0)
    before = track.advanced_at
    track.observe(None, 50.0)
    assert track.advanced_at == before
    assert track.value == 100.0


# ---------------------------------------------------------------------------
# 5. Actions are edge-triggered
# ---------------------------------------------------------------------------


def test_each_rung_acts_once_per_flat_episode() -> None:
    """The ladder is level-triggered; its ACTIONS are not.

    A diagnosis captured on every 15 s sample is a log flood, and a SIGTERM
    re-sent on every sample is a child that never gets its window to unwind.
    """

    track = _track()
    track.observe(100.0, 0.0)
    assert track.claim(RUNG_TERM) is True
    assert track.claim(RUNG_TERM) is False
    # ...and an advance re-arms every rung, because the next flat episode is a
    # different episode.
    track.observe(200.0, 10.0)
    assert track.claim(RUNG_TERM) is True


# ---------------------------------------------------------------------------
# 6. The diagnosis rung produces a real artifact
# ---------------------------------------------------------------------------


def test_the_diagnosis_reads_this_process_and_names_its_state(
    tmp_path: Path,
) -> None:
    """Run against a REAL pid — this one — so the reader is proven against real
    `/proc` rather than a fixture that always agrees with it."""

    report = procdiag.capture(os.getpid(), tmp_path)
    assert f"pid={os.getpid()}" in report
    assert "[wchan]" in report and "[kernel stack]" in report
    assert "[python stack]" in report
    written = list(tmp_path.glob("liveness-diagnosis-*.txt"))
    assert len(written) == 1, "the artifact must survive a pod with no logs API"
    assert written[0].read_text().startswith(f"pid={os.getpid()}")
    # This process is running, so the state letter must be a real one.
    assert procdiag.read_proc_state(os.getpid()) in ("R", "S", "D", "t", "T")


def test_the_diagnosis_never_raises_on_a_process_that_is_gone() -> None:
    """A diagnostic that can raise turns a stall into a crash. The child being
    diagnosed is by definition in a bad state; it may also have just exited."""

    report = procdiag.capture(2_147_483_646)
    assert "2147483646" in report
    # Every source reports its own absence rather than propagating.
    assert "<absent>" in report or "unreadable" in report or "not permitted" in report


def test_a_D_state_is_named_because_it_is_not_the_childs_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single most valuable line the diagnosis can carry.

    `State: D` is an uninterruptible wait: the child cannot answer a SIGTERM
    and did not choose to be there. This fleet has seen exactly that shape on a
    wedged mount, and a kill report that does not say so sends the next reader
    hunting the wrong process.
    """

    monkeypatch.setattr(procdiag, "read_proc_state", lambda pid: "D")
    report = procdiag.capture(os.getpid())
    assert "UNINTERRUPTIBLE WAIT" in report


# ---------------------------------------------------------------------------
# 7. The ladder, driven through the real `_ChildSlot` method
# ---------------------------------------------------------------------------


class _Proc:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.signals: list[str] = []

    def terminate(self) -> None:
        self.signals.append("TERM")

    def kill(self) -> None:
        self.signals.append("KILL")


class _Slot:
    """A structural stand-in for the attributes the ladder reads.

    `_walk_liveness_ladder` is called UNBOUND on this, so the code under test is
    the production method rather than a copy: constructing a real `_ChildSlot`
    needs a live parent, a spawned subprocess and a socket, none of which the
    decision touches.
    """

    def __init__(self, track: EvidenceTrack, dials: list[str]) -> None:
        self.evidence = track
        self.ordinal = 0
        self.label = "g0"
        self.in_flight: dict[Any, Any] = {}
        self.liveness_activity = ""
        self.last_frame_at = 0.0
        self.last_liveness_at = 0.0
        self.watchdog_fired = False

        async def dial(detail: str) -> None:
            dials.append(detail)

        self.p = type("P", (), {
            "_dial_detail": staticmethod(dial),
            "_postmortem_dir": None,
        })()

    # Borrowed from production, not reimplemented: the labels are part of what
    # the reports are asserted to carry.
    from gen_worker.procsplit.parent import _ChildSlot as _Real

    _liveness_labels = _Real._liveness_labels


def test_the_real_ladder_signals_TERM_before_KILL_and_says_so() -> None:
    """End to end through `_ChildSlot._walk_liveness_ladder`.

    The old code went straight to `proc.kill()`. A SIGTERM first is what lets a
    child that is merely slow flush its results instead of losing them, and the
    rungs must be distinguishable on the wire — an operator reading
    `worker_activity_events` has to be able to tell a report from a kill.
    """

    from gen_worker.procsplit.parent import _ChildSlot

    dials: list[str] = []
    track = _track()
    track.observe(100.0, 0.0)
    slot = _Slot(track, dials)
    proc = _Proc()
    w = track.window_s

    async def walk() -> None:
        for t in (w * 0.5, w * 1.5, w * 2.5, w * 3.5, w * 4.5, w * 5.5):
            await _ChildSlot._walk_liveness_ladder(slot, proc, t)  # type: ignore[arg-type]

    asyncio.run(walk())

    assert proc.signals == ["TERM", "KILL"], (
        f"the ladder must ask before it kills, once each; got {proc.signals}"
    )
    phases = [d.split()[0] for d in dials]
    assert phases == [
        "phase=compute_child_stalled",
        "phase=compute_liveness_diagnosis",
        "phase=compute_liveness_term",
        "phase=compute_liveness_kill",
    ], phases
    # Every report shows its working, and carries the labels as LABELS.
    assert all("window_s=" in d and "largest_observed_gap_s=" in d for d in dials)
    assert all("label_activity=" in d for d in dials)


def test_the_real_ladder_signals_NOTHING_while_evidence_advances() -> None:
    """THE pgw#1613 REGRESSION, through the production method.

    No activity is ever declared, the loop has been silent since time zero, and
    the child is burning CPU. The old ladder SIGKILLed this. It must now be
    completely uneventful — no signal, and not even a report.
    """

    from gen_worker.procsplit.parent import _ChildSlot

    dials: list[str] = []
    track = _track()
    slot = _Slot(track, dials)
    proc = _Proc()

    async def walk() -> None:
        now, value = 0.0, 0.0
        for _ in range(200):
            now += 15.0        # the real sampling cadence: budget/4
            value += 1.0
            track.observe(value, now)
            await _ChildSlot._walk_liveness_ladder(slot, proc, now)  # type: ignore[arg-type]

    asyncio.run(walk())

    assert proc.signals == [], (
        "a child with continuously advancing kernel evidence was signalled — "
        "this is exactly the pgw#1613 kill"
    )
    assert dials == [], "and it should not even have been reported as stalled"
    assert slot.watchdog_fired is False


def test_an_unmeasurable_child_is_reported_and_never_signalled() -> None:
    """Failure to measure holds and REPORTS — the report is the product."""

    from gen_worker.procsplit.parent import _ChildSlot

    dials: list[str] = []
    track = _track()
    slot = _Slot(track, dials)
    proc = _Proc()

    async def walk() -> None:
        for t in (10.0, 10_000.0, 100_000.0):
            track.observe(None, t)
            await _ChildSlot._walk_liveness_ladder(slot, proc, t)  # type: ignore[arg-type]

    asyncio.run(walk())

    assert proc.signals == []
    assert len(dials) == 1, "reported once per episode, not once per sample"
    assert dials[0].startswith("phase=compute_liveness_unmeasurable")
    assert "holding, never killing" in dials[0]


# ---------------------------------------------------------------------------
# 8. pgw#1613's landed scope stays landed
# ---------------------------------------------------------------------------


def test_the_pgw1613_activity_scope_survives_as_TELEMETRY() -> None:
    """The `activity.running(KIND_BOOT_MATERIALIZE)` scope pgw#1613 added is
    GOOD telemetry and stays — it is what makes a stall report say WHAT the
    child was doing. What changed is that it decides nothing.

    Its own acceptance tests in `test_boot_materialize.py` are untouched and
    still green; this asserts the scope is still there, so a future "it decides
    nothing, delete it" reading of pgw#1630 fails here first.
    """

    from gen_worker import boot_materialize

    source = Path(boot_materialize.__file__).read_text()
    assert "activity.running(activity.KIND_BOOT_MATERIALIZE)" in source, (
        "pgw#1613's fetch activity is gone; the stall report can no longer say "
        "what the child was doing"
    )


@pytest.fixture(autouse=True)
def _clean_activity() -> Iterator[None]:
    activity.reset_for_tests()
    yield
    activity.reset_for_tests()
