"""pgw#1630: the parent's kill decision, and NOTHING ELSE decides it.

## The fact this module exists for

In the pgw#1613 kills the child PASSED the kernel-evidence rung. The CAS fill
kept ``tree_evidence`` advancing, so ``no_work_accrued`` did not fire — and the
kill came from the next rung down, ``loop_wedged_no_activity``. The parent
looked at kernel-accounted proof of life and killed anyway, because a LABEL was
missing. `watchdog_loop`'s own docstring promises the opposite: *"The parent
kills only what is provably NOT RUNNING; a child that runs but serves nothing is
the hub's stall clock to reap."* Two pods, two pins, byte-identical death, exit
137, `all_declared_functions_disabled`, rental burned.

pgw#1613's fix opened an activity at the one site that had been killed. That
closed one site and left the CLASS: every future long-running path owes a ritual
or dies, and the price of keeping the cooperative design is a mandatory
reprieve-test at every declaration site, forever.

## The design

1. **One evidence source.** ``proc_evidence.tree_evidence`` — tree CPU seconds
   (live + reaped) plus IO MB, sampled by the parent against a high-water mark.
   Zero cooperation, whole lifecycle including spawn->hello. A tenant can fake
   it into looking ALIVE, never into looking WEDGED, and "alive but serving
   nothing" is the hub's stall clock, exactly as the module already says.
2. **One binary verdict.** Progress within the window -> alive. Flat beyond it
   -> escalate. Loop pings, declared activities and in-flight counts decide
   NOTHING; they are carried as the report's LABEL and nothing more.
3. **Escalation, not a cliff.** report -> diagnose -> SIGTERM -> SIGKILL, one
   window apart, and every rung requires flatness to have CONTINUED. Any
   evidence advance resets the ladder to zero: a wedge that un-wedges mid-ladder
   was never a wedge.
4. **The window is DERIVED FROM OBSERVATION.** ``W = max(floor, k x the largest
   inter-progress gap this child has actually demonstrated)``. A child that has
   shown 40 s gaps during a compile earns a proportionally longer leash; one
   that ticks every 200 ms gets a short one. This is the no-magic-timeouts rule
   satisfied rather than restated: the only clock left measures ABSENCE OF
   PROGRESS, and its scale comes from the child's own progress.
5. **Failure to measure HOLDS.** An unreadable ``/proc`` — psutil import
   failure, a race on exit — is absence of instrument, not guilt. It reports
   and holds. Inverting the burden of proof there is how an instrument outage
   becomes a kill.

The asymmetry that justifies all of it: a false kill costs a cold reboot, a pod
retirement and a burned rental; a missed wedge costs minutes until the hub's own
request timeout catches it. So the reaper is biased hard toward "held" and kills
only on what the kernel can prove.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "DEFAULT_FLATNESS_FLOOR_S",
    "EvidenceTrack",
    "RUNG_ALIVE",
    "RUNG_DIAGNOSE",
    "RUNG_KILL",
    "RUNG_ORDER",
    "RUNG_REPORT",
    "RUNG_TERM",
    "RUNG_UNMEASURABLE",
    "WINDOW_SLACK",
]

#: The child is accruing kernel-accounted work. Nothing else needs to be true.
RUNG_ALIVE = "alive"
#: Flat past one window: SAY SO, with whatever label the telemetry has. No
#: signal, no state change — the report is the product.
RUNG_REPORT = "report"
#: Flat past two windows: capture a diagnosis. This is the rung that turns the
#: NEXT kill from a mystery into a filed bug, and it is the reason the ladder
#: exists at all rather than a single threshold.
RUNG_DIAGNOSE = "diagnose"
#: Flat past three windows: SIGTERM. The child gets a window to unwind.
RUNG_TERM = "term"
#: Flat past four windows, having ignored the SIGTERM: SIGKILL.
RUNG_KILL = "kill"
#: The instrument could not be read. HOLD and report — absence of instrument is
#: never guilt. Deliberately not part of the ordered ladder.
RUNG_UNMEASURABLE = "unmeasurable"

#: The ladder, in the order a continuously-flat child walks it.
RUNG_ORDER = (RUNG_ALIVE, RUNG_REPORT, RUNG_DIAGNOSE, RUNG_TERM, RUNG_KILL)

#: How many times the largest OBSERVED inter-progress gap the window allows.
#:
#: Not a timeout: it scales a measurement. A child whose evidence has ticked at
#: most every G seconds is given 3G of silence before anyone even reports —
#: three consecutive worst-observed gaps with nothing in between, which no
#: shape that has been ticking at G produces by chance. The kill is at 4W = 12G.
WINDOW_SLACK = 3.0

#: The FLOOR under the derived window, in seconds. Settings-overridable, which
#: is the operator lever pgw#1613 proved must exist ("NO KNOB").
#:
#: DERIVATION, because a number in this file has to earn its place. The floor
#: covers the case the observed-gap term cannot: a child whose demonstrated gaps
#: are small and which then enters a single long stall that accrues NOTHING
#: measurable — an uninterruptible D-state wait on a slow mount is the shape
#: this fleet has actually seen. The floor must exceed such a stall's plausible
#: length, and the ladder then multiplies it: the EARLIEST possible kill is
#: 4 x floor = 8 minutes of provably-zero kernel work, with a report at 2 min
#: and a captured diagnosis at 4 min. It never SHORTENS a window — the
#: observed-gap term can only lengthen it — so a child that demonstrates long
#: legitimate gaps is not held to this number at all.
DEFAULT_FLATNESS_FLOOR_S = 120.0


@dataclass
class EvidenceTrack:
    """One child's kernel evidence, its derived window, and its ladder position.

    Pure: it is fed samples and a clock and returns a rung. Every test in
    `test_watchdog_evidence_pgw1630.py` drives THIS object, so the decision under
    test is the production decision and not a copy of its logic.
    """

    #: Settings-supplied lower bound on the flatness window.
    floor_s: float = DEFAULT_FLATNESS_FLOOR_S
    slack: float = WINDOW_SLACK
    #: Minimum advance that counts as progress. Mirrors `activity._EVIDENCE_EPS`.
    eps: float = 0.05

    #: High-water evidence and when it last ADVANCED. `None` = never measured.
    value: Optional[float] = None
    advanced_at: float = 0.0
    #: The largest interval between two successive ADVANCES this boot. It grows
    #: only when evidence actually moves again, which is what makes it a
    #: measurement of the child rather than of the wall clock: a gap is only
    #: known to have been a gap once it ends.
    largest_gap_s: float = 0.0
    #: True once the instrument has failed to read at least once — reported, and
    #: never used as evidence of anything.
    unreadable: bool = False
    #: Rungs already acted on, so each fires once per flat episode.
    fired: "set[str]" = field(default_factory=set)

    # -- sampling ----------------------------------------------------------

    def observe(self, evidence: Optional[float], now: float) -> bool:
        """Feed one sample. Returns True when the ladder RESET (evidence moved).

        A `None` sample is an instrument failure, not a flat reading: it leaves
        `advanced_at` exactly where it was, so an unreadable `/proc` neither
        proves life nor accrues flatness against the child.
        """

        if evidence is None:
            # An instrument failure is NOT a flat reading. `advanced_at` stays
            # exactly where it was, and `verdict` refuses to walk the ladder
            # while this is set — see the RUNG_UNMEASURABLE branch there.
            self.unreadable = True
            return False
        self.unreadable = False
        if self.value is None:
            # First measurement: the flatness clock starts HERE, not at spawn.
            # Anything before the first successful read is unmeasured, and
            # unmeasured never counts against a child.
            self.value = evidence
            self.advanced_at = now
            self.fired.clear()
            return True
        if evidence - self.value < self.eps:
            # Flat. And the flatness accumulated ACROSS an instrument outage is
            # real rather than assumed: tree CPU seconds and IO bytes are
            # monotonic, so a reading equal to the pre-outage one proves zero
            # work happened in between. That is why a recovered instrument may
            # hand a child straight to a late rung — the evidence is retroactive.
            return False
        gap = now - self.advanced_at
        if gap > self.largest_gap_s:
            self.largest_gap_s = gap
        self.value = evidence
        self.advanced_at = now
        self.fired.clear()
        return True

    # -- the window --------------------------------------------------------

    @property
    def window_s(self) -> float:
        """How long evidence may be flat before the first rung.

        Derived from what this child has DEMONSTRATED, floored by the operator
        lever. Nothing here is a declared duration for a piece of work.
        """
        return max(float(self.floor_s), float(self.slack) * float(self.largest_gap_s))

    def flat_for(self, now: float) -> float:
        if self.value is None:
            return 0.0
        return max(0.0, now - self.advanced_at)

    # -- the verdict -------------------------------------------------------

    def verdict(self, now: float) -> str:
        """The rung this child is on. The ONE decision, from ONE input."""

        if self.value is None or self.unreadable:
            # Never measured, or NOT MEASURABLE RIGHT NOW. Both hold.
            #
            # This replaces `no_evidence_source -> KILL`, which made an
            # unreadable `/proc` a death sentence. The second half matters just
            # as much as the first: if the instrument fails AFTER a successful
            # read, elapsed time keeps accruing while nothing is being measured,
            # and a ladder that walked on it would kill a healthy child on the
            # strength of an instrument outage. We cannot say a child is flat
            # when we cannot read it.
            return RUNG_UNMEASURABLE
        window = self.window_s
        flat = self.flat_for(now)
        if flat <= window:
            return RUNG_ALIVE
        if flat <= 2.0 * window:
            return RUNG_REPORT
        if flat <= 3.0 * window:
            return RUNG_DIAGNOSE
        if flat <= 4.0 * window:
            return RUNG_TERM
        return RUNG_KILL

    def claim(self, rung: str) -> bool:
        """True the FIRST time this flat episode reaches ``rung``.

        The ladder is level-triggered (it is recomputed from flatness on every
        sample) but its ACTIONS are edge-triggered: a diagnosis captured every
        tick is a log flood, and a SIGTERM re-sent every tick is a child that
        never gets its window to unwind. `observe` clears this on any advance,
        which is what "any evidence advance resets the ladder" means in code.
        """
        if rung in self.fired:
            return False
        self.fired.add(rung)
        return True

    def describe(self, now: float) -> str:
        """The arithmetic, for a report that shows its working."""
        return (
            f"flat_s={self.flat_for(now):.1f} window_s={self.window_s:.1f} "
            f"largest_observed_gap_s={self.largest_gap_s:.1f} "
            f"floor_s={float(self.floor_s):.0f} slack={float(self.slack):.1f} "
            f"evidence={('none' if self.value is None else f'{self.value:.1f}')} "
            f"instrument={'unreadable' if self.unreadable else 'ok'}"
        )
