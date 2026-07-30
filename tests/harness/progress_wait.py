"""Progress-gated waiting for the suite (pgw#795).

A test that gives up on a fixed wall clock is not asserting a property of the
code; it is asserting that today's runner is fast enough. pgw#795 is what that
costs: a 15-second deadline in a file the release never touched failed the
0.78.0 publish job TWICE, after observing 2 of the 3 events it wanted. Worse
than the delay: the clock also LIED about the cause. The real defect was a race
with a ~2 s worker-side promotion (see tests/test_residency_republish_pgw628.py),
and a wall clock cannot tell "slow" from "this can never happen now" — so it
reported the only thing it knows, and two releases were spent believing it.

Paul's standing rule (gw#666, ``gen_worker.stall``): a give-up decision comes
from LIVENESS plus PROGRESS-STALENESS, never from a duration. That rule is not
about production code, it is about the shape of the decision — and a test makes
exactly the same decision, with a release attached to it.

So a wait here ends in one of three ways:

* the thing happened (the only success),
* the peer is provably gone — definitive, immediate, no clock,
* nothing has advanced for a staleness window that this run itself MEASURED.

A test whose property is genuinely broken must still FAIL, and quickly enough
to be useful — so the give-up is real, it just keys on evidence.

:class:`Cadence` is the third one. Every advance a wait actually sits through is
a sample of how fast this machine answers today; the window is a multiple of the
slowest such sample, shared across the session because a slow runner is slow for
every test on it. Ten times slower a machine, ten times wider the window. The
floor is what an uncalibrated first wait gets, and it is a SILENCE floor, not a
work budget: it bounds how long we tolerate hearing nothing at all.

And when a wait does expire, it says what it was waiting on, what it last saw,
how long the silence was, and how the window was derived — "saw 2 of 3" was the
only reason pgw#795 was diagnosable at all, and that is the bar.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from gen_worker.stall import SilenceWindow

__all__ = ["Cadence", "StalledError", "await_count", "await_progress"]


class StalledError(TimeoutError):
    """Nothing advanced, so the wait gave up.

    Subclasses ``TimeoutError`` because callers that deliberately probe for
    ABSENCE (``wait_for(..., timeout=2.0)`` inside ``except TimeoutError``)
    predate this and mean it.
    """


class Cadence:
    """The staleness window this run has earned, in seconds.

    ``record(waited_s)`` feeds it one sample: how long a wait sat before the
    thing it wanted advanced. ``window_s`` is ``headroom`` times the slowest
    sample seen (by this cadence or anywhere in the session), floored.

    There is deliberately no total-wait bound. A wait whose peer keeps
    advancing runs as long as the work does, exactly like
    :class:`gen_worker.stall.SilenceWindow`, which this composes.
    """

    _session_lock = threading.Lock()
    _session_slowest = 0.0

    def __init__(
        self, *, floor_s: float = 30.0, headroom: float = 10.0, share: bool = True
    ) -> None:
        if floor_s <= 0 or headroom <= 0:
            raise ValueError("floor_s and headroom must be positive")
        self.floor_s = float(floor_s)
        self.headroom = float(headroom)
        self.share = bool(share)
        self._slowest = 0.0

    def record(self, waited_s: float) -> None:
        waited_s = max(0.0, float(waited_s))
        self._slowest = max(self._slowest, waited_s)
        if self.share:
            with Cadence._session_lock:
                Cadence._session_slowest = max(Cadence._session_slowest, waited_s)

    @property
    def slowest_s(self) -> float:
        if not self.share:
            return self._slowest
        with Cadence._session_lock:
            return max(self._slowest, Cadence._session_slowest)

    @property
    def window_s(self) -> float:
        return max(self.floor_s, self.headroom * self.slowest_s)

    def describe(self) -> str:
        slowest = self.slowest_s
        if self.headroom * slowest <= self.floor_s:
            return f"{self.window_s:.1f}s silence floor (slowest advance {slowest:.2f}s)"
        return (
            f"{self.window_s:.1f}s = {self.headroom:g}x the slowest advance "
            f"this run measured ({slowest:.2f}s)"
        )


def await_progress(
    observe: Callable[[], Any],
    settled: Callable[[Any], bool],
    *,
    what: str,
    cadence: Cadence,
    gone: Optional[Callable[[], Optional[str]]] = None,
    render: Optional[Callable[[Any], str]] = None,
    poll_s: float = 0.02,
) -> Any:
    """Wait until ``settled(observe())``, giving up only on stall or death.

    ``observe`` is the thing under test; a CHANGE in its value is the only
    thing that counts as an advance. Deliberately: the peer being alive, or
    busy, or merely talking about something else is not progress toward what
    you asked for, and a window that refreshes on it never closes — measured
    twice while authoring this (a stuck-but-not-done task, and unrelated
    connection traffic, each held a broken test open indefinitely). A test that
    cannot fail is worse than one that flakes.

    ``gone`` returns a reason string once no further progress is possible (the
    stream ended, the thread exited) and fails the wait immediately, with no
    clock involved.
    """
    window = SilenceWindow(cadence.window_s)
    value = observe()
    last_advance = time.monotonic()

    while not settled(value):
        reason = gone() if gone is not None else None
        if reason is not None:
            raise StalledError(
                f"waiting for {what}: {reason} — last saw "
                f"{render(value) if render else value!r}"
            )
        time.sleep(poll_s)

        now = time.monotonic()
        fresh = observe()
        if fresh != value:
            cadence.record(now - last_advance)
            value, last_advance = fresh, now
            window.touch()
            continue

        # Re-read the window every poll: a slow advance recorded mid-wait
        # widens it, which is the calibration doing its job.
        window.window_s = cadence.window_s
        if window.stalled():
            raise StalledError(
                f"waiting for {what}: nothing advanced for "
                f"{window.silent_for():.1f}s (staleness window "
                f"{cadence.describe()}); last saw "
                f"{render(value) if render else value!r}"
            )

    cadence.record(time.monotonic() - last_advance)
    return value


def await_count(
    observe: Callable[[], int],
    want: int,
    *,
    what: str,
    cadence: Cadence,
    gone: Optional[Callable[[], Optional[str]]] = None,
    poll_s: float = 0.02,
) -> int:
    """``await_progress`` for a monotonic count: wait until it reaches ``want``."""
    return int(
        await_progress(
            observe,
            lambda seen: seen >= want,
            what=f"{want} {what}",
            cadence=cadence,
            gone=gone,
            render=lambda seen: f"{seen} of {want}",
            poll_s=poll_s,
        )
    )
