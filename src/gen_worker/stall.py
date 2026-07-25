"""Progress-based give-up for retry and poll loops (gw#666).

Paul's standing rule, and the whole point of the th#1166 audit: nothing that
can END REAL WORK may key on a wall clock. A fixed budget cannot tell a
healthy three-hour transfer from a wedge, so it is either useless or it
kills work that was about to succeed. The honest question is always "has
anything advanced lately?".

Two tiny values answer it, and every long loop in the SDK is built from
them:

:class:`SilenceWindow`
    Time since the last ADVANCE, where the caller decides what an advance
    is — a peer that answered, a state token that changed, a floor that was
    cleared. ``stalled()`` is the give-up test. There is deliberately no
    total-runtime bound: a loop that keeps advancing runs as long as the
    work does.

:class:`ProgressFloor`
    A monotonic total that must GROW BY A FLOOR to count as an advance
    (gw#655's rule: a trickle is a stall). Without it, a byte a minute
    keeps an any-progress watchdog alive forever.

Both take their clock by injection so tests drive them deterministically
rather than by sleeping.
"""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["ProgressFloor", "SilenceWindow"]

_UNSET: Any = object()


class SilenceWindow:
    """How long since anything advanced — the only legitimate give-up test.

    ``window_s`` is a SILENCE window, never a work budget: it bounds how
    long the loop tolerates hearing nothing, not how long the work may take.
    Derive it from the loop's own cadence (a per-call network timeout, a
    protocol's reporting interval) times headroom, so it is a statement
    about the channel rather than a guess about the job.
    """

    __slots__ = ("window_s", "_now", "_last", "_marker")

    def __init__(
        self, window_s: float, *, now: Callable[[], float] = time.monotonic
    ) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)
        self._now = now
        self._last = now()
        self._marker: Any = _UNSET

    def touch(self) -> None:
        """Record an advance."""
        self._last = self._now()

    def touch_if_changed(self, marker: Any) -> bool:
        """Advance only when a peer-reported ``marker`` differs from the last
        one observed. Returns whether this counted as an advance (the first
        observation always does)."""
        if self._marker is not _UNSET and marker == self._marker:
            return False
        self._marker = marker
        self.touch()
        return True

    def silent_for(self) -> float:
        """Seconds since the last advance."""
        return max(0.0, self._now() - self._last)

    def stalled(self) -> bool:
        return self.silent_for() >= self.window_s


class ProgressFloor:
    """A monotonic total that must move by ``floor`` to count as progress.

    ``cleared(total)`` rebases and returns True only once ``total`` has grown
    by at least ``floor`` since the last clearance — so a transfer that
    trickles can never masquerade as a healthy one (gw#655).
    """

    __slots__ = ("floor", "_base")

    def __init__(self, floor: int, *, base: int = 0) -> None:
        if floor <= 0:
            raise ValueError("floor must be positive")
        self.floor = int(floor)
        self._base = int(base)

    def moved(self, total: int) -> int:
        """How far ``total`` has come since the window opened."""
        return int(total) - self._base

    def cleared(self, total: int) -> bool:
        if self.moved(total) < self.floor:
            return False
        self._base = int(total)
        return True
