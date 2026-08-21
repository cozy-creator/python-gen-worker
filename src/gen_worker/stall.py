"""Progress-based give-up for retry and poll loops."""

from __future__ import annotations

import time
from typing import Any, Callable

__all__ = ["ProgressFloor", "SilenceWindow"]

_UNSET: Any = object()


class SilenceWindow:
    """How long since anything advanced — the only legitimate give-up test."""

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
        """Advance only when a peer-reported ``marker`` differs from the last one observed."""
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
    """A monotonic total that must move by ``floor`` to count as progress."""

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
