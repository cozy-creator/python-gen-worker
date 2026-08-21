from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

from gen_worker.stall import SilenceWindow

__all__ = [
    "Cadence",
    "StalledError",
    "await_count",
    "await_progress",
    "await_progress_async",
]


class StalledError(TimeoutError):
    """Nothing advanced, so the wait gave up."""


class Cadence:
    """The staleness window ONE wait has earned, in seconds."""

    def __init__(self, *, floor_s: float = 30.0, headroom: float = 10.0) -> None:
        if floor_s <= 0 or headroom <= 0:
            raise ValueError("floor_s and headroom must be positive")
        self.floor_s = float(floor_s)
        self.headroom = float(headroom)
        self._slowest = 0.0

    def record(self, waited_s: float) -> None:
        self._slowest = max(self._slowest, max(0.0, float(waited_s)))

    @property
    def slowest_s(self) -> float:
        return self._slowest

    @property
    def window_s(self) -> float:
        return max(self.floor_s, self.headroom * self._slowest)

    def describe(self) -> str:
        if self.headroom * self._slowest <= self.floor_s:
            return (
                f"{self.window_s:.1f}s silence floor (slowest advance in THIS "
                f"wait {self._slowest:.2f}s)"
            )
        return (
            f"{self.window_s:.1f}s = {self.headroom:g}x the slowest advance in "
            f"THIS wait ({self._slowest:.2f}s)"
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
    """Wait until ``settled(observe())``, giving up only on stall or death."""
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


async def await_progress_async(
    observe: Callable[[], Any],
    settled: Callable[[Any], bool],
    *,
    what: str,
    cadence: Cadence,
    gone: Optional[Callable[[], Optional[str]]] = None,
    render: Optional[Callable[[Any], str]] = None,
    poll_s: float = 0.005,
) -> Any:
    """:func:`await_progress` for a wait that must not block the event loop."""
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
        await asyncio.sleep(poll_s)

        now = time.monotonic()
        fresh = observe()
        if fresh != value:
            cadence.record(now - last_advance)
            value, last_advance = fresh, now
            window.touch()
            continue

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
