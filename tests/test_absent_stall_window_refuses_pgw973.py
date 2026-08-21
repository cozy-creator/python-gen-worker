"""An unset limit is a REFUSAL."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

import pytest

from gen_worker.models.download import DownloadStalledError, _run_with_stall_watchdog


def _watchdog(byte_totals: Iterator[int], *, stall_timeout: float) -> str:
    done = threading.Event()
    last = [0]

    def _download() -> str:
        done.wait(timeout=30.0)
        return "/tmp/pgw973-done"

    def _scan(_root: Path) -> int:
        try:
            last[0] = next(byte_totals)
        except StopIteration:
            done.set()
        return last[0]

    try:
        return _run_with_stall_watchdog(
            _download,
            label="pgw973",
            progress_root=Path("/nonexistent"),
            progress_callback=None,
            total_hint=None,
            stall_timeout=stall_timeout,
            min_window_bytes=1024,
            scan_bytes=_scan,
            poll_interval=0.01,
        )
    finally:
        done.set()


@pytest.mark.parametrize("absent", [0, 0.0, -1.0])
def test_an_absent_stall_window_is_refused_not_silently_infinite(absent: float) -> None:
    with pytest.raises(ValueError) as exc:
        _watchdog(iter([1, 2, 3]), stall_timeout=absent)
    assert "window_s must be positive" in str(exc.value), (
        "a non-positive stall window must be refused by SilenceWindow's own guard; "
        "substituting an unbounded window is how a watchdog silently stops existing"
    )


def test_the_watchdog_still_catches_the_runaway_it_exists_for() -> None:
    """The surviving bound, unchanged: a trickle below the progress floor is a stall however long it keeps dribbling."""
    trickle = (n for n in range(1, 10_000))
    with pytest.raises(DownloadStalledError) as exc:
        _watchdog(trickle, stall_timeout=0.2)
    assert "stalled" in str(exc.value)


def test_a_healthy_transfer_is_still_admitted() -> None:
    """And a transfer clearing the floor every window runs to completion — the refusal above is on absence, not on slow-but-advancing work."""
    healthy = iter([2048 * n for n in range(1, 6)])
    assert _watchdog(healthy, stall_timeout=5.0) == "/tmp/pgw973-done"
