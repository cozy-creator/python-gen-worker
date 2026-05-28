"""#379: the HF download stall watchdog.

A wedged HTTP/hf_xet read used to block snapshot_download forever while the
emit-only-when-growing progress poller stayed silent — the worker sat in
models_downloading with no progress, no error, no disconnect. The watchdog
converts that into a bounded, OBSERVABLE DownloadStalledError so the worker
reports model.download.failed and the orchestrator reaps + replaces it.
"""

import threading
import time
from pathlib import Path

import pytest

from gen_worker.models.hf_downloader import (
    DownloadStalledError,
    _run_with_stall_watchdog,
)


def test_stall_trips_when_no_byte_progress():
    # Download "hangs" (never returns); scan_bytes never grows -> stall.
    release = threading.Event()

    def hang() -> str:
        release.wait(30)  # blocks well past the stall window
        return "/never"

    t0 = time.monotonic()
    with pytest.raises(DownloadStalledError):
        _run_with_stall_watchdog(
            hang,
            label="acme/m@main",
            progress_root=Path("/tmp"),  # scanned, but scan returns constant 0
            progress_callback=None,
            total_hint=None,
            stall_timeout=0.3,
            wall_clock_max=0.0,
            scan_bytes=lambda _p: 0,  # no growth -> stall
            poll_interval=0.05,
        )
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, "watchdog should trip promptly after the stall window"
    release.set()  # let the abandoned daemon thread finish


def test_wallclock_cap_trips_without_progress_dir():
    # No progress_root (no callback path) -> only the wall-clock cap can trip.
    release = threading.Event()

    def hang() -> str:
        release.wait(30)
        return "/never"

    with pytest.raises(DownloadStalledError):
        _run_with_stall_watchdog(
            hang,
            label="acme/m@main",
            progress_root=None,
            progress_callback=None,
            total_hint=None,
            stall_timeout=0.0,   # stall detection disabled
            wall_clock_max=0.3,  # hard cap
            scan_bytes=lambda _p: 0,
            poll_interval=0.05,
        )
    release.set()


def test_progress_keeps_it_alive_then_completes():
    # Bytes grow each poll, so the stall window never elapses; the download
    # completes and its path is returned. progress_callback sees the growth.
    counter = {"n": 0}

    def grow(_p: Path) -> int:
        counter["n"] += 1
        return counter["n"] * 1000  # always growing

    def download() -> str:
        time.sleep(0.4)  # longer than stall_timeout, but progress keeps it alive
        return "/done/path"

    seen: list[int] = []
    out = _run_with_stall_watchdog(
        download,
        label="acme/m@main",
        progress_root=Path("/tmp"),
        progress_callback=lambda b, _t: seen.append(b),
        total_hint=10_000,
        stall_timeout=0.2,
        wall_clock_max=0.0,
        scan_bytes=grow,
        poll_interval=0.05,
    )
    assert out == "/done/path"
    assert seen and seen == sorted(seen), "progress callback should see monotonic growth"


def test_download_exception_propagates_unchanged():
    # A real download error must propagate as itself, NOT as DownloadStalledError.
    def boom() -> str:
        raise ValueError("404 not found")

    with pytest.raises(ValueError, match="404 not found"):
        _run_with_stall_watchdog(
            boom,
            label="acme/m@main",
            progress_root=None,
            progress_callback=None,
            total_hint=None,
            stall_timeout=5.0,
            wall_clock_max=0.0,
            scan_bytes=lambda _p: 0,
            poll_interval=0.05,
        )


def test_fast_success_returns_immediately():
    out = _run_with_stall_watchdog(
        lambda: "/quick",
        label="acme/m@main",
        progress_root=None,
        progress_callback=None,
        total_hint=None,
        stall_timeout=5.0,
        wall_clock_max=0.0,
        scan_bytes=lambda _p: 0,
        poll_interval=0.05,
    )
    assert out == "/quick"
