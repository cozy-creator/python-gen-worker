"""pgw#655: a worker never reports READY-with-a-serveable-function while that
function's model is absent, and a download that trickles is a stall.

Two independent halves of the same live wedge (Paul's fleet audit, DEBT
R3): boot prefetch logged "failed terminally" and walked on to READY, so the
hub dispatched paid GPU jobs that each re-discovered the missing model; and
the download watchdog reset its window on ANY byte, so a trickle could pin
DOWNLOADING_MODELS forever (the wall-clock cap that was supposed to catch it
had been 0.0 — off — for its whole life).
"""

from __future__ import annotations

import threading

from pathlib import Path

import pytest

from gen_worker.models.download import (
    _HF_DOWNLOAD_MIN_WINDOW_BYTES,
    DownloadStalledError,
    _run_with_stall_watchdog,
)

from harness.hf_bound_endpoints_pgw655 import UPSTREAM_REF
from harness.hub_double import hub_double, is_fn_unavailable, is_ready

_MODULE = "harness.hf_bound_endpoints_pgw655"


# ---------------------------------------------------------------------------
# Half 1 — boot prefetch failure gates the function it belongs to.
# ---------------------------------------------------------------------------


class _Upstream404(Exception):
    """What a missing/private HF repo looks like to the download classifier:
    a 4xx-carrying exception, hence terminal, hence no retry storm."""

    status_code = 404


def test_absent_worker_fetched_model_gates_its_function(monkeypatch) -> None:
    import gen_worker.models.download as download_mod

    calls: list = []

    def _refuse(ref, **kwargs):  # network boundary — the only fake here
        calls.append(ref)
        raise _Upstream404("404 Client Error: Repository Not Found")

    monkeypatch.setattr(download_mod, "download_hf", _refuse)

    with hub_double(modules=(_MODULE,)) as (scheduler, harness):
        conn = scheduler.wait_connection(0)
        unavailable = conn.wait_for(is_fn_unavailable("hf-echo")).fn_unavailable
        assert unavailable.reason == "model_unavailable"
        assert "404" in unavailable.detail
        assert unavailable.axes["ref"] == UPSTREAM_REF

        ready = conn.wait_for(is_ready).state_delta
        # The wedge: READY while advertising a function with no model.
        assert "hf-echo" not in ready.available_functions
        # ...and it is not merely "still loading" either — it is refused.
        assert "hf-echo" not in ready.loading_functions
        # Gating is per function, never the process: the model-free sibling
        # serves, and the worker is alive.
        assert "plain-echo" in ready.available_functions
        assert harness.exit_code is None
        assert calls, "the prefetch never reached the download path"


# ---------------------------------------------------------------------------
# Half 2 — the progress floor. Real watchdog; fakes at the clock/disk boundary.
# ---------------------------------------------------------------------------


def _watchdog(byte_series, *, stall_timeout: float, min_window_bytes: int) -> str:
    """Run the real watchdog over a scripted bytes-on-disk series. The
    download thread finishes only when the series is exhausted, so a stall
    verdict comes from the watchdog, never from the download returning."""
    done = threading.Event()
    seen = iter(byte_series)
    last = [0]

    def _download() -> str:
        done.wait(timeout=30.0)
        return "/tmp/pgw655-done"

    def _scan(_root: Path) -> int:
        try:
            last[0] = next(seen)
        except StopIteration:
            done.set()
        return last[0]

    try:
        return _run_with_stall_watchdog(
            _download,
            label="pgw655",
            progress_root=Path("/nonexistent"),
            progress_callback=None,
            total_hint=None,
            stall_timeout=stall_timeout,
            min_window_bytes=min_window_bytes,
            scan_bytes=_scan,
            poll_interval=0.01,
        )
    finally:
        done.set()


def test_trickling_download_is_a_stall() -> None:
    """One byte per poll forever used to reset the window on every tick —
    the exact shape that pinned DOWNLOADING_MODELS with no wall-clock cap."""
    trickle = (n for n in range(1, 10_000))
    with pytest.raises(DownloadStalledError) as exc:
        _watchdog(trickle, stall_timeout=0.2, min_window_bytes=1024)
    assert "stalled" in str(exc.value)


def test_download_clearing_the_floor_is_not_a_stall() -> None:
    series = [4096 * i for i in range(1, 40)]
    assert _watchdog(series, stall_timeout=0.2, min_window_bytes=1024)


def test_zero_progress_is_still_a_stall() -> None:
    with pytest.raises(DownloadStalledError):
        _watchdog([0] * 10_000, stall_timeout=0.2, min_window_bytes=1024)


def test_bursty_transfer_survives() -> None:
    """A transfer that sits still for several polls and then jumps past the
    floor must NOT be killed — the floor is a rate over the window, not a
    per-tick demand."""
    series: list[int] = []
    total = 0
    for _ in range(20):
        series.extend([total] * 5)
        total += 4096
    assert _watchdog(iter(series), stall_timeout=0.3, min_window_bytes=1024)


def test_the_shipped_floor_is_non_zero() -> None:
    """The constant this issue exists to fix: the wall-clock cap was 0.0 —
    'no bound at all' — for its entire life."""
    import gen_worker.models.download as download_mod

    assert _HF_DOWNLOAD_MIN_WINDOW_BYTES > 0
    assert not hasattr(download_mod, "_HF_DOWNLOAD_MAX_SECONDS"), (
        "the dead wall-clock knob must stay deleted (pgw#655)"
    )
