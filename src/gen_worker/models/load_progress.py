"""pgw#1041: byte-level progress for the model load path.

The ie#615 attempt-4 load ran 94 minutes with ZERO progress telemetry and
died to an unattributable SIGKILL. The activity/beat machinery (gw#621,
th#1451) was already in place — what was missing is a PRODUCER during the
load: fetch feeds counters, but staging/hydration is one long blocking
diffusers call with nothing ticking, so the hub's stall clock had nothing
to hold the worker to and the death dial had no "where".

One reporter fixes both, riding the EXISTING channels (WORKER-CONTRACTS §1:
no parallel heartbeat systems):

* a sampler thread ticks a byte counter from ``/proc/self/io`` read_bytes
  plus anonymous-RSS growth — real external evidence of staging progress,
  independent of any loader hook — which the 10s app beat then carries to
  the hub as counter advancement (``activity.on_beat``);
* every tick overwrites a small breadcrumb file
  (:func:`gen_worker.postmortem.write_load_progress`); a SIGKILL mid-load
  is then attributed by the surviving parent as "died at
  hydrate:transformer, N/M GiB" instead of a blank.

Off-Linux (/proc missing) the reporter is an honest no-op.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .. import activity as activity_mod
from .. import postmortem
from .. import progress as progress_mod

logger = logging.getLogger(__name__)

#: One counter name for the whole load path; the hub's stall clock runs on
#: non-advancement of whatever counter is freshest, not on the name.
COUNTER_NAME = "load:staged_bytes"

_INTERVAL_S = 5.0

_lock = threading.Lock()
_active: Optional["LoadProgressReporter"] = None


def _proc_read_bytes() -> Optional[int]:
    try:
        for line in open("/proc/self/io"):
            if line.startswith("read_bytes:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _proc_rss_anon_kb() -> Optional[int]:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


class LoadProgressReporter:
    """Samples staging progress while a model load blocks the caller.

    ``label`` names the load (function/ref); ``total_bytes`` is the on-disk
    tree size about to be staged (0 = unknown; the counter then reports
    done-only). ``phase`` is updated by the loader as it moves through
    components (:func:`set_phase`)."""

    def __init__(
        self,
        label: str,
        total_bytes: int,
        *,
        marker_path: Optional[Path] = None,
        interval_s: float = _INTERVAL_S,
    ) -> None:
        self.label = label
        self.total_bytes = max(0, int(total_bytes))
        self.marker_path = marker_path
        self.interval_s = max(0.5, float(interval_s))
        self._phase = "load"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._io0: Optional[int] = None
        self._started_unix = 0.0

    # -- loader-facing ------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        self._phase = phase
        self._tick()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "LoadProgressReporter":
        global _active
        self._io0 = _proc_read_bytes()
        self._started_unix = time.time()
        with _lock:
            _active = self
        self._tick()
        t = threading.Thread(
            target=self._run, name="load-progress", daemon=True)
        self._thread = t
        t.start()
        return self

    def stop(self, *, clean: bool) -> None:
        global _active
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self.interval_s + 1.0)
        with _lock:
            if _active is self:
                _active = None
        try:
            c = progress_mod.counter(COUNTER_NAME, "bytes", self.total_bytes)
            c.finish()
        except Exception:  # noqa: BLE001 - reporting must never break a load
            pass
        if clean:
            postmortem.clear_load_progress(self.marker_path)
        # An unclean stop leaves the breadcrumb for the death attribution.

    def __enter__(self) -> "LoadProgressReporter":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop(clean=exc_type is None)

    # -- sampling -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._tick()

    def _tick(self) -> None:
        try:
            io_now = _proc_read_bytes()
            read = (
                io_now - self._io0
                if io_now is not None and self._io0 is not None else 0
            )
            rss_anon_kb = _proc_rss_anon_kb() or 0
            # Bytes STAGED is evidenced by whichever is further along:
            # cold reads show in io, page-cache-warm loads show as anon RSS.
            done = max(0, read, rss_anon_kb * 1024)
            c = progress_mod.counter(
                COUNTER_NAME, "bytes",
                max(self.total_bytes, done),
            )
            c.set_done(done)
            activity_mod.note_progress()
            postmortem.write_load_progress({
                "label": self.label,
                "phase": self._phase,
                "read_bytes": read,
                "rss_anon_kb": rss_anon_kb,
                "staged_bytes": done,
                "total_bytes": self.total_bytes,
                "started_unix": self._started_unix,
                "ts_unix": time.time(),
                "pid": os.getpid(),
            }, self.marker_path)
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-progress tick dropped", exc_info=True)


def set_phase(phase: str) -> None:
    """Update the active reporter's phase (no-op when no load is running)."""
    with _lock:
        rep = _active
    if rep is not None:
        rep.set_phase(phase)


__all__ = ["COUNTER_NAME", "LoadProgressReporter", "set_phase"]
