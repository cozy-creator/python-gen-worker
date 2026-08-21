"""Byte-level progress for the model load path."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .. import activity as activity_mod
from .. import byte_sources
from .. import postmortem

logger = logging.getLogger(__name__)

#: One counter name for the whole load path; the hub's stall clock runs on
#: non-advancement of whatever counter is freshest, not on the name.
#:
#: pgw#1632: was ``load:staged_bytes``. It is sampled from ``/proc/self/io``
#: read_bytes (with anon-RSS growth as the page-cache-warm alternate) — a READ
#: meter that wore a WRITE verb, and th#2246's first mechanism lane spent a pass
#: hunting the per-child write it implied. The rename is wire-safe because the
#: hub keys on advancement, not on the name (see the module docstring).
COUNTER_NAME = "load:ingested_bytes"
#: How :data:`COUNTER_NAME` is measured. `byte_sources` holds the registry the
#: pgw#1632 lint reads; this is the declaration at the site that produces it.
COUNTER_SOURCE = byte_sources.Source.PROC_READ_IO

EVENT_PHASE = "load_phase"
EVENT_PHASE_DONE = "load_phase_done"
EVENT_PHASE_THRASH = "load_phase_thrash"

_INTERVAL_S = 5.0
_GIB = float(1 << 30)

_REREAD_MULTIPLE = 3.0
_CEILING_FRACTION = 0.9

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


def _gib(n: float) -> str:
    return f"{n / _GIB:.2f} GiB"


class LoadProgressReporter:
    """Samples staging progress while a model load blocks the caller."""

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
        self._phase_bytes = 0
        self._phase_started = time.monotonic()
        self._staged = 0
        self._phase_read0 = 0
        self._phase_anon0 = 0
        self._thrash = ""
        self._cgroup_limit = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._io0: Optional[int] = None
        self._started_unix = 0.0

    def set_phase(self, phase: str, nbytes: int = 0) -> None:
        """Enter ``phase`` (``nbytes`` = the tree this phase stages, when the caller knows it)."""
        if phase == self._phase:
            self._phase_bytes = max(self._phase_bytes, int(nbytes))
            return
        self._close_phase()
        self._phase, self._phase_bytes = phase, max(0, int(nbytes))
        self._phase_started = time.monotonic()
        self._phase_read0 = _proc_read_bytes() or 0
        self._phase_anon0 = (_proc_rss_anon_kb() or 0) * 1024
        self._announce_phase()
        self._tick()

    def _progress(self) -> str:
        return (f"resident {_gib(self._staged)} anon; tree on disk "
                f"{_gib(self.total_bytes)}")

    def _announce_phase(self) -> None:
        try:
            sized = f", {_gib(self._phase_bytes)} tree" if self._phase_bytes else ""
            activity_mod.emit_event(
                EVENT_PHASE,
                f"{self.label}: {self._phase}{sized}; {self._progress()}",
                phase=self._phase,
            )
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-phase event dropped", exc_info=True)

    def _close_phase(self, outcome: str = "") -> None:
        try:
            span_ms = int(max(0.0, time.monotonic() - self._phase_started) * 1000)
            activity_mod.emit_event(
                EVENT_PHASE_DONE,
                f"{self.label}: {self._phase} ended{outcome}; "
                f"{self._progress()}",
                phase=self._phase, duration_ms=span_ms,
            )
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-phase event dropped", exc_info=True)

    def start(self) -> "LoadProgressReporter":
        global _active
        self._io0 = _proc_read_bytes()
        self._started_unix = time.time()
        self._phase_started = time.monotonic()
        with _lock:
            _active = self
        self._tick()
        self._announce_phase()
        t = threading.Thread(
            target=self._run, name="load-progress", daemon=True)
        self._thread = t
        t.start()
        return self

    def stop(self, *, clean: bool, raised: bool = False) -> None:
        """``clean`` clears the death breadcrumb; ``raised`` says the load this reporter wrapped is unwinding."""
        global _active
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self.interval_s + 1.0)
        with _lock:
            if _active is self:
                _active = None
        self._close_phase(" on a RAISE" if raised else " without error")
        try:
            c = activity_mod.scoped_counter(
                COUNTER_NAME, "bytes", self.total_bytes)
            c.finish()
        except Exception:  # noqa: BLE001 - reporting must never break a load
            pass
        if clean:
            postmortem.clear_load_progress(self.marker_path)

    def __enter__(self) -> "LoadProgressReporter":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop(clean=exc_type is None)

    @property
    def thrash(self) -> str:
        """This load's re-read verdict, or ``""``."""
        return self._thrash

    def _cgroup_limit_bytes(self) -> int:
        if not self._cgroup_limit:
            from .memory import cgroup_memory_limit_bytes

            self._cgroup_limit = int(cgroup_memory_limit_bytes() or 0)
        return self._cgroup_limit

    def _check_thrash(self, read: Optional[int], anon: int) -> None:
        if self._thrash or self._phase_bytes <= 0 or read is None:
            return
        phase_read = max(0, read - self._phase_read0)
        staged = max(0, anon - self._phase_anon0)
        limit = self._cgroup_limit_bytes()
        if phase_read <= _REREAD_MULTIPLE * self._phase_bytes:
            return
        if limit <= 0 or anon < _CEILING_FRACTION * limit:
            return
        self._thrash = (
            f"{self._phase}: read {_gib(phase_read)} for a "
            f"{_gib(self._phase_bytes)} set "
            f"({phase_read / float(self._phase_bytes):.1f}x re-read) having "
            f"staged {_gib(staged)}, with anon RSS "
            f"{_gib(anon)} against a {_gib(limit)} cgroup limit — this load "
            f"is re-reading its own bytes through direct reclaim, not "
            f"staging them"
        )
        logger.error("load thrash (pgw#1063) %s: %s", self.label, self._thrash)
        try:
            activity_mod.emit_event(
                EVENT_PHASE_THRASH, f"{self.label}: {self._thrash}",
                phase=self._phase,
            )
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-thrash event dropped", exc_info=True)

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
            readable = self.total_bytes or read
            done = max(0, min(read, readable), rss_anon_kb * 1024)
            self._staged = done
            self._check_thrash(io_now, rss_anon_kb * 1024)
            c = activity_mod.scoped_counter(
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
                "ingested_bytes": done,
                "total_bytes": self.total_bytes,
                "started_unix": self._started_unix,
                "ts_unix": time.time(),
                "pid": os.getpid(),
                "thrash": self._thrash,
            }, self.marker_path)
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-progress tick dropped", exc_info=True)


def set_phase(phase: str, nbytes: int = 0) -> None:
    """Update the active reporter's phase (no-op when no load is running)."""
    with _lock:
        rep = _active
    if rep is not None:
        rep.set_phase(phase, nbytes)


def thrash_verdict() -> str:
    """The active load's re-read verdict, or ``""``."""
    with _lock:
        rep = _active
    return rep.thrash if rep is not None else ""


__all__ = [
    "COUNTER_NAME",
    "COUNTER_SOURCE",
    "EVENT_PHASE",
    "EVENT_PHASE_DONE",
    "EVENT_PHASE_THRASH",
    "LoadProgressReporter",
    "set_phase",
    "thrash_verdict",
]
