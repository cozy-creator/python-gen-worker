"""Byte-level progress for the model load path.

Staging/hydration is one long blocking diffusers call with nothing ticking, so
without a PRODUCER during the load the hub's stall clock has nothing to hold
the worker to and a SIGKILL has no "where".

One reporter fixes both, riding the EXISTING channels (no parallel heartbeat
systems):

* a sampler thread ticks a byte counter from ``/proc/self/io`` read_bytes
  plus anonymous-RSS growth — real external evidence of staging progress,
  independent of any loader hook — which the 10s app beat then carries to
  the hub as counter advancement (``activity.on_beat``);
* every tick overwrites a small breadcrumb file
  (:func:`gen_worker.postmortem.write_load_progress`); a SIGKILL mid-load
  is then attributed by the surviving parent as "died at
  hydrate:transformer, N/M GiB" instead of a blank.
* every phase transition emits a typed, DURABLE activity event naming the
  component — ``load_phase`` on entry, ``load_phase_done`` on exit with the
  measured span in ``duration_ms``. The counter says a number is moving; the
  events say WHAT the worker is doing, which is what a long load needs while it
  is still ALIVE. The breadcrumb only reaches the hub through a death, and a
  slow load that eventually succeeds leaves no death to read.

Why events and not the counter name or the activity's phase: the hub makes a
RUNNING update durable only on a phase transition, and this load
does not own the phase of the activity it runs under — ``ensure_setup``'s
``self_mint_compile`` does. ``emit_event`` sends a COMPLETED update, which is
always durable, and deliberately does not touch the open activity.

Off-Linux (/proc missing) the reporter still reports phases; only the byte
sampling is an honest no-op.
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

logger = logging.getLogger(__name__)

#: One counter name for the whole load path; the hub's stall clock runs on
#: non-advancement of whatever counter is freshest, not on the name.
COUNTER_NAME = "load:staged_bytes"

#: A load phase STARTED (``duration_ms`` 0 — nothing is measured yet). This
#: is the row that names the component a hung load is hung on.
EVENT_PHASE = "load_phase"
#: A load phase FINISHED, carrying its measured span (the number goes in the
#: column, never interpolated into the detail).
EVENT_PHASE_DONE = "load_phase_done"
#: This phase is RE-READING its own set instead of staging it — the
#: direct-reclaim crawl, confessed while the worker is still alive.
EVENT_PHASE_THRASH = "load_phase_thrash"

_INTERVAL_S = 5.0
_GIB = float(1 << 30)

#: How many full passes over a phase's own bytes count as staging before the
#: reads are re-reads. THREAT: a staging admitted against an estimate that
#: turned out low crawls in direct reclaim (measured: 1.578 TB read for a 105 GB
#: set, a 15x re-read, across 37 minutes of billed H100 before the kernel
#: OOM-killed it). Derivation: a cold load reads each byte ONCE (1x) and a
#: page-cache-warm load reads ~0; 3x is three full passes, which no legitimate
#: staging shape reaches. Conjunctive with the ceiling fraction below, so a
#: merely large load cannot trip it.
_REREAD_MULTIPLE = 3.0
#: Fraction of the cgroup limit anon RSS must occupy for the re-reads to be
#: attributable to reclaim pressure rather than to a big tree. Measured at the
#: incident: 232.9 GiB anon of a 233.76 GiB cgv1 ceiling = 99.6%.
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
        self._phase_bytes = 0
        self._phase_started = time.monotonic()
        self._staged = 0
        # This phase's own read/anon baselines and the verdict once it has been
        # reached (sticky — the regime does not un-happen, and the loader reads
        # it between components).
        self._phase_read0 = 0
        self._phase_anon0 = 0
        self._thrash = ""
        self._cgroup_limit = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._io0: Optional[int] = None
        self._started_unix = 0.0

    # -- loader-facing ------------------------------------------------------

    def set_phase(self, phase: str, nbytes: int = 0) -> None:
        """Enter ``phase`` (``nbytes`` = the tree this phase stages, when the
        caller knows it). Closes the phase being left, so the pair of events
        is a phase table with real spans."""
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

    # -- phase events -------------------------------------------------------

    def _announce_phase(self) -> None:
        try:
            sized = f", {_gib(self._phase_bytes)} tree" if self._phase_bytes else ""
            activity_mod.emit_event(
                EVENT_PHASE,
                f"{self.label}: {self._phase}{sized}; staged "
                f"{_gib(self._staged)} of {_gib(self.total_bytes)}",
                phase=self._phase,
            )
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-phase event dropped", exc_info=True)

    def _close_phase(self) -> None:
        try:
            span_ms = int(max(0.0, time.monotonic() - self._phase_started) * 1000)
            # "ended", never "complete": `load_slot` stops the reporter from a
            # `finally`, so this row also closes a phase that RAISED. The span
            # is the honest fact either way; the outcome is the raised error's
            # to report.
            activity_mod.emit_event(
                EVENT_PHASE_DONE,
                f"{self.label}: {self._phase} ended; staged "
                f"{_gib(self._staged)} of {_gib(self.total_bytes)}",
                phase=self._phase, duration_ms=span_ms,
            )
        except Exception:  # noqa: BLE001 - reporting must never break a load
            logger.debug("load-phase event dropped", exc_info=True)

    # -- lifecycle ----------------------------------------------------------

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

    def stop(self, *, clean: bool) -> None:
        global _active
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self.interval_s + 1.0)
        with _lock:
            if _active is self:
                _active = None
        # The last phase closes on ANY stop: a raise is still a span that
        # ended, and the event names where the time went. Only a kernel kill
        # skips this — that one is the breadcrumb's job.
        self._close_phase()
        try:
            c = activity_mod.scoped_counter(
                COUNTER_NAME, "bytes", self.total_bytes)
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

    # -- the re-read (direct-reclaim) verdict ---------------------

    @property
    def thrash(self) -> str:
        """This load's re-read verdict, or ``""``. Sticky once reached."""
        return self._thrash

    def _cgroup_limit_bytes(self) -> int:
        if not self._cgroup_limit:
            # CYCLE: memory imports nothing from here.
            from .memory import cgroup_memory_limit_bytes

            self._cgroup_limit = int(cgroup_memory_limit_bytes() or 0)
        return self._cgroup_limit

    def _check_thrash(self, read: Optional[int], anon: int) -> None:
        """Decide whether this phase is STAGING or RE-READING.

        THREAT: a staging admitted against an estimate that turned out low
        does not fail — it crawls. Every fault goes through direct reclaim, the
        process reads its own set over and over, and the only thing that ends
        it is the kernel's OOM killer, tens of minutes and tens of dollars
        later (measured: 1.578 TB read for a 105 GB set, 37 minutes, rss_anon
        232.9 GiB of a 233.76 GiB ceiling).

        The observable that decides it correctly is a CONJUNCTION, because
        each half alone has an innocent explanation: a big tree reads a lot,
        and a load with room may sit high. Together — reads past several
        full passes over this phase's OWN bytes, WHILE anon sits against the
        cgroup limit — there is no innocent reading left: a healthy load
        reads each byte about once and a page-cache-warm one reads ~none.

        Anon GROWTH is deliberately not treated as proof of progress: in the
        incident it grew the whole time (130 -> 232.9 GiB) while the estimate
        was simply wrong about what the set weighed.

        The verdict is a confession, not a kill: the counter above already
        stopped crediting re-reads, so the existing stall authority sees a
        stalled load, and the loader refuses the next component with these
        numbers instead of admitting one more into the crawl.
        """
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
            # Reads BEYOND the set's own size are re-reads, not progress —
            # crediting them lets a direct-reclaim crawl report TB of
            # "advancement" so the stall clock never sees a stalled load. Cap
            # the read-derived credit at what there is to read; anon growth
            # (real staged bytes) still advances it honestly.
            readable = self.total_bytes or read
            done = max(0, min(read, readable), rss_anon_kb * 1024)
            self._staged = done
            # ABSOLUTE counters: the phase baselines are absolute too, and a
            # delta-from-load-start would silently zero the comparison.
            self._check_thrash(io_now, rss_anon_kb * 1024)
            # Activity-owned, so a load advances the clock of the phase doing it
            # and not whatever else is open on this pod.
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
                "staged_bytes": done,
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
    """The active load's re-read verdict, or ``""``.

    Non-empty means THIS process has been measured re-reading a set it
    cannot hold instead of staging it. The loader refuses the next
    component with it rather than admitting one more into the crawl."""
    with _lock:
        rep = _active
    return rep.thrash if rep is not None else ""


__all__ = [
    "COUNTER_NAME",
    "EVENT_PHASE",
    "EVENT_PHASE_DONE",
    "EVENT_PHASE_THRASH",
    "LoadProgressReporter",
    "set_phase",
    "thrash_verdict",
]
