"""Progress registry (gw#621 / th#994): named monotonic counters for
long-running phases.

Long phases register a counter (download per-ref bytes, watchdog evidence
during load/compile, warmup jobs, upload bytes, inference steps); the 10s
app heartbeat (activity.on_beat) snapshots the registry onto the wire and
self-diagnoses a counter stalled past its per-phase window. The hub kills
on counter non-advancement or that confession — never on CPU inference.

The counter-name family (prefix before ":") selects the self-diagnosis
window. Windows are code constants — no env knobs — and all sit under the
hub's 10-minute layer-3 backstop so a worker that can still speak
confesses before the hub must infer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

UNIT_BYTES = "bytes"
UNIT_STEPS = "steps"
UNIT_GRAPHS = "graphs"
# Combined watchdog evidence (process+children CPU seconds + process disk
# IO MB, see activity._default_evidence) — covers load/compile phases with
# no natural app-level counter.
UNIT_EVIDENCE = "evidence"

STALL_WINDOW_S: Dict[str, float] = {
    "download": 180.0,
    "load": 240.0,
    "compile": 600.0,
    "warmup": 300.0,
    "upload": 180.0,
    "infer": 300.0,
    "evidence": 300.0,
}
DEFAULT_STALL_WINDOW_S = 300.0

# Overridable in tests (fake clock).
_now = time.monotonic

_lock = threading.Lock()
_counters: Dict[str, "Counter"] = {}


@dataclass(frozen=True)
class Snapshot:
    name: str
    unit: str
    done: float
    total: float  # 0 = unknown
    rate_per_s: float
    age_s: float  # since last advance
    window_s: float
    elapsed_s: float


def window_for(name: str) -> float:
    return STALL_WINDOW_S.get(name.split(":", 1)[0], DEFAULT_STALL_WINDOW_S)


class Counter:
    """One named monotonic counter; open until finish()."""

    def __init__(self, name: str, unit: str, total: float = 0.0) -> None:
        self.name, self.unit = name, unit
        now = _now()
        self._done = 0.0
        self._total = max(0.0, float(total))
        self._started = now
        self._advanced = now
        # Rate sample anchor, refreshed by each snapshot() call.
        self._rate_t = now
        self._rate_v = 0.0
        self._rate = 0.0

    def add(self, n: float) -> None:
        if n > 0:
            with _lock:
                self._done += n
                self._advanced = _now()

    def set_done(self, done: float) -> None:
        with _lock:
            if done > self._done:
                self._done = float(done)
                self._advanced = _now()

    def set_total(self, total: float) -> None:
        with _lock:
            if total > 0:
                self._total = float(total)

    def finish(self) -> None:
        with _lock:
            if _counters.get(self.name) is self:
                del _counters[self.name]

    def _snapshot_locked(self, now: float) -> Snapshot:
        dt = now - self._rate_t
        if dt >= 1.0:
            self._rate = max(0.0, (self._done - self._rate_v) / dt)
            self._rate_t, self._rate_v = now, self._done
        return Snapshot(
            name=self.name, unit=self.unit, done=self._done, total=self._total,
            rate_per_s=self._rate, age_s=max(0.0, now - self._advanced),
            window_s=window_for(self.name), elapsed_s=max(0.0, now - self._started),
        )


def counter(name: str, unit: str, total: float = 0.0) -> Counter:
    """Register-or-get the open counter `name` (idempotent)."""
    with _lock:
        existing = _counters.get(name)
        if existing is not None:
            if total > 0:
                existing._total = float(total)
            return existing
        c = Counter(name, unit, total)
        _counters[name] = c
        return c


class tracking:
    """Context manager: register on enter, finish on exit."""

    def __init__(self, name: str, unit: str, total: float = 0.0) -> None:
        self._args = (name, unit, total)

    def __enter__(self) -> Counter:
        self._counter = counter(*self._args)
        return self._counter

    def __exit__(self, *exc: object) -> None:
        self._counter.finish()


def snapshot() -> List[Snapshot]:
    now = _now()
    with _lock:
        return [c._snapshot_locked(now) for c in _counters.values()]


def freshest() -> Optional[Snapshot]:
    """The most recently advanced open counter — the liveness view. ANY
    advancing counter proves the process is doing real work."""
    snaps = snapshot()
    return min(snaps, key=lambda s: s.age_s) if snaps else None


def self_diagnosis() -> Optional[Snapshot]:
    """Non-None when EVERY open counter is stale past its own window (i.e.
    even the freshest one) — the typed self_stalled confession the beat
    reports so the hub kills on fact, not inference."""
    fresh = freshest()
    if fresh is not None and fresh.age_s > fresh.window_s:
        return fresh
    return None


def reset() -> None:
    """Test hook: drop all open counters."""
    with _lock:
        _counters.clear()
