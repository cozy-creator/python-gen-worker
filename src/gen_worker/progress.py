"""Progress registry: named monotonic counters for long-running phases."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

UNIT_BYTES = "bytes"
UNIT_STEPS = "steps"
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

_now = time.monotonic

_lock = threading.Lock()

_counters: Dict[Tuple[str, str], "Counter"] = {}


@dataclass(frozen=True)
class Snapshot:
    name: str
    unit: str
    done: float
    total: float
    rate_per_s: float
    age_s: float
    window_s: float
    elapsed_s: float
    owner: str = ""


def window_for(name: str) -> float:
    return STALL_WINDOW_S.get(name.split(":", 1)[0], DEFAULT_STALL_WINDOW_S)


class Counter:
    """One named monotonic counter; open until finish()."""

    def __init__(
        self, name: str, unit: str, total: float = 0.0, owner: str = "",
    ) -> None:
        self.name, self.unit, self.owner = name, unit, owner
        now = _now()
        self._done = 0.0
        self._total = max(0.0, float(total))
        self._started = now
        self._advanced = now
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
            if _counters.get((self.owner, self.name)) is self:
                del _counters[(self.owner, self.name)]

    def _snapshot_locked(self, now: float) -> Snapshot:
        dt = now - self._rate_t
        if dt >= 1.0:
            self._rate = max(0.0, (self._done - self._rate_v) / dt)
            self._rate_t, self._rate_v = now, self._done
        return Snapshot(
            name=self.name, unit=self.unit, owner=self.owner,
            done=self._done, total=self._total,
            rate_per_s=self._rate, age_s=max(0.0, now - self._advanced),
            window_s=window_for(self.name), elapsed_s=max(0.0, now - self._started),
        )


def counter(
    name: str, unit: str, total: float = 0.0, *, owner: str = "",
) -> Counter:
    """Register-or-get the open counter `name` within `owner` (idempotent)."""
    key = (owner, name)
    with _lock:
        existing = _counters.get(key)
        if existing is not None:
            if total > 0:
                existing._total = float(total)
            return existing
        c = Counter(name, unit, total, owner)
        _counters[key] = c
        return c


class tracking:
    """Context manager: register on enter, finish on exit."""

    def __init__(
        self, name: str, unit: str, total: float = 0.0, *, owner: str = "",
    ) -> None:
        self._args = (name, unit, total)
        self._owner = owner

    def __enter__(self) -> Counter:
        self._counter = counter(*self._args, owner=self._owner)
        return self._counter

    def __exit__(self, *exc: object) -> None:
        self._counter.finish()


def snapshot(owner: Optional[str] = None) -> List[Snapshot]:
    """Every open counter, or only ``owner``'s when one is named."""
    now = _now()
    with _lock:
        return [
            c._snapshot_locked(now) for c in _counters.values()
            if owner is None or c.owner == owner
        ]


def freshest(owner: Optional[str] = None) -> Optional[Snapshot]:
    """The most recently advanced open counter."""
    snaps = snapshot(owner)
    return min(snaps, key=lambda s: s.age_s) if snaps else None


def self_diagnosis(owner: Optional[str] = None) -> Optional[Snapshot]:
    """Non-None when even the FRESHEST open counter is stale past its own window — the typed self_stalled confession the beat reports so the hub kills on fact, not inference."""
    fresh = freshest(owner)
    if fresh is not None and fresh.age_s > fresh.window_s:
        return fresh
    return None


def reset() -> None:
    """Test hook: drop all open counters."""
    with _lock:
        _counters.clear()
