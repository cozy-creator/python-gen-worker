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
from typing import Dict, List, Optional, Tuple

UNIT_BYTES = "bytes"
UNIT_STEPS = "steps"
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

#: ``(owner, name) -> Counter``. pgw#894: the key used to be the NAME alone, so
#: one process-global namespace held every phase's counters and `freshest()`
#: returned whichever of them advanced most recently — regardless of which work
#: it described. A serving request's `infer:steps` therefore refreshed a
#: background mint's stall clock: measured on the standing chaos hub, 28 log
#: lines reporting `infer:steps` under `self_mint_compile`, one of which
#: declined a condemnation because that mint activity was "0s ago".
#:
#: The OWNER is the scope the counter belongs to (an activity id, a request
#: id). Registry-wide queries still exist and still mean what they meant — "is
#: this process doing anything at all" — but a scope's stall verdict is now
#: computed from that scope's own counters.
_counters: Dict[Tuple[str, str], "Counter"] = {}


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
    #: The scope that owns this counter ("" = unowned/process-wide). LAST and
    #: defaulted on purpose: a Snapshot is constructed by hand in tests and by
    #: readers that do not care whose counter it is, and a new field must be
    #: additive rather than a positional break (pgw#894).
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
    """Register-or-get the open counter `name` within `owner` (idempotent).

    ``owner`` scopes the counter to the work it describes (pgw#894). Two
    scopes may use the same NAME — two concurrent requests both counting
    ``infer:steps`` is the ordinary case — and neither can advance the
    other's clock.
    """
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
    """The most recently advanced open counter.

    ``owner=None`` is the PROCESS liveness view and is unchanged: any
    advancing counter proves the process is doing real work, which is exactly
    what a "did this pod wedge" question wants.

    ``owner="..."`` is the SCOPE view, and it is the one a stall verdict must
    use (pgw#894). A mint asking "am I still advancing" must not be answered
    by a request that happens to be running beside it.
    """
    snaps = snapshot(owner)
    return min(snaps, key=lambda s: s.age_s) if snaps else None


def self_diagnosis(owner: Optional[str] = None) -> Optional[Snapshot]:
    """Non-None when even the FRESHEST open counter is stale past its own
    window — the typed self_stalled confession the beat reports so the hub
    kills on fact, not inference.

    Scoped to ``owner`` when one is named, registry-wide otherwise. pgw#894:
    registry-wide was the ONLY form, and it is the right answer to "is this
    process alive" and the wrong one to "is this mint stalled" — a request
    running beside a wedged mint answered for it. `Activity.on_beat` passes
    the activity's own id.

    Counter LIFETIME still has to be honest within a scope: a counter left
    open after its producer's phase ended is the min-age counter of a phase it
    knows nothing about, and confesses for it. `Activity.counter()` scopes
    them to the phase for that reason (pgw#962)."""
    fresh = freshest(owner)
    if fresh is not None and fresh.age_s > fresh.window_s:
        return fresh
    return None


def reset() -> None:
    """Test hook: drop all open counters."""
    with _lock:
        _counters.clear()
