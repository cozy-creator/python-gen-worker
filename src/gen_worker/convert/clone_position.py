"""The clone's DECLARED POSITION — one monotone integer the hub differences.

Hub job liveness is position ADVANCE inside a phase budget: `AdvanceJobProgress`
takes an int64 and accepts it only on a STRICT increase. So a phase that reports
a 0..1 fraction (`int(0.02) == 0`) or reports nothing at all is indistinguishable
from a wedged one the moment its budget expires, whatever it is really doing.

The unit is MOVEMENT: one per MiB of bytes moved, one per completed remote item
(a file enumerated, a header fetched). Only real work advances it — a counter
that ticks on a clock manufactures exactly the liveness this instrument exists
to disprove — and entering a phase counts once, because entering one is work.

The position lives here rather than in a closure inside `run_clone` because that
is where it was, and an unrelated refactor deleted it and its test together
(pgw#1667): every phase of every clone went back to declaring nothing, and the
next large job died at `clone.plan 0` having moved no bytes at all.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

#: The position's byte unit. MiB, not bytes: the hub stores an int64 and an
#: operator reads it.
MIB = 1024 * 1024

#: ``(fraction, phase, position, total)`` — the position and its optional
#: ceiling, plus the 0..1 fraction the user-facing feed renders.
EmitFn = Callable[[float, str, int, Optional[int]], None]


class ClonePosition:
    """One clone job's position, monotone across every phase.

    Thread-safe: the publish leg advances it from the uploader's threads while
    the main thread is emitting phase entries.
    """

    def __init__(self, emit: EmitFn) -> None:
        self._emit = emit
        self._lock = threading.Lock()
        self._position = 0
        self._base = 0

    def enter(self, fraction: float, phase: str) -> None:
        """Enter a phase: one unit of movement, and the origin for its own units."""
        with self._lock:
            self._position += 1
            self._base = self._position
            self._emit(fraction, phase, self._position, None)

    def units(
        self, fraction: float, phase: str, done: int, total: Optional[int] = None
    ) -> None:
        """Advance to ``done`` completed work items within the current phase."""
        self._advance(fraction, phase, int(done or 0),
                      int(total) if total else None)

    def bytes_moved(
        self, fraction: float, phase: str, done: int, total: Optional[int] = None
    ) -> None:
        """Advance by the bytes this phase has actually moved."""
        self._advance(fraction, phase, int(done or 0) // MIB,
                      (int(total) // MIB) if total else None)

    def _advance(
        self, fraction: float, phase: str, done: int, total: Optional[int]
    ) -> None:
        with self._lock:
            position = self._base + max(0, done)
            if position <= self._position:
                return
            self._position = position
            self._emit(fraction, phase, position,
                       (self._base + total) if total else None)
