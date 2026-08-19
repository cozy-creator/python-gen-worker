"""pgw#1455: the BYTE POSITION of a serving-path weight download, as a wire fact.

Rental #6 (e2e#1910) parked a request on `model_download_pending` for 49 minutes
and ~$0.59, and nothing anywhere could say whether the download was finishing.
The scheduler's `attempts=2963` counts its own 1 Hz placement poll and tracks
wall-clock seconds 1:1 — it reads the same whether a download is healthy, slow
or wedged, so it carries no download information at all. `worker_activity_events`
held 0 rows for the pod. The honest state was "nobody can tell".

**No clocks.** The judgment a consumer, an operator and the scheduler all need is
*is the position advancing* — not *how long has it been*. So this emits a
POSITION and lets the reader difference it, exactly as [[pgw#1397]] did for the
clone path (the hub advances on strict increase).

## Integral MiB, never a fraction — pgw#1397's documented trap

The hub parses an INTEGER off the position field (`ActivityUpdate.step` ->
`worker_activity_events.step`). A position expressed in GiB spends the first
1073741823 bytes truncating to `int(0.97) == 0`, so a healthy multi-GB transfer
reports a frozen position and reads as the wedge it is not. The position here is
`bytes // MiB` — already an integer, monotone, and one unit of it is small
enough that a live transfer moves it within a second.

## What the position MEANS, and why it is the aggregate

`bytes accounted for the snapshot`, not `bytes off the wire`. The fetch loop
(`models/cozy_snapshot._ensure_objects`) runs a residency scan first — ~1 s/GiB
warm, ~14 s/GiB cold-and-contended — during which no wire byte moves and a
wire-only position would sit at 0 for minutes, which is the same lie #1397 fixed
(`progress_position = 0` for the whole fetch). The consumer-side question is
"how much of this model does the pod have", and the aggregate answers it through
both phases.

## Rows, not a beat

Positions ride `emit_event` (a self-contained COMPLETED event) rather than a
running activity, for `snapshot_pull`'s reason: an extra RUNNING activity would
put "a message arrived recently" into the hub's serving-blocked and stall
predicates, and a download must never be able to satisfy a liveness rule by
existing. Cadence is bounded by BOTH a byte stride and a time floor, so a fast
7 GB pull writes ~28 rows and a slow one writes about one a minute.

`started` is emitted unconditionally, at position 0. Without it a transfer that
wedges before its first MiB emits nothing, and "no rows" is the exact reading
that cost rental #6 — absence has to render as a row that says zero.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

#: The fetch opened; position 0. Always emitted, so a wedge at zero is a ROW.
PHASE_STARTED = "started"
#: One position, mid-transfer.
PHASE_FETCHING = "fetching"
#: The fetch returned; position is what landed.
PHASE_FETCHED = "fetched"
#: The fetch raised or was cancelled; position is where it stopped.
PHASE_ABANDONED = "abandoned"
#: pgw#1485: the ref was ALREADY RESIDENT — nothing was transferred and no
#: `model_download` record was opened hub-side. A distinct word, because
#: "fetched at position 0" and "there was nothing to fetch" are different facts
#: and th#2204 cost a rental to the first being unsayable.
PHASE_ALREADY_RESIDENT = "already_resident"

#: Emit at most one row per this many MiB of advance...
STRIDE_MIB = 256
#: ...and at least one per this many seconds while the position moves. Matched
#: to the hub's own `workerActivityProgressEventInterval` (1 min), which is the
#: cadence at which it makes an in-phase position durable.
MIN_INTERVAL_S = 60.0


def _token(value: str) -> str:
    """A `k=v` value with no spaces in it, and never empty — the grammar the
    th#1839 route serves verbatim and every reader parses as `(\\w+)=(\\S+)`."""
    cleaned = "".join(ch for ch in str(value or "") if not ch.isspace())
    return cleaned or "-"


class FetchPosition:
    """The position reporter for ONE ref's weight fetch.

    ``progress(done, total)`` is deliberately the ``ProgressFn`` shape the
    download layer already calls, so the reporter is wired by handing this
    method to the same callback chain rather than by threading a second one.
    """

    def __init__(self, ref: str, total_bytes: int = 0) -> None:
        self.ref = str(ref or "")
        self._total_bytes = max(0, int(total_bytes or 0))
        self._pos_bytes = 0
        #: Last position PUT ON THE WIRE, in MiB. Rows are emitted on strict
        #: increase only, which is what makes the hub's #1243 rule applicable
        #: to them unchanged.
        self._emitted_mib = -1
        self._emitted_at = 0.0
        self._opened = False
        self._closed = False

    # -- the position itself ------------------------------------------------

    @property
    def position_mib(self) -> int:
        """Integral MiB accounted for. Never a fraction; see the module note."""
        return self._pos_bytes // MIB

    @property
    def total_mib(self) -> int:
        return self._total_bytes // MIB

    # -- the three call sites -----------------------------------------------

    def open(self) -> None:
        """State position 0 before a byte moves."""
        if self._opened:
            return
        self._opened = True
        self._emit(PHASE_STARTED)

    def progress(self, done: int, total: Optional[int] = None) -> None:
        """One ``ProgressFn`` tick from the fetch loop."""
        try:
            self._pos_bytes = max(0, int(done))
            if total:
                self._total_bytes = max(self._total_bytes, int(total))
        except (TypeError, ValueError):
            return
        pos = self.position_mib
        if pos <= self._emitted_mib:  # strict increase, always
            return
        now = time.monotonic()
        if (pos - self._emitted_mib) < STRIDE_MIB and (
            now - self._emitted_at
        ) < MIN_INTERVAL_S:
            return
        self._emit(PHASE_FETCHING)

    def close(self, ok: bool = True, *, resident: bool = False) -> None:
        """The terminal position. Emitted whether or not it advanced — where
        the transfer STOPPED is the fact, and a fetch that moved less than one
        MiB has to leave a row saying so.

        ``resident=True`` says the ref was already on the pod and no transfer
        happened at all (pgw#1485)."""
        if self._closed:
            return
        self._closed = True
        if ok and resident:
            self._emit(PHASE_ALREADY_RESIDENT)
            return
        self._emit(PHASE_FETCHED if ok else PHASE_ABANDONED)

    def already_resident(self) -> None:
        """Open AND close in one row: the resolver found the ref resident, so
        there is no transfer to report positions for. One row, so the reader
        can tell "the pod already had it" from "nothing ever ran"."""
        if self._closed:
            return
        self._opened = True
        self._closed = True
        self._emit(PHASE_ALREADY_RESIDENT)

    # -- emission -----------------------------------------------------------

    def _emit(self, phase: str) -> None:
        pos = self.position_mib
        self._emitted_mib = pos
        self._emitted_at = time.monotonic()
        detail = (
            f"ref={_token(self.ref)} pos_mib={pos} total_mib={self.total_mib} "
            f"pos_bytes={self._pos_bytes} total_bytes={self._total_bytes}"
        )
        try:
            # Lazy: `activity` pulls in protobuf and psutil, and this module is
            # imported from the download path the discovery build keeps thin.
            from . import activity as activity_mod

            activity_mod.emit_event(
                activity_mod.KIND_WEIGHT_FETCH,
                detail,
                phase=phase,
                step=pos,
                total_steps=self.total_mib,
            )
        except Exception:  # pragma: no cover — telemetry never fails a fetch
            logger.debug("weight position event dropped", exc_info=True)
