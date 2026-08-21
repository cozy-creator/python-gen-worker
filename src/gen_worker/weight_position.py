"""The BYTE POSITION of a serving-path weight download, as a wire fact. NO CLOCKS: the judgment every consumer needs is "is the position advancing" — a POSITION is emitted and the reader differences it (the hub advances on strict increase). Integral MiB, never a fraction: the hub parses an INTEGER off the position field (ActivityUpdate.step), so a position in GiB truncates int(0.97)==0 for the first ~1 GiB and a healthy multi-GB transfer reads as the wedge it is not — the position is bytes // MiB. It means bytes ACCOUNTED for the snapshot (residency scan + wire), not bytes off the wire, so a warm scan does not sit at 0 for minutes. Positions ride emit_event rows, never a RUNNING activity — a download must never satisfy a liveness rule by existing — and `started` is emitted unconditionally at position 0, so a transfer that wedges before its first MiB still renders as a row that says zero."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Iterator, Optional

from . import byte_sources

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

PHASE_STARTED = "started"
PHASE_FETCHING = "fetching"
PHASE_FETCHED = "fetched"
PHASE_ABANDONED = "abandoned"
PHASE_ALREADY_RESIDENT = "already_resident"

STRIDE_MIB = 256
MIN_INTERVAL_S = 60.0


def _token(value: str) -> str:
    cleaned = "".join(ch for ch in str(value or "") if not ch.isspace())
    return cleaned or "-"


class FetchPosition:
    """The position reporter for ONE ref's weight fetch."""

    #: pgw#1632 — HOW this number is measured, declared at the site that
    #: produces it. `CAS_ACCOUNTED`, not a wire meter: the position credits
    #: bytes that are PRESENT for the manifest whether they came off a socket,
    #: off the endpoint volume, or were already banked (see the module
    #: docstring's "what the position MEANS"). No read verb and no write verb
    #: can say that truthfully, which is why the source admits neither.
    SOURCE = byte_sources.Source.CAS_ACCOUNTED

    def __init__(self, ref: str, total_bytes: int = 0) -> None:
        self.ref = str(ref or "")
        self._total_bytes = max(0, int(total_bytes or 0))
        self._pos_bytes = 0
        self._emitted_mib = -1
        self._emitted_at = 0.0
        self._opened = False
        self._closed = False
        self._opened_at = 0.0

    @property
    def position_mib(self) -> int:
        """Integral MiB accounted for."""
        return self._pos_bytes // MIB

    @property
    def total_mib(self) -> int:
        return self._total_bytes // MIB

    def open(self) -> None:
        """State position 0 before a byte moves."""
        if self._opened:
            return
        self._opened = True
        self._opened_at = time.monotonic()
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
        if pos <= self._emitted_mib:
            return
        now = time.monotonic()
        if (pos - self._emitted_mib) < STRIDE_MIB and (
            now - self._emitted_at
        ) < MIN_INTERVAL_S:
            return
        self._emit(PHASE_FETCHING)

    def close(self, ok: bool = True, *, resident: bool = False) -> None:
        """The terminal position."""
        if self._closed:
            return
        self._closed = True
        if ok and resident:
            self._emit(PHASE_ALREADY_RESIDENT)
            return
        self._emit(PHASE_FETCHED if ok else PHASE_ABANDONED)

    def already_resident(self) -> None:
        """Open AND close in one row: the resolver found the ref resident, so there is no transfer to report positions for."""
        if self._closed:
            return
        self._opened = True
        self._closed = True
        self._emit(PHASE_ALREADY_RESIDENT)

    def _terminal_detail(self) -> str:
        if not self._opened_at:
            return ""
        elapsed = max(0.0, time.monotonic() - self._opened_at)
        out = f" elapsed_ms={int(elapsed * 1000)}"
        if elapsed > 0 and self._pos_bytes > 0:
            out += f" mib_s={(self._pos_bytes / MIB) / elapsed:.1f}"
        return out

    def _emit(self, phase: str) -> None:
        pos = self.position_mib
        self._emitted_mib = pos
        self._emitted_at = time.monotonic()
        detail = (
            f"ref={_token(self.ref)} pos_mib={pos} total_mib={self.total_mib} "
            f"pos_bytes={self._pos_bytes} total_bytes={self._total_bytes}"
        )
        if phase in (PHASE_FETCHED, PHASE_ABANDONED):
            detail += self._terminal_detail()
        try:
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


def snapshot_bytes(snapshot: Any) -> int:
    """Total declared bytes of a resolved snapshot (0 when unknown)."""
    files = getattr(snapshot, "files", None) or ()
    try:
        return sum(int(getattr(f, "size_bytes", 0) or 0) for f in files)
    except (TypeError, ValueError):
        return 0


@contextlib.contextmanager
def track(ref: str, total_bytes: int = 0) -> Iterator[FetchPosition]:
    """Open a position, hand it over, and CLOSE IT ON EVERY EXIT PATH."""
    position = FetchPosition(ref, total_bytes=total_bytes)
    position.open()
    try:
        yield position
    except BaseException:
        position.close(ok=False)
        raise
    position.close(ok=True)
