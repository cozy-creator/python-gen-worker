from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PHASE_PULLED = "pulled"


@dataclass(frozen=True, slots=True)
class SnapshotPullStats:
    """One snapshot pull, reduced to what it cost the wire."""

    requested_objects: int = 0
    tree_bytes: int = 0

    fetched_objects: int = 0
    fetched_bytes: int = 0

    resident_objects: int = 0
    resident_bytes: int = 0

    filled_objects: int = 0
    filled_bytes: int = 0

    late_resident_objects: int = 0

    def accounted(self) -> int:
        """Objects this record accounts for."""
        return (
            self.fetched_objects
            + self.resident_objects
            + self.filled_objects
            + self.late_resident_objects
        )

    def detail(self, *, snapshot: str, key: str, components: int) -> str:
        return (
            f"snapshot={_token(snapshot)} key={_token(key)} "
            f"components={components} "
            f"requested_objects={self.requested_objects} "
            f"tree_bytes={self.tree_bytes} "
            f"fetched_objects={self.fetched_objects} "
            f"fetched_bytes={self.fetched_bytes} "
            f"resident_objects={self.resident_objects} "
            f"resident_bytes={self.resident_bytes} "
            f"filled_objects={self.filled_objects} "
            f"filled_bytes={self.filled_bytes} "
            f"late_resident_objects={self.late_resident_objects} "
            f"accounted_objects={self.accounted()}"
        )


def _token(value: str) -> str:
    cleaned = "".join(ch for ch in str(value or "") if not ch.isspace())
    return cleaned or "-"


def emit_pull(
    stats: SnapshotPullStats,
    *,
    snapshot: str,
    key: str,
    components: int = 0,
    duration_ms: int = 0,
) -> None:
    """One completed event stating what this pull moved."""
    try:
        from . import activity as activity_mod

        activity_mod.emit_event(
            activity_mod.KIND_SNAPSHOT_PULL,
            stats.detail(snapshot=snapshot, key=key, components=components),
            phase=PHASE_PULLED,
            duration_ms=duration_ms,
        )
    except Exception:  # pragma: no cover — telemetry never fails a boot
        logger.debug("snapshot pull event failed", exc_info=True)
