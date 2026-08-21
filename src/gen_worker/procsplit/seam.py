"""THE INVARIANT: the seam carries CONTROL, not DATA."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from .actions import CONTROL_BODY_CEILING_BYTES

logger = logging.getLogger(__name__)

__all__ = ["SeamAccountant", "SeamViolation", "CONTROL_FRAME_CEILING_BYTES",
           "DIAL_PHASE"]

CONTROL_FRAME_CEILING_BYTES = CONTROL_BODY_CEILING_BYTES

DIAL_PHASE = "procsplit_data_on_control_seam"


@dataclass(frozen=True)
class SeamViolation:
    kind: str
    size_bytes: int
    group: int
    detail: str

    def format(self) -> str:
        return (
            f"phase={DIAL_PHASE} kind={self.kind} bytes={self.size_bytes} "
            f"group={self.group} ceiling={CONTROL_FRAME_CEILING_BYTES} "
            f"— {self.detail}"
        )


@dataclass
class SeamAccountant:
    """Per-worker byte accounting across the parent<->child seam."""

    ceiling_bytes: int = CONTROL_FRAME_CEILING_BYTES
    bytes_by_kind: Dict[str, int] = field(default_factory=dict)
    frames_by_kind: Dict[str, int] = field(default_factory=dict)
    violations: List[SeamViolation] = field(default_factory=list)

    _JOB_KINDS = ("job_result", "job_progress", "job_accepted")

    def record(self, kind: str, size_bytes: int, *, group: int = 0) -> bool:
        """Account one relayed message."""
        self.bytes_by_kind[kind] = self.bytes_by_kind.get(kind, 0) + size_bytes
        self.frames_by_kind[kind] = self.frames_by_kind.get(kind, 0) + 1
        if kind not in self._JOB_KINDS or size_bytes <= self.ceiling_bytes:
            return True
        violation = SeamViolation(
            kind=kind,
            size_bytes=size_bytes,
            group=group,
            detail=(
                "job DATA crossed the control parent's interpreter. Results "
                ">64KB must leave the CHILD as presigned object-store PUTs "
                "with JobResult carrying blob_ref; routing them through the "
                "parent recreates the GIL bottleneck pgw#782 measured one "
                "layer up and costs the 4x this split exists for"
            ),
        )
        self.violations.append(violation)
        logger.error("procsplit seam: %s", violation.format())
        return False

    @property
    def job_payload_bytes(self) -> int:
        """Bytes of job-shaped messages the parent has relayed."""
        return sum(self.bytes_by_kind.get(k, 0) for k in self._JOB_KINDS)

    @property
    def clean(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        parts = ", ".join(
            f"{k}={self.bytes_by_kind[k]}B/{self.frames_by_kind[k]}f"
            for k in sorted(self.bytes_by_kind)
        )
        return f"seam[{parts or 'idle'}] violations={len(self.violations)}"
