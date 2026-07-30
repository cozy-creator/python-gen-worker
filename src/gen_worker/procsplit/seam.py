"""pgw#783 THE INVARIANT: the seam carries CONTROL, not DATA.

This is the constraint that decides whether pgw#782's measured 4.00x survives
the split. If G children's job DATA — tensors, images, video, chunk
reassembly, fused hashing — routes through the PARENT's single interpreter,
the GIL bottleneck is recreated one layer up and the whole exercise becomes a
rewrite of the bottleneck's address. Data goes child -> object store directly
(per-job scoped grants) or via shared memory. Never through the parent.

**A job whose result crosses the parent's interpreter is a BUG, not a slow
path.** Three enforcements:

1. STRUCTURAL, already true and to be kept true: results >64KB leave the child
   as presigned R2 PUTs and ``JobResult`` carries ``blob_ref``; inputs arrive
   as refs/URLs that ``input_assets``/``url_fetch`` resolve child-side; the
   pgw#769/pgw#781 fused hash and chunk reassembly are DATA-plane and stay in
   the child. The parent grows no upload path, no hashing, no reassembly —
   ever, as policy, not as an accident of the current code.
2. MEASURED, here: every relayed frame is accounted by type, and a relayed
   ``job_result`` above a hard control ceiling is recorded as a violation and
   dialed typed. **Reported, never refused** — refusing in production turns a
   payload quirk into a lost job, and the point of this module is that a
   regression is VISIBLE, not that it is fatal to a caller.
3. TESTED: a job producing megabytes must relay ~nothing through the parent,
   and the guard itself is red-verified against an inlined payload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from .actions import CONTROL_BODY_CEILING_BYTES

logger = logging.getLogger(__name__)

__all__ = ["SeamAccountant", "SeamViolation", "CONTROL_FRAME_CEILING_BYTES",
           "DIAL_PHASE"]

# A control message is small. 256 KiB is roomy for the largest legitimate
# JobResult (refs, metrics, safe_message, a stage_ms map) and four orders of
# magnitude below a single generated image. Shared verbatim with the delta-1
# action broker's body cap so the two seam halves cannot drift.
CONTROL_FRAME_CEILING_BYTES = CONTROL_BODY_CEILING_BYTES

DIAL_PHASE = "procsplit_data_on_control_seam"


@dataclass(frozen=True)
class SeamViolation:
    kind: str            # the WorkerMessage oneof name, e.g. "job_result"
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

    # Frame kinds whose bytes are, by design, job payload-free. These are the
    # ones the ceiling is enforced on; a residency snapshot or a lifecycle
    # projection is legitimately large and is not job data.
    _JOB_KINDS = ("job_result", "job_progress", "job_accepted")

    def record(self, kind: str, size_bytes: int, *, group: int = 0) -> bool:
        """Account one relayed message. Returns False on a violation."""
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

    # ---- observability ---------------------------------------------------

    @property
    def job_payload_bytes(self) -> int:
        """Bytes of job-shaped messages the parent has relayed. The number the
        control-not-data test asserts stays small while the child provably
        produced megabytes."""
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
