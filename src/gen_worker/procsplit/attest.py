"""Parent-side attestation of billable job metrics: JobMetrics is produced by the code being billed, so the child's numbers arrive as a CLAIM and the parent asserts what it can observe, never inflating what it cannot. Wall-clock quantities are CLAMPED one-directionally to the parent's own dispatch->result window (overstatement is the billing-relevant direction); concurrency_at_start and rss_at_end_bytes are REPLACED with the parent's own observations. output_media_duration_s, token counts, output_count and peak_vram_bytes stay child-reported — measuring them parent-side would route the OUTPUT through the parent, breaking the seam-carries-CONTROL-not-DATA invariant — and are NAMED in the attestation record so the divergence is visible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

CLAMP_SLACK_MS = 250


@dataclass
class JobObservation:
    """What the parent watched, from outside the process doing the work."""

    function: str = ""
    relayed_at: float = 0.0
    concurrency_at_relay: int = 0


def attest(
    metrics: Any,
    obs: JobObservation,
    *,
    now: float,
    child_rss_bytes: int = 0,
    status_ok: bool = False,
) -> List[str]:
    """Rewrite ``metrics`` in place to the parent's observations."""
    found: List[str] = []
    if obs.relayed_at:
        observed_wall_ms = max(0, int((now - obs.relayed_at) * 1000))
        bound = observed_wall_ms + CLAMP_SLACK_MS
        for name in ("runtime_ms", "queue_ms", "slot_held_ms", "finalize_wall_ms"):
            claimed = int(getattr(metrics, name, 0) or 0)
            if claimed > bound:
                found.append(
                    f"{name}={claimed} exceeds the parent-observed "
                    f"dispatch->result wall of {observed_wall_ms}ms; clamped"
                )
                setattr(metrics, name, bound)

    claimed_conc = int(getattr(metrics, "concurrency_at_start", 0) or 0)
    if claimed_conc != obs.concurrency_at_relay:
        found.append(
            f"concurrency_at_start={claimed_conc} replaced with the parent's "
            f"dispatch-time count {obs.concurrency_at_relay}"
        )
    metrics.concurrency_at_start = max(0, int(obs.concurrency_at_relay))

    if child_rss_bytes > 0:
        claimed_rss = int(getattr(metrics, "rss_at_end_bytes", 0) or 0)
        if claimed_rss and abs(claimed_rss - child_rss_bytes) > (claimed_rss // 4):
            found.append(
                f"rss_at_end_bytes={claimed_rss} replaced with the parent's "
                f"/proc reading {child_rss_bytes}"
            )
        metrics.rss_at_end_bytes = int(child_rss_bytes)

    return found


__all__ = ["CLAMP_SLACK_MS", "JobObservation", "attest"]
