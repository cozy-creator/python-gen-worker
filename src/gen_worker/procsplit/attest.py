"""Parent-side attestation of billable job metrics (pgw#763 delta 3).

th#1309: ``JobMetrics`` is the sole billing source and it is produced by the
code being billed. Under the split that code runs in the compute child, so the
child's numbers arrive at the parent as a claim — and the parent is the one
component that watched the whole job from outside it.

The rule this file implements: **the parent asserts what it can observe, and
never inflates what it cannot.**

* Wall-clock quantities (``runtime_ms``, ``queue_ms``, ``slot_held_ms``,
  ``finalize_wall_ms``) are CLAMPED to the parent's own dispatch->result
  measurement on the stream. Clamping is one-directional on purpose:
  overstatement is the billing-relevant direction for per-second pricing, and a
  child that understates its own runtime is only charging itself less. The
  parent's window is a strict superset of the handler's, so a claim outside it
  is impossible rather than merely suspicious.
* ``concurrency_at_start`` is REPLACED with the parent's in-flight count at
  dispatch. The parent dispatches every job, so this is not an estimate.
* ``rss_at_end_bytes`` is REPLACED with the parent's ``/proc`` reading of the
  child at result time — the same discipline as pgw#771's liveness evidence: a
  process is not the witness for its own resource use.

What is NOT attested here, and why it is a scoped follow-up rather than a gap
this file pretends to close: ``output_media_duration_s``, the token counts,
``output_count`` and ``peak_vram_bytes``. Measuring media seconds or token
counts parent-side would mean routing the OUTPUT through the parent, which is
precisely the "seam carries CONTROL, not DATA" invariant the multi-GPU 4x
depends on. Those stay child-reported and need hub-side plausibility bounds
(th#1309's half). This module NAMES them in the attestation record so the
divergence is visible instead of implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

# Slack on the wall-clock bound. The parent times the frame write and the frame
# read, so its window already contains the child's; the margin only absorbs
# clock granularity, never a real overstatement.
CLAMP_SLACK_MS = 250


@dataclass
class JobObservation:
    """What the parent watched, from outside the process doing the work."""

    function: str = ""
    relayed_at: float = 0.0          # monotonic, when the RunJob frame was written
    concurrency_at_relay: int = 0    # jobs this parent already had open
    accepted_at: Optional[float] = None
    divergences: List[str] = field(default_factory=list)


def attest(
    metrics: Any,
    obs: JobObservation,
    *,
    now: float,
    child_rss_bytes: int = 0,
    status_ok: bool = False,
) -> List[str]:
    """Rewrite ``metrics`` in place to the parent's observations.

    Returns the divergences worth reporting — an empty list means the child's
    self-report agreed with what the parent watched happen.
    """
    found: List[str] = []
    # `if obs.relayed_at` — never `if observed_wall_ms`: a job the parent
    # watched take under a millisecond measures 0, and treating 0 as "no
    # observation" would exempt the fastest jobs, which are exactly the ones a
    # forged runtime is most profitable on.
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
        # Only worth naming when the child's claim is not merely stale.
        if claimed_rss and abs(claimed_rss - child_rss_bytes) > (claimed_rss // 4):
            found.append(
                f"rss_at_end_bytes={claimed_rss} replaced with the parent's "
                f"/proc reading {child_rss_bytes}"
            )
        metrics.rss_at_end_bytes = int(child_rss_bytes)

    # Named, not enforced: the parent cannot see the output without pulling the
    # data plane through its own interpreter. th#1309 owns the hub-side bound.
    if status_ok:
        if float(getattr(metrics, "output_media_duration_s", 0.0) or 0.0) <= 0.0 \
                and int(getattr(metrics, "output_count", 0) or 0) > 0:
            found.append(
                "output_media_duration_s=0 on a successful job with "
                f"output_count={int(metrics.output_count)} — the child-reported "
                "billing quantity the parent cannot corroborate (th#1309)"
            )
    obs.divergences = found
    return found


__all__ = ["CLAMP_SLACK_MS", "JobObservation", "attest"]
