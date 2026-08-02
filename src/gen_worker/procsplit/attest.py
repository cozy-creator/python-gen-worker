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

from dataclasses import dataclass
from typing import Any, List

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

    # th#1364: the `output_media_duration_s == 0` divergence is DELETED, and
    # deleting it is the fix rather than relabelling what it emitted.
    #
    # `_scan_output_assets` sums `Asset.duration_s`, which only a TEMPORAL asset
    # has. A still image legitimately reports 0.0, so the check fired on every
    # successful image job — the overwhelming majority of the fleet's work — and
    # it could never have done otherwise, because the parent has no way to tell
    # "correctly zero for an image" from "wrongly zero for a video". Its own
    # comment said as much: the parent cannot see the output without pulling the
    # data plane through its own interpreter.
    #
    # It was not merely noisy. Each divergence dials the post-mortem carrier,
    # which opens a SEPARATE Connect; at the time this was found that carrier
    # authenticated with `settings.worker_jwt` — the boot token, which rotation
    # never updated — so past pod-create + TTL every dial was
    # `worker_token_expired`, and three of them terminate the pod
    # (`worker_auth_wedge`). pgw#846 attempt sixteen died exactly there, 35
    # minutes into the longest AOT mint in the program's history. **A false
    # positive in a billing attestation became the proximate cause of a pod
    # death.** The carrier now reads `worker_credential.current()` (pgw#848
    # `7fa4eeb`) and a diagnostic dial can no longer condemn a pod (th#1359
    # `705c316a`); this removes the false positive that fired it. Three
    # independent layers — keep all three.
    #
    # th#1309 keeps the property, on the side that can actually hold it: the HUB
    # knows the endpoint's settlement model, so it alone can say whether a zero
    # duration is meaningful (`per_output_second` — fail closed) or expected
    # (`per_output` on images). Re-adding a worker-side version of this check
    # needs a temporal-asset count on the wire, not a heuristic over `output_count`.
    return found


__all__ = ["CLAMP_SLACK_MS", "JobObservation", "attest"]
