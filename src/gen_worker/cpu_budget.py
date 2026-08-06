"""Per-group host-CPU budget (pgw#782).

torch sizes its intra-op thread pool from the HOST's logical processors. It
knows nothing about (a) the container's cpu quota or (b) how many execution
groups share this process. On the measured 4xA40 pod that is FOUR 48-thread
teams (from 96 host processors) against a 32.3-core quota — and the boot
window really does get CFS-throttled under it (``nr_throttled`` 4,
``throttled_usec`` 9.5 s in a 146-period window, before a single request).

WHAT THIS DOES **NOT** FIX, measured (samples/dpfix, sdxl 1024^2/28 steps,
4xA40, width-4 burst): 37.1 s with torch's 48-thread default vs 36.7 s pinned
to 8 — within noise. The width-4 collapse is the shared interpreter (pgw#783),
not thread oversubscription: at width 4 the process uses 2.3 of its 32.3
allowed cores, so CPU was never the scarce resource on THIS pod. An earlier
run appeared to show 74.0 s -> 42.3 s from this change; that was the probe's
own per-thread /proc sampler getting cheaper as the thread count fell, not the
workload getting faster. The claim is retracted.

It is kept because 192 threads on a 32.3-core quota is a misconfiguration on
its own terms, it is de-escalation-only (so it can never slow anything down),
and the pods where it WOULD bite are the narrow ones — a 4-GPU pod with 8
vCPUs gets 192 threads on 8 cores, and the CPU-heavy tails (x264, image
post-processing, hashing) are exactly where that lands.

Three rules:

* The **allowance is a fact**, read from the cgroup quota / affinity mask /
  cpu count — never from ``/proc/cpuinfo`` alone, which is the host's and is
  what torch already gets wrong.
* The **divisor is the delivered topology's group count** — the number of
  requests this process runs at once.
* The imposition is **CODE**, never an env var: pgw#718 scrubs ``OMP_*`` and
  ``MKL_*`` wholesale precisely so that no base image and no operator decides
  this, and it then left the decision to a library default reading the wrong
  CPU count. This closes that gap.

De-escalation only: the imposed value is ``min(torch's own default, budget)``,
so a pod whose allowance already exceeds torch's default is untouched and a
single-group pod can only ever have oversubscription removed, never gain
threads.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
from .procsplit import host_siblings

logger = logging.getLogger(__name__)

_CGROUP_V2 = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def cgroup_cpu_quota() -> Optional[float]:
    """CPU cores this cgroup may use, or None when unlimited/unreadable."""
    try:
        raw = Path(_CGROUP_V2).read_text().split()
        if raw and raw[0] != "max":
            return int(raw[0]) / int(raw[1])
        if raw:
            return None
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path(_CGROUP_V1_QUOTA).read_text().strip())
        period = int(Path(_CGROUP_V1_PERIOD).read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def cpu_allowance() -> float:
    """The narrowest true bound on this process's CPU: cgroup quota,
    affinity mask, and cpu_count — whichever is smallest."""
    bounds = []
    quota = cgroup_cpu_quota()
    if quota and quota > 0:
        bounds.append(float(quota))
    try:
        bounds.append(float(len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):
        pass
    count = os.cpu_count()
    if count:
        bounds.append(float(count))
    return min(bounds) if bounds else 1.0


def per_group_threads(allowance: float, groups: int) -> int:
    """Intra-op threads one execution group may use. Floor of 1: a group
    always gets a thread, even on a fractional-core pod."""
    return max(1, int(math.floor(allowance / max(1, int(groups)))))


def impose_intra_op_threads(groups: int) -> Dict[str, Any]:
    """Size torch's intra-op (and, when still settable, inter-op) pools for
    ONE group's share of this process. Returns the recorded facts; never
    raises — a host that will not answer keeps torch's default.

    pgw#783: the divisor is the number of execution groups sharing this CPU
    cgroup, which is ``groups`` (this process's own) TIMES the compute-child
    sibling count. Under the process split each child rewrites its own
    ``WORKER_EXECUTION_TOPOLOGY`` to a single local group, so ``groups`` reads 1
    in every child — but G of them share one cgroup, and without the sibling
    multiplier each would claim the whole allowance and reinstate exactly the
    192-threads-on-32-cores oversubscription pgw#782 removed. ``host_siblings()``
    is 1 for every pod that is not running the split, so this is unchanged for
    them (byte-identical at G=1)."""

    siblings = host_siblings()
    effective = max(1, int(groups)) * siblings
    facts: Dict[str, Any] = {"execution_groups": int(groups), "host_siblings": siblings,
                             "concurrency": effective}
    try:
        import torch
    except Exception:  # noqa: BLE001  (torch-free contexts: tools, tests)
        facts["skipped"] = "torch unavailable"
        return facts
    allowance = cpu_allowance()
    budget = per_group_threads(allowance, effective)
    default = int(torch.get_num_threads())
    imposed = min(default, budget)
    facts.update(allowance=round(allowance, 2), budget=budget,
                 torch_default=default, imposed=imposed)
    if imposed == default:
        # Already at or under budget — nothing to de-escalate.
        return facts
    try:
        torch.set_num_threads(imposed)
        facts["intra_op"] = int(torch.get_num_threads())
    except Exception as exc:  # noqa: BLE001
        facts["intra_op_error"] = str(exc)[:200]
    try:
        # Only settable before the inter-op pool is first used; a pod that
        # already launched work keeps its pool and says so.
        torch.set_num_interop_threads(imposed)
        facts["inter_op"] = int(torch.get_num_interop_threads())
    except Exception as exc:  # noqa: BLE001
        facts["inter_op_error"] = str(exc)[:120]
    logger.info(
        "pgw#782 cpu budget: %d local group(s) x %d sibling child(ren) = %d "
        "sharing a %.2f-core allowance -> intra-op threads %d (torch default "
        "was %d)",
        facts["execution_groups"], siblings, effective, allowance, imposed, default)
    return facts


__all__ = [
    "cgroup_cpu_quota",
    "cpu_allowance",
    "impose_intra_op_threads",
    "per_group_threads",
]
