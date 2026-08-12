"""pgw#975: declare the OOM victim order the pgw#763 split silently depends on.

The split's reporting guarantee is that the control parent OUTLIVES the compute
child it supervises, so a kernel kill of the child is reported as ``WTERMSIG`` /
``oom_kill`` over the wire instead of vanishing with the pod. Nothing anywhere
told the kernel that. The parent survived because a bare interpreter (44.4 MiB,
397 modules) happens to be smaller than one that has imported torch (493.0 MiB,
1181) — an accident that erodes with every dependency the control plane gains,
and that is INVERTED for the 9.5-11.7s (pgw#833, measured on a hub-launched pod)
a freshly spawned child spends at 43.0 MiB before it reaches ``import torch``.

The compute child raises its OWN score. Direction is load-bearing: raising is
unprivileged, lowering needs ``CAP_SYS_RESOURCE`` and returns ``EACCES`` in a
hardened container (``__set_oom_adj``, ``fs/proc/base.c``). Placement follows
``aot_compile_pool.arm_parent_death_signal``: the child does it to itself, not
through the parent's ``preexec_fn`` — that hook is async-signal-unsafe territory,
forces ``fork()`` over ``posix_spawn()``, and pgw#932 is removing it.

The mint child (pgw#784) and the AOT pool's entry children are descendants of a
compute child, and ``oom_score_adj`` is inherited across fork and preserved
across exec — so the whole compute subtree is covered by this one call. There is
deliberately no second knob for them (§4.24 item 1): the fattest process is
already the one the kernel picks, and every death below the parent is
recoverable.

Nothing here imports grpc or torch: this runs at the very top of the compute
child, so a child that dies during its own imports is already ranked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..postmortem import cgroup_nodes

logger = logging.getLogger(__name__)

__all__ = [
    "DEGRADE_PHASE",
    "OomRank",
    "oom_domain_bytes",
    "parent_ceiling_bytes",
    "score_adj_delta_for_domain",
    "raise_own_oom_score_adj",
]

#: Stable token for the degradation. Grep-able in a pod log, and the child's
#: stderr is teed to the container log AND ring-buffered into the parent's death
#: dial by the pgw#833 pump, so an ERROR here already reaches the wire.
DEGRADE_PHASE = "procsplit_oom_rank_unset"

#: The control parent with every control module imported, measured (44.4 MiB /
#: 397 modules, CPython 3.12) and rounded up.
_PARENT_RESIDENT_BYTES = 48 * 1024 * 1024

#: The only thing in the parent that grows with LOAD: ``transport.RESHIP_WINDOW``
#: (256) frames, each capped by ``seam.CONTROL_FRAME_CEILING_BYTES`` (256 KiB).
#: Written out rather than imported because this module must not pull grpc into
#: the child's first milliseconds; `test_pgw975_*` fails if either drifts. (The
#: per-group stderr ring is 32 KiB x G — four orders down, ignored.)
_PARENT_BUFFER_BYTES = 256 * 256 * 1024

#: The seam ceiling is enforced REPORTED-NOT-REFUSED (``seam.py``), so the parent
#: may legitimately exceed its declared budget transiently. One full over-run is
#: covered; more would volunteer more of a shared host than the guarantee needs.
_OVERRUN_ALLOWANCE = 2

#: Tightest memory domain this worker has ever been observed running in (the
#: ``ram_total_gb: 14.9`` container of the 0.56.2 report). Used only when /proc
#: is unreadable, so an unknown domain degrades TOWARD protecting the reporter.
_TIGHTEST_OBSERVED_DOMAIN_BYTES = 14 * 1024 ** 3

_SELF_OOM_SCORE_ADJ = Path("/proc/self/oom_score_adj")
_PROC_MEMINFO = Path("/proc/meminfo")

#: Kernel range for ``oom_score_adj`` (``fs/proc/base.c``). Not ours to pick.
_OOM_SCORE_ADJ_MIN = -1000
_OOM_SCORE_ADJ_MAX = 1000


@dataclass(frozen=True)
class OomRank:
    """What this process asked the kernel for, and what it got.

    ``applied`` false is a DEGRADATION, not a detail: ``unprotected`` names the
    guarantee that has stopped holding.
    """

    applied: bool
    value: int
    previous: Optional[int]
    domain_bytes: int
    ceiling_bytes: int
    reason: str = ""
    unprotected: str = ""

    def format(self) -> str:
        return (
            f"phase={DEGRADE_PHASE} value={self.value} previous={self.previous} "
            f"domain_bytes={self.domain_bytes} ceiling_bytes={self.ceiling_bytes} "
            f"reason={self.reason} — {self.unprotected}"
        )


def parent_ceiling_bytes() -> int:
    """Every byte the control parent can hold: measured resident set + the one
    buffer that grows with load."""
    return _PARENT_RESIDENT_BYTES + _PARENT_BUFFER_BYTES


def _read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw in ("", "max"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_memory_max() -> Optional[int]:
    """Tightest finite ``memory.max`` on this process's cgroup chain.

    None on both RunPod shapes measured live 2026-07-30 (CPU pod, 4090 SECURE):
    the provider sets no ceiling, so the OOM domain is the whole shared host and
    the adjudicator is the host killer — which is precisely why victim selection
    has to be declared rather than delegated to a cgroup that does not exist.
    """
    best: Optional[int] = None
    for node in cgroup_nodes():
        value = _read_int(node / "memory.max")
        if value is not None:
            best = value if best is None else min(best, value)
    return best


def _meminfo_total_bytes() -> Optional[int]:
    try:
        for line in _PROC_MEMINFO.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def oom_domain_bytes() -> int:
    """The memory the kernel scores badness against: the cgroup ceiling when one
    exists, otherwise the host. 0 when neither is readable."""
    limit = _cgroup_memory_max()
    total = _meminfo_total_bytes()
    if limit is not None and total is not None:
        return min(limit, total)
    return limit or total or 0


def score_adj_delta_for_domain(domain_bytes: int, ceiling_bytes: int) -> int:
    """The GAP over the inherited baseline: the parent's whole footprint, twice
    over, in the kernel's own units.

    A delta, not an absolute, because ``oom_score_adj`` is inherited: whatever
    the child reads at boot IS the control parent's own value. An absolute set
    creates no gap at all whenever the ambient baseline already exceeds it —
    measured, not hypothesised: on a desktop session inheriting 200 from
    ``gnome-terminal`` an absolute 6 was a no-op and parent and child ranked
    identically. Which baseline a container starts from is not ours to assume,
    and assuming it is the failure this whole issue is about.

    THREAT: a control parent that outweighs the compute child is the one the OOM
    killer picks, and then the death of the process whose ONLY job is to report
    the child's death goes unreported — the pod dies mute. Nothing else prevents
    it: the guards upstream (host_move_guard, the process caps) bound
    allocation, not victim selection, and RunPod sets no cgroup ceiling to fire
    first. Sized to the parent and no larger, because that OOM domain is a
    SHARED host: a bigger number volunteers our paying child ahead of a
    co-tenant's leak, which spares the actual hog and costs us the job.

    ``oom_badness()`` scales the adjustment by ``totalpages / 1000``, so one
    point is 0.1% of the domain: +1 on a 755 GiB host, +2 on a 125 GiB host,
    +15 on a 14.9 GiB container.
    """
    if domain_bytes <= 0:
        domain_bytes = _TIGHTEST_OBSERVED_DOMAIN_BYTES
    raw = math.ceil(1000 * _OVERRUN_ALLOWANCE * ceiling_bytes / domain_bytes)
    return max(1, min(_OOM_SCORE_ADJ_MAX, raw))


_UNPROTECTED = (
    "the pgw#763 control parent is no longer preferentially spared. A kernel "
    "OOM may take the REPORTER instead of the compute child, and that death is "
    "then reported only by the next boot's leftover post-mortem record — if the "
    "container restarts at all"
)


def raise_own_oom_score_adj() -> OomRank:
    """Make THIS process a more attractive OOM victim than the control parent.

    Only ever raises, by construction: the delta is >= 1 and it is added to the
    value we inherited. Lowering would need ``CAP_SYS_RESOURCE`` we do not have.
    """
    domain = oom_domain_bytes()
    ceiling = parent_ceiling_bytes()
    delta = score_adj_delta_for_domain(domain, ceiling)
    previous = _read_int(_SELF_OOM_SCORE_ADJ)
    # An unreadable baseline is not a reason to skip: the delta alone is still a
    # raise over the kernel's own default of 0.
    want = min(_OOM_SCORE_ADJ_MAX, (previous or 0) + delta)
    if previous is not None and want <= previous:
        # Only reachable at the ceiling: the parent is ALREADY maximally
        # killable, so no child can be ranked above it.
        rank = OomRank(
            False, want, previous, domain, ceiling,
            reason="baseline_at_kernel_maximum", unprotected=_UNPROTECTED,
        )
        logger.error("procsplit oom rank: %s", rank.format())
        return rank
    try:
        _SELF_OOM_SCORE_ADJ.write_text(f"{want}\n")
    except OSError as exc:
        rank = OomRank(
            False, want, previous, domain, ceiling,
            reason=f"{type(exc).__name__}:errno={exc.errno}",
            unprotected=_UNPROTECTED,
        )
        logger.error("procsplit oom rank: %s", rank.format())
        return rank
    logger.info(
        "procsplit oom rank: compute child oom_score_adj %s -> %d (+%d) "
        "(domain=%.2f GiB, parent ceiling=%.0f MiB); descendants inherit",
        "unreadable" if previous is None else previous, want, delta,
        domain / 1024 ** 3, ceiling / 1024 ** 2,
    )
    return OomRank(True, want, previous, domain, ceiling, reason="set")
