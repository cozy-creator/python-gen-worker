"""Declare the OOM victim order the process split silently depends on."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import hostfacts
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

DEGRADE_PHASE = "procsplit_oom_rank_unset"

_PARENT_RESIDENT_BYTES = 48 * 1024 * 1024

_PARENT_BUFFER_BYTES = 256 * 256 * 1024

_OVERRUN_ALLOWANCE = 2

_TIGHTEST_OBSERVED_DOMAIN_BYTES = 14 * 1024 ** 3

_SELF_OOM_SCORE_ADJ = Path("/proc/self/oom_score_adj")

_OOM_SCORE_ADJ_MIN = -1000
_OOM_SCORE_ADJ_MAX = 1000


@dataclass(frozen=True)
class OomRank:
    """What this process asked the kernel for, and what it got."""

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
    """Every byte the control parent can hold: measured resident set + the one buffer that grows with load."""
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
    best: Optional[int] = None
    for node in cgroup_nodes():
        value = _read_int(node / "memory.max")
        if value is not None:
            best = value if best is None else min(best, value)
    return best


def _meminfo_total_bytes() -> Optional[int]:
    total = hostfacts.meminfo_kb().get("MemTotal")
    return total * 1024 if total else None


def oom_domain_bytes() -> int:
    """The memory the kernel scores badness against: the cgroup ceiling when one exists, otherwise the host."""
    limit = _cgroup_memory_max()
    total = _meminfo_total_bytes()
    if limit is not None and total is not None:
        return min(limit, total)
    return limit or total or 0


def score_adj_delta_for_domain(domain_bytes: int, ceiling_bytes: int) -> int:
    """The GAP over the inherited baseline: the parent's whole footprint, twice over, in the kernel's own units."""
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
    """Make THIS process a more attractive OOM victim than the control parent."""
    domain = oom_domain_bytes()
    ceiling = parent_ceiling_bytes()
    delta = score_adj_delta_for_domain(domain, ceiling)
    previous = _read_int(_SELF_OOM_SCORE_ADJ)
    want = min(_OOM_SCORE_ADJ_MAX, (previous or 0) + delta)
    if previous is not None and want <= previous:
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
