"""Kernel-accounted evidence that another process is doing real work.

The producer every progress-keyed bound over a LOCAL child is built from
(gw#666, §4.24): a monotonic number that rises while a process tree burns CPU
or moves bytes, and stops rising the moment it wedges. It needs no cooperation
from the process being watched, which is what makes it usable against a
follower that has no protocol of its own between spawn and ready.

One implementation, because the two callers had drifted into being one:
``procsplit.parent`` grew this to watch a compute child and
``parallel.group`` needed exactly it to replace two flat wall clocks.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["tree_evidence"]


def _cpu_seconds(proc: Any) -> float:
    t = proc.cpu_times()
    return (
        float(t.user) + float(t.system)
        + float(getattr(t, "children_user", 0.0) or 0.0)
        + float(getattr(t, "children_system", 0.0) or 0.0)
    )


def tree_evidence(pid: int) -> Optional[float]:
    """This process tree's kernel-accounted work, or ``None`` when it cannot
    be read at all.

    CPU seconds for the WHOLE tree — live AND already-reaped descendants
    (pgw#964) — plus process disk I/O MB, the same combination
    ``activity._default_evidence`` trusts, measured from ``/proc``. Either
    source advancing on its own proves life: a weights download is CPU-light
    and moves real bytes; an inductor compile burns child CPU with flat I/O;
    a true hang advances neither.

    The reaped half is not optional. A descendant's CPU moves into its
    parent's ``cutime``/``cstime`` when it is waited for, so a tree summed
    over live members only would go DOWN whenever a subprocess finishes —
    which is why every caller compares this against a HIGH-WATER MARK rather
    than against the previous sample.
    """
    try:
        import psutil
    except Exception:
        return None

    try:
        proc = psutil.Process(int(pid))
        total = _cpu_seconds(proc)
        try:
            io = proc.io_counters()
            total += (io.read_bytes + io.write_bytes) / float(1 << 20)
        except (psutil.Error, AttributeError, NotImplementedError):
            pass
        for child in proc.children(recursive=True):
            try:
                total += _cpu_seconds(child)
            except psutil.Error:
                continue
        return total
    except (psutil.Error, ValueError, OSError):
        # ValueError: psutil refuses a non-positive pid outright, which is
        # "cannot say" like every other unreadable process here.
        return None
