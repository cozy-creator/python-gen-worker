"""Kernel-accounted evidence that another process is doing real work."""

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
    """This process tree's kernel-accounted work, or ``None`` when it cannot be read at all."""
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
        return None
