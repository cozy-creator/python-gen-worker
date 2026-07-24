"""gw#640: name a death the dying process cannot report.

`worker_fatal` (0.56.1) covers every death Python can observe: an exception
anywhere in boot/run, and the clean `return 0` from the run loop. RUN 9 of the
th#1085 cold-boot gate produced SIX process restarts and ZERO `worker_fatal`
rows, which proves the remaining class: the process dies BELOW Python — a
signal (cgroup OOM SIGKILL, SIGSEGV in a C extension, an external kill). No
`except` catches that and no in-process reporter can dial out after it.

So the reporter is the NEXT process, not the dying one. Two carriers:

  * the supervisor parent (``supervisor.py``) survives the child and reads its
    ``waitpid`` status directly — WIFSIGNALED / WTERMSIG / WCOREDUMP;
  * a boot record on the container filesystem covers the case where the whole
    cgroup goes (``memory.oom.group``) or the container is restarted: the next
    boot finds an unfinished record and reports it.

Both carry the container's memory facts — ``memory.max`` vs ``memory.current``
vs ``memory.peak``, and the ``memory.events`` ``oom_kill`` counter delta — so
"the kernel OOM-killed us" is a fact in the report, not an inference.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_GIB = 1024 ** 3

# Default location of the boot record. Container-local and restart-persistent:
# RunPod restarts the container in place, so this file outlives the death.
BOOT_RECORD_PATH = Path(
    os.environ.get("GEN_WORKER_BOOT_RECORD", "/tmp/gen-worker-boot-record.json")
)


def _cgroup_nodes(
    root: Path = _CGROUP_ROOT, proc_self_cgroup: Path = _PROC_SELF_CGROUP
) -> list[Path]:
    """cgroup-v2 dirs from root down to this process's own cgroup."""
    rel = ""
    try:
        for line in proc_self_cgroup.read_text().splitlines():
            if line.startswith("0::"):
                rel = line[3:].strip().strip("/")
                break
    except OSError:
        pass
    nodes = [root]
    node = root
    for part in Path(rel).parts:
        node = node / part
        nodes.append(node)
    return nodes


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_int(path: Path) -> Optional[int]:
    raw = _read_text(path)
    if raw is None or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_keyed(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    raw = _read_text(path)
    if not raw:
        return out
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return out


def _deepest(name: str) -> Optional[Path]:
    for node in reversed(_cgroup_nodes()):
        p = node / name
        if p.exists():
            return p
    return None


def oom_kill_count() -> int:
    """``memory.events`` oom_kill for this cgroup (0 when unreadable).

    Counts kernel OOM kills in this cgroup since it was created. A delta
    across a worker death is direct proof the kernel did it.
    """
    p = _deepest("memory.events")
    if p is None:
        return 0
    events = _read_keyed(p)
    return int(events.get("oom_kill", 0) or 0)


def container_limits() -> Dict[str, Any]:
    """Memory/CPU facts for the container we are actually running inside.

    Everything the "we sized for a bigger box than we got" family of bugs
    needs: the cgroup ceiling, what we are using against it, the high-water
    mark, the OOM counters, and the CPU quota vs the host's core count (the
    number ``os.cpu_count()`` reports, which is the HOST's, not ours).
    """
    facts: Dict[str, Any] = {}
    mem_max = _deepest("memory.max")
    mem_cur = _deepest("memory.current")
    mem_peak = _deepest("memory.peak")
    swap_max = _deepest("memory.swap.max")
    facts["memory_max_bytes"] = _read_int(mem_max) if mem_max else None
    facts["memory_current_bytes"] = _read_int(mem_cur) if mem_cur else None
    facts["memory_peak_bytes"] = _read_int(mem_peak) if mem_peak else None
    facts["memory_swap_max_bytes"] = _read_int(swap_max) if swap_max else None
    ev = _deepest("memory.events")
    facts["memory_events"] = _read_keyed(ev) if ev else {}
    cpu_max = _deepest("cpu.max")
    raw_cpu = _read_text(cpu_max) if cpu_max else None
    facts["cpu_max"] = raw_cpu
    facts["cpu_quota_cores"] = cpu_quota_cores()
    facts["host_cpu_count"] = os.cpu_count() or 0
    try:
        facts["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        facts["affinity_cpus"] = None
    meminfo = _read_meminfo()
    facts["meminfo_total_kb"] = meminfo.get("MemTotal")
    facts["meminfo_available_kb"] = meminfo.get("MemAvailable")
    return facts


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> Dict[str, int]:
    """``/proc/meminfo`` as {key: kB} ("MemTotal:  65...  kB" -> 3 fields)."""
    out: Dict[str, int] = {}
    raw = _read_text(path)
    if not raw:
        return out
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].endswith(":"):
            continue
        try:
            out[parts[0][:-1]] = int(parts[1])
        except ValueError:
            continue
    return out


def cpu_quota_cores() -> Optional[float]:
    """Cores this cgroup may actually use, from ``cpu.max`` (None = uncapped).

    ``os.cpu_count()`` reports the HOST's cores — 32 on a pod that owns 4.
    Anything that sizes work by core count must use THIS number.
    """
    p = _deepest("cpu.max")
    raw = _read_text(p) if p else None
    if not raw:
        return None
    parts = raw.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    try:
        quota, period = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if period <= 0 or quota <= 0:
        return None
    return quota / period


def effective_cpu_count() -> int:
    """Honest usable-core count: host cores min'd with affinity and quota."""
    candidates = [os.cpu_count() or 1]
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    quota = cpu_quota_cores()
    if quota is not None:
        candidates.append(max(1, int(quota + 0.5)))
    return max(1, min(candidates))


def describe_exit(status: int) -> Dict[str, Any]:
    """Decode a ``waitpid`` status into a reportable verdict."""
    out: Dict[str, Any] = {"raw_status": int(status)}
    if os.WIFSIGNALED(status):
        sig = os.WTERMSIG(status)
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"SIG{sig}"
        out.update(
            signaled=True,
            signal=sig,
            signal_name=name,
            core_dumped=bool(os.WCOREDUMP(status)),
            exit_code=128 + sig,
        )
    elif os.WIFEXITED(status):
        out.update(signaled=False, exit_code=int(os.WEXITSTATUS(status)))
    else:
        out.update(signaled=False, exit_code=-1)
    return out


def _gb(value: Optional[int]) -> str:
    if value is None:
        return "unlimited"
    return f"{value / _GIB:.2f}GiB"


def format_detail(
    *,
    phase: str,
    verdict: Dict[str, Any],
    limits: Dict[str, Any],
    oom_kill_delta: Optional[int] = None,
    lifetime_s: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """One human-readable blob for the ``worker_fatal`` carrier's ``detail``."""
    head = [f"phase={phase} exit_code={verdict.get('exit_code')}"]
    if verdict.get("signaled"):
        head.append(
            f"KILLED BY SIGNAL {verdict.get('signal_name')}"
            f"({verdict.get('signal')}) core_dumped={verdict.get('core_dumped')}"
        )
    else:
        head.append(f"exited normally code={verdict.get('exit_code')}")
    if lifetime_s is not None:
        head.append(f"lifetime_s={lifetime_s:.1f}")
    if oom_kill_delta is not None:
        head.append(
            f"cgroup_oom_kill_delta={oom_kill_delta}"
            + ("  <-- THE KERNEL OOM-KILLED US" if oom_kill_delta > 0 else "")
        )
    head.append(
        "memory.max=%s memory.current=%s memory.peak=%s swap.max=%s"
        % (
            _gb(limits.get("memory_max_bytes")),
            _gb(limits.get("memory_current_bytes")),
            _gb(limits.get("memory_peak_bytes")),
            _gb(limits.get("memory_swap_max_bytes")),
        )
    )
    head.append(
        "cpu.max=%s quota_cores=%s host_cpu_count=%s affinity=%s"
        % (
            limits.get("cpu_max"),
            limits.get("cpu_quota_cores"),
            limits.get("host_cpu_count"),
            limits.get("affinity_cpus"),
        )
    )
    head.append(f"memory.events={json.dumps(limits.get('memory_events') or {}, sort_keys=True)}")
    if limits.get("meminfo_total_kb"):
        head.append(
            "meminfo_total=%.2fGiB meminfo_available=%.2fGiB"
            % (
                (limits.get("meminfo_total_kb") or 0) / (1024 * 1024),
                (limits.get("meminfo_available_kb") or 0) / (1024 * 1024),
            )
        )
    if extra:
        head.append(json.dumps(extra, sort_keys=True, default=str))
    return "\n".join(head)


# ---- boot record ----------------------------------------------------------
#
# Covers the death the supervisor parent cannot survive (memory.oom.group, an
# external `docker kill`, the whole container going): the record is written at
# boot and cleared on a clean exit, so an unfinished record found at the NEXT
# boot IS the previous process's unreported death.


def write_boot_record(path: Path = BOOT_RECORD_PATH, **extra: Any) -> None:
    """Stamp this boot: pid, time, and the OOM counter to diff against."""
    record = {
        "pid": os.getpid(),
        "boot_unix": time.time(),
        "oom_kill_at_boot": oom_kill_count(),
        "limits": container_limits(),
    }
    record.update(extra)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, default=str))
    except OSError:
        pass


def clear_boot_record(path: Path = BOOT_RECORD_PATH) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def take_boot_record(path: Path = BOOT_RECORD_PATH) -> Optional[Dict[str, Any]]:
    """Read and consume a previous boot's record (None when absent/garbage)."""
    try:
        raw = path.read_text()
    except OSError:
        return None
    clear_boot_record(path)
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def previous_boot_detail(path: Path = BOOT_RECORD_PATH) -> Optional[str]:
    """A report for a previous process that vanished without clearing its
    record — i.e. one killed so hard even the supervisor did not survive."""
    record = take_boot_record(path)
    if record is None:
        return None
    limits = container_limits()
    before = int(record.get("oom_kill_at_boot") or 0)
    now = oom_kill_count()
    lifetime = None
    try:
        lifetime = max(0.0, time.time() - float(record.get("boot_unix") or 0.0))
    except (TypeError, ValueError):
        pass
    return format_detail(
        phase="previous_container_death",
        verdict={"exit_code": None, "signaled": None},
        limits=limits,
        oom_kill_delta=max(0, now - before) if now >= before else None,
        lifetime_s=lifetime,
        extra={
            "previous_pid": record.get("pid"),
            "limits_at_previous_boot": record.get("limits"),
            "note": (
                "the previous process left an unfinished boot record: it died "
                "without its supervisor surviving to report (whole-cgroup OOM "
                "kill, container restart, or external kill)"
            ),
        },
    )


__all__ = [
    "BOOT_RECORD_PATH",
    "clear_boot_record",
    "container_limits",
    "cpu_quota_cores",
    "describe_exit",
    "effective_cpu_count",
    "format_detail",
    "oom_kill_count",
    "previous_boot_detail",
    "take_boot_record",
    "write_boot_record",
]
