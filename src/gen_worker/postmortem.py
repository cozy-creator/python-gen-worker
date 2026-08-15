"""Name a death the dying process cannot report.

`worker_fatal` covers every death Python can observe: an exception anywhere in
boot/run, and the clean `return 0` from the run loop. This module covers the
remaining class — the process dies BELOW Python, by signal (cgroup OOM
SIGKILL, SIGSEGV in a C extension, an external kill). No `except` catches that
and no in-process reporter can dial out after it.

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

import contextlib
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import faulthandler
from . import hostfacts
from .procsplit import group_ordinal, host_siblings

logger = logging.getLogger(__name__)

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_PROC_MOUNTS = Path("/proc/mounts")
_GIB = 1024 ** 3

# Filesystems whose contents die with the container's memory — i.e. exactly
# the death this record exists to report.
_VOLATILE_FSTYPES = {"tmpfs", "ramfs"}

_BOOT_RECORD_NAME = "gen-worker-boot-record.json"


def _fstype_for(path: Path, mounts: Path = _PROC_MOUNTS) -> str:
    """fstype of the longest mount point that is a prefix of ``path``.
    Empty string when /proc/mounts is unreadable (non-Linux, sandboxes)."""
    try:
        lines = mounts.read_text().splitlines()
    except OSError:
        return ""
    target = os.path.abspath(str(path))
    best, best_type = "", ""
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        point, fstype = parts[1].replace("\\040", " "), parts[2]
        if (target == point or target.startswith(point.rstrip("/") + "/")) and (
            len(point) > len(best)
        ):
            best, best_type = point, fstype
    return best_type


def boot_record_is_volatile(path: Path, mounts: Path = _PROC_MOUNTS) -> bool:
    """Whether the record's carrier is wiped by the very death it instruments.

    ``/tmp`` is tmpfs — RAM — on many container images. A cgroup OOM kill (the
    headline case this module reports) frees that RAM, so the evidence dies
    with the process and the next boot finds nothing, indistinguishable from
    "no death happened".
    """
    probe = path if path.exists() else path.parent
    return _fstype_for(probe, mounts) in _VOLATILE_FSTYPES


def _default_boot_record_path() -> Path:
    """Prefer a DURABLE carrier. ``GEN_WORKER_BOOT_RECORD`` wins; otherwise the
    model-cache volume (a real RunPod disk) when it is not itself volatile;
    ``/tmp`` only as the last resort, and then :func:`write_boot_record` says
    so loudly rather than pretending the record is durable."""
    explicit = os.environ.get("GEN_WORKER_BOOT_RECORD", "").strip()
    if explicit:
        return Path(explicit)
    cache = os.environ.get("TENSORHUB_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / _BOOT_RECORD_NAME
        if not boot_record_is_volatile(candidate):
            return candidate
    return Path("/tmp") / _BOOT_RECORD_NAME


BOOT_RECORD_PATH = _default_boot_record_path()


def cgroup_nodes(
    root: Path = _CGROUP_ROOT, proc_self_cgroup: Path = _PROC_SELF_CGROUP
) -> list[Path]:
    """cgroup-v2 dirs from root down to this process's own cgroup."""
    return hostfacts.cgroup_nodes(root, proc_self_cgroup)


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
    for node in reversed(cgroup_nodes()):
        p = node / name
        if p.exists():
            return p
    return None


# ---- cgroup v1 --------------------------------------------------
#
# Some fleets (RunPod's AP-JP-1 H100 hosts among them) mount the memory
# controller at /sys/fs/cgroup/memory and have NO memory.max /
# memory.current / memory.events; without this fallback the postmortem is
# blind on exactly the pod class that OOM-kills. v1 spells the same facts
# differently:
#     memory.limit_in_bytes / memory.usage_in_bytes / memory.max_usage_in_bytes
#     memory.oom_control ("oom_kill N" line, kernel >= 4.13)

_V1_MEM = _CGROUP_ROOT / "memory"
# v1 "unlimited" territory (kernel reports ~0x7ffffffffffff000).
_V1_UNLIMITED = 1 << 60


def _v1_int(name: str) -> Optional[int]:
    v = _read_int(_V1_MEM / name)
    if v is None or not 0 <= v < _V1_UNLIMITED:
        return None
    return v


def _v1_oom_control() -> Dict[str, int]:
    return _read_keyed(_V1_MEM / "memory.oom_control")


def oom_kill_count() -> int:
    """Kernel OOM kills in this cgroup since creation (0 when unreadable).

    v2: ``memory.events`` oom_kill; v1 fallback: the ``oom_kill`` counter in
    ``memory.oom_control``. A delta across a worker death is direct proof
    the kernel did it.
    """
    p = _deepest("memory.events")
    if p is not None:
        events = _read_keyed(p)
        return int(events.get("oom_kill", 0) or 0)
    return int(_v1_oom_control().get("oom_kill", 0) or 0)


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
    facts["cgroup_flavor"] = "v2" if mem_max is not None else "none"
    if mem_max is None and (_V1_MEM / "memory.limit_in_bytes").exists():
        # v1 host — the same facts under their v1 names.
        facts["cgroup_flavor"] = "v1"
        facts["memory_max_bytes"] = _v1_int("memory.limit_in_bytes")
        facts["memory_current_bytes"] = _v1_int("memory.usage_in_bytes")
        facts["memory_peak_bytes"] = _v1_int("memory.max_usage_in_bytes")
        facts["memory_swap_max_bytes"] = _v1_int("memory.memsw.limit_in_bytes")
        facts["memory_events"] = _v1_oom_control()
    # The one fact deciding whether the process split can report at all:
    # `memory.oom.group=1` makes the kernel kill the whole cgroup as a unit,
    # parent included, and `mem_cgroup_get_oom_group()` is consulted on the
    # GLOBAL oom path too, so `memory.max=unlimited` does not rule it out.
    # Every death and every boot record carries it.
    oom_group = _deepest("memory.oom.group")
    facts["memory_oom_group"] = _read_int(oom_group) if oom_group else None
    facts["cpu_max"] = hostfacts.cpu_quota_raw()
    facts["cpu_quota_cores"] = cpu_quota_cores()
    facts["host_cpu_count"] = os.cpu_count() or 0
    try:
        facts["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        facts["affinity_cpus"] = None
    meminfo = hostfacts.meminfo_kb()
    facts["meminfo_total_kb"] = meminfo.get("MemTotal")
    facts["meminfo_available_kb"] = meminfo.get("MemAvailable")
    return facts


def cpu_quota_cores() -> Optional[float]:
    """Cores this cgroup may actually use (None = uncapped). One reader,
    in :mod:`hostfacts` — this module's copy had no cgroup-v1 fallback while
    ``cpu_budget``'s had no chain walk, so a v1 host and a nested cgroup got
    different answers from the two."""
    return hostfacts.cpu_quota()


def effective_cpu_count() -> int:
    """Honest usable-core count: host cores min'd with affinity and quota.

    ``floor``, not ``int(x + 0.5)``: this used to round a 2.5-core quota UP to
    3 while ``cpu_budget.cpu_allowance`` kept 2.5, so the fleet planned against
    3 and torch ran 2.5 cores' worth of threads under a throttling kernel. The
    integer derivation is stated once, in :class:`hostfacts.CpuAllowance`.
    """
    return hostfacts.cpu_allowance().whole_cores


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
    # `1` means the kernel kills the cgroup as a unit, so this report is only
    # reaching you because the parent happened to outlive it.
    oom_group = limits.get("memory_oom_group")
    head.append(
        "memory.oom.group=%s%s"
        % (
            "unreadable" if oom_group is None else oom_group,
            "  <-- GROUP KILL: the pgw#763 reporter dies with the child"
            if oom_group == 1 else "",
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
    volatile = boot_record_is_volatile(path)
    record = {
        "pid": os.getpid(),
        "boot_unix": time.time(),
        "oom_kill_at_boot": oom_kill_count(),
        "limits": container_limits(),
        # Carried IN the record so a reader can tell "the pod did not die"
        # from "the evidence was on RAM and died with it".
        "carrier_volatile": volatile,
    }
    record.update(extra)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, default=str))
    except OSError:
        pass
    if volatile:
        logger.warning(
            "POSTMORTEM CARRIER IS VOLATILE: boot record at %s lives on %s — a "
            "cgroup OOM kill frees it, so the death this record exists to report "
            "will look like no death at all. Point GEN_WORKER_BOOT_RECORD at a "
            "real volume (pgw#657).",
            path, _fstype_for(path if path.exists() else path.parent) or "an unknown fs",
        )


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
    extra: Dict[str, Any] = {
        "previous_pid": record.get("pid"),
        "limits_at_previous_boot": record.get("limits"),
        "note": (
            "the previous process left an unfinished boot record: it died "
            "without its supervisor surviving to report (whole-cgroup OOM "
            "kill, container restart, or external kill)"
        ),
    }
    # The previous process's in-flight marker + fault dump are the death's
    # attribution; consuming them here also feeds the crash registry so this
    # boot's gate can refuse a crash-streak function.
    extra.update(attribute_all_signal_deaths(
        signal_name="container_death", marker_dir=path.parent,
    ))
    return format_detail(
        phase="previous_container_death",
        verdict={"exit_code": None, "signaled": None},
        limits=limits,
        oom_kill_delta=max(0, now - before) if now >= before else None,
        lifetime_s=lifetime,
        extra=extra,
    )


# ---- in-flight marker + native-crash streaks --------------------
#
# A bare exit_code=139 carries NOTHING: no frame, no function, and the
# restarted process takes the same request shape and dies again. Three pieces
# close that class:
#
#   * a faulthandler dump file the dying process writes below Python — the
#     surviving supervisor attaches its tail, so a signal death carries the
#     Python stacks of every thread;
#   * an in-flight marker naming what was executing (function, kind,
#     request id) — written at request/warmup start, cleared on finish;
#   * a per-pod crash registry: a function whose in-flight execution died by
#     signal ``NATIVE_CRASH_REFUSE_STREAK`` times is refused at the next
#     boot's gate (degrade-never-die across process death: siblings keep
#     serving, the refusal is loud and typed, the hub reroutes).

_INFLIGHT_NAME = "gen-worker-inflight.json"
_CRASH_REGISTRY_NAME = "gen-worker-crash-streaks.json"
_FAULT_DUMP_NAME = "gen-worker-fault-dump.txt"
_LOAD_PROGRESS_NAME = "gen-worker-load-progress.json"

#: Signal deaths mid-flight on one function, on one pod, before the gate
#: refuses it. 2 = one free retry for a genuinely transient fault.
NATIVE_CRASH_REFUSE_STREAK = 2

_FAULT_DUMP_TAIL_BYTES = 8000


def _sibling(name: str) -> Path:
    """Same durable carrier the boot record chose."""
    return BOOT_RECORD_PATH.parent / name


# A compute group is a separate OS process, so its transient markers need one
# writer each; otherwise one child replaces or truncates a sibling's evidence.
# The crash registry stays pod-wide on purpose: it is the worker-level refusal
# fact consumed by every group on its next boot.
def _group_marker_path(
    name: str, ordinal: int, marker_dir: Optional[Path] = None,
) -> Path:
    root = marker_dir or BOOT_RECORD_PATH.parent
    return root / f"g{max(0, int(ordinal))}" / name


def group_inflight_path(
    ordinal: int, marker_dir: Optional[Path] = None,
) -> Path:
    return _group_marker_path(_INFLIGHT_NAME, ordinal, marker_dir)


def group_fault_dump_path(
    ordinal: int, marker_dir: Optional[Path] = None,
) -> Path:
    return _group_marker_path(_FAULT_DUMP_NAME, ordinal, marker_dir)


def _local_marker_path(name: str) -> Path:
    # Importing procsplit's tiny environment helpers here keeps the reserved
    # names canonical without pulling in the parent or any compute dependency.

    if host_siblings() > 1:
        return _group_marker_path(name, group_ordinal())
    return _sibling(name)


INFLIGHT_PATH = _local_marker_path(_INFLIGHT_NAME)
CRASH_REGISTRY_PATH = _sibling(_CRASH_REGISTRY_NAME)
FAULT_DUMP_PATH = _local_marker_path(_FAULT_DUMP_NAME)
LOAD_PROGRESS_PATH = _local_marker_path(_LOAD_PROGRESS_NAME)


def group_load_progress_path(
    ordinal: int, marker_dir: Optional[Path] = None,
) -> Path:
    return _group_marker_path(_LOAD_PROGRESS_NAME, ordinal, marker_dir)


def write_load_progress(
    record: Dict[str, Any], path: Optional[Path] = None,
) -> None:
    """The load path's death breadcrumb — overwritten every reporter tick,
    consumed by the death attribution, so a SIGKILL mid-load names the
    phase/component and the last byte count instead of nothing."""
    path = path or LOAD_PROGRESS_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, default=str))
        os.replace(tmp, path)
    except OSError:
        pass


def clear_load_progress(path: Optional[Path] = None) -> None:
    path = path or LOAD_PROGRESS_PATH
    try:
        path.unlink()
    except OSError:
        pass


def take_load_progress(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Read and consume the last load-progress breadcrumb (None when absent)."""
    path = path or LOAD_PROGRESS_PATH
    try:
        raw = path.read_text()
    except OSError:
        return None
    clear_load_progress(path)
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None

_fault_dump_file: Optional[Any] = None


def enable_fault_dump(path: Optional[Path] = None) -> None:
    """Point ``faulthandler`` at a file the supervisor can read after we die.

    ``faulthandler.enable`` writes every thread's Python stack from inside
    the signal handler (SIGSEGV/SIGFPE/SIGABRT/SIGBUS) without allocating,
    then the default action re-raises — so the file has content by the time
    ``waitpid`` returns to the parent."""

    global _fault_dump_file
    path = path or FAULT_DUMP_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _fault_dump_file = open(path, "w", buffering=1)
        faulthandler.enable(file=_fault_dump_file, all_threads=True)
    except (OSError, ValueError):
        logger.warning("fault-dump file unavailable at %s; faulthandler "
                       "falls back to stderr", path)
        faulthandler.enable(all_threads=True)


def fault_dump_tail(path: Optional[Path] = None,
                    limit: int = _FAULT_DUMP_TAIL_BYTES) -> str:
    path = path or FAULT_DUMP_PATH
    try:
        raw = path.read_text(errors="replace").strip()
    except OSError:
        return ""
    return raw[-limit:]


def clear_fault_dump(path: Optional[Path] = None) -> None:
    path = path or FAULT_DUMP_PATH
    try:
        path.unlink()
    except OSError:
        pass


_inflight_lock = threading.Lock()
_inflight_active: Dict[int, Dict[str, Any]] = {}
_inflight_next_token = 0


def _write_inflight(path: Path) -> None:
    try:
        if not _inflight_active:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"active": list(_inflight_active.values())}))
    except OSError:
        pass


def note_inflight(
    kind: str, function: str, *, request_id: str = "",
    path: Optional[Path] = None,
) -> int:
    """Stamp an execution about to touch the GPU; returns a token for
    :func:`clear_inflight`. The file carries EVERY active execution (a
    background-mint seed can overlap another instance's request), so a
    signal death attributes to all of them — usually exactly one. Cheap:
    one tiny json write off the compute path."""
    global _inflight_next_token
    path = path or INFLIGHT_PATH
    record = {
        "kind": kind,
        "function": function,
        "request_id": request_id,
        "pid": os.getpid(),
        "started_unix": time.time(),
    }
    with _inflight_lock:
        _inflight_next_token += 1
        token = _inflight_next_token
        _inflight_active[token] = record
        _write_inflight(path)
    return token


def clear_inflight(
    token: Optional[int] = None, path: Optional[Path] = None,
) -> None:
    """Retire one execution (or, with no token, every marker — boot/exit
    hygiene)."""
    path = path or INFLIGHT_PATH
    with _inflight_lock:
        if token is None:
            _inflight_active.clear()
        else:
            _inflight_active.pop(token, None)
        _write_inflight(path)


def current_inflight_request(kind: str = "request") -> str:
    """The request id of the newest LIVE in-flight execution of ``kind``, so a
    serve-time guard-miss confession names the exact request that hit it. ''
    when none is marked."""
    with _inflight_lock:
        for _token, record in sorted(_inflight_active.items(), reverse=True):
            if record.get("kind") == kind and record.get("request_id"):
                return str(record["request_id"])
    return ""


def take_inflight(path: Optional[Path] = None) -> list[Dict[str, Any]]:
    """Read and consume the active-execution list a dead process left."""
    path = path or INFLIGHT_PATH
    try:
        raw = path.read_text()
    except OSError:
        return []
    try:
        path.unlink()
    except OSError:
        pass
    try:
        record = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(record, dict):
        return []
    active = record.get("active")
    return [r for r in active if isinstance(r, dict)] if isinstance(
        active, list) else []


#: How many contributing request ids a streak row remembers. Only needs to
#: outlive one request's retry ladder.
_CRASH_REQUEST_MEMORY = 8


def record_native_crash(
    function: str, *, kind: str = "", signal_name: str = "",
    request_id: str = "", path: Optional[Path] = None,
) -> int:
    """Count one signal death attributed to ``function``; returns the new
    streak. The registry lives on the pod's container fs, so it survives
    process restarts and dies with the pod — per-SKU-instance by
    construction.

    The streak counts DISTINCT REQUESTS, not attempts: the hub's blame ladder
    re-runs one deterministically fatal payload on the same pod, and counting
    attempts would condemn a healthy pod `worker_native_crash_loop` on retries
    of a SINGLE request. The gate exists for a function that keeps killing this
    pod across DIFFERENT work. A death with no request id (a background
    compile) still counts every time.
    """
    path = path or CRASH_REGISTRY_PATH
    streaks = native_crash_streaks(path)
    row = streaks.get(function) or {"count": 0}
    request_id = str(request_id or "").strip()
    seen = [str(r) for r in (row.get("requests") or []) if r]
    if request_id and request_id in seen:
        # Same request, another attempt: one fault, already counted.
        row["last_kind"] = kind
        row["last_signal"] = signal_name
        row["last_unix"] = time.time()
        streaks[function] = row
        _write_streaks(path, streaks)
        return int(row.get("count") or 0)
    row["count"] = int(row.get("count") or 0) + 1
    if request_id:
        seen.append(request_id)
        row["requests"] = seen[-_CRASH_REQUEST_MEMORY:]
    row["last_kind"] = kind
    row["last_signal"] = signal_name
    row["last_unix"] = time.time()
    streaks[function] = row
    _write_streaks(path, streaks)
    return int(row["count"])


def _write_streaks(path: Path, streaks: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(streaks, default=str))
    except OSError:
        pass


def native_crash_streaks(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    path = path or CRASH_REGISTRY_PATH
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(fn): row for fn, row in raw.items() if isinstance(row, dict)
    }


#: Inflight-marker kind for a background torch.compile (hot-swap warm thread,
#: mint compile units). Its presence at a signal death makes the COMPILE the
#: prime suspect — dynamo/inductor run native codegen — so the streak is
#: recorded against the compile marker, never the tenant request that happened
#: to be in flight (that misattribution condemns a whole SKU for a software
#: race).
COMPILE_KIND = "compile"
_COMPILE_FN_PREFIX = "compile:"


def compile_marker(label: str) -> str:
    """Registry/marker function name for a background compile of ``label``.
    Namespaced so it can never collide with (or refuse) a serving function."""
    return _COMPILE_FN_PREFIX + str(label or "unknown")


@contextlib.contextmanager
def compile_inflight(
    label: str, *, path: Optional[Path] = None,
) -> Iterator[None]:
    """Name a device-touching COMPILE span before it starts, and retire it on
    every exit (pgw#1262).

    The `finally` is the load-bearing half: a leaked marker would make the NEXT
    unrelated signal death read as a compile crash, which is the same
    misattribution this exists to prevent, pointed the other way.

    Wrapping a span in this is what makes :func:`compile_crash_rows` — and so
    pgw#714's eager-only reboot — cover it. A device-touching compile/adopt
    phase that does NOT hold one of these is invisible to that mechanism, and
    its deaths are charged to whatever tenant request happened to be in flight.
    """
    token = note_inflight(COMPILE_KIND, compile_marker(label), path=path)
    try:
        yield
    finally:
        clear_inflight(token, path=path)


def compile_crash_rows(
    path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Crash-registry rows attributed to background compiles."""
    return {
        fn: row for fn, row in native_crash_streaks(path).items()
        if fn.startswith(_COMPILE_FN_PREFIX)
        or str(row.get("last_kind") or "") == COMPILE_KIND
    }


def attribute_signal_death(
    *, signal_name: str,
    inflight_path: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    dump_path: Optional[Path] = None,
    load_progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Everything the post-mortem reporter can attach to a signal death:
    the in-flight markers (consumed), the fault-dump tail, and — for each
    marker naming a function — the recorded crash streak.

    When a ``compile`` marker is in flight the streak is recorded ONLY against
    the compile marker(s): a background dynamo/inductor compile racing a tenant
    forward is the native suspect, and charging the request's function would
    refuse serving and condemn the SKU for a software bug."""
    extra: Dict[str, Any] = {}
    inflight = take_inflight(inflight_path)
    if inflight:
        extra["inflight"] = inflight
        compile_rows = [
            row for row in inflight
            if str(row.get("kind") or "") == COMPILE_KIND
        ]
        blamed = compile_rows or inflight
        streaks: Dict[str, int] = {}
        for row in blamed:
            fn = str(row.get("function") or "")
            if fn:
                streaks[fn] = record_native_crash(
                    fn, kind=str(row.get("kind") or ""),
                    signal_name=signal_name,
                    request_id=str(row.get("request_id") or ""),
                    path=registry_path)
        if streaks:
            extra["native_crash_streaks"] = streaks
    tail = fault_dump_tail(dump_path)
    if tail:
        extra["fault_dump_tail"] = tail
    # A death mid-load names the phase/component and last byte count — the
    # difference between "SIGKILL" and "SIGKILL at hydrate:transformer,
    # 48.2/94.3 GiB staged".
    progress = take_load_progress(load_progress_path)
    if progress:
        extra["last_load_progress"] = progress
    return extra


def _group_marker_dirs(marker_dir: Optional[Path] = None) -> list[Path]:
    """Existing ``gN`` marker dirs, sorted by ordinal.

    The whole-worker supervisor and next-container reporter do not own the
    delivered topology, so the durable filesystem is their census.  Only the
    exact directory form this module creates is admitted.
    """
    marker_dir = marker_dir or BOOT_RECORD_PATH.parent
    try:
        dirs = [
            p for p in marker_dir.iterdir()
            if p.is_dir() and p.name.startswith("g") and p.name[1:].isdigit()
        ]
    except OSError:
        return []
    return sorted(dirs, key=lambda p: int(p.name[1:]))


def clear_all_inflight(marker_dir: Optional[Path] = None) -> None:
    """Remove transient markers for the whole worker, including every group."""
    paths = {
        marker_dir / _INFLIGHT_NAME if marker_dir is not None else INFLIGHT_PATH
    }
    paths.update(p / _INFLIGHT_NAME for p in _group_marker_dirs(marker_dir))
    for path in paths:
        clear_inflight(path=path)


def clear_all_fault_dumps(marker_dir: Optional[Path] = None) -> None:
    """Remove fault dumps for the whole worker, including every group."""
    paths = {
        marker_dir / _FAULT_DUMP_NAME if marker_dir is not None else FAULT_DUMP_PATH
    }
    paths.update(p / _FAULT_DUMP_NAME for p in _group_marker_dirs(marker_dir))
    for path in paths:
        clear_fault_dump(path=path)


def attribute_all_signal_deaths(
    *, signal_name: str, marker_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Consume signal evidence after the whole control process/container dies.

    A per-slot death calls :func:`attribute_signal_death` with its ordinal.
    The outer supervisor and next boot instead lost the owner of every slot,
    so they aggregate all one-writer group files without letting one group's
    record overwrite another's.
    """
    pairs = [(
        "worker",
        marker_dir / _INFLIGHT_NAME if marker_dir is not None else INFLIGHT_PATH,
        marker_dir / _FAULT_DUMP_NAME if marker_dir is not None else FAULT_DUMP_PATH,
        marker_dir / _LOAD_PROGRESS_NAME if marker_dir is not None
        else LOAD_PROGRESS_PATH,
    )]
    pairs.extend(
        (p.name, p / _INFLIGHT_NAME, p / _FAULT_DUMP_NAME,
         p / _LOAD_PROGRESS_NAME)
        for p in _group_marker_dirs(marker_dir)
    )
    seen: set[tuple[Path, Path]] = set()
    inflight: list[Dict[str, Any]] = []
    streaks: Dict[str, int] = {}
    tails: list[str] = []
    progress_rows: Dict[str, Dict[str, Any]] = {}
    for label, inflight_path, dump_path, progress_path in pairs:
        key = (inflight_path, dump_path)
        if key in seen:
            continue
        seen.add(key)
        if not (inflight_path.exists() or dump_path.exists()
                or progress_path.exists()):
            continue
        detail = attribute_signal_death(
            signal_name=signal_name,
            inflight_path=inflight_path,
            dump_path=dump_path,
            load_progress_path=progress_path,
        )
        row = detail.get("last_load_progress")
        if isinstance(row, dict):
            progress_rows[label] = row
        rows = detail.get("inflight")
        if isinstance(rows, list):
            inflight.extend(r for r in rows if isinstance(r, dict))
        counts = detail.get("native_crash_streaks")
        if isinstance(counts, dict):
            streaks.update({str(k): int(v) for k, v in counts.items()})
        tail = str(detail.get("fault_dump_tail") or "")
        if tail:
            tails.append(f"[{label}]\n{tail}")
    extra: Dict[str, Any] = {}
    if inflight:
        extra["inflight"] = inflight
    if streaks:
        extra["native_crash_streaks"] = streaks
    if tails:
        extra["fault_dump_tail"] = "\n\n".join(tails)[-_FAULT_DUMP_TAIL_BYTES:]
    if progress_rows:
        extra["last_load_progress"] = (
            progress_rows["worker"] if list(progress_rows) == ["worker"]
            else progress_rows
        )
    return extra


__all__ = [
    "BOOT_RECORD_PATH",
    "CRASH_REGISTRY_PATH",
    "FAULT_DUMP_PATH",
    "INFLIGHT_PATH",
    "COMPILE_KIND",
    "NATIVE_CRASH_REFUSE_STREAK",
    "attribute_all_signal_deaths",
    "attribute_signal_death",
    "compile_crash_rows",
    "compile_inflight",
    "compile_marker",
    "boot_record_is_volatile",
    "clear_boot_record",
    "clear_all_fault_dumps",
    "clear_all_inflight",
    "clear_fault_dump",
    "clear_inflight",
    "container_limits",
    "cpu_quota_cores",
    "current_inflight_request",
    "describe_exit",
    "effective_cpu_count",
    "enable_fault_dump",
    "fault_dump_tail",
    "format_detail",
    "group_fault_dump_path",
    "group_inflight_path",
    "group_load_progress_path",
    "LOAD_PROGRESS_PATH",
    "clear_load_progress",
    "take_load_progress",
    "write_load_progress",
    "native_crash_streaks",
    "note_inflight",
    "oom_kill_count",
    "previous_boot_detail",
    "record_native_crash",
    "take_boot_record",
    "take_inflight",
    "write_boot_record",
]
