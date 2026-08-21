"""ONE home for every primitive fact this worker observes about its own host."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgspec

logger = logging.getLogger(__name__)

DEVICE_PRESENT = "cuda"
DEVICE_ABSENT = "none"
DEVICE_UNREADABLE = "unreadable"


class CudaState(msgspec.Struct, frozen=True):
    """Which of the three CUDA states this host is in, and the evidence."""

    state: str
    probe_class: str = ""
    detail: str = ""

    @property
    def present(self) -> bool:
        return self.state == DEVICE_PRESENT

    @property
    def absent(self) -> bool:
        return self.state == DEVICE_ABSENT

    @property
    def unreadable(self) -> bool:
        return self.state == DEVICE_UNREADABLE


_cuda_state: Optional[CudaState] = None


def cuda_ready() -> bool:
    """Does torch have a usable CUDA device right now? The ONLY ``torch.cuda.is_available()`` call site in ``src/`` — fenced by ``tests/test_hostfacts_pgw896.py``."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — a probe never changes an outcome
        return False


def cuda_state(*, device_index: int = 0) -> CudaState:
    """The three-valued CUDA verdict, measured once per process."""
    global _cuda_state
    if _cuda_state is not None:
        return _cuda_state
    from .cuda_probe import CARDLESS_PROBE_CLASSES, classify_probe_failure, probe_cuda

    result = probe_cuda(device_index)
    if result.ok:
        _cuda_state = CudaState(DEVICE_PRESENT)
        return _cuda_state
    klass = classify_probe_failure(result.reason)
    _cuda_state = CudaState(
        DEVICE_ABSENT if klass in CARDLESS_PROBE_CLASSES else DEVICE_UNREADABLE,
        probe_class=klass,
        detail=result.reason,
    )
    return _cuda_state


def reset_cuda_state() -> None:
    """Forget the cached verdict (tests; a re-probe after a device reset)."""
    global _cuda_state
    _cuda_state = None


def _mem_get_info(device: Optional[int]) -> Optional[Tuple[int, int]]:
    if not cuda_ready():
        return None
    try:
        import torch

        free, total = torch.cuda.mem_get_info(
            torch.cuda.current_device() if device is None else int(device)
        )
        return int(free), int(total)
    except Exception as exc:  # noqa: BLE001
        logger.debug("mem_get_info failed: %s: %s", type(exc).__name__, exc)
        return None


def free_vram_bytes(device: Optional[int] = None) -> Optional[int]:
    """Driver-free bytes on ONE card right now."""
    reading = _mem_get_info(device)
    return None if reading is None else reading[0]


def total_vram_bytes(device: Optional[int] = None) -> Optional[int]:
    """Total bytes on ONE card, honestly measured (an H100-80GB reports 79.19 GiB — declare that, not the marketing number)."""
    reading = _mem_get_info(device)
    return None if reading is None else reading[1]


def headroom_bytes(device: Optional[int] = None) -> Optional[int]:
    """Bytes a new allocation on ONE card could actually get."""
    reading = _mem_get_info(device)
    if reading is None:
        return None
    free, _total = reading
    try:
        import torch

        index = torch.cuda.current_device() if device is None else int(device)
        reclaimable = max(
            0,
            int(torch.cuda.memory_reserved(index))
            - int(torch.cuda.memory_allocated(index)),
        )
    except Exception:  # noqa: BLE001
        reclaimable = 0
    return int(free) + reclaimable


def process_ceiling_bytes(device: Optional[int] = None) -> Optional[int]:
    """Bytes THIS process may occupy on ONE card at its peak."""
    reading = _mem_get_info(device)
    if reading is None:
        return None
    free, _total = reading
    try:
        import torch

        index = torch.cuda.current_device() if device is None else int(device)
        allocated = int(torch.cuda.memory_allocated(index))
    except Exception:  # noqa: BLE001
        allocated = 0
    return int(free) + allocated


def device_count() -> int:
    """Cards this process can see."""
    if not cuda_ready():
        return 0
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def device_identity(device: Optional[int] = None) -> Tuple[str, str]:
    """``(name, sm)`` for one card — ``("", "")`` off-GPU."""
    if not cuda_ready():
        return "", ""
    name = sm = ""
    try:
        import torch

        index = torch.cuda.current_device() if device is None else int(device)
    except Exception:  # noqa: BLE001
        return "", ""
    try:
        name = str(torch.cuda.get_device_properties(index).name)
    except Exception:  # noqa: BLE001
        pass
    try:
        major, minor = torch.cuda.get_device_capability(index)
        sm = f"sm_{major}{minor}"
    except Exception:  # noqa: BLE001
        pass
    return name, sm


_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
PROC_MEMINFO = Path("/proc/meminfo")


def cgroup_nodes(
    root: Optional[Path] = None, proc_self_cgroup: Optional[Path] = None
) -> List[Path]:
    """Cgroup-v2 dirs from ``root`` down to this process's own cgroup."""
    root = _CGROUP_ROOT if root is None else root
    proc_self_cgroup = (
        _PROC_SELF_CGROUP if proc_self_cgroup is None else proc_self_cgroup)
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


def _read_text(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _deepest(
    name: str, root: Optional[Path], proc_self_cgroup: Optional[Path]
) -> Optional[Path]:
    for node in reversed(cgroup_nodes(root, proc_self_cgroup)):
        candidate = node / name
        if candidate.exists():
            return candidate
    return None


def cpu_quota(
    *,
    root: Optional[Path] = None,
    proc_self_cgroup: Optional[Path] = None,
) -> Optional[float]:
    """Cores this cgroup may use, or ``None`` when uncapped/unreadable."""
    raw = _read_text(_deepest("cpu.max", root, proc_self_cgroup))
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return quota / period
            except ValueError:
                pass
        if parts:
            return None
    try:
        v1 = (_CGROUP_ROOT if root is None else root) / "cpu"
        quota = int((v1 / "cpu.cfs_quota_us").read_text().strip())
        period = int((v1 / "cpu.cfs_period_us").read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def cpu_quota_raw(
    *,
    root: Optional[Path] = None,
    proc_self_cgroup: Optional[Path] = None,
) -> Optional[str]:
    """The verbatim ``cpu.max`` line, for a postmortem to echo."""
    return _read_text(_deepest("cpu.max", root, proc_self_cgroup))


class CpuAllowance(msgspec.Struct, frozen=True):
    """The narrowest true bound on this process's CPU, and which one bound."""

    cores: float
    basis: str
    os_count: int
    affinity: int
    quota_cores: float

    @property
    def whole_cores(self) -> int:
        """The integer allowance: ``floor``, never ``int(x + 0.5)``."""
        return max(1, int(math.floor(self.cores)))


def cpu_allowance(
    *,
    root: Optional[Path] = None,
    proc_self_cgroup: Optional[Path] = None,
) -> CpuAllowance:
    """The ONE ``min()`` over cgroup quota, affinity mask and host core count."""
    os_count = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os_count
    quota = cpu_quota(root=root, proc_self_cgroup=proc_self_cgroup)
    candidates: List[Tuple[float, str]] = [
        (float(os_count), "cpu_count"),
        (float(affinity), "affinity"),
    ]
    if quota is not None and quota > 0:
        candidates.append((float(quota), "quota"))
    cores, basis = min(candidates)
    return CpuAllowance(
        cores=max(0.0, cores),
        basis=basis,
        os_count=os_count,
        affinity=affinity,
        quota_cores=float(quota) if quota is not None else -1.0,
    )


def meminfo_kb(path: Optional[Path] = None) -> Dict[str, int]:
    """``/proc/meminfo`` as ``{key: kB}`` — one parser, in kB as the file is."""
    out: Dict[str, int] = {}
    try:
        raw = (PROC_MEMINFO if path is None else path).read_text()
    except OSError:
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


class HostFacts(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the fleet is told about this host, measured ONCE."""

    gpu_count: int = 0
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    gpu_name: str = ""
    gpu_sm: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    driver_version: str = ""
    installed_libs: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {f: getattr(self, f) for f in self.__struct_fields__}


__all__ = [
    "DEVICE_ABSENT",
    "DEVICE_PRESENT",
    "DEVICE_UNREADABLE",
    "CpuAllowance",
    "CudaState",
    "HostFacts",
    "cgroup_nodes",
    "cpu_allowance",
    "cpu_quota",
    "cpu_quota_raw",
    "cuda_ready",
    "cuda_state",
    "device_count",
    "device_identity",
    "free_vram_bytes",
    "headroom_bytes",
    "PROC_MEMINFO",
    "meminfo_kb",
    "process_ceiling_bytes",
    "reset_cuda_state",
    "total_vram_bytes",
]
