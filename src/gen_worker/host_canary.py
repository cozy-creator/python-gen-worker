"""Boot host canary: measure the host ONCE, report at registration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from .cuda_probe import NVIDIA_SMI_TIMEOUT_S as _NVIDIA_SMI_TIMEOUT_S
from .postmortem import effective_cpu_count
import tempfile
from .hostfacts import cuda_ready

logger = logging.getLogger(__name__)

_BUF_BYTES = 256 << 20
_MEMCPY_REPS = 3
_PCIE_REPS = 3
_CPU_SLICE_S = 0.25
_HASH_BLOCK = 1 << 20
_PEER_REPS = 3

INTERCONNECT_NONE = ""
INTERCONNECT_HOST_STAGED = "host-staged"
INTERCONNECT_PCIE_P2P = "pcie-p2p"
INTERCONNECT_NVLINK = "nvlink"

# SP_MIN_PEER_GBPS is the measured-bandwidth floor a pod must clear to carry a platform-sharded group, IN ADDITION to classifying nvlink — tensorhub's topology.SPMinPeerGbps verbatim. Hub and worker gate independently on the same measurement with the same two-term predicate, so the constants must move together; there is deliberately no HelloAck demote field.
SP_MIN_PEER_GBPS = 200.0


def sp_admits(interconnect: str, peer_gbps: float) -> bool:
    """Whether this pod's MEASURED fabric may carry a platform-sharded group."""
    return interconnect == INTERCONNECT_NVLINK and peer_gbps >= SP_MIN_PEER_GBPS


def is_fabric_wedge(peer_access: bool, peer_gbps: float) -> bool:
    """An NCCL WEDGE, not a slow host: peer access reported, bandwidth measured exactly zero."""
    return peer_access and peer_gbps == 0.0

PRODUCTION_ACTIVATION_SHAPE: Tuple[int, ...] = (1, 40, 37800, 128)
PRODUCTION_COLLECTIVES_PER_CALL = 160


@dataclass(frozen=True)
class HostCanaryReport:
    """One boot-time host measurement (zeros = axis not measurable)."""

    memcpy_gbps: float = 0.0
    h2d_gbps: float = 0.0
    d2h_gbps: float = 0.0
    pinned_alloc_ok: bool = False
    cpu_single_mbps: float = 0.0
    cpu_multi_mbps: float = 0.0
    vcpus: int = 0
    ram_total_gb: float = 0.0
    duration_ms: int = 0
    gpu_count: int = 0
    interconnect: str = INTERCONNECT_NONE
    peer_gbps: float = 0.0
    peer_access: bool = False
    topo_link: str = ""


def _measure_memcpy_gbps() -> float:
    import numpy as np

    src = np.ones(_BUF_BYTES, dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)
    t0 = time.perf_counter()
    for _ in range(_MEMCPY_REPS):
        np.copyto(dst, src)
    dt = time.perf_counter() - t0
    return (_MEMCPY_REPS * _BUF_BYTES) / dt / 1e9 if dt > 0 else 0.0


def _measure_cpu_mbps(workers: int) -> float:

    block = b"\xa5" * _HASH_BLOCK

    def one() -> int:
        n = 0
        h = hashlib.sha256()
        deadline = time.perf_counter() + _CPU_SLICE_S
        while time.perf_counter() < deadline:
            h.update(block)
            n += _HASH_BLOCK
        return n

    if workers <= 1:
        t0 = time.perf_counter()
        total = one()
        dt = time.perf_counter() - t0
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            t0 = time.perf_counter()
            total = sum(pool.map(lambda _: one(), range(workers)))
            dt = time.perf_counter() - t0
    return total / dt / 1e6 if dt > 0 else 0.0


def _measure_pcie() -> tuple[float, float, bool]:
    try:
        import torch

        if not cuda_ready():
            return 0.0, 0.0, False
    except Exception:
        return 0.0, 0.0, False

    pinned_ok = True
    try:
        host = torch.empty(_BUF_BYTES, dtype=torch.uint8, pin_memory=True)
    except Exception:
        pinned_ok = False
        host = torch.empty(_BUF_BYTES, dtype=torch.uint8)
    try:
        dev = torch.empty(_BUF_BYTES, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            dev.copy_(host, non_blocking=True)
        stream.synchronize()

        def bw(direction: str) -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(_PCIE_REPS):
                    if direction == "h2d":
                        dev.copy_(host, non_blocking=True)
                    else:
                        host.copy_(dev, non_blocking=True)
                end.record(stream)
            end.synchronize()
            ms = start.elapsed_time(end)
            return (_PCIE_REPS * _BUF_BYTES) / (ms / 1e3) / 1e9 if ms > 0 else 0.0

        h2d = bw("h2d")
        d2h = bw("d2h")
        return h2d, d2h, pinned_ok
    except Exception:
        logger.warning("host canary: PCIe probe failed", exc_info=True)
        return 0.0, 0.0, pinned_ok


def parse_nvidia_smi_topo(text: str, a: int, b: int) -> str:
    """The raw link code ``nvidia-smi topo -m`` reports for the (a, b) pair."""
    header: Optional[List[str]] = None
    for raw in text.splitlines():
        graphs = raw.split()
        if not graphs:
            continue
        if header is None:
            if any(c.upper().startswith("GPU") for c in graphs):
                header = [c.upper() for c in graphs]
            continue
        if graphs[0].upper() != f"GPU{a}":
            continue
        try:
            col = header.index(f"GPU{b}")
        except ValueError:
            return ""
        values = graphs[1:]
        if col < len(values):
            return values[col].strip()
        return ""
    return ""


def _nvidia_smi_topo_link(a: int, b: int) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except Exception:
        return ""
    if out.returncode != 0:
        return ""
    try:
        return parse_nvidia_smi_topo(out.stdout, a, b)
    except Exception:
        return ""


def classify_interconnect(topo_link: str, peer_access: bool) -> str:
    """Map (topology code, peer-access capability) onto the fabric class the latency model cares about."""
    link = (topo_link or "").strip().upper()
    if not peer_access:
        return INTERCONNECT_HOST_STAGED
    if link.startswith("NV"):
        return INTERCONNECT_NVLINK
    if link == "SYS":
        return INTERCONNECT_HOST_STAGED
    return INTERCONNECT_PCIE_P2P


def _measure_peer(devices: Tuple[int, int] = (0, 1)) -> Tuple[str, float, bool, str]:
    a, b = int(devices[0]), int(devices[1])
    try:
        import torch

        if not cuda_ready() or torch.cuda.device_count() <= max(a, b):
            return INTERCONNECT_NONE, 0.0, False, ""
    except Exception:
        return INTERCONNECT_NONE, 0.0, False, ""

    peer_access = False
    try:
        peer_access = bool(torch.cuda.can_device_access_peer(a, b))
    except Exception:
        logger.warning("host canary: peer-access query failed", exc_info=True)
    topo_link = _nvidia_smi_topo_link(a, b)
    interconnect = classify_interconnect(topo_link, peer_access)

    gbps = 0.0
    try:
        src = torch.empty(_BUF_BYTES, dtype=torch.uint8, device=f"cuda:{a}")
        dst = torch.empty(_BUF_BYTES, dtype=torch.uint8, device=f"cuda:{b}")
        stream = torch.cuda.Stream(device=a)
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(_PEER_REPS):
                dst.copy_(src, non_blocking=True)
            end.record(stream)
        end.synchronize()
        ms = start.elapsed_time(end)
        gbps = (_PEER_REPS * _BUF_BYTES) / (ms / 1e3) / 1e9 if ms > 0 else 0.0
        del src, dst
        torch.cuda.empty_cache()
    except Exception:
        logger.warning("host canary: peer bandwidth probe failed", exc_info=True)

    return interconnect, gbps, peer_access, topo_link


def measure_host_canary() -> HostCanaryReport:
    """Run every axis once; failures zero their axis instead of raising."""
    t0 = time.perf_counter()
    memcpy = single = multi = 0.0
    vcpus = effective_cpu_count()
    try:
        memcpy = _measure_memcpy_gbps()
    except Exception:
        logger.warning("host canary: memcpy probe failed", exc_info=True)
    try:
        single = _measure_cpu_mbps(1)
        multi = _measure_cpu_mbps(min(vcpus, 16)) if vcpus > 1 else single
    except Exception:
        logger.warning("host canary: cpu probe failed", exc_info=True)
    h2d, d2h, pinned_ok = _measure_pcie()
    ram_total = 0.0
    try:
        from .models.memory import probe_host_ram

        ram_total = probe_host_ram().total_gb
    except Exception:
        pass
    gpu_count = 0
    try:
        import torch

        gpu_count = int(torch.cuda.device_count()) if cuda_ready() else 0
    except Exception:
        pass
    interconnect, peer_gbps, peer_access, topo_link = (
        _measure_peer() if gpu_count > 1 else (INTERCONNECT_NONE, 0.0, False, "")
    )
    report = HostCanaryReport(
        memcpy_gbps=round(memcpy, 2),
        h2d_gbps=round(h2d, 2),
        d2h_gbps=round(d2h, 2),
        pinned_alloc_ok=pinned_ok,
        cpu_single_mbps=round(single, 1),
        cpu_multi_mbps=round(multi, 1),
        vcpus=vcpus,
        ram_total_gb=round(ram_total, 1),
        gpu_count=gpu_count,
        interconnect=interconnect,
        peer_gbps=round(peer_gbps, 2),
        peer_access=peer_access,
        topo_link=topo_link,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )
    logger.info(
        "HOST_CANARY memcpy_gbps=%.2f h2d_gbps=%.2f d2h_gbps=%.2f "
        "pinned_alloc_ok=%s cpu_single_mbps=%.1f cpu_multi_mbps=%.1f "
        "vcpus=%d ram_total_gb=%.1f gpu_count=%d interconnect=%s "
        "peer_gbps=%.2f peer_access=%s topo_link=%s duration_ms=%d",
        report.memcpy_gbps, report.h2d_gbps, report.d2h_gbps,
        report.pinned_alloc_ok, report.cpu_single_mbps, report.cpu_multi_mbps,
        report.vcpus, report.ram_total_gb, report.gpu_count,
        report.interconnect or "-", report.peer_gbps, report.peer_access,
        report.topo_link or "-", report.duration_ms,
    )
    return report


_cached: Optional[HostCanaryReport] = None


def get_host_canary() -> HostCanaryReport:
    """Process-once cached measurement (boot rides the pre-READY window)."""
    global _cached
    if _cached is None:
        _cached = measure_host_canary()
    return _cached


def _collective_rank_main(
    rank: int, world_size: int, shape: Tuple[int, ...], iters: int, out_dir: str,
) -> None:
    import torch
    import torch.distributed as dist

    result: Dict[str, Any] = {"rank": rank}
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        numel = 1
        for d in shape:
            numel *= int(d)
        buf = torch.ones(numel, dtype=torch.bfloat16, device=f"cuda:{rank}")
        src = buf.view(world_size, -1)
        dst = torch.empty_like(src)
        tensor_bytes = buf.element_size() * numel

        def timed(n: int) -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            start.record()
            for _ in range(n):
                dist.all_to_all_single(dst, src)
            end.record()
            torch.cuda.synchronize()
            return float(start.elapsed_time(end)) / n

        timed(3)
        ms = timed(iters)
        over_link = tensor_bytes * (world_size - 1) / world_size
        result.update(
            ok=True,
            world_size=world_size,
            shape=list(shape),
            iters=iters,
            tensor_bytes=tensor_bytes,
            ms_per_call=round(ms, 4),
            link_gbps=round(over_link / (ms / 1e3) / 1e9, 2) if ms > 0 else 0.0,
            algo_gbps=round(tensor_bytes / (ms / 1e3) / 1e9, 2) if ms > 0 else 0.0,
            model_call_ms=round(ms * PRODUCTION_COLLECTIVES_PER_CALL, 2),
            nccl_p2p_disable=os.environ.get("NCCL_P2P_DISABLE", ""),
        )
    except Exception as exc:  # pragma: no cover - pod-only path
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            import torch.distributed as _d

            if _d.is_initialized():
                _d.destroy_process_group()
        except Exception:
            pass
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(result, fh)


def measure_peer_collective(
    world_size: int = 2,
    *,
    shape: Tuple[int, ...] = PRODUCTION_ACTIVATION_SHAPE,
    iters: int = 20,
    p2p_disable: bool = False,
    port: int = 29577,
) -> Dict[str, Any]:
    """Ground truth for the cheap leg: a real NCCL ``all_to_all_single`` on the production activation shape, across ``world_size`` spawned ranks."""

    import torch
    import torch.multiprocessing as mp

    if not cuda_ready() or torch.cuda.device_count() < world_size:
        return {"ok": False, "error": f"needs {world_size} visible CUDA devices"}

    env_saved = {k: os.environ.get(k) for k in
                 ("MASTER_ADDR", "MASTER_PORT", "NCCL_P2P_DISABLE")}
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    if p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    else:
        os.environ.pop("NCCL_P2P_DISABLE", None)
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            mp.spawn(
                _collective_rank_main,
                args=(world_size, tuple(shape), int(iters), out_dir),
                nprocs=world_size,
                join=True,
            )
            ranks = []
            for r in range(world_size):
                path = os.path.join(out_dir, f"rank{r}.json")
                if os.path.exists(path):
                    with open(path) as fh:
                        ranks.append(json.load(fh))
            if not ranks:
                return {"ok": False, "error": "no rank produced a result"}
            head = dict(ranks[0])
            head["p2p_disable"] = p2p_disable
            head["ranks"] = ranks
            return head
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "p2p_disable": p2p_disable}
    finally:
        for k, v in env_saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _main() -> None:  # pragma: no cover - operator entry point
    logging.basicConfig(level=logging.INFO)
    report = measure_host_canary()
    out: Dict[str, Any] = {"canary": asdict(report)}
    if report.gpu_count > 1:
        out["collective_p2p"] = measure_peer_collective(min(report.gpu_count, 2))
        out["collective_no_p2p"] = measure_peer_collective(
            min(report.gpu_count, 2), p2p_disable=True, port=29578)
    print(json.dumps(out, indent=2, sort_keys=True))


__all__ = [
    "HostCanaryReport",
    "get_host_canary",
    "measure_host_canary",
    "measure_peer_collective",
    "classify_interconnect",
    "parse_nvidia_smi_topo",
    "PRODUCTION_ACTIVATION_SHAPE",
    "SP_MIN_PEER_GBPS",
    "sp_admits",
    "is_fabric_wedge",
]


if __name__ == "__main__":  # pragma: no cover
    _main()
