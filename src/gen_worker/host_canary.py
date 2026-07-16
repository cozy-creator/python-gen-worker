"""Boot host canary (gw#550): measure the host ONCE, report at registration.

ie#484 proved the GPU-pod host lottery lives entirely in the CPU-bound
stages: identical per-step denoise times across every host sampled, while
VAE-decode tails ranged 26-147 s and mp4-encode 27-301 s on the SAME job.
The hub's degraded-host watch (th#740) only trips AFTER slow completions.
This canary measures the three host properties those tails depend on —
host memcpy bandwidth, pinned PCIe H2D/D2H bandwidth, raw CPU throughput —
in a bounded ~1.5 s at boot, and ships them in ``Hello.resources`` so a bad
host is visible BEFORE it serves.

No config knobs: sizes and repetitions are fixed; thresholds live hub-side,
justified from the measured fleet distribution.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Fixed measurement geometry (bounded: ~6 memcpy passes over 256 MiB, 3 PCIe
# round-trips of 256 MiB, and <=0.6 s of hashing).
_BUF_BYTES = 256 << 20
_MEMCPY_REPS = 3
_PCIE_REPS = 3
_CPU_SLICE_S = 0.25
_HASH_BLOCK = 1 << 20


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


def _measure_memcpy_gbps() -> float:
    """Host-RAM copy bandwidth over a buffer far larger than LLC."""
    import numpy as np

    src = np.ones(_BUF_BYTES, dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)  # warm faults
    t0 = time.perf_counter()
    for _ in range(_MEMCPY_REPS):
        np.copyto(dst, src)
    dt = time.perf_counter() - t0
    return (_MEMCPY_REPS * _BUF_BYTES) / dt / 1e9 if dt > 0 else 0.0


def _measure_cpu_mbps(workers: int) -> float:
    """Aggregate sha256 throughput (MB/s) across ``workers`` threads.

    sha256 releases the GIL and stresses the ALU + memory system — a stable
    proxy for the x264/VAE-postprocess CPU work the encode tail is made of.
    """
    from concurrent.futures import ThreadPoolExecutor

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
    """Pinned H2D/D2H bandwidth (GB/s) — the exact transfer the gw#549 media
    staging path uses. Returns (h2d, d2h, pinned_alloc_ok); zeros sans CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
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
            dev.copy_(host, non_blocking=True)  # warm
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


def measure_host_canary() -> HostCanaryReport:
    """Run every axis once; failures zero their axis instead of raising."""
    t0 = time.perf_counter()
    memcpy = single = multi = 0.0
    vcpus = os.cpu_count() or 0
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
    report = HostCanaryReport(
        memcpy_gbps=round(memcpy, 2),
        h2d_gbps=round(h2d, 2),
        d2h_gbps=round(d2h, 2),
        pinned_alloc_ok=pinned_ok,
        cpu_single_mbps=round(single, 1),
        cpu_multi_mbps=round(multi, 1),
        vcpus=vcpus,
        ram_total_gb=round(ram_total, 1),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )
    logger.info(
        "HOST_CANARY memcpy_gbps=%.2f h2d_gbps=%.2f d2h_gbps=%.2f "
        "pinned_alloc_ok=%s cpu_single_mbps=%.1f cpu_multi_mbps=%.1f "
        "vcpus=%d ram_total_gb=%.1f duration_ms=%d",
        report.memcpy_gbps, report.h2d_gbps, report.d2h_gbps,
        report.pinned_alloc_ok, report.cpu_single_mbps, report.cpu_multi_mbps,
        report.vcpus, report.ram_total_gb, report.duration_ms,
    )
    return report


_cached: Optional[HostCanaryReport] = None


def get_host_canary() -> HostCanaryReport:
    """Process-once cached measurement (boot rides the pre-READY window)."""
    global _cached
    if _cached is None:
        _cached = measure_host_canary()
    return _cached


__all__ = ["HostCanaryReport", "get_host_canary", "measure_host_canary"]
