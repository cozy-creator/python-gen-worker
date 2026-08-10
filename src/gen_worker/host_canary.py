"""Boot host canary (gw#550): measure the host ONCE, report at registration.

ie#484 proved the GPU-pod host lottery lives entirely in the CPU-bound
stages: identical per-step denoise times across every host sampled, while
VAE-decode tails ranged 26-147 s and mp4-encode 27-301 s on the SAME job.
The hub's degraded-host watch (th#740) only trips AFTER slow completions.
This canary measures the three host properties those tails depend on —
host memcpy bandwidth, pinned PCIe H2D/D2H bandwidth, raw CPU throughput —
in a bounded ~1.5 s at boot, and ships them in ``Hello.resources`` so a bad
host is visible BEFORE it serves.

pgw#748 phase 0 adds a **2-GPU leg** on the same principle. Sequence
parallelism's whole latency case rests on the GPU-to-GPU fabric a delivered
pod actually has, and the hub cannot infer it: the SKU says SXM, the pod
says nothing about whether ITS two cards have peer access. The leg measures
what the fabric really is (nvlink / pcie-p2p / host-staged) plus the achieved
pairwise bandwidth, so the biggest unknown in the design becomes a fleet
observable at every boot rather than a one-off probe.

Two depths, one file:

- :func:`measure_host_canary` runs the **cheap** leg — topology, peer access,
  and a timed cross-device copy. Bounded, boot-safe, no subprocesses.
- :func:`measure_peer_collective` runs the **deep** leg — a real NCCL
  ``all_to_all_single`` on the production activation shape across N spawned
  ranks. Never at boot (it costs seconds and forks); it is the seqpar-p0
  probe, kept here so the cheap leg's classification can be validated
  against ground truth on any multi-GPU pod we ever rent.
  ``python -m gen_worker.host_canary`` runs both and prints JSON.

No config knobs: sizes and repetitions are fixed; thresholds live hub-side,
justified from the measured fleet distribution.
"""

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

logger = logging.getLogger(__name__)

# Fixed measurement geometry (bounded: ~6 memcpy passes over 256 MiB, 3 PCIe
# round-trips of 256 MiB, and <=0.6 s of hashing).
_BUF_BYTES = 256 << 20
_MEMCPY_REPS = 3
_PCIE_REPS = 3
_CPU_SLICE_S = 0.25
_HASH_BLOCK = 1 << 20
# 2-GPU leg (pgw#748): same 256 MiB buffer, 3 timed peer copies.
_PEER_REPS = 3

# Measured interconnect classes, worst to best.
INTERCONNECT_NONE = ""            # single GPU / not measurable
INTERCONNECT_HOST_STAGED = "host-staged"
INTERCONNECT_PCIE_P2P = "pcie-p2p"
INTERCONNECT_NVLINK = "nvlink"

# SP_MIN_PEER_GBPS (pgw#818) is the measured-bandwidth floor a pod must clear
# to carry a platform-sharded group, IN ADDITION to classifying ``nvlink`` —
# the hub's ``topology.SPMinPeerGbps`` (tensorhub topology/interconnect.go),
# verbatim. Class alone is a string a degraded NV4 host also prints. The
# 2026-07-29 fleet survey found the populations separated by an empty band
# under EITHER leg a canary might report into ``peer_gbps``:
#
#   achieved all-to-all    NVLink 241.9 - 273.9  |  everything else <= 30.2
#   device-to-device copy  NVLink 388.2 - 389.8  |  everything else <= 52.9
#
# 200 GB/s sits inside both gaps. Hub and worker gate INDEPENDENTLY on the
# same measurement with the same two-term predicate (th#1285 interpretation 4
# holds only while the predicates match — pgw#818 is what happened when one
# side grew a floor alone); there is deliberately no HelloAck demote field.
SP_MIN_PEER_GBPS = 200.0


def sp_admits(interconnect: str, peer_gbps: float) -> bool:
    """Whether this pod's MEASURED fabric may carry a platform-sharded group.

    Both terms required: the class must be ``nvlink`` AND the bandwidth must
    clear ``SP_MIN_PEER_GBPS``. Unknown/unmeasured admits nothing — the fast
    tier's promise is only kept when the pod proved it.
    """
    return interconnect == INTERCONNECT_NVLINK and peer_gbps >= SP_MIN_PEER_GBPS


def is_fabric_wedge(peer_access: bool, peer_gbps: float) -> bool:
    """An NCCL WEDGE, not a slow host: peer access reported, bandwidth
    measured exactly zero. Reproduced twice on machine 8n9k05n0sz03 (H100
    NVL, SECURE): the collective HUNG — no exception, no timeout, no log. A
    pod serving in that state strands every request routed to it. Zero
    WITHOUT peer access is just "not measured" (a 1-GPU pod), no verdict.
    Mirrors the hub's ``topology.IsFabricWedge``.
    """
    return peer_access and peer_gbps == 0.0

# The wan-2.2 A14B production activation shape one Ulysses all-to-all moves
# (SEQPAR-DESIGN §3): [batch, heads, tokens, head_dim] bf16 = 387 MB.
PRODUCTION_ACTIVATION_SHAPE: Tuple[int, ...] = (1, 40, 37800, 128)
# 4 all-to-alls per layer x 40 layers = one model call's worth of collectives.
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
    # pgw#748 2-GPU leg. All inert on a 1-GPU pod: gpu_count<=1 leaves
    # interconnect "" and peer_gbps 0, which the hub reads as "not measured".
    gpu_count: int = 0
    interconnect: str = INTERCONNECT_NONE
    peer_gbps: float = 0.0
    peer_access: bool = False
    topo_link: str = ""  # raw nvidia-smi code for the first pair (NV18/PIX/SYS/...)


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


# ---------------------------------------------------------------------------
# The 2-GPU leg (pgw#748): what fabric does THIS pod actually have?
# ---------------------------------------------------------------------------


def parse_nvidia_smi_topo(text: str, a: int, b: int) -> str:
    """The raw link code ``nvidia-smi topo -m`` reports for the (a, b) pair.

    The matrix is a header row of GPU columns followed by one row per GPU;
    trailing columns (CPU/NUMA affinity) are not GPU columns and are skipped
    by indexing off the header's GPU labels rather than off the row end.
    Returns "" when the pair is not in the matrix.
    """
    header: Optional[List[str]] = None
    for raw in text.splitlines():
        cells = raw.split()
        if not cells:
            continue
        if header is None:
            if any(c.upper().startswith("GPU") for c in cells):
                header = [c.upper() for c in cells]
            continue
        if cells[0].upper() != f"GPU{a}":
            continue
        try:
            col = header.index(f"GPU{b}")
        except ValueError:
            return ""
        # Row cells after the row label line up with the header columns, but
        # "X" on the diagonal is printed as a bare token like every other.
        values = cells[1:]
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
    """Map (topology code, peer-access capability) onto the fabric class the
    latency model cares about. These are PERFORMANCE classes, not mechanisms:
    each names a row of the projected-wall table, and the pgw#748 probe
    measured all three on real RunPod pods.

    ``nvidia-smi`` reports NV# for an NVLink pair; PIX/PXB/PHB/NODE/SYS are
    PCIe paths of decreasing quality.

    Two rules, both measured rather than assumed:

    - **Peer access overrules good-looking wiring.** Without it every transfer
      is staged through host RAM however adjacent the cards look. Two RTX 4090s
      reporting ``NODE`` and ``peer_access=False`` achieved **1.96 GB/s** on an
      all-to-all — and the identical number with ``NCCL_P2P_DISABLE=1``, which
      is the proof NCCL was already host-staging.
    - **``SYS`` overrules peer access.** A cross-socket pair traverses the CPU
      interconnect, and the P2P flag then buys nothing: two H100 PCIe cards
      reporting ``SYS`` and ``peer_access=True`` achieved **14.5 GB/s**, again
      bit-identical with P2P disabled. That is the host-staged row of the
      table (assumed 15 GB/s), not the PCIe-P2P row (assumed 40). Calling it
      ``pcie-p2p`` would overstate the achievable wall by 2.8x.

    The raw code always rides alongside in ``topo_link``, so a consumer that
    wants the mechanism rather than the class still has it.
    """
    link = (topo_link or "").strip().upper()
    if not peer_access:
        return INTERCONNECT_HOST_STAGED
    if link.startswith("NV"):
        return INTERCONNECT_NVLINK
    if link == "SYS":
        return INTERCONNECT_HOST_STAGED
    return INTERCONNECT_PCIE_P2P


def _measure_peer(devices: Tuple[int, int] = (0, 1)) -> Tuple[str, float, bool, str]:
    """Cheap boot leg: (interconnect, peer_gbps, peer_access, topo_link).

    ``peer_gbps`` is a timed device-to-device copy of the same 256 MiB buffer
    the other legs use — the raw fabric throughput NCCL's collectives ride.
    """
    a, b = int(devices[0]), int(devices[1])
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() <= max(a, b):
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
            dst.copy_(src, non_blocking=True)  # warm (may allocate peer mappings)
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
    # gw#640: os.cpu_count() reports the HOST's cores — 32 on a pod that owns
    # 4 — and shipping that next to a cgroup-derived ram_total_gb produced a
    # "32 vCPUs / 14.9 GB" report nobody could interpret. Report what this
    # container may actually use, and benchmark with that many threads.

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

        gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
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


# ---------------------------------------------------------------------------
# The deep leg — the seqpar-p0 probe, kept next to the canary it validates.
# NEVER runs at boot: it forks ranks and costs seconds.
# ---------------------------------------------------------------------------


def _collective_rank_main(
    rank: int, world_size: int, shape: Tuple[int, ...], iters: int, out_dir: str,
) -> None:
    """One NCCL rank: init, time ``all_to_all_single``, write its JSON."""
    import torch
    import torch.distributed as dist

    result: Dict[str, Any] = {"rank": rank}
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # all_to_all_single splits along dim 0, so the production activation
        # is viewed as [world_size, -1]: identical byte volume, divisible.
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

        timed(3)  # warm
        ms = timed(iters)
        # Bytes that actually cross the fabric per rank per call: every chunk
        # but this rank's own.
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
    """Ground truth for the cheap leg: a real NCCL ``all_to_all_single`` on
    the production activation shape, across ``world_size`` spawned ranks.

    ``p2p_disable=True`` sets ``NCCL_P2P_DISABLE=1`` for the spawned ranks,
    which bounds the host-staged floor — NCCL reads it at init, so it can
    only be varied across process groups, never within one.
    """

    import torch
    import torch.multiprocessing as mp

    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
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
    """``python -m gen_worker.host_canary`` — the boot leg plus, on a
    multi-GPU host, both collective arms. Prints one JSON object."""
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
