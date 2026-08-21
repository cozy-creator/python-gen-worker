"""Pre-import hardware + canary measurement for the control parent (delta 2)."""

from __future__ import annotations

import json
import os
import logging
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from ..hostfacts import HostFacts

logger = logging.getLogger(__name__)


def measure() -> Dict[str, Any]:
    """Hardware facts, the boot host canary, and the build identity."""
    out: Dict[str, Any] = {"hardware": {}, "canary": None, "gen_worker_version": ""}
    attempts = _CENSUS_RETRIES if gpu_devices_present() else 1
    for attempt in range(attempts):
        try:
            facts = probe_hardware()
        except Exception as exc:
            out["hardware_error"] = f"{type(exc).__name__}: {exc}"
            break
        out.pop("hardware_error", None)
        out["hardware"] = facts.as_dict()
        gaps = _census_gaps(facts)
        out["census_gaps"] = list(gaps)
        if not gaps or attempt == attempts - 1:
            break
        backoff = _CENSUS_BACKOFF_S * (attempt + 1)
        logger.warning(
            "host census incomplete on attempt %d/%d (missing: %s) while %s "
            "exists — a GPU was assigned to this container, so this is "
            "UNREADABLE, not absent; retrying in %.1fs (pgw#1414/#1417)",
            attempt + 1, attempts, ", ".join(gaps),
            next((n for n in _GPU_DEVICE_NODES if os.path.exists(n)), "?"),
            backoff,
        )
        time.sleep(backoff)

    if "hardware_error" not in out and gpu_devices_present():
        facts_dict = out.get("hardware") or {}
        if (facts_dict.get("gpu_count") or facts_dict.get("gpu_name")
                or facts_dict.get("driver_version")) and not facts_dict.get("gpu_sm"):
            reason_class, detail = _capability_reason()
            out["capability_reason_class"] = reason_class
            out["capability_detail"] = detail
            out["capability_unreadable"] = (
                f"GPU {facts_dict.get('gpu_name') or '(unnamed)'} "
                f"(driver {facts_dict.get('driver_version') or '?'}) was read, "
                f"but its compute capability was not, across {attempts} "
                f"attempt(s). `gpu_sm` is empty, so every request carrying a "
                f"derived min_sm will refuse gpu_capability_incompatible. The "
                f"CUDA RUNTIME, not the driver, answers this — the driver is "
                f"clearly up. Probe says {reason_class}: {detail}"
            )
            logger.error(
                "capability_unreadable: %s (pgw#1417/#1436)",
                out["capability_unreadable"],
            )
        elif not (facts_dict.get("gpu_count") or facts_dict.get("gpu_name")
                  or facts_dict.get("driver_version")):
            out["census_unreadable"] = (
                f"GPU device nodes exist "
                f"({', '.join(n for n in _GPU_DEVICE_NODES if os.path.exists(n))}) "
                f"but {attempts} census attempt(s) read no driver, no device "
                f"and no name. A card may be present and not answering — this "
                f"host must NOT be registered as cpu-class."
            )
            logger.error(
                "census_unreadable: %s (pgw#1414)", out["census_unreadable"]
            )
    try:
        from ..host_canary import get_host_canary

        c = get_host_canary()
        out["canary"] = {
            "memcpy_gbps": c.memcpy_gbps,
            "d2h_gbps": c.d2h_gbps,
            "pinned_alloc_ok": c.pinned_alloc_ok,
            "cpu_single_mbps": c.cpu_single_mbps,
            "cpu_multi_mbps": c.cpu_multi_mbps,
            "vcpus": c.vcpus,
            "ram_total_gb": c.ram_total_gb,
            "duration_ms": c.duration_ms,
            "interconnect": c.interconnect,
            "peer_gbps": c.peer_gbps,
            "peer_access": c.peer_access,
            "topo_link": c.topo_link,
        }
    except Exception as exc:
        out["canary_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from ..toolchain import gen_worker_version

        out["gen_worker_version"] = gen_worker_version()
    except Exception:
        pass
    return out


def main() -> int:
    sys.stdout.write(json.dumps(measure()))
    sys.stdout.flush()
    return 0


_GPU_DEVICE_NODES = ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm")

_CENSUS_RETRIES = 4
_CENSUS_BACKOFF_S = 1.5


def _capability_reason() -> Tuple[str, str]:
    try:
        from ..hostfacts import cuda_state

        state = cuda_state()
        klass = (getattr(state, "probe_class", "") or "").strip() or "unknown"
        detail = (getattr(state, "detail", "") or "").strip()
        return klass, detail or "(probe returned no detail)"
    except Exception as exc:  # noqa: BLE001 — a probe never changes an outcome
        return "unknown", f"cuda_state() itself failed: {type(exc).__name__}: {exc}"


def gpu_devices_present() -> bool:
    """Whether this container was handed GPU device nodes."""
    return any(os.path.exists(node) for node in _GPU_DEVICE_NODES)


def _census_gaps(facts: "HostFacts") -> Tuple[str, ...]:
    if not (facts.gpu_count or facts.gpu_name or facts.driver_version):
        return ("device",)
    if not facts.gpu_sm:
        return ("capability",)
    return ()


def probe_hardware() -> "HostFacts":
    """Measure this host ONCE into one immutable :class:`HostFacts`."""
    from ..hostfacts import (
        HostFacts, device_count, device_identity, free_vram_bytes,
        total_vram_bytes,
    )
    from ..topology import TopologyError as _TopologyError

    gpu_count = 0
    vram_total = 0
    vram_free = 0
    gpu_name = ""
    gpu_sm = ""
    torch_version = ""
    cuda_version = ""
    driver_version = ""
    installed_libs: tuple[str, ...] = ()
    try:
        from ..hardware_report import _nvidia_smi_driver_and_gpu

        driver_version, gpu_name = _nvidia_smi_driver_and_gpu()
    except Exception:
        pass
    try:
        count = device_count()
        if count:
            gpu_count = count
            gpu_name = device_identity(0)[0] or gpu_name
            vram_total = total_vram_bytes(0) or 0
            card0_free = free_vram_bytes(0) or 0
            vram_free = card0_free
            worst = _worst_group_free_vram_bytes()
            if worst and worst != card0_free:
                logger.info(
                    "fit inputs: gating on the least-free group (%d bytes) "
                    "instead of card 0 (%d bytes) — pgw#776",
                    int(worst), int(card0_free))
                vram_free = int(worst)
    except _TopologyError:
        raise
    except Exception:
        pass
    try:
        from ..models.hub_policy import detect_worker_capabilities

        caps = detect_worker_capabilities()
        installed_libs = tuple(str(x) for x in (caps.installed_libs or []))
        torch_version = str(caps.torch_version or "")
        cuda_version = str(caps.cuda_version or "")
        if caps.gpu_sm:
            gpu_sm = str(int(caps.gpu_sm))
    except Exception:
        pass
    return HostFacts(
        gpu_count=gpu_count,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        gpu_name=gpu_name,
        gpu_sm=gpu_sm,
        torch_version=torch_version,
        cuda_version=cuda_version,
        driver_version=driver_version,
        installed_libs=installed_libs,
    )

def _worst_group_free_vram_bytes() -> int:

    from ..topology import delivered_topology

    groups = delivered_topology().all_groups()
    return int(min((g.free_vram_bytes() for g in groups), default=0))


if __name__ == "__main__":
    sys.exit(main())
