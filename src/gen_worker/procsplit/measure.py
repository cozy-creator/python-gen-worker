"""Pre-import hardware + canary measurement for the control parent (delta 2).

A handful of floats a worker reports about itself become FLEET-WIDE verdicts:
``HardwareUnsuitable`` fences a machine; ``HostCanary`` condemns a SKU on the
SPFabricLedger; the reported ``gpu_name`` chooses which verdict key gets written.
Measured inside the worker process — which has already imported tenant endpoint
code — any of them could be replaced by a handler, a module import side effect,
or a monkeypatched ``torch``. The hub-side corroboration gate contains that; this
removes the ability.

The control parent must stay torch-free (it is the process that survives a CUDA
death), so it cannot measure the silicon itself. It runs THIS module as a
short-lived subprocess instead, before and independently of any compute child:

    python -m gen_worker.procsplit.measure

It imports gen_worker and torch and nothing of the tenant's — no endpoint
module is ever named — measures, prints one JSON object, and exits. The parent
keeps the numbers and stamps them onto every Hello it relays. There is no
window in which tenant code has run and the measurement has not.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # heavy edges stay off this module's import scope
    from ..hostfacts import HostFacts

logger = logging.getLogger(__name__)


def measure() -> Dict[str, Any]:
    """Hardware facts, the boot host canary, and the build identity."""
    out: Dict[str, Any] = {"hardware": {}, "canary": None, "gen_worker_version": ""}
    try:
        out["hardware"] = probe_hardware().as_dict()
    except Exception as exc:  # never fatal: an unmeasured axis is a zero
        out["hardware_error"] = f"{type(exc).__name__}: {exc}"
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
        from ..compile_cache import gen_worker_version

        out["gen_worker_version"] = gen_worker_version()
    except Exception:
        pass
    return out


def main() -> int:
    # One JSON object on stdout and nothing else. Every log line the imports
    # emit goes to stderr, where the parent forwards it to the pod log.
    sys.stdout.write(json.dumps(measure()))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# pgw#1373: `probe_hardware` LIVES HERE NOW. It came from the deleted
# `lifecycle.py`, and its obvious home looked like `hostfacts` — it returns a
# `HostFacts`. That was wrong, and `lint_serve_role_closure` said so
# immediately: the probe needs `topology` and `models.hub_policy`, and putting
# it in `hostfacts` put that edge on the MODEL-FREE serve surface, dragging
# `models.residency -> models.memory -> structure_only -> diffusers` onto a
# path whose whole point is that it never imports a model library. `hostfacts`
# says "import this module freely" in its own header and has to keep meaning it.
#
# This module is the only caller and it runs in a DEDICATED measurement
# subprocess, so weight here costs nothing.
#
# Leaving the function deleted was a SILENT degradation, which is why it is
# restored rather than dropped: the `except Exception` above turns an
# ImportError into `hardware_error` plus an EMPTY hardware dict, so the parent
# shipped a measurement with no gpu_name, no gpu_count and no torch version,
# and every consumer read those zeros as a CPU-only box instead of as a missing
# measurement.


def probe_hardware() -> "HostFacts":
    """Measure this host ONCE into one immutable :class:`HostFacts`.

    The single producer of the facts the fleet acts on. ``vram_free_bytes`` is
    the input `gate_functions` turns into `unavailable` + `serve_plans` for
    EVERY function, so on a wide pod it must be the free pool of the group with
    the LEAST room, not card 0's. The MIN is the honest single scalar until the
    fit ladder is per-rank: it never promises a group room it does not have.
    """
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
    # The HOST driver, read from NVML rather than torch: it must stay readable
    # when the CUDA runtime is not. A cu130 build on a 570.x host imports fine,
    # reports every version string correctly, and dies on its first allocation.
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
        # A WEDGED fabric is the one topology fault that must reach the caller:
        # `delivered_topology` raises it so the hub re-packs instead of this
        # worker booting and hanging every collective. Swallowed here it read
        # as "card 0, fine" and the pod went on to serve.
        raise
    except Exception:
        pass
    try:
        from .models.hub_policy import detect_worker_capabilities

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
    """The free pool of the least-roomy execution group, or 0 if unknowable.

    Reads the DELIVERED topology: on a pod this worker itself demoted for a
    non-NVLink fabric, ``from_env`` describes a packing that is not being
    served, so reporting from it makes reported and served topologies disagree.
    """

    from ..topology import delivered_topology

    groups = delivered_topology().all_groups()
    return int(min((g.free_vram_bytes() for g in groups), default=0))
