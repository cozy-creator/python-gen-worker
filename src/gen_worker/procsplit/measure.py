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
import sys
from typing import Any, Dict


def measure() -> Dict[str, Any]:
    """Hardware facts, the boot host canary, and the build identity."""
    out: Dict[str, Any] = {"hardware": {}, "canary": None, "gen_worker_version": ""}
    try:
        from ..lifecycle import probe_hardware

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
