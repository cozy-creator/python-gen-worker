"""Boot-time CUDA health probe (gw#529).

RunPod occasionally allocates a host whose CUDA device is present but wedged
("CUDA-capable device(s) is/are busy or unavailable"). Left unchecked, the
worker says hello, accepts a job, and terminal-fails it at model load —
burning the invoker's request on a provider fault we could catch first.
Call ``probe_cuda`` before the orchestrator handshake; on failure the caller
logs ``CUDA_PROBE_FAILED_MARKER`` and exits nonzero so the pod is replaced.
"""

from __future__ import annotations

from typing import Any, Optional

import msgspec

CUDA_PROBE_FAILED_MARKER = "GEN_WORKER_CUDA_PROBE_FAILED"


class CudaProbeResult(msgspec.Struct, frozen=True):
    ok: bool
    reason: str = ""


def probe_cuda(device_index: int = 0) -> CudaProbeResult:
    """Allocate, run one op on, sync, and free a tiny tensor on
    ``device_index``. Never raises — every failure (import, is_available,
    alloc, op, sync) becomes a typed ``CudaProbeResult(ok=False, reason=...)``.
    """
    try:
        import torch
    except Exception as e:
        return CudaProbeResult(ok=False, reason=f"torch unavailable: {e}")

    try:
        if not torch.cuda.is_available():
            return CudaProbeResult(ok=False, reason="torch.cuda.is_available() is False")
        t = torch.ones(2, 2, device=f"cuda:{device_index}")
        t = t + 1
        torch.cuda.synchronize(device_index)
        del t
        torch.cuda.empty_cache()
    except Exception as e:
        return CudaProbeResult(ok=False, reason=f"{type(e).__name__}: {e}")

    return CudaProbeResult(ok=True)


def manifest_needs_cuda(manifest: Optional[dict[str, Any]]) -> bool:
    """True iff any function in the discovery manifest declares
    ``resources.gpu`` — the only build-time-known signal of whether this
    worker is a GPU (accelerator=cuda) vs CPU (accelerator=none) deployment."""
    if not manifest:
        return False
    for fn in manifest.get("functions", []) or []:
        if bool((fn.get("resources") or {}).get("gpu")):
            return True
    return False
