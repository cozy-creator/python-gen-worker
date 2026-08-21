"""Boot-time CUDA health probe."""

from __future__ import annotations

from typing import Any, Optional

import msgspec
from .hostfacts import cuda_ready
from .manifest_blocks import declaration_rows

CUDA_PROBE_FAILED_MARKER = "GEN_WORKER_CUDA_PROBE_FAILED"

NO_DEVICE_REASON = "hostfacts.cuda_ready() is False"

# Wall budget for one nvidia-smi subprocess: the wedged-CUDA fault this module diagnoses is also the fault that makes nvidia-smi hang indefinitely, so a call with no timeout turns the diagnostic into the hang it was diagnosing.
NVIDIA_SMI_TIMEOUT_S = 5.0


class CudaProbeResult(msgspec.Struct, frozen=True):
    ok: bool
    reason: str = ""


def probe_cuda(device_index: int = 0) -> CudaProbeResult:
    """Allocate, run one op on, sync, and free a tiny tensor on ``device_index``."""
    try:
        import torch
    except Exception as e:
        return CudaProbeResult(ok=False, reason=f"torch unavailable: {e}")

    try:
        if not cuda_ready():
            return CudaProbeResult(ok=False, reason=NO_DEVICE_REASON)
        t = torch.ones(2, 2, device=f"cuda:{device_index}")
        t = t + 1
        torch.cuda.synchronize(device_index)
        del t
        torch.cuda.empty_cache()
    except Exception as e:
        return CudaProbeResult(ok=False, reason=f"{type(e).__name__}: {e}")

    return CudaProbeResult(ok=True)


CARDLESS_PROBE_CLASSES = frozenset({"cuda_unavailable", "torch_unavailable"})


def classify_probe_failure(reason: str) -> str:
    """Typed vocabulary for ``CudaProbeResult.reason`` — the wire class the hub's pod_events row and death-taxonomy correlation key on."""
    r = (reason or "").strip().lower()
    if not r:
        return "unknown"
    if "torch unavailable" in r:
        return "torch_unavailable"
    if NO_DEVICE_REASON.lower() in r:
        return "cuda_unavailable"
    if "driver too old" in r or "found version" in r:
        return "driver_too_old"
    return "cuda_error"


def should_probe_cuda(
    manifest: Optional[dict[str, Any]], *, cuda_build: Optional[bool] = None
) -> bool:
    """Whether this concrete worker image must pass the CUDA health probe."""
    rows = declaration_rows(manifest)
    gpu_requirements = [bool((row.get("resources") or {}).get("gpu")) for row in rows]
    if not any(gpu_requirements):
        return False
    if all(gpu_requirements):
        return True
    if cuda_build is None:
        try:
            import torch

            cuda_build = bool(torch.version.cuda)
        except Exception:
            return True
    return cuda_build
