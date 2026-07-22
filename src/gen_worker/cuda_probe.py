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


def classify_probe_failure(reason: str) -> str:
    """Typed vocabulary for ``CudaProbeResult.reason`` (gw#619) — the wire
    class the hub's pod_events row and death-taxonomy correlation key on.
    Never free-form: torch_unavailable | cuda_unavailable | driver_too_old |
    cuda_error | unknown. ``driver_too_old`` matches torch's own CUDA
    initialization message ("driver too old (found version ...)", the exact
    th#591/th#979 signature) — a strictly stronger, in-container-measured
    version of the same fact th#979's pre-rent provider-telemetry floor
    infers from the outside.
    """
    r = (reason or "").strip().lower()
    if not r:
        return "unknown"
    if "torch unavailable" in r:
        return "torch_unavailable"
    if "is_available() is false" in r:
        return "cuda_unavailable"
    if "driver too old" in r or "found version" in r:
        return "driver_too_old"
    return "cuda_error"


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


def should_probe_cuda(
    manifest: Optional[dict[str, Any]], *, cuda_build: Optional[bool] = None
) -> bool:
    """Whether this concrete worker image must pass the CUDA health probe.

    A manifest may contain both CPU and GPU functions because one endpoint
    release can publish separate ``accelerator=none`` and ``accelerator=cuda``
    images.  In that mixed case the installed torch build is the authoritative
    signal: probe CUDA images and let CPU-only images serve the CPU lane.  A
    GPU-only manifest is always probed so an accidentally CPU-built image
    fails before it can register.
    """
    functions = (manifest or {}).get("functions", []) or []
    gpu_requirements = [bool((fn.get("resources") or {}).get("gpu")) for fn in functions]
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
