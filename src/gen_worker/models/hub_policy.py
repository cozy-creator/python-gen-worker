from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from ..hostfacts import cuda_ready


@dataclass(frozen=True)
class TensorhubWorkerCapabilities:
    cuda_version: str
    gpu_sm: int
    torch_version: str
    installed_libs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_version": self.cuda_version,
            "gpu_sm": self.gpu_sm,
            "torch_version": self.torch_version,
            "installed_libs": list(self.installed_libs),
        }


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def detect_worker_capabilities(*, extra_libs: Optional[List[str]] = None) -> TensorhubWorkerCapabilities:
    """Detect worker capabilities for Cozy Hub artifact selection."""
    installed: List[str] = []

    known = ["bitsandbytes", "torchao", "transformer_engine",
             "deepcompressor", "modelopt"]
    if extra_libs:
        known.extend(extra_libs)
    for name in known:
        mod = name
        if name == "transformer_engine":
            mod = "transformer_engine"
        if _is_importable(mod):
            installed.append(name)
    cuda_version = ""
    gpu_sm = 0
    torch_version = ""
    try:
        import torch

        torch_version = str(getattr(torch, "__version__", "") or "")
        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        if getattr(torch, "cuda", None) is not None and cuda_ready():
            major, minor = torch.cuda.get_device_capability()
            gpu_sm = int(major) * 10 + int(minor)
    except Exception:
        pass

    installed.sort()
    return TensorhubWorkerCapabilities(
        cuda_version=cuda_version,
        gpu_sm=gpu_sm,
        torch_version=torch_version,
        installed_libs=installed,
    )


FIT_FITS = "fits"
FIT_INCOMPATIBLE = "incompatible"

def variant_fit(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
) -> tuple[str, str]:
    """Fit verdict for ONE function's Resources on this machine — exactly TWO verdicts: "incompatible" (no CUDA device at all, or a quant library this build lacks) and "fits" (no opinion about the card's size; real placement is measured at load by models/memory). free_vram_gb stays in the signature and is DELIBERATELY UNUSED: re-admitting it is the single edit that would rebuild the estimate-as-floor th#1867 deleted — nothing may reintroduce a size comparison here."""
    needs_gpu = bool(getattr(resources, "gpu", False))
    libs = tuple(getattr(resources, "libraries", ()) or ())
    if needs_gpu and caps.gpu_sm <= 0:
        return FIT_INCOMPATIBLE, "no CUDA GPU detected"
    missing = [lib for lib in libs if lib not in (caps.installed_libs or [])]
    if missing:
        return FIT_INCOMPATIBLE, f"missing libraries: {', '.join(missing)}"
    return FIT_FITS, ""

