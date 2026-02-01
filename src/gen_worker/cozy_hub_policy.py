from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CozyHubWorkerCapabilities:
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


def detect_worker_capabilities(*, extra_libs: Optional[List[str]] = None) -> CozyHubWorkerCapabilities:
    """
    Detect worker capabilities for Cozy Hub artifact selection.

    This is intentionally conservative: if torch/cuda isn't available, we report
    empty/zero values so Cozy Hub can avoid selecting capability-gated artifacts
    (e.g. fp8 or flashpack) unless explicitly supported.
    """
    installed: List[str] = []

    # Known optional libs that affect artifact compatibility.
    # Keep this hardcoded (no env config), per Cozy design.
    known = ["flashpack", "bitsandbytes", "torchao", "transformer_engine"]
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
        import torch  # type: ignore

        torch_version = str(getattr(torch, "__version__", "") or "")
        cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            gpu_sm = int(major) * 10 + int(minor)
    except Exception:
        pass

    installed.sort()
    return CozyHubWorkerCapabilities(
        cuda_version=cuda_version,
        gpu_sm=gpu_sm,
        torch_version=torch_version,
        installed_libs=installed,
    )


def default_resolve_preferences() -> Dict[str, List[str]]:
    """
    Hardcoded worker-side preference order for Cozy Hub selection.

    Policy:
      - Prefer file_type: flashpack -> safetensors
      - Prefer quantization: fp8 -> bf16 -> fp16 (fp16 is a pragmatic fallback)
      - Prefer file_layout: diffusers
    """
    return {
        "file_type_preference": ["flashpack", "safetensors"],
        "quantization_preference": ["fp8", "bf16", "fp16"],
        "file_layout_preference": ["diffusers"],
    }

