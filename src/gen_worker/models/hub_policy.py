from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from .memory import effective_vram_requirement_gb
from .loading import EMERGENCY_FIT_FACTOR, FP8_STORAGE_FIT_FACTOR


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
    """
    Detect worker capabilities for Cozy Hub artifact selection.

    This is intentionally conservative: if torch/cuda isn't available, we report
    empty/zero values so Cozy Hub can avoid selecting capability-gated artifacts
    (e.g. fp8) unless explicitly supported.
    """
    installed: List[str] = []

    # Known optional libs that affect artifact compatibility.
    # Keep this hardcoded (no env config), per Cozy design.
    known = ["bitsandbytes", "torchao", "transformer_engine",
             "nunchaku", "deepcompressor", "modelopt"]
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
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
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


# ---------------------------------------------------------------------------
# Fit-verdict policy (#380) — classify ONE function's Resources on THIS
# machine. Pure logic over (Resources, capabilities, free VRAM); consumed by
# serve_fit.plan_serve (the executor's flavor-fit ladder) and `run --list`.
# The old select_variant ranking helper was deleted in pgw#527 — dead since
# `--variant auto` was removed (pgw#226/#515); ranking now lives hub-side.
# ---------------------------------------------------------------------------

FIT_FITS = "fits"
FIT_FP8 = "fp8"
FIT_NVFP4 = "nvfp4"
FIT_SVDQ_FP4 = "svdq_fp4"
FIT_SVDQ_INT4 = "svdq_int4"
FIT_EMERGENCY_FP8 = "emergency_fp8"
FIT_EMERGENCY = "emergency_quant"
FIT_GGUF = "gguf_quant"
FIT_OFFLOAD = "offload"
FIT_INCOMPATIBLE = "incompatible"

# pgw#1148 (§1.32(d), th#1803): the STORED-PRECISION classification of a
# binding is gone from this planner. It read `binding.flavor` — the
# arbitrary-string sub-selector Paul killed — and with that field deleted
# `svdq_flavor_kind`/`quant_flavor_kind` could only ever have answered "".
# A binding names a tag or a digest; WHAT THE BYTES ARE is the checkpoint's
# TENSOR-LAYOUT CONTRACT (§1.33), which the loaders gate on for real
# (`@implements_contract`, models/svdq_layout.py, w4a4.py, w8a8.py) instead
# of trusting a token in a ref. The svdq SM-window and nvfp4-Blackwell
# refusals therefore live where the artifact declares itself, not here.
# Re-deriving a LOCAL fit verdict from the resolved checkpoint's contract is
# a BUILD, filed as a pgw#1148 residual — this planner now answers on
# resources + declared cast alone, which is what it can honestly know.


def variant_fit(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
) -> tuple[str, str]:
    """Fit verdict for ONE function/variant's ``Resources`` on this machine.

    - ``incompatible``: hard gates unmet (no CUDA GPU, required quant
      libraries not installed).
    - ``emergency_fp8``: does not fit as stored, but runtime fp8-E4M3 weight
      storage (loading.apply_fp8_storage: fp8 bytes resident, bf16 compute,
      no fp8 silicon required) would fit — quality ~= a stored #fp8 flavor.
    - ``emergency_quant``: even the fp8 estimate does not fit, but the
      emergency nf4 rung (th#546 emergency lane; loading layer, automatic on
      CUDA hosts) applies and the 4-bit estimate fits — runs at
      below-platform quality, loudly.
    - ``offload``: runnable, but the recommended card size (vram_gb minus the
      fixed GPU reserve) exceeds free VRAM — the models/memory.py offload
      ladder carries it (slower).
    - ``fits``: full-VRAM residency expected.
    """
    needs_gpu = bool(getattr(resources, "gpu", False))
    libs = tuple(getattr(resources, "libraries", ()) or ())
    if needs_gpu and caps.gpu_sm <= 0:
        return FIT_INCOMPATIBLE, "no CUDA GPU detected"
    # SDK v2 (pgw#647): no declared compute-capability gate — the fit
    # ladder picks precision per card.
    missing = [lib for lib in libs if lib not in (caps.installed_libs or [])]
    if missing:
        return FIT_INCOMPATIBLE, f"missing libraries: {', '.join(missing)}"
    vram = getattr(resources, "vram_gb_hint", None)
    # vram_gb recommends a card SIZE (total VRAM), so an idle card of exactly
    # that size counts as fitting: subtract the fixed driver/framebuffer/CUDA
    # reserve before comparing against measured free VRAM.

    if vram is None or effective_vram_requirement_gb(vram) <= float(free_vram_gb):
        return FIT_FITS, ""

    # Runtime rungs are automatic on CUDA hosts (gw#420) — pure functions of
    # the declared capabilities, so verdicts don't depend on the probing host.
    if caps.gpu_sm > 0:
        # fp8-E4M3 runtime storage: weights ~halve, bf16 compute, quality
        # ~= a stored #fp8 flavor — try before dropping to 4-bit.
        if float(vram) * FP8_STORAGE_FIT_FACTOR <= float(free_vram_gb):
            return FIT_EMERGENCY_FP8, (
                f"runs (fp8 storage): {float(vram):g} GB VRAM via runtime "
                f"fp8-E4M3 weight storage, {float(free_vram_gb):.1f} GB free"
            )
        if float(vram) * EMERGENCY_FIT_FACTOR <= float(free_vram_gb):
            return FIT_EMERGENCY, (
                f"runs (emergency quality): {float(vram):g} GB VRAM via 4-bit "
                f"emergency quantization, {float(free_vram_gb):.1f} GB free"
            )
    return FIT_OFFLOAD, (
        f"declares {float(vram):g} GB VRAM, {float(free_vram_gb):.1f} GB free"
    )

