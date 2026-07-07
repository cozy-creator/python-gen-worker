from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
    known = ["bitsandbytes", "torchao", "transformer_engine", "tensorrt"]
    if extra_libs:
        known.extend(extra_libs)
    for name in known:
        mod = name
        if name == "transformer_engine":
            mod = "transformer_engine"
        if _is_importable(mod):
            installed.append(name)
    # TRT engine cells (#390) are version-locked plans: the hub needs the
    # runtime's trt version to resolve/schedule matching cells, so advertise
    # a second `tensorrt==<full version>` entry alongside the plain name
    # (plain-name library gating keeps working).
    if "tensorrt" in installed:
        try:
            import tensorrt  # type: ignore

            installed.append(f"tensorrt=={tensorrt.__version__}")
        except Exception:
            pass

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
# Variant auto-selection (#380) — pick the best routable variant for THIS
# machine from `variants={name: (binding, Resources)}` declarations. Pure
# logic over (Resources, capabilities, free VRAM); consumer CLIs (cozy) call
# it through `run --list` / `--variant auto` instead of reimplementing
# hardware policy in Go.
# ---------------------------------------------------------------------------

FIT_FITS = "fits"
FIT_EMERGENCY = "emergency_quant"
FIT_OFFLOAD = "offload"
FIT_INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class VariantChoice:
    name: str
    fit: str  # fits | emergency_quant | offload
    reason: str = ""


def variant_fit(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
) -> tuple[str, str]:
    """Fit verdict for ONE function/variant's ``Resources`` on this machine.

    - ``incompatible``: hard gates unmet (no CUDA GPU, compute_capability
      above this GPU's SM, required quant libraries not installed).
    - ``emergency_quant``: does not fit, but the env-gated emergency nf4 rung
      (th#546 emergency lane; loading layer) is enabled and the 4-bit
      estimate fits — runs at below-platform quality, loudly.
    - ``offload``: runnable, but declared vram_gb exceeds free VRAM — the
      models/memory.py offload ladder carries it (slower).
    - ``fits``: full-VRAM residency expected.
    """
    needs_gpu = bool(getattr(resources, "gpu", False))
    req_cc = getattr(resources, "compute_capability", None)
    libs = tuple(getattr(resources, "libraries", ()) or ())
    if needs_gpu and caps.gpu_sm <= 0:
        return FIT_INCOMPATIBLE, "no CUDA GPU detected"
    if req_cc is not None and caps.gpu_sm < int(round(float(req_cc) * 10)):
        return FIT_INCOMPATIBLE, (
            f"requires compute capability >= {float(req_cc):g}, "
            f"GPU is SM{caps.gpu_sm}"
        )
    missing = [lib for lib in libs if lib not in (caps.installed_libs or [])]
    if missing:
        return FIT_INCOMPATIBLE, f"missing libraries: {', '.join(missing)}"
    vram = getattr(resources, "vram_gb", None)
    if vram is None or float(vram) <= float(free_vram_gb):
        return FIT_FITS, ""
    from .loading import EMERGENCY_FIT_FACTOR, emergency_quant_enabled

    if emergency_quant_enabled() and \
            float(vram) * EMERGENCY_FIT_FACTOR <= float(free_vram_gb):
        return FIT_EMERGENCY, (
            f"runs (emergency quality): {float(vram):g} GB VRAM via 4-bit "
            f"emergency quantization, {float(free_vram_gb):.1f} GB free"
        )
    return FIT_OFFLOAD, (
        f"declares {float(vram):g} GB VRAM, {float(free_vram_gb):.1f} GB free"
    )


def select_variant(
    variants: List[tuple[str, Any]],
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    base: Optional[tuple[str, Any]] = None,
) -> Optional[VariantChoice]:
    """Pick the best routable variant.

    ``variants`` is ``[(fn_name, Resources), ...]``; ``base`` is the base
    binding's row when the class declares one. Policy:
      1. drop incompatible rows (SM / library gates);
      2. among variants that FIT free VRAM, prefer the largest declared
         vram_gb (bigger = higher-quality precision);
      3. none fit → an emergency_quant row if the env-gated nf4 rung applies
         (runs at below-platform quality, loudly);
      4. else the base binding + the offload ladder;
      5. no base → the smallest-VRAM compatible variant, offloaded;
      6. nothing routable → None.
    """
    def _vram(res: Any) -> float:
        v = getattr(res, "vram_gb", None)
        return float(v) if v is not None else 0.0

    compat: List[tuple[str, Any, str, str]] = []
    for name, res in variants:
        fit, reason = variant_fit(res, caps, free_vram_gb)
        if fit != FIT_INCOMPATIBLE:
            compat.append((name, res, fit, reason))

    fitting = [row for row in compat if row[2] == FIT_FITS]
    if fitting:
        name, _res, fit, reason = max(fitting, key=lambda r: _vram(r[1]))
        return VariantChoice(name=name, fit=fit, reason=reason)

    # Emergency rung (th#546): nothing fits, but 4-bit emergency quantization
    # would — prefer it over the offload ladder (ladder position: after
    # stored flavors, before CPU offload).
    emergency = [row for row in compat if row[2] == FIT_EMERGENCY]
    if emergency:
        name, _res, fit, reason = max(emergency, key=lambda r: _vram(r[1]))
        return VariantChoice(name=name, fit=fit, reason=reason)

    if base is not None:
        base_name, base_res = base
        fit, reason = variant_fit(base_res, caps, free_vram_gb)
        if fit != FIT_INCOMPATIBLE:
            return VariantChoice(name=base_name, fit=fit, reason=reason)

    if compat:
        name, _res, fit, reason = min(compat, key=lambda r: _vram(r[1]))
        return VariantChoice(name=name, fit=fit, reason=reason)
    return None
