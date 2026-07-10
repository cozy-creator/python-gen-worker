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
    known = ["bitsandbytes", "torchao", "transformer_engine", "tensorrt",
             "nunchaku", "deepcompressor"]
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
FIT_FP8 = "fp8"
FIT_NVFP4 = "nvfp4"
FIT_SVDQ_FP4 = "svdq_fp4"
FIT_SVDQ_INT4 = "svdq_int4"
FIT_EMERGENCY_FP8 = "emergency_fp8"
FIT_EMERGENCY = "emergency_quant"
FIT_OFFLOAD = "offload"
FIT_INCOMPATIBLE = "incompatible"

# Stored-flavor hardware windows (th#683 fit ladder), same SM-gating pattern
# as the svdq rows: a flavor only serves on silicon whose kernels/format
# support it. fp8 flavors are an Ada-and-newer rung (sm_89 Ada, sm_90 Hopper,
# sm_100+ Blackwell); nvfp4 tensor-core kernels exist on Blackwell only.
FP8_FLAVOR_MIN_SM = 89
NVFP4_FLAVOR_MIN_SM = 100


def svdq_flavor_kind(binding: Any) -> str:
    """"fp4" / "int4" / "" — is this binding an svdq stored-flavor row?
    The flavor token is the selector (th#597): ``svdq-fp4-*`` / ``svdq-int4-*``."""
    flavor = str(getattr(binding, "flavor", "") or "").strip().lower()
    if flavor.startswith("svdq-fp4"):
        return "fp4"
    if flavor.startswith("svdq-int4"):
        return "int4"
    return ""


def quant_flavor_kind(binding: Any) -> str:
    """"fp8" / "nvfp4" / "" — a stored quantized-flavor row this planner
    HW-gates (svdq rows are classified separately by svdq_flavor_kind)."""
    flavor = str(getattr(binding, "flavor", "") or "").strip().lower()
    if flavor.startswith("svdq"):
        return ""
    if flavor.startswith("fp8"):
        return "fp8"
    if flavor.startswith("nvfp4"):
        return "nvfp4"
    return ""


@dataclass(frozen=True)
class VariantChoice:
    name: str
    fit: str  # one of the FIT_* verdicts (never incompatible)
    reason: str = ""


def variant_fit(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
) -> tuple[str, str]:
    """Fit verdict for ONE function/variant's ``Resources`` on this machine.

    - ``incompatible``: hard gates unmet (no CUDA GPU, compute_capability
      above this GPU's SM, required quant libraries not installed, an svdq
      row whose SM window / nunchaku-diffusers pin matrix fails, or an
      fp8/nvfp4 stored-flavor row outside its SM window).
    - ``fp8`` / ``nvfp4``: the binding is a stored quantized flavor whose
      hardware window passes (fp8: Ada/Hopper+, nvfp4: Blackwell) and it
      fits free VRAM — native runs on their silicon, ranked below a fitting
      full-precision row (bf16 > fp8 > nvfp4 quality order).
    - ``svdq_fp4`` / ``svdq_int4``: the binding is a stored SVDQuant flavor
      (gw#415) and every gate passes — on Blackwell, svdq_fp4 OUTRANKS every
      other fitting row (faster AND smaller, QUANTIZATION-POLICY fit ladder);
      svdq_int4 is a fit rung ahead of emergency-nf4 only.
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
    svdq_kind = svdq_flavor_kind(binding)
    if svdq_kind:
        from .svdq import (
            SVDQ_FP4_SMS,
            SVDQ_INT4_SMS,
            svdq_precision_for_sm,
            svdq_stack_reason,
        )

        window = SVDQ_FP4_SMS if svdq_kind == "fp4" else SVDQ_INT4_SMS
        if caps.gpu_sm not in window:
            runs = svdq_precision_for_sm(caps.gpu_sm)
            return FIT_INCOMPATIBLE, (
                f"svdq-{svdq_kind} kernels need SM in "
                f"{'/'.join(str(x) for x in window)}; GPU is SM{caps.gpu_sm}"
                + (f" (runs svdq-{runs})" if runs and runs != svdq_kind else "")
            )
        reason = svdq_stack_reason()
        if reason is not None:
            return FIT_INCOMPATIBLE, reason
    quant_kind = quant_flavor_kind(binding)
    if quant_kind == "fp8" and caps.gpu_sm < FP8_FLAVOR_MIN_SM:
        return FIT_INCOMPATIBLE, (
            f"fp8 stored flavor needs SM >= {FP8_FLAVOR_MIN_SM} (Ada/Hopper+); "
            f"GPU is SM{caps.gpu_sm}"
        )
    if quant_kind == "nvfp4" and caps.gpu_sm < NVFP4_FLAVOR_MIN_SM:
        return FIT_INCOMPATIBLE, (
            f"nvfp4 stored flavor needs Blackwell (SM >= {NVFP4_FLAVOR_MIN_SM}); "
            f"GPU is SM{caps.gpu_sm}"
        )
    vram = getattr(resources, "vram_gb", None)
    # vram_gb recommends a card SIZE (total VRAM), so an idle card of exactly
    # that size counts as fitting: subtract the fixed driver/framebuffer/CUDA
    # reserve before comparing against measured free VRAM.
    from .memory import effective_vram_requirement_gb

    if vram is None or effective_vram_requirement_gb(vram) <= float(free_vram_gb):
        if svdq_kind == "fp4":
            return FIT_SVDQ_FP4, "svdq-fp4 stored flavor (Blackwell 4-bit rung)"
        if svdq_kind == "int4":
            return FIT_SVDQ_INT4, "svdq-int4 stored flavor (4-bit fit rung)"
        if quant_kind == "fp8":
            return FIT_FP8, "fp8 stored flavor (Ada/Hopper+ rung)"
        if quant_kind == "nvfp4":
            return FIT_NVFP4, "nvfp4 stored flavor (Blackwell rung)"
        return FIT_FITS, ""
    from .loading import EMERGENCY_FIT_FACTOR, FP8_STORAGE_FIT_FACTOR

    # Runtime rungs are automatic on CUDA hosts (gw#420) — pure functions of
    # the declared capabilities, so verdicts don't depend on the probing host.
    # Already-quantized flavors (svdq/fp8/nvfp4) can't be re-quantized: they
    # fall straight to the offload ladder when they don't fit.
    if caps.gpu_sm > 0 and not svdq_kind and not quant_kind:
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


def select_variant(
    variants: List[tuple],
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    base: Optional[tuple] = None,
) -> Optional[VariantChoice]:
    """Pick the best routable variant.

    ``variants`` rows are ``(fn_name, Resources)`` or
    ``(fn_name, Resources, binding)``; ``base`` is the base binding's row when
    the class declares one. Policy (QUANTIZATION-POLICY fit ladder,
    bf16 -> fp8 -> nvfp4 -> 4-bit over the HW-compatible fitting set):
      1. drop incompatible rows (SM / library / svdq pin-matrix / fp8-nvfp4
         SM-window gates);
      2. a fitting svdq_fp4 row wins outright — on Blackwell the SVDQuant
         flavor beats fp8/bf16 on BOTH latency and VRAM (gw#405 measured);
      3. among plain rows that FIT free VRAM, prefer the largest declared
         vram_gb (bigger = higher-quality precision);
      4. a fitting stored fp8 row (Ada/Hopper+), then a fitting stored
         nvfp4 row (Blackwell);
      5. a fitting svdq_int4 row (4-bit fit rung: nothing bigger fits);
      6. an emergency_fp8 row (runtime fp8-E4M3 storage), then an
         emergency_quant row if the nf4 rung applies
         (runs at below-platform quality, loudly);
      7. else the base binding + the offload ladder;
      8. no base → the smallest-VRAM compatible variant, offloaded;
      9. nothing routable → None.
    """
    def _vram(res: Any) -> float:
        v = getattr(res, "vram_gb", None)
        return float(v) if v is not None else 0.0

    compat: List[tuple[str, Any, str, str]] = []
    for row in variants:
        name, res = row[0], row[1]
        row_binding = row[2] if len(row) > 2 else None
        fit, reason = variant_fit(res, caps, free_vram_gb, binding=row_binding)
        if fit != FIT_INCOMPATIBLE:
            compat.append((name, res, fit, reason))

    for rung in (FIT_SVDQ_FP4, FIT_FITS, FIT_FP8, FIT_NVFP4):
        fitting = [row for row in compat if row[2] == rung]
        if fitting:
            name, _res, fit, reason = max(fitting, key=lambda r: _vram(r[1]))
            return VariantChoice(name=name, fit=fit, reason=reason)

    # svdq_int4 (gw#415): a stored 4-bit flavor that fits when nothing
    # full-precision does — ahead of the emergency runtime-quant rungs.
    # Emergency rungs (th#546 / th#683): nothing stored fits, but runtime
    # fp8-E4M3 storage or 4-bit quantization would — prefer them over the
    # offload ladder (ladder position: after stored flavors, before offload).
    for rung in (FIT_SVDQ_INT4, FIT_EMERGENCY_FP8, FIT_EMERGENCY):
        rows = [row for row in compat if row[2] == rung]
        if rows:
            name, _res, fit, reason = max(rows, key=lambda r: _vram(r[1]))
            return VariantChoice(name=name, fit=fit, reason=reason)

    if base is not None:
        base_name, base_res = base[0], base[1]
        base_binding = base[2] if len(base) > 2 else None
        fit, reason = variant_fit(base_res, caps, free_vram_gb, binding=base_binding)
        if fit != FIT_INCOMPATIBLE:
            return VariantChoice(name=base_name, fit=fit, reason=reason)

    if compat:
        name, _res, fit, reason = min(compat, key=lambda r: _vram(r[1]))
        return VariantChoice(name=name, fit=fit, reason=reason)
    return None
