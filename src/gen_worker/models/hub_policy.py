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

# Stored-flavor hardware windows. nvfp4 is a genuine native-compute format:
# its tensor-core kernels exist on Blackwell only, so an nvfp4 stored flavor
# on older silicon is a hard refusal. fp8 is DIFFERENT — fp8-E4M3 storage
# upcasts per-layer to bf16 at compute (loading.apply_fp8_storage: fp8 bytes
# resident, no fp8 silicon required), so a stored #fp8 flavor SERVES on ANY
# CUDA card and is never refused on SM. Whether fp8 is PREFERRED over bf16 is
# the SM-conditional question (ladder.FP8_COMPUTE_MIN_SM): fp8 tensor cores
# on SM>=89 make it faster+smaller, below that it merely halves storage.
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
      nvfp4 stored-flavor row below Blackwell). A stored #fp8 flavor is
      NEVER incompatible on SM — its storage upcasts to bf16 at compute.
    - ``fp8`` / ``nvfp4``: the binding is a stored quantized flavor and it
      fits free VRAM — nvfp4 is Blackwell-native; fp8 serves on ANY card
      (bf16-upcast). Ranking is SM-aware: on fp8-compute silicon (SM>=89)
      fp8 outranks bf16 (faster+smaller); below it fp8 is a fit fallback
      only (bf16 preferred when it fits).
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
    # fp8 has NO SM refusal — a stored #fp8 flavor upcasts to bf16 at compute
    # and serves on any CUDA card (the refuse-bug fix: the hub could place
    # #fp8 on an older card the worker then refused). nvfp4 stays gated: it is
    # a genuine Blackwell-native tensor-core format with no upcast path here.
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
            return FIT_FP8, "fp8 stored flavor (universal; bf16-upcast below SM89)"
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
    the class declares one. Policy — SM-aware (the shared-ladder ordering,
    ``FP8_COMPUTE_MIN_SM``):
      1. drop incompatible rows (SM / library / svdq pin-matrix / nvfp4
         Blackwell gate; fp8 is never dropped on SM);
      2. a fitting svdq_fp4 row wins outright — on Blackwell the SVDQuant
         flavor beats fp8/bf16 on BOTH latency and VRAM (gw#405 measured);
      3. fp8-vs-bf16 over the fitting set is SM-conditional (Paul's ruling
         "run fp8 wherever we can"): on fp8-compute silicon (SM>=89) a
         fitting stored fp8 row outranks bf16 (faster+smaller); below SM89
         bf16-if-it-fits wins and fp8 is only a fit fallback. Within each
         rung prefer the largest declared vram_gb; nvfp4 (Blackwell) trails;
      4. a fitting svdq_int4 row (4-bit fit rung: nothing bigger fits);
      5. an emergency_fp8 row (runtime fp8-E4M3 storage), then an
         emergency_quant row if the nf4 rung applies
         (runs at below-platform quality, loudly);
      6. else the base binding + the offload ladder;
      7. no base → the smallest-VRAM compatible variant, offloaded;
      8. nothing routable → None.
    """
    from .ladder import FP8_COMPUTE_MIN_SM

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

    # SM>=89: fp8 tensor cores → prefer fp8 over bf16. Below: bf16-if-fits first.
    if caps.gpu_sm >= FP8_COMPUTE_MIN_SM:
        native_order = (FIT_SVDQ_FP4, FIT_FP8, FIT_FITS, FIT_NVFP4)
    else:
        native_order = (FIT_SVDQ_FP4, FIT_FITS, FIT_FP8, FIT_NVFP4)
    for rung in native_order:
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
