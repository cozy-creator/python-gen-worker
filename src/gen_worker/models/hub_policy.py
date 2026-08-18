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
    """
    Detect worker capabilities for Cozy Hub artifact selection.

    This is intentionally conservative: if torch/cuda isn't available, we report
    empty/zero values so Cozy Hub can avoid selecting capability-gated artifacts
    (e.g. fp8) unless explicitly supported.
    """
    installed: List[str] = []

    # Known optional libs that affect artifact compatibility.
    # Keep this hardcoded (no env config), per Cozy design.
    #
    # pgw#1300: "nunchaku" is GONE from this probe. pgw#1298 kept it as an
    # ADMISSION TOKEN — the hub refused any svdq row whose stamped
    # `engines=["nunchaku"]` was not in `installed_libs`. th#2055 (`65f0882f2`)
    # deleted that gate outright (`precision/ladder.go` `admitted()` and
    # `AdmitLiteral`'s engine arm are gone), and no hub reader consults this
    # list for `nunchaku` any more; the ones that remain ask for `torchao` /
    # `modelopt` backends. So the token claims a capability we deleted in
    # pgw#1298 and gates nothing — reporting it is a lie with no consumer.
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


# ---------------------------------------------------------------------------
# Fit-verdict policy — classify ONE function's Resources on THIS machine. Pure
# logic over (Resources, capabilities, free VRAM); consumed by
# serve_fit.plan_serve (the executor's flavor-fit ladder) and `run --list`.
# Ranking lives hub-side, never here.
# ---------------------------------------------------------------------------

FIT_FITS = "fits"
FIT_INCOMPATIBLE = "incompatible"

# th#1867 (§1.35): this module returns NO size verdicts. Every one of them was
# derived from an author DECLARATION compared against free VRAM — a prediction
# made before anything is measured, which §4.33 forbids from acting as a floor
# and §1.35 forbids from acting at all. The rungs themselves did not go
# anywhere — they are chosen at LOAD time by `models/memory.select_auto_mode`
# from the
# pipeline's real size against the card's real free VRAM, and reported as they
# happen (`serve_fit.replan` -> FnDegraded). What is deleted is the guess, not
# the ladder.

# The STORED-PRECISION classification of a binding is NOT this planner's. A
# binding names a tag or a digest; WHAT THE BYTES ARE is the checkpoint's
# TENSOR-LAYOUT CONTRACT (§1.33), which the loaders gate on for real
# (`@implements_contract`, models/svdq_layout.py, w4a4.py, w8a8.py) instead
# of trusting a token in a ref. The svdq SM-window and nvfp4-Blackwell
# refusals therefore live where the artifact declares itself, not here. This
# planner answers on resources + declared cast alone, which is what it can
# honestly know.


def variant_fit(
    resources: Any,
    caps: TensorhubWorkerCapabilities,
    free_vram_gb: float,
    *,
    binding: Any = None,
) -> tuple[str, str]:
    """Fit verdict for ONE function/variant's ``Resources`` on this machine.

    Exactly TWO verdicts remain after th#1867, and both name OUR code rather
    than the card (§1.35 amendment 2):

    - ``incompatible``: no CUDA device at all, or a quant library this build
      does not carry.
    - ``fits``: everything else. Not a promise of full residency — a promise
      that nothing here has an opinion about the card's size. How the pipeline
      actually lands is measured at load time by ``models/memory``.

    ``free_vram_gb`` is retained in the signature and deliberately unused: it
    is the number the deleted comparison was made against, and re-admitting it
    is the single edit that would rebuild the estimate-as-floor this ruling
    removed. Nothing may reintroduce a size comparison here.
    """
    needs_gpu = bool(getattr(resources, "gpu", False))
    libs = tuple(getattr(resources, "libraries", ()) or ())
    if needs_gpu and caps.gpu_sm <= 0:
        return FIT_INCOMPATIBLE, "no CUDA GPU detected"
    # SDK v2: no declared compute-capability gate — the fit
    # ladder picks precision per card.
    missing = [lib for lib in libs if lib not in (caps.installed_libs or [])]
    if missing:
        return FIT_INCOMPATIBLE, f"missing libraries: {', '.join(missing)}"
    return FIT_FITS, ""

