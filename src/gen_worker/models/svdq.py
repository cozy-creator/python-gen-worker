"""SVDQuant/nunchaku 4-bit loader mode (gw#415).

A ``#svdq-fp4-*`` / ``#svdq-int4-*`` flavor is a normal diffusers tree whose
denoiser directory holds ONE nunchaku-format single-file checkpoint instead of
plain safetensors shards. The file self-describes in its safetensors
``__metadata__``: ``model_class`` names the nunchaku transformer class and
``quantization_config`` carries ``{"method": "svdquant", "weight": {"dtype":
"fp4_e2m1_all" | "int4"}, "rank": N}``. Loading swaps the nunchaku module into
the standard pipeline — same pipeline class, same handler, no new endpoint.

Hardware: fp4 kernels exist ONLY on consumer Blackwell (sm_120/121); int4 on
sm_75–89. No sm_90/100 (datacenter stays the TRT lane). Version coupling is
HARD (gw#405, live-hit): nunchaku wheels are per-(torch minor, CUDA) and each
nunchaku release calls diffusers transformer forwards positionally against one
diffusers signature window — 1.2.x/1.3.x require diffusers 0.36/0.37 and crash
on 0.38+. The pin matrix below is enforced with typed errors both at variant
selection (fit gating) and at load."""

from __future__ import annotations

import json
import logging
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from ..component_vocab import denoiser_components
from typing import Any, Optional
import importlib.metadata as md

logger = logging.getLogger(__name__)

SVDQ_METHOD = "svdquant"
# Nunchaku ships fp4 kernels for consumer Blackwell only and int4 kernels for
# Turing→Ada. sm_90 (Hopper) and sm_100 (B200) are deliberately absent.
SVDQ_FP4_SMS = (120, 121)
SVDQ_INT4_SMS = (75, 80, 86, 89)

_MAX_HEADER_BYTES = 100 << 20


class SvdqError(RuntimeError):
    """Base class for typed svdq loader-mode failures."""


class SvdqStackError(SvdqError):
    """The (nunchaku, diffusers, torch/cuda) stack violates the pin matrix."""


class SvdqHardwareError(SvdqError):
    """The artifact's precision has no kernels on this GPU."""


class SvdqSnapshotError(SvdqError):
    """The flavor snapshot is not a loadable svdq tree."""


# ---------------------------------------------------------------------------
# Pin matrix — (nunchaku minor) -> required diffusers window.
#
# nunchaku calls ``super().forward(...)`` POSITIONALLY against the diffusers
# transformer signatures it was released against; a newer diffusers inserts
# arguments and the call crashes mid-denoise (gw#405: 1.2.1 on 0.38.0.dev0 and
# 0.39.0 -> "TypeError: argument of type 'int' is not iterable"). Torch/CUDA
# coupling is carried in the wheel's local version tag (e.g.
# ``1.2.1+cu13.0torch2.11``) and checked against the running interpreter.
# Re-verify and extend this table on every nunchaku release.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SvdqPin:
    nunchaku_minor: tuple[int, int]
    diffusers_min: tuple[int, int]
    diffusers_max_exclusive: tuple[int, int]


_PIN_MATRIX: tuple[SvdqPin, ...] = (
    # 1.2.x: CI-pinned to diffusers 0.36 (verified live in gw#405).
    SvdqPin((1, 2), (0, 36), (0, 37)),
    # 1.3.x <-> diffusers 0.39: UNVERIFIED — static signature check only, live
    # A/B still owed (th#1211 G4/G4a). Admitted add-then-verify per gw#405's own
    # precedent; inert until a nunchaku 1.3 wheel is actually installed, which
    # today is only th#1211's benchmark-only serve variant.
    # Static basis: nunchaku's forward and diffusers 0.39's HAVE drifted
    # positionally (nunchaku keeps txt_seq_lens, 0.39 dropped it and added
    # additional_t_cond, so position 6+ misbinds on a positional call), BUT
    # diffusers 0.39's QwenImagePipeline calls the transformer with keyword
    # arguments only and never passes either drifted param — so the gw#405
    # positional hazard is void and no unexpected-kwarg error can fire on the
    # t2i path. Numerical equivalence is NOT established by that check.
    # t2i only: QwenImageEditPlusPipeline was not checked.
    SvdqPin((1, 3), (0, 39), (0, 40)),
)


def _version_tuple(raw: str, n: int = 2) -> tuple[int, ...]:
    """Leading numeric components of a version string ("0.38.0.dev0" ->
    (0, 38)). Missing/unparseable parts are 0."""
    parts: list[int] = []
    for tok in re.split(r"[.+]", str(raw or "").strip()):
        m = re.match(r"^(\d+)", tok)
        if m is None:
            break
        parts.append(int(m.group(1)))
        if len(parts) >= n:
            break
    while len(parts) < n:
        parts.append(0)
    return tuple(parts)


def _wheel_local_tag(nunchaku_version: str) -> tuple[str, str]:
    """(cuda, torch) from a nunchaku wheel version's local tag, e.g.
    ``1.2.1+cu13.0torch2.11`` -> ("13.0", "2.11"). Empty strings when the
    tag is absent (source builds)."""
    if "+" not in str(nunchaku_version or ""):
        return "", ""
    local = str(nunchaku_version).split("+", 1)[1]
    m = re.match(r"^cu([\d.]+)torch([\d.]+)$", local)
    if m is None:
        return "", ""
    return m.group(1), m.group(2)


def check_svdq_stack_versions(
    *, nunchaku_version: str, diffusers_version: str,
    torch_version: str = "", cuda_version: str = "",
) -> None:
    """Pure pin-matrix check over version strings. Raises SvdqStackError with
    the precise coupling that failed; returns None when the stack is legal."""
    nv = _version_tuple(nunchaku_version)
    pin = next((p for p in _PIN_MATRIX if p.nunchaku_minor == nv), None)
    if pin is None:
        known = ", ".join(f"{a}.{b}" for a, b in (p.nunchaku_minor for p in _PIN_MATRIX))
        raise SvdqStackError(
            f"nunchaku {nunchaku_version} is not in the svdq pin matrix "
            f"(known: {known}); its diffusers signature window must be "
            f"verified and added before the svdq rung can serve on it"
        )
    dv = _version_tuple(diffusers_version)
    if not (pin.diffusers_min <= dv < pin.diffusers_max_exclusive):
        lo = ".".join(map(str, pin.diffusers_min))
        hi = ".".join(map(str, pin.diffusers_max_exclusive))
        raise SvdqStackError(
            f"nunchaku {nunchaku_version} requires diffusers>={lo},<{hi} "
            f"(positional transformer forward coupling, gw#405); installed "
            f"diffusers is {diffusers_version}"
        )
    wheel_cuda, wheel_torch = _wheel_local_tag(nunchaku_version)
    if wheel_torch and torch_version and _version_tuple(torch_version) != _version_tuple(wheel_torch):
        raise SvdqStackError(
            f"nunchaku wheel {nunchaku_version} was built for torch "
            f"{wheel_torch}.x; installed torch is {torch_version}"
        )
    if wheel_cuda and cuda_version and _version_tuple(wheel_cuda, 1) != _version_tuple(cuda_version, 1):
        raise SvdqStackError(
            f"nunchaku wheel {nunchaku_version} was built for CUDA "
            f"{wheel_cuda}; torch reports CUDA {cuda_version}"
        )


def svdq_stack_reason() -> Optional[str]:
    """Non-raising stack check against the INSTALLED environment (importlib
    metadata only — does not import nunchaku's CUDA extension). Returns the
    failure reason, or None when the svdq lane is servable here."""

    try:
        nunchaku_version = md.version("nunchaku")
    except md.PackageNotFoundError:
        return "nunchaku is not installed"
    try:
        diffusers_version = md.version("diffusers")
    except md.PackageNotFoundError:
        return "diffusers is not installed"
    torch_version = cuda_version = ""
    try:
        import torch

        torch_version = str(torch.__version__ or "")
        cuda_version = str(getattr(torch.version, "cuda", "") or "")
    except ImportError:
        pass
    try:
        check_svdq_stack_versions(
            nunchaku_version=nunchaku_version,
            diffusers_version=diffusers_version,
            torch_version=torch_version,
            cuda_version=cuda_version,
        )
    except SvdqStackError as exc:
        return str(exc)
    return None


def svdq_precision_for_sm(gpu_sm: int) -> str:
    """"fp4" / "int4" / "" — which svdq kernel family NUNCHAKU can run here."""
    if gpu_sm in SVDQ_FP4_SMS:
        return "fp4"
    if gpu_sm in SVDQ_INT4_SMS:
        return "int4"
    return ""


# ---------------------------------------------------------------------------
# Engine selection (pgw#685) — nunchaku is no longer the only way to serve an
# svdq artifact.
# ---------------------------------------------------------------------------

SVDQ_ENGINES = ("native", "nunchaku")
_ENGINE_ENV = "GEN_WORKER_SVDQ_ENGINE"


def svdq_engine_override() -> str:
    """Operational kill-switch: pin the engine for this process. Empty (the
    default) means "choose per artifact + host"."""
    import os

    raw = str(os.environ.get(_ENGINE_ENV, "") or "").strip().lower()
    if raw and raw not in SVDQ_ENGINES:
        raise SvdqError(
            f"{_ENGINE_ENV}={raw!r} is not a known svdq engine "
            f"({', '.join(SVDQ_ENGINES)})")
    return raw


def svdq_engine_candidates(precision: str) -> tuple[str, ...]:
    """Engines that could serve this precision, best first.

    NATIVE is preferred for fp4: no nunchaku wheel, no diffusers signature
    window, no pin-matrix row, no torch downgrade (gw#405 / th#1211's whole
    blocker class), it compiles, and it covers sm_100/103 which nunchaku never
    will. int4 has no native implementation — the native module is nvfp4
    block-scaled ``_scaled_mm``, and int4 svdq is a different (single-level,
    group-64) scale path."""
    if str(precision) == "int4":
        return ("nunchaku",)
    return ("native", "nunchaku")


def select_svdq_engine(precision: str, *, override: str = "",
                       ) -> tuple[str, dict[str, str]]:
    """Pick the engine for an svdq artifact on THIS host.

    Returns ``(engine, reasons)`` — ``engine`` is "" when none can serve, and
    ``reasons`` maps each rejected engine to why, so the caller can raise one
    error naming every closed door. An explicit ``override`` is honored
    strictly: if that engine cannot serve, nothing else is substituted."""
    from .svdq_native import svdq_native_reason

    chosen = str(override or "").strip().lower() or svdq_engine_override()
    if chosen and chosen not in SVDQ_ENGINES:
        raise SvdqError(f"unknown svdq engine {chosen!r} "
                        f"({', '.join(SVDQ_ENGINES)})")

    candidates = ((chosen,) if chosen
                  else svdq_engine_candidates(precision))
    reasons: dict[str, str] = {}
    for engine in candidates:
        if engine == "native":
            if str(precision) == "int4":
                reasons[engine] = ("the native engine implements nvfp4 only; "
                                   "int4 svdq is a different scale path")
                continue
            reason = svdq_native_reason()
        else:
            reason = svdq_stack_reason()
        if reason is None:
            return engine, reasons
        reasons[engine] = reason
    return "", reasons


# ---------------------------------------------------------------------------
# Artifact detection — safetensors __metadata__ sniff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SvdqArtifact:
    component: str      # denoiser dir name inside the tree ("transformer"/"unet")
    file: Path          # the nunchaku single-file checkpoint
    model_class: str    # e.g. "NunchakuZImageTransformer2DModel"
    precision: str      # "fp4" | "int4"
    rank: int


def _read_safetensors_metadata(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {}
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or n > _MAX_HEADER_BYTES:
                return {}
            header = json.loads(f.read(n))
    except (OSError, ValueError):
        return {}
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    return meta if isinstance(meta, dict) else {}


def _svdq_from_file(component: str, path: Path) -> Optional[SvdqArtifact]:
    meta = _read_safetensors_metadata(path)
    model_class = str(meta.get("model_class") or "")
    qc_raw = meta.get("quantization_config")
    try:
        qc = json.loads(qc_raw) if isinstance(qc_raw, str) else (qc_raw or {})
    except ValueError:
        qc = {}
    if not isinstance(qc, dict) or str(qc.get("method") or "") != SVDQ_METHOD:
        return None
    if not model_class.startswith("Nunchaku"):
        return None
    weight_dtype = str((qc.get("weight") or {}).get("dtype") or "").lower()
    if "fp4" in weight_dtype:
        precision = "fp4"
    elif "int4" in weight_dtype:
        precision = "int4"
    else:
        logger.warning("svdq artifact %s has unknown weight dtype %r", path, weight_dtype)
        return None
    return SvdqArtifact(
        component=component,
        file=path,
        model_class=model_class,
        precision=precision,
        rank=int(qc.get("rank") or 0),
    )


def detect_svdq_artifact(model_path: Path) -> Optional[SvdqArtifact]:
    """Find the nunchaku single-file checkpoint inside a snapshot: the
    denoiser dir's (or, for a bare artifact, the root's) sole svdq-tagged
    safetensors. Cheap — header reads only."""
    root = Path(model_path)
    if root.is_file():
        return _svdq_from_file("", root) if root.suffix == ".safetensors" else None
    if not root.is_dir():
        return None
    for comp in denoiser_components():
        comp_dir = root / comp
        if not comp_dir.is_dir():
            continue
        for f in sorted(comp_dir.glob("*.safetensors")):
            art = _svdq_from_file(comp, f)
            if art is not None:
                return art
    for f in sorted(root.glob("*.safetensors")):
        art = _svdq_from_file("", f)
        if art is not None:
            return art
    return None


# ---------------------------------------------------------------------------
# Loading — nunchaku transformer swap into the standard pipeline
# ---------------------------------------------------------------------------

def check_svdq_loadable(art: SvdqArtifact) -> None:
    """All typed gates for actually serving ``art`` on this machine."""
    reason = svdq_stack_reason()
    if reason is not None:
        raise SvdqStackError(reason)
    import torch

    if not torch.cuda.is_available():
        raise SvdqHardwareError("svdq artifacts require a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    expected = svdq_precision_for_sm(sm)
    if expected != art.precision:
        raise SvdqHardwareError(
            f"svdq-{art.precision} has no kernels on SM{sm} "
            f"(fp4: sm_120/121, int4: sm_75-89"
            + (f"; this GPU runs svdq-{expected}" if expected else "")
            + ")"
        )


def load_svdq_pipeline(cls: Any, path: Path, art: SvdqArtifact, *,
                       engine: str = "") -> Any:
    """Serve an svdq artifact through whichever ENGINE can run it here.

    Engine choice is :func:`select_svdq_engine` (native preferred for fp4);
    ``engine`` pins it explicitly. Callers need no engine awareness — this is
    the single entry point the loading layer already uses."""
    chosen, reasons = select_svdq_engine(art.precision, override=engine)
    if not chosen:
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
        raise SvdqHardwareError(
            f"no svdq engine can serve svdq-{art.precision} here — {detail}")
    if chosen == "native":
        from .svdq_native import load_svdq_native_pipeline

        logger.info("svdq engine: native (%s %s r%d, file %s)",
                    art.precision, art.component, art.rank, art.file.name)
        return load_svdq_native_pipeline(cls, path, art)
    return load_svdq_nunchaku_pipeline(cls, path, art)


def load_svdq_nunchaku_pipeline(cls: Any, path: Path, art: SvdqArtifact) -> Any:
    """Build the pipeline with the nunchaku transformer swapped in.

    Compute dtype is pinned to bf16 — nunchaku's kernels and the surrounding
    scales are bf16-oriented (every upstream example and the gw#405 probe used
    bf16); honoring an fp16 binding here would change numerics silently."""
    check_svdq_loadable(art)
    import nunchaku
    import torch

    if not art.component:
        raise SvdqSnapshotError(
            f"svdq snapshot {path} is a bare single-file transformer; a "
            f"servable #svdq flavor must be a full diffusers tree with the "
            f"nunchaku file under its denoiser directory"
        )
    ncls = getattr(nunchaku, art.model_class, None)
    if ncls is None:
        raise SvdqStackError(
            f"installed nunchaku has no {art.model_class} (artifact needs a "
            f"newer nunchaku release / unsupported family)"
        )
    dtype = torch.bfloat16
    logger.info(
        "svdq loader mode: %s %s r%d via %s (file %s)",
        art.precision, art.component, art.rank, art.model_class, art.file.name,
    )
    denoiser = ncls.from_pretrained(str(art.file), torch_dtype=dtype)
    # low_cpu_mem_usage=False is nunchaku's documented requirement for the
    # component-swap pipeline build (meta-device init breaks its buffers).
    return cls.from_pretrained(
        str(path), torch_dtype=dtype, low_cpu_mem_usage=False,
        **{art.component: denoiser},
    )


__all__ = [
    "SVDQ_ENGINES",
    "SVDQ_FP4_SMS",
    "SVDQ_INT4_SMS",
    "SvdqArtifact",
    "SvdqError",
    "SvdqHardwareError",
    "SvdqSnapshotError",
    "SvdqStackError",
    "check_svdq_loadable",
    "check_svdq_stack_versions",
    "detect_svdq_artifact",
    "load_svdq_nunchaku_pipeline",
    "load_svdq_pipeline",
    "select_svdq_engine",
    "svdq_engine_candidates",
    "svdq_engine_override",
    "svdq_precision_for_sm",
    "svdq_stack_reason",
]
