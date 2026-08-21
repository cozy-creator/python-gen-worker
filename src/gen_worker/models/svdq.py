"""SVDQuant 4-bit loader mode."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from ..component_vocab import denoiser_components
from .safetensors_header import read_metadata
from typing import Any, Optional

logger = logging.getLogger(__name__)

SVDQ_METHOD = "svdquant"


class SvdqError(RuntimeError):
    """Base class for typed svdq loader-mode failures."""


class SvdqHardwareError(SvdqError):
    """The artifact's precision has no kernels on this GPU."""


class SvdqInt4Unsupported(SvdqError):
    """svdq-int4: no native engine, and no other engine is installed."""


SVDQ_INT4_REFUSAL = (
    "{path} is svdq-int4. The native engine implements the nvfp4 "
    "block-scaled path only; int4 svdq is a different (single-level, "
    "group-64) scale path with no native implementation, so loading it would "
    "require the nunchaku runtime this worker deliberately does not use "
    "(pgw#1298 deleted that engine). Serve the fp4 artifact of this family, "
    "or file int4 native support as a pgw# follow-up."
)


@dataclass(frozen=True)
class SvdqArtifact:
    component: str
    file: Path
    model_class: str
    precision: str
    rank: int


def _read_safetensors_metadata(path: Path) -> dict:

    return read_metadata(
        path,
        why="a checkpoint whose __metadata__ goes unread stops being detected "
            "as svdq and is loaded down the plain bf16 lane",
    )


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
    """Find the nunchaku single-file checkpoint inside a snapshot: the denoiser dir's (or, for a bare artifact, the root's) sole svdq-tagged safetensors."""
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


def check_svdq_servable(art: SvdqArtifact, path: Any = "") -> None:
    """Every typed refusal for ``art``, raised on the DETECTED artifact."""
    if str(art.precision) == "int4":
        raise SvdqInt4Unsupported(
            SVDQ_INT4_REFUSAL.format(path=path or art.file))
    from .svdq_native import svdq_native_reason

    reason = svdq_native_reason()
    if reason is not None:
        raise SvdqHardwareError(
            f"cannot serve svdq-{art.precision} here — {reason}")


def load_svdq_pipeline(cls: Any, path: Path, art: SvdqArtifact) -> Any:
    """Serve an svdq artifact on the native engine."""
    check_svdq_servable(art, path)
    from .svdq_native import load_svdq_native_pipeline

    logger.info("svdq: native engine (%s %s r%d, file %s)",
                art.precision, art.component, art.rank, art.file.name)
    return load_svdq_native_pipeline(cls, path, art)


__all__ = [
    "SVDQ_INT4_REFUSAL",
    "SvdqArtifact",
    "SvdqError",
    "SvdqHardwareError",
    "SvdqInt4Unsupported",
    "check_svdq_servable",
    "detect_svdq_artifact",
    "load_svdq_pipeline",
]
