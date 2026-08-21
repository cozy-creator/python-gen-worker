"""SVDQuant flavor trees — build + mirror."""

from __future__ import annotations

from ..component_vocab import denoiser_components

import logging
import os
from pathlib import Path
from typing import Optional

from ..models.svdq import SvdqArtifact, detect_svdq_artifact
from .writer import copy_non_weight_files
from ..net import hf

logger = logging.getLogger(__name__)

# The hub's checkpoint-commit grant allows 64 GiB/file and uploads go out as presigned multipart parts, so a large single file is fine. Sharding a nunchaku checkpoint WOULD strip the __metadata__ its loader needs — it must publish whole, and publishing whole is allowed.
MAX_SVDQ_FILE_BYTES = 64 * 1024**3


def svdq_flavor_label(art: SvdqArtifact) -> str:
    """Canonical flavor token: precision + rank explicit (``svdq-fp4-r128``)."""
    return f"svdq-{art.precision}-r{art.rank}"


def build_svdq_flavor_tree(
    base_dir: Path,
    svdq_file: Path,
    out_dir: Path,
    *,
    component: Optional[str] = None,
) -> tuple[Path, dict[str, str]]:
    """Materialize one svdq flavor tree: the full base tree minus the denoiser's weights, plus the nunchaku single-file checkpoint under the denoiser directory."""
    base_dir = Path(base_dir)
    svdq_file = Path(svdq_file)
    out_dir = Path(out_dir)

    art = detect_svdq_artifact(svdq_file)
    if art is None:
        raise ValueError(
            f"{svdq_file} is not a nunchaku SVDQuant checkpoint (missing "
            "model_class/quantization_config safetensors metadata)"
        )
    size = svdq_file.stat().st_size
    if size > MAX_SVDQ_FILE_BYTES:
        raise ValueError(
            f"svdq file {svdq_file.name} is {size / 1e9:.1f} GB > "
            f"{MAX_SVDQ_FILE_BYTES / 1024**3:.0f} GiB hub per-file ceiling; it "
            f"must publish whole (sharding strips nunchaku metadata), so there "
            f"is no fallback for a file this large"
        )
    if component is None:
        component = next(
            (c for c in denoiser_components() if (base_dir / c).is_dir()), "",
        )
    if not component:
        raise ValueError(
            f"base tree {base_dir} has no transformer/unet component to swap"
        )
    if not (base_dir / "model_index.json").exists():
        raise ValueError(
            f"base tree {base_dir} is not a diffusers pipeline layout "
            "(model_index.json missing) — svdq flavors swap a pipeline's denoiser"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    copy_non_weight_files(base_dir, out_dir, skip_components={component})
    dest_dir = out_dir / component
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / svdq_file.name
    if dest.exists():
        dest.unlink()
    try:
        os.link(svdq_file, dest)
    except OSError:
        import shutil

        shutil.copy2(svdq_file, dest)

    flavor = svdq_flavor_label(art)
    attrs = {
        "flavor": flavor,
        "quantization_method": "svdquant",
        "quantization_library": "nunchaku",
        "svdq_precision": art.precision,
        "svdq_rank": str(art.rank),
        "svdq_model_class": art.model_class,
        "svdq_component": component,
    }
    logger.info(
        "built svdq flavor tree %s: %s <- %s (%.2f GB)",
        flavor, out_dir, svdq_file.name, size / 1e9,
    )
    return out_dir, attrs


def fetch_svdq_checkpoint(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    *,
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Path:
    """Download ONE nunchaku checkpoint file from an HF repo (mirror lane)."""

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = hf().hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(dest_dir),
        token=hf_token or None,
    )
    return Path(local)


__all__ = [
    "MAX_SVDQ_FILE_BYTES",
    "build_svdq_flavor_tree",
    "fetch_svdq_checkpoint",
    "svdq_flavor_label",
]
