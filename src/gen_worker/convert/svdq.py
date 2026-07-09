"""SVDQuant flavor trees (gw#415) — build + mirror.

A ``#svdq-fp4-rN`` / ``#svdq-int4-rN`` flavor is the base checkpoint's
diffusers tree with the denoiser's plain weights replaced by ONE
nunchaku-format single-file checkpoint (self-describing safetensors
``__metadata__``; see ``gen_worker.models.svdq``). CAS dedup makes the shared
components (VAE / text encoders / scheduler / tokenizer) free — the flavor
stores only the 4-bit denoiser.

Two producers emit this shape:
  - MIRROR: fetch an official nunchaku artifact (e.g.
    ``nunchaku-ai/nunchaku-z-image-turbo``) and marry it to our mirrored base.
  - PRODUCE: a deepcompressor SVDQuant run whose ``convert.py`` output is the
    same single-file format.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ..models.svdq import SvdqArtifact, detect_svdq_artifact
from .writer import copy_non_weight_files

logger = logging.getLogger(__name__)

# R2 single-PUT cap (clone.py reshards bigger files) — but sharding a nunchaku
# checkpoint strips the __metadata__ its loader needs, so oversize svdq
# artifacts are refused until component-level reassembly lands (qwen-image
# files are 11.5-13.1 GB; z-image/flux fit). Tracked in the qwen rollout issue.
MAX_SVDQ_FILE_BYTES = 5 * 1000**3


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
    """Materialize one svdq flavor tree: the full base tree minus the
    denoiser's weights, plus the nunchaku single-file checkpoint under the
    denoiser directory. Returns ``(tree_root, attrs)``."""
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
            f"svdq file {svdq_file.name} is {size / 1e9:.1f} GB > single-PUT "
            f"cap; sharding would strip nunchaku metadata — component-level "
            f"reassembly is not implemented yet (gw#415 follow-up)"
        )
    if component is None:
        component = next(
            (c for c in ("transformer", "unet") if (base_dir / c).is_dir()), "",
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
    from huggingface_hub import hf_hub_download

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
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
