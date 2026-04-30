"""Model-family / variant detection from a HuggingFace repo's file listing.

File-selection logic moved to :mod:`gen_worker.conversion.hf_classifier` (see
e2e progress.json #67). This module now only contains downstream metadata
inference: given a repo_dir + file list, what model family / variant is this?
The output feeds the destination checkpoint's tags so inference workers can
pick the right pipeline class.

The legacy ``select_huggingface_source_files`` / ``HFSourceFileSelection``
that lived here were replaced by :mod:`hf_classifier`'s classify_huggingface_repo
+ per-strategy selectors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_DIFFUSERS_COMPONENT_DIRS = {
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
    "transformer",
}
_SD15_SD2_HINTS = (
    "stable-diffusion-v1",
    "stable-diffusion-v2",
    "sd-v1",
    "sd-v2",
    "v1-inference.yaml",
    "v2-inference.yaml",
)
_SD15_SD2_PATTERN = re.compile(r"(?:^|[^a-z0-9])(?:sd|stable[-_]?diffusion)[-_ ]?v?[12](?:[^0-9]|$)")
_SD15_SD2_CHECKPOINT_PATTERN = re.compile(r"(?:^|[^a-z0-9])v(?:1|2)[-_](?:[0-9])")
_SD15_VARIANT_HINTS = (
    "stable-diffusion-v1-5",
    "stable-diffusion-1-5",
    "sd-v1-5",
    "sd15",
)
_SD15_VARIANT_PATTERN = re.compile(r"(?:^|[^a-z0-9])(?:sd|stable[-_ ]?diffusion)[-_ ]?(?:v)?1(?:[-_. ]?5)?(?:[^0-9]|$)")


def _normalize_letters_digits(raw: str) -> str:
    chars: list[str] = []
    for ch in str(raw or "").strip().lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            chars.append(ch)
    return "".join(chars)


def canonical_model_family_from_variant(variant: str) -> str:
    raw = str(variant or "").strip().lower()
    if raw in {"flux1", "flux2", "flex2", "z_image", "qwen_image"}:
        return "flux"
    if raw in {"wan21", "wan22", "wan"}:
        return "wan"
    if raw in {"auraflow"}:
        return "auraflow"
    if raw == "sdxl":
        return "sdxl"
    if raw == "sd15":
        return "sd15_sd2"
    return "unknown"


def infer_model_family_variant_from_hint(value: str | None) -> str:
    hint = str(value or "").strip().lower()
    normalized = _normalize_letters_digits(hint)
    if hint == "":
        return "unknown"
    if "auraflow" in normalized:
        return "auraflow"
    if "flex2" in normalized or ("ostris" in normalized and "flex" in normalized):
        return "flex2"
    if "wan22" in normalized:
        return "wan22"
    if "wan21" in normalized:
        return "wan21"
    if "wan" in normalized and any(tok in normalized for tok in ("video", "i2v", "t2v", "vace")):
        return "wan22"
    if "qwenimage" in normalized:
        return "qwen_image"
    if "zimage" in normalized:
        return "z_image"
    if "flux2" in normalized or ("flux" in normalized and "klein" in normalized):
        return "flux2"
    if "flux1" in normalized:
        return "flux1"
    if "flux" in normalized:
        return "flux1"
    if "sdxl" in normalized or "stablediffusionxl" in normalized:
        return "sdxl"
    if any(token in hint for token in _SD15_VARIANT_HINTS):
        return "sd15"
    if _SD15_VARIANT_PATTERN.search(hint) is not None:
        return "sd15"
    return "unknown"


def infer_model_family_from_hint(value: str | None) -> str:
    hint = str(value or "").strip().lower()
    variant = infer_model_family_variant_from_hint(value)
    if variant != "unknown":
        return canonical_model_family_from_variant(variant)
    if any(token in hint for token in _SD15_SD2_HINTS):
        return "sd15_sd2"
    if _SD15_SD2_PATTERN.search(hint) is not None:
        return "sd15_sd2"
    if _SD15_SD2_CHECKPOINT_PATTERN.search(hint) is not None and (
        "pruned" in hint or "emaonly" in hint or "ckpt" in hint or "safetensors" in hint
    ):
        return "sd15_sd2"
    return "unknown"


@dataclass(frozen=True)
class SourceLayoutInfo:
    """Lightweight metadata about a detected HF repo (for tagging only).

    File selection is the classifier's job — see hf_classifier. This struct is
    populated for downstream taggers that need a model-family hint.
    """

    source_layout: str
    model_family: str
    model_family_variant: str
    detection_reason: str


def _normalize_paths(files: list[str]) -> list[str]:
    out: list[str] = []
    for raw in files:
        clean = str(raw or "").strip().replace("\\", "/").lstrip("/")
        if clean == "" or ".." in clean.split("/"):
            continue
        out.append(clean)
    return out


def _has_diffusers_layout_signals(paths: list[str]) -> bool:
    if "model_index.json" in paths:
        return True
    top_dirs = {p.split("/", 1)[0] for p in paths if "/" in p}
    return bool(_DIFFUSERS_COMPONENT_DIRS.intersection(top_dirs))


def _detect_family_variant_from_model_index(repo_dir: Path) -> str:
    model_index_path = repo_dir / "model_index.json"
    if not model_index_path.exists():
        return "unknown"
    try:
        payload = json.loads(model_index_path.read_text("utf-8"))
    except Exception:
        return "unknown"
    name_or_path = str(payload.get("_name_or_path") or "").strip()
    detected_from_name = infer_model_family_variant_from_hint(name_or_path)
    if detected_from_name != "unknown":
        return detected_from_name
    cls = str(payload.get("_class_name") or "").strip().lower()
    if "auraflow" in cls:
        return "auraflow"
    if cls.startswith("wan") or "wanpipeline" in cls:
        return "wan22"
    if "qwenimage" in cls:
        return "qwen_image"
    if "zimage" in cls:
        return "z_image"
    if "flux2" in cls:
        return "flux2"
    if "flux" in cls:
        return "flux1"
    if "stablediffusionxl" in cls:
        return "sdxl"
    if "stablediffusion" in cls:
        return "sd15"
    return "unknown"


def _detect_family_variant_from_components(paths: list[str]) -> str:
    top_dirs = {p.split("/", 1)[0] for p in paths if "/" in p}
    if "transformer_2" in top_dirs and "text_encoder" in top_dirs and "vae" in top_dirs and "unet" not in top_dirs:
        return "wan22"
    if (
        "image_encoder" in top_dirs
        and "transformer" in top_dirs
        and "text_encoder" in top_dirs
        and "unet" not in top_dirs
        and "text_encoder_2" not in top_dirs
    ):
        return "wan22"
    if "transformer" in top_dirs:
        # FLUX.1 generally carries a second text encoder; FLUX.2-like layouts usually do not.
        if "text_encoder_2" in top_dirs:
            return "flux1"
        return "flux2"
    if "text_encoder_2" in top_dirs:
        return "sdxl"
    if "unet" in top_dirs and "vae" in top_dirs and "text_encoder" in top_dirs:
        return "sd15"
    return "unknown"


def _detect_family_variant_from_singlefile_paths(paths: list[str]) -> str:
    for path in paths:
        detected = infer_model_family_variant_from_hint(path)
        if detected != "unknown":
            return detected
    return "unknown"


def detect_huggingface_source_layout(*, repo_dir: Path, files: list[str]) -> SourceLayoutInfo:
    """Tagging-only metadata: detect diffusers-vs-singlefile shape + family variant.

    Used by ingest_from_source to populate ``model_family`` / ``model_family_variant``
    in the destination checkpoint metadata. Not a load-bearing decision —
    the file-selection strategy is determined upstream by the classifier.
    """
    normalized = _normalize_paths(files)
    if _has_diffusers_layout_signals(normalized):
        source_layout = "diffusers"
        reason = "diffusers_layout_signals_present"
    elif any(p.lower().endswith((".safetensors", ".gguf")) for p in normalized):
        source_layout = "singlefile"
        reason = "single_file_weight_signals_present"
    else:
        source_layout = "unknown"
        reason = "layout_signals_missing"

    model_family_variant = _detect_family_variant_from_model_index(repo_dir)
    if model_family_variant == "unknown":
        model_family_variant = _detect_family_variant_from_components(normalized)
    if model_family_variant == "unknown" and source_layout == "singlefile":
        model_family_variant = _detect_family_variant_from_singlefile_paths(normalized)
    model_family = canonical_model_family_from_variant(model_family_variant)
    if model_family == "unknown":
        model_family = infer_model_family_from_hint(" ".join(normalized[:64]))

    return SourceLayoutInfo(
        source_layout=source_layout,
        model_family=model_family,
        model_family_variant=model_family_variant,
        detection_reason=reason,
    )


__all__ = [
    "SourceLayoutInfo",
    "canonical_model_family_from_variant",
    "detect_huggingface_source_layout",
    "infer_model_family_variant_from_hint",
    "infer_model_family_from_hint",
]
