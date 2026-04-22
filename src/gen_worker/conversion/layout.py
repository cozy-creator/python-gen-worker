from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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
_WEIGHT_EXTS = (".safetensors", ".ckpt", ".pt", ".bin")
_AIO_EXCLUDED_NAME_HINTS = (
    "lora",
    "lycoris",
    "adapter",
    "controlnet",
    "ip_adapter",
    "ip-adapter",
    "textual_inversion",
    "textual-inversion",
    "embedding",
    "text_encoder",
    "text-encoder",
    "vae",
)
_AIO_EXCLUDED_DIR_HINTS = {
    "adapters",
    "adapter",
    "lora",
    "loras",
    "embeddings",
    "embedding",
    "textual_inversion",
    "textual-inversion",
    "controlnet",
    "ip_adapter",
    "ip-adapter",
    "text_encoder",
    "text_encoder_2",
    "text-encoder",
    "text-encoder-2",
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
    source_layout: str
    model_family: str
    model_family_variant: str
    detection_reason: str


@dataclass(frozen=True)
class HFSourceFileSelection:
    source_layout_preference: str
    selected_source_layout: str
    selected_paths: list[str]
    selected_aio_path: str
    selection_reason: str
    detection_reason: str
    # Per-component dtype selected from `source_dtype_preference`. Present only
    # when dtype filtering kicked in. Keys are the canonical `base_key` tokens
    # from `dtype_detection.parse_weight_file` (e.g. "unet/diffusion_pytorch_model").
    dtype_by_component: dict[str, str] = field(default_factory=dict)
    dropped_paths: tuple[str, ...] = ()


def _normalize_paths(files: list[str]) -> list[str]:
    out: list[str] = []
    for raw in files:
        clean = str(raw or "").strip().replace("\\", "/").lstrip("/")
        if clean == "" or ".." in clean.split("/"):
            continue
        out.append(clean)
    return out


def _analyze_layout_signals(paths: list[str]) -> tuple[bool, bool, bool]:
    has_model_index = "model_index.json" in paths
    top_dirs = {p.split("/", 1)[0] for p in paths if "/" in p}
    has_diffusers_components = bool(_DIFFUSERS_COMPONENT_DIRS.intersection(top_dirs))
    has_weight_file = any(p.lower().endswith(_WEIGHT_EXTS) for p in paths)
    return has_model_index, has_diffusers_components, has_weight_file


def _detect_source_layout(paths: list[str]) -> tuple[str, str]:
    has_model_index, has_diffusers_components, has_weight_file = _analyze_layout_signals(paths)

    if (has_model_index or has_diffusers_components) and has_weight_file:
        return "diffusers", "both_layout_signals_present_prefer_diffusers"
    if has_model_index or has_diffusers_components:
        return "diffusers", "diffusers_layout_signals_present"
    if has_weight_file:
        return "singlefile", "single_file_weight_signals_present"
    return "unknown", "layout_signals_missing"


def _is_aio_weight_candidate(path: str) -> bool:
    lower = str(path or "").strip().lower()
    if lower == "" or not lower.endswith(_WEIGHT_EXTS):
        return False
    parts = [p for p in lower.split("/") if p]
    if not parts:
        return False
    if any(part in _DIFFUSERS_COMPONENT_DIRS for part in parts):
        return False
    if any(part in _AIO_EXCLUDED_DIR_HINTS for part in parts):
        return False
    filename = parts[-1]
    if any(hint in filename for hint in _AIO_EXCLUDED_NAME_HINTS):
        return False
    return True


def _aio_candidate_rank(path: str) -> tuple[int, int, int, str]:
    lower = str(path or "").strip().lower()
    ext_score = 9
    for idx, ext in enumerate(_WEIGHT_EXTS):
        if lower.endswith(ext):
            ext_score = idx
            break
    depth = lower.count("/")
    return (ext_score, depth, len(lower), lower)


def select_huggingface_source_files(
    *,
    files: list[str],
    source_layout_preference: str | None = None,
    source_dtype_preference: list[str] | None = None,
) -> HFSourceFileSelection:
    normalized = _normalize_paths(files)
    preference = str(source_layout_preference or "auto").strip().lower() or "auto"
    if preference not in {"auto", "diffusers", "aio"}:
        raise ValueError("source_layout_preference must be one of: auto, diffusers, aio")
    if not normalized:
        return HFSourceFileSelection(
            source_layout_preference=preference,
            selected_source_layout="unknown",
            selected_paths=[],
            selected_aio_path="",
            selection_reason="no_files",
            detection_reason="layout_signals_missing",
        )

    detected_layout, detection_reason = _detect_source_layout(normalized)
    has_model_index, has_diffusers_components, _ = _analyze_layout_signals(normalized)
    diffusers_available = has_model_index or has_diffusers_components
    aio_candidates = sorted([path for path in normalized if _is_aio_weight_candidate(path)], key=_aio_candidate_rank)

    selected_layout = "unknown"
    selection_reason = "unknown"
    if preference == "diffusers":
        if not diffusers_available:
            raise ValueError("requested_source_layout_unavailable:diffusers")
        selected_layout = "diffusers"
        selection_reason = "preference_diffusers"
    elif preference == "aio":
        if not aio_candidates:
            raise ValueError("requested_source_layout_unavailable:aio")
        selected_layout = "singlefile"
        selection_reason = "preference_aio"
    else:
        if diffusers_available:
            selected_layout = "diffusers"
            selection_reason = "auto_prefer_diffusers" if aio_candidates else "auto_diffusers_only"
        elif aio_candidates:
            selected_layout = "singlefile"
            selection_reason = "auto_aio_only"
        else:
            selected_layout = detected_layout
            selection_reason = "auto_fallback_detected_layout"

    selected_paths: list[str]
    selected_aio_path = ""
    if selected_layout == "diffusers":
        selected_paths = [path for path in normalized if not _is_aio_weight_candidate(path)]
        if not selected_paths:
            selected_paths = list(normalized)
    elif selected_layout == "singlefile":
        selected_aio_path = aio_candidates[0] if aio_candidates else ""
        selected_paths = [selected_aio_path] if selected_aio_path != "" else list(normalized)
    else:
        selected_paths = list(normalized)

    # Dtype-aware pruning: among the layout-selected files, group weight files
    # by `(component, base_name)` and keep only the preferred dtype sibling.
    # Non-weight files (configs/tokenizers/scheduler configs) pass through
    # unconditionally.
    from .dtype_detection import filter_by_source_dtype_preference, parse_weight_file

    weight_infos = []
    non_weight_paths = []
    dropped_weight_paths: list[str] = []
    for p in selected_paths:
        info = parse_weight_file(p)
        if info is None:
            non_weight_paths.append(p)
        else:
            weight_infos.append(info)

    dtype_by_component: dict[str, str] = {}
    if weight_infos:
        kept_weights, dropped_weights, chosen_by_base = filter_by_source_dtype_preference(
            weight_infos,
            preference=list(source_dtype_preference or []),
        )
        selected_paths = non_weight_paths + [info.rel_path for info in kept_weights]
        dtype_by_component = dict(chosen_by_base)
        dropped_weight_paths = [info.rel_path for info in dropped_weights]

    return HFSourceFileSelection(
        source_layout_preference=preference,
        selected_source_layout=selected_layout,
        selected_paths=selected_paths,
        selected_aio_path=selected_aio_path,
        selection_reason=selection_reason,
        detection_reason=detection_reason,
        dtype_by_component=dtype_by_component,
        dropped_paths=tuple(dropped_weight_paths),
    )


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
    normalized = _normalize_paths(files)
    source_layout, reason = _detect_source_layout(normalized)

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
    "HFSourceFileSelection",
    "SourceLayoutInfo",
    "canonical_model_family_from_variant",
    "detect_huggingface_source_layout",
    "infer_model_family_variant_from_hint",
    "infer_model_family_from_hint",
    "select_huggingface_source_files",
]
