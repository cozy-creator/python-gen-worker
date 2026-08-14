"""Source-layout / family detection — the generic engine over declared matchers.

File-selection logic lives in :mod:`gen_worker.convert.classifier`. This module
answers the downstream metadata question: given a repo_dir + file list, which
model family / variant is this? The output feeds destination checkpoint tags.

Family ladders are :class:`~.layout_spec.LayoutDeclaration` records registered by
the endpoint that owns the family. This file evaluates them; it names no family.
The four detection channels run in a fixed order:

1. ``model_index.json``'s ``_name_or_path`` as a free-text hint
2. ``model_index.json``'s ``_class_name``
3. the top-level component directory set
4. per-file hints, then root-file sentinels, then the whole listing as one hint

Ordering within a channel is the declaration's ``order`` field, which is how a
more specific variant (``flux2``) stays ahead of a broader one (``flux1``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .layout_spec import LayoutSignals, normalize_letters_digits
from .registry import registered_layouts
from ..models.file_layout import MULTI_FILE, SINGLE_FILE

_DEFAULT_SIGNALS = LayoutSignals()


def canonical_model_family_from_variant(variant: str) -> str:
    """Roll a fine-grained variant slug up to its declared canonical family."""
    raw = str(variant or "").strip().lower()
    if raw == "":
        return "unknown"
    for decl in registered_layouts():
        if decl.variant and decl.variant.lower() == raw:
            return decl.family
    for decl in registered_layouts():
        if decl.family.lower() == raw:
            return decl.family
    return "unknown"


def infer_model_family_variant_from_hint(value: str | None) -> str:
    """First declared variant whose hint matchers accept this free text."""
    hint = str(value or "").strip().lower()
    if hint == "":
        return "unknown"
    for decl in registered_layouts():
        if decl.variant and decl.matches_hint(hint):
            return decl.variant
    return "unknown"


def infer_model_family_from_hint(value: str | None) -> str:
    """Family-level hint resolution: variant matchers first, then family-only ones."""
    hint = str(value or "").strip().lower()
    if hint == "":
        return "unknown"
    variant = infer_model_family_variant_from_hint(hint)
    if variant != "unknown":
        return canonical_model_family_from_variant(variant)
    for decl in registered_layouts():
        if decl.matches_hint(hint):
            return decl.family
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


def _top_dirs(paths: list[str]) -> frozenset[str]:
    return frozenset(p.split("/", 1)[0] for p in paths if "/" in p)


def _root_files(paths: list[str]) -> frozenset[str]:
    return frozenset(p.lower() for p in paths if "/" not in p)


def _has_diffusers_layout_signals(paths: list[str], signals: LayoutSignals) -> bool:
    if signals.index_file in paths:
        return True
    return bool(signals.component_dirs & _top_dirs(paths))


def _detect_variant_from_model_index(repo_dir: Path) -> str:
    model_index_path = repo_dir / _DEFAULT_SIGNALS.index_file
    if not model_index_path.exists():
        return "unknown"
    try:
        payload = json.loads(model_index_path.read_text("utf-8"))
    except Exception:
        return "unknown"
    detected = infer_model_family_variant_from_hint(str(payload.get("_name_or_path") or "").strip())
    if detected != "unknown":
        return detected
    cls = str(payload.get("_class_name") or "").strip()
    if cls == "":
        return "unknown"
    for decl in registered_layouts():
        if decl.variant and decl.matches_class(cls):
            return decl.variant
    return "unknown"


def _detect_variant_from_components(paths: list[str]) -> str:
    dirs = _top_dirs(paths)
    for decl in registered_layouts():
        if decl.variant and decl.dirs and decl.matches_dirs(dirs):
            return decl.variant
    return "unknown"


def _detect_variant_from_paths(paths: list[str]) -> str:
    for path in paths:
        detected = infer_model_family_variant_from_hint(path)
        if detected != "unknown":
            return detected
    return "unknown"


def _detect_variant_from_sentinels(paths: list[str]) -> str:
    """Root-file sentinels: a repackage layout whose filenames carry no family token.

    The LTX-2 case (``video_vae_encoder.safetensors`` + friends) is ONE
    declaration, read by both the worker and the trainer.
    """
    root = _root_files(paths)
    for decl in registered_layouts():
        if decl.variant and decl.matches_sentinels(root):
            return decl.variant
    return "unknown"


def detect_huggingface_source_layout(*, repo_dir: Path, files: list[str]) -> SourceLayoutInfo:
    """Tagging-only metadata: detect diffusers-vs-singlefile shape + family variant.

    Used by ingest_from_source to populate ``model_family`` / ``model_family_variant``
    in the destination checkpoint metadata. Not a load-bearing decision —
    the file-selection strategy is determined upstream by the classifier.
    """
    signals = _DEFAULT_SIGNALS
    normalized = _normalize_paths(files)
    # Annotated `str`, not inferred: the ruled tokens are `Final[Literal[...]]`
    # so an inferred type here would narrow to the first branch's literal.
    source_layout: str
    if _has_diffusers_layout_signals(normalized, signals):
        source_layout = MULTI_FILE
        reason = "diffusers_layout_signals_present"
    elif any(p.lower().endswith(signals.weight_suffixes) for p in normalized):
        source_layout = SINGLE_FILE
        reason = "single_file_weight_signals_present"
    else:
        source_layout = "unknown"
        reason = "layout_signals_missing"

    variant = _detect_variant_from_model_index(repo_dir)
    if variant == "unknown":
        variant = _detect_variant_from_components(normalized)
    if variant == "unknown" and source_layout == SINGLE_FILE:
        variant = _detect_variant_from_paths(normalized)
    if variant == "unknown":
        variant = _detect_variant_from_sentinels(normalized)

    model_family = canonical_model_family_from_variant(variant)
    if model_family == "unknown":
        model_family = infer_model_family_from_hint(" ".join(normalized[:64]))

    return SourceLayoutInfo(
        source_layout=source_layout,
        model_family=model_family,
        model_family_variant=variant,
        detection_reason=reason,
    )


__all__ = [
    "SourceLayoutInfo",
    "canonical_model_family_from_variant",
    "detect_huggingface_source_layout",
    "infer_model_family_from_hint",
    "infer_model_family_variant_from_hint",
    "normalize_letters_digits",
]
