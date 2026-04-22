"""Dtype detection + dtype-preference filtering for HuggingFace source repos.

HF diffusers / transformers repos ship the same weights in multiple dtypes
under a shared component directory, distinguished by filename suffix:

    unet/diffusion_pytorch_model.safetensors         -> fp32 (unsuffixed)
    unet/diffusion_pytorch_model.fp16.safetensors    -> fp16
    unet/diffusion_pytorch_model.bf16.safetensors    -> bf16

The filename suffix is a strong hint but not authoritative — some repos
ship only `.safetensors` that internally are fp16. `parse_weight_file_dtype`
returns the suffix-inferred dtype; if that returns `"fp32"` and the caller
wants authoritative detection, they can fall back to reading the safetensors
JSON header.

A sharded weight looks like either:

    model-00001-of-00002.safetensors
    diffusion_pytorch_model-00001-of-00002.fp16.safetensors

Sharded shards all share a dtype. They group together via `base_key`.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_WEIGHT_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")

# Recognized dtype suffixes that sit immediately before the weight extension.
# Order matters for longest-match: check multi-char tokens before short ones.
_DTYPE_SUFFIXES = [
    "bf16",
    "fp16",
    "fp32",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8",
    "int8",
    "int4",
    "nf4",
    "nvfp4",
]

# sharded-weight pattern: "...-00001-of-00042..."
_SHARD_RE = re.compile(r"-(\d{4,6})-of-(\d{4,6})(?=[.]|$)")


@dataclass(frozen=True)
class WeightFileInfo:
    """Per-weight-file metadata used by dtype-aware file selection."""

    rel_path: str
    component: str
    base_key: str  # (component, base_name_without_dtype_or_shard)
    dtype: str  # canonical dtype label; "fp32" when unsuffixed
    extension: str
    is_sharded: bool
    shard_index: int | None
    shard_count: int | None


def _split_ext(name: str) -> tuple[str, str]:
    lower = name.lower()
    for ext in _WEIGHT_EXTS:
        if lower.endswith(ext):
            return name[: -len(ext)], ext
    return name, ""


def _peel_dtype_suffix(stem: str) -> tuple[str, str]:
    """Strip a trailing `.<dtype>` segment from a filename stem.

    Returns `(stem_without_dtype, detected_dtype)`. When no recognized
    dtype suffix is present, returns `(stem, "fp32")` — the diffusers
    convention where `.safetensors` without a suffix is the master
    (usually fp32) copy.
    """
    lower = stem.lower()
    for sfx in _DTYPE_SUFFIXES:
        marker = "." + sfx
        if lower.endswith(marker):
            return stem[: -len(marker)], _canonicalize_dtype(sfx)
    return stem, "fp32"


def _canonicalize_dtype(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"fp8_e4m3", "fp8-e4m3"}:
        return "fp8:e4m3"
    if s in {"fp8_e5m2", "fp8-e5m2"}:
        return "fp8:e5m2"
    return s


def _peel_shard_suffix(stem: str) -> tuple[str, int | None, int | None]:
    match = _SHARD_RE.search(stem)
    if not match:
        return stem, None, None
    idx = int(match.group(1))
    count = int(match.group(2))
    return stem[: match.start()] + stem[match.end():], idx, count


def parse_weight_file(rel_path: str) -> WeightFileInfo | None:
    """Return `WeightFileInfo` for a diffusers-style weight file, else None.

    Non-weight files (configs, tokenizers, README, scheduler configs, etc.)
    return None and should be kept unconditionally by the caller.
    """
    clean = str(rel_path or "").strip().replace("\\", "/").lstrip("/")
    if clean == "":
        return None

    path = PurePosixPath(clean)
    name = path.name
    if not name:
        return None

    stem, ext = _split_ext(name)
    if ext == "":
        return None

    stem, shard_index, shard_count = _peel_shard_suffix(stem)
    stem, dtype = _peel_dtype_suffix(stem)

    parts = list(path.parts)
    if len(parts) >= 2:
        component = parts[0].lower()
    else:
        component = ""

    base_key = f"{component}/{stem}" if component else stem

    return WeightFileInfo(
        rel_path=clean,
        component=component,
        base_key=base_key,
        dtype=dtype,
        extension=ext,
        is_sharded=shard_index is not None,
        shard_index=shard_index,
        shard_count=shard_count,
    )


_DEFAULT_DTYPE_PREFERENCE = ("bf16", "fp16", "fp32")


def filter_by_source_dtype_preference(
    weight_files: list[WeightFileInfo],
    *,
    preference: list[str] | None = None,
) -> tuple[list[WeightFileInfo], list[WeightFileInfo], dict[str, str]]:
    """Group weight files by `base_key`, pick the preferred dtype per group.

    Returns `(kept, dropped, chosen_by_base)`:
      - `kept`: file infos we want to download/use.
      - `dropped`: file infos belonging to rejected-dtype siblings.
      - `chosen_by_base`: `base_key -> chosen dtype` (for lineage metadata).

    Preference order of fallback: when none of the preferred dtypes exist
    for a group, keep the lowest-precision fallback (fp32 > fp16 > anything
    exotic) so we always end up with *something* per group.
    """
    prefs = [str(p or "").strip().lower() for p in (preference or []) if str(p or "").strip()]
    prefs = [_canonicalize_dtype(p) for p in prefs]
    if not prefs:
        prefs = list(_DEFAULT_DTYPE_PREFERENCE)

    groups: dict[str, list[WeightFileInfo]] = {}
    for info in weight_files:
        groups.setdefault(info.base_key, []).append(info)

    kept: list[WeightFileInfo] = []
    dropped: list[WeightFileInfo] = []
    chosen_by_base: dict[str, str] = {}

    for base_key, group in groups.items():
        available = {info.dtype for info in group}
        chosen_dtype = ""
        for pref in prefs:
            if pref in available:
                chosen_dtype = pref
                break
        if chosen_dtype == "":
            # Nothing matches preferences — fall back to the most "canonical"
            # dtype the group has, preferring fp32 > fp16 > bf16 > others so
            # inference can always find a weight.
            fallback_order = ["fp32", "fp16", "bf16"] + [d for d in available if d not in {"fp32", "fp16", "bf16"}]
            for candidate in fallback_order:
                if candidate in available:
                    chosen_dtype = candidate
                    break
        chosen_by_base[base_key] = chosen_dtype
        for info in group:
            if info.dtype == chosen_dtype:
                kept.append(info)
            else:
                dropped.append(info)

    return kept, dropped, chosen_by_base


# Map safetensors header dtype strings to our canonical labels.
_SAFETENSORS_DTYPE_MAP = {
    "BF16": "bf16",
    "F16": "fp16",
    "F32": "fp32",
    "F64": "fp64",
    "F8_E4M3": "fp8:e4m3",
    "F8_E5M2": "fp8:e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def read_safetensors_header_dtype(path: str | Path) -> str | None:
    """Peek the safetensors header and return the dominant tensor dtype.

    Reads only the header (first 8 bytes = little-endian u64 header length,
    then that many bytes of JSON). No tensor data is loaded.

    Returns the canonical dtype label (matches `_DTYPE_SUFFIXES` vocabulary)
    of the largest tensor by element count, or ``None`` if the header can't
    be parsed. We pick the largest tensor because a weights file may carry
    small bookkeeping tensors in a different dtype (e.g. int64 index vectors)
    alongside the bulk fp16/bf16 weight; what we want is the dtype of the
    weights, not of an incidental int tensor.
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return None
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or header_len > 100 * 1024 * 1024:
                # Sanity cap: safetensors headers are tens of MB at most.
                return None
            header_bytes = f.read(int(header_len))
            if len(header_bytes) != header_len:
                return None
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(header, dict):
        return None

    best_dtype: str | None = None
    best_numel = -1
    for key, entry in header.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        raw_dtype = entry.get("dtype")
        if not isinstance(raw_dtype, str):
            continue
        canonical = _SAFETENSORS_DTYPE_MAP.get(raw_dtype.upper().strip())
        if canonical is None:
            continue
        shape = entry.get("shape")
        if not isinstance(shape, list):
            continue
        numel = 1
        try:
            for dim in shape:
                numel *= int(dim)
        except Exception:
            continue
        if numel > best_numel:
            best_numel = numel
            best_dtype = canonical
    return best_dtype


__all__ = [
    "WeightFileInfo",
    "parse_weight_file",
    "filter_by_source_dtype_preference",
    "read_safetensors_header_dtype",
]
