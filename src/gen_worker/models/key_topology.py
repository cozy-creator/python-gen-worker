"""Which tensor-KEY convention an artifact on disk is written in.

Header reads only — no tensor data, no torch, no model construction. That is
the whole point: the answer must be available BEFORE a 71 GB fetch turns into
a rented pod that discovers the mismatch as `Cannot detect the model type`
from an md5-over-key:shape lookup five libraries down.

Classification is by the keys that DIFFER, not by counting: the minimax-h3
diffusers repackaging and the minimax-native tree share exactly one key out of
638/535, and the attention projection is where they part — fused
`blocks.N.attn.qkv_proj` versus split `transformer_blocks.N.attn.to_q`.

An unrecognized key set returns ``""``: UNKNOWN, which refuses nothing. A
classifier that refused what it could not name would be an upper bound
wearing a lower bound's clothes, and the fleet is full of legal trees this
one has never seen.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Iterable

from .safetensors_header import header_len_ok
from .tensor_layout_contract import (
    KEYS_DIFFUSERS_SPLIT_QKV,
    KEYS_NATIVE_FUSED_QKV,
    KEYS_TRANSFORMERS_NATIVE,
)

# Ordered: the first rule whose pattern appears wins, and the two denoiser
# conventions are checked before the generic transformers one because a
# transformers-style prefix can appear inside either.
_RULES: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (KEYS_NATIVE_FUSED_QKV,
     re.compile(r"(^|\.)blocks\.\d+\..*\.(qkv_proj|qkv)\.")),
    (KEYS_DIFFUSERS_SPLIT_QKV,
     re.compile(r"(^|\.)transformer_blocks\.\d+\..*\.to_q\.")),
    (KEYS_TRANSFORMERS_NATIVE,
     re.compile(r"(^|\.)(model\.layers|encoder\.layer|encoder\.block)\.\d+\.")),
)

#: Header bytes are bounded by `header_len_ok`; this caps how many FILES we
#: open, because a sharded tree's shards share one convention.
_MAX_FILES = 8


def tensor_keys(files: Iterable[Path]) -> tuple[str, ...]:
    """Every tensor name in the given safetensors files' headers."""
    keys: list[str] = []
    for count, path in enumerate(files):
        if count >= _MAX_FILES:
            break
        try:
            with open(path, "rb") as handle:
                raw = handle.read(8)
                if len(raw) < 8:
                    continue
                (length,) = struct.unpack("<Q", raw)
                if not header_len_ok(length):
                    continue
                header = json.loads(handle.read(length))
        except (OSError, ValueError, struct.error):
            continue
        if isinstance(header, dict):
            keys.extend(k for k in header if k != "__metadata__")
    return tuple(keys)


def identify_keys(keys: Iterable[str]) -> str:
    """The key convention a tensor-name set is written in, or ``""``."""
    names = list(keys)
    for token, pattern in _RULES:
        if any(pattern.search(name) for name in names):
            return token
    return ""


def identify_snapshot_keys(root: Path, component: str = "") -> str:
    """The key convention of a snapshot's denoiser tree, or ``""``.

    ``component`` names the subdirectory to read; empty scans the root's own
    safetensors (single-file and root-layout trees).
    """
    from ..component_vocab import denoiser_components

    base = Path(root)
    if base.is_file():
        return identify_keys(tensor_keys([base]))
    if not base.is_dir():
        return ""
    candidates = [base / component] if component else [
        base / name for name in denoiser_components()
    ] + [base]
    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(p for p in directory.glob("*.safetensors") if p.is_file())
        if not files:
            continue
        found = identify_keys(tensor_keys(files))
        if found:
            return found
    return ""
