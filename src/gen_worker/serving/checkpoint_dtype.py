"""What dtype a checkpoint IS, for a lane that declares none (pgw#1488).

A layout contract states its load dtype, and a class that declares a contract
lane reads it there. A class that declares NO contract still has to load at
some precision, and the trace's precision is graph identity — a bf16 trace and
an fp32 trace are different graphs, so "whatever happens" is not an answer.

The checkpoint is the answer. Two sources, in order of how directly they know:

1. the root config's ``torch_dtype``/``dtype`` (``model_index.json`` for a
   diffusers pipeline, ``config.json`` for a bare module) — what the packager
   SAID;
2. the safetensors headers — what the tensors ARE. Headers are read as headers
   (an 8-byte length and a JSON blob at the front of the file), so this costs
   a few KB and never opens a weight.

(1) leads because a packager who states a dtype is stating the serving
intent — an fp32 file saved from a bf16 recipe is real. (2) is the fallback
because it cannot be absent, and it is the only source for the many
checkpoints that ship raw ``.safetensors`` with no config at all. ``None``
means neither source spoke, and the author's loader keeps its own default.

Both ends of the pipeline read this: the trace (``TraceLoadContext.load``) and
the serve (``LoadContext``). One function, so they cannot disagree about the
precision a derived lane runs at.
"""

from __future__ import annotations

import json
import logging
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_LOG = logging.getLogger("gen_worker.serving.checkpoint_dtype")

#: safetensors dtype spellings -> torch attribute names.
_ST_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}

#: Root config files, most specific first.
_CONFIGS = ("model_index.json", "config.json")

#: Header prefix cap. A safetensors header is JSON metadata; anything past
#: this is a corrupt or hostile file, not a header we want in memory.
_MAX_HEADER = 100 * 1024 * 1024


def torch_dtype(name: Any) -> Any:
    """A dtype SPELLING (safetensors or torch) as a real ``torch.dtype``."""

    import torch

    if name is None or isinstance(name, torch.dtype):
        return name
    if not isinstance(name, str):
        return None
    spelled = _ST_DTYPES.get(name.upper(), name.lower())
    if spelled.startswith("torch."):
        spelled = spelled[len("torch."):]
    candidate = getattr(torch, spelled, None)
    return candidate if isinstance(candidate, torch.dtype) else None


def _config_dtype(tree: Path) -> Any:
    for name in _CONFIGS:
        path = tree / name
        if not path.is_file():
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        for key in ("torch_dtype", "dtype"):
            resolved = torch_dtype(config.get(key))
            if resolved is not None:
                return resolved
    return None


def _header_dtypes(path: Path) -> Counter[str]:
    """The floating dtype tally of one safetensors file's header."""

    tally: Counter[str] = Counter()
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                return tally
            length = struct.unpack("<Q", raw)[0]
            if length <= 0 or length > _MAX_HEADER:
                return tally
            header = json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return tally
    if not isinstance(header, dict):
        return tally
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        spelling = entry.get("dtype")
        if isinstance(spelling, str) and spelling.upper() in _ST_DTYPES:
            tally[spelling.upper()] += 1
    return tally


def _tensor_dtype(tree: Path) -> Any:
    """The dominant floating dtype across the tree's safetensors headers.

    Every shard is read (headers only) rather than just the first: a pipeline
    keeps its text encoder and its VAE beside the denoiser, and the file that
    happens to sort first is not the one that decides the recipe. The majority
    of DECLARED TENSORS wins, which is the denoiser in every real checkpoint.
    """

    tally: Counter[str] = Counter()
    for path in sorted(tree.rglob("*.safetensors")):
        tally.update(_header_dtypes(path))
    if not tally:
        return None
    return torch_dtype(tally.most_common(1)[0][0])


def checkpoint_dtype(tree: Optional[Path]) -> Any:
    """The dtype ``tree`` advertises, or ``None`` when it advertises none."""

    if tree is None:
        return None
    path = Path(tree)
    if not path.is_dir():
        return None
    return _config_dtype(path) or _tensor_dtype(path)


__all__ = ["checkpoint_dtype", "torch_dtype"]
