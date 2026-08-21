from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_ST_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}

_CONFIGS = ("model_index.json", "config.json")


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

    from ..models.safetensors_header import read_header

    tally: Counter[str] = Counter()
    try:
        header = read_header(
            path,
            why="a lane with no contract takes its load dtype from the "
                "checkpoint, and a header that reads as empty would silently "
                "become the loader's default precision instead",
        )
    except Exception:  # noqa: BLE001 - a dtype PROBE never kills a load
        return tally
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        spelling = entry.get("dtype")
        if isinstance(spelling, str) and spelling.upper() in _ST_DTYPES:
            tally[spelling.upper()] += 1
    return tally


def _tensor_dtype(tree: Path) -> Any:

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
