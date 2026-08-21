from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

MULTI_FILE: Final[Literal["multi-file"]] = "multi-file"

SINGLE_FILE: Final[Literal["single-file"]] = "single-file"

NOT_APPLICABLE: Final[Literal[""]] = ""

FileLayout = Literal["multi-file", "single-file"]

KNOWN_FILE_LAYOUTS: frozenset[str] = frozenset({MULTI_FILE, SINGLE_FILE})

_DEAD_SPELLINGS = {
    "singlefile": f"this repo's pre-th#1937 Literal; write {SINGLE_FILE!r}",
    "diffusers": f"a LIBRARY name — it belongs in library_name, and this axis wants {MULTI_FILE!r}",
    "multifile": f"write {MULTI_FILE!r}",
    "transformers": "a LIBRARY name — a transformers tree declares no file_layout at all",
    "single_file": f"write {SINGLE_FILE!r}",
    "multi_file": f"write {MULTI_FILE!r}",
}


def validate_file_layout(token: str) -> str:
    """Return the token, or raise ``ValueError`` naming the vocabulary."""
    token = (token or "").strip()
    if token == NOT_APPLICABLE or token in KNOWN_FILE_LAYOUTS:
        return token
    why = _DEAD_SPELLINGS.get(token.lower())
    detail = f" ({why})" if why else ""
    raise ValueError(
        f"file_layout_unknown_token: {token!r} is not a file_layout token{detail}. "
        f"The vocabulary is {SINGLE_FILE} | {MULTI_FILE}, ruled at th#1937 with no aliases"
    )


def is_single_file_snapshot(path: Path) -> bool:
    """The single-file SHAPE, from names alone."""
    if path.is_file():
        return path.suffix == ".safetensors"
    if not path.is_dir():
        return False
    if (path / "model_index.json").exists() or (path / "config.json").exists():
        return False
    singles = [p for p in path.glob("*.safetensors") if p.is_file()]
    if len(singles) == 1:
        return True
    return bool(singles) and len(list(path.glob("*.safetensors.index.json"))) == 1


def observed_file_layout(path: Path) -> str:
    """The layout a tree ON DISK is in, or :data:`NOT_APPLICABLE`."""
    p = Path(path)
    if is_single_file_snapshot(p):
        return SINGLE_FILE
    if p.is_dir() and (
            (p / "model_index.json").exists() or (p / "config.json").exists()):
        return MULTI_FILE
    return NOT_APPLICABLE


__all__ = [
    "FileLayout",
    "KNOWN_FILE_LAYOUTS",
    "MULTI_FILE",
    "NOT_APPLICABLE",
    "SINGLE_FILE",
    "is_single_file_snapshot",
    "observed_file_layout",
    "validate_file_layout",
]
