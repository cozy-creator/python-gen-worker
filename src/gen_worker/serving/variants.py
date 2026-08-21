from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_SHARD = r"\d{5}-of-\d{5}"

_SUFFIXES = ("safetensors", "bin")

_TOKEN = r"[A-Za-z0-9][A-Za-z0-9_]{0,15}"

_VARIANT_WEIGHT = re.compile(
    rf"^(?P<stem>.+?)\.(?P<token>{_TOKEN})(?:-{_SHARD})?\.(?P<suffix>{'|'.join(_SUFFIXES)})$"
)
_PLAIN_WEIGHT = re.compile(
    rf"^(?P<stem>.+?)(?:-{_SHARD})?\.(?P<suffix>{'|'.join(_SUFFIXES)})$"
)


class VariantAmbiguous(RuntimeError):
    """A tree offers several weight variants and none is plainly named."""


def _component_dirs(tree: Path) -> list[Path]:

    dirs = [tree]
    try:
        dirs += sorted(p for p in tree.iterdir() if p.is_dir())
    except OSError:
        return [tree]
    return dirs


def _classify(directory: Path) -> Tuple[set[str], bool]:

    tokens: set[str] = set()
    plain = False
    try:
        names = sorted(p.name for p in directory.iterdir() if p.is_file())
    except OSError:
        return tokens, plain
    for name in names:
        variant = _VARIANT_WEIGHT.match(name)
        if variant is not None:
            tokens.add(variant.group("token"))
            continue
        if _PLAIN_WEIGHT.match(name) is not None:
            plain = True
    return tokens, plain


def detect_variant(tree: Path | str) -> Optional[str]:
    """The variant this tree must be loaded with, or ``None``."""

    root = Path(tree)
    if not root.is_dir():
        return None
    tokens: set[str] = set()
    for directory in _component_dirs(root):
        found, plain = _classify(directory)
        if plain:
            return None
        tokens |= found
    if not tokens:
        return None
    if len(tokens) > 1:
        raise VariantAmbiguous(
            f"{root} offers weight variants {sorted(tokens)!r} and no plainly "
            f"named weight file. Which one to serve is a statement about "
            f"PRECISION and it belongs to whoever published the tree, not to "
            f"the worker — project or convert the one you mean."
        )
    token = tokens.pop()
    logger.info(
        "ctx.load: %s carries only %r-variant weight files and no plainly "
        "named ones, so the eager bridge loads it with variant=%r (pgw#1473)",
        root, token, token,
    )
    return token


__all__ = ["VariantAmbiguous", "detect_variant"]
