"""The source-layout detection schema — declarative matchers the engine runs."""

from __future__ import annotations

import re
from ..component_vocab import pipeline_component_dirs
from dataclasses import dataclass, field

from .repack_spec import DeclarationError


def normalize_letters_digits(raw: str) -> str:
    """``FLUX.2-klein-9B`` -> ``flux2klein9b``."""
    return "".join(ch for ch in str(raw or "").strip().lower() if ch.isalnum())


@dataclass(frozen=True)
class HintMatch:
    """A match against a free-text hint (repo name, ``_name_or_path``, a filename)."""

    all_tokens: tuple[str, ...] = ()
    any_tokens: tuple[str, ...] = ()
    raw_substrings: tuple[str, ...] = ()
    raw_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.all_tokens or self.any_tokens or self.raw_substrings or self.raw_patterns):
            raise DeclarationError("HintMatch declares no condition — it would match everything")
        for pattern in self.raw_patterns:
            re.compile(pattern)

    def matches(self, raw_hint: str) -> bool:
        raw = str(raw_hint or "").strip().lower()
        if raw == "":
            return False
        normalized = normalize_letters_digits(raw)
        if self.all_tokens and not all(tok in normalized for tok in self.all_tokens):
            return False
        if self.any_tokens and not any(tok in normalized for tok in self.any_tokens):
            return False
        if self.raw_substrings and not any(sub in raw for sub in self.raw_substrings):
            return False
        if self.raw_patterns and not any(re.search(p, raw) is not None for p in self.raw_patterns):
            return False
        return True


@dataclass(frozen=True)
class DirMatch:
    """A match against the set of top-level component directories in a tree."""

    requires: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (self.requires or self.any_of):
            raise DeclarationError("DirMatch must require at least one directory")

    def matches(self, dirs: frozenset[str]) -> bool:
        if not all(name in dirs for name in self.requires):
            return False
        if any(name in dirs for name in self.forbids):
            return False
        if self.any_of and not any(name in dirs for name in self.any_of):
            return False
        return True


@dataclass(frozen=True)
class LayoutDeclaration:
    """How one family variant is recognized from a downloaded repo."""

    variant: str
    family: str
    order: int = 100
    hints: tuple[HintMatch, ...] = ()
    class_match: tuple[HintMatch, ...] = ()
    dirs: tuple[DirMatch, ...] = ()
    root_sentinels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.family or "").strip():
            raise DeclarationError("LayoutDeclaration.family is empty")
        if not (self.hints or self.class_match or self.dirs or self.root_sentinels):
            raise DeclarationError(
                f"LayoutDeclaration({self.family!r}) declares no detection channel"
            )

    def matches_hint(self, hint: str) -> bool:
        return any(h.matches(hint) for h in self.hints)

    def matches_class(self, class_name: str) -> bool:
        return any(h.matches(class_name) for h in self.class_match)

    def matches_dirs(self, dirs: frozenset[str]) -> bool:
        return any(d.matches(dirs) for d in self.dirs)

    def matches_sentinels(self, root_files: frozenset[str]) -> bool:
        if not self.root_sentinels:
            return False
        return all(name.lower() in root_files for name in self.root_sentinels)


@dataclass(frozen=True)
class LayoutSignals:
    """Generic, family-free evidence that a directory is a diffusers tree at all."""

    component_dirs: frozenset[str] = field(
        default_factory=lambda: frozenset(pipeline_component_dirs())
    )
    index_file: str = "model_index.json"
    weight_suffixes: tuple[str, ...] = (".safetensors", ".gguf")


__all__ = [
    "DirMatch",
    "HintMatch",
    "LayoutDeclaration",
    "LayoutSignals",
    "normalize_letters_digits",
]
