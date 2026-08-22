"""The TREE-REPACK declaration schema — a key map, not a model.

This is the second repackage vocabulary in ``convert/`` and the two are not
alternatives. :mod:`repack_spec` declares a LOAD-and-resave: a diffusers
pipeline class is named, instantiated from a single file and written back out
(``SinglefileTarget(pipeline_class=…)``). That needs a class the diffusers SDK
can name, weights in memory, and a card's worth of RAM.

A tree repack is a transform over HEADERS AND NAMES. It re-routes a flat
transformers-shaped tree's keys into diffusers component directories, writes
each component's ``config.json`` from the source config by declared field, moves
the tokenizer files into their own directory, and emits ``model_index.json``.
No tensor is ever materialized: tensor DATA is copied as byte ranges, so every
tensor in the produced tree is byte-identical to the one it came from.

Why the platform needs one (se#840's ruling, pgw#1670): a mirror reproduces the
upstream layout, and a flat root beside a ``config.json`` whose ``auto_map``
names ``.py`` files that do not exist cannot be ``ctx.load``ed hollow-legally —
the hollow session intercepts only diffusers/transformers loaders, so an
endpoint's component classes have to be reachable through a
``model_index.json``. Doing it as a CONVERSION LEG keeps one producer of the
published artifact and keeps the serving path free of per-endpoint tree-shape
knowledge.

⚠️ A declaration is refused at declaration time for anything that would DROP
keys silently — an unroutable key, two catch-alls, a component that carries
neither weights nor files. Losing a tensor is the failure mode this schema
exists to make impossible, and it is not detectable downstream: the tree still
loads, one weight short.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .repack_spec import DeclarationError, RenameRule

__all__ = [
    "ComponentConfig",
    "ConfigField",
    "DeclarationError",
    "FileRoute",
    "RepackComponent",
    "TreeRepack",
    "TreeRepackError",
]


class TreeRepackError(ValueError):
    """A tree cannot be repacked as declared — a REFUSAL, never a fallback."""


@dataclass(frozen=True)
class ConfigField:
    """One field of a produced JSON document, and where its value comes from.

    Exactly one of ``source`` (a dotted path in the source document) and
    ``value`` (a literal) supplies it. A declared ``source`` that the source
    document does not carry REFUSES when ``required`` — a config field silently
    defaulting is how a serving config drifts from the checkpoint it describes.
    """

    target: str
    source: str = ""
    value: Any = None
    required: bool = True

    def __post_init__(self) -> None:
        if not str(self.target or "").strip():
            raise DeclarationError("ConfigField.target is empty")
        if bool(str(self.source or "").strip()) == (self.value is not None):
            raise DeclarationError(
                f"ConfigField {self.target!r} must declare exactly one of "
                "source=<dotted path> or value=<literal>"
            )


@dataclass(frozen=True)
class ComponentConfig:
    """How one component's ``config.json`` is derived from the source tree."""

    source: str = "config.json"
    fields: tuple[ConfigField, ...] = ()

    def __post_init__(self) -> None:
        if not self.fields:
            raise DeclarationError("ComponentConfig declares no fields")
        seen: set[str] = set()
        for f in self.fields:
            if f.target in seen:
                raise DeclarationError(f"config field {f.target!r} declared twice")
            seen.add(f.target)
        if any(f.source for f in self.fields) and not str(self.source or "").strip():
            raise DeclarationError(
                "ComponentConfig reads source fields but names no source document")


@dataclass(frozen=True)
class FileRoute:
    """A non-weight file that MOVES from the tree root into a component."""

    source: str
    target_name: str = ""
    json_overrides: tuple[ConfigField, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        name = str(self.source or "").strip()
        if not name or name.startswith("/") or ".." in name.split("/"):
            raise DeclarationError(f"FileRoute.source is not a tree-relative path: {self.source!r}")
        for override in self.json_overrides:
            if override.source:
                raise DeclarationError(
                    f"FileRoute {name!r}: a json override sets a literal; "
                    f"{override.target!r} declares source={override.source!r}")

    @property
    def name(self) -> str:
        return str(self.target_name or "").strip() or self.source.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class RepackComponent:
    """One directory of the produced tree, and one row of ``model_index.json``."""

    name: str
    library: str
    class_name: str
    key_prefixes: tuple[str, ...] = ()
    rules: tuple[RenameRule, ...] = ()
    weight_stem: str = ""
    config: ComponentConfig | None = None
    files: tuple[FileRoute, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name or "/" in name or name.startswith("."):
            raise DeclarationError(f"RepackComponent.name is not a directory name: {self.name!r}")
        if not str(self.library or "").strip():
            raise DeclarationError(f"component {name!r} declares no model_index library")
        if not str(self.class_name or "").strip():
            raise DeclarationError(f"component {name!r} declares no model_index class")
        if self.key_prefixes and not self.weight_stem:
            raise DeclarationError(
                f"component {name!r} routes keys but declares no weight_stem — "
                "the tensors would have nowhere to land")
        if not self.carries_weights and not self.files:
            raise DeclarationError(
                f"component {name!r} carries neither weights nor files: it would be an "
                "EMPTY directory named by model_index.json, which loads as a missing "
                "component rather than as an error")
        if self.rules and not self.carries_weights:
            raise DeclarationError(f"component {name!r} declares key rules but no weights")

    @property
    def carries_weights(self) -> bool:
        return bool(str(self.weight_stem or "").strip())

    @property
    def is_catch_all(self) -> bool:
        return self.carries_weights and not self.key_prefixes

    def claims(self, key: str) -> bool:
        return any(key.startswith(p) for p in self.key_prefixes)

    def rename(self, key: str) -> str:
        out = key
        for rule in self.rules:
            out = rule.apply(out)
        return out


@dataclass(frozen=True)
class TreeRepack:
    """One family's complete flat-to-diffusers key map."""

    name: str
    pipeline_class: str
    components: tuple[RepackComponent, ...]
    requires_key_prefixes: tuple[str, ...] = ()
    diffusers_version: str = "0.39.0"
    keep_root: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise DeclarationError("TreeRepack.name is empty")
        if not str(self.pipeline_class or "").strip():
            raise DeclarationError(f"repack {self.name!r} declares no pipeline_class")
        if not self.components:
            raise DeclarationError(f"repack {self.name!r} declares no components")
        seen: set[str] = set()
        for comp in self.components:
            if comp.name in seen:
                raise DeclarationError(
                    f"repack {self.name!r} declares component {comp.name!r} twice")
            seen.add(comp.name)
        weighted = [c for c in self.components if c.carries_weights]
        if not weighted:
            raise DeclarationError(
                f"repack {self.name!r} routes no weights — a repack that moves no tensor "
                "produces a tree with no model in it")
        catch_alls = [c.name for c in weighted if c.is_catch_all]
        if len(catch_alls) > 1:
            raise DeclarationError(
                f"repack {self.name!r} declares {len(catch_alls)} catch-all weight components "
                f"({', '.join(catch_alls)}) — routing would depend on declaration order and "
                "the loser would silently take no keys")
        if catch_alls and weighted[-1].name != catch_alls[0]:
            raise DeclarationError(
                f"repack {self.name!r}: the catch-all component {catch_alls[0]!r} must be the "
                "LAST weight component, otherwise the components after it can never match")
        routed = {f.source for c in self.components for f in c.files}
        if len(routed) != sum(len(c.files) for c in self.components):
            raise DeclarationError(
                f"repack {self.name!r} routes one source file into two components")

    @property
    def weight_components(self) -> tuple[RepackComponent, ...]:
        return tuple(c for c in self.components if c.carries_weights)

    @property
    def has_catch_all(self) -> bool:
        return any(c.is_catch_all for c in self.components)

    @property
    def is_pure_move(self) -> bool:
        """True when no member can ever need rewriting.

        One weight component takes every key and no key is renamed, so each
        source member becomes a component member by ``os.replace``: zero bytes
        read, zero bytes written, zero extra disk. The disk preflight prices
        this property rather than assuming either answer.
        """

        return len(self.weight_components) == 1 and not any(c.rules for c in self.components)

    def component_for(self, key: str) -> RepackComponent | None:
        """The component a source key lands in, or ``None`` if nothing claims it."""
        for comp in self.weight_components:
            if comp.is_catch_all or comp.claims(key):
                return comp
        return None
