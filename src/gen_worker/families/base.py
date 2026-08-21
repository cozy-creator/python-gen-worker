"""Shared per-family inference-defaults vocabulary."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import msgspec

_REGISTRY: Dict[Tuple[str, str], Type["GenerationDefaults"]] = {}

KIND_CHECKPOINT = "checkpoint"
KIND_LORA = "lora"
_VALID_KINDS = (KIND_CHECKPOINT, KIND_LORA)

F = TypeVar("F", bound="GenerationDefaults")


def _normalize_kind(kind: str) -> str:
    k = str(kind or KIND_CHECKPOINT).strip().lower() or KIND_CHECKPOINT
    if k not in _VALID_KINDS:
        raise ValueError(f"kind={kind!r} must be one of {_VALID_KINDS}")
    return k


class GenerationDefaults(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True,
):
    """Base for a per-family inference-defaults/constraints vocabulary."""

    schema_version: int = 1

    @property
    def family(self) -> str:
        """This instance's registered family name (``""`` for a subclass no family ever registered — an authoring mistake, not a valid preset)."""
        return str(getattr(type(self), "__gen_worker_family__", "") or "")

    @property
    def kind(self) -> str:
        """This instance's registered kind (``"checkpoint"`` | ``"lora"``); ``"checkpoint"`` for a subclass that never got a kind."""
        return str(getattr(type(self), "__gen_worker_kind__", "") or KIND_CHECKPOINT)


def register_family(
    name: str, cls: Type[F], *, kind: str = KIND_CHECKPOINT
) -> Type[F]:
    """Register a :class:`GenerationDefaults` subclass under ``(name, kind)``."""
    fam = str(name or "").strip()
    if not fam:
        raise ValueError("register_family(name=...) requires a non-empty name")
    knd = _normalize_kind(kind)
    if not (isinstance(cls, type) and issubclass(cls, GenerationDefaults)):
        raise TypeError(
            f"register_family({fam!r}, kind={knd!r}) needs a GenerationDefaults "
            f"subclass, got {cls!r}"
        )
    key = (fam, knd)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls:
        same_declaration = (
            existing.__module__ == cls.__module__
            and existing.__qualname__ == cls.__qualname__
        )
        if not same_declaration:
            raise ValueError(
                f"family {fam!r} kind {knd!r} already registered by "
                f"{existing.__module__}.{existing.__qualname__} "
                f"(redeclared by {cls.__module__}.{cls.__qualname__})"
            )
    cls.__gen_worker_family__ = fam  # type: ignore[attr-defined]
    cls.__gen_worker_kind__ = knd  # type: ignore[attr-defined]
    _REGISTRY[key] = cls
    return cls


def family_registry(*, kind: str = KIND_CHECKPOINT) -> Dict[str, Type[GenerationDefaults]]:
    """Every registered family of ``kind``, name -> struct class."""
    knd = _normalize_kind(kind)
    return {fam: cls for (fam, k), cls in _REGISTRY.items() if k == knd}


def family_for(name: str, *, kind: str = KIND_CHECKPOINT) -> Optional[Type[GenerationDefaults]]:
    """The registered family class for ``(name, kind)``, or ``None``."""
    return _REGISTRY.get((str(name or "").strip(), _normalize_kind(kind))) or None


def _clean_descriptions(node: Dict[str, Any]) -> None:
    desc = node.get("description")
    if isinstance(desc, str):
        node["description"] = inspect.cleandoc(desc)


def schema_filename(name: str, *, kind: str = KIND_CHECKPOINT) -> str:
    """The ``<family>[.lora].schema.json`` filename convention a family's exported schema is written under — shared by :func:`export_all_schemas`' caller (the ``families export-schemas`` CLI) and tensorhu..."""
    knd = _normalize_kind(kind)
    suffix = "" if knd == KIND_CHECKPOINT else f".{knd}"
    return f"{name}{suffix}.schema.json"


def export_json_schema(name: str, *, kind: str = KIND_CHECKPOINT) -> Dict[str, Any]:
    """Standalone JSON Schema (draft 2020-12) for one registered ``(family, kind)`` pair."""
    knd = _normalize_kind(kind)
    cls = family_for(name, kind=knd)
    if cls is None:
        registered = sorted(f"{fam}:{k}" for fam, k in _REGISTRY) or "(none)"
        raise KeyError(f"no family registered as {name!r} kind {knd!r}; registered: {registered}")
    raw = msgspec.json.schema(cls)
    defs = dict(raw.get("$defs") or {})
    body = defs.pop(cls.__name__, None)
    if body is None:
        body = {k: v for k, v in raw.items() if k not in ("$ref", "$defs")}
    _clean_descriptions(body)
    for d in defs.values():
        if isinstance(d, dict):
            _clean_descriptions(d)
    schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://schemas.cozy.art/gen-worker/families/{schema_filename(name, kind=knd)}",
        "title": cls.__name__,
        **body,
    }
    if defs:
        schema["$defs"] = defs
    return msgspec.json.decode(msgspec.json.encode(schema))


def export_all_schemas() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``{(family_name, kind): schema}`` for every registered family."""
    return {
        (fam, knd): export_json_schema(fam, kind=knd)
        for (fam, knd) in sorted(_REGISTRY)
    }


__all__ = [
    "KIND_CHECKPOINT",
    "KIND_LORA",
    "GenerationDefaults",
    "export_all_schemas",
    "export_json_schema",
    "family_for",
    "family_registry",
    "register_family",
    "schema_filename",
]
