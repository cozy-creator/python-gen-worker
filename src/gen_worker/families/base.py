"""Shared per-family inference-defaults vocabulary (pgw#520 / th#767).

The MODEL SET is catalog, not code (th#767) — but the SHAPE of a family's
inference defaults/constraints (scheduler choices, step counts, guidance
clamps, ...) is still a code-side contract: it is what lets tensorhub
validate REPO METADATA at PUT time (fail at config, not at invoke) and what
lets an endpoint's :class:`~gen_worker.api.slot.Slot` carry a typed
FALLBACK preset for repos that publish no metadata.

A family struct is declared once per architecture and shared by every
endpoint that serves it::

    @family("sdxl")
    class SdxlDefaults(FamilyDefaults):
        scheduler: Literal["euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"] = "euler_a"
        steps: int = 28
        guidance: float = 6.0
        quality_preamble: str = ""
        negative: str = ""
        max_guidance: float | None = None  # a CLAMP, never a wire reshape

``@family(...)`` self-registers the class in the module-level registry
(keyed by name) — :func:`family_for` / :func:`family_registry` look it up
by name, the way :class:`~gen_worker.api.slot.Slot`'s resolution chain does
when repo metadata JSON arrives with no code fallback to decode against.

A DECORATOR, not a ``class X(FamilyDefaults, family="sdxl")`` class kwarg:
msgspec's own ``StructMeta`` does not forward unrecognized class keywords to
``__init_subclass__`` (verified: it raises ``TypeError`` on an unknown
kwarg), and mypy cannot type-check a metaclass computed at runtime
(``type(msgspec.Struct)``) as a valid base for a wrapping metaclass either —
both dead ends given the "mypy 0" gate, hence the decorator.

``forbid_unknown_fields=True`` on the base is what makes
:func:`export_json_schema` emit ``additionalProperties: false`` — the
contract tensorhub validates repo metadata against: a closed vocabulary per
family, not an open bag of keys.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

import msgspec

_REGISTRY: Dict[str, Type["FamilyDefaults"]] = {}

F = TypeVar("F", bound="FamilyDefaults")


class FamilyDefaults(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True,
):
    """Base for a per-family inference-defaults/constraints vocabulary.

    Versioned: ``schema_version`` lets tensorhub gate stored repo metadata
    against the version it was validated under when a family's fields
    change shape. Subclasses add their own typed fields (scheduler enum,
    step counts, guidance clamps, prompt dialect strings, ...) and register
    with ``@family("...")`` (see the module docstring).

    ``forbid_unknown_fields=True`` is inherited by every family — the
    exported JSON schema is closed (``additionalProperties: false``): a
    contract, not a suggestion.
    """

    schema_version: int = 1

    @property
    def family(self) -> str:
        """This instance's registered family name (``""`` for a subclass
        that never got ``@family("...")`` — an authoring mistake, not a
        valid preset)."""
        return str(getattr(type(self), "__gen_worker_family__", "") or "")


def family(name: str) -> Callable[[Type[F]], Type[F]]:
    """Class decorator: register a :class:`FamilyDefaults` subclass under
    ``name`` — the key :func:`family_for` / tensorhub's repo-metadata
    validation / a :class:`~gen_worker.api.slot.Slot`'s ``Compile(family=)``
    reconciliation all look it up by."""
    fam = str(name or "").strip()
    if not fam:
        raise ValueError("family(name=...) requires a non-empty name")

    def deco(cls: Type[F]) -> Type[F]:
        if not (isinstance(cls, type) and issubclass(cls, FamilyDefaults)):
            raise TypeError(
                f"@family({fam!r}) must decorate a FamilyDefaults subclass, "
                f"got {cls!r}"
            )
        existing = _REGISTRY.get(fam)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"family {fam!r} already registered by "
                f"{existing.__module__}.{existing.__qualname__} "
                f"(redeclared by {cls.__module__}.{cls.__qualname__})"
            )
        cls.__gen_worker_family__ = fam  # type: ignore[attr-defined]
        _REGISTRY[fam] = cls
        return cls

    return deco


def family_registry() -> Dict[str, Type[FamilyDefaults]]:
    """Every registered family, name -> struct class."""
    return dict(_REGISTRY)


def family_for(name: str) -> Optional[Type[FamilyDefaults]]:
    """The registered family class for ``name``, or ``None``."""
    return _REGISTRY.get(str(name or "").strip()) or None


def export_json_schema(name: str) -> Dict[str, Any]:
    """Standalone JSON Schema (draft 2020-12) for one registered family.

    Flattens msgspec's ``{$ref, $defs}`` output into one closed document —
    the shape tensorhub validates repo metadata against at PUT time.
    """
    cls = family_for(name)
    if cls is None:
        raise KeyError(
            f"no family registered as {name!r}; registered: "
            f"{sorted(_REGISTRY) or '(none)'}"
        )
    raw = msgspec.json.schema(cls)
    defs = dict(raw.get("$defs") or {})
    body = defs.pop(cls.__name__, None)
    if body is None:
        # No nested defs (a family with no struct-valued fields): the
        # top-level schema IS the body.
        body = {k: v for k, v in raw.items() if k not in ("$ref", "$defs")}
    schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://schemas.cozy.art/gen-worker/families/{name}.schema.json",
        "title": cls.__name__,
        **body,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def export_all_schemas() -> Dict[str, Dict[str, Any]]:
    """``{family_name: schema}`` for every registered family."""
    return {name: export_json_schema(name) for name in sorted(_REGISTRY)}


__all__ = [
    "FamilyDefaults",
    "export_all_schemas",
    "export_json_schema",
    "family",
    "family_for",
    "family_registry",
]
