"""Shared per-family inference-defaults vocabulary.

The MODEL SET is catalog, not code — but the SHAPE of a family's
inference defaults/constraints (scheduler choices, step counts, guidance
clamps, ...) is still a code-side contract: it is what lets tensorhub
validate REPO METADATA at PUT time (fail at config, not at invoke) and what
lets an endpoint's :class:`~gen_worker.api.slot.Slot` carry a typed
FALLBACK preset for repos that publish no metadata.

A family struct is declared once per architecture and shared by every
endpoint that serves it. **The struct belongs to a family object** — pgw#1332,
Paul's naming ruling — so the family declares it and the family registers it::

    from gen_worker.model import GraphModelSpec, TunedValues

    class SdxlTuned(TunedValues):
        scheduler: Literal["euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"] = "euler_a"
        steps: int = 28
        guidance: float = 6.0
        max_guidance: float | None = None  # a CLAMP, never a wire reshape

    SDXL = GraphModelSpec(name="sdxl", tuned=SdxlTuned, ...)

**The free-standing ``@family("sdxl")`` class decorator is GONE** (pgw#1332).
It owned the word ``family`` while describing only a defaults vocabulary, and
the typed ModelSpec SDK needs that word for the thing that actually is a family —
graph classes, buckets, loop, instances. Resolving the collision in favour of
the family-owned schema means the registrar is now the plain function
:func:`register_family`, which :class:`gen_worker.model.ModelSpec` calls for its
own schemas. Nothing about the REGISTRY changed: same ``(name, kind)`` key,
same idempotent-re-import rule, same exported JSON Schema, same
``gen-worker families export-schemas``. What changed is that a struct can no
longer claim a family name it does not belong to.

**Kind axis.** A family name has (up to) two vocabularies: the CHECKPOINT
recipe (``kind="checkpoint"``, the default) and the
LORA overlay's recipe OPINIONS (``kind="lora"``), a separate typed struct
sharing the same family name — declared as ``ModelSpec(lora_tuned=...)``.

A separate KIND axis rather than a second family namespace (``"sdxl-lora"``):
a LoRA targets the SAME architecture root as its base checkpoint
(``modelfamily.Root`` on the tensorhub side), so keying by ``(family, kind)``
keeps that identity explicit. tensorhub's schema registry mirrors this:
``<root>.schema.json`` (checkpoint) vs ``<root>.lora.schema.json`` (lora) —
see ``export_json_schema``.

A FUNCTION, not a ``class X(GenerationDefaults, family="sdxl")`` class kwarg:
msgspec's ``StructMeta`` does not forward unrecognized class keywords to
``__init_subclass__`` (it raises ``TypeError`` on an unknown kwarg), and mypy
cannot type-check a metaclass computed at runtime (``type(msgspec.Struct)``) as
a base for a wrapping metaclass.

``forbid_unknown_fields=True`` on the base is what makes
:func:`export_json_schema` emit ``additionalProperties: false`` — the
contract tensorhub validates repo metadata against: a closed vocabulary per
family, not an open bag of keys.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import msgspec

# Keyed (family_name, kind) — "checkpoint" | "lora" (see module docstring's
# kind-axis section). Kind is normalized/defaulted to "checkpoint", so a bare
# ``register_family("sdxl", cls)`` registers the checkpoint recipe.
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
    """Base for a per-family inference-defaults/constraints vocabulary.

    Versioned: ``schema_version`` lets tensorhub gate stored repo metadata
    against the version it was validated under when a family's fields
    change shape. Subclasses add their own typed fields (scheduler enum,
    step counts, guidance clamps, prompt dialect strings, ...) and are
    registered by the family that owns them (see the module docstring).

    ``forbid_unknown_fields=True`` is inherited by every family — the
    exported JSON schema is closed (``additionalProperties: false``): a
    contract, not a suggestion.

    **Positional construction:** this base's own ``kw_only=True``
    only marks ITS field (``schema_version``) keyword-only — msgspec's
    ``kw_only`` does not propagate to a subclass's own fields. A preset row
    like ``SdxlDefaults("euler_a", 28, 6.0)`` (declaration order) works
    fine; it is NOT a TypeError. Still, prefer keyword args in your own
    presets — positional order follows field DECLARATION order, and msgspec
    does not type-check plain construction, so a misordered positional call
    silently lands values on the wrong field instead of raising.
    """

    schema_version: int = 1

    @property
    def family(self) -> str:
        """This instance's registered family name (``""`` for a subclass no
        family ever registered — an authoring mistake, not a valid preset)."""
        return str(getattr(type(self), "__gen_worker_family__", "") or "")

    @property
    def kind(self) -> str:
        """This instance's registered kind (``"checkpoint"`` | ``"lora"``);
        ``"checkpoint"`` for a subclass that never got a kind."""
        return str(getattr(type(self), "__gen_worker_kind__", "") or KIND_CHECKPOINT)


def register_family(
    name: str, cls: Type[F], *, kind: str = KIND_CHECKPOINT
) -> Type[F]:
    """Register a :class:`GenerationDefaults` subclass under ``(name, kind)``.

    The key :func:`family_for` / tensorhub's repo-metadata validation / a
    :class:`~gen_worker.api.slot.Slot`'s ``Compile(family=)`` reconciliation
    all look it up by.

    ``kind`` defaults to ``"checkpoint"``. A LoRA overlay's vocabulary
    registers under the SAME family name with ``kind="lora"`` (see the module
    docstring's kind-axis section) — it is a separate struct, not a merge of
    the checkpoint one.

    **Call it through a family**, not directly:
    :class:`gen_worker.model.ModelSpec` registers its own ``tuned`` and
    ``lora_tuned`` schemas from ``__post_init__``. This function is the
    mechanism, and it stays public only because the CLI, the fences and the
    hub-side contract check need a registrar that does not require a whole
    family declaration.
    """
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
        # A RE-IMPORT is not a collision. Registration is an import side
        # effect, so re-importing a declaration module builds a NEW class
        # object for the same declaration — same module, same qualname —
        # and an identity check alone reads that as two families fighting
        # over one name. It is the same one, twice, and the replacement is
        # what the caller asked for.
        #
        # A different module, or a different class in the same module, is
        # still a genuine collision and still refused: that is two
        # declarations claiming one family, which nothing can adjudicate.
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
    """Dedent docstring-derived ``description`` values in place.

    msgspec 0.21 emits the RAW class docstring (leading indentation intact);
    older msgspec dedented it. The exported schema is a stable contract
    (golden-file-tested), so normalize with ``inspect.cleandoc`` — one
    output regardless of msgspec's docstring handling."""
    desc = node.get("description")
    if isinstance(desc, str):
        node["description"] = inspect.cleandoc(desc)


def schema_filename(name: str, *, kind: str = KIND_CHECKPOINT) -> str:
    """The ``<family>[.lora].schema.json`` filename convention a family's
    exported schema is written under — shared by :func:`export_all_schemas`'
    caller (the ``families export-schemas`` CLI) and tensorhub's own
    ``internal/modelfamily/inferencedefaults`` registry loader, which key
    off this SAME convention: ``<root>.schema.json`` for ``checkpoint``,
    ``<root>.lora.schema.json`` for ``lora``. Keep both sides in lockstep."""
    knd = _normalize_kind(kind)
    suffix = "" if knd == KIND_CHECKPOINT else f".{knd}"
    return f"{name}{suffix}.schema.json"


def export_json_schema(name: str, *, kind: str = KIND_CHECKPOINT) -> Dict[str, Any]:
    """Standalone JSON Schema (draft 2020-12) for one registered
    ``(family, kind)`` pair.

    Flattens msgspec's ``{$ref, $defs}`` output into one closed document —
    the shape tensorhub validates repo metadata against at PUT time.
    """
    knd = _normalize_kind(kind)
    cls = family_for(name, kind=knd)
    if cls is None:
        registered = sorted(f"{fam}:{k}" for fam, k in _REGISTRY) or "(none)"
        raise KeyError(f"no family registered as {name!r} kind {knd!r}; registered: {registered}")
    raw = msgspec.json.schema(cls)
    defs = dict(raw.get("$defs") or {})
    body = defs.pop(cls.__name__, None)
    if body is None:
        # No nested defs (a family with no struct-valued fields): the
        # top-level schema IS the body.
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
    # Canonicalize types: msgspec.json.schema() emits Python-native default
    # VALUES straight off the struct (e.g. a tuple-typed field's default
    # stays a `tuple`, not a JSON `array`) — round-trip through JSON so
    # every caller (not just the CLI, which already serializes with
    # json.dumps) sees the same JSON-safe shape this function's docstring
    # promises ("Standalone JSON Schema").
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
