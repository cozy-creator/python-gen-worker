"""Registration hooks for ``convert/`` — the third registry, beside ``@family``
(serving defaults) and ``Compile(inputs=...)`` (export inputs). Without it, ``convert/``
family knowledge would have to live in the SDK.

Two registries, one refusal doctrine: an unknown family is refused **by name**,
listing what IS registered. Nothing here ever guesses, and nothing here falls
back to a "close enough" family — that is precisely how an SDXL tree reached the
SD1.5 converter.
"""

from __future__ import annotations

import importlib
from threading import RLock

from .layout_spec import LayoutDeclaration
from .repack_spec import DeclarationError, RepackageFamily

_lock = RLock()
_repack: dict[str, RepackageFamily] = {}
_layouts: dict[str, LayoutDeclaration] = {}


class UnknownFamilyError(LookupError):
    """No declaration is registered for the requested family."""


def register_repackage_family(spec: RepackageFamily, *, replace: bool = False) -> RepackageFamily:
    """Register one family's repackage declaration. Idempotent for an identical spec."""
    with _lock:
        existing = _repack.get(spec.family)
        if existing is not None and existing != spec and not replace:
            raise DeclarationError(
                f"family {spec.family!r} is already registered with a different declaration; "
                "pass replace=True only if you own both"
            )
        for alias in (*spec.aliases, spec.family):
            owner = _alias_owner(alias)
            if owner is not None and owner != spec.family:
                raise DeclarationError(
                    f"alias {alias!r} is already claimed by family {owner!r}"
                )
        _repack[spec.family] = spec
        return spec


def register_layout(decl: LayoutDeclaration, *, replace: bool = False) -> LayoutDeclaration:
    """Register one source-layout detection declaration.

    Keyed by (family, variant, order): one variant legitimately needs more than
    one rung — a broad rule that must sit BELOW a sibling variant's rule, which
    is exactly what the hand-written ladder's line order used to express.
    """
    key = f"{decl.family}/{decl.variant}/{decl.order}"
    with _lock:
        existing = _layouts.get(key)
        if existing is not None and existing != decl and not replace:
            raise DeclarationError(
                f"layout declaration {key!r} is already registered with different content"
            )
        _layouts[key] = decl
        return decl


def registered_repackage_families() -> tuple[str, ...]:
    with _lock:
        return tuple(sorted(_repack))


def registered_layouts() -> tuple[LayoutDeclaration, ...]:
    """Declarations in evaluation order: ``order`` first, then family/variant."""
    with _lock:
        return tuple(sorted(_layouts.values(), key=lambda d: (d.order, d.family, d.variant)))


def normalize_family(name: str | None) -> str:
    """Collapse a family slug (incl. fine-tune lineages) onto its declared family.

    Replaces the SDK's hand-written ``_normalize_family`` ladder: the lineages a
    family absorbs (``sdxl-illustrious`` -> ``sdxl``) are declared by whoever owns
    the family. Returns ``"unknown"`` when nothing claims the slug.
    """
    raw = str(name or "").strip().lower()
    if raw == "":
        return "unknown"
    with _lock:
        if raw in _repack:
            return raw
        owner = _alias_owner(raw)
    return owner or "unknown"


def _alias_owner(slug: str) -> str | None:
    """Caller holds the lock (or accepts a racy read during registration)."""
    for spec in _repack.values():
        if slug == spec.family or slug in spec.aliases:
            return spec.family
    best: tuple[int, str] | None = None
    for spec in _repack.values():
        for prefix in spec.alias_prefixes:
            if slug.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), spec.family)
    return best[1] if best is not None else None


def repackage_family(name: str | None) -> RepackageFamily | None:
    normalized = normalize_family(name)
    with _lock:
        return _repack.get(normalized)


def require_repackage_family(name: str | None) -> RepackageFamily:
    spec = repackage_family(name)
    if spec is None:
        known = ", ".join(registered_repackage_families()) or "<none>"
        raise UnknownFamilyError(
            f"no repackage declaration registered for family {str(name or '').strip()!r}. "
            f"Registered: {known}. Family layout knowledge lives in the endpoint that owns "
            "the family — register it with gen_worker.convert.register_repackage_family() "
            "from your endpoint module (see convert/repack_spec.py for the schema)."
        )
    return spec


def load_declaration_module(dotted: str) -> None:
    """Import a module so its registrations run. Mirrors ``cli/families.py --module``."""
    name = str(dotted or "").strip()
    if not name:
        raise DeclarationError("empty declaration module")
    importlib.import_module(name)


def reset_registries() -> None:
    """Drop every registration. Tests only."""
    with _lock:
        _repack.clear()
        _layouts.clear()


__all__ = [
    "UnknownFamilyError",
    "load_declaration_module",
    "normalize_family",
    "register_layout",
    "register_repackage_family",
    "registered_layouts",
    "registered_repackage_families",
    "repackage_family",
    "require_repackage_family",
    "reset_registries",
]
