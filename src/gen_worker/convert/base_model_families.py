"""Foreign-catalog enum adapters — injected, never shipped.

Civitai's ``baseModel`` enum is a *foreign catalog's* vocabulary: 46 keys
mapping onto 35 canonical slugs, changing whenever civitai adds a base model.
It has nothing to do with what a worker can execute, so pgw#740 (B13) moved the
table out of the SDK: the ingest endpoint (or the hub catalog) injects it with
:func:`declare_foreign_family_map` and the SDK only performs the lookup.

An unmapped value returns ``None`` — the caller stamps nothing rather than
guessing a family from a string it does not recognize.
"""

from __future__ import annotations

from threading import RLock
from typing import Mapping, Optional

_lock = RLock()
_maps: dict[str, dict[str, str]] = {}

CIVITAI = "civitai"


def declare_foreign_family_map(
    source: str, mapping: Mapping[str, str], *, replace: bool = False
) -> None:
    """Inject one foreign catalog's ``enum value -> canonical family`` table.

    Keys are matched case-sensitively after stripping, exactly as the upstream
    API emits them.
    """
    name = str(source or "").strip().lower()
    if not name:
        raise ValueError("foreign family map needs a source name")
    table = {str(k).strip(): str(v).strip() for k, v in mapping.items() if str(k).strip()}
    with _lock:
        if name in _maps and not replace:
            _maps[name].update(table)
        else:
            _maps[name] = table


def foreign_to_family(source: str, value: str) -> Optional[str]:
    """Map a foreign catalog's enum value onto the canonical family, or None."""
    name = str(source or "").strip().lower()
    key = str(value or "").strip()
    if not key:
        return None
    with _lock:
        return _maps.get(name, {}).get(key)


def civitai_to_family(value: str) -> Optional[str]:
    """Map a Civitai ``baseModel`` enum value to the canonical family.
    Returns None for unrecognized inputs, and for every input until the ingest
    endpoint has injected the table."""
    return foreign_to_family(CIVITAI, value)


def registered_foreign_sources() -> tuple[str, ...]:
    with _lock:
        return tuple(sorted(_maps))


def reset_foreign_family_maps() -> None:
    """Tests only."""
    with _lock:
        _maps.clear()


__all__ = [
    "CIVITAI",
    "civitai_to_family",
    "declare_foreign_family_map",
    "foreign_to_family",
    "registered_foreign_sources",
    "reset_foreign_family_maps",
]
