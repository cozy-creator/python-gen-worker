"""Foreign-catalog enum adapters — injected, never shipped."""

from __future__ import annotations

from threading import RLock
from typing import Mapping, Optional

_lock = RLock()
_maps: dict[str, dict[str, str]] = {}

CIVITAI = "civitai"


def declare_foreign_family_map(
    source: str, mapping: Mapping[str, str], *, replace: bool = False
) -> None:
    """Inject one foreign catalog's ``enum value -> canonical family`` table."""
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
    """Map a Civitai ``baseModel`` enum value to the canonical family."""
    return foreign_to_family(CIVITAI, value)


def reset_foreign_family_maps() -> None:
    """Tests only."""
    with _lock:
        _maps.clear()


__all__ = [
    "CIVITAI",
    "civitai_to_family",
    "declare_foreign_family_map",
    "foreign_to_family",
    "reset_foreign_family_maps",
]
