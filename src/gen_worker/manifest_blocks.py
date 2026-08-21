"""The declaration rows of a baked ``endpoint.lock``."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

DECLARATION_BLOCK = "entrypoints"


def declaration_rows(manifest: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every declaration row in ``manifest``."""
    if not isinstance(manifest, Mapping):
        return []
    return [
        row for row in (manifest.get(DECLARATION_BLOCK) or ())
        if isinstance(row, dict)
    ]


def declared_row_count(manifest: Optional[Mapping[str, Any]]) -> int:
    """How many declarations this image carries."""
    return len(declaration_rows(manifest))
