"""The declaration rows of a baked ``endpoint.lock``.

**ONE block (pgw#1373).** ``entrypoints[]`` is the only shape a release
declares: Paul's SDK hardcut deleted ``@endpoint``/``@job`` and with them the
``functions[]``/``jobs[]`` blocks. So there is no block LIST any more, and
deliberately no ``DECLARATION_BLOCKS`` constant — a name for "the set of
blocks" is a second home for the vocabulary, and that second home is exactly
what went stale twice: ``functions``-only walks went blind on jobs-only images
(pgw#1354) and then on entrypoints-only images (pgw#1395), each time because a
reader and the block list disagreed. One block, one reader, nothing to sync.

Stdlib only, and no ``gen_worker`` imports: the control parent reads this and
must stay a bare interpreter with no path to torch (pgw#763).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

#: The one block, spelled once. Readers ask for rows, not for the name.
DECLARATION_BLOCK = "entrypoints"


def declaration_rows(manifest: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every declaration row in ``manifest``.

    Tolerant of a malformed or absent block — a lock this process cannot read
    must not raise here, because the callers turn "nothing declared" into a
    typed refusal that names the gap.
    """
    if not isinstance(manifest, Mapping):
        return []
    return [
        row for row in (manifest.get(DECLARATION_BLOCK) or ())
        if isinstance(row, dict)
    ]


def declared_row_count(manifest: Optional[Mapping[str, Any]]) -> int:
    """How many declarations this image carries."""
    return len(declaration_rows(manifest))
