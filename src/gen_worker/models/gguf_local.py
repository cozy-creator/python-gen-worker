"""Local-only GGUF snapshot detection.

This is the DETECTION half only: a materialized snapshot dir that carries
:data:`GGUF_MARKER` is a composed gguf tree, and the loading layer reads the
marker to pick its lane.

There is deliberately no SELECTION half. Selecting a gguf member by
`owner/repo#gguf-<qtype>` addressed a flavor column that no longer exists —
`repo_tags` is keyed on (repo, tag, checkpoint), the resolve emits `tag_members`
and rejects `?flavor=`, and the `#` tail is not an address. Re-addressing the
gguf member by CHECKPOINT DIGEST and re-composing is a build that has not
happened.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Written into a composed snapshot dir after materialization; the loading
# layer's lane detection reads it (with a structural fallback for a dir that
# lost the marker mid-crash).
GGUF_MARKER = ".cozy-gguf.json"

GGUF_QUALITY_ORDER = (
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q5_1",
    "q5_0",
    "q4_k_m",
    "q4_k_s",
    "q4_1",
    "q4_0",
    "q3_k_m",
    "q3_k_s",
    "q2_k",
)


def gguf_qtype(token: str) -> str:
    """The quant type named by a ``gguf-<qtype>`` token; "" when it names no
    known gguf quantization. The token comes off a MARKER on disk now, never
    off a ref."""
    t = str(token or "").strip().lower()
    if not t.startswith("gguf-"):
        return ""
    qtype = t.removeprefix("gguf-")
    return qtype if qtype in GGUF_QUALITY_ORDER else ""


def read_marker(snap_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(snap_dir) / GGUF_MARKER
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
