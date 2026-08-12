"""Local-only GGUF snapshot detection (cl#27, GGUF-DESIGN consumption half).

What survives here is the DETECTION half: a materialized snapshot dir that
carries :data:`GGUF_MARKER` is a composed gguf tree, and the loading layer
reads the marker to pick its lane.

pgw#1148 DELETED the SELECTION half — `select_gguf`, `maybe_rebind_gguf`,
`compose_resolved`, `write_marker` and `fetch_gguf_snapshot`. All of it
addressed the artifact as `owner/repo#gguf-<qtype>` and chose among the
resolve's `sibling_flavors` rows. th#1803 deleted both: `repo_tags` is
re-keyed to (repo, tag, checkpoint), the flavor column is gone, the resolve
emits `tag_members` (checkpoint rows) and no longer accepts `?flavor=`, and
§1.32(d) makes the `#` tail a non-address. The code could not have worked
against the current hub — this is a deletion of a broken path, not of a
working feature. Re-addressing the gguf member by CHECKPOINT DIGEST and
re-composing is a BUILD, filed as a pgw#1148 residual.
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
