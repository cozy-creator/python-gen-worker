"""The ONE payload/return schema rendering, shared by both manifest halves."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import msgspec


def type_schema_and_hash(t: type) -> Tuple[Dict[str, Any], str]:
    """JSON schema + its SHA256 over the canonical (sorted, tight) encoding."""
    schema = msgspec.json.schema(t)
    raw = json.dumps(schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return schema, hashlib.sha256(raw).hexdigest()


__all__ = ["type_schema_and_hash"]
