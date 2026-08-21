"""ProducedFlavor — what a tenant transform hands to publish_flavors, one entry per produced flavor. Tenants declare ONLY tenant-specific attributes: dtype/file_layout/file_type/kind/library are SERVER-INFERRED from the uploaded bytes (tenant-supplied values are ignored), and attribute keys starting with "_" are rejected by the server (reserved for server-computed fields)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import msgspec


_PathField = Annotated[Path, msgspec.Meta(extra_json_schema={"type": "string"})]


class ProducedFlavor(msgspec.Struct):
    """One checkpoint flavor emitted by a transform tenant function."""

    path: _PathField
    attributes: dict = msgspec.field(default_factory=dict)
    extra_files: list[_PathField] = msgspec.field(default_factory=list)


__all__ = ["ProducedFlavor"]
