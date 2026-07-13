"""Per-family inference-defaults vocabulary (pgw#520 / th#767).

Importing this package registers every SHIPPED family (``sdxl``, ...) —
third-party families register themselves the same way, decorating with
``@family("...")`` anywhere that gets imported before ``gen-worker families
export-schemas`` or a build's discovery walk runs.
"""

from __future__ import annotations

from .base import (
    KIND_CHECKPOINT,
    KIND_LORA,
    FamilyDefaults,
    export_all_schemas,
    export_json_schema,
    family,
    family_for,
    family_registry,
    schema_filename,
)
from .sdxl import SdxlDefaults, SdxlLoraDefaults, SdxlScheduler

__all__ = [
    "KIND_CHECKPOINT",
    "KIND_LORA",
    "FamilyDefaults",
    "SdxlDefaults",
    "SdxlLoraDefaults",
    "SdxlScheduler",
    "export_all_schemas",
    "export_json_schema",
    "family",
    "family_for",
    "family_registry",
    "schema_filename",
]
