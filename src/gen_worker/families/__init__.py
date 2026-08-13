"""Per-family inference-defaults vocabulary.

This package ships the REGISTRY and nothing else. A family's vocabulary is
declared by the endpoint that owns the family — decorate a
:class:`GenerationDefaults` subclass with ``@family("...")`` anywhere that gets
imported before ``gen-worker families export-schemas --module <endpoint>`` or a
build's discovery walk runs.

``SdxlDefaults`` / ``WanDefaults`` used to ship here. No SDK code
ever consumed them, and a vocabulary in the library is a vocabulary that needs a
wheel release to change — they now live in inference-endpoints/sdxl and
inference-endpoints/wan-2.2 respectively.
"""

from __future__ import annotations

from .base import (
    KIND_CHECKPOINT,
    KIND_LORA,
    GenerationDefaults,
    export_all_schemas,
    export_json_schema,
    family,
    family_for,
    family_registry,
    schema_filename,
)

__all__ = [
    "KIND_CHECKPOINT",
    "KIND_LORA",
    "GenerationDefaults",
    "export_all_schemas",
    "export_json_schema",
    "family",
    "family_for",
    "family_registry",
    "schema_filename",
]
