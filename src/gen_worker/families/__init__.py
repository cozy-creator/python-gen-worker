"""Per-family inference-defaults vocabulary.

This package ships the REGISTRY and nothing else. A family's vocabulary is
declared by the FAMILY that owns it — ``gen_worker.model.ModelSpec(tuned=...)``
registers it through :func:`register_family` — anywhere that gets imported
before ``gen-worker families export-schemas --module <endpoint>`` or a build's
discovery walk runs. The old free-standing ``@family("...")`` class decorator
is gone (pgw#1332): it held the word ``family`` for a defaults vocabulary while
the typed ModelSpec SDK needed it for the family itself.

A defaults vocabulary lives with the ENDPOINT that uses it, never here: a
vocabulary in the library is one that needs a wheel release to change.
"""

from __future__ import annotations

from .base import (
    KIND_CHECKPOINT,
    KIND_LORA,
    GenerationDefaults,
    export_all_schemas,
    export_json_schema,
    family_for,
    family_registry,
    register_family,
    schema_filename,
)

__all__ = [
    "KIND_CHECKPOINT",
    "KIND_LORA",
    "GenerationDefaults",
    "export_all_schemas",
    "export_json_schema",
    "family_for",
    "family_registry",
    "register_family",
    "schema_filename",
]
