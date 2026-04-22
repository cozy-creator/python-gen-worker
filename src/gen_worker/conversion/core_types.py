"""Core types shared by transform tenants, clone_pipeline, and library internals.

Moved from conversion-endpoints/shared.py as part of e2e progress.json #9.
Tenants using the new @training_function contract typically don't touch
these directly — they return ``list[ProducedVariant]`` which the library
adapts. But legacy clone_pipeline + any tenant that needs richer output
metadata (multi-artifact outputs with sharded indices) can import these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import msgspec

from ..api.types import Tensors


class ConversionArtifact(msgspec.Struct):
    """One (relative-path, Tensors) artifact emitted alongside a primary.

    Populated by converters that produce multiple files for a single logical
    output — e.g. when the converted result is re-sharded (.index.json +
    shard files). Manifest builders emit one manifest entry per
    ``ConversionArtifact``.
    """

    rel_name: str
    tensors: Tensors


class ConversionOutput(msgspec.Struct):
    """The legacy transform/clone function return type.

    For new @training_function tenants, prefer returning
    ``list[ProducedVariant]`` and let the library handle upload +
    attributes. ``ConversionOutput`` remains for clone_pipeline + any code
    path that needs a single primary artifact with a typed metadata dict.
    """

    weights: Tensors
    metadata: dict[str, str] = msgspec.field(default_factory=dict)
    # Additional artifacts attached to this output (e.g. shard files when
    # the converted result is re-sharded). Each entry is (rel_name, tensors).
    # ``weights`` is the primary entry-point artifact (.safetensors for
    # unsharded, .index.json for sharded).
    additional_artifacts: list[ConversionArtifact] = msgspec.field(default_factory=list)


@dataclass
class IngestResult:
    """Typed sidecar returned alongside ConversionOutput from ingest paths.

    Replaces the former ``_internal_*``-prefixed stringly-keyed side channel
    that smuggled typed values through a dict[str, str]. Consumers read
    these attributes directly — nothing here crosses the orchestrator→worker
    wire or the tenant→pipeline contract.
    """

    source_repo_dir: str | None = None
    all_weight_files: list[Any] = field(default_factory=list)
    component_groups: dict[str, Any] = field(default_factory=dict)
    all_file_tensors: list[Any] = field(default_factory=list)
    source_dtype_by_component: dict[str, str] = field(default_factory=dict)
    source_dtype_preference: list[str] = field(default_factory=list)


def tensors_with(t: Tensors, **overrides: Any) -> Tensors:
    """Return a copy of ``t`` with field overrides, preserving every other field.

    Uses ``msgspec.structs.replace`` so adding fields to Tensors later doesn't
    silently drop them the way hand-rolled mirror helpers used to.
    """
    return msgspec.structs.replace(t, **overrides)


__all__ = [
    "ConversionArtifact",
    "ConversionOutput",
    "IngestResult",
    "tensors_with",
]
