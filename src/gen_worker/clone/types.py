"""Public data types for the `gen_worker.clone` API (issue #20).

Stable — signature-bound to the public methods. Minor version bump for
additive changes, major for breaking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CheckpointRef:
    """Reference to a finalized checkpoint in tensorhub.

    Returned from `clone.from_huggingface` / `clone.from_civitai`. Opaque
    to tenant code — carry it through to whatever downstream call needs
    the destination_repo + checkpoint_id pair (e.g. an explicit
    ``publish_repo_revision`` retag, or simply reporting back to the
    caller).
    """

    destination_repo: str  # "owner/repo"
    checkpoint_id: str     # content-addressed snapshot digest
    kind: str = ""         # "model" | "lora" | etc.
    dtype: str = ""
    file_layout: str = ""
    file_type: str = ""
    size_bytes: int = 0


@dataclass(frozen=True)
class HFMeta:
    """Parsed HuggingFace repo metadata. Fields optional / empty when absent."""

    source_ref: str
    revision: str = ""
    library: str = ""              # "diffusers" | "transformers" | ""
    model_type: str = ""           # from config.json
    architectures: tuple[str, ...] = field(default_factory=tuple)
    base_model: str = ""           # from adapter_config.json or model card
    target_modules: tuple[str, ...] = field(default_factory=tuple)
    license: str = ""
    file_layout_guess: str = ""    # "diffusers" | "singlefile" | ""
    dtype_hints: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CivitaiMeta:
    """Parsed Civitai model-version metadata."""

    model_version_id: int
    model_id: int = 0
    name: str = ""
    base_model: str = ""
    file_layout_guess: str = ""
    files: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["CheckpointRef", "HFMeta", "CivitaiMeta"]
