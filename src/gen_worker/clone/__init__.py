"""Public API for cloning model checkpoints from external upstreams (issue #20).

# Six-method surface

## High-level (one-call — 99% of tenants use these):

- `clone.from_huggingface(ctx, payload)` — full pipeline (download, convert,
  upload) for a HuggingFace repo. Returns the ingested `ConversionOutput`.
- `clone.from_civitai(ctx, payload)` — same for Civitai.

## Intermediate (escape hatches for tenants building custom flows):

- `clone.fetch_huggingface_snapshot(source_ref, revision, dest_dir, ...)` —
  download-only; no upload. Returns local snapshot path.
- `clone.fetch_civitai_file(model_version_id, file_id, dest_dir)` — download
  a specific Civitai file.
- `clone.parse_huggingface_metadata(source_ref)` — fetch + parse repo
  metadata without downloading weights.
- `clone.parse_civitai_metadata(model_version_id)` — same for Civitai.

## Versioning commitment

- High-level APIs (from_huggingface, from_civitai): STABLE. Signature
  changes = major version bump.
- Intermediate APIs: stable, narrower use-cases — changes flagged in
  release notes; minor bump additive, major bump breaking.
- Everything else inside `gen_worker.clone` (including `pipeline.py`,
  `_shared.py`, `_flashpack.py`): LIBRARY-INTERNAL. No stability promise;
  refactor freely.

## Architectural principle

Tenant endpoint code — both `@inference_function` and `@training_function`
— should never touch tensorhub's upload contract (sessions, presigns,
finalize, publish). This module owns all that machinery for clone flows.
Tenant endpoints call `clone.from_huggingface(ctx, payload)` and nothing
more.
"""

from __future__ import annotations

from typing import Any

from .pipeline import run_clone as _run_clone
from .types import CheckpointRef, CivitaiMeta, HFMeta


def from_huggingface(ctx: Any, payload: Any) -> Any:
    """Clone a Hugging Face repo end-to-end: download, convert, upload.

    Returns the ingested ConversionOutput (same shape as the pre-issue-#20
    tenant-code path, just moved into the library).
    """
    return _run_clone(
        ctx,
        provider="huggingface",
        source_ref=getattr(payload, "huggingface_repo", None),
        source_version_id=None,
        source_revision=getattr(payload, "source_revision", None),
        source_metadata_overrides=None,
        destination_repo=getattr(payload, "destination_repo", None),
        destination_repo_tags=getattr(payload, "destination_repo_tags", None),
        target_layout=getattr(payload, "target_layout", None),
        source_layout_preference=getattr(payload, "source_layout_preference", None),
        source_dtype_preference=getattr(payload, "source_dtype_preference", None),
        outputs=getattr(payload, "outputs", None),
        save_formats=getattr(payload, "save_formats", None),
        output_ref=getattr(payload, "output_ref", None),
        quantize_components=getattr(payload, "quantize_components", None),
        auto_publish_public=bool(getattr(payload, "auto_publish_public", False)),
        overwrite_repo=bool(getattr(payload, "overwrite_repo", False)),
    )


def from_civitai(ctx: Any, payload: Any) -> Any:
    """Clone a Civitai model end-to-end: download, convert, upload.

    Supports both URL-based (`source_url`) and ID-based (`civitai_model_version_id` +
    optional `civitai_file_id`) inputs — the pipeline auto-routes.
    """
    # Civitai supports two input modes; keep both pass-throughs. The tenant
    # wrapper used to handle the URL-vs-ID disambiguation; moved here per
    # the architectural principle.
    source_url = str(getattr(payload, "source_url", "") or "").strip()
    version_id = int(getattr(payload, "civitai_model_version_id", 0) or 0)
    file_id = int(getattr(payload, "civitai_file_id", 0) or 0)

    source_metadata_overrides: dict[str, str] = {}
    if file_id:
        source_metadata_overrides["civitai_file_id"] = str(file_id)

    source_ref = source_url if source_url else str(version_id)

    return _run_clone(
        ctx,
        provider="civitai",
        source_ref=source_ref,
        source_version_id=str(version_id) if version_id else None,
        source_revision=None,
        source_metadata_overrides=source_metadata_overrides or None,
        destination_repo=getattr(payload, "destination_repo", None),
        destination_repo_tags=getattr(payload, "destination_repo_tags", None),
        target_layout=getattr(payload, "target_layout", None),
        source_layout_preference=getattr(payload, "source_layout_preference", None),
        source_dtype_preference=getattr(payload, "source_dtype_preference", None),
        outputs=getattr(payload, "outputs", None),
        save_formats=getattr(payload, "save_formats", None),
        output_ref=getattr(payload, "output_ref", None),
        quantize_components=getattr(payload, "quantize_components", None),
        auto_publish_public=bool(getattr(payload, "auto_publish_public", False)),
        overwrite_repo=bool(getattr(payload, "overwrite_repo", False)),
    )


def fetch_huggingface_snapshot(
    source_ref: str,
    revision: str | None = None,
    dest_dir: str | None = None,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Download a HuggingFace snapshot to a local directory.

    Intermediate API: download only; no upload, no lineage. Returns the
    local snapshot path. For a tenant who needs "download then do something
    custom," compose this with `ctx.save_file(...)` manually.
    """
    # Thin wrapper over the library's HF loader; implementation delegates to
    # the pipeline's internal fetch path.
    from .pipeline import _fetch_hf_snapshot_only  # type: ignore[attr-defined]
    return _fetch_hf_snapshot_only(
        source_ref=source_ref,
        revision=revision,
        dest_dir=dest_dir,
        include=include,
        exclude=exclude,
    )


def fetch_civitai_file(model_version_id: int, file_id: int, dest_dir: str) -> str:
    """Download a specific Civitai file to dest_dir. Returns local path."""
    from .pipeline import _fetch_civitai_file_only  # type: ignore[attr-defined]
    return _fetch_civitai_file_only(
        model_version_id=model_version_id,
        file_id=file_id,
        dest_dir=dest_dir,
    )


def parse_huggingface_metadata(source_ref: str, revision: str | None = None) -> dict[str, Any]:
    """Fetch + parse HF repo metadata (library, architectures, etc.) without downloading weights."""
    from .pipeline import _parse_hf_metadata_only  # type: ignore[attr-defined]
    return _parse_hf_metadata_only(source_ref=source_ref, revision=revision)


def parse_civitai_metadata(model_version_id: int) -> dict[str, Any]:
    """Fetch + parse Civitai model-version metadata without downloading."""
    from .pipeline import _parse_civitai_metadata_only  # type: ignore[attr-defined]
    return _parse_civitai_metadata_only(model_version_id=model_version_id)


__all__ = [
    "from_huggingface",
    "from_civitai",
    "fetch_huggingface_snapshot",
    "fetch_civitai_file",
    "parse_huggingface_metadata",
    "parse_civitai_metadata",
    "CheckpointRef",
    "HFMeta",
    "CivitaiMeta",
]
