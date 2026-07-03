"""Public API for cloning model checkpoints from external upstreams (issue #20).

Two entry points:

- `clone.from_huggingface(ctx, payload)` — full pipeline (download, convert,
  upload) for a HuggingFace repo. Returns the ingested `ConversionOutput`.
- `clone.from_civitai(ctx, payload)` — same for Civitai.

Everything else inside `gen_worker.clone` (including `pipeline.py`,
`_shared.py`) is LIBRARY-INTERNAL. No stability promise; refactor freely.

## Architectural principle

Tenant endpoint code — both `@inference` and `@conversion`
— should never touch tensorhub's upload contract (sessions, presigns,
finalize, publish). This module owns all that machinery for clone flows.
Tenant endpoints call `clone.from_huggingface(ctx, payload)` and nothing
more.
"""

from __future__ import annotations

from typing import Any

from .pipeline import run_clone as _run_clone
from .size_walk import compute_size_facts
from .types import CheckpointRef, CivitaiMeta, HFMeta


def from_huggingface(ctx: Any, payload: Any, *, hf_token: str | None = None) -> Any:
    """Clone a Hugging Face repo end-to-end: download, convert, upload.

    Returns the ingested ConversionOutput (same shape as the pre-issue-#20
    tenant-code path, just moved into the library).

    `hf_token` is the credential used for the source download. The tenant
    endpoint resolves precedence (per-request token > endpoint env) and passes
    the effective value here; when None the library falls back to
    `ctx.hf_token`. Per-request tokens let an invoker pull a gated/private repo
    their own account can access. Never logged.
    """
    return _run_clone(
        ctx,
        provider="huggingface",
        source_ref=getattr(payload, "huggingface_repo", None),
        source_version_id=None,
        source_revision=getattr(payload, "source_revision", None),
        source_metadata_overrides=None,
        hf_token=hf_token,
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
        gguf_quant=getattr(payload, "gguf_quant", None),
    )


def from_civitai(ctx: Any, payload: Any, *, civitai_api_key: str | None = None) -> Any:
    """Clone a Civitai model end-to-end: download, convert, upload.

    Accepts a `civitai_model_version_id` plus optional `civitai_file_id`.
    Arbitrary model-weight URLs are not a supported provider surface.

    `civitai_api_key` is the credential used for the source download. The tenant
    endpoint resolves precedence (per-request key > endpoint env) and passes the
    effective value here. Per-request keys let an invoker pull a gated/early-
    access model their own account can access. Never logged.
    """
    source_url = str(getattr(payload, "source_url", "") or "").strip()
    if source_url:
        raise ValueError("civitai source_url is not supported; use civitai_model_version_id")
    version_id = int(getattr(payload, "civitai_model_version_id", 0) or 0)
    if version_id <= 0:
        raise ValueError("civitai_model_version_id is required")
    file_id = int(getattr(payload, "civitai_file_id", 0) or 0)

    source_metadata_overrides: dict[str, str] = {}
    if file_id:
        source_metadata_overrides["civitai_file_id"] = str(file_id)

    return _run_clone(
        ctx,
        provider="civitai",
        source_ref=str(version_id),
        source_version_id=str(version_id) if version_id else None,
        source_revision=None,
        source_metadata_overrides=source_metadata_overrides or None,
        civitai_api_key=civitai_api_key,
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
        gguf_quant=None,  # not relevant for civitai (no GGUF on that platform)
    )


__all__ = [
    "from_huggingface",
    "from_civitai",
    "compute_size_facts",
    "CheckpointRef",
    "HFMeta",
    "CivitaiMeta",
]
