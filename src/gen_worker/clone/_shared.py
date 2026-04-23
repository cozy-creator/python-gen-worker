"""Clone-pipeline-specific helpers.

The transform-endpoint tenants (cast_dtype / torchao_quantization / ...) use
gen_worker.conversion (library-side). This file is the narrower slice used
only by clone_huggingface / clone_civitai / clone_pipeline for the
external-URL ingest path.

Legacy progress-tracking machinery (_emit_upload_progress + 300 LOC of
stream-mode detection, retry env defaults, chunked-upload orchestration)
was removed: gen-worker's ctx.save_checkpoint already emits progress events
and handles retries itself. See e2e progress.json #9.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from gen_worker import RequestContext, Tensors
from gen_worker.conversion.core_types import (
    ConversionArtifact,
    ConversionOutput,
    IngestResult,
    tensors_with,
)
from gen_worker.conversion.ingest import (
    CivitaiResolvedIdentity,
    download_civitai_model_version_files,
    download_huggingface_repo_files,
    source_url_to_cas,
)
from gen_worker.conversion.layout import (
    detect_huggingface_source_layout,
    infer_model_family_from_hint,
)


def require_local_weights(weights: Tensors) -> Path:
    """Return the local path on disk for ``weights``, raising if missing."""
    local = str(weights.local_path or "").strip()
    if local == "":
        raise ValueError("weights.local_path is required")
    path = Path(local)
    if not path.exists():
        raise ValueError(f"weights.local_path does not exist: {path}")
    return path


def default_output_ref(ctx: RequestContext, stem: str, ext: str = ".safetensors") -> str:
    """Job-scoped output ref: ``jobs/<request_id>/outputs/<stem><ext>``."""
    suffix = ext if ext.startswith(".") else f".{ext}"
    return f"jobs/{ctx.request_id}/outputs/{stem}{suffix}"


def _infer_source_ext_from_url(source_url: str) -> str:
    try:
        parsed = urlparse(str(source_url or "").strip())
        ext = Path(parsed.path).suffix.lower()
    except Exception:
        ext = ""
    if ext in {".safetensors", ".ckpt", ".pt", ".bin", ".flashpack"}:
        return ext
    return ".bin"


def save_checkpoint_chunked(
    ctx: RequestContext,
    *,
    input_path: Path,
    ref: str,
    format: str,
) -> Tensors:
    """Upload ``input_path`` as a checkpoint under ``ref``.

    Thin wrapper over ``ctx.save_checkpoint`` — kept as a stable name for
    clone_pipeline callers. The library itself handles chunking, streaming,
    retries, and progress emission.
    """
    return ctx.save_checkpoint(ref, str(input_path), format=format)




def ingest_from_url(
    ctx: RequestContext,
    *,
    source_url: str,
    output_ref: Optional[str],
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> ConversionOutput:
    src = str(source_url or "").strip()
    if not (src.startswith("http://") or src.startswith("https://")):
        raise ValueError("ingest source must be an http(s) URL")

    td = tempfile.mkdtemp(prefix=f"conv-{ctx.request_id}-ingest-")
    ext = _infer_source_ext_from_url(src)
    raw = Path(td) / f"ingested{ext}"
    info = source_url_to_cas(src, raw, progress_callback=progress_callback)
    ref = str(output_ref or "").strip() or default_output_ref(ctx, "ingested-source", ext=ext)
    saved = save_checkpoint_chunked(ctx, input_path=raw, ref=ref, format=ext.lstrip(".") or "bin")
    if str(saved.local_path or "").strip() == "":
        saved = tensors_with(saved, local_path=str(raw))
    return ConversionOutput(
        weights=saved,
        metadata={
            "source_url": src,
            "source_layout": "singlefile",
            "model_family": "unknown",
            "sha256": str(info.get("sha256") or ""),
            "size_bytes": str(info.get("size_bytes") or ""),
        },
    )


def ingest_from_source(
    ctx: RequestContext,
    *,
    provider: str,
    source_ref: str,
    source_revision: str | None,
    source_layout_preference: str = "auto",
    source_dtype_preference: list[str] | None = None,
    source_expected_sha256: str | None = None,
    source_expected_size_bytes: int | None = None,
    civitai_model_version_id: int | None = None,
    civitai_file_id: int | None = None,
    resolved_civitai_identity: CivitaiResolvedIdentity | None = None,
    output_ref: Optional[str],
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> tuple[ConversionOutput, IngestResult]:
    src = str(source_ref or "").strip()
    provider_norm = str(provider or "").strip().lower()
    if src == "":
        raise ValueError("source_ref is required")

    td = tempfile.mkdtemp(prefix=f"conv-{ctx.request_id}-ingest-")
    raw = Path(td) / "ingested.bin"
    if src.startswith("http://") or src.startswith("https://"):
        source_ext = _infer_source_ext_from_url(src)
        raw = Path(td) / f"ingested{source_ext}"
        info = source_url_to_cas(
            src,
            raw,
            progress_callback=progress_callback,
            source_provider=provider_norm,
            expected_size_bytes=source_expected_size_bytes,
            expected_sha256=str(source_expected_sha256 or ""),
        )
        guessed_family = infer_model_family_from_hint(src)
        source_meta: dict[str, str] = {
            "source_url": src,
            "source_layout": "singlefile",
            "model_family": guessed_family,
        }
        ref = str(output_ref or "").strip() or default_output_ref(ctx, "ingested-source", ext=source_ext)
        saved = save_checkpoint_chunked(
            ctx,
            input_path=raw,
            ref=ref,
            format=source_ext.lstrip(".") or "bin",
        )
        if str(saved.local_path or "").strip() == "":
            saved = tensors_with(saved, local_path=str(raw))
        source_meta["sha256"] = str(info.get("sha256") or "")
        source_meta["size_bytes"] = str(info.get("size_bytes") or "")
        return ConversionOutput(weights=saved, metadata=source_meta), IngestResult()
    elif provider_norm == "huggingface":
        repo_dir = Path(td) / "source-repo"
        info = download_huggingface_repo_files(
            src,
            repo_dir,
            source_revision=source_revision,
            source_layout_preference=source_layout_preference,
            source_dtype_preference=list(source_dtype_preference or []),
            progress_callback=progress_callback,
        )

        files = [dict(item) for item in list(info.get("files") or []) if isinstance(item, dict)]
        if not files:
            raise ValueError("huggingface repo has no files")

        # File extensions to skip during HF ingest — not needed for inference.
        _SKIP_EXTS = {
            ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp",  # images
            ".onnx", ".pb", ".tflite",  # non-pytorch formats
            ".msgpack",  # JAX/Flax format
        }
        _SKIP_NAMES = {".gitattributes", ".gitignore"}

        refs: list[str] = []
        primary_saved: Tensors | None = None
        primary_saved_size: int = 0
        primary_weight_exts = (".safetensors", ".ckpt", ".pt", ".bin")
        all_weight_files: list[tuple[Tensors, str, int]] = []  # (saved, rel_path, size_bytes)
        # Every uploaded file (weights + configs + tokenizers + schedulers)
        # captured as Tensors so finalize_clone can list them all in the
        # snapshot manifest. Without this, diffusers inference fails with
        # `model_index.json not found` because only weight files were
        # registered for download.
        all_file_tensors: list[tuple[Tensors, str, int]] = []
        # component_name -> {is_sharded, index_tensors, index_rel_path, expected_shards, shards}
        component_groups: dict[str, dict[str, object]] = {}

        def _component_of(rel_path: str) -> str:
            parts = [p for p in rel_path.split("/") if p]
            return parts[0].lower() if len(parts) > 1 else ""

        def _ensure_group(component: str) -> dict[str, object]:
            if component not in component_groups:
                component_groups[component] = {
                    "is_sharded": False,
                    "index_tensors": None,
                    "index_rel_path": None,
                    "expected_shards": [],
                    "shards": [],
                }
            return component_groups[component]

        # Collect non-weight files (configs, tokenizers, scheduler, model_index,
        # README, LICENSE, etc.) as a separate list so we can upload them
        # unconditionally and keep weight files local-only until _finalize_clone
        # decides which variants want them.
        non_weight_items: list[dict[str, object]] = []
        weight_items: list[dict[str, object]] = []
        for item in files:
            rel_path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
            local_path = str(item.get("local_path") or "").strip()
            if rel_path == "" or local_path == "":
                continue
            if ".." in rel_path.split("/"):
                continue
            filename = Path(rel_path).name.lower()
            ext = Path(rel_path).suffix.lower()
            if ext in _SKIP_EXTS or filename in _SKIP_NAMES:
                continue
            row = {
                "rel_path": rel_path,
                "local_path": local_path,
                "filename": filename,
                "ext": ext,
            }
            if rel_path.lower().endswith(primary_weight_exts):
                weight_items.append(row)
            else:
                non_weight_items.append(row)

        # Upload every non-weight file to repo-CAS immediately. Every output
        # variant needs these (model_index.json, tokenizer configs, etc.) to
        # reconstruct a working diffusers pipeline regardless of which dtype
        # was requested.
        for row in non_weight_items:
            rel_path = str(row["rel_path"])
            local_path = str(row["local_path"])
            ref = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}"
            saved = save_checkpoint_chunked(
                ctx,
                input_path=Path(local_path),
                ref=ref,
                format=Path(rel_path).suffix.lstrip(".") or "bin",
            )
            if str(saved.local_path or "").strip() == "":
                saved = tensors_with(saved, local_path=local_path)
            refs.append(str(saved.ref or ref))
            all_file_tensors.append((saved, rel_path, int(saved.size_bytes or 0)))

            filename = str(row["filename"])
            component = _component_of(rel_path)
            # Track safetensors index sidecars — they carry the weight_map for sharded components.
            if filename.endswith(".safetensors.index.json") and component != "":
                group = _ensure_group(component)
                group["is_sharded"] = True
                group["index_tensors"] = saved
                group["index_rel_path"] = rel_path
                try:
                    idx_data = json.loads(Path(local_path).read_text("utf-8"))
                    weight_map = idx_data.get("weight_map") or {}
                    expected = sorted({str(v) for v in weight_map.values() if v})
                except Exception:
                    expected = []
                group["expected_shards"] = expected

        # For weight files, synthesise "local-only" Tensors (with local_path +
        # size_bytes but no blob_digest yet). `_finalize_clone` calls
        # `save_checkpoint_chunked` on the subset actually required by the
        # requested OutputSpec list, so weight bytes that nobody references
        # never hit repo-CAS.
        for row in weight_items:
            rel_path = str(row["rel_path"])
            local_path = str(row["local_path"])
            file_size = int(Path(local_path).stat().st_size)
            ref = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}"
            saved = Tensors(
                ref=ref,
                owner=None,
                local_path=local_path,
                format=Path(rel_path).suffix.lstrip(".") or "bin",
                size_bytes=file_size,
                sha256=None,
                blake3=None,
                blob_digest=None,
                blob_domain=None,
                blob_path=None,
                snapshot_digest=None,
                download_token=None,
            )
            all_file_tensors.append((saved, rel_path, file_size))

            component = _component_of(rel_path)
            all_weight_files.append((saved, rel_path, file_size))
            if component != "":
                _ensure_group(component)["shards"].append((saved, rel_path, file_size))
            if primary_saved is None or file_size > primary_saved_size:
                primary_saved = saved
                primary_saved_size = file_size

        # If a repo has zero weight files (shouldn't happen for diffusers/transformers
        # mirrors, but guard anyway), fall back to the first non-weight file as primary.
        if primary_saved is None:
            for saved_tuple in all_file_tensors:
                primary_saved = saved_tuple[0]
                break

        if primary_saved is None:
            raise ValueError("huggingface repo ingestion produced no uploadable files")

        manifest_lines = [
            f"{str(item.get('path') or '').strip()}:{int(item.get('size_bytes') or 0)}"
            for item in files
            if str(item.get("path") or "").strip() != ""
        ]
        manifest_hash = hashlib.sha256("\n".join(manifest_lines).encode("utf-8")).hexdigest()

        layout_info = detect_huggingface_source_layout(
            repo_dir=repo_dir,
            files=[str(item.get("path") or "") for item in files],
        )

        source_meta = {
            "source_repo": str(info.get("source_repo") or src),
            "source_revision": str(info.get("source_revision") or str(source_revision or "").strip()),
            "source_layout_preference": str(source_layout_preference or "auto").strip().lower() or "auto",
            "selected_source_layout": str(info.get("selected_source_layout") or str(layout_info.source_layout)),
            "selected_aio_path": str(info.get("selected_aio_path") or ""),
            "source_layout_selection_reason": str(info.get("source_layout_selection_reason") or ""),
            "source_layout": str(layout_info.source_layout),
            "model_family": str(layout_info.model_family),
            "model_family_variant": str(layout_info.model_family_variant),
            "source_layout_detection_reason": str(
                info.get("source_layout_detection_reason") or str(layout_info.detection_reason)
            ),
            "source_file_count": str(len(files)),
            "source_total_bytes": str(int(info.get("total_bytes") or 0)),
            "ingested_file_count": str(len(files)),
            "ingested_total_bytes": str(int(info.get("total_bytes") or 0)),
            "source_manifest_sha256": manifest_hash,
            "source_artifact_refs": ";".join(refs),
        }
        ingest_result = IngestResult(
            source_repo_dir=str(repo_dir),
            all_weight_files=all_weight_files,
            component_groups=component_groups,
            all_file_tensors=all_file_tensors,
            source_dtype_by_component=dict(info.get("source_dtype_by_component") or {}),
            source_dtype_preference=list(info.get("source_dtype_preference") or []),
        )
        return ConversionOutput(weights=primary_saved, metadata=source_meta), ingest_result
    elif provider_norm == "civitai" and int(civitai_model_version_id or 0) > 0:
        repo_dir = Path(td) / "source-repo"
        info = download_civitai_model_version_files(
            int(civitai_model_version_id or 0),
            repo_dir,
            civitai_file_id=(int(civitai_file_id or 0) or None),
            progress_callback=progress_callback,
            resolved_identity=resolved_civitai_identity,
        )

        files = [dict(item) for item in list(info.get("files") or []) if isinstance(item, dict)]
        if not files:
            raise ValueError("civitai_no_supported_files")

        refs: list[str] = []
        primary_saved: Tensors | None = None
        primary_weight_exts = (".safetensors", ".ckpt", ".pt", ".bin")
        selected_manifest: list[dict[str, object]] = []

        for item in files:
            rel_path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
            local_path = str(item.get("local_path") or "").strip()
            if rel_path == "" or local_path == "":
                continue
            if ".." in rel_path.split("/"):
                continue
            ref = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}"
            saved = save_checkpoint_chunked(
                ctx,
                input_path=Path(local_path),
                ref=ref,
                format=Path(rel_path).suffix.lstrip(".") or "bin",
            )
            if str(saved.local_path or "").strip() == "":
                saved = tensors_with(saved, local_path=local_path)
            refs.append(str(saved.ref or ref))

            if primary_saved is None:
                primary_saved = saved
            if rel_path.lower().endswith(primary_weight_exts):
                if primary_saved is None or not str(primary_saved.ref or "").lower().endswith(primary_weight_exts):
                    primary_saved = saved

            selected_manifest.append(
                {
                    "path": rel_path,
                    "file_id": int(item.get("file_id") or 0),
                    "name": str(item.get("file_name") or ""),
                    "size_bytes": int(item.get("size_bytes") or 0),
                    "sha256": str(item.get("sha256") or ""),
                    "fingerprint": str(item.get("file_fingerprint") or ""),
                    "primary": bool(item.get("primary")),
                    "hashes": dict(item.get("hashes") or {}),
                }
            )

        if primary_saved is None:
            raise ValueError("civitai_no_supported_files")

        manifest_lines = [
            f"{str(item.get('path') or '').strip()}:{int(item.get('size_bytes') or 0)}:{str(item.get('file_fingerprint') or '')}"
            for item in files
            if str(item.get("path") or "").strip() != ""
        ]
        manifest_hash = hashlib.sha256("\n".join(manifest_lines).encode("utf-8")).hexdigest()

        layout_info = detect_huggingface_source_layout(
            repo_dir=repo_dir,
            files=[str(item.get("path") or "") for item in files],
        )
        model_family = str(layout_info.model_family or "").strip().lower()
        if model_family == "" or model_family == "unknown":
            hinted = str(info.get("model_family") or "").strip().lower()
            if hinted != "":
                model_family = hinted
        if model_family == "":
            model_family = "unknown"
        model_family_variant = str(layout_info.model_family_variant or "").strip().lower()
        if model_family_variant == "" or model_family_variant == "unknown":
            hinted_variant = str(info.get("model_family_variant") or "").strip().lower()
            if hinted_variant != "":
                model_family_variant = hinted_variant
        if model_family_variant == "":
            model_family_variant = "unknown"

        source_meta = {
            "source_kind": "civitai_model_version",
            "source_repo": str(info.get("source_ref") or src),
            "source_revision": str(info.get("source_revision") or str(source_revision or "").strip()),
            "source_layout": str(layout_info.source_layout),
            "model_family": model_family,
            "model_family_variant": model_family_variant,
            "source_layout_detection_reason": str(layout_info.detection_reason),
            "source_file_count": str(len(files)),
            "source_total_bytes": str(int(info.get("total_bytes") or 0)),
            "source_manifest_sha256": str(info.get("source_manifest_sha256") or manifest_hash),
            "source_artifact_refs": ";".join(refs),
            "civitai_model_version_id": str(int(info.get("model_version_id") or 0)),
            "civitai_model_id": str(int(info.get("model_id") or 0)),
            "civitai_base_model": str(info.get("base_model") or ""),
            "civitai_base_model_type": str(info.get("base_model_type") or ""),
            "civitai_air": str(info.get("air") or ""),
            "civitai_pipeline_hint": str(info.get("pipeline_hint") or ""),
            "civitai_model_family_variant": str(info.get("model_family_variant") or model_family_variant),
            "civitai_file_fingerprints_json": json.dumps(
                dict(info.get("file_fingerprints") or {}),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "civitai_selected_files_json": json.dumps(
                selected_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        selected_id = int(info.get("selected_file_id") or 0)
        if selected_id > 0:
            source_meta["civitai_file_id"] = str(selected_id)
        return ConversionOutput(weights=primary_saved, metadata=source_meta), IngestResult(source_repo_dir=str(repo_dir))
    else:
        raise ValueError("ingest source must be an http(s) URL")


__all__ = [
    # Re-exports from gen_worker.conversion.core_types
    "ConversionArtifact",
    "ConversionOutput",
    "IngestResult",
    "tensors_with",
    # Clone-pipeline-local helpers
    "require_local_weights",
    "default_output_ref",
    "save_checkpoint_chunked",
    "ingest_from_url",
    "ingest_from_source",
]
