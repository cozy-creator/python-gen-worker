"""Clone-pipeline-specific helpers.

Transform endpoints use gen_worker.conversion on the library side. This file is
the narrower slice used only by clone_huggingface / clone_civitai /
clone_pipeline for provider-specific ingest paths.

Legacy progress-tracking machinery (_emit_upload_progress + 300 LOC of
stream-mode detection, retry env defaults, chunked-upload orchestration)
was removed: gen-worker's ctx.save_checkpoint already emits progress events
and handles retries itself.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from gen_worker import RequestContext
from gen_worker.api.types import Tensors
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
)
from gen_worker.conversion.layout import (
    detect_huggingface_source_layout,
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
    gguf_quant: str | None = None,
    # When set with len > 1, the HF branch runs
    # the per-strategy selector once per concrete dtype and emits
    # `IngestResult.classifier_attrs_per_checkpoint` so finalize can
    # publish N checkpoints under one tag with distinct dtype attrs.
    # Single-element / None falls back to the existing single-checkpoint
    # path. Civitai / URL providers ignore this (multi-dtype only makes
    # sense for HF source repos that ship variants side-by-side).
    dtype_outputs: list[str] | None = None,
    # Per-invocation download credentials. When set, the source download
    # authenticates with these instead of (HF) `ctx.hf_token`. The caller
    # (run_clone) has already applied per-request > env precedence. Empty /
    # None falls back to `ctx.hf_token` for HF and unauthenticated for civitai.
    # Never logged.
    hf_token: str | None = None,
    civitai_api_key: str | None = None,
) -> tuple[ConversionOutput, IngestResult]:
    src = str(source_ref or "").strip()
    provider_norm = str(provider or "").strip().lower()
    if src == "":
        raise ValueError("source_ref is required")

    effective_hf_token = str(hf_token or "").strip() or str(getattr(ctx, "hf_token", "") or "").strip()
    effective_civitai_api_key = str(civitai_api_key or "").strip()

    td = tempfile.mkdtemp(prefix=f"conv-{ctx.request_id}-ingest-")
    if src.startswith("http://") or src.startswith("https://"):
        raise ValueError("arbitrary URL model sources are not supported")
    elif provider_norm == "huggingface":
        repo_dir = Path(td) / "source-repo"
        info = download_huggingface_repo_files(
            src,
            repo_dir,
            source_revision=source_revision,
            source_layout_preference=source_layout_preference,
            source_dtype_preference=list(source_dtype_preference or []),
            gguf_quant=gguf_quant,
            progress_callback=progress_callback,
            dtype_outputs=list(dtype_outputs or []),
            hf_token=effective_hf_token,
        )

        files = [dict(item) for item in list(info.get("files") or []) if isinstance(item, dict)]
        if not files:
            raise ValueError("huggingface repo has no files")

        # The classifier in download_huggingface_repo_files already filtered
        # pickle / ONNX / OpenVINO / Flax / demo media at selection time. We
        # only need a tiny `.gitattributes` / `.gitignore` skip here for HF's
        # repo plumbing files that the classifier may have allowed through.
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
            if filename in _SKIP_NAMES:
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
        #
        # Issue #269/#13/#19: fan out across the fixed file-level upload
        # pool so the worker pipelines disk read + hash + multipart PUT
        # across files without unbounded R2 PUT fan-out.
        from gen_worker.request_context._concurrent_upload import parallel_map_uploads

        def _upload_non_weight(row: dict[str, object]) -> tuple[dict[str, object], object]:
            rel_path_local = str(row["rel_path"])
            local_path_local = str(row["local_path"])
            ref_local = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path_local}"
            saved_local = ctx.save_checkpoint(
                ref_local,
                local_path_local,
                format=Path(rel_path_local).suffix.lstrip(".") or "bin",
            )
            if str(saved_local.local_path or "").strip() == "":
                saved_local = tensors_with(saved_local, local_path=local_path_local)
            return row, saved_local

        for row, saved in parallel_map_uploads(
            non_weight_items, _upload_non_weight, label="clone-nonweight"
        ):
            rel_path = str(row["rel_path"])
            local_path = str(row["local_path"])
            ref = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}"
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
        # `ctx.save_checkpoint` on the subset actually required by the
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

        classifier_attrs = {
            str(k): str(v)
            for k, v in (info.get("attrs") or {}).items()
            if v is not None and str(v) != ""
        }
        runtime_library = str(info.get("runtime_library") or classifier_attrs.get("runtime_library") or "").strip()
        library_name = {
            "diffusers-single-file": "diffusers",
            "diffusers-lora": "diffusers",
            "llama-cpp": "llama.cpp",
        }.get(runtime_library, runtime_library)
        repo_type = "adapter" if runtime_library in {"peft", "diffusers-lora"} else "model"
        class_name = str(
            classifier_attrs.get("pipeline_class")
            or classifier_attrs.get("architecture")
            or ""
        ).strip()
        model_family = str(layout_info.model_family or "").strip()
        if repo_type == "adapter":
            model_family = str(classifier_attrs.get("base_model_family") or model_family).strip()
        if hasattr(ctx, "set_repo_spec"):
            ctx.set_repo_spec(
                kind=repo_type,
                library_name=library_name,
                model_family=model_family,
                class_name=class_name,
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
            # Classifier-derived metadata. These are
            # also available on IngestResult.classifier_attrs for finalize to
            # thread onto each ProducedFlavor.attributes.
            "ingest_strategy": str(info.get("strategy") or ""),
            "runtime_library": str(info.get("runtime_library") or ""),
            "subtype": str(info.get("subtype") or ""),
            "selected_count": str(int(info.get("selected_count") or 0)),
            "skipped_count": str(int(info.get("skipped_count") or 0)),
            "selected_bytes": str(int(info.get("selected_bytes") or 0)),
            "skipped_bytes": str(int(info.get("skipped_bytes") or 0)),
            "pickle_files_refused_count": str(len(info.get("pickle_files_refused") or [])),
        }
        # Fold classifier attrs into source_meta with `classifier_` prefix so
        # they don't collide with the legacy keys above.
        for k, v in classifier_attrs.items():
            source_meta.setdefault(f"classifier_{k}", v)
        # Surface per-checkpoint attrs from the
        # downloader's `selections` field so the finalize layer can
        # publish N checkpoints (one per resolved concrete dtype) under
        # the same destination tag.
        per_checkpoint = []
        for entry in (info.get("selections") or []):
            if not isinstance(entry, dict):
                continue
            attrs = entry.get("attrs") or {}
            if not isinstance(attrs, dict):
                continue
            per_checkpoint.append({str(k): str(v) for k, v in attrs.items()})

        ingest_result = IngestResult(
            source_repo_dir=str(repo_dir),
            all_weight_files=all_weight_files,
            component_groups=component_groups,
            all_file_tensors=all_file_tensors,
            source_dtype_by_component=dict(info.get("source_dtype_by_component") or {}),
            source_dtype_preference=list(info.get("source_dtype_preference") or []),
            classifier_attrs=classifier_attrs,
            classifier_attrs_per_checkpoint=per_checkpoint,
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
            civitai_api_key=effective_civitai_api_key,
        )

        files = [dict(item) for item in list(info.get("files") or []) if isinstance(item, dict)]
        if not files:
            raise ValueError("civitai_no_supported_files")

        refs: list[str] = []
        primary_saved: Tensors | None = None
        primary_weight_exts = (".safetensors", ".ckpt", ".pt", ".bin")
        selected_manifest: list[dict[str, object]] = []

        # Filter once so the upload fan-out + the manifest pass see the
        # same list (and parallel_map_uploads preserves index alignment).
        usable_items: list[dict[str, object]] = []
        for item in files:
            rel_path = str(item.get("path") or "").strip().replace("\\", "/").lstrip("/")
            local_path = str(item.get("local_path") or "").strip()
            if rel_path == "" or local_path == "":
                continue
            if ".." in rel_path.split("/"):
                continue
            item = dict(item)
            item["_rel_path"] = rel_path
            item["_local_path"] = local_path
            usable_items.append(item)

        # Issue #269/#13/#19: parallelize per-file upload across the
        # fixed file pool. Each future owns one file hash→PUT→complete
        # cycle; results returned in input order.
        from gen_worker.request_context._concurrent_upload import parallel_map_uploads

        def _upload_civitai(item: dict[str, object]) -> "Tensors":
            rel_path_local = str(item["_rel_path"])
            local_path_local = str(item["_local_path"])
            ref_local = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path_local}"
            saved_local = ctx.save_checkpoint(
                ref_local,
                local_path_local,
                format=Path(rel_path_local).suffix.lstrip(".") or "bin",
            )
            if str(saved_local.local_path or "").strip() == "":
                saved_local = tensors_with(saved_local, local_path=local_path_local)
            return saved_local

        uploaded = parallel_map_uploads(usable_items, _upload_civitai, label="civitai")

        for item, saved in zip(usable_items, uploaded):
            rel_path = str(item["_rel_path"])
            ref = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}"
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

        # Populate the structured base-model
        # lineage attributes from the Civitai API response. The
        # `baseModel` enum is the family signal; the `air` URN is the
        # specific reference within civitai's universe (not directly
        # mappable to an HF repo, but uniquely identifies the upload).
        civitai_base_model_raw = str(info.get("base_model") or "").strip()
        try:
            from gen_worker.conversion.base_model_families import civitai_to_family
            base_family = civitai_to_family(civitai_base_model_raw) or ""
        except Exception:
            base_family = ""
        # `base_model_specific_hint` is set from kohya `ss_sd_model_name`
        # when the safetensors header carries it (LoRAs trained on a
        # specific community fine-tune like Juggernaut/RealisticVision).
        # `info.kohya_metadata` may carry it from the civitai download
        # path; if not, leave empty.
        kohya_meta = info.get("kohya_metadata") if isinstance(info.get("kohya_metadata"), dict) else {}
        ss_sd_model_name = ""
        ss_sd_model_hash = ""
        if isinstance(kohya_meta, dict):
            ss_sd_model_name = str(kohya_meta.get("ss_sd_model_name") or "").strip()
            ss_sd_model_hash = str(kohya_meta.get("ss_sd_model_hash") or "").strip()

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
            "civitai_base_model": civitai_base_model_raw,
            "civitai_base_model_type": str(info.get("base_model_type") or ""),
            "civitai_air": str(info.get("air") or ""),
            "civitai_pipeline_hint": str(info.get("pipeline_hint") or ""),
            "civitai_model_family_variant": str(info.get("model_family_variant") or model_family_variant),
            # Structured lineage attrs (see
            # gen_worker/conversion/base_model_families.py for the family
            # enum). These are mirrored into IngestResult.classifier_attrs
            # below so the destination checkpoint catalog row carries the
            # right `base_model_family` / `base_model_civitai_baseModel`
            # / `lineage_source` values.
            "base_model_family": base_family,
            "base_model_civitai_baseModel": civitai_base_model_raw,
            "base_model_specific_hint": ss_sd_model_name,
            "lineage_source": "civitai_baseModel" if civitai_base_model_raw else "unknown",
            "kohya_ss_sd_model_hash": ss_sd_model_hash,
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

        # Thread the structured base-model
        # lineage onto IngestResult.classifier_attrs so the finalize
        # path stamps the same fields onto the destination checkpoint
        # the HF clone path does (`base_model_family`,
        # `base_model_civitai_baseModel`, `lineage_source`,
        # `base_model_specific_hint`). The runtime_library hint is
        # the civitai pipeline_hint when present, else
        # diffusers-single-file (full-checkpoint civitai uploads) /
        # diffusers-lora (kohya-style LoRA, detected by metadata).
        runtime_library_hint = ""
        pipeline_hint = str(info.get("pipeline_hint") or "").strip()
        if pipeline_hint:
            runtime_library_hint = pipeline_hint
        elif kohya_meta:
            runtime_library_hint = "diffusers-lora"
        else:
            runtime_library_hint = "diffusers-single-file"
        civitai_classifier_attrs = {
            "runtime_library": runtime_library_hint,
            "base_model_family": base_family,
            "base_model_civitai_baseModel": civitai_base_model_raw,
            "base_model_specific_hint": ss_sd_model_name,
            "lineage_source": "civitai_baseModel" if civitai_base_model_raw else "unknown",
        }
        return (
            ConversionOutput(weights=primary_saved, metadata=source_meta),
            IngestResult(
                source_repo_dir=str(repo_dir),
                classifier_attrs=civitai_classifier_attrs,
            ),
        )
    else:
        raise ValueError(f"unsupported ingest provider: {provider_norm}")


__all__ = [
    # Re-exports from gen_worker.conversion.core_types
    "ConversionArtifact",
    "ConversionOutput",
    "IngestResult",
    "tensors_with",
    # Clone-pipeline-local helpers
    "require_local_weights",
    "default_output_ref",
    "ingest_from_source",
]
