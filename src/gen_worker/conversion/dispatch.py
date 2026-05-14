"""Finalize helpers for the class-shape conversion / training / dataset
endpoints (#332).

After the #332 hard-cut, all transform-kind tenants are class-shape
(``@conversion`` / ``@training`` / ``@dataset``). The worker's
``_execute_conversion_class_request`` invokes the tenant's
``generate()`` method, collects the returned ``list[ProducedFlavor]``,
and hands it to ``_finalize_produced_variants`` here for upload +
publish.

This module owns the upload + publish contract: given a list of
ProducedFlavors and the ``RequestContext`` carrying ``destination`` /
``source`` info, it walks each variant's output dir (or single file),
uploads every file via ``ctx.save_checkpoint`` (which routes to
tensorhub's CAS), builds the snapshot_manifest list of
``(path, digest, size_bytes)`` entries, and calls the appropriate
publish RPC:

  - ``publish_repo_revision`` for model checkpoints (conversion /
    training kinds — the ``sub_kind`` field on the endpoint class
    distinguishes ``format-conversion`` / ``quantization`` / etc.).
  - ``publish_dataset_revision`` for dataset artifacts (dataset kind,
    or any sub_kind starting with ``dataset-generation``).

The function-shape ``@conversion`` decorator was removed in
#332. Tenants now declare:

    @conversion(sub_kind="format-conversion", models={...})
    class MyConversion:
        def setup(self): ...
        @conversion.function
        def generate(self, ctx, payload) -> Iterator[ProducedFlavor]: ...
        def shutdown(self): ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .produced import ProducedFlavor

if TYPE_CHECKING:
    from ..request_context import RequestContext

_log = logging.getLogger(__name__)


# Mirror of training-endpoints canonical_quant.SCHEME_LIBRARY for the
# canonical (scheme → default library) mapping. Kept inline here so
# python-gen-worker doesn't take a hard dependency on training-endpoints
# at import time. Issue #258 task 3.
_CANONICAL_QUANT_SCHEME_LIBRARY: dict[str, str] = {
    "nf4":      "bitsandbytes",
    "int4_awq": "modelopt",
    "w4a8_awq": "modelopt",
    "int4_wo":  "torchao",
    "svdquant": "nunchaku",
    "hqq":      "hqq",
    "nvfp4":    "modelopt",
    "fp8":      "modelopt",
    "int8":     "modelopt",
}

_CHECKPOINT_METADATA_DROP_KEYS = {
    "flavor",
    "flavors",
    "produced_by_kind",
    "produced_by_job_id",
}


def _checkpoint_metadata_from_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-identity, non-provenance checkpoint metadata."""
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        clean = str(key or "").strip()
        if not clean or clean in _CHECKPOINT_METADATA_DROP_KEYS:
            continue
        out[clean] = value
    return out


def _resolve_quant_method(flavor: str, v_attrs: dict[str, Any], merged_attrs: dict[str, Any]) -> str:
    """Pick the canonical quantization method for a quantization-edge publish.

    Resolution order:
      1. Explicit `quantization_method` / `quant_method` / `quant_scheme`
         on the variant's attributes.
      2. The variant flavor itself, if it's a canonical scheme name
         (post-normalization the worker emits canonical names directly).
      3. Empty string — caller decides whether to skip lineage metadata.
    """
    for key in ("quantization_method", "quant_method", "quant_scheme"):
        v = str(v_attrs.get(key) or merged_attrs.get(key) or "").strip()
        if v:
            return v
    fl = str(flavor or "").strip()
    if fl in _CANONICAL_QUANT_SCHEME_LIBRARY:
        return fl
    return ""


def _resolve_quant_library(method: str, v_attrs: dict[str, Any], merged_attrs: dict[str, Any]) -> str:
    """Pick the canonical library that produced the bytes.

    Resolution order:
      1. Explicit `quantization_library` / `quant_library` on attrs.
      2. Canonical default for `method` from _CANONICAL_QUANT_SCHEME_LIBRARY.
      3. Empty string.
    """
    for key in ("quantization_library", "quant_library"):
        v = str(v_attrs.get(key) or merged_attrs.get(key) or "").strip()
        if v:
            return v
    return _CANONICAL_QUANT_SCHEME_LIBRARY.get(str(method).strip(), "")


def _finalize_produced_variants(
    request_context: "RequestContext",
    variants: list[ProducedFlavor],
    *,
    kind: str,
) -> None:
    """Upload each ProducedFlavor's files, build its manifest +
    snapshot_digest, and publish to the correct tensorhub subsystem
    based on ``kind``.

    Routes by kind:

    - ``kind.startswith("dataset-generation")`` → publish into
      ``tensorhub.datasets`` via ``publish_dataset_revision``. File bytes
      land in CAS same as checkpoints; the difference is the publish
      target: dataset artifacts shouldn't pollute the model-checkpoint
      search space.
    - everything else → publish into ``tensorhub.repo_checkpoints`` via
      ``publish_repo_revision`` (the original path — model weights).

    Upload flow in both branches:
      1. Walks the variant's output dir (or single file) and uploads
         each file via ``ctx.save_checkpoint``, capturing the
         blake3-keyed Tensors handle the upload returns.
      2. Builds a snapshot_manifest entry list from those handles:
         ``[{path, digest, size_bytes}]``.
      3. Computes ``snapshot_digest = sha256(canonical_json(entries))`` —
         content-addresses the whole variant snapshot.
      4. Calls the appropriate publish method with the snapshot_manifest
         + metadata payload.

    ``destination.tags`` (e.g. ``:prod``) are forwarded to
    ``publish_repo_revision`` in the checkpoint path so the tag +
    checkpoint land atomically. Datasets use naming-based versioning,
    not tags — the publish_dataset_revision path ignores destination.tags.
    """
    if not variants:
        return

    is_dataset_gen = (kind or "").startswith("dataset-generation")
    if is_dataset_gen:
        _finalize_dataset_variants(request_context, variants, kind=kind)
        return

    # ctx.destination / ctx.source are dicts ({ref, tags, ...} / {ref, attributes, ...}).
    destination = getattr(request_context, "destination", None) or {}
    if not isinstance(destination, dict):
        destination = {}
    dest_ref = str(destination.get("ref") or "").strip()

    source_info = getattr(request_context, "source", None) or {}
    if not isinstance(source_info, dict):
        source_info = {}
    source_ref = str(source_info.get("ref") or "").strip()
    source_checkpoint_id = str(
        source_info.get("checkpoint_id")
        or ""
    ).strip()
    # Fallback: extract the snapshot_digest from the materialized source path.
    # _materialize_source_for_training caches under
    # ``<cache>/cas/snapshots/<algo>:<hex>``; the last path segment IS the
    # checkpoint_id. Covers attribute-selector resolves where the caller
    # didn't know the concrete checkpoint_id up front.
    if not source_checkpoint_id:
        src_path = getattr(request_context, "source_path", None)
        if src_path:
            tail = Path(str(src_path)).name
            if tail and (":" in tail or len(tail) == 64):
                source_checkpoint_id = tail

    relationship_map = {
        "quantization":      "quantization",
        "format-conversion": "format-conversion",
        "fine-tune":         "fine-tune",
        "fine-tuning":       "fine-tune",
        "distillation":      "distillation",
        "pruning":           "pruning",
    }
    rel_kind = relationship_map.get(str(kind or "").lower(), "manual-fork")

    publish_fn = getattr(request_context, "publish_repo_revision", None)
    destination_tags = list(destination.get("tags") or [])

    publish_checkpoint_flavors: list[dict[str, Any]] = []
    aggregate_manifest_entries: list[dict[str, Any]] = []

    for variant in variants:
        v_attrs = dict(variant.attributes or {})
        checkpoint_metadata = _checkpoint_metadata_from_attrs(v_attrs)
        flavor = str(
            getattr(variant, "flavor", "")
            or v_attrs.get("flavor")
            or ""
        ).strip()
        flavors: list[str] = []
        raw_flavors = getattr(variant, "flavors", []) or []
        if isinstance(raw_flavors, list):
            for raw_flavor in raw_flavors:
                item = str(raw_flavor or "").strip()
                if item and item not in flavors:
                    flavors.append(item)
        if flavor and flavor not in flavors:
            flavors.insert(0, flavor)
        path = variant.path
        _log.info("finalize flavor: path=%s flavor=%s is_file=%s is_dir=%s dest=%s", path, flavor, path.is_file() if path else False, path.is_dir() if path else False, dest_ref)

        # Upload every file under the variant and collect the returned Tensors.
        # Each Tensors handle carries the blake3 digest + size_bytes that
        # tensorhub's CAS upload flow committed — that's what we put in the
        # snapshot_manifest.
        uploaded: list[tuple[str, Any]] = []
        if path.is_file():
            t = _upload_single_file_flavor(
                request_context, path, attributes=checkpoint_metadata,
            )
            if t is not None:
                uploaded.append((path.name, t))
        elif path.is_dir():
            uploaded.extend(_upload_directory_flavor(
                request_context, path, attributes=checkpoint_metadata,
            ))
        else:
            raise FileNotFoundError(f"ProducedFlavor.path does not exist: {path}")
        _log.info("finalize uploaded %d files for variant at %s", len(uploaded), path)

        # Build manifest entries from the uploaded Tensors. Each entry is
        # {path, digest, size_bytes}. Entries without a digest (e.g. save_fn
        # stubbed out in tests) are dropped — they can't participate in the
        # content-addressed snapshot.
        manifest_entries: list[dict[str, Any]] = []
        for rel_path, t in uploaded:
            digest = _tensors_blake3_digest(t)
            if not digest:
                continue
            entry = {
                "path":       rel_path,
                "digest":     digest,
                "size_bytes": int(getattr(t, "size_bytes", 0) or 0),
            }
            manifest_entries.append(entry)

        # Hand the manifest to tensorhub — the server computes checkpoint_id
        # as sha256(canonical(entries)) and persists the checkpoint row. The
        # worker intentionally doesn't know the hashing rule; that lives in
        # one place server-side.
        _log.info("finalize: manifest_entries=%d publish_callable=%s dest_ref=%s", len(manifest_entries), callable(publish_fn), dest_ref)
        if callable(publish_fn) and dest_ref and manifest_entries:
            # Issue #22: server-authoritative metadata. The body
            # carries the manifest + flavor labels + size only; server
            # infers kind / library / dtype / file_layout / file_type
            # from the uploaded file contents and writes the canonical
            # values.
            label = str(flavor or "").strip()
            if not label:
                recipe = str(v_attrs.get("quant_recipe") or "").strip()
                if recipe:
                    label = recipe.replace(":", "-").replace("/", "-")
            if not label:
                label = variant.path.name
            publish_checkpoint_flavor = {
                "flavor":            flavor or label,
                "flavors":           flavors or ([flavor or label] if flavor or label else []),
                # Per-variant manifest — publish_repo_revision forwards this
                # to tensorhub so each checkpoint_id is hashed from only
                # this flavor's files, not an aggregate across flavors.
                "snapshot_manifest": manifest_entries,
                "display_label":      label,
            }
            if checkpoint_metadata:
                publish_checkpoint_flavor["metadata"] = dict(checkpoint_metadata)
            # Issue #258 task 3: for quantization edges, attach the
            # canonical method + library to the publish payload so
            # tensorhub's lineage edge metadata carries the quant identity
            # (validated server-side by ValidateQuantizationLineageMetadata).
            if rel_kind == "quantization":
                quant_method = _resolve_quant_method(flavor, v_attrs, checkpoint_metadata)
                quant_library = _resolve_quant_library(quant_method, v_attrs, checkpoint_metadata)
                if quant_method or quant_library:
                    quant_meta: dict[str, Any] = {}
                    if quant_method:
                        quant_meta["quantization_method"] = quant_method
                    if quant_library:
                        quant_meta["quantization_library"] = quant_library
                    quant_params = v_attrs.get("quantization_params") or checkpoint_metadata.get("quantization_params")
                    if isinstance(quant_params, dict):
                        quant_meta["quantization_params"] = quant_params
                    publish_checkpoint_flavor["lineage_metadata"] = quant_meta
            publish_checkpoint_flavors.append(publish_checkpoint_flavor)
            aggregate_manifest_entries.extend(manifest_entries)

    if callable(publish_fn) and dest_ref and publish_checkpoint_flavors:
        metadata = {
            "checkpoint_flavors": publish_checkpoint_flavors,
        }
        # Each checkpoint flavor carries its own per-flavor manifest above;
        # the aggregate manifest is kept only as request-level context.
        snapshot_manifest = {
            "version": 1,
            "entries": aggregate_manifest_entries,
        }
        try:
            publish_fn(
                destination_repo=dest_ref,
                metadata=metadata,
                source_repo=source_ref,
                source_version_id=source_checkpoint_id,
                snapshot_manifest=snapshot_manifest,
                relationship_kind=rel_kind,
                auto_create_external_parent=False,
                destination_repo_tags=destination_tags,
            )
        except Exception as exc:  # noqa: BLE001 — surface via event, don't fail the job
            emit = getattr(request_context, "emit", None)
            if emit is not None:
                emit("transform.publish_failed", {
                    "ref": dest_ref,
                    "error": f"{type(exc).__name__}: {exc}",
                })


def _finalize_dataset_variants(
    request_context: "RequestContext",
    variants: list[ProducedFlavor],
    *,
    kind: str,
) -> None:
    """Publish dataset-generation tenant outputs into ``tensorhub.datasets``.

    Called from ``_finalize_produced_variants`` when the tenant's kind starts
    with ``dataset-generation``. The flow:

      1. Upload every file under the variant via ``ctx.save_checkpoint``
         (reuses the same blob-CAS path as checkpoints; the bytes are
         identical, only the catalog metadata differs).
      2. Read ``dataset_info.json`` at the variant root — it's the
         authoritative source of truth for features_json / kind / num_rows.
      3. Call ``publish_dataset_revision`` to create (or update) the
         ``tensorhub.datasets`` row. The dataset table is mutable;
         versioning is by naming convention.

    Failures in publish are surfaced as ``dataset.publish_failed`` events
    rather than re-raised — the blob uploads already landed, so the
    content is recoverable even if the catalog update fails.
    """
    import json as _json

    destination = getattr(request_context, "destination", None) or {}
    if not isinstance(destination, dict):
        destination = {}
    dest_ref = str(destination.get("ref") or "").strip()
    if not dest_ref:
        _log.warning("dataset publish: no destination.ref; uploads will succeed but no dataset row created")
        return

    visibility = str(destination.get("visibility") or "").strip().lower()
    if visibility not in ("private", "public"):
        visibility = "private"

    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "") or "")
    library_provenance: dict[str, str] = {"produced_by_job_id": job_id, "produced_by_kind": kind}

    publish_fn = getattr(request_context, "publish_dataset_revision", None)

    for variant in variants:
        merged_attrs = {**library_provenance, **dict(variant.attributes or {})}
        path = variant.path
        _log.info("dataset finalize: path=%s dest=%s", path, dest_ref)

        # Upload files — same as checkpoint path, just lands in CAS.
        uploaded: list[tuple[str, Any]] = []
        if path.is_file():
            t = _upload_single_file_flavor(request_context, path, attributes=merged_attrs)
            if t is not None:
                uploaded.append((path.name, t))
        elif path.is_dir():
            uploaded.extend(_upload_directory_flavor(request_context, path, attributes=merged_attrs))
        else:
            raise FileNotFoundError(f"ProducedFlavor.path does not exist: {path}")

        # Build the snapshot_manifest from uploaded blob digests — records
        # content identity for the revision even though the dataset row
        # is mutable.
        manifest_entries: list[dict[str, Any]] = []
        for rel_path, t in uploaded:
            digest = _tensors_blake3_digest(t)
            if not digest:
                continue
            manifest_entries.append({
                "path": rel_path,
                "digest": digest,
                "size_bytes": int(getattr(t, "size_bytes", 0) or 0),
            })

        # Parse dataset_info.json for features_json + kind. Falls back to
        # a minimal schema if the file is missing.
        info_path = path / "dataset_info.json" if path.is_dir() else None
        features_json: dict[str, Any] = {}
        dataset_kind: str = ""
        dataset_info: dict[str, Any] = {}
        if info_path is not None and info_path.exists():
            try:
                with open(info_path) as f:
                    dataset_info = _json.load(f)
                if isinstance(dataset_info, dict):
                    features_json = dict(dataset_info.get("features") or {})
                    dataset_kind = str(dataset_info.get("kind") or "")
            except Exception as exc:  # noqa: BLE001
                _log.warning("dataset finalize: failed to parse dataset_info.json (%s)", exc)

        if not callable(publish_fn):
            _log.warning(
                "dataset finalize: request_context.publish_dataset_revision missing; "
                "blobs uploaded to CAS but no tensorhub.datasets row created. "
                "gen-worker release needs bump."
            )
            continue

        try:
            result = publish_fn(
                destination_dataset=dest_ref,
                features_json=features_json,
                row_artifacts_json=None,
                snapshot_manifest=manifest_entries,
                visibility=visibility,
                kind=dataset_kind,
                dataset_info=dataset_info,
            )
            _log.info("dataset finalize: published %s -> %s", dest_ref, result)
        except Exception as exc:  # noqa: BLE001
            emit = getattr(request_context, "emit", None)
            if emit is not None:
                emit("dataset.publish_failed", {
                    "ref": dest_ref,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            _log.warning("dataset finalize: publish failed for %s: %s", dest_ref, exc)


def _tensors_blake3_digest(t: Any) -> str:
    """Return the ``blake3:<hex>`` digest the upload flow bound for this
    Tensors handle. Tensorhub's CAS indexes blobs by blake3; the commit
    step verifies the uploaded bytes hashed to this digest, so we trust
    it here. Returns ``""`` when the upload path didn't populate a digest
    (e.g. test stubs), in which case the manifest entry is dropped.
    """
    bd = str(getattr(t, "blob_digest", "") or "").strip()
    if bd:
        return bd if ":" in bd else f"blake3:{bd}"
    b3 = str(getattr(t, "blake3", "") or "").strip()
    if b3:
        return b3 if ":" in b3 else f"blake3:{b3}"
    return ""


def _upload_single_file_flavor(
    request_context: "RequestContext",
    path: Path,
    *,
    attributes: dict[str, str],
) -> Any:
    """Upload one safetensors/gguf/flashpack file; return the Tensors handle.

    Returns ``None`` when ``save_checkpoint`` is missing (test stubs).
    """
    save_fn = getattr(request_context, "save_checkpoint", None)
    if save_fn is None:
        return None
    ref = _default_flavor_ref(request_context, path)
    format_hint = path.suffix.lstrip(".") or "bin"
    return save_fn(ref, str(path), format=format_hint, attributes=attributes)


def _upload_directory_flavor(
    request_context: "RequestContext",
    dir_path: Path,
    *,
    attributes: dict[str, str],
) -> list[tuple[str, Any]]:
    """Upload every file under a directory flavor; return ``[(rel_path, Tensors)]``."""
    save_fn = getattr(request_context, "save_checkpoint", None)
    if save_fn is None:
        return []
    base_ref = _default_flavor_ref(request_context, dir_path)
    uploaded: list[tuple[str, Any]] = []
    for f in sorted(dir_path.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(dir_path).as_posix()
        ref = f"{base_ref}/{rel}"
        format_hint = f.suffix.lstrip(".") or "bin"
        t = save_fn(ref, str(f), format=format_hint, attributes=attributes)
        uploaded.append((rel, t))
    return uploaded


def _default_flavor_ref(request_context: "RequestContext", path: Path) -> str:
    """Return a job-scoped ref for a produced flavor."""
    job_id = str(getattr(request_context, "job_id", "") or "") \
        or str(getattr(request_context, "request_id", "r")) \
        or "r"
    return f"jobs/{job_id}/outputs/{path.name}"


__all__ = [
    "_finalize_produced_variants",
    "_finalize_dataset_variants",
]
