from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gen_worker import RequestContext, Tensors

# clone_pipeline calls the library streaming primitives directly (see
# _apply_save_format) instead of going through the transform-endpoint tenant
# functions — those use a different dispatch shape (@training_function) that
# expects ctx.source_path to point at a materialized tensorhub snapshot, but
# clone_pipeline works from a local Tensors object that's already been
# ingested from an external URL.
from gen_worker.conversion.ingest import (
    CivitaiResolvedIdentity,
    resolve_civitai_source_identity,
    resolve_huggingface_source_identity,
)
# Layout repackage (singlefile↔diffusers) calls library primitives directly
# for the same reason _apply_save_format does — clone_pipeline works on local
# paths, not a tensorhub snapshot.
from gen_worker.conversion.repackage import (
    diffusers_to_singlefile,
    singlefile_to_diffusers,
)
from ._shared import ConversionOutput, IngestResult, default_output_ref, ingest_from_source, tensors_with


_PUBLIC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_PUBLIC_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_LAYOUT_POLICY = "diffusers->diffusers,diffusers->singlefile,singlefile->diffusers,singlefile->singlefile"
_SOURCE_LAYOUT_PREFERENCE_POLICY = "auto,diffusers,aio"


def normalize_destination_repo_name(value: str) -> str:
    destination_repo_name = str(value or "").strip().lower()
    if destination_repo_name == "":
        raise ValueError("destination_repo is invalid")
    if _PUBLIC_NAME_RE.match(destination_repo_name) is None:
        raise ValueError("destination_repo is invalid")
    return destination_repo_name


def normalize_destination_owner(value: str) -> str:
    destination_owner = str(value or "").strip().lower()
    if destination_owner == "":
        raise ValueError("invoker owner is required")
    if _PUBLIC_NAME_RE.match(destination_owner) is None:
        raise ValueError("invoker owner is invalid")
    return destination_owner


def normalize_destination_ref(value: str) -> str:
    destination_ref = str(value or "").strip().lower()
    if destination_ref == "":
        raise ValueError("destination_repo is required")
    # Strip the optional "cozy:" scheme prefix (e.g. "cozy:owner/repo" -> "owner/repo")
    if destination_ref.startswith("cozy:"):
        destination_ref = destination_ref[len("cozy:"):]
    parts = destination_ref.split("/", 1)
    if len(parts) != 2:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    _ = normalize_destination_owner(parts[0])
    _ = normalize_destination_repo_name(parts[1])
    return destination_ref


def normalize_target_layout(value: str | None) -> str:
    layout = str(value or "diffusers").strip().lower()
    if layout == "":
        layout = "diffusers"
    if layout not in {"singlefile", "diffusers"}:
        raise ValueError("target_layout must be one of: singlefile, diffusers")
    return layout


def normalize_source_layout_preference(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode == "":
        mode = "auto"
    if mode not in {"auto", "diffusers", "aio"}:
        raise ValueError("source_layout_preference must be one of: auto, diffusers, aio")
    return mode


def normalize_save_formats(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        fmt = str(raw or "").strip().lower()
        if fmt == "" or fmt in seen:
            continue
        seen.add(fmt)
        out.append(fmt)
    return out


# ---------------------------------------------------------------------------
# OutputSpec: the canonical per-variant request.
# ---------------------------------------------------------------------------

# Canonical output dtypes recognized by the conversion pipeline. "pt" dtypes
# like fp16/fp32 are passthrough-eligible when the source weights already
# match; bf16/fp8:*/nvfp4 always require a conversion pass.
_KNOWN_DTYPES = {
    "fp32",
    "fp16",
    "bf16",
    "fp8",
    "fp8:e4m3",
    "fp8:e5m2",
    "nvfp4",
    "int8",
    "int8:awq",
    "int8:gptq",
    "int4",
    "int4:wo",
    "int4:nf4",
    "int4:fp4",
    "int4:awq",
    "int4:gptq",
    "nf4",
    "fp4",
}

_KNOWN_FILE_LAYOUTS = {"diffusers", "singlefile", "aio"}
_KNOWN_FILE_TYPES = {"safetensors", "flashpack", "gguf", "bin"}


@dataclass(frozen=True)
class OutputSpec:
    """One requested output flavor: dtype + file layout + container format."""

    dtype: str
    file_layout: str
    file_type: str

    @property
    def label(self) -> str:
        """Canonical flavor label used for checkpoint publish."""
        return f"{self.dtype}-{self.file_layout}-{self.file_type}".replace(":", "-")


_DEFAULT_OUTPUT_SPEC = OutputSpec(dtype="bf16", file_layout="diffusers", file_type="safetensors")


def _normalize_dtype_token(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if s in {"fp8-e4m3", "fp8_e4m3"}:
        return "fp8:e4m3"
    if s in {"fp8-e5m2", "fp8_e5m2"}:
        return "fp8:e5m2"
    return s


def normalize_outputs(
    values: Iterable[Any],
    *,
    target_layout_hint: str = "diffusers",
) -> list[OutputSpec]:
    """Accept a list of dicts/OutputSpec/None; return deduped OutputSpec list.

    Unknown dtype/layout/filetype tokens raise ValueError so the client gets a
    clean error at submit time rather than a silent fallthrough later.
    """
    raw_items = list(values or [])
    out: list[OutputSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, OutputSpec):
            dtype = _normalize_dtype_token(item.dtype)
            layout = str(item.file_layout or "").strip().lower() or target_layout_hint
            ftype = str(item.file_type or "").strip().lower() or "safetensors"
        elif isinstance(item, dict):
            dtype = _normalize_dtype_token(str(item.get("dtype") or ""))
            layout = str(item.get("file_layout") or "").strip().lower() or target_layout_hint
            ftype = str(item.get("file_type") or "").strip().lower() or "safetensors"
        else:
            # Struct-like (msgspec.Struct) — duck-type via getattr.
            dtype = _normalize_dtype_token(str(getattr(item, "dtype", "") or ""))
            layout = str(getattr(item, "file_layout", "") or "").strip().lower() or target_layout_hint
            ftype = str(getattr(item, "file_type", "") or "").strip().lower() or "safetensors"
        if dtype == "":
            raise ValueError("output.dtype is required")
        if dtype not in _KNOWN_DTYPES:
            raise ValueError(f"unsupported output.dtype: {dtype!r}")
        if layout not in _KNOWN_FILE_LAYOUTS:
            raise ValueError(f"unsupported output.file_layout: {layout!r}")
        if ftype not in _KNOWN_FILE_TYPES:
            raise ValueError(f"unsupported output.file_type: {ftype!r}")
        key = (dtype, layout, ftype)
        if key in seen:
            continue
        seen.add(key)
        out.append(OutputSpec(dtype=dtype, file_layout=layout, file_type=ftype))
    if not out:
        out.append(OutputSpec(
            dtype=_DEFAULT_OUTPUT_SPEC.dtype,
            file_layout=target_layout_hint if target_layout_hint in _KNOWN_FILE_LAYOUTS else _DEFAULT_OUTPUT_SPEC.file_layout,
            file_type=_DEFAULT_OUTPUT_SPEC.file_type,
        ))
    return out


def save_formats_to_outputs(
    save_formats: Iterable[str],
    *,
    target_layout: str,
) -> list[OutputSpec]:
    """Translate the legacy `save_formats` list into OutputSpec entries.

    Preserves ordering and dedupes. Used as a compatibility bridge while
    callers still send `save_formats` instead of `outputs`.
    """
    layout = str(target_layout or "").strip().lower() or "diffusers"
    if layout not in _KNOWN_FILE_LAYOUTS:
        layout = "diffusers"
    specs: list[OutputSpec] = []
    for raw in save_formats or []:
        fmt = str(raw or "").strip().lower()
        if fmt == "":
            continue
        if fmt == "flashpack":
            specs.append(OutputSpec(dtype="bf16", file_layout="singlefile", file_type="flashpack"))
        elif fmt == "bf16":
            specs.append(OutputSpec(dtype="bf16", file_layout=layout, file_type="safetensors"))
        elif fmt == "fp16":
            specs.append(OutputSpec(dtype="fp16", file_layout=layout, file_type="safetensors"))
        elif fmt == "fp32":
            specs.append(OutputSpec(dtype="fp32", file_layout=layout, file_type="safetensors"))
        elif fmt == "nvfp4":
            specs.append(OutputSpec(dtype="nvfp4", file_layout=layout, file_type="safetensors"))
        elif fmt.startswith("fp8"):
            specs.append(OutputSpec(dtype=fmt, file_layout=layout, file_type="safetensors"))
        elif fmt in {"int8", "int4", "nf4", "fp4"} or fmt.startswith("int8:") or fmt.startswith("int4:"):
            # Issue #73 inline weight-only quantization. int8 → torchao Int8Tensor;
            # int4/nf4/fp4 → bitsandbytes Params4bit (CPU-friendly); int4:awq /
            # int4:gptq → calibrated, refused downstream with separate-job hint.
            specs.append(OutputSpec(dtype=fmt, file_layout=layout, file_type="safetensors"))
        elif fmt.startswith("gguf"):
            encoding = fmt.split(":", 1)[1] if ":" in fmt else "f16"
            dtype = "fp16" if encoding in {"f16", "fp16"} else ("bf16" if encoding in {"bf16"} else "fp16")
            specs.append(OutputSpec(dtype=dtype, file_layout="singlefile", file_type="gguf"))
        else:
            raise ValueError(f"unsupported legacy save_format: {fmt!r}")
    return normalize_outputs(specs, target_layout_hint=layout) if specs else []


def output_spec_to_save_format(spec: OutputSpec) -> str:
    """Map OutputSpec → legacy save_format string for `_apply_save_format`.

    Only intended for non-passthrough outputs. Passthrough outputs should
    bypass `_apply_save_format` entirely and upload source weights directly.
    """
    if spec.file_type == "flashpack":
        return "flashpack"
    if spec.file_type == "gguf":
        # GGUF outputs route by their dtype directly — q4_k_m, q8_0, f16, bf16.
        # The inline gguf converter handles both direct encodings (f16/bf16/q8_0)
        # and llama-quantize two-step targets (q4_k_m, q5_k_m, …).
        return f"gguf:{spec.dtype}"
    # safetensors container — dispatch by dtype.
    if spec.dtype in {"bf16", "fp16", "fp32"}:
        return spec.dtype
    if spec.dtype.startswith("fp8"):
        return spec.dtype  # "fp8:e4m3", "fp8:e5m2"
    if spec.dtype == "nvfp4":
        return "nvfp4"
    if spec.dtype in {"int8", "int4", "nf4", "fp4"} or spec.dtype.startswith("int4:") or spec.dtype.startswith("int8:"):
        return spec.dtype
    # Unknown safetensors dtype — caller treats this as passthrough-only.
    return ""


def normalize_destination_repo_tags(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        tag = str(raw or "").strip().lower()
        if tag == "" or tag in seen:
            continue
        if _PUBLIC_TAG_RE.match(tag) is None:
            raise ValueError("destination_repo_tags contains an invalid tag")
        if tag == "latest":
            raise ValueError("destination_repo_tags must not include latest")
        seen.add(tag)
        out.append(tag)
    out.sort()
    return out


def normalize_source_ref(value: str) -> str:
    source_ref = str(value or "").strip()
    if source_ref == "":
        raise ValueError("source_ref is required")
    return source_ref


def compute_source_hash(*, provider: str, source_ref: str, source_revision: str | None) -> str:
    normalized = "\n".join(
        [
            str(provider or "").strip().lower(),
            str(source_ref or "").strip().lower(),
            str(source_revision or "").strip(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _compute_identity_hash(*, provider: str, source_ref: str, source_revision: str | None) -> str:
    return "sha256:" + compute_source_hash(
        provider=provider,
        source_ref=source_ref,
        source_revision=source_revision,
    )


def _normalize_sha256(raw: str | None) -> str:
    value = str(raw or "").strip().lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1].strip().lower()
    if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
        return value
    return ""


def _parse_positive_int(raw: object) -> int | None:
    try:
        parsed = int(str(raw or "").strip())
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _is_http_source(source_ref: str) -> bool:
    raw = str(source_ref or "").strip().lower()
    return raw.startswith("http://") or raw.startswith("https://")


@dataclass(frozen=True)
class MirrorPreflight:
    destination_exists: bool
    destination_metadata: dict[str, Any]
    noop: bool
    mirror_metadata: dict[str, str]


@dataclass(frozen=True)
class SourceIdentity:
    provider: str
    source_ref: str
    source_revision: str
    identity_hash: str
    dedupe_supported: bool
    civitai_model_version_id: int | None
    civitai_file_id: int | None
    source_metadata: dict[str, str]
    resolved_civitai_identity: CivitaiResolvedIdentity | None


@dataclass(frozen=True)
class FinalizeCloneResult:
    output: ConversionOutput
    published_version_id: str


def _pick_primary_source_file(repo_dir: Path) -> Path:
    """Pick the canonical safetensors entry-point for inline conversion.

    Preference order:
      1. ``model.safetensors.index.json`` (sharded transformers)
      2. ``model.safetensors`` (single-file transformers)
      3. The largest ``*.safetensors`` file in the repo root
      4. The largest ``*.safetensors`` file anywhere under repo_dir

    Used by the publish-as-is path's inline conversion to give
    ``streaming_dtype_cast`` an entry point. The streaming reader
    handles the index → shards expansion internally.
    """
    repo_dir = Path(repo_dir)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"source repo dir is not a directory: {repo_dir}")
    canonical_index = repo_dir / "model.safetensors.index.json"
    if canonical_index.is_file():
        return canonical_index
    canonical_single = repo_dir / "model.safetensors"
    if canonical_single.is_file():
        return canonical_single
    root_safetensors = sorted(
        (p for p in repo_dir.iterdir() if p.is_file() and p.suffix == ".safetensors"),
        key=lambda p: p.stat().st_size, reverse=True,
    )
    if root_safetensors:
        return root_safetensors[0]
    nested_safetensors = sorted(
        (p for p in repo_dir.rglob("*.safetensors") if p.is_file()),
        key=lambda p: p.stat().st_size, reverse=True,
    )
    if nested_safetensors:
        return nested_safetensors[0]
    raise FileNotFoundError(f"no .safetensors entry point found under {repo_dir}")


def _tensors_artifact_module(t: Any) -> dict[str, Any]:
    """Extract artifact dict from a Tensors object. Module-level mirror of the
    inner `_tensors_artifact` defined inside `_finalize_clone` — used by
    `_finalize_publish_as_is` which is at
    module scope and can't see the inner function."""
    art: dict[str, Any] = {}
    digest = str(getattr(t, "blob_digest", "") or getattr(t, "blake3", "") or "").strip()
    if digest and ":" not in digest:
        digest = f"blake3:{digest}"
    if digest:
        art["digest"] = digest
    path = str(getattr(t, "ref", "") or "").strip()
    if path:
        art["path"] = path
    size = getattr(t, "size_bytes", None)
    if size is not None:
        art["size_bytes"] = int(size)
    domain = str(getattr(t, "blob_domain", "") or "private").strip()
    art["domain"] = domain
    return art


def _finalize_publish_as_is(
    ctx: RequestContext,
    *,
    source_identity: SourceIdentity,
    source_version_id: str | None,
    destination_repo: str,
    destination_repo_name: str,
    destination_repo_tags: list[str],
    ingested: ConversionOutput,
    ingest_result: IngestResult,
    emit_stage: Any,
    auto_publish_public: bool,
    overwrite_repo: bool,
    output_specs: list[OutputSpec] | None = None,
) -> FinalizeCloneResult:
    """Thin finalize for non-diffusers library classes.

    Uploads every ingested file to repo CAS, builds a snapshot manifest from
    the upload set, attaches the classifier-derived attributes, publishes.
    For requested OutputSpecs whose dtype the source already ships, this is
    a direct passthrough; for non-matching dtypes we run inline conversion
    via gen_worker.conversion.inline_convert — the same library code paths
    other worker functions can call. Calibrated quants (int4:awq etc.) raise
    InlineConversionNotPossible upstream; this path catches them and
    surfaces failed_flavors in metadata.

    The diffusers-aware monolith in `_finalize_clone` keeps handling
    diffusers trees that need multi-component layout repackage and per-output
    save-format conversion. Everything else lands here.
    """
    classifier_attrs = dict(ingest_result.classifier_attrs or {})
    runtime_library = str(classifier_attrs.get("runtime_library") or "").strip().lower()
    requested_specs = list(output_specs or [])

    # Promote every locally-staged weight tensor to a real CAS upload, and
    # build the snapshot_manifest using REPO-RELATIVE paths (`config.json`,
    # `unet/diffusion_pytorch_model.safetensors`, etc.) — not the upload
    # `ref` paths which are `jobs/<rid>/outputs/source-repo/<rel>`. Tensorhub's
    # layout-contract validator inspects manifest path strings; entries must
    # be canonical or the validator's `path == "config.json"` check fails.
    promoted_artifacts: list[dict[str, Any]] = []
    snapshot_manifest: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()  # (path, digest) dedup

    for t, rel_path, _size in list(ingest_result.all_file_tensors or []):
        rel_path = str(rel_path or "").strip().replace("\\", "/").lstrip("/")
        if not rel_path:
            continue
        # If the tensor already has a blob_digest, it was uploaded during
        # the ingest's non-weight-file pass. Reuse the digest.
        if t.blob_digest:
            saved = t
        else:
            local_path = str(t.local_path or "").strip()
            if not local_path:
                continue
            ref = f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}"
            ext = Path(rel_path).suffix.lstrip(".") or "bin"
            try:
                saved = ctx.save_checkpoint(ref, local_path, format=ext)
            except Exception as exc:
                raise RuntimeError(f"finalize_publish_as_is: failed to upload {rel_path}: {exc}") from exc

        art = _tensors_artifact_module(saved)
        digest = str(art.get("digest") or "").strip()
        if not digest:
            continue
        size = int(art.get("size_bytes") or 0)
        # Manifest entry uses the repo-relative path so the layout-contract
        # validator and downstream resolvers see the canonical structure.
        # Tensorhub's SnapshotManifestFile struct strict-decodes; only
        # path / type / size_bytes / digest / blake3 / url / validator* are
        # accepted.
        manifest_entry: dict[str, Any] = {
            "path": rel_path,
            "digest": digest,
            "size_bytes": size,
        }
        # Artifact entry (artifact_refs) keeps the upload-side `path` so
        # publish_repo_revision can build artifact_refs correctly.
        artifact_entry = dict(art)
        # Override `path` on the artifact to be the rel_path too — that's
        # what publish_repo_revision propagates as "path" into snapshot
        # entries when no per-flavor manifest is supplied; aligning both
        # avoids a mismatch.
        artifact_entry["path"] = rel_path

        key = (rel_path, digest)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        promoted_artifacts.append(artifact_entry)
        snapshot_manifest.append(manifest_entry)

    if not promoted_artifacts:
        raise RuntimeError("finalize_publish_as_is: no artifacts produced")

    # Single flavor entry covering everything we ingested.
    primary_dtype = str(classifier_attrs.get("dtype") or ingested.metadata.get("dtype") or "bf16").strip().lower()
    primary_filetype = str(classifier_attrs.get("file_type") or "safetensors").strip().lower()
    # `file_layout` is the tensorhub-validated axis.
    # Only diffusers uses non-empty values (`multi-file` / `single-file`).
    # transformers / peft / sentence-transformers / gguf / native_lora — empty.
    # The classifier may have stamped a strategy-name into `file_layout` for
    # tagging purposes; that label rides under a different attr key
    # (`layout_kind`) so it doesn't collide with the validator's enum.
    classifier_layout = str(classifier_attrs.get("file_layout") or "").strip().lower()
    primary_layout = ""  # empty for non-diffusers; validator requires this
    primary_label_layout = classifier_layout or runtime_library or "transformers"
    flavor_label_parts = [primary_dtype, primary_label_layout, primary_filetype]
    flavor_label = "-".join([p for p in flavor_label_parts if p])

    commit_checkpoint_flavors: list[dict[str, Any]] = [{
        "flavor": flavor_label,
        "flavors": [primary_dtype, primary_layout, primary_filetype],
        "display_label": flavor_label,
        "artifacts": promoted_artifacts,
        # The publish_repo_revision wrapper looks at v.get("snapshot_manifest")
        # per-flavor before falling back to the top-level manifest entries.
        # Put the manifest here so it lands on the publish request body.
        "snapshot_manifest": snapshot_manifest,
    }]

    # When the HF classifier resolved multiple
    # concrete dtypes in the source repo (multi-quant GGUF, side-by-side
    # transformers variants), publish one flavor per resolved dtype under
    # the same destination tag. Each per-checkpoint attrs dict comes from
    # `IngestResult.classifier_attrs_per_checkpoint` and carries its own
    # `dtype` value; we filter to entries whose dtype != primary_dtype
    # (the primary flavor above already covers that one) and synthesize
    # one extra flavor per remaining dtype, sharing the snapshot_manifest
    # since the union of files was downloaded once.
    per_checkpoint_attrs = list(ingest_result.classifier_attrs_per_checkpoint or [])
    for ck_attrs in per_checkpoint_attrs:
        ck_dtype = str(ck_attrs.get("dtype") or "").strip().lower()
        if not ck_dtype or ck_dtype == primary_dtype:
            continue
        ck_filetype = str(ck_attrs.get("file_type") or primary_filetype).strip().lower()
        ck_label_layout = str(ck_attrs.get("layout_kind") or runtime_library or "transformers").strip().lower()
        ck_label = "-".join([p for p in (ck_dtype, ck_label_layout, ck_filetype) if p])
        commit_checkpoint_flavors.append({
            "flavor": ck_label,
            "flavors": [ck_dtype, "", ck_filetype],
            "display_label": ck_label,
            "artifacts": promoted_artifacts,
            "snapshot_manifest": snapshot_manifest,
        })

    # For each requested OutputSpec whose dtype differs from the source, run
    # inline conversion and emit a per-spec flavor entry. Specs whose dtype
    # matches the source's primary dtype are already covered by the passthrough
    # flavor above. Calibrated quants raise InlineConversionNotPossible; we
    # record structured requirements so callers can render their own guidance.
    failed_flavors: list[dict[str, Any]] = []
    if requested_specs:
        from gen_worker.conversion.inline_convert import (
            InlineConversionNotPossible,
            run_inline_conversion,
        )
        repo_dir_str = str(ingest_result.source_repo_dir or "").strip()
        repo_dir = Path(repo_dir_str) if repo_dir_str else None

        # Tag the existing primary flavor with the spec.label that matches its
        # dtype, so callers can map "spec → flavor" cleanly. If no spec
        # matches, the primary flavor stays as-is (legacy behavior).
        primary_matched_spec: OutputSpec | None = None
        for spec in requested_specs:
            if str(spec.dtype or "").strip().lower() == primary_dtype:
                primary_matched_spec = spec
                break
        if primary_matched_spec is not None:
            commit_checkpoint_flavors[0]["flavor"] = primary_matched_spec.label
            commit_checkpoint_flavors[0]["display_label"] = primary_matched_spec.label

        for spec in requested_specs:
            if spec is primary_matched_spec:
                continue  # already covered by the passthrough flavor
            spec_dtype = str(spec.dtype or "").strip().lower()
            if spec_dtype == "" or spec_dtype == primary_dtype:
                continue
            if repo_dir is None or not repo_dir.exists():
                failed_flavors.append({
                    "spec_label": spec.label,
                    "dtype": spec.dtype,
                    "file_type": spec.file_type,
                    "reason": "source repo dir is not available for inline conversion",
                })
                continue
            try:
                inline_out_dir = repo_dir.parent / f"_inline_{spec.label}"
                inline_out_dir.mkdir(parents=True, exist_ok=True)
                # Pick a "primary" weight file from the source for the cast
                # path's input. For transformers / sentence-transformers /
                # peft this is usually `model.safetensors` (or the index).
                primary_source = _pick_primary_source_file(repo_dir)
                inline_result = run_inline_conversion(
                    source_path=primary_source,
                    out_dir=inline_out_dir,
                    target_dtype=spec_dtype,
                    target_file_type=str(spec.file_type or "safetensors").strip().lower(),
                    source_repo_dir=repo_dir,
                )
            except InlineConversionNotPossible as exc:
                failed: dict[str, Any] = {
                    "spec_label": spec.label,
                    "dtype": spec.dtype,
                    "file_type": spec.file_type,
                    "reason": exc.reason,
                }
                if exc.deferred_requirement is not None:
                    failed["deferred_requirement"] = exc.deferred_requirement.as_dict()
                failed_flavors.append(failed)
                if emit_stage is not None:
                    payload: dict[str, Any] = {
                        "output_spec": spec.label,
                        "reason": exc.reason,
                    }
                    if exc.deferred_requirement is not None:
                        payload["deferred_requirement"] = exc.deferred_requirement.as_dict()
                    emit_stage("clone.save_format.skipped", 0.85, payload)
                continue
            except Exception as exc:  # noqa: BLE001 — record + continue
                import traceback
                tb = traceback.format_exc()
                failed_flavors.append({
                    "spec_label": spec.label,
                    "dtype": spec.dtype,
                    "file_type": spec.file_type,
                    "reason": f"inline conversion failed: {exc}",
                    "traceback": tb,
                })
                # Print a structured worker-side log so operators can see WHAT
                # failed (the failed_flavors map only reaches the API caller's
                # response payload — inline conversion failures are common
                # enough during ramp-up that we want them in the worker log).
                import sys
                print(
                    f"[inline-convert] flavor={spec.label} FAILED: {exc}\n{tb}",
                    file=sys.stderr,
                    flush=True,
                )
                if emit_stage is not None:
                    emit_stage("clone.save_format.failed", 0.85, {
                        "output_spec": spec.label,
                        "reason": str(exc),
                    })
                continue

            # Promote the converted output files to CAS and build a per-spec
            # flavor entry. Each output file lives at
            # `<repo>/<filename>` for the destination snapshot.
            spec_artifacts: list[dict[str, Any]] = []
            spec_manifest: list[dict[str, Any]] = []
            for out_path in inline_result.output_paths:
                rel = out_path.name
                ref_key = f"jobs/{ctx.request_id}/outputs/inline-{spec.label}/{rel}"
                ext = Path(rel).suffix.lstrip(".") or "bin"
                try:
                    saved = ctx.save_checkpoint(ref_key, str(out_path), format=ext)
                except Exception as exc:
                    failed_flavors.append({
                        "spec_label": spec.label,
                        "dtype": spec.dtype,
                        "file_type": spec.file_type,
                        "reason": f"inline conversion produced output but upload failed: {exc}",
                    })
                    spec_artifacts = []
                    spec_manifest = []
                    break
                art = _tensors_artifact_module(saved)
                digest = str(art.get("digest") or "").strip()
                if not digest:
                    continue
                size = int(art.get("size_bytes") or 0)
                spec_manifest.append({
                    "path": rel,
                    "digest": digest,
                    "size_bytes": size,
                })
                art_entry = dict(art)
                art_entry["path"] = rel
                spec_artifacts.append(art_entry)
            # Carry every non-weight source file (config.json, tokenizer
            # files, …) onto the converted flavor too — without these the
            # inference loader can't reconstruct the model.
            for src_art in promoted_artifacts:
                src_rel = str(src_art.get("path") or "").strip()
                if not src_rel:
                    continue
                # Skip weight files from the source; the inline conversion
                # produced the converted equivalents.
                if src_rel.endswith(".safetensors") or src_rel.endswith(".bin"):
                    continue
                if src_rel.endswith(".safetensors.index.json"):
                    continue
                spec_artifacts.append(dict(src_art))
                # Mirror the file into the per-spec manifest too.
                src_manifest_entry = next(
                    (
                        dict(m) for m in snapshot_manifest
                        if str(m.get("path") or "") == src_rel
                    ),
                    None,
                )
                if src_manifest_entry is not None:
                    spec_manifest.append(src_manifest_entry)
            if not spec_artifacts:
                failed_flavors.append({
                    "spec_label": spec.label,
                    "dtype": spec.dtype,
                    "file_type": spec.file_type,
                    "reason": "inline conversion produced no uploadable artifacts",
                })
                continue

            commit_checkpoint_flavors.append({
                "flavor": spec.label,
                "flavors": [spec_dtype, primary_layout, str(spec.file_type or "safetensors").strip().lower()],
                "display_label": spec.label,
                "artifacts": spec_artifacts,
                "snapshot_manifest": spec_manifest,
            })

    # Lineage: pin the upstream as parent if known.
    parent_repo = ""
    parent_checkpoint_id = ""
    src_ref = str(source_identity.source_ref or "").strip()
    if src_ref:
        parent_repo = src_ref
        parent_checkpoint_id = str(source_version_id or "").strip()

    metadata: dict[str, Any] = {}
    metadata.update(ingested.metadata or {})
    metadata["destination_repo"] = destination_repo
    metadata["checkpoint_flavors"] = commit_checkpoint_flavors
    if failed_flavors:
        metadata["failed_flavors"] = failed_flavors
        metadata["failed_flavor_count"] = str(len(failed_flavors))

    publish_fn = getattr(ctx, "publish_repo_revision", None)
    if not callable(publish_fn):
        metadata["publish_skipped_reason"] = "publish_repo_revision_missing"
        result = ConversionOutput(weights=tensors_with(ingested.weights, local_path=None), metadata=metadata)
        if emit_stage is not None:
            emit_stage("clone.completed", 1.0)
        return FinalizeCloneResult(output=result, published_version_id="")

    publish_kwargs: dict[str, Any] = {
        "destination_repo": destination_repo,
        "artifact_refs": [str(art.get("ref") or art.get("path") or "") for art in promoted_artifacts if art.get("ref") or art.get("path")],
        "metadata": metadata,
        "create_if_missing": True,
        "source_repo": parent_repo,
        "source_version_id": parent_checkpoint_id,
        "snapshot_manifest": snapshot_manifest,
        "relationship_kind": "import",
        "auto_create_external_parent": True,
        "merge_with_existing": not overwrite_repo,
    }
    if destination_repo_tags:
        publish_kwargs["destination_repo_tags"] = destination_repo_tags

    publish_result = publish_fn(**publish_kwargs)
    if not isinstance(publish_result, dict):
        publish_result = {"ok": True}
    outputs = [
        str(v or "").strip().lower()
        for v in list((publish_result or {}).get("output_versions") or [])
        if str(v or "").strip()
    ]
    published_version_id = outputs[0] if outputs else ""
    if published_version_id:
        metadata["published_version_id"] = published_version_id

    result = ConversionOutput(weights=tensors_with(ingested.weights, local_path=None), metadata=metadata)
    if emit_stage is not None:
        emit_stage("clone.completed", 1.0)
    return FinalizeCloneResult(output=result, published_version_id=published_version_id)


def _resolve_source_identity(
    *,
    provider: str,
    source_ref: str,
    source_revision: str | None,
    hf_token: str = "",
    civitai_model_version_id: int | None = None,
    civitai_file_id: int | None = None,
    source_metadata_overrides: dict[str, str] | None = None,
) -> SourceIdentity:
    provider_norm = str(provider or "").strip().lower()
    source_ref_norm = normalize_source_ref(source_ref)
    source_revision_norm = str(source_revision or "").strip()
    dedupe_supported = False
    civitai_model_version_norm = int(civitai_model_version_id or 0)
    civitai_file_norm = int(civitai_file_id or 0)
    source_metadata: dict[str, str] = {}
    resolved_civitai_identity: CivitaiResolvedIdentity | None = None
    override_expected_sha = ""
    override_expected_size: int | None = None

    if provider_norm == "civitai" and civitai_model_version_norm > 0:
        resolved_civitai_identity = resolve_civitai_source_identity(
            civitai_model_version_norm,
            civitai_file_id=(civitai_file_norm or None),
        )
        source_ref_norm = resolved_civitai_identity.source_ref
        source_revision_norm = resolved_civitai_identity.source_revision
        dedupe_supported = bool(source_revision_norm)
        source_metadata = {
            "source_kind": "civitai_model_version",
            "civitai_model_version_id": str(resolved_civitai_identity.model_version_id),
            "civitai_model_id": str(resolved_civitai_identity.model_id),
            "civitai_base_model": str(resolved_civitai_identity.base_model),
            "civitai_base_model_type": str(resolved_civitai_identity.base_model_type),
            "civitai_air": str(resolved_civitai_identity.air),
            "civitai_source_manifest_sha256": str(resolved_civitai_identity.source_manifest_sha256),
            "civitai_selected_file_count": str(len(resolved_civitai_identity.selected_files)),
            "civitai_file_fingerprints_json": json.dumps(
                dict(resolved_civitai_identity.file_fingerprints),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        if resolved_civitai_identity.selected_file_id is not None:
            source_metadata["civitai_file_id"] = str(int(resolved_civitai_identity.selected_file_id))
        selected_manifest = [
            {
                "file_id": int(item.file_id),
                "path": str(item.rel_path),
                "name": str(item.name),
                "size_bytes": int(item.size_bytes or 0),
                "primary": bool(item.primary),
                "fingerprint": str(item.fingerprint),
                "hashes": dict(item.hashes),
            }
            for item in list(resolved_civitai_identity.selected_files)
        ]
        source_metadata["civitai_selected_files_json"] = json.dumps(
            selected_manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
        source_metadata["civitai_pipeline_hint"] = str(resolved_civitai_identity.pipeline_hint)
        source_metadata["civitai_model_family_hint"] = str(resolved_civitai_identity.model_family)
        source_metadata["civitai_model_family_variant_hint"] = str(resolved_civitai_identity.model_family_variant)

    # Tenant-owned source identity normalization stays in endpoint code.
    if provider_norm == "huggingface" and not _is_http_source(source_ref_norm):
        repo_id, resolved_revision = resolve_huggingface_source_identity(
            source_ref_norm,
            source_revision=source_revision_norm or None,
            hf_token=hf_token,
        )
        source_ref_norm = str(repo_id).strip()
        source_revision_norm = str(resolved_revision).strip()
        dedupe_supported = source_revision_norm != ""

    if source_metadata_overrides:
        override_expected_sha = _normalize_sha256(source_metadata_overrides.get("source_expected_sha256"))
        override_expected_size = _parse_positive_int(source_metadata_overrides.get("source_expected_size_bytes"))
        for key, value in source_metadata_overrides.items():
            k = str(key or "").strip()
            if k == "":
                continue
            if k in {"source_expected_sha256", "source_expected_size_bytes"}:
                continue
            source_metadata[k] = str(value or "")

    if _is_http_source(source_ref_norm) and override_expected_sha != "":
        source_revision_norm = f"sha256:{override_expected_sha}"
        dedupe_supported = True
        source_metadata["source_expected_sha256"] = override_expected_sha
        if isinstance(override_expected_size, int) and override_expected_size > 0:
            source_metadata["source_expected_size_bytes"] = str(override_expected_size)

    return SourceIdentity(
        provider=provider_norm,
        source_ref=source_ref_norm,
        source_revision=source_revision_norm,
        identity_hash=_compute_identity_hash(
            provider=provider_norm,
            source_ref=source_ref_norm,
            source_revision=source_revision_norm,
        ),
        dedupe_supported=dedupe_supported,
        civitai_model_version_id=(civitai_model_version_norm if civitai_model_version_norm > 0 else None),
        civitai_file_id=(civitai_file_norm if civitai_file_norm > 0 else None),
        source_metadata=source_metadata,
        resolved_civitai_identity=resolved_civitai_identity,
    )




# Skip files under 100 MB — small configs, embeddings, etc. not worth quantizing.
_MIN_QUANTIZE_BYTES = 100 * 1024 * 1024


def _component_name_from_rel_path(rel_path: str) -> str:
    """Extract the top-level component name (e.g. `"transformer"`) from a
    relative path like `transformer/model.safetensors`. Returns empty string
    if the path has no directory prefix (which can happen for flat sources).
    """
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    return parts[0].lower() if parts else ""


def _build_conversion_jobs(
    *,
    output_specs: list[OutputSpec],
    ingested: ConversionOutput,
    component_groups: dict[str, Any],
    all_weight_files: list[Any],
    quantize_components: list[str] | None,
    is_passthrough_spec: Callable[[OutputSpec], bool],
) -> list[dict[str, Any]]:
    """Build the list of per-component conversion jobs.

    One job per weight component that needs a non-passthrough conversion.
    Three code paths, in preference order:

    1. Diffusers-layout sources with per-component groups (HF clones): one job
       per component; sharded components emit a single index-anchored job
       whose `snapshot_rel_path` points at the de-sharded output location.
    2. Flat weight-file lists without component grouping (fallback): one job
       per file that meets the minimum-size threshold.
    3. Single-file sources (Civitai): one job for the primary artifact with
       `component="primary"` and no snapshot prefix.

    Each job is a dict with the shape consumed by `_apply_save_format`:
    `{component, weights, extra_shards, shard_rel_paths, weights_rel_path,
    snapshot_rel_path, size_bytes}`.
    """
    target_components = list(quantize_components or ["transformer", "text_encoder"])
    use_all = "all" in target_components
    need_conversion_jobs = any(not is_passthrough_spec(spec) for spec in output_specs)
    conversion_jobs: list[dict[str, Any]] = []

    if not need_conversion_jobs:
        return conversion_jobs

    if component_groups:
        for component, group in component_groups.items():
            if not (use_all or component in target_components):
                continue
            shards = group.get("shards") or []
            if not shards:
                continue
            total_size = sum(int(s[2]) for s in shards)
            if total_size < _MIN_QUANTIZE_BYTES:
                continue
            is_sharded = (
                bool(group.get("is_sharded"))
                and len(shards) > 1
                and group.get("index_tensors") is not None
            )
            if is_sharded:
                index_tensors = group["index_tensors"]
                index_rel_path = str(group.get("index_rel_path") or "")
                idx_name, snapshot_rel = _derive_sharded_snapshot_path(index_rel_path, component)
                conversion_jobs.append({
                    "component": component,
                    "weights": index_tensors,
                    "extra_shards": [s[0] for s in shards],
                    # Shard basenames are passed to `_apply_save_format` to name
                    # the staged shard symlinks under the tempdir; they are NOT
                    # the final snapshot paths (the output is de-sharded).
                    "shard_rel_paths": [Path(s[1]).name for s in shards],
                    "weights_rel_path": idx_name,
                    "snapshot_rel_path": snapshot_rel,
                    "size_bytes": total_size,
                })
            else:
                for saved_tensors, rel_path, file_size in shards:
                    if file_size < _MIN_QUANTIZE_BYTES:
                        continue
                    conversion_jobs.append({
                        "component": component,
                        "weights": saved_tensors,
                        "extra_shards": [],
                        "shard_rel_paths": [],
                        # `weights_rel_path` is the basename for staging; the
                        # full `snapshot_rel_path` places the converted output
                        # at `<component>/<file>` in the published snapshot.
                        "weights_rel_path": Path(rel_path).name,
                        "snapshot_rel_path": rel_path,
                        "size_bytes": file_size,
                    })
        return conversion_jobs

    if all_weight_files:
        for saved_tensors, rel_path, file_size in all_weight_files:
            component = _component_name_from_rel_path(rel_path)
            if (use_all or component in target_components) and file_size >= _MIN_QUANTIZE_BYTES:
                conversion_jobs.append({
                    "component": component,
                    "weights": saved_tensors,
                    "extra_shards": [],
                    "shard_rel_paths": [],
                    "weights_rel_path": Path(rel_path).name,
                    "snapshot_rel_path": rel_path,
                    "size_bytes": file_size,
                })
        return conversion_jobs

    # Civitai / single-file path: convert the primary artifact.
    conversion_jobs.append({
        "component": "primary",
        "weights": ingested.weights,
        "extra_shards": [],
        "shard_rel_paths": [],
        "weights_rel_path": None,
        "snapshot_rel_path": None,
        "size_bytes": int(ingested.weights.size_bytes or 0),
    })
    return conversion_jobs


def _derive_sharded_snapshot_path(index_rel_path: str, component: str) -> tuple[str, str]:
    """Derive the index basename and snapshot path for a sharded component.

    `streaming_dtype_cast` de-shards the conversion input, so the canonical
    output path is a single `<component>/<stem>.safetensors` (the `.index.json`
    suffix of the source's index is stripped). This is the inverse of
    Bug 1: the earlier code stashed only the index basename on the
    `conversion_jobs` entry, which caused the manifest builder to place the
    index JSON at the snapshot root (`model.safetensors.index.json`) instead
    of the expected `<component>/<stem>.safetensors` path.

    Returns `(idx_name, snapshot_rel)` where:
      - `idx_name` is the basename of the index file (used to stage the
        conversion input; may still be an `.index.json` filename).
      - `snapshot_rel` is the component-prefixed full path the converted
        single-file output should land at in the published snapshot.
    """
    idx_name = Path(index_rel_path).name if index_rel_path else "model.safetensors.index.json"
    if idx_name.endswith(".safetensors.index.json"):
        deshard_name = idx_name[: -len(".index.json")]
    else:
        deshard_name = "model.safetensors"
    idx_parent = str(Path(index_rel_path).parent).replace("\\", "/") if index_rel_path else component
    if idx_parent and idx_parent != ".":
        snapshot_rel = f"{idx_parent}/{deshard_name}"
    else:
        snapshot_rel = deshard_name
    return idx_name, snapshot_rel


def _compute_stale_source_index_paths(save_format_outputs: list[dict[str, Any]]) -> set[str]:
    """Return the set of source-side `.safetensors.index.json` snapshot paths
    that must be dropped from the manifest because their component was
    converted to a single de-sharded `.safetensors` file.

    For every converted output whose `snapshot_rel_path` is
    `<component>/<name>.safetensors`, the corresponding stale source index is
    `<component>/<name>.safetensors.index.json`. Diffusers prefers an index.json
    when both exist and will then attempt sharded loading against shard paths
    that don't exist in our snapshot — so we drop the source index up front.
    """
    stale: set[str] = set()
    for entry in save_format_outputs:
        snap = str(entry.get("snapshot_rel_path") or "").strip()
        if snap.endswith(".safetensors") and not snap.endswith(".index.json"):
            stale.add(snap + ".index.json")
    return stale


def _manifest_entry_path(weights_ref: str, metadata: dict[str, Any]) -> str:
    source_filename = str(metadata.get("source_filename") or "").strip()
    if source_filename:
        return source_filename
    ref = str(weights_ref or "").strip()
    if ref:
        return ref.rsplit("/", 1)[-1] or ref
    return "weights.bin"


def _weights_blake3_digest(weights: Any) -> str:
    """Return a canonical `blake3:<hex>` digest for a Tensors-like object.

    Tensorhub's `blobs`/`blob_reverse_lookup` tables are keyed on blake3 because
    that is what the CAS upload flow verified against. Using sha256 (as an
    older iteration of this helper did) fails the commit-time binding check
    silently — the variant row is never written. Return "" only when the
    upload path didn't populate any digest at all.
    """
    bd = str(getattr(weights, "blob_digest", "") or "").strip()
    if bd:
        return bd if ":" in bd else f"blake3:{bd}"
    b3 = str(getattr(weights, "blake3", "") or "").strip()
    if b3:
        return b3 if ":" in b3 else f"blake3:{b3}"
    return ""


def _manifest_entry_for_weights(
    weights: Any,
    *,
    path_override: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    digest = _weights_blake3_digest(weights)
    if digest == "":
        return None
    ref = str(getattr(weights, "ref", "") or "").strip()
    if path_override and str(path_override).strip():
        path = str(path_override).strip()
    elif ref:
        path = ref.rsplit("/", 1)[-1] or ref
    else:
        path = _manifest_entry_path(ref, metadata)
    entry: dict[str, Any] = {
        "path": path,
        "digest": digest,
    }
    size_bytes = int(getattr(weights, "size_bytes", 0) or 0)
    if size_bytes > 0:
        entry["size_bytes"] = size_bytes
    return entry


def _log_manifest_debug(msg: str) -> None:
    """Single-line debug for manifest assembly; visible in worker logs."""
    import sys
    print(f"[clone_pipeline.snapshot_manifest] {msg}", file=sys.stderr, flush=True)


def _build_snapshot_manifest(
    output: ConversionOutput,
    snapshot_digest: str,
    metadata: dict[str, Any],
    *,
    extra_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the snapshot manifest that `persistSnapshotManifest` will expand into
    `tensorhub.snapshots` + `tensorhub.blobs` + `tensorhub.blob_reverse_lookup`.

    Must include every blob that any `checkpoint_flavors[i].artifacts[j]` will
    reference — otherwise `validateSnapshotArtifacts` rejects the variant at
    upload-complete time (see `tensorhub/internal/api/repo_job_presigned.go`).

    Entries are keyed on blake3 digest to match the CAS storage. Duplicate
    digests (e.g. dedup across save_format outputs) are de-duped by digest.
    """
    entries: list[dict[str, Any]] = []
    seen_digests: set[str] = set()

    skipped_no_digest = 0
    skipped_dupe = 0

    def _append(entry: dict[str, Any] | None) -> None:
        nonlocal skipped_no_digest, skipped_dupe
        if entry is None:
            skipped_no_digest += 1
            return
        digest = str(entry.get("digest") or "").strip()
        if digest == "":
            skipped_no_digest += 1
            return
        if digest in seen_digests:
            skipped_dupe += 1
            return
        seen_digests.add(digest)
        entries.append(entry)

    _append(_manifest_entry_for_weights(
        output.weights,
        path_override=_manifest_entry_path(str(getattr(output.weights, "ref", "") or ""), metadata),
        metadata=metadata,
    ))

    extras_list = list(extra_outputs or [])
    for extra in extras_list:
        weights = extra.get("weights")
        if weights is None:
            continue
        bd = str(getattr(weights, "blob_digest", "") or "").strip()
        b3 = str(getattr(weights, "blake3", "") or "").strip()
        _log_manifest_debug(
            f"extra path={extra.get('path')!r} ref={getattr(weights, 'ref', '')!r} "
            f"blob_digest={bd!r} blake3={b3!r} size={getattr(weights, 'size_bytes', None)!r}"
        )
        _append(_manifest_entry_for_weights(
            weights,
            path_override=extra.get("path"),
            metadata=metadata,
        ))

    _log_manifest_debug(
        f"built: entries={len(entries)} extras_in={len(extras_list)} "
        f"skipped_no_digest={skipped_no_digest} skipped_dupe={skipped_dupe}"
    )

    return {
        "version": 1,
        "snapshot_digest": str(snapshot_digest or "").strip(),
        "entries": entries,
    }


def preflight_clone(
    ctx: RequestContext,
    *,
    provider: str,
    source_ref: str,
    source_revision: str | None,
    destination_repo: str,
) -> MirrorPreflight:
    _ = ctx
    _ = provider
    _ = source_ref
    _ = source_revision
    _ = destination_repo
    return MirrorPreflight(destination_exists=False, destination_metadata={}, noop=False, mirror_metadata={})


_FP8_PROFILE_TO_TORCH_DTYPE = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


def _stage_sharded_input(
    ctx: RequestContext,
    td: Path,
    primary: Path,
    *,
    weights_rel_path: str | None,
    extra_shards: list[Tensors],
    shard_rel_paths: list[str],
) -> Path:
    """Symlink the index.json + all shards into a single dir so streaming readers see them co-located."""
    from ._shared import require_local_weights
    import os
    if len(extra_shards) != len(shard_rel_paths):
        raise ValueError("extra_shards and shard_rel_paths must have matching length")
    shard_dir = td / "sharded-input"
    shard_dir.mkdir(parents=True, exist_ok=True)
    idx_name = (weights_rel_path or "").strip() or primary.name
    idx_dest = shard_dir / idx_name
    if not idx_dest.exists():
        os.symlink(primary, idx_dest)
    for shard_tensors, shard_name in zip(extra_shards, shard_rel_paths):
        bare = Path(shard_name).name
        if not bare:
            raise ValueError("shard_rel_paths entries must be non-empty filenames")
        dest = shard_dir / bare
        if not dest.exists():
            os.symlink(require_local_weights(shard_tensors), dest)
    return idx_dest


def _derive_output_stem(
    *, snapshot_rel_path: str | None, weights_rel_path: str | None, fallback: str
) -> tuple[str, str]:
    """Derive ``(component, stem)`` for a converted output.

    HuggingFace/diffusers convention: each component directory (``transformer``,
    ``text_encoder``, ``vae``, ...) holds a ``<stem>.safetensors`` (or its
    ``<stem>-00001-of-NNN.safetensors`` + ``<stem>.safetensors.index.json``
    shards). We want our conversion outputs to use the SAME stem as the source
    so e.g. a ``transformer/diffusion_pytorch_model.safetensors`` input still
    lands at ``transformer/diffusion_pytorch_model-<N>-of-<NN>.safetensors``
    after a dtype cast — not at ``weights-bf16-<N>-of-<NN>.safetensors``.

    Returns ``(component, stem)`` where either may be ``""`` for the Civitai
    single-file case (caller falls back to ``fallback``).
    """
    snap = (snapshot_rel_path or "").strip().replace("\\", "/").lstrip("/")
    if snap:
        parts = snap.rsplit("/", 1)
        if len(parts) == 2:
            component, filename = parts
        else:
            component, filename = "", parts[0]
        stem = filename
        for suffix in (".safetensors.index.json", ".safetensors"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem == "":
            stem = fallback
        return component, stem
    rel = (weights_rel_path or "").strip()
    if rel:
        stem = Path(rel).name
        for suffix in (".safetensors.index.json", ".safetensors"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem != "":
            return "", stem
    return "", fallback


def _apply_save_format(
    ctx: RequestContext,
    *,
    weights: Tensors,
    save_format: str,
    source_repo_dir: str | None = None,
    source_revision: str | None = None,
    extra_shards: list[Tensors] | None = None,
    shard_rel_paths: list[str] | None = None,
    weights_rel_path: str | None = None,
    snapshot_rel_path: str | None = None,
) -> ConversionOutput:
    """Apply one conversion (bf16 / flashpack / fp8 / nvfp4 / gguf) to local weights.

    Calls gen_worker.conversion streaming primitives directly rather than
    dispatching through the @training_function tenant path (those expect
    a materialized tensorhub snapshot; clone_pipeline works from a local
    Tensors that's already been ingested from an external URL).
    """
    from ._shared import require_local_weights
    from gen_worker.conversion.safetensors_io import (
        materialize_safetensors_input,
        persist_safetensors_output,
    )
    from ._flashpack import convert_safetensors_to_flashpack
    from gen_worker.conversion.streaming_primitives import (
        streaming_dtype_cast,
        streaming_nvfp4_quantize,
    )

    value = (save_format or "").strip().lower()
    parts = value.split(":")
    tag = parts[0]
    inp = require_local_weights(weights)
    shards = list(extra_shards or [])
    shard_names = list(shard_rel_paths or [])

    def _output_ref(*, ext: str = ".safetensors", fallback_stem: str) -> tuple[str, str]:
        """Compute ``(output_ref, shard_prefix)`` for one conversion.

        Shard filenames use the source stem so the published paths match
        HF/diffusers convention:
          * source ``transformer/diffusion_pytorch_model.safetensors`` →
            single shard published at
            ``transformer/diffusion_pytorch_model.safetensors``;
          * shardable → ``transformer/diffusion_pytorch_model-00001-of-NN.safetensors``
            + ``transformer/diffusion_pytorch_model.safetensors.index.json``.
        """
        component, stem = _derive_output_stem(
            snapshot_rel_path=snapshot_rel_path,
            weights_rel_path=weights_rel_path,
            fallback=fallback_stem,
        )
        suffix = ext if ext.startswith(".") else f".{ext}"
        if component:
            ref = f"jobs/{ctx.request_id}/outputs/{component}/{stem}{suffix}"
        else:
            ref = default_output_ref(ctx, stem, ext=ext)
        return ref, stem

    td = Path(tempfile.mkdtemp(prefix=f"clone-{ctx.request_id}-{tag}-"))
    try:
        # Materialize input into a safetensors-file path the streaming readers can open.
        if shards:
            prepared = _stage_sharded_input(
                ctx, td, inp,
                weights_rel_path=weights_rel_path,
                extra_shards=shards,
                shard_rel_paths=shard_names,
            )
            sharded_meta = {"input_sharded": "1", "input_shard_count": str(len(shards))}
        else:
            prepared, sharded_meta = materialize_safetensors_input(inp, td)

        if tag in {"bf16", "fp16", "fp32"}:
            import torch
            target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[tag]
            ref, shard_prefix = _output_ref(fallback_stem=f"weights-{tag}")
            shard_dir = td / "shards"
            result = streaming_dtype_cast(
                prepared, shard_dir, target_dtype=target_dtype, shard_prefix=shard_prefix,
            )
            saved, additional, sharding_meta = persist_safetensors_output(
                ctx,
                shard_paths=result["output_paths"],
                index_path=result["index_path"],
                output_ref=ref,
            )
            return ConversionOutput(
                weights=saved,
                metadata={**sharded_meta, **sharding_meta},
                additional_artifacts=additional,
            )

        if tag == "flashpack":
            ref, _stem = _output_ref(ext=".flashpack", fallback_stem="weights-flashpack")
            converted = td / "converted.flashpack"
            convert_safetensors_to_flashpack(prepared, converted, target_dtype="preserve")
            saved = ctx.save_checkpoint(ref, str(converted), format="flashpack")
            return ConversionOutput(weights=saved, metadata={"output_format": "flashpack"})

        if (
            tag in {"fp8", "int8", "int4", "nf4", "fp4"}
            or save_format.startswith("fp8:")
            or save_format.startswith("int4:")
            or save_format.startswith("int8:")
        ):
            # Inline weight-only quantization via torchao.
            # Loads the component as an HF model with a TorchAoConfig, runs the
            # quant pass, and re-shards through the streaming writer so the
            # output matches the rest of the clone snapshot. Calibrated quants
            # (int4:awq / int4:gptq) raise InlineConversionNotPossible upstream.
            from gen_worker.conversion.inline_convert import (
                InlineConversionNotPossible,
                run_inline_conversion,
            )
            ref, shard_prefix = _output_ref(fallback_stem=f"weights-{tag}")
            shard_dir = td / "shards"
            # Per-component scope: when called from the diffusers monolith
            # path (`_finalize_clone` builds one job per component), the
            # full repo dir IS a diffusers tree (has model_index.json), but
            # we only want to quantize THIS component. Derive the component
            # subdir from `weights_rel_path` and pass that as the scope so
            # `run_inline_conversion` takes the transformers path against
            # the component's config.json — not the whole-repo fan-out.
            scoped_repo_dir = Path(source_repo_dir) if source_repo_dir else None
            comp, _stem = _derive_output_stem(
                snapshot_rel_path=snapshot_rel_path,
                weights_rel_path=weights_rel_path,
                fallback="",
            )
            if scoped_repo_dir is not None and comp:
                comp_dir = scoped_repo_dir / comp
                if (comp_dir / "config.json").is_file():
                    scoped_repo_dir = comp_dir
            inline_result = run_inline_conversion(
                source_path=prepared,
                out_dir=shard_dir,
                target_dtype=save_format,
                target_file_type="safetensors",
                source_repo_dir=scoped_repo_dir,
                shard_prefix=shard_prefix,
            )
            # Inline torchao/bnb produce a directory of files (sharded
            # safetensors + tokenizer/config sidecars). For per-component
            # diffusers calls, only the safetensors counts as "the
            # component's weights" — the persist_safetensors_output path
            # expects a list of safetensors files. Filter accordingly.
            shard_paths = [
                p for p in inline_result.output_paths
                if p.suffix == ".safetensors"
                and not p.name.endswith(".safetensors.index.json")
            ]
            if not shard_paths:
                shard_paths = list(inline_result.output_paths)
            saved, additional, sharding_meta = persist_safetensors_output(
                ctx,
                shard_paths=shard_paths,
                index_path=inline_result.index_path,
                output_ref=ref,
            )
            meta = {**sharded_meta, **sharding_meta, **inline_result.attributes}
            return ConversionOutput(
                weights=saved,
                metadata=meta,
                additional_artifacts=additional,
            )

        if tag == "nvfp4":
            # nvfp4 needs a calibration dataset to be useful at inference; the
            # raw streaming primitive emits int8+scale sidecars that diffusers/
            # transformers loaders can't materialize. Refuse cleanly with
            # structured requirements; deployment-specific rendering belongs
            # outside gen-worker.
            from gen_worker.conversion.inline_convert import (
                InlineConversionNotPossible,
                deferred_conversion_requirement,
            )
            raise InlineConversionNotPossible(
                reason=(
                    "nvfp4 needs a calibration dataset; the clone path doesn't "
                    "run dataset-dependent quantization inline"
                ),
                target_dtype="nvfp4",
                deferred_requirement=deferred_conversion_requirement("nvfp4"),
            )

        if tag == "gguf":
            if len(parts) > 2:
                # Per-quant encoding may itself contain a colon (rare). Reassemble.
                encoding = ":".join(parts[1:])
            else:
                encoding = (parts[1] if len(parts) == 2 else "f16").strip().lower() or "f16"
            # Single inline-convert path handles both direct encodings (f16/
            # bf16/q8_0) and llama-quantize two-step targets (q4_k_m, q6_k, …).
            from gen_worker.conversion.inline_convert import run_inline_conversion
            inline_result = run_inline_conversion(
                source_path=prepared,
                out_dir=td / "gguf-out",
                target_dtype=encoding,
                target_file_type="gguf",
                source_repo_dir=Path(source_repo_dir) if source_repo_dir else None,
            )
            converted = inline_result.output_paths[0]
            ref = default_output_ref(ctx, f"weights-gguf-{encoding}", ext=".gguf")
            saved = ctx.save_checkpoint(ref, str(converted), format="gguf")
            meta: dict[str, str] = {"encoding": encoding, "output_format": "gguf"}
            meta.update(inline_result.attributes)
            return ConversionOutput(weights=saved, metadata=meta)

        raise ValueError(f"unsupported save format: {save_format}")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _run_layout_repackage(
    ctx: RequestContext,
    *,
    source_weights: Tensors,
    source_layout: str,
    target_layout: str,
    model_family: str,
    source_repo_dir: str | None,
) -> ConversionOutput:
    """Run singlefile↔diffusers layout transform against a local source dir.

    Uploads the result (directory for diffusers, single safetensors for
    singlefile) via ctx.save_checkpoint / a directory-tree walk.
    """
    from ._shared import require_local_weights

    td = Path(tempfile.mkdtemp(prefix=f"clone-repackage-{ctx.request_id}-"))
    try:
        if source_layout == "singlefile" and target_layout == "diffusers":
            primary = require_local_weights(source_weights)
            out_dir = td / "diffusers"
            singlefile_to_diffusers(primary, out_dir, model_family=model_family)
            # Upload every file in the diffusers tree under a common prefix.
            ref_root = f"jobs/{ctx.request_id}/outputs/weights-diffusers"
            primary_saved: Tensors | None = None
            additional: list[Any] = []
            for f in sorted(out_dir.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(out_dir).as_posix()
                saved = ctx.save_checkpoint(
                    f"{ref_root}/{rel}", str(f), format=f.suffix.lstrip(".") or "bin",
                )
                if primary_saved is None and rel.endswith(".safetensors"):
                    primary_saved = saved
                else:
                    additional.append(saved)
            if primary_saved is None:
                raise RuntimeError("singlefile_to_diffusers_emitted_no_safetensors")
            return ConversionOutput(
                weights=primary_saved,
                metadata={"file_layout": "diffusers", "model_family": model_family},
            )
        if source_layout == "diffusers" and target_layout == "singlefile":
            if not source_repo_dir:
                raise ValueError("diffusers→singlefile repackage requires source_repo_dir")
            out_file = td / "model.safetensors"
            diffusers_to_singlefile(Path(source_repo_dir), out_file, model_family=model_family)
            ref = default_output_ref(ctx, "weights-singlefile", ext=".safetensors")
            saved = ctx.save_checkpoint(ref, str(out_file), format="safetensors")
            return ConversionOutput(
                weights=saved,
                metadata={"file_layout": "singlefile", "model_family": model_family},
            )
        raise ValueError(
            f"unsupported repackage: {source_layout} -> {target_layout}"
        )
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _finalize_clone(
    ctx: RequestContext,
    *,
    source_identity: SourceIdentity,
    source_version_id: str | None,
    destination_repo: str,
    destination_owner: str,
    destination_repo_name: str,
    destination_repo_tags: list[str],
    target_layout: str,
    outputs: list[OutputSpec] | None = None,
    save_formats: list[str] | None = None,
    ingested: ConversionOutput,
    ingest_result: "IngestResult | None" = None,
    emit_stage: Any = None,
    quantize_components: list[str] | None = None,
    auto_publish_public: bool = False,
    overwrite_repo: bool = False,
) -> FinalizeCloneResult:
    if ingest_result is None:
        ingest_result = IngestResult()

    selected_layout = normalize_target_layout(target_layout)
    # Resolve the final `outputs` list: callers may send either `outputs`
    # (canonical) or legacy `save_formats` (translated here). Empty → default
    # to `[bf16 / diffusers / safetensors]`. Computed before the dispatch on
    # runtime_library so the publish-as-is path also gets per-spec inline
    # conversion (issue #73).
    if outputs is not None and len(list(outputs)) > 0:
        output_specs = normalize_outputs(outputs, target_layout_hint=selected_layout)
    else:
        legacy = list(save_formats or [])
        if legacy:
            output_specs = save_formats_to_outputs(legacy, target_layout=selected_layout)
            if not output_specs:
                output_specs = normalize_outputs([], target_layout_hint=selected_layout)
        else:
            output_specs = normalize_outputs([], target_layout_hint=selected_layout)

    # Dispatch on the classifier's runtime_library.
    # Non-diffusers kinds (transformers / peft / sentence-transformers / gguf
    # / diffusers-lora) take a thin "publish what we ingested" path that
    # skips layout-repackage and save-format conversion. Today's monolithic
    # _finalize_clone code below assumes everything is a diffusers tree.
    runtime_library = str((ingest_result.classifier_attrs or {}).get("runtime_library") or "").strip().lower()
    if runtime_library not in {"", "diffusers", "diffusers-single-file"}:
        # Reject `--save-formats` with non-default values when the destination
        # is not a diffusers repo. The default OutputSpec for a
        # transformers/peft/llama.cpp
        # source is "ingest as-is", which is what _finalize_publish_as_is
        # already does. Anything else (`fp8:e4m3`, `int4:awq`, …) means
        # the caller wants quantization, which clone won't do for these kinds.
        non_default_save_formats = [
            f
            for f in normalize_save_formats(save_formats or [])
            if f and f.lower() not in {"bf16", "fp16", "default"}
        ]
        if non_default_save_formats:
            raise ValueError(
                "save_formats with non-default values is only supported for "
                f"diffusers sources; got runtime_library={runtime_library!r} "
                f"and save_formats={non_default_save_formats!r}. "
                "Run a separate conversion job against the cloned destination "
                "for those dtypes."
            )
        return _finalize_publish_as_is(
            ctx,
            source_identity=source_identity,
            source_version_id=source_version_id,
            destination_repo=destination_repo,
            destination_repo_name=destination_repo_name,
            destination_repo_tags=destination_repo_tags,
            ingested=ingested,
            ingest_result=ingest_result,
            emit_stage=emit_stage,
            auto_publish_public=auto_publish_public,
            overwrite_repo=overwrite_repo,
            output_specs=output_specs,
        )
    # Internal save_format list derived from OutputSpecs for the existing
    # per-component conversion loop. One save_format per non-passthrough spec;
    # passthrough specs are handled separately (source weight upload).
    source_layout = str(ingested.metadata.get("source_layout") or "unknown").strip().lower()
    if source_layout not in {"singlefile", "diffusers"}:
        source_layout = "unknown"
    model_family = str(ingested.metadata.get("model_family") or "unknown").strip().lower() or "unknown"

    # Detect per-component source dtype (from ingest's dtype-filtered selection).
    # When all components share a dtype we can cleanly mark an output "passthrough".
    source_dtype_by_base: dict[str, str] = dict(
        ingest_result.source_dtype_by_component
    )
    distinct_source_dtypes = {str(v or "").strip().lower() for v in source_dtype_by_base.values() if v}
    homogeneous_source_dtype = next(iter(distinct_source_dtypes)) if len(distinct_source_dtypes) == 1 else ""

    def _is_passthrough_spec(spec: OutputSpec) -> bool:
        # Passthrough fires when the target output bytes would be
        # byte-for-byte identical to the source bytes. That holds when
        # dtype/file_type/file_layout all match the source. Size never
        # disqualifies — oversize safetensors get sharded by byte offset
        # at upload time (shard_safetensors_by_offset), still zero decode.
        if homogeneous_source_dtype == "":
            return False
        if spec.dtype != homogeneous_source_dtype:
            return False
        if spec.file_type != "safetensors":
            return False
        if spec.file_layout != source_layout:
            return False
        return True

    # Per-spec fan-out: save_format_outputs is indexed by output_spec.label
    # so the variant emission pass below can assemble each variant's artifact
    # list independently.
    save_format_outputs: list[dict[str, Any]] = []  # one per (component, output_spec) conversion
    passthrough_uploads: dict[str, list[tuple[Any, str]]] = {}  # spec.label -> [(Tensors, rel_path)]
    flavor_refs: list[str] = []
    # Failed flavors recorded during partial-success conversion. Each entry
    # carries reason plus optional structured deferred_requirement metadata.
    # Surfaces in the clone result so the caller can render a per-flavor
    # warning instead of either crashing the whole job or silently dropping
    # the failed dtype.
    failed_flavors: list[dict[str, Any]] = []
    failed_spec_labels: set[str] = set()

    # publish_artifact_refs tracks every CAS-bound ref that accumulates during
    # finalize (for the publish call's artifact_refs list). Only include
    # things that are actually uploaded — source weight tensors arrive
    # "local-only" from ingest so we don't reference them unless a passthrough
    # spec uploads them in this function.
    publish_artifact_refs: list[str] = []
    extra_source_refs_raw = str(ingested.metadata.get("source_artifact_refs") or "").strip()
    if extra_source_refs_raw != "":
        for raw_ref in extra_source_refs_raw.split(";"):
            ref = str(raw_ref or "").strip()
            if ref != "":
                publish_artifact_refs.append(ref)

    conversion_jobs = _build_conversion_jobs(
        output_specs=output_specs,
        ingested=ingested,
        component_groups=ingest_result.component_groups,
        all_weight_files=ingest_result.all_weight_files,
        quantize_components=quantize_components,
        is_passthrough_spec=_is_passthrough_spec,
    )

    # Per-OutputSpec execution:
    #   - passthrough specs upload the ingested source weight files as-is.
    #   - non-passthrough specs run `_apply_save_format` on each component
    #     with the spec translated back to the legacy save_format token.
    total_conversions = max(1, len(output_specs) * max(1, len(conversion_jobs)))
    conversion_idx = 0
    for spec in output_specs:
        if _is_passthrough_spec(spec):
            uploaded_for_spec: list[tuple[Any, str]] = []
            from gen_worker.conversion._sharding import MAX_SAFETENSORS_SHARD_BYTES
            from gen_worker.conversion.safetensors_io import shard_safetensors_by_offset
            # Upload every source weight file (they arrived local-only from ingest).
            # Files ≤ MAX_SAFETENSORS_SHARD_BYTES go up as-is. Oversize
            # safetensors get split via shard_safetensors_by_offset — raw byte
            # copy, no tensor decode, emits HF-canonical
            # `{base}-00001-of-NNNNN.safetensors` + `{base}.safetensors.index.json`
            # under the source's parent directory (preserving the component
            # path: `transformer/diffusion_pytorch_model.safetensors` →
            # `transformer/diffusion_pytorch_model-00001-of-00002.safetensors`).
            for saved_tensors, rel_path, file_size in ingest_result.all_weight_files:
                local_path = str(getattr(saved_tensors, "local_path", "") or "").strip()
                if local_path == "":
                    continue
                fmt = str(saved_tensors.format or Path(rel_path).suffix.lstrip(".") or "safetensors")
                up_ref = str(saved_tensors.ref or f"jobs/{ctx.request_id}/outputs/source-repo/{rel_path}")

                # Passthrough fast path for normal-sized files.
                needs_shard = (
                    fmt == "safetensors"
                    and int(file_size or 0) > int(MAX_SAFETENSORS_SHARD_BYTES)
                )
                if not needs_shard:
                    up = ctx.save_checkpoint(up_ref, local_path, format=fmt)
                    if str(up.local_path or "").strip() == "":
                        up = tensors_with(up, local_path=local_path)
                    uploaded_for_spec.append((up, rel_path))
                    if str(up.ref or "").strip():
                        publish_artifact_refs.append(str(up.ref).strip())
                    continue

                # Oversize safetensors: shard by byte offset, upload each
                # shard + the index.json. No decode; no combined intermediate.
                rel_path_obj = Path(rel_path)
                base_name = rel_path_obj.stem  # e.g. "diffusion_pytorch_model"
                shard_stage = Path(local_path).parent / f".__shard_stage__{base_name}"
                shard_stage.mkdir(parents=True, exist_ok=True)
                try:
                    shard_paths, index_path, _shard_sizes = shard_safetensors_by_offset(
                        Path(local_path),
                        shard_stage,
                        max_shard_bytes=MAX_SAFETENSORS_SHARD_BYTES,
                        shard_prefix=base_name,
                    )
                    rel_dir = rel_path_obj.parent
                    for shard_path in shard_paths:
                        shard_rel = str((rel_dir / shard_path.name).as_posix())
                        shard_ref = f"jobs/{ctx.request_id}/outputs/source-repo/{shard_rel}"
                        up = ctx.save_checkpoint(shard_ref, str(shard_path), format="safetensors")
                        if str(up.local_path or "").strip() == "":
                            up = tensors_with(up, local_path=str(shard_path))
                        uploaded_for_spec.append((up, shard_rel))
                        if str(up.ref or "").strip():
                            publish_artifact_refs.append(str(up.ref).strip())
                    # Only upload the index when we actually sharded into
                    # multiple files (planner may return a single-shard plan
                    # if the file happens to fit).
                    if len(shard_paths) > 1:
                        index_rel = str((rel_dir / index_path.name).as_posix())
                        index_ref = f"jobs/{ctx.request_id}/outputs/source-repo/{index_rel}"
                        idx_up = ctx.save_checkpoint(index_ref, str(index_path), format="json")
                        if str(idx_up.local_path or "").strip() == "":
                            idx_up = tensors_with(idx_up, local_path=str(index_path))
                        uploaded_for_spec.append((idx_up, index_rel))
                        if str(idx_up.ref or "").strip():
                            publish_artifact_refs.append(str(idx_up.ref).strip())
                finally:
                    # Leave shard files on disk so the stream upload's retry
                    # path can re-read; caller (worker-level temp dir cleanup)
                    # wipes the whole request tmp at end of request.
                    pass
            passthrough_uploads[spec.label] = uploaded_for_spec
            conversion_idx += max(1, len(conversion_jobs))
            emit_stage("clone.save_format.completed",
                0.60 + (0.25 * float(conversion_idx) / float(total_conversions)),
                {
                    "save_format": spec.label,
                    "component": "all",
                    "passthrough": True,
                },
            )
            continue

        legacy_fmt = output_spec_to_save_format(spec)
        if legacy_fmt == "":
            raise ValueError(
                f"output dtype={spec.dtype} file_layout={spec.file_layout} "
                f"file_type={spec.file_type} is not supported yet (no converter wired)"
            )
        # Partial-success: if any component conversion raises
        # InlineConversionNotPossible (calibrated quant, missing toolchain),
        # record the failure and skip the rest of THIS spec's components —
        # we never publish a half-converted flavor. Other specs continue.
        from gen_worker.conversion.inline_convert import InlineConversionNotPossible
        spec_failed_exc: InlineConversionNotPossible | None = None
        spec_outputs: list[tuple[dict[str, Any], ConversionOutput, float]] = []
        for job in conversion_jobs:
            fmt_started = time.monotonic()
            try:
                out = _apply_save_format(
                    ctx,
                    weights=job["weights"],
                    save_format=legacy_fmt,
                    source_repo_dir=ingest_result.source_repo_dir,
                    source_revision=source_identity.source_revision,
                    extra_shards=job.get("extra_shards") or [],
                    shard_rel_paths=job.get("shard_rel_paths") or [],
                    weights_rel_path=job.get("weights_rel_path"),
                    snapshot_rel_path=job.get("snapshot_rel_path"),
                )
            except InlineConversionNotPossible as exc:
                spec_failed_exc = exc
                break
            spec_outputs.append((job, out, fmt_started))
        if spec_failed_exc is not None:
            failed: dict[str, Any] = {
                "spec_label": spec.label,
                "dtype": spec.dtype,
                "file_type": spec.file_type,
                "reason": spec_failed_exc.reason,
            }
            if spec_failed_exc.deferred_requirement is not None:
                failed["deferred_requirement"] = spec_failed_exc.deferred_requirement.as_dict()
            failed_flavors.append(failed)
            failed_spec_labels.add(spec.label)
            conversion_idx += max(1, len(conversion_jobs))
            payload: dict[str, Any] = {
                "save_format": legacy_fmt,
                "output_spec": spec.label,
                "reason": spec_failed_exc.reason,
            }
            if spec_failed_exc.deferred_requirement is not None:
                payload["deferred_requirement"] = spec_failed_exc.deferred_requirement.as_dict()
            emit_stage(
                "clone.save_format.skipped",
                0.60 + (0.25 * float(conversion_idx) / float(total_conversions)),
                payload,
            )
            continue
        for job, out, fmt_started in spec_outputs:
            component = str(job["component"])
            if str(out.weights.ref or "").strip():
                publish_artifact_refs.append(str(out.weights.ref).strip())
            extra_flavor_refs_raw = str(out.metadata.get("source_artifact_refs") or "").strip()
            if extra_flavor_refs_raw != "":
                for raw_ref in extra_flavor_refs_raw.split(";"):
                    ref = str(raw_ref or "").strip()
                    if ref != "":
                        publish_artifact_refs.append(ref)
            flavor_refs.append(f"{spec.label}:{component}={out.weights.ref}")
            save_format_outputs.append({
                "spec_label": spec.label,
                "component": component,
                "save_format": legacy_fmt,
                "output": out,
                "weights_rel_path": job.get("weights_rel_path"),
                "snapshot_rel_path": job.get("snapshot_rel_path"),
            })
            elapsed = max(time.monotonic() - fmt_started, 0.0)
            size_bytes = int(out.weights.size_bytes or 0)
            conversion_idx += 1
            emit_stage("clone.save_format.completed",
                0.60 + (0.25 * float(conversion_idx) / float(total_conversions)),
                {
                    "save_format": legacy_fmt,
                    "output_spec": spec.label,
                    "component": component,
                    "elapsed_s": float(elapsed),
                    "avg_bytes_per_sec": float(size_bytes) / max(elapsed, 1e-6),
                },
            )

    # Families that support singlefile↔diffusers layout repackaging.
    _REPACKAGE_FAMILIES = {
        "stable-diffusion", "sd", "sd15", "sd1", "sd2", "sdxl",
        "flux", "pixart", "dit", "kandinsky", "playground",
    }

    layout_output: ConversionOutput | None = None
    repackaged = False
    repackage_toolchain = "none"
    if (
        source_layout in {"singlefile", "diffusers"}
        and selected_layout != source_layout
        and model_family in _REPACKAGE_FAMILIES
    ):
        emit_stage("clone.repackage.started",
            0.90,
            {
                "source_layout": source_layout,
                "target_layout": selected_layout,
                "model_family": model_family,
            },
        )
        layout_output = _run_layout_repackage(
            ctx,
            source_weights=ingested.weights,
            source_layout=source_layout,
            target_layout=selected_layout,
            model_family=model_family,
            source_repo_dir=ingest_result.source_repo_dir,
        )
        repackaged = True
        repackage_toolchain = f"{source_layout}_to_{selected_layout}:v1"
        emit_stage("clone.repackage.completed",
            0.94,
            {
                "source_layout": source_layout,
                "target_layout": selected_layout,
                "model_family": model_family,
                "repackage_toolchain": repackage_toolchain,
            },
        )
    emit_stage("clone.layout.completed",
        0.95,
        {
            "source_layout": source_layout,
            "target_layout": selected_layout,
            "model_family": model_family,
            "repackaged": bool(repackaged),
        },
    )

    output = layout_output if layout_output is not None else ingested
    # `output.weights` is local-only when `output is ingested` (post-refactor:
    # ingest no longer uploads source weights). Only append its ref to the
    # publish artifact refs when we actually have a CAS-bound blob, i.e. when
    # layout_output produced a repackaged file that did upload via
    # ctx.save_checkpoint.
    if layout_output is not None and str(output.weights.ref or "").strip():
        publish_artifact_refs.append(str(output.weights.ref).strip())
    if layout_output is not None:
        layout_refs_raw = str(layout_output.metadata.get("source_artifact_refs") or "").strip()
        if layout_refs_raw != "":
            for raw_ref in layout_refs_raw.split(";"):
                ref = str(raw_ref or "").strip()
                if ref != "":
                    publish_artifact_refs.append(ref)

    # `_internal_*` filter no longer needed — those values now flow via the
    # typed `IngestResult` sidecar instead of the metadata dict.
    metadata = dict(ingested.metadata)
    if layout_output is not None:
        metadata.update(layout_output.metadata)
    metadata.update({k: str(v) for k, v in dict(source_identity.source_metadata).items()})
    metadata.update(
        {
            "source_provider": source_identity.provider,
            "source_ref": source_identity.source_ref,
            "source_version_id": str(source_version_id or "").strip(),
            "source_revision": source_identity.source_revision,
            "source_hash": source_identity.identity_hash,
            "destination_repo": destination_repo,
            "destination_owner": destination_owner,
            "destination_repo_name": destination_repo_name,
            "destination_repo_tags": ",".join(destination_repo_tags),
            "mirror.mode": "mirror",
            "mirror.provider": source_identity.provider,
            "mirror.source_ref": source_identity.source_ref,
            "mirror.source_revision": source_identity.source_revision,
            "mirror.source_hash": source_identity.identity_hash,
            "source_layout": source_layout,
            "target_layout": selected_layout,
            "model_family": model_family,
            "repackage_toolchain": repackage_toolchain,
            "layout_policy": _LAYOUT_POLICY,
            "outputs": ",".join(spec.label for spec in output_specs),
            "source_dtype_preference": ",".join(
                str(v or "").strip().lower()
                for v in (ingest_result.source_dtype_preference)
            ),
            "source_dtype_by_component": ",".join(
                f"{k}={v}" for k, v in sorted(source_dtype_by_base.items())
            ),
        }
    )
    if flavor_refs:
        metadata["flavor_refs"] = ";".join(flavor_refs)

    dedup_refs: list[str] = []
    seen_refs: set[str] = set()
    for ref in publish_artifact_refs:
        clean = str(ref or "").strip()
        if clean == "" or clean in seen_refs:
            continue
        seen_refs.add(clean)
        dedup_refs.append(clean)

    # Build structured checkpoint_flavors for the TensorHub publish call.
    # The identity_hash must use algo:hex format for TensorHub lineage validation.
    identity_hash_raw = str(source_identity.identity_hash or "").strip().lower()
    version_id = f"sha256:{identity_hash_raw}" if identity_hash_raw and ":" not in identity_hash_raw else identity_hash_raw

    def _tensors_artifact(t: Any) -> dict[str, Any]:
        """Extract artifact dict from a Tensors object."""
        art: dict[str, Any] = {}
        digest = str(getattr(t, "blob_digest", "") or getattr(t, "blake3", "") or "").strip()
        if digest and ":" not in digest:
            digest = f"blake3:{digest}"
        if digest:
            art["digest"] = digest
        path = str(getattr(t, "ref", "") or "").strip()
        if path:
            art["path"] = path
        size = getattr(t, "size_bytes", None)
        if size is not None:
            art["size_bytes"] = int(size)
        domain = str(getattr(t, "blob_domain", "") or "private").strip()
        art["domain"] = domain
        return art

    commit_checkpoint_flavors: list[dict[str, Any]] = []

    # Non-weight files (configs/tokenizers/scheduler/model_index.json/README/
    # LICENSE) always belong to every emitted flavor so inference can load
    # the pipeline regardless of which dtype/layout/filetype was picked.
    # Source-side `.safetensors.index.json` files whose component had a
    # non-passthrough conversion become stale: the converted output is a
    # single de-sharded file, so the source's sharded index points at
    # `model-00001-of-0000N.safetensors` shard paths that don't exist in
    # the snapshot. Diffusers' `from_pretrained` prefers index.json over
    # the single file, tries sharded load, and dies. Drop those.
    stale_index_paths = _compute_stale_source_index_paths(save_format_outputs)
    non_weight_file_tensors: list[Any] = []
    for t, _rel, _sz in list(ingest_result.all_file_tensors):
        digest = str(getattr(t, "blob_digest", "") or getattr(t, "blake3", "") or "").strip()
        if digest == "":
            # Weight files were kept local-only earlier; they don't have a blob
            # digest unless the passthrough path uploaded them below.
            continue
        if _rel in stale_index_paths:
            continue
        non_weight_file_tensors.append(t)

    # One checkpoint flavor per OutputSpec. Artifacts for a flavor =
    # converted weights produced by that spec's conversions (or passthrough
    # source weights) + every non-weight file. Specs that failed inline
    # conversion are skipped here — they're surfaced via `failed_flavors`
    # in metadata so the caller can render per-flavor warnings.
    for spec in output_specs:
        if spec.label in failed_spec_labels:
            continue
        artifact_tensors: list[Any] = []
        if _is_passthrough_spec(spec):
            for tensors, _rel in passthrough_uploads.get(spec.label, []):
                artifact_tensors.append(tensors)
        else:
            for entry in save_format_outputs:
                if str(entry.get("spec_label") or "") != spec.label:
                    continue
                out_obj = entry.get("output")
                if out_obj is None:
                    continue
                artifact_tensors.append(out_obj.weights)
        artifact_tensors.extend(non_weight_file_tensors)

        artifacts: list[dict[str, Any]] = []
        seen_digests: set[str] = set()
        for t in artifact_tensors:
            art = _tensors_artifact(t)
            digest = str(art.get("digest") or "").strip()
            if digest == "" or digest in seen_digests:
                continue
            seen_digests.add(digest)
            artifacts.append(art)

        if not version_id or not artifacts:
            # Nothing uploaded for this spec — surface as a soft warning in
            # the metadata below instead of emitting an invalid row.
            continue

        commit_checkpoint_flavors.append({
            "flavor": spec.label,
            "flavors": [spec.dtype, spec.file_layout, spec.file_type],
            "display_label": spec.label,
            "artifacts": artifacts,
        })

    # Forward the full flavor list to publish_repo_revision. If for some reason
    # we produced nothing (no digests at all), still omit the key and let the
    # publish path fall back — but that's a bug worth surfacing in logs.
    if commit_checkpoint_flavors:
        metadata["checkpoint_flavors"] = commit_checkpoint_flavors

    # Surface partial-success failures (calibrated quants etc.) so callers can
    # render per-flavor warnings. Each entry carries the requested
    # dtype/file_type, the reason, and optional structured follow-up metadata.
    # Empty when every requested flavor landed cleanly.
    if failed_flavors:
        metadata["failed_flavors"] = failed_flavors
        metadata["failed_flavor_count"] = str(len(failed_flavors))

    publish_fn = getattr(ctx, "publish_repo_revision", None)
    if not callable(publish_fn):
        metadata["publish_skipped_reason"] = "publish_repo_revision_missing"
        result = ConversionOutput(weights=tensors_with(output.weights, local_path=None), metadata=metadata)
        emit_stage("clone.completed", 1.0)
        return FinalizeCloneResult(output=result, published_version_id="")

    # Build snapshot manifest for the publish. Entries must include every blob
    # referenced by `commit_checkpoint_flavors[i].artifacts` plus every file the
    # ingested repo produced (configs, tokenizers, schedulers) — otherwise the
    # commit-time `validateSnapshotArtifacts` check on Tensorhub rejects the
    # flavor, AND inference (e.g. `StableDiffusionPipeline.from_pretrained`)
    # later fails with `model_index.json not found`.
    snapshot_digest = version_id  # Align with the version_id used on variant rows.
    extra_manifest_outputs: list[dict[str, Any]] = []
    # Non-weight files uploaded during ingest (model_index.json, configs,
    # tokenizers, scheduler, README, LICENSE). Local-only weight Tensors are
    # also in this list but get skipped by `_build_snapshot_manifest`
    # because they have no blob_digest.
    # `stale_index_paths` was computed above when building
    # `non_weight_file_tensors`. Reuse the same set here so the manifest and
    # per-variant artifacts agree on which source index files to drop.
    all_file_tensors = ingest_result.all_file_tensors
    for tensors, rel_path, _size in list(all_file_tensors):
        if rel_path in stale_index_paths:
            continue
        extra_manifest_outputs.append({
            "weights": tensors,
            "path": rel_path,
        })
    # Passthrough uploads (source weights that a passthrough OutputSpec pushed
    # to CAS just now).
    for passthrough_list in passthrough_uploads.values():
        for tensors, rel_path in passthrough_list:
            extra_manifest_outputs.append({
                "weights": tensors,
                "path": rel_path,
            })
    # Converted save-format outputs (one entry per (output_spec, component)).
    # Use the FULL `snapshot_rel_path` (e.g. `text_encoder/model.fp16.safetensors`)
    # so the diffusers loader finds the file under the expected component
    # directory. `weights_rel_path` is only the basename (used to stage the
    # conversion input).
    for entry in save_format_outputs:
        out_obj = entry.get("output")
        if out_obj is None:
            continue
        full_path = entry.get("snapshot_rel_path")
        if not full_path:
            # Civitai / single-file path legitimately has no component prefix;
            # accept the basename in that one case. Every diffusers-layout
            # conversion must set `snapshot_rel_path` explicitly so we do not
            # silently publish bare basenames that diffusers can't resolve.
            if entry.get("component") == "primary":
                full_path = entry.get("weights_rel_path") or None
            else:
                raise RuntimeError(
                    f"_finalize_clone: conversion output missing snapshot_rel_path "
                    f"(component={entry.get('component')!r}, save_format={entry.get('save_format')!r}). "
                    f"This would publish at a bare basename and diffusers inference would fail."
                )
        # Sharded output handling: when `persist_safetensors_output` re-sharded
        # the conversion result (output > _MAX_SAFETENSORS_SHARD_BYTES or the
        # caller requested a specific shard count), `out_obj.weights` is the
        # `.index.json` and `out_obj.additional_artifacts` holds per-shard
        # Tensors. We emit one manifest entry per file under the component's
        # parent directory (`<component>/<rel_name>`); the primary index sits
        # at `<full_path>.index.json`.
        sharded_output = bool(getattr(out_obj, "additional_artifacts", None))
        if sharded_output:
            parent = str(Path(full_path or "").parent).replace("\\", "/")
            if parent == "." or parent == "":
                parent = ""
            index_path = str(full_path) + ".index.json" if full_path else None
            extra_manifest_outputs.append({
                "weights": out_obj.weights,
                "path": index_path,
            })
            for art in out_obj.additional_artifacts:
                art_rel = str(getattr(art, "rel_name", "") or "").strip()
                if art_rel == "":
                    continue
                # The `additional_artifacts` list from
                # `persist_safetensors_output` includes the index as its first
                # entry; skip that when we've already emitted it above to
                # avoid duplicate manifest entries at the same path.
                if art_rel.endswith(".index.json"):
                    continue
                art_full = f"{parent}/{art_rel}" if parent else art_rel
                extra_manifest_outputs.append({
                    "weights": art.tensors,
                    "path": art_full,
                })
        else:
            extra_manifest_outputs.append({
                "weights": out_obj.weights,
                "path": full_path or None,
            })
    snapshot_manifest = _build_snapshot_manifest(
        output,
        snapshot_digest,
        metadata,
        extra_outputs=extra_manifest_outputs,
    )

    # Prefix identity_hash for target_version_id.
    target_vid = ""
    if source_identity.dedupe_supported and identity_hash_raw:
        target_vid = version_id

    # HARD-CUT issue #14: imports land with lineage edge
    # external-sources/upstream@hf:<org>/<name> -> destination checkpoint,
    # relationship_kind='import'. The publish endpoint auto-creates the
    # placeholder parent when auto_create_external_parent=True.
    provider = (source_identity.provider or "").strip().lower()
    source_ref = (source_identity.source_ref or "").strip()
    if provider in ("huggingface", "hf"):
        parent_repo = "external-sources/upstream"
        parent_checkpoint_id = f"hf:{source_ref}"
    elif provider == "civitai":
        parent_repo = "external-sources/upstream"
        parent_checkpoint_id = f"civitai:{source_ref}"
    else:
        parent_repo = source_ref
        parent_checkpoint_id = str(source_version_id or "").strip()

    publish_kwargs: dict[str, Any] = {
        "destination_repo": destination_repo,
        "artifact_refs": dedup_refs,
        "metadata": metadata,
        "create_if_missing": True,
        "source_repo": parent_repo,
        "source_version_id": parent_checkpoint_id,
        "snapshot_manifest": snapshot_manifest,
        "relationship_kind": "import",
        "auto_create_external_parent": True,
        "merge_with_existing": not overwrite_repo,
    }
    if destination_repo_tags:
        publish_kwargs["destination_repo_tags"] = destination_repo_tags
    if target_vid:
        publish_kwargs["target_version_id"] = target_vid

    publish_result = publish_fn(**publish_kwargs)
    if not isinstance(publish_result, dict):
        publish_result = {"ok": True}

    job_id = str((publish_result or {}).get("job_id") or "").strip()
    if job_id:
        metadata["publish_job_id"] = job_id
    outputs = [
        str(v or "").strip().lower()
        for v in list((publish_result or {}).get("output_versions") or [])
        if str(v or "").strip()
    ]
    published_version_id = outputs[0] if outputs else ""

    result = ConversionOutput(weights=tensors_with(output.weights, local_path=None), metadata=metadata)
    emit_stage("clone.completed", 1.0)
    return FinalizeCloneResult(output=result, published_version_id=published_version_id)


def finalize_clone(
    ctx: RequestContext,
    *,
    source_identity: SourceIdentity,
    source_version_id: str | None,
    destination_repo: str,
    destination_owner: str,
    destination_repo_name: str,
    destination_repo_tags: list[str],
    target_layout: str,
    outputs: list[OutputSpec] | None = None,
    save_formats: list[str] | None = None,
    ingested: ConversionOutput,
    emit_stage: Any = None,
    quantize_components: list[str] | None = None,
) -> ConversionOutput:
    finalized = _finalize_clone(
        ctx,
        source_identity=source_identity,
        source_version_id=source_version_id,
        destination_repo=destination_repo,
        destination_owner=destination_owner,
        destination_repo_name=destination_repo_name,
        destination_repo_tags=destination_repo_tags,
        target_layout=target_layout,
        outputs=outputs,
        save_formats=save_formats,
        ingested=ingested,
        emit_stage=emit_stage,
        quantize_components=quantize_components,
    )
    return finalized.output


def run_clone(
    ctx: RequestContext,
    *,
    provider: str,
    source_ref: str,
    source_version_id: str | None,
    source_revision: str | None,
    civitai_model_version_id: int | None = None,
    civitai_file_id: int | None = None,
    source_metadata_overrides: dict[str, str] | None = None,
    destination_repo: str,
    destination_repo_tags: list[str],
    target_layout: str,
    source_layout_preference: str,
    source_dtype_preference: list[str] | None = None,
    outputs: list[OutputSpec] | None = None,
    save_formats: list[str] | None = None,
    output_ref: str | None = None,
    quantize_components: list[str] | None = None,
    auto_publish_public: bool = False,
    overwrite_repo: bool = False,
    gguf_quant: str | None = None,
) -> ConversionOutput:
    started_at = time.monotonic()
    stage_emit_state: dict[str, Any] = {
        "stage": "",
        "progress_pct": None,
        "last_emit_ts": 0.0,
    }
    download_stage_heartbeat_s = 5.0

    def emit_stage(stage: str, progress: float, extra: dict[str, Any] | None = None) -> None:
        now = time.monotonic()
        # Deduplicate noisy stage updates while preserving meaningful progress changes.
        progress_clamped = max(0.0, min(1.0, float(progress)))
        progress_pct = int(round(progress_clamped * 100.0))
        last_stage = str(stage_emit_state.get("stage") or "")
        last_progress_pct = stage_emit_state.get("progress_pct")
        last_emit_ts = float(stage_emit_state.get("last_emit_ts") or 0.0)

        if stage == last_stage and progress_pct == last_progress_pct:
            if stage != "clone.ingest.downloading":
                return
            if (now - last_emit_ts) < download_stage_heartbeat_s:
                return

        stage_emit_state["stage"] = stage
        stage_emit_state["progress_pct"] = progress_pct
        stage_emit_state["last_emit_ts"] = now

        payload: dict[str, Any] = {
            "stage": stage,
            "progress": float(progress_clamped),
            "elapsed_s": float(max(now - started_at, 0.0)),
        }
        if extra:
            payload.update(extra)
        if hasattr(ctx, "emit"):
            ctx.emit("conversion.clone_progress", payload)
        if hasattr(ctx, "progress"):
            ctx.progress(float(progress_clamped), stage=stage)

    emit_stage("clone.started", 0.05, {"provider": str(provider or "").strip().lower()})

    normalized_source = normalize_source_ref(source_ref)
    normalized_destination = normalize_destination_ref(destination_repo)
    destination_owner, normalized_destination_name = normalized_destination.split("/", 1)
    normalized_tags = normalize_destination_repo_tags(destination_repo_tags)
    normalized_layout = normalize_target_layout(target_layout)
    normalized_source_layout_preference = normalize_source_layout_preference(source_layout_preference)
    normalized_save_formats = normalize_save_formats(save_formats or [])
    normalized_source_dtype_preference = [
        str(p or "").strip().lower() for p in (source_dtype_preference or []) if str(p or "").strip()
    ]
    # Resolve the canonical OutputSpec list once at entry, so downstream calls
    # see a consistent view regardless of whether the client sent `outputs`
    # or legacy `save_formats`.
    if outputs is not None and len(list(outputs)) > 0:
        resolved_outputs = normalize_outputs(outputs, target_layout_hint=normalized_layout)
    elif normalized_save_formats:
        resolved_outputs = save_formats_to_outputs(normalized_save_formats, target_layout=normalized_layout)
        if not resolved_outputs:
            resolved_outputs = normalize_outputs([], target_layout_hint=normalized_layout)
    else:
        resolved_outputs = normalize_outputs([], target_layout_hint=normalized_layout)

    source_identity = _resolve_source_identity(
        provider=provider,
        source_ref=normalized_source,
        source_revision=source_revision,
        hf_token=ctx.hf_token,
        civitai_model_version_id=civitai_model_version_id,
        civitai_file_id=civitai_file_id,
        source_metadata_overrides=source_metadata_overrides,
    )
    emit_stage(
        "clone.source_identity_resolved",
        0.15,
        {
            "source_ref": source_identity.source_ref,
            "source_revision": source_identity.source_revision,
            "dedupe_supported": bool(source_identity.dedupe_supported),
            "civitai_model_version_id": source_identity.civitai_model_version_id,
            "civitai_file_id": source_identity.civitai_file_id,
        },
    )

    emit_stage("clone.ingest.mode_selected",
        0.17,
        {
            "ingest_mode": "auto",
            "source_layout_preference": normalized_source_layout_preference,
        },
    )

    dl_state = {"last_ts": time.monotonic(), "last_bytes": 0}

    def on_download_progress(bytes_written: int, total_bytes: int | None) -> None:
        now = time.monotonic()
        elapsed = max(now - started_at, 1e-6)
        delta_t = max(now - float(dl_state["last_ts"]), 1e-6)
        delta_b = max(0, int(bytes_written) - int(dl_state["last_bytes"]))
        avg_bps = float(bytes_written) / elapsed
        inst_bps = float(delta_b) / delta_t
        total_int = int(total_bytes) if total_bytes is not None else None
        download_ratio: float | None = None
        progress = 0.25
        if total_int is not None and total_int > 0:
            download_ratio = min(1.0, float(bytes_written) / float(total_int))
            progress += download_ratio * 0.30
        remaining_bytes = None
        if total_int is not None and total_int > 0:
            remaining_bytes = max(0, total_int - int(bytes_written))
        payload: dict[str, Any] = {
            "bytes_written": int(bytes_written),
            "bytes_total": total_int,
            "bytes_remaining": remaining_bytes,
            "avg_bytes_per_sec": float(avg_bps),
            "inst_bytes_per_sec": float(inst_bps),
            "download_bytes_downloaded": int(bytes_written),
            "download_bytes_total": total_int,
            "download_bytes_remaining": remaining_bytes,
            "download_bps_avg": float(avg_bps),
            "download_bps_inst": float(inst_bps),
        }
        if download_ratio is not None:
            progress_pct = int(round(download_ratio * 100.0))
            payload["stage_progress"] = float(download_ratio)
            payload["stage_progress_pct"] = progress_pct
            payload["download_progress"] = float(download_ratio)
            payload["download_progress_pct"] = progress_pct
        emit_stage(
            "clone.ingest.downloading",
            progress,
            payload,
        )
        dl_state["last_ts"] = now
        dl_state["last_bytes"] = int(bytes_written)

    def execute_miss() -> dict[str, Any]:
        source_expected_sha256 = str(source_identity.source_metadata.get("source_expected_sha256") or "").strip()
        source_expected_size_bytes = _parse_positive_int(
            source_identity.source_metadata.get("source_expected_size_bytes")
        )
        # Thread the requested concrete dtypes
        # from `resolved_outputs` so the HF downloader can multi-select
        # when N>1. Single-element / unset falls back to single-checkpoint
        # behavior. Drop entries with empty `dtype` (defaults from the
        # `[]` no-outputs case).
        dtype_outputs_pref = [
            str(s.dtype or "").strip().lower()
            for s in resolved_outputs
            if str(s.dtype or "").strip()
        ]
        ingested, ingest_result = ingest_from_source(
            ctx,
            provider=source_identity.provider,
            source_ref=source_identity.source_ref,
            source_revision=source_identity.source_revision,
            source_layout_preference=normalized_source_layout_preference,
            source_dtype_preference=list(normalized_source_dtype_preference),
            source_expected_sha256=(source_expected_sha256 if source_expected_sha256 != "" else None),
            source_expected_size_bytes=source_expected_size_bytes,
            civitai_model_version_id=source_identity.civitai_model_version_id,
            civitai_file_id=source_identity.civitai_file_id,
            resolved_civitai_identity=source_identity.resolved_civitai_identity,
            output_ref=output_ref,
            progress_callback=on_download_progress,
            gguf_quant=gguf_quant,
            dtype_outputs=dtype_outputs_pref,
        )
        emit_stage("clone.layout.detected",
            0.58,
            {
                "source_layout": str(ingested.metadata.get("source_layout") or "unknown"),
                "model_family": str(ingested.metadata.get("model_family") or "unknown"),
                "target_layout": normalized_layout,
            },
        )
        emit_stage("clone.ingest.completed", 0.60)

        finalized = _finalize_clone(
            ctx,
            source_identity=source_identity,
            source_version_id=source_version_id,
            destination_repo=normalized_destination,
            destination_owner=destination_owner,
            destination_repo_name=normalized_destination_name,
            destination_repo_tags=normalized_tags,
            target_layout=normalized_layout,
            outputs=list(resolved_outputs),
            save_formats=None,
            ingested=ingested,
            ingest_result=ingest_result,
            emit_stage=emit_stage,
            quantize_components=quantize_components,
            auto_publish_public=auto_publish_public,
            overwrite_repo=overwrite_repo,
        )
        return {
            "output": finalized.output,
            "version_id": finalized.published_version_id,
            "primary_artifact_ref": str(finalized.output.weights.ref or "").strip(),
            "primary_artifact_format": str(finalized.output.weights.format or "").strip(),
        }

    emit_stage("clone.dedupe.skipped",
        0.18,
        {"reason": "removed"},
    )
    return execute_miss()["output"]


def maybe_noop(
    preflight: MirrorPreflight,
    *,
    destination_owner: str,
    destination_repo_name: str,
) -> ConversionOutput | None:
    _ = preflight
    _ = destination_owner
    _ = destination_repo_name
    return None


__all__ = [
    "MirrorPreflight",
    "SourceIdentity",
    "compute_source_hash",
    "finalize_clone",
    "maybe_noop",
    "normalize_destination_owner",
    "normalize_destination_repo_name",
    "normalize_destination_repo_tags",
    "normalize_save_formats",
    "normalize_source_ref",
    "normalize_source_layout_preference",
    "normalize_target_layout",
    "preflight_clone",
    "run_clone",
]


# ---------------------------------------------------------------------------
# Intermediate API stubs (issue #20 — `gen_worker.clone` six-method surface).
# These expose library-internal pieces of the clone pipeline as public
# entry points for tenants who need custom flows. Currently stubs pending
# refactor of the pipeline to expose these cleanly — `from_huggingface` /
# `from_civitai` are the documented stable surface.
# ---------------------------------------------------------------------------


def _fetch_hf_snapshot_only(
    *,
    source_ref: str,
    revision: str | None = None,
    dest_dir: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> str:
    raise NotImplementedError(
        "fetch_huggingface_snapshot is pending extraction from pipeline internals; "
        "use clone.from_huggingface(ctx, payload) for the full clone flow."
    )


def _fetch_civitai_file_only(*, model_version_id: int, file_id: int, dest_dir: str) -> str:
    raise NotImplementedError(
        "fetch_civitai_file is pending extraction from pipeline internals; "
        "use clone.from_civitai(ctx, payload) for the full clone flow."
    )


def _parse_hf_metadata_only(*, source_ref: str, revision: str | None = None) -> dict[str, Any]:
    raise NotImplementedError(
        "parse_huggingface_metadata is pending extraction; use clone.from_huggingface for now."
    )


def _parse_civitai_metadata_only(*, model_version_id: int) -> dict[str, Any]:
    raise NotImplementedError(
        "parse_civitai_metadata is pending extraction; use clone.from_civitai for now."
    )
