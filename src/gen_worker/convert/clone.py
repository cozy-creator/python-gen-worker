"""Clone: mirror an external checkpoint into Tensorhub, optionally converting.

download → (repackage) → (cast / quant / gguf) → ONE finalize:
each requested output flavor becomes one local file tree and one Tensorhub
commit (``POST /commits`` + part PUTs + finalize). ``overwrite_repo`` maps to
``mode="replace"`` — no enumerate-prior-and-delete.

Entry points: :func:`from_huggingface` / :func:`from_civitai` (payload-shaped)
and :func:`run_clone` (keyword-explicit).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .hub import HubClient, files_from_tree
from .ingest import IngestedSource, ingest_civitai, ingest_huggingface
from .writer import (
    FP8_DEFAULT_COMPONENTS,
    MAX_SAFETENSORS_SHARD_BYTES,
    copy_non_weight_files,
    shard_safetensors_by_offset,
    snapshot_weight_groups,
)

logger = logging.getLogger(__name__)

_PUBLIC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_PUBLIC_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")

_KNOWN_DTYPES = {
    # "source" = publish the source's own weights untouched (pure mirror);
    # the flavor's recorded dtype is the detected on-disk dtype.
    "source",
    "fp32", "fp16", "bf16", "fp8", "nvfp4",
    "int4", "int4:nf4", "int4:fp4", "nf4", "fp4",
    # GGUF encodings ride the dtype axis with file_type="gguf".
    "f16", "f32", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
    "q4_0", "q4_1", "q3_k_m", "q3_k_s", "q2_k",
}
_KNOWN_FILE_LAYOUTS = {"diffusers", "singlefile"}
_KNOWN_FILE_TYPES = {"safetensors", "gguf"}

# Families whose singlefile<->diffusers repackage is implemented, in the
# repackage module's normalized vocabulary (fine-tune lineages like
# sdxl-illustrious / flux1-dev / z-image-turbo normalize onto these).
_REPACKAGE_NORMALIZED_FAMILIES = {"sd15_sd2", "sdxl", "flux", "zimage"}

_DEFAULT_QUANT_COMPONENTS = ("transformer", "unet", "text_encoder", "text_encoder_2",
                             "text_encoder_3", "image_encoder", "prior", "controlnet")
_MIN_CONVERT_BYTES = 100 * 1024 * 1024  # leave tiny weights (embeddings) untouched


@dataclass(frozen=True)
class OutputSpec:
    """One requested output flavor: dtype + file layout + container."""

    dtype: str
    file_layout: str
    file_type: str

    @property
    def label(self) -> str:
        return f"{self.dtype}-{self.file_layout}-{self.file_type}".replace(":", "-")


@dataclass
class CloneResult:
    destination_repo: str
    published: list[dict[str, Any]] = field(default_factory=list)
    failed_flavors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_destination_ref(value: str) -> str:
    ref = str(value or "").strip().lower()
    if not ref:
        raise ValueError("destination_repo is required")
    for p in ("tensorhub:", "hf:", "civitai:", "huggingface:"):
        if ref.startswith(p):
            raise ValueError("destination_repo must be bare owner/repo (no provider prefix)")
    parts = ref.split("/", 1)
    if len(parts) != 2 or not all(_PUBLIC_NAME_RE.match(p) for p in parts):
        raise ValueError("destination_repo must be '<owner>/<repo>'")
    return ref


def normalize_tags(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for raw in values or []:
        tag = str(raw or "").strip().lower()
        if not tag or tag in out:
            continue
        if _PUBLIC_TAG_RE.match(tag) is None:
            raise ValueError(f"invalid destination tag: {tag!r}")
        out.append(tag)
    return sorted(out)


def normalize_outputs(values: Iterable[Any] | None, *, layout_hint: str = "diffusers") -> list[OutputSpec]:
    out: list[OutputSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for item in values or []:
        if item is None:
            continue
        get = (lambda k: item.get(k)) if isinstance(item, dict) else (lambda k: getattr(item, k, None))
        # fp8 spellings collapse onto the flavor name: e4m3 is THE fp8 format.
        dtype = str(get("dtype") or "").strip().lower()
        dtype = {"fp8-e4m3": "fp8", "fp8:e4m3": "fp8"}.get(dtype, dtype)
        layout = str(get("file_layout") or "").strip().lower() or layout_hint
        ftype = str(get("file_type") or "").strip().lower() or "safetensors"
        if not dtype:
            raise ValueError("output.dtype is required")
        if dtype not in _KNOWN_DTYPES:
            raise ValueError(f"unsupported output.dtype: {dtype!r}")
        if layout not in _KNOWN_FILE_LAYOUTS:
            raise ValueError(f"unsupported output.file_layout: {layout!r}")
        if ftype not in _KNOWN_FILE_TYPES:
            raise ValueError(f"unsupported output.file_type: {ftype!r}")
        key = (dtype, layout, ftype)
        if key not in seen:
            seen.add(key)
            out.append(OutputSpec(dtype=dtype, file_layout=layout, file_type=ftype))
    if not out:
        layout = layout_hint if layout_hint in _KNOWN_FILE_LAYOUTS else "diffusers"
        out.append(OutputSpec(dtype="bf16", file_layout=layout, file_type="safetensors"))
    return out


# ---------------------------------------------------------------------------
# Flavor tree construction
# ---------------------------------------------------------------------------

def _stage_oversize_safetensors(tree: Path) -> None:
    """Replace any >5 GB safetensors in the tree with HF-convention byte-offset
    shards + index (raw copy, no decode)."""
    for f in sorted(tree.rglob("*.safetensors")):
        if not f.is_file() or f.stat().st_size <= MAX_SAFETENSORS_SHARD_BYTES:
            continue
        stem = f.stem
        stage = f.parent / f".__shard__{stem}"
        shard_paths, index_path, _ = shard_safetensors_by_offset(
            f, stage, shard_prefix=stem)
        if len(shard_paths) > 1:
            f.unlink()
            for sp in shard_paths:
                sp.rename(f.parent / sp.name)
            index_path.rename(f.parent / index_path.name)
        shutil.rmtree(stage, ignore_errors=True)


def build_flavor_tree(
    source: IngestedSource,
    spec: OutputSpec,
    out_dir: Path,
    *,
    quantize_components: list[str] | None = None,
) -> tuple[Path, dict[str, str]]:
    """Materialize one output flavor as a local file tree.

    Passthrough (dtype/layout/container match the source) hardlinks the
    snapshot. Otherwise: optional layout repackage, then per-weight-set
    cast / quant / gguf via :mod:`gen_worker.convert.convert`.

    Returns ``(tree_root, attrs)``. Raises ``InlineConversionNotPossible``
    for calibrated dtypes.
    """
    from .convert import run_inline_conversion

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attrs: dict[str, str] = {"dtype": spec.dtype, "file_layout": spec.file_layout,
                             "file_type": spec.file_type}

    source_dir = Path(source.dir)
    source_layout = source.layout if source.layout in _KNOWN_FILE_LAYOUTS else "singlefile"
    source_dtype = str(source.attrs.get("dtype") or "").strip().lower()

    # dtype="source": pure mirror — publish the source weights untouched and
    # record the detected on-disk dtype. Refuses combos that would still
    # rewrite tensors (layout/container changes) or an undetectable source.
    if spec.dtype == "source":
        if spec.file_type != "safetensors":
            raise ValueError('dtype="source" requires file_type="safetensors"')
        if spec.file_layout != source_layout:
            raise ValueError(
                f'dtype="source" cannot repackage {source_layout}->{spec.file_layout}; '
                "request an explicit dtype")
        if not source_dtype:
            raise ValueError(
                'dtype="source" needs a detectable on-disk dtype; request an explicit dtype')
        attrs["dtype"] = source_dtype
        copy_non_weight_files(source_dir, out_dir, skip_components=set())
        _stage_oversize_safetensors(out_dir)
        return out_dir, attrs

    # GGUF: single-artifact container.
    if spec.file_type == "gguf":
        groups = snapshot_weight_groups(source_dir, source_layout)
        if not groups:
            raise ValueError("no safetensors weights found for gguf conversion")
        result = run_inline_conversion(
            source_path=groups[0][1], out_dir=out_dir, target_dtype=spec.dtype,
            target_file_type="gguf",
            source_repo_dir=(source_dir / groups[0][0]) if groups[0][0] else source_dir,
        )
        attrs.update(result.attributes)
        return out_dir, attrs
    # Layout repackage (singlefile <-> diffusers) when requested + supported.
    work_root = source_dir
    work_layout = source_layout
    if spec.file_layout != source_layout:
        from .repackage import _normalize_family, diffusers_to_singlefile, singlefile_to_diffusers

        family = str(source.model_family or "").strip().lower()
        if _normalize_family(family) not in _REPACKAGE_NORMALIZED_FAMILIES:
            raise ValueError(
                f"layout repackage {source_layout}->{spec.file_layout} unsupported "
                f"for model_family={family!r}")

        repack_dir = out_dir.parent / f"{out_dir.name}.__repack__"
        repack_dir.mkdir(parents=True, exist_ok=True)
        if source_layout == "singlefile":
            groups = snapshot_weight_groups(source_dir, "singlefile")
            if not groups:
                raise ValueError("no safetensors entry for repackage")
            singlefile_to_diffusers(
                groups[0][1], repack_dir, model_family=family, output_dtype=spec.dtype)
        else:
            diffusers_to_singlefile(source_dir, repack_dir / "model.safetensors",
                                    model_family=family)
        work_root = repack_dir
        work_layout = spec.file_layout
        attrs["repackage_toolchain"] = f"{source_layout}_to_{spec.file_layout}:v1"

    # Passthrough: dtype already matches (or repackage output is final).
    needs_dtype_pass = spec.dtype != source_dtype or work_root is not source_dir
    if spec.dtype == source_dtype and work_root is source_dir:
        copy_non_weight_files(source_dir, out_dir, skip_components=set())
        _stage_oversize_safetensors(out_dir)
        return out_dir, attrs

    # Dtype conversion per weight set.
    groups = snapshot_weight_groups(work_root, work_layout)
    is_fp8 = spec.dtype == "fp8"
    if is_fp8 and work_layout != "diffusers":
        raise ValueError(
            "fp8 storage flavors are component-scoped; request "
            'file_layout="diffusers" (repackage first for singlefile sources)')
    if quantize_components:
        target_names = set(quantize_components)
    elif is_fp8:
        # Denoiser-only by default: matches apply_fp8_storage's consumption
        # scope; TEs join explicitly via the component-wise ladder (gw#392).
        target_names = set(FP8_DEFAULT_COMPONENTS)
    else:
        target_names = set(_DEFAULT_QUANT_COMPONENTS)
    is_quant = spec.dtype not in {"bf16", "fp16", "fp32", "f16", "f32"}
    converted: set[str] = set()
    for comp, entry in groups:
        comp_dir = (work_root / comp) if comp else work_root
        size = sum(f.stat().st_size for f in comp_dir.glob("*.safetensors") if f.is_file())
        # Quant targets only the requested components; casts apply everywhere.
        if is_quant and comp and comp not in target_names:
            continue
        if size < _MIN_CONVERT_BYTES and is_quant:
            continue
        dest = (out_dir / comp) if comp else out_dir
        stem = entry.name
        for suffix in (".safetensors.index.json", ".safetensors"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        result = run_inline_conversion(
            source_path=entry, out_dir=dest, target_dtype=spec.dtype,
            target_file_type="safetensors", shard_prefix=stem or "model",
            source_repo_dir=comp_dir,
        )
        attrs.update({k: v for k, v in result.attributes.items() if k not in attrs})
        converted.add(comp)
    if needs_dtype_pass and not converted and not groups:
        raise ValueError("no safetensors weights found to convert")

    copy_non_weight_files(work_root, out_dir, skip_components=converted)
    _stage_oversize_safetensors(out_dir)
    if work_root is not source_dir:
        shutil.rmtree(work_root, ignore_errors=True)
    return out_dir, attrs


# ---------------------------------------------------------------------------
# run_clone — ingest, convert, ONE finalize path
# ---------------------------------------------------------------------------

def _clone_workdir(provider: str, source_key: str, destination: str) -> Path:
    """Persistent workdir keyed by (provider, source, destination): a failed
    clone keeps its downloaded snapshot so a retry resumes instead of
    re-downloading. Deleted on success. Base dir: ``$COZY_CONVERT_WORKDIR``
    or ``<tmp>/gen-worker-convert``."""
    base = Path(os.environ.get("COZY_CONVERT_WORKDIR", "").strip()
                or Path(tempfile.gettempdir()) / "gen-worker-convert")
    digest = hashlib.sha256(
        f"{provider}|{source_key}|{destination}".encode("utf-8")).hexdigest()[:16]
    workdir = base / f"clone-{digest}"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def run_clone(
    ctx: Any,
    *,
    provider: str,
    source_ref: str = "",
    source_revision: str | None = None,
    civitai_model_version_id: int | None = None,
    destination_repo: str,
    destination_repo_tags: Iterable[str] | None = None,
    target_layout: str | None = None,
    source_dtype_preference: list[str] | None = None,
    outputs: Iterable[Any] | None = None,
    quantize_components: list[str] | None = None,
    overwrite_repo: bool = False,
    gguf_quant: str | None = None,
    hf_token: str | None = None,
    civitai_api_key: str | None = None,
) -> CloneResult:
    provider = str(provider or "").strip().lower()
    destination = normalize_destination_ref(destination_repo)
    tags = normalize_tags(destination_repo_tags)
    layout_hint = str(target_layout or "diffusers").strip().lower() or "diffusers"
    specs = normalize_outputs(outputs, layout_hint=layout_hint)
    effective_hf_token = str(hf_token or "").strip() or str(getattr(ctx, "hf_token", "") or "").strip()

    def _progress(p: float, stage: str) -> None:
        fn = getattr(ctx, "progress", None)
        if callable(fn):
            try:
                fn(p, stage=stage)
            except Exception:
                pass

    source_key = source_ref if provider == "huggingface" else str(civitai_model_version_id or 0)
    if source_revision:
        source_key = f"{source_key}@{source_revision}"
    workdir = _clone_workdir(provider, source_key, destination)
    succeeded = False
    try:
        _progress(0.05, "clone.ingest")

        def _dl_progress(done: int, total: Optional[int]) -> None:
            if total:
                _progress(0.05 + 0.45 * min(1.0, done / total), "clone.download")

        if provider == "huggingface":
            source = ingest_huggingface(
                source_ref, workdir / "source",
                revision=source_revision,
                dtype_preference=source_dtype_preference,
                gguf_quant=gguf_quant,
                hf_token=effective_hf_token,
                progress=_dl_progress,
            )
            lineage_parent = f"hf:{source.source_ref}"
        elif provider == "civitai":
            source = ingest_civitai(
                int(civitai_model_version_id or 0), workdir / "source",
                civitai_api_key=civitai_api_key, progress=_dl_progress,
            )
            lineage_parent = f"civitai:{source.source_ref}"
        else:
            raise ValueError(f"unsupported clone provider: {provider!r}")

        _progress(0.5, "clone.convert")
        from .convert import InlineConversionNotPossible

        hubclient = HubClient.from_ctx(ctx)
        result = CloneResult(destination_repo=destination, metadata=dict(source.metadata))
        lineage = [{
            "parent_repo": "external-sources/upstream",
            "parent_checkpoint_id": lineage_parent,
            "relationship_kind": "import",
        }]
        mode = "replace" if overwrite_repo else "merge"

        # Non-diffusers-class sources publish as-is; extra output specs that
        # would need conversion are refused per-flavor, not per-job.
        strategy = source.classification.strategy if source.classification is not None else ""
        publish_as_is = strategy in {
            "transformers", "peft", "sentence_transformers", "gguf", "native_lora",
        }

        for i, spec in enumerate(specs):
            flavor_label = spec.dtype
            try:
                if publish_as_is:
                    source_dtype = str(source.attrs.get("dtype") or "").strip().lower()
                    if i > 0 or (source_dtype and spec.dtype != source_dtype and spec.dtype != "bf16"):
                        # Only the source's own flavor is published for
                        # non-diffusers classes (quant runs as a separate job).
                        if spec.dtype != source_dtype:
                            raise InlineConversionNotPossible(
                                reason=f"{strategy} sources publish as-is; "
                                       f"run a conversion job for {spec.dtype}",
                                target_dtype=spec.dtype,
                            )
                    tree = source.dir
                    attrs = dict(source.attrs)
                    flavor_label = source_dtype or spec.dtype
                else:
                    # Wipe any partial flavor tree from a prior failed run —
                    # only the downloaded source is resumable.
                    flavor_dir = workdir / f"flavor-{spec.label}"
                    shutil.rmtree(flavor_dir, ignore_errors=True)
                    shutil.rmtree(workdir / f"flavor-{spec.label}.__repack__",
                                  ignore_errors=True)
                    tree, attrs = build_flavor_tree(
                        source, spec, flavor_dir,
                        quantize_components=quantize_components,
                    )
                    # dtype="source" resolves to the detected on-disk dtype.
                    flavor_label = str(attrs.get("dtype") or spec.dtype)
            except InlineConversionNotPossible as exc:
                entry: dict[str, Any] = {
                    "spec_label": spec.label, "dtype": spec.dtype,
                    "file_type": spec.file_type, "reason": exc.reason,
                }
                deferred = getattr(exc, "deferred_requirement", None)
                if deferred is not None:
                    entry["deferred_requirement"] = deferred.as_dict()
                result.failed_flavors.append(entry)
                continue
            except Exception as exc:  # noqa: BLE001 — partial success per flavor
                result.failed_flavors.append({
                    "spec_label": spec.label, "dtype": spec.dtype,
                    "file_type": spec.file_type, "reason": str(exc),
                })
                continue

            files = files_from_tree(tree)
            if not files:
                result.failed_flavors.append({
                    "spec_label": spec.label, "dtype": spec.dtype,
                    "file_type": spec.file_type, "reason": "flavor tree is empty",
                })
                continue

            # size facts for VRAM-aware placement (advisory).
            metadata: dict[str, Any] = {k: v for k, v in source.metadata.items()}
            try:
                from .size_walk import compute_size_facts

                facts = compute_size_facts(str(tree))
                if facts.get("full_model_bytes"):
                    metadata["size_facts"] = facts
            except Exception:
                pass
            for k, v in attrs.items():
                metadata.setdefault(f"attr_{k}", str(v))

            _progress(0.55 + 0.4 * (i / max(1, len(specs))), f"clone.publish.{spec.label}")
            commit = hubclient.commit(
                destination_repo=destination,
                files=files,
                tags=tags,
                mode=mode if i == 0 else "merge",
                flavor=flavor_label,
                dtype=str(attrs.get("dtype") or spec.dtype),
                file_layout=str(attrs.get("file_layout") or spec.file_layout),
                file_type=str(attrs.get("file_type") or spec.file_type),
                message=f"clone {provider}:{source.source_ref}@{source.source_revision}",
                metadata=metadata,
                lineage=lineage,
                repo_spec=source.repo_spec,
                auto_create_external_parent=True,
            )
            result.published.append({
                "flavor": flavor_label,
                "spec_label": spec.label,
                "revision_id": commit.revision_id,
                "uploaded": commit.uploaded,
                "deduped": commit.deduped,
                "total_bytes": commit.total_bytes,
            })

        if not result.published:
            reasons = "; ".join(
                str(f.get("reason") or "") for f in result.failed_flavors
            ) or "no output spec produced anything"
            raise RuntimeError(f"clone produced no publishable flavor: {reasons}")

        result.metadata["destination_repo"] = destination
        result.metadata["published_count"] = str(len(result.published))
        if result.failed_flavors:
            result.metadata["failed_flavor_count"] = str(len(result.failed_flavors))
        _progress(1.0, "clone.completed")
        succeeded = True
        return result
    finally:
        if succeeded:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            logger.warning("clone failed; workdir retained for resume: %s", workdir)


def from_huggingface(ctx: Any, payload: Any, *, hf_token: str | None = None) -> CloneResult:
    """Clone a Hugging Face repo end-to-end: download, convert, publish."""
    return run_clone(
        ctx,
        provider="huggingface",
        source_ref=str(getattr(payload, "huggingface_repo", "") or ""),
        source_revision=getattr(payload, "source_revision", None),
        destination_repo=str(getattr(payload, "destination_repo", "") or ""),
        destination_repo_tags=getattr(payload, "destination_repo_tags", None),
        target_layout=getattr(payload, "target_layout", None),
        source_dtype_preference=getattr(payload, "source_dtype_preference", None),
        outputs=getattr(payload, "outputs", None),
        quantize_components=getattr(payload, "quantize_components", None),
        overwrite_repo=bool(getattr(payload, "overwrite_repo", False)),
        gguf_quant=getattr(payload, "gguf_quant", None),
        hf_token=hf_token,
    )


def from_civitai(ctx: Any, payload: Any, *, civitai_api_key: str | None = None) -> CloneResult:
    """Clone a Civitai model version end-to-end (no arbitrary-URL sources)."""
    version_id = int(getattr(payload, "civitai_model_version_id", 0) or 0)
    if version_id <= 0:
        raise ValueError("civitai_model_version_id is required")
    return run_clone(
        ctx,
        provider="civitai",
        civitai_model_version_id=version_id,
        destination_repo=str(getattr(payload, "destination_repo", "") or ""),
        destination_repo_tags=getattr(payload, "destination_repo_tags", None),
        target_layout=getattr(payload, "target_layout", None),
        outputs=getattr(payload, "outputs", None),
        quantize_components=getattr(payload, "quantize_components", None),
        overwrite_repo=bool(getattr(payload, "overwrite_repo", False)),
        civitai_api_key=civitai_api_key,
    )


__all__ = [
    "CloneResult",
    "OutputSpec",
    "build_flavor_tree",
    "from_civitai",
    "from_huggingface",
    "normalize_destination_ref",
    "normalize_outputs",
    "normalize_tags",
    "run_clone",
]
