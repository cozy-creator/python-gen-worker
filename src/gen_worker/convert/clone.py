"""Clone: mirror an external checkpoint into Tensorhub, optionally converting.

download → (repackage) → (cast / quant / gguf) → ONE finalize:
each requested output flavor becomes one local file tree and one Tensorhub
commit (``POST /commits`` + part PUTs + finalize). ``overwrite_repo`` maps to
``mode="replace"`` — no enumerate-prior-and-delete.

Entry points: :func:`from_huggingface` / :func:`from_civitai` (payload-shaped)
and :func:`run_clone` (keyword-explicit).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .bank import build_bank_payload, flavor_bank_key
from .hub import CommitFile, HubClient, HubPublishError, files_from_tree
from .ingest import (
    IngestedSource,
    _is_multi_weight_names,
    ingest_civitai,
    ingest_huggingface,
    plan_civitai,
    plan_huggingface,
)
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
        from gen_worker.models.refs import flavor_token

        return flavor_token(f"{self.dtype}-{self.file_layout}-{self.file_type}")


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

_CAST_NORMALIZE_DTYPES = {"fp16", "bf16", "fp32", "f16", "f32"}

_VARIANT_WEIGHT_NAME_RE = re.compile(
    r"^(?P<base>.+)\.(?P<v>fp16|bf16|fp32)"
    r"(?P<shard>-\d{5}-of-\d{5})?\.safetensors(?P<idx>\.index\.json)?$"
)


def _normalize_variant_filenames(tree: Path) -> None:
    """Strip dtype-variant tokens from published weight filenames.

    dtype is a checkpoint axis (flavor) in repo-cas — one dtype per tree — so
    HF variant suffixes are redundant, and the resharder composes an index
    name diffusers cannot find (J23 live: juggernaut-xl published
    "diffusion_pytorch_model.fp16.safetensors.index.json" where diffusers'
    _add_variant expects "diffusion_pytorch_model.safetensors.index.fp16.json";
    variant=fp16 serve setup died). Canonical names sidestep the class.
    A directory whose canonical twin already exists is left untouched
    (dual-dtype upstream trees must not collide). Quant flavors (fp8/int4/…)
    are never normalized — their names carry loader semantics."""
    dirs = sorted({p.parent for p in tree.rglob("*.safetensors*") if p.is_file()})
    for d in dirs:
        renames: dict[str, str] = {}
        for p in sorted(d.iterdir()):
            m = _VARIANT_WEIGHT_NAME_RE.match(p.name)
            if not m:
                continue
            new_name = f"{m['base']}{m['shard'] or ''}.safetensors{m['idx'] or ''}"
            if (d / new_name).exists():
                renames.clear()
                break
            renames[p.name] = new_name
        for old_name, new_name in renames.items():
            (d / old_name).rename(d / new_name)
        if not renames:
            continue
        for idx in d.glob("*.safetensors.index.json"):
            try:
                payload = json.loads(idx.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, dict):
                continue
            changed = False
            for key, shard in weight_map.items():
                if shard in renames:
                    weight_map[key] = renames[shard]
                    changed = True
            if changed:
                idx.write_text(
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    encoding="utf-8",
                )


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
        if source_dtype in _CAST_NORMALIZE_DTYPES:
            _normalize_variant_filenames(out_dir)
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
        if spec.dtype in _CAST_NORMALIZE_DTYPES:
            _normalize_variant_filenames(out_dir)
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
    if spec.dtype in _CAST_NORMALIZE_DTYPES:
        _normalize_variant_filenames(out_dir)
    if work_root is not source_dir:
        shutil.rmtree(work_root, ignore_errors=True)
    return out_dir, attrs


# ---------------------------------------------------------------------------
# th#592 download-skip: publish from banked manifests (zero bytes downloaded)
# ---------------------------------------------------------------------------

def _publish_from_bank(
    hubclient: HubClient,
    *,
    plan: Any,
    provider: str,
    specs: list[OutputSpec],
    bank_keys: dict[str, str],
    destination: str,
    tags: list[str],
    mode: str,
    progress: Any,
) -> Optional[CloneResult]:
    """Try to publish EVERY requested flavor from the hub's banked manifests
    (commit-by-CAS-reference; no local bytes). Returns None on any miss or
    error — the caller falls through to the full download path (fail-open).
    All-or-nothing on purpose: the download is shared across specs, so one
    miss means downloading anyway."""
    if not bank_keys or any(not k for k in bank_keys.values()):
        return None
    try:
        lookup = hubclient.lookup_clone_manifests(
            destination, sorted(set(bank_keys.values())))
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.warning("download-skip bank lookup failed (fail-open, full clone): %s", exc)
        return None

    payloads: dict[str, dict[str, Any]] = {}
    for spec in specs:
        r = lookup.get(bank_keys[spec.label]) or {}
        payload = r.get("payload")
        if not r.get("ready") or not isinstance(payload, dict) or not payload.get("files"):
            logger.info(
                "download-skip bank miss for %s:%s spec=%s (found=%s ready=%s)",
                provider, plan.source_ref, spec.label,
                bool(r.get("found")), bool(r.get("ready")))
            return None
        payloads[spec.label] = payload

    source_bytes = sum(size for _, size, _ in plan.bank_files())
    revision = str(getattr(plan, "revision", "") or "")
    provenance = {"upstream_revision": revision}
    result = CloneResult(destination_repo=destination)
    try:
        for i, spec in enumerate(specs):
            payload = payloads[spec.label]
            files = [
                CommitFile(
                    path=str(f.get("path") or ""),
                    local_path=None,
                    size_bytes=int(f.get("size_bytes") or 0),
                    blake3=str(f.get("blake3") or ""),
                )
                for f in payload["files"]
            ]
            metadata_raw = payload.get("metadata")
            metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
            metadata = dict(metadata)
            metadata["download_skip"] = "bank"
            # Banked repo_specs are frozen at bank time and can carry a stale
            # pre-gw#477 library_name="diffusers" for multi-weight bundles —
            # recompute the opt-out from the banked file names so a bank-path
            # publish never (re)creates a repo the layout contract rejects.
            repo_spec = {k: str(v) for k, v in dict(payload.get("repo_spec") or {}).items()}
            if repo_spec.get("library_name") == "diffusers" and _is_multi_weight_names(
                    str(f.get("path") or "") for f in payload["files"]
                    if "/" not in str(f.get("path") or "")):
                repo_spec["library_name"] = ""
                metadata["multi_weight_bundle"] = "true"
            if callable(progress):
                progress(0.1 + 0.85 * (i / max(1, len(specs))),
                         f"clone.publish.{spec.label}")
            commit = hubclient.commit(
                destination_repo=destination,
                files=files,
                tags=tags,
                mode=mode if i == 0 else "merge",
                flavor=str(payload.get("flavor") or spec.dtype),
                dtype=str(payload.get("dtype") or spec.dtype),
                file_layout=str(payload.get("file_layout") or spec.file_layout),
                file_type=str(payload.get("file_type") or spec.file_type),
                message=(
                    f"clone {provider}:{plan.source_ref}"
                    f"@{revision or payload.get('source_revision') or ''}"
                ),
                metadata=metadata,
                provenance=provenance,
                repo_spec=repo_spec,
            )
            result.published.append({
                "flavor": str(payload.get("flavor") or spec.dtype),
                "spec_label": spec.label,
                "revision_id": commit.revision_id,
                "checkpoint_id": commit.checkpoint_id,
                "uploaded": commit.uploaded,
                "deduped": commit.deduped,
                "total_bytes": commit.total_bytes,
                "banked": True,
            })
    except HubPublishError as exc:
        # Includes BankedBlobGoneError (CAS GC'd a blob between lookup and
        # commit). Never fatal: the full clone re-creates everything.
        logger.warning(
            "download-skip bank publish failed (%s); falling back to full clone", exc)
        return None

    first = payloads[specs[0].label]
    if isinstance(first.get("metadata"), dict):
        result.metadata.update({
            str(k): str(v) for k, v in first["metadata"].items() if isinstance(v, str)
        })
    result.metadata["destination_repo"] = destination
    result.metadata["published_count"] = str(len(result.published))
    result.metadata["download_skip"] = "bank"
    result.metadata["source_bytes_downloaded"] = "0"
    result.metadata["source_bytes_avoided"] = str(source_bytes)
    logger.info(
        "download-skip engaged: %s:%s -> %s published %d spec(s) by CAS reference; "
        "%d source files (%.2f GB) NOT downloaded",
        provider, plan.source_ref, destination, len(result.published),
        len(plan.bank_files()), source_bytes / 1e9)
    if callable(progress):
        progress(1.0, "clone.completed")
    return result


# ---------------------------------------------------------------------------
# run_clone — ingest, convert, ONE finalize path
# ---------------------------------------------------------------------------

class CloneDiskSpaceError(RuntimeError):
    """Preflight found too little free disk for the clone — fail fast and
    actionably instead of ENOSPC minutes into a 40GB download (gw#462)."""


# Free space required = source_bytes x headroom: the source snapshot and each
# converted flavor tree (+ repack temp) coexist on disk during publish.
_DISK_HEADROOM_DEFAULT = 2.5
_DISK_MARGIN_BYTES = 2 * 1024**3


def _preflight_disk(workdir: Path, plan: Any) -> None:
    """Fail fast when the disk cannot fit the clone. The source plan knows
    every selected file's size before a byte is downloaded (HF list_repo_tree
    / civitai version API); no plan (metadata fetch failed) skips the check
    (fail-open — the download surfaces its own error)."""
    if plan is None:
        return
    try:
        source_bytes = sum(int(size) for _, size, _ in plan.bank_files())
    except Exception:  # noqa: BLE001 — preflight is best-effort on odd plans
        return
    if source_bytes <= 0:
        return
    headroom = float(os.environ.get("COZY_CONVERT_DISK_HEADROOM", "") or _DISK_HEADROOM_DEFAULT)
    required = int(source_bytes * headroom) + _DISK_MARGIN_BYTES
    free = shutil.disk_usage(workdir).free
    if free < required:
        gib = float(1024**3)
        raise CloneDiskSpaceError(
            f"not enough disk for clone: need ~{required / gib:.1f} GiB free "
            f"(source {source_bytes / gib:.1f} GiB x {headroom:g} headroom "
            f"+ {_DISK_MARGIN_BYTES / gib:.0f} GiB margin), have {free / gib:.1f} GiB "
            f"at {workdir}")


def _sweep_stale_workdirs(base: Path, *, keep: Optional[Path] = None) -> None:
    """Remove clone scratch left by crashed predecessors: dirs whose flock is
    free and that have been idle past COZY_CONVERT_SCRATCH_TTL_S (default 1h).
    A long-running conversion worker otherwise accumulates each crashed job's
    scratch until any disk fills (gw#462)."""
    try:
        entries = sorted(base.glob("clone-*"))
    except OSError:
        return
    ttl_s = float(os.environ.get("COZY_CONVERT_SCRATCH_TTL_S", "") or 3600.0)
    now = time.time()
    for d in entries:
        if not d.is_dir() or (keep is not None and d == keep):
            continue
        try:
            if now - d.stat().st_mtime < ttl_s:
                continue
        except OSError:
            continue
        lock_path = base / f".{d.name}.lock"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # a live clone holds this workdir
            shutil.rmtree(d, ignore_errors=True)
            logger.info("swept stale clone scratch: %s", d)
        finally:
            os.close(fd)


def _clone_workdir(provider: str, source_key: str, destination: str) -> Path:
    """Workdir keyed by (provider, source, destination) so concurrent
    duplicates of the same clone serialize on one flock. Removed after every
    job — success or failure (COZY_CONVERT_RETAIN_WORKDIR=1 keeps a failed
    job's scratch for debugging). Base dir: ``$COZY_CONVERT_WORKDIR`` or
    ``<tmp>/gen-worker-convert``."""
    base = Path(os.environ.get("COZY_CONVERT_WORKDIR", "").strip()
                or Path(tempfile.gettempdir()) / "gen-worker-convert")
    digest = hashlib.sha256(
        f"{provider}|{source_key}|{destination}".encode("utf-8")).hexdigest()[:16]
    workdir = base / f"clone-{digest}"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _acquire_workdir_lock(workdir: Path) -> int:
    """Exclusive flock serializing clones that share one keyed workdir.
    Concurrent duplicates (crash-recovery re-queues of the same clone)
    otherwise corrupt the shared snapshot: hf_hub's local-dir download
    unlinks + re-fetches files the peer clone is mid-reading."""
    lock_path = workdir.parent / f".{workdir.name}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info(
            "workdir %s held by a concurrent clone of the same source; waiting", workdir)
        fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


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
    _sweep_stale_workdirs(workdir.parent, keep=workdir)
    lock_fd = _acquire_workdir_lock(workdir)
    succeeded = False
    try:
        if provider not in {"huggingface", "civitai"}:
            raise ValueError(f"unsupported clone provider: {provider!r}")

        hubclient = HubClient.from_ctx(ctx)
        mode = "replace" if overwrite_repo else "merge"

        # th#592 download-skip: derive the source's identity from provider
        # metadata alone (no bytes), then try to publish every requested
        # flavor from the hub's banked manifests. Any miss/error falls
        # through to the full clone below (fail-open).
        _progress(0.02, "clone.plan")
        plan: Any = None
        try:
            if provider == "huggingface":
                plan = plan_huggingface(
                    source_ref,
                    revision=source_revision,
                    dtype_preference=source_dtype_preference,
                    gguf_quant=gguf_quant,
                    hf_token=effective_hf_token,
                )
            else:
                plan = plan_civitai(
                    int(civitai_model_version_id or 0),
                    civitai_api_key=civitai_api_key,
                    gguf_quant=gguf_quant,
                )
        except Exception as exc:
            logger.warning(
                "clone source plan failed (download-skip disabled for this run): %s", exc)

        bank_keys: dict[str, str] = {}
        if plan is not None:
            for spec in specs:
                bank_keys[spec.label] = flavor_bank_key(
                    plan, spec.label,
                    layout_hint=layout_hint,
                    quantize_components=quantize_components,
                    gguf_quant=gguf_quant,
                )
            banked = _publish_from_bank(
                hubclient,
                plan=plan,
                provider=provider,
                specs=specs,
                bank_keys=bank_keys,
                destination=destination,
                tags=tags,
                mode=mode,
                progress=_progress,
            )
            if banked is not None:
                succeeded = True
                return banked

        # gw#462: the plan already knows every selected file's size — fail
        # fast on an undersized disk instead of ENOSPC mid-download.
        _preflight_disk(workdir, plan)

        _progress(0.05, "clone.ingest")
        dl_bytes = {"done": 0}

        def _dl_progress(done: int, total: Optional[int]) -> None:
            dl_bytes["done"] = max(dl_bytes["done"], int(done or 0))
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
                plan=plan,
            )
        else:
            source = ingest_civitai(
                int(civitai_model_version_id or 0), workdir / "source",
                civitai_api_key=civitai_api_key, progress=_dl_progress,
                gguf_quant=gguf_quant,
            )

        _progress(0.5, "clone.convert")
        from .convert import InlineConversionNotPossible

        result = CloneResult(destination_repo=destination, metadata=dict(source.metadata))
        # th#606: upstream identity (upstream_ref + derivation_op=import) is
        # orchestrator-derived and rides the capability token; the worker only
        # ADDS the revision it actually resolved during download.
        provenance = {"upstream_revision": str(source.source_revision or "")}
        bank_records: list[dict[str, Any]] = []

        # Non-diffusers-class sources publish as-is; extra output specs that
        # would need conversion are refused per-flavor, not per-job.
        strategy = source.classification.strategy if source.classification is not None else ""
        publish_as_is = strategy in {
            "transformers", "peft", "sentence_transformers", "gguf", "native_lora",
            "pipeline_tree",
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
                # Hub flavor tokens are [a-z0-9][a-z0-9._-]{0,63}: the gguf
                # dtype-axis label ("gguf:q4_k_m") publishes as "gguf-q4_k_m"
                # (the th#611 flavor convention).
                from gen_worker.models.refs import flavor_token

                flavor_label = flavor_token(flavor_label)
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
                # gw#419: the PRIMARY output also owns the bare selector row —
                # tensorhub (th#597 C1) never moves flavor='' for an
                # explicit-flavor publish unless default_flavor names it, and
                # every serve/convert flow references mirrors bare (repo:tag).
                default_flavor=flavor_label if i == 0 else "",
                dtype=str(attrs.get("dtype") or spec.dtype),
                file_layout=str(attrs.get("file_layout") or spec.file_layout),
                file_type=str(attrs.get("file_type") or spec.file_type),
                message=f"clone {provider}:{source.source_ref}@{source.source_revision}",
                metadata=metadata,
                provenance=provenance,
                repo_spec=source.repo_spec,
            )
            result.published.append({
                "flavor": flavor_label,
                "spec_label": spec.label,
                "revision_id": commit.revision_id,
                "checkpoint_id": commit.checkpoint_id,
                "uploaded": commit.uploaded,
                "deduped": commit.deduped,
                "total_bytes": commit.total_bytes,
            })

            # th#592: bank this flavor's published manifest for future
            # download-skips. HF repackaged flavors are excluded — their
            # output depends on model_family detected from DOWNLOADED bytes,
            # which the pre-download bank key cannot see. (Civitai family
            # comes from the version API's baseModel: pre-download, in-key.)
            key = bank_keys.get(spec.label, "")
            hf_repackaged = provider == "huggingface" and bool(attrs.get("repackage_toolchain"))
            if key and not hf_repackaged:
                bank_records.append({
                    "key": key,
                    "payload": build_bank_payload(
                        files=[
                            {"path": f.path, "blake3": f.blake3, "size_bytes": f.size_bytes}
                            for f in files
                        ],
                        flavor=flavor_label,
                        dtype=str(attrs.get("dtype") or spec.dtype),
                        file_layout=str(attrs.get("file_layout") or spec.file_layout),
                        file_type=str(attrs.get("file_type") or spec.file_type),
                        metadata=metadata,
                        repo_spec=source.repo_spec,
                        source_revision=source.source_revision,
                    ),
                })

        if bank_records:
            try:
                statuses = hubclient.record_clone_manifests(destination, bank_records)
                logger.info("download-skip bank recorded %d manifest(s): %s",
                            len(bank_records), statuses)
            except Exception as exc:  # noqa: BLE001 — banking is best-effort
                logger.warning("download-skip bank record failed (non-fatal): %s", exc)

        if not result.published:
            reasons = "; ".join(
                str(f.get("reason") or "") for f in result.failed_flavors
            ) or "no output spec produced anything"
            raise RuntimeError(f"clone produced no publishable flavor: {reasons}")

        result.metadata["destination_repo"] = destination
        result.metadata["published_count"] = str(len(result.published))
        result.metadata["source_bytes_downloaded"] = str(dl_bytes["done"])
        if result.failed_flavors:
            result.metadata["failed_flavor_count"] = str(len(result.failed_flavors))
        _progress(1.0, "clone.completed")
        succeeded = True
        return result
    finally:
        # gw#462: a long-running worker must not leak scratch — the workdir
        # goes after EVERY job. Cross-run resume lives in the publish bank
        # (th#592) + CAS dedup, not in retained local bytes.
        retain = os.environ.get("COZY_CONVERT_RETAIN_WORKDIR", "").strip() == "1"
        if succeeded or not retain:
            shutil.rmtree(workdir, ignore_errors=True)
            if not succeeded:
                logger.warning(
                    "clone failed; workdir %s removed "
                    "(COZY_CONVERT_RETAIN_WORKDIR=1 keeps it for debugging)", workdir)
        else:
            logger.warning("clone failed; workdir retained: %s", workdir)
        os.close(lock_fd)  # releases the flock


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
        gguf_quant=getattr(payload, "gguf_quant", None),
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
