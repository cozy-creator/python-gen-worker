"""Clone: mirror an external checkpoint into Tensorhub, optionally converting."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from gen_worker.api.errors import ValidationError

from ..serving_facts import OBJECTIVES
from ..hubio.client import HubClient, _dtype_token, files_from_tree
from ..hubio.publish_state import JOURNAL_NAME, ProducerRecovery
from .convert import run_inline_conversion
from .dtype_pins import (
    DTYPE_BITS as _DTYPE_STORAGE_BITS,
)
from .dtype_pins import (
    cast_exempt_components,
    check_explicit_pin_conflict,
    verify_produced_tree,
)
from .ingest import (
    IngestedSource,
    ingest_civitai,
    ingest_huggingface,
    plan_civitai,
    plan_huggingface,
)
from .keepalive import HubKeepalive
from .publish import destination_release as _destination_release
from .layout import canonical_model_family_from_variant, infer_model_family_variant_from_hint
from .registry import repackage_family
from .writer import (
    CAST_NORMALIZE_DTYPES as _CAST_NORMALIZE_DTYPES,
)
from .writer import (
    apply_objective_scheduler_config,
    copy_non_weight_files,
    deshard_mirror_tree,
    fp8_default_components,
    snapshot_weight_groups,
    tree_has_sharded_safetensors,
)
from .writer import (
    normalize_variant_filenames as _normalize_variant_filenames,
)
from ..scratchrepo import PREFIX as SCRATCH_PREFIX, is_scratch_name

logger = logging.getLogger(__name__)

_PUBLIC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_SHARD_MEMBER_RE = re.compile(
    r"^(?P<group>.*?[^/]+)-\d{5}-of-\d{5}\.safetensors$")

_KNOWN_DTYPES = {
    "source",
    "fp32", "fp16", "bf16", "fp8", "nvfp4",
    "int4", "int4:nf4", "int4:fp4", "nf4", "fp4",
    "f16", "f32", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
    "q4_0", "q4_1", "q3_k_m", "q3_k_s", "q2_k",
}
from ..component_vocab import quant_candidate_components
from ..models.file_layout import KNOWN_FILE_LAYOUTS, MULTI_FILE, SINGLE_FILE

_KNOWN_FILE_LAYOUTS = set(KNOWN_FILE_LAYOUTS)
_KNOWN_FILE_TYPES = {"safetensors", "gguf"}

_default_quant_components = quant_candidate_components
_MIN_CONVERT_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class OutputSpec:
    """One requested output flavor: dtype + file layout + container."""

    dtype: str
    file_layout: str
    file_type: str

    @property
    def label(self) -> str:

        return _dtype_token(f"{self.dtype}-{self.file_layout}-{self.file_type}")


@dataclass
class CloneResult:
    destination_repo: str
    published: list[dict[str, Any]] = field(default_factory=list)
    failed_flavors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def _payload_destination_release(payload: Any) -> str:
    dest = getattr(payload, "destination", None)
    if dest is not None:
        get = dest.get if isinstance(dest, dict) else (lambda k: getattr(dest, k, None))
        rel = str(get("release") or "").strip()
        if rel:
            return rel
    return str(getattr(payload, "destination_release", "") or "").strip()


def _valid_repo_name(name: str) -> bool:
    if is_scratch_name(name):
        return bool(_PUBLIC_NAME_RE.match(name[len(SCRATCH_PREFIX):]))
    return bool(_PUBLIC_NAME_RE.match(name))


def normalize_destination_ref(value: str) -> str:
    ref = str(value or "").strip().lower()
    if not ref:
        raise ValueError("destination_repo is required")
    for p in ("tensorhub:", "hf:", "civitai:", "huggingface:"):
        if ref.startswith(p):
            raise ValueError("destination_repo must be bare owner/repo (no provider prefix)")
    parts = ref.split("/", 1)
    if len(parts) != 2 or not _PUBLIC_NAME_RE.match(parts[0]) or not _valid_repo_name(parts[1]):
        raise ValueError("destination_repo must be '<owner>/<repo>'")
    return ref


def normalize_source_include(value: Any) -> tuple[str, ...]:
    """Dual-form clone-request field disambiguating a multi-checkpoint-bundle source repo: compact form is a single glob string, structured form is a list of globs."""
    if value is None:
        return ()
    if isinstance(value, str):
        v = value.strip()
        return (v,) if v else ()
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
        return tuple(out)
    raise ValueError(
        f"source_include must be a string or a list of strings, got {type(value).__name__}")




def normalize_outputs(values: Iterable[Any] | None, *, layout_hint: str = MULTI_FILE) -> list[OutputSpec]:
    out: list[OutputSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for item in values or []:
        if item is None:
            continue
        get = (lambda k: item.get(k)) if isinstance(item, dict) else (lambda k: getattr(item, k, None))
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
        layout = layout_hint if layout_hint in _KNOWN_FILE_LAYOUTS else MULTI_FILE
        out.append(OutputSpec(dtype="bf16", file_layout=layout, file_type="safetensors"))
    return out


def build_flavor_tree(
    source: IngestedSource,
    spec: OutputSpec,
    out_dir: Path,
    *,
    quantize_components: list[str] | None = None,
    objective: str = "",
    distilled: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Materialize one output flavor as a local file tree."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attrs: dict[str, str] = {"dtype": spec.dtype, "file_layout": spec.file_layout,
                             "file_type": spec.file_type}

    source_dir = Path(source.dir)
    source_layout = source.layout if source.layout in _KNOWN_FILE_LAYOUTS else SINGLE_FILE
    source_dtype = str(source.attrs.get("dtype") or "").strip().lower()

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
        deshard_mirror_tree(out_dir)
        if source_dtype in _CAST_NORMALIZE_DTYPES:
            _normalize_variant_filenames(out_dir)
        apply_objective_scheduler_config(out_dir, objective, distilled)
        return out_dir, attrs

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
    work_root = source_dir
    work_layout = source_layout
    if spec.file_layout != source_layout:
        from .repackage import diffusers_to_singlefile, singlefile_to_diffusers

        family = str(source.model_family or "").strip().lower()
        declared = repackage_family(family)
        if declared is None or not (
            declared.supports_singlefile_to_diffusers if source_layout == SINGLE_FILE
            else declared.supports_diffusers_to_singlefile
        ):
            raise ValueError(
                f"layout repackage {source_layout}->{spec.file_layout} unsupported "
                f"for model_family={family!r}")

        repack_dir = out_dir.parent / f"{out_dir.name}.__repack__"
        repack_dir.mkdir(parents=True, exist_ok=True)
        if source_layout == SINGLE_FILE:
            groups = snapshot_weight_groups(source_dir, SINGLE_FILE)
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

    needs_dtype_pass = spec.dtype != source_dtype or work_root is not source_dir
    if spec.dtype == source_dtype and work_root is source_dir:
        copy_non_weight_files(source_dir, out_dir, skip_components=set())
        deshard_mirror_tree(out_dir)
        if spec.dtype in _CAST_NORMALIZE_DTYPES:
            _normalize_variant_filenames(out_dir)
        apply_objective_scheduler_config(out_dir, objective, distilled)
        return out_dir, attrs

    groups = snapshot_weight_groups(work_root, work_layout)
    is_fp8 = spec.dtype == "fp8"
    fp8_block_scope = False
    if is_fp8 and work_layout != MULTI_FILE:
        if len(groups) != 1 or groups[0][0] != "":
            raise ValueError(
                "fp8 storage flavors need component identity: a diffusers "
                "layout, or a single root weight set (transformers "
                f"backbone) — found {len(groups)} weight set(s)")
        fp8_block_scope = True
    if quantize_components:
        target_names = set(quantize_components)
    elif is_fp8:
        target_names = set(fp8_default_components())
    else:
        target_names = set(_default_quant_components())
    is_quant = spec.dtype not in {"bf16", "fp16", "fp32", "f16", "f32"}
    check_explicit_pin_conflict(
        work_root, spec.dtype, quantize_components if is_quant else None)
    pin_exempt = cast_exempt_components(work_root, spec.dtype)
    skipped_pins: dict[str, str] = {}
    converted: set[str] = set()
    for comp, entry in groups:
        comp_dir = (work_root / comp) if comp else work_root
        size = sum(f.stat().st_size for f in comp_dir.glob("*.safetensors") if f.is_file())
        if is_quant and comp and comp not in target_names:
            continue
        if size < _MIN_CONVERT_BYTES and is_quant:
            continue
        fact = pin_exempt.get(comp)
        if fact is not None:
            skipped_pins[comp] = fact.dtype
            logger.info("clone.cast.pin_skip component=%s pin=%s requested=%s reason=%s",
                        comp, fact.dtype, spec.dtype, fact.reason)
            continue
        dest = (out_dir / comp) if comp else out_dir
        stem = entry.name
        for suffix in (".safetensors.index.json", ".safetensors"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        result = run_inline_conversion(
            source_path=entry, out_dir=dest, target_dtype=spec.dtype,
            target_file_type="safetensors", output_stem=stem or "model",
            source_repo_dir=comp_dir, fp8_block_scope=fp8_block_scope,
        )
        attrs.update({k: v for k, v in result.attributes.items() if k not in attrs})
        converted.add(comp)
    if needs_dtype_pass and not converted and not groups:
        raise ValueError("no safetensors weights found to convert")

    copy_non_weight_files(work_root, out_dir, skip_components=converted)
    deshard_mirror_tree(out_dir)
    if spec.dtype in _CAST_NORMALIZE_DTYPES:
        _normalize_variant_filenames(out_dir)
    apply_objective_scheduler_config(out_dir, objective, distilled)
    if skipped_pins:
        attrs["dtype_pinned_components"] = ",".join(
            f"{c}:{d}" for c, d in sorted(skipped_pins.items()))
    if work_root is not source_dir:
        shutil.rmtree(work_root, ignore_errors=True)
    return out_dir, attrs


class CloneDiskSpaceError(RuntimeError):
    """Preflight found too little free disk for the clone — fail fast and actionably instead of ENOSPC minutes into a 40GB download."""


_DISK_MARGIN_BYTES = 2 * 1024**3
_PUBLISH_AS_IS_STRATEGIES = frozenset({
    "transformers", "peft", "sentence_transformers", "gguf", "native_lora",
    "pipeline_tree", "diffusers_component",
})
_CAST_ELIGIBLE_PUBLISH_AS_IS_STRATEGIES = frozenset({
    "transformers", "peft", "sentence_transformers", "native_lora",
    "diffusers_component",
})
_DIRECT_GGUF_ENCODINGS = frozenset({"f32", "f16", "bf16", "q8_0"})
_UNRESOLVED_SOURCE_BITS = 32


def _plan_has_no_repackager(plan: Any) -> bool:

    for p in getattr(plan, "paths", None) or ():
        variant = infer_model_family_variant_from_hint(str(p))
        if variant == "unknown":
            continue
        declared = repackage_family(canonical_model_family_from_variant(variant))
        return declared is None or not declared.supports_singlefile_to_diffusers
    return False


def _output_tree_bytes(
    spec: OutputSpec, source_bytes: int, source_bits: int, measured_bits: int,
) -> int:
    out_bits = _DTYPE_STORAGE_BITS.get(spec.dtype)
    if out_bits is None:
        return source_bytes
    if not measured_bits and out_bits <= source_bits:
        return source_bytes
    return (source_bytes * out_bits + source_bits - 1) // source_bits


def _preflight_disk(workdir: Path, plan: Any, specs: list[OutputSpec]) -> None:
    if plan is None:
        return
    try:
        files = [(str(path), int(size)) for path, size, _ in plan.bank_files()]
        source_bytes = sum(size for _, size in files)
        provider = str(getattr(plan, "provider", "") or "").strip().lower()
        classification = getattr(plan, "classification", None)
        strategy = str(getattr(classification, "strategy", "") or "").strip().lower()
        raw_attrs = getattr(classification, "attrs", None)
        attrs = {
            str(k): str(v).strip().lower()
            for k, v in (raw_attrs.items() if isinstance(raw_attrs, dict) else ())
        }
        if classification is None and provider == "civitai":
            source_type = (
                "gguf" if files and all(path.lower().endswith(".gguf")
                                        for path, _ in files)
                else "safetensors"
            )
            attrs = {"file_layout": SINGLE_FILE, "file_type": source_type}
            if source_type == "gguf":
                strategy = "gguf"
        no_repackager = (
            strategy == "aio_singlefile" and provider == "huggingface"
            and _plan_has_no_repackager(plan)
        )
    except Exception:  # noqa: BLE001 — preflight is best-effort on odd plans
        return
    if source_bytes <= 0:
        return

    if strategy in _PUBLISH_AS_IS_STRATEGIES or no_repackager:
        required = source_bytes + _DISK_MARGIN_BYTES
        operation = f"{strategy} publishes the source tree directly"
    else:
        source_layout = attrs.get("file_layout", "")
        source_dtype = attrs.get("dtype", "")
        source_type = attrs.get("file_type", "") or (
            "gguf" if any(path.lower().endswith(".gguf") for path, _ in files)
            else "safetensors"
        )
        passthrough = [
            spec for spec in specs
            if source_layout and spec.file_layout == source_layout
            and spec.file_type == source_type
            and (spec.dtype == "source" or (source_dtype and spec.dtype == source_dtype))
        ]
        materialized = [spec for spec in specs if spec not in passthrough]
        shard_groups: dict[str, int] = {}
        for path, size in files:
            m = _SHARD_MEMBER_RE.match(path)
            if m:
                shard_groups[m.group("group")] = shard_groups.get(m.group("group"), 0) + size
        deshard_bytes = len(passthrough) * max(shard_groups.values(), default=0)
        measured_bits = int(getattr(plan, "source_storage_bits", 0) or 0)
        source_bits = measured_bits or _DTYPE_STORAGE_BITS.get(
            source_dtype, _UNRESOLVED_SOURCE_BITS)
        output_sizes = [
            _output_tree_bytes(spec, source_bytes, source_bits, measured_bits)
            for spec in materialized
        ]
        gguf_intermediate = max(
            (
                (source_bytes * 16 + source_bits - 1) // source_bits
                for spec in materialized
                if spec.file_type == "gguf"
                and spec.dtype not in _DIRECT_GGUF_ENCODINGS
            ),
            default=0,
        )
        repack = max(
            (
                size for spec, size in zip(materialized, output_sizes)
                if spec.file_type != "gguf"
                and source_layout and spec.file_layout != source_layout
            ),
            default=0,
        )
        required = (
            source_bytes + sum(output_sizes) + gguf_intermediate
            + deshard_bytes + repack + _DISK_MARGIN_BYTES
        )
        parts = []
        if passthrough:
            parts.append("hardlink passthrough")
        if materialized:
            parts.append(f"{len(materialized)} materialized output tree(s)")
        if deshard_bytes:
            parts.append("one merged de-shard output")
        if gguf_intermediate:
            parts.append("one intermediate F16 GGUF tree")
        if repack:
            parts.append("one layout-repack tree")
        if (materialized and not measured_bits
                and source_dtype not in _DTYPE_STORAGE_BITS):
            parts.append(
                "source dtype unreadable, assumed "
                f"{_UNRESOLVED_SOURCE_BITS}-bit")
        operation = ", ".join(parts) or "source tree"

    free = shutil.disk_usage(workdir).free
    if free < required:
        gib = float(1024**3)
        raise CloneDiskSpaceError(
            f"not enough disk for clone: need ~{required / gib:.1f} GiB free "
            f"(source {source_bytes / gib:.1f} GiB; {operation}; "
            f"{_DISK_MARGIN_BYTES / gib:.0f} GiB margin), have {free / gib:.1f} GiB "
            f"at {workdir}")

def _reusable_flavor_tree(
    workdir: Path, spec_label: str, flavor_dir: Path,
) -> Optional[dict[str, str]]:
    if not flavor_dir.is_dir():
        return None
    recovery = ProducerRecovery(workdir / JOURNAL_NAME)
    entry = recovery.find(spec_label=str(spec_label), tree=str(flavor_dir))
    if entry is None:
        return None
    attrs = entry.producer_state.get("attrs")
    if not isinstance(attrs, dict):
        return None
    if not entry.declares([f.path for f in files_from_tree(flavor_dir)]):
        logger.info(
            "flavor-%s: retained tree no longer matches publish %s's declaration; "
            "rebuilding", spec_label, entry.session_id)
        return None
    logger.warning(
        "flavor-%s: REUSING the retained cast output for publish %s — "
        "re-uploading rather than re-casting (pgw#1003)",
        spec_label, entry.session_id)
    return {str(k): str(v) for k, v in attrs.items()}


def _sweep_stale_workdirs(base: Path, *, keep: Optional[Path] = None) -> None:
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
                continue
            shutil.rmtree(d, ignore_errors=True)
            logger.info("swept stale clone scratch: %s", d)
        finally:
            os.close(fd)


def _clone_workdir(provider: str, source_key: str, destination: str) -> Path:
    base = Path(os.environ.get("COZY_CONVERT_WORKDIR", "").strip()
                or Path(tempfile.gettempdir()) / "gen-worker-convert")
    digest = hashlib.sha256(
        f"{provider}|{source_key}|{destination}".encode("utf-8")).hexdigest()[:16]
    workdir = base / f"clone-{digest}"
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _acquire_workdir_lock(workdir: Path) -> int:
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
    destination_release: str = "",
    target_layout: str | None = None,
    source_dtype_preference: list[str] | None = None,
    outputs: Iterable[Any] | None = None,
    quantize_components: list[str] | None = None,
    overwrite_repo: bool = False,
    gguf_quant: str | None = None,
    hf_token: str | None = None,
    civitai_api_key: str | None = None,
    source_include: Any = None,
    objective: str | None = None,
    distilled: bool = False,
) -> CloneResult:

    provider = str(provider or "").strip().lower()
    destination = normalize_destination_ref(destination_repo)
    release = _destination_release(ctx, destination_release, destination)
    layout_hint = str(target_layout or MULTI_FILE).strip().lower() or MULTI_FILE
    specs = normalize_outputs(outputs, layout_hint=layout_hint)
    include = normalize_source_include(source_include)
    objective_fact = str(objective or "").strip().lower()
    if objective_fact and objective_fact not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES} (or unset), got {objective!r}")
    distilled_fact = bool(distilled)
    if include and provider != "huggingface":
        raise ValueError("source_include is only supported for provider='huggingface'")
    explicit_outputs = bool(outputs)
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
    keepalive: Optional[HubKeepalive] = None
    try:
        if provider not in {"huggingface", "civitai"}:
            raise ValueError(f"unsupported clone provider: {provider!r}")

        hubclient = HubClient.from_ctx(ctx)
        mode = "replace" if overwrite_repo else "merge"

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
                    source_include=include,
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

        _preflight_disk(workdir, plan, specs)

        _progress(0.05, "clone.ingest")
        dl_bytes = {"done": 0}

        def _dl_progress(done: int, total: Optional[int]) -> None:
            dl_bytes["done"] = max(dl_bytes["done"], int(done or 0))
            if total:
                _progress(0.05 + 0.45 * min(1.0, done / total), "clone.download")

        keepalive = HubKeepalive(
            hubclient, hubclient._repo_path(destination),
            log=getattr(ctx, "log", None))
        keepalive.start()

        if provider == "huggingface":
            source = ingest_huggingface(
                source_ref, workdir / "source",
                revision=source_revision,
                dtype_preference=source_dtype_preference,
                gguf_quant=gguf_quant,
                hf_token=effective_hf_token,
                progress=_dl_progress,
                plan=plan,
                source_include=include,
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
        provenance = {"upstream_revision": str(source.source_revision or "")}
        strategy = source.classification.strategy if source.classification is not None else ""
        declared = repackage_family(source.model_family)
        no_repackager = strategy == "aio_singlefile" and (
            declared is None or not declared.supports_singlefile_to_diffusers)
        publish_as_is = strategy in _PUBLISH_AS_IS_STRATEGIES or no_repackager

        for i, spec in enumerate(specs):
            dtype_label = spec.dtype
            try:
                if publish_as_is:
                    source_dtype = str(source.attrs.get("dtype") or "").strip().lower()
                    dtype_matches = (not source_dtype) or spec.dtype == source_dtype
                    if dtype_matches or not explicit_outputs:
                        tree = source.dir
                        attrs = dict(source.attrs)
                        dtype_label = source_dtype or spec.dtype
                        if spec.file_type == "safetensors" \
                                and tree_has_sharded_safetensors(Path(tree)):
                            deshard_dir = workdir / f"flavor-{spec.label}.__deshard__"
                            shutil.rmtree(deshard_dir, ignore_errors=True)
                            deshard_dir.mkdir(parents=True, exist_ok=True)
                            copy_non_weight_files(Path(tree), deshard_dir, skip_components=set())
                            deshard_mirror_tree(deshard_dir)
                            tree = deshard_dir
                    elif i == 0 and spec.file_type == "safetensors" \
                            and (strategy in _CAST_ELIGIBLE_PUBLISH_AS_IS_STRATEGIES
                                 or no_repackager):
                        effective_layout = (
                            source.layout if source.layout in _KNOWN_FILE_LAYOUTS
                            else SINGLE_FILE
                        )
                        cast_spec = OutputSpec(
                            dtype=spec.dtype, file_layout=effective_layout,
                            file_type=spec.file_type,
                        )
                        flavor_dir = workdir / f"flavor-{spec.label}"
                        shutil.rmtree(flavor_dir, ignore_errors=True)
                        shutil.rmtree(workdir / f"flavor-{spec.label}.__repack__",
                                      ignore_errors=True)
                        tree, attrs = build_flavor_tree(
                            source, cast_spec, flavor_dir,
                            quantize_components=quantize_components,
                            objective=objective_fact,
                            distilled=distilled_fact,
                        )
                        dtype_label = str(attrs.get("dtype") or spec.dtype)
                    else:
                        raise InlineConversionNotPossible(
                            reason=f"{strategy} sources publish as-is; "
                                   f"run a conversion job for {spec.dtype}",
                            target_dtype=spec.dtype,
                        )
                else:
                    flavor_dir = workdir / f"flavor-{spec.label}"
                    reused = _reusable_flavor_tree(workdir, spec.label, flavor_dir)
                    if reused is not None:
                        tree, attrs = flavor_dir, reused
                    else:
                        shutil.rmtree(flavor_dir, ignore_errors=True)
                        shutil.rmtree(workdir / f"flavor-{spec.label}.__repack__",
                                      ignore_errors=True)
                        tree, attrs = build_flavor_tree(
                            source, spec, flavor_dir,
                            quantize_components=quantize_components,
                            objective=objective_fact,
                            distilled=distilled_fact,
                        )
                    dtype_label = str(attrs.get("dtype") or spec.dtype)
                dtype_label = _dtype_token(dtype_label)
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
                    "input_rejection": isinstance(exc, (ValueError, ValidationError)),
                })
                continue

            if spec.file_type != "gguf" and dtype_label in _CAST_NORMALIZE_DTYPES:
                _normalize_variant_filenames(Path(tree))

            files = files_from_tree(tree)
            if not files:
                result.failed_flavors.append({
                    "spec_label": spec.label, "dtype": spec.dtype,
                    "file_type": spec.file_type, "reason": "flavor tree is empty",
                })
                continue

            try:
                produced_dtypes = verify_produced_tree(
                    tree, source_dir=Path(source.dir))
            except Exception as exc:  # noqa: BLE001 — one flavor, not the run
                result.failed_flavors.append({
                    "spec_label": spec.label, "dtype": spec.dtype,
                    "file_type": spec.file_type, "reason": str(exc),
                    "component_dtype_pin_violation": True,
                })
                continue

            metadata: dict[str, Any] = {k: v for k, v in source.metadata.items()}
            try:
                from .size_walk import compute_size_facts

                facts = compute_size_facts(str(tree))
                if facts.get("full_model_bytes"):
                    metadata["size_facts"] = facts
            except Exception:
                pass
            if produced_dtypes:
                metadata["component_dtypes"] = dict(produced_dtypes)
            for k, v in attrs.items():
                metadata.setdefault(f"attr_{k}", str(v))

            _progress(0.55 + 0.4 * (i / max(1, len(specs))), f"clone.publish.{spec.label}")
            commit = hubclient.publish_v2(
                destination_repo=destination,
                files=files,
                release=release,
                mode=mode if i == 0 else "merge",
                dtype=str(attrs.get("dtype") or spec.dtype),
                file_layout=str(attrs.get("file_layout") or spec.file_layout),
                file_type=str(attrs.get("file_type") or spec.file_type),
                objective=objective_fact,
                distilled=distilled_fact,
                metadata=metadata,
                provenance=provenance,
                repo_spec=source.repo_spec,
                journal_path=workdir / JOURNAL_NAME,
                journal_state={
                    "spec_label": str(spec.label),
                    "tree": str(tree),
                    "attrs": {str(k): str(v) for k, v in attrs.items()},
                },
            )
            result.published.append({
                "dtype": dtype_label,
                "spec_label": spec.label,
                "revision_id": commit.revision_id,
                "checkpoint_id": commit.checkpoint_id,
                "uploaded": commit.uploaded,
                "deduped": commit.deduped,
                "total_bytes": commit.total_bytes,
            })

        if not result.published:
            reasons = "; ".join(
                str(f.get("reason") or "") for f in result.failed_flavors
            ) or "no output spec produced anything"
            if result.failed_flavors and all(f.get("input_rejection") for f in result.failed_flavors):
                raise ValidationError(f"clone produced no publishable flavor: {reasons}")
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
        if keepalive is not None:
            keepalive.stop()
            if keepalive.longest_outage_s or keepalive.reachable is False:
                logger.warning(
                    "clone hub keepalive: %d probes, longest observed outage "
                    "%.0fs, reachable at exit=%s",
                    keepalive.probes, keepalive.longest_outage_s,
                    keepalive.reachable)
        resumable = (
            0 if succeeded else ProducerRecovery(workdir / JOURNAL_NAME).count()
        )
        if resumable:
            logger.warning(
                "clone failed with %d publish session(s) still resumable; "
                "RETAINING %s so a retry re-uploads instead of re-cloning "
                "(swept after COZY_CONVERT_SCRATCH_TTL_S)", resumable, workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)
            if not succeeded:
                logger.warning("clone failed; workdir %s removed", workdir)
        os.close(lock_fd)


def from_huggingface(ctx: Any, payload: Any, *, hf_token: str | None = None) -> CloneResult:
    """Clone a Hugging Face repo end-to-end: download, convert, publish."""
    return run_clone(
        ctx,
        provider="huggingface",
        source_ref=str(getattr(payload, "huggingface_repo", "") or ""),
        source_revision=getattr(payload, "source_revision", None),
        destination_repo=str(getattr(payload, "destination_repo", "") or ""),
        destination_release=_payload_destination_release(payload),
        target_layout=getattr(payload, "target_layout", None),
        source_dtype_preference=getattr(payload, "source_dtype_preference", None),
        outputs=getattr(payload, "outputs", None),
        quantize_components=getattr(payload, "quantize_components", None),
        overwrite_repo=bool(getattr(payload, "overwrite_repo", False)),
        gguf_quant=getattr(payload, "gguf_quant", None),
        hf_token=hf_token,
        source_include=getattr(payload, "source_include", None),
        objective=getattr(payload, "objective", None),
        distilled=bool(getattr(payload, "distilled", False)),
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
        destination_release=_payload_destination_release(payload),
        target_layout=getattr(payload, "target_layout", None),
        outputs=getattr(payload, "outputs", None),
        quantize_components=getattr(payload, "quantize_components", None),
        overwrite_repo=bool(getattr(payload, "overwrite_repo", False)),
        gguf_quant=getattr(payload, "gguf_quant", None),
        civitai_api_key=civitai_api_key,
        objective=getattr(payload, "objective", None),
        distilled=bool(getattr(payload, "distilled", False)),
    )


__all__ = [
    "CloneResult",
    "OutputSpec",
    "build_flavor_tree",
    "from_civitai",
    "from_huggingface",
    "normalize_destination_ref",
    "normalize_outputs",
    "normalize_source_include",
    "run_clone",
]
