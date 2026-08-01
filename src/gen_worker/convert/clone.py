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

from gen_worker.api.errors import ValidationError

from .hub import HubClient, files_from_tree
from .keepalive import HubKeepalive
from .ingest import (
    IngestedSource,
    ingest_civitai,
    ingest_huggingface,
    plan_civitai,
    plan_huggingface,
)
from .writer import (
    CAST_NORMALIZE_DTYPES as _CAST_NORMALIZE_DTYPES,
    fp8_default_components,
    apply_objective_scheduler_config,
    copy_non_weight_files,
    merge_safetensors_by_offset,
    normalize_variant_filenames as _normalize_variant_filenames,
    snapshot_weight_groups,
)
from .convert import run_inline_conversion
from .layout import canonical_model_family_from_variant, infer_model_family_variant_from_hint
from .registry import repackage_family
from ..api.slot import OBJECTIVES
from gen_worker.models.refs import flavor_token

logger = logging.getLogger(__name__)

_PUBLIC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
# An HF shard member; `group` is dir + prefix, i.e. the set it belongs to.
_SHARD_MEMBER_RE = re.compile(
    r"^(?P<group>.*?[^/]+)-\d{5}-of-\d{5}\.safetensors$")
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
from ..component_vocab import quant_candidate_components
_KNOWN_FILE_LAYOUTS = {"diffusers", "singlefile"}
_KNOWN_FILE_TYPES = {"safetensors", "gguf"}

_default_quant_components = quant_candidate_components
_MIN_CONVERT_BYTES = 100 * 1024 * 1024  # leave tiny weights (embeddings) untouched


@dataclass(frozen=True)
class OutputSpec:
    """One requested output flavor: dtype + file layout + container."""

    dtype: str
    file_layout: str
    file_type: str

    @property
    def label(self) -> str:

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


def normalize_source_include(value: Any) -> tuple[str, ...]:
    """gw#593 item 2: dual-form clone-request field disambiguating a
    multi-checkpoint-bundle source repo (e.g. Lightricks/LTX-2.3's
    dev/distilled/lora/upscaler root bundle) — compact form is a single glob
    string, structured form is a list of globs. Both mean the same thing:
    an explicit allowlist matched against repo-relative paths, applied
    before classification (see :func:`gen_worker.convert.classifier.
    apply_source_include`). ``None``/empty means "today's heuristic,
    unrestricted" — the default, unchanged behavior.
    """
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

def _deshard_indexed_safetensors(index_path: Path) -> Path:
    """Collapse ONE HF shard set into a single ``<prefix>.safetensors``.

    th#1362: repos WE pull are normalised to one file per component on the way
    in, including pure pass-through mirrors, so the corpus we own has one shape.
    Chunked CAS already gives resumable, parallel, partial-failure-tolerant
    transfer BELOW the file, so the shard set buys nothing and costs an index
    that can disagree with the bytes it names — the bug that made klein-4b
    publish an unloadable text_encoder.

    Provenance is unaffected: upstream per-file digests are recorded by the
    provenance layer regardless of the layout we store.
    """
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid safetensors index: {index_path}")
    member_names: list[str] = []
    for name in weight_map.values():
        name = str(name)
        if Path(name).name != name:
            raise ValueError(
                f"safetensors index member must be a basename: {index_path}")
        if name not in member_names:
            member_names.append(name)
    members = [index_path.parent / name for name in member_names]
    missing = [m for m in members if not m.is_file()]
    if missing:
        raise ValueError(f"safetensors index references a missing shard: {index_path}")

    prefix = index_path.name.removesuffix(".safetensors.index.json")
    merged = index_path.parent / f"{prefix}.safetensors"
    if merged.exists() and merged not in members:
        raise ValueError(f"deshard destination already exists: {merged}")
    merge_safetensors_by_offset(members, merged)

    # The merged header must name exactly the tensors the index claimed — the
    # index-vs-bytes disagreement this whole change exists to kill is caught
    # here, at ingest, instead of at load time on a GPU pod.
    with open(merged, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
    got = {k for k in header if k != "__metadata__"}
    want = {str(k) for k in weight_map}
    if got != want:
        merged.unlink(missing_ok=True)
        raise ValueError(
            f"deshard of {index_path} produced {len(got)} tensors, index names "
            f"{len(want)} (missing={sorted(want - got)[:5]}, "
            f"extra={sorted(got - want)[:5]})")

    for member in members:
        if member != merged:
            member.unlink()
    index_path.unlink()
    logger.info("deshard_mirror index=%s shards=%d -> %s",
                index_path.name, len(members), merged.name)
    return merged


def tree_has_sharded_safetensors(tree: Path) -> bool:
    """Does this tree carry an HF shard set that mirror ingest must collapse?"""
    return any(Path(tree).rglob("*.safetensors.index.json"))


def deshard_mirror_tree(tree: Path) -> int:
    """De-shard every HF shard set in an ingested mirror tree, in place.

    One component at a time, unlinking each set's members as soon as its merged
    file is verified, so the peak extra disk is the LARGEST component and not
    the whole tree.
    """
    n = 0
    for index_path in sorted(Path(tree).rglob("*.safetensors.index.json")):
        _deshard_indexed_safetensors(index_path)
        n += 1
    return n


def build_flavor_tree(
    source: IngestedSource,
    spec: OutputSpec,
    out_dir: Path,
    *,
    quantize_components: list[str] | None = None,
    objective: str = "",
    distilled: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Materialize one output flavor as a local file tree.

    Passthrough (dtype/layout/container match the source) hardlinks the
    snapshot. Otherwise: optional layout repackage, then per-weight-set
    cast / quant / gguf via :mod:`gen_worker.convert.convert`.

    ``objective``/``distilled`` (pgw#654) stamp objective-correct scheduler
    config into the produced tree's ``scheduler/config.json`` (no-op for an
    epsilon/unstamped non-distilled checkpoint or a layout with no
    scheduler component).

    Returns ``(tree_root, attrs)``. Raises ``InlineConversionNotPossible``
    for calibrated dtypes.
    """

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
        deshard_mirror_tree(out_dir)
        if source_dtype in _CAST_NORMALIZE_DTYPES:
            _normalize_variant_filenames(out_dir)
        apply_objective_scheduler_config(out_dir, objective, distilled)
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
        from .repackage import diffusers_to_singlefile, singlefile_to_diffusers

        family = str(source.model_family or "").strip().lower()
        # pgw#740 (B16): capability is registry membership, not a hand-synced
        # allowlist. The declaration also states the DIRECTION it supports, so
        # a family that can only go singlefile->diffusers is refused for the
        # reverse instead of failing deep inside the converter.
        declared = repackage_family(family)
        if declared is None or not (
            declared.supports_singlefile_to_diffusers if source_layout == "singlefile"
            else declared.supports_diffusers_to_singlefile
        ):
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
        deshard_mirror_tree(out_dir)
        if spec.dtype in _CAST_NORMALIZE_DTYPES:
            _normalize_variant_filenames(out_dir)
        apply_objective_scheduler_config(out_dir, objective, distilled)
        return out_dir, attrs

    # Dtype conversion per weight set.
    groups = snapshot_weight_groups(work_root, work_layout)
    is_fp8 = spec.dtype == "fp8"
    fp8_block_scope = False
    if is_fp8 and work_layout != "diffusers":
        # One root weight set = a transformers backbone (the whole checkpoint
        # IS the denoiser, e.g. HiDream-O1's UiT): block-scoped fp8 cast.
        # Multi-set singlefile bundles still refuse — component identity is
        # ambiguous there.
        if len(groups) != 1 or groups[0][0] != "":
            raise ValueError(
                "fp8 storage flavors need component identity: a diffusers "
                "layout, or a single root weight set (transformers "
                f"backbone) — found {len(groups)} weight set(s)")
        fp8_block_scope = True
    if quantize_components:
        target_names = set(quantize_components)
    elif is_fp8:
        # Denoiser-only by default: matches apply_fp8_storage's consumption
        # scope; TEs join explicitly via the component-wise ladder (gw#392).
        target_names = set(fp8_default_components())
    else:
        target_names = set(_default_quant_components())
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
    if work_root is not source_dir:
        shutil.rmtree(work_root, ignore_errors=True)
    return out_dir, attrs


# ---------------------------------------------------------------------------
# run_clone — ingest, convert, ONE finalize path
# ---------------------------------------------------------------------------

class CloneDiskSpaceError(RuntimeError):
    """Preflight found too little free disk for the clone — fail fast and
    actionably instead of ENOSPC minutes into a 40GB download (gw#462)."""


_DISK_MARGIN_BYTES = 2 * 1024**3
_PUBLISH_AS_IS_STRATEGIES = frozenset({
    "transformers", "peft", "sentence_transformers", "gguf", "native_lora",
    "pipeline_tree", "diffusers_component",
})
# th#901: publish_as_is strategies whose weight set is an ordinary dense
# safetensors tree (not a binary quant container like gguf, and not a
# mixed-dtype multi-checkpoint bundle like pipeline_tree) — a genuinely
# mismatched requested dtype is real, in-line-castable work via
# build_flavor_tree, not something to silently swallow or unconditionally
# refuse.
_CAST_ELIGIBLE_PUBLISH_AS_IS_STRATEGIES = frozenset({
    "transformers", "peft", "sentence_transformers", "native_lora",
    "diffusers_component",
})
_DIRECT_GGUF_ENCODINGS = frozenset({"f32", "f16", "bf16", "q8_0"})
_DTYPE_STORAGE_BITS = {
    "fp32": 32, "f32": 32, "float32": 32,
    "bf16": 16, "fp16": 16, "f16": 16, "float16": 16,
    "fp8": 8, "fp8:e5m2": 8, "q8_0": 8,
    "q6_k": 6,
    "q5_k_m": 5, "q5_k_s": 5,
    "nvfp4": 4, "int4": 4, "int4:nf4": 4, "int4:fp4": 4,
    "nf4": 4, "fp4": 4, "q4_k_m": 4, "q4_k_s": 4, "q4_0": 4,
    "q4_1": 4,
    "q3_k_m": 3, "q3_k_s": 3,
    "q2_k": 2,
}


def _plan_has_no_repackager(plan: Any) -> bool:
    """Pre-download capability probe, mirroring run_clone's post-download check.

    Uses ONLY the file-listing paths ``plan_huggingface`` already has (no bytes
    fetched yet): a checkpoint filename carrying a family token resolves the
    family through the same declared hint matchers ``detect_huggingface_source_layout``
    uses post-download.

    pgw#740 (B16): this used to be spelled ``_hf_plan_looks_like_ltx2`` — one
    family named twice in this file to route AROUND the repackager. A source
    whose family declares no singlefile->diffusers converter simply has no
    repackager, LTX-2 or otherwise, and publishes as-is."""

    for p in getattr(plan, "paths", None) or ():
        variant = infer_model_family_variant_from_hint(str(p))
        if variant == "unknown":
            continue
        declared = repackage_family(canonical_model_family_from_variant(variant))
        return declared is None or not declared.supports_singlefile_to_diffusers
    return False


def _preflight_disk(workdir: Path, plan: Any, specs: list[OutputSpec]) -> None:
    """Fail fast when the disk cannot fit the clone. The source plan knows
    every selected file's size before a byte is downloaded (HF list_repo_tree
    / civitai version API). The bound covers plan-known files; a repackage
    tool that fetches missing base components can still fail on its provider
    download. An unavailable plan skips the check (fail-open)."""
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
            attrs = {"file_layout": "singlefile", "file_type": source_type}
            if source_type == "gguf":
                strategy = "gguf"
        # gw#592/gw#593: run_clone routes a strategy="aio_singlefile" source
        # whose family declares no singlefile->diffusers converter through
        # publish_as_is regardless of the requested output layout — but this
        # preflight only sees the pre-download classification, which has no
        # no-repackager concept, so it was estimating a full layout-repack +
        # materialized-dtype-tree budget (388GB for a 43GB source) for a
        # clone that actually only ever needs the source bytes + margin.
        # Found live: e2e#185 ltx-firstlight run 7, CloneDiskSpaceError on a
        # 200GB pod for a 43GB LTX-2.3 dev-checkpoint clone.
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
        # th#1362: mirror ingest de-shards, so the transient cost is the
        # merged copy of the LARGEST shard set — its members are unlinked as
        # soon as it verifies, one component at a time.
        shard_groups: dict[str, int] = {}
        for path, size in files:
            m = _SHARD_MEMBER_RE.match(path)
            if m:
                shard_groups[m.group("group")] = shard_groups.get(m.group("group"), 0) + size
        deshard_bytes = len(passthrough) * max(shard_groups.values(), default=0)
        # Untagged safetensors may still be a packed 4-bit tree. Mirrors do
        # not use this estimate; explicit widening conversions do.
        source_bits = _DTYPE_STORAGE_BITS.get(source_dtype, 4)
        output_sizes = [
            source_bytes if _DTYPE_STORAGE_BITS.get(spec.dtype, source_bits) <= source_bits
            else (
                source_bytes * _DTYPE_STORAGE_BITS[spec.dtype] + source_bits - 1
            ) // source_bits
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
        operation = ", ".join(parts) or "source tree"

    free = shutil.disk_usage(workdir).free
    if free < required:
        gib = float(1024**3)
        raise CloneDiskSpaceError(
            f"not enough disk for clone: need ~{required / gib:.1f} GiB free "
            f"(source {source_bytes / gib:.1f} GiB; {operation}; "
            f"{_DISK_MARGIN_BYTES / gib:.0f} GiB margin), have {free / gib:.1f} GiB "
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
    source_include: Any = None,
    objective: str | None = None,
    distilled: bool = False,
) -> CloneResult:

    provider = str(provider or "").strip().lower()
    destination = normalize_destination_ref(destination_repo)
    tags = normalize_tags(destination_repo_tags)
    layout_hint = str(target_layout or "diffusers").strip().lower() or "diffusers"
    specs = normalize_outputs(outputs, layout_hint=layout_hint)
    include = normalize_source_include(source_include)
    objective_fact = str(objective or "").strip().lower()
    if objective_fact and objective_fact not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES} (or unset), got {objective!r}")
    distilled_fact = bool(distilled)
    if include and provider != "huggingface":
        raise ValueError("source_include is only supported for provider='huggingface'")
    # th#901: normalize_outputs collapses "caller asked for nothing" onto a
    # schema default (dtype="bf16") — indistinguishable from an EXPLICIT
    # bf16 request once normalized. Only an explicit request may force a
    # publish_as_is source through a real conversion; an unspecified request
    # keeps mirroring the source's own dtype untouched (no cast nobody asked
    # for).
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

        # Derive the source's identity from provider metadata alone (no
        # bytes) — the disk preflight below needs every selected file's size.
        #
        # This used to ALSO drive th#592's download-skip: a bank key over that
        # metadata, a banked manifest lookup, and a publish BY CAS REFERENCE
        # that downloaded nothing. That path is gone with the v1 protocol
        # (pgw#807): its adds carried a caller-asserted blake3 and no local
        # bytes, which is precisely what the chunked-sha256 route cannot and
        # must not accept — a claimed digest stops being assertable. It had
        # also been dead in practice since the hub froze v1 writes.
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

        # gw#462: the plan already knows every selected file's size — fail
        # fast on an undersized disk instead of ENOSPC mid-download.
        _preflight_disk(workdir, plan, specs)

        _progress(0.05, "clone.ingest")
        dl_bytes = {"done": 0}

        def _dl_progress(done: int, total: Optional[int]) -> None:
            dl_bytes["done"] = max(dl_bytes["done"], int(done or 0))
            if total:
                _progress(0.05 + 0.45 * min(1.0, done / total), "clone.download")

        # pgw#743: download and cast are the two phases that talk to the hub
        # NOT AT ALL — an hour of them, and then the first upload discovers
        # whatever happened to the path in the meantime. Both losses of a paid
        # 53 GiB download landed exactly there. Keep the path warm and, either
        # way, keep an actual record of when the hub was reachable.
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
        # th#606: upstream identity (upstream_ref + derivation_op=import) is
        # orchestrator-derived and rides the capability token; the worker only
        # ADDS the revision it actually resolved during download.
        provenance = {"upstream_revision": str(source.source_revision or "")}
        # Non-diffusers-class sources publish as-is; extra output specs that
        # would need conversion are refused per-flavor, not per-job.
        strategy = source.classification.strategy if source.classification is not None else ""
        # gw#592/pgw#740 (B16): a bare single/multi-root safetensors repo whose
        # family declares no singlefile->diffusers converter has nothing to
        # repackage INTO, so it publishes as-is rather than being handed to a
        # repackager with no rule for it. LTX-2 is the motivating case (the
        # te#70 trainer resolves its native singlefile snapshot directly, and
        # nobody consumes an "ltx2 diffusers layout") but the family name no
        # longer appears here — capability is registry membership.
        declared = repackage_family(source.model_family)
        no_repackager = strategy == "aio_singlefile" and (
            declared is None or not declared.supports_singlefile_to_diffusers)
        publish_as_is = strategy in _PUBLISH_AS_IS_STRATEGIES or no_repackager

        for i, spec in enumerate(specs):
            flavor_label = spec.dtype
            try:
                if publish_as_is:
                    source_dtype = str(source.attrs.get("dtype") or "").strip().lower()
                    dtype_matches = (not source_dtype) or spec.dtype == source_dtype
                    if dtype_matches or not explicit_outputs:
                        # Already satisfied, OR nobody actually asked for a
                        # specific dtype (normalize_outputs' schema default
                        # just landed on bf16) — mirror the source's own
                        # dtype untouched rather than force a cast nobody
                        # requested.
                        tree = source.dir
                        attrs = dict(source.attrs)
                        flavor_label = source_dtype or spec.dtype
                        # th#1362: this passthrough bypasses build_flavor_tree
                        # entirely (every one of ITS branches de-shards), so
                        # de-shard it here too — the ruling is EXPLICIT that
                        # pure pass-through mirrors are normalised as well, so
                        # that the corpus we own has ONE shape. Hardlink into a
                        # scratch tree first; only the sharded components are
                        # actually rewritten, everything else stays a link.
                        #
                        # An oversized MONOLITHIC file needs no handling at all
                        # any more: the old code resharded it because
                        # tensorhub's commit API rejected it
                        # (request_too_large, e2e#185 ltx-firstlight run 8 with
                        # LTX-2.3's 46 GB single file). The checkpoint grant is
                        # now sized for exactly this — 64 GiB per file, "single
                        # files up to full unsharded checkpoints" — and chunked
                        # CAS carries the transfer.
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
                        # th#901: an EXPLICITLY mismatched requested dtype is
                        # real, in-line-castable work for these strategies
                        # (ordinary dense safetensors trees) —
                        # build_flavor_tree already knows how to cast a
                        # single/few-weight-set tree. Never silently
                        # republish the source's own dtype under the
                        # requested flavor's label. These strategies publish
                        # as-is ORGANIZATIONALLY too — no layout repackage is
                        # attempted here (that stays a separate job) — so the
                        # cast targets the source's own on-disk layout, only
                        # the dtype changes.
                        effective_layout = (
                            source.layout if source.layout in _KNOWN_FILE_LAYOUTS
                            else "singlefile"
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
                        flavor_label = str(attrs.get("dtype") or spec.dtype)
                    else:
                        # Only the source's own flavor is published for
                        # classes this worker cannot cast in-line (quant/gguf
                        # containers, mixed-dtype multi-checkpoint bundles —
                        # those run as a separate job) or for i>0 extra
                        # specs. Fail loud instead of silently republishing
                        # under the wrong flavor label.
                        raise InlineConversionNotPossible(
                            reason=f"{strategy} sources publish as-is; "
                                   f"run a conversion job for {spec.dtype}",
                            target_dtype=spec.dtype,
                        )
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
                        objective=objective_fact,
                        distilled=distilled_fact,
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
                    # th#1084: spec/source combination rejections are input
                    # verdicts, not conversion faults.
                    "input_rejection": isinstance(exc, (ValueError, ValidationError)),
                })
                continue

            # gw#522: EVERY publish path emits canonical filenames — this
            # seam also covers the publish-as-is lane build_flavor_tree
            # never touches. Idempotent on already-normalized trees.
            if spec.file_type != "gguf" and flavor_label in _CAST_NORMALIZE_DTYPES:
                _normalize_variant_filenames(Path(tree))

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
            # th#1303 phase 3: the mirror is the FIRST producer class flipped to
            # the chunked sha256 CAS, chosen for blast radius — a mirror is
            # re-runnable from upstream, so a bad publish costs a re-clone and
            # never an unrecoverable artifact. Every file here is a real local
            # file (`files_from_tree`), which v2 requires: its guarantee is that
            # a digest is PROVEN from bytes in hand. The BANK path at :556 keeps
            # `commit()` precisely because its adds are by-reference.
            #
            # `message` has no v2 equivalent and is NOT silently dropped — it
            # was `clone {provider}:{ref}@{revision}`, which is exactly
            # `upstream_ref` + `upstream_revision` in the provenance stamp the
            # hub now resolves authoritatively (th#1331). The fact moved to a
            # structured, queryable home; it did not disappear.
            #
            # A `merge` into a prior v1/blake3 manifest is a TYPED refusal
            # (`mixed_algorithm_manifest`) rather than a silent partial — the
            # right behaviour until the phase-2 backfill re-keys the repo.
            commit = hubclient.publish_v2(
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

        if not result.published:
            reasons = "; ".join(
                str(f.get("reason") or "") for f in result.failed_flavors
            ) or "no output spec produced anything"
            # th#1084: when every failure was an input-combination rejection
            # (e.g. dtype="source" with a non-safetensors source), the verdict
            # is about the request, not the release — INVALID, not FATAL.
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
        destination_repo_tags=getattr(payload, "destination_repo_tags", None),
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
    "normalize_tags",
    "run_clone",
]
