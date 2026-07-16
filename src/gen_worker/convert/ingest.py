"""Hub-API ingest: materialize a source model snapshot locally.

HuggingFace: ``HfApi.list_repo_files`` → :mod:`gen_worker.convert.classifier` →
``snapshot_download(allow_patterns=...)``. Civitai: the provider-bounded
fetch from ``gen_worker.models.download`` plus clone-side metadata
(baseModel lineage, kohya hints). No arbitrary-URL sources.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..net import hf, install_hf_http_timeouts
from .classifier import RepoClassification, classify_repo
from .layout import detect_huggingface_source_layout

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, Optional[int]], None]


class CloneDownloadError(RuntimeError):
    """Source download failed after bounded retries (gw#456) — the clone job
    fails cleanly instead of hanging."""


_DOWNLOAD_ATTEMPTS_ENV = "COZY_CLONE_DOWNLOAD_ATTEMPTS"


def _download_attempts() -> int:
    raw = os.environ.get(_DOWNLOAD_ATTEMPTS_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 3

_SAFETENSORS_DTYPE_NAMES = {
    "F32": "fp32", "F16": "fp16", "BF16": "bf16",
    "F8_E4M3": "fp8", "F8_E5M2": "fp8:e5m2",
}
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _detect_snapshot_dtype(root: Path) -> str:
    """Majority weight dtype across a snapshot's safetensors headers
    ('bf16' / 'fp16' / 'fp32' / 'fp8', '' when undetectable). Civitai
    version metadata is unreliable (fp labels routinely contradict the
    file bytes), so read the headers."""
    import struct

    counts: dict[str, int] = {}
    try:
        for p in sorted(Path(root).rglob("*.safetensors")):
            with open(p, "rb") as f:
                raw = f.read(8)
                if len(raw) < 8:
                    continue
                (n,) = struct.unpack("<Q", raw)
                if n <= 0 or n > _MAX_SAFETENSORS_HEADER_BYTES:
                    continue
                header = json.loads(f.read(n))
            for value in header.values():
                if isinstance(value, dict) and "dtype" in value:
                    counts[str(value["dtype"])] = counts.get(str(value["dtype"]), 0) + 1
    except (OSError, ValueError):
        return ""
    if not counts:
        return ""
    top = max(counts, key=lambda k: counts[k])
    return _SAFETENSORS_DTYPE_NAMES.get(top, "")


@dataclass
class IngestedSource:
    """A materialized local snapshot + everything finalize needs to know."""

    provider: str                 # huggingface | civitai
    source_ref: str               # canonical repo id / civitai version id
    source_revision: str          # resolved commit sha / manifest hash
    dir: Path                     # local snapshot root
    layout: str                   # diffusers | singlefile | unknown
    model_family: str
    model_family_variant: str
    classification: Optional[RepoClassification] = None
    attrs: dict[str, str] = field(default_factory=dict)      # checkpoint attributes
    metadata: dict[str, str] = field(default_factory=dict)   # provenance metadata
    repo_spec: dict[str, str] = field(default_factory=dict)  # repo auto-create fields


def resolve_hf_identity(
    source_ref: str,
    *,
    revision: str | None = None,
    hf_token: str | None = None,
) -> tuple[str, str]:
    """Resolve (repo_id, commit_sha) via the HF hub API."""
    repo_id = str(source_ref or "").strip()
    if repo_id.count("/") != 1:
        raise ValueError(f"huggingface source ref must be org/name, got {source_ref!r}")
    info = hf().HfApi(token=(hf_token or None)).repo_info(
        repo_id, revision=(str(revision).strip() or None) if revision else None)
    return repo_id, str(getattr(info, "sha", "") or "")


def _hf_classification_inputs(
    repo_id: str,
    revision: str | None,
    hf_token: str | None,
) -> tuple[list[str], dict[str, int], dict[str, Any], dict[str, str]]:
    """One list_repo_tree walk: paths, sizes, small side signals, and the
    provider's per-file content ids (lfs sha256 / git blob oid) — the
    latter feed the th#592 download-skip bank key."""
    api = hf().HfApi(token=(hf_token or None))
    paths: list[str] = []
    sizes: dict[str, int] = {}
    content_ids: dict[str, str] = {}
    for entry in api.list_repo_tree(repo_id, revision=revision, recursive=True):
        path = str(getattr(entry, "path", "") or "")
        size = getattr(entry, "size", None)
        if not path or size is None:
            continue  # skip directory rows
        paths.append(path)
        sizes[path] = int(size or 0)
        lfs = getattr(entry, "lfs", None)
        sha = ""
        if lfs is not None:
            sha = str(getattr(lfs, "sha256", "") or "").strip().lower()
            if not sha and isinstance(lfs, dict):
                sha = str(lfs.get("sha256") or lfs.get("oid") or "").strip().lower()
        blob_id = str(getattr(entry, "blob_id", "") or "").strip()
        if sha:
            content_ids[path] = f"sha256:{sha}"
        elif blob_id:
            content_ids[path] = f"git:{blob_id}"

    side: dict[str, Any] = {}
    if "config.json" in paths:
        try:
            local = hf().hf_hub_download(repo_id, "config.json", revision=revision,
                                         token=(hf_token or None))
            side["config_json"] = json.loads(Path(local).read_text(encoding="utf-8"))
        except Exception:
            side["config_json"] = None

    root_st = [p for p in paths if "/" not in p and p.lower().endswith(".safetensors")]
    if root_st and "model_index.json" not in paths and "adapter_config.json" not in paths:
        # LoRA sniff — remote safetensors __metadata__ via the hub API.
        try:
            st_md = hf().get_safetensors_metadata(repo_id, revision=revision, token=(hf_token or None))
            for fmeta in (getattr(st_md, "files_metadata", None) or {}).values():
                md = getattr(fmeta, "metadata", None)
                if md:
                    side["safetensors_metadata"] = {str(k): str(v) for k, v in md.items()}
                    break
        except Exception:
            pass
        try:
            card = hf().ModelCard.load(repo_id, token=(hf_token or None))
            tags = getattr(card.data, "tags", None) or []
            side["readme_tags"] = [str(t) for t in tags]
        except Exception:
            pass
    return paths, sizes, side, content_ids


@dataclass
class HFSourcePlan:
    """Pre-download identity of one HF clone source (metadata calls only).

    Feeds the th#592 download-skip: ``content_ids`` carries the provider's
    own per-file content hashes (lfs.oid sha256 for LFS files, git blob oid
    for small files), so a bank key derives without downloading a byte.
    """

    repo_id: str
    revision: str                      # resolved commit sha
    paths: list[str]
    sizes: dict[str, int]
    side: dict[str, Any]
    classification: RepoClassification
    content_ids: dict[str, str]        # path -> "sha256:<hex>" | "git:<oid>"

    @property
    def provider(self) -> str:
        return "huggingface"

    @property
    def source_ref(self) -> str:
        return self.repo_id

    def bank_files(self) -> list[tuple[str, int, str]]:
        """(path, size, content_id) for every file the clone would download.
        Empty when any selected file lacks a content id (no safe key)."""
        out: list[tuple[str, int, str]] = []
        for p in self.classification.allow_patterns:
            cid = self.content_ids.get(p, "")
            if not cid:
                return []
            out.append((p, int(self.sizes.get(p, 0)), cid))
        return sorted(out)

    def bank_extra(self) -> dict[str, str]:
        attrs = {str(k): str(v) for k, v in (self.classification.attrs or {}).items()}
        return {
            "strategy": str(self.classification.strategy),
            "attrs": json.dumps(attrs, sort_keys=True, separators=(",", ":")),
        }


@dataclass
class CivitaiSourcePlan:
    """Pre-download identity of one civitai model version (one API call)."""

    version_id: int
    payload: dict[str, Any]
    files: list[dict[str, Any]]        # download.{name,size_bytes,sha256,...}
    revision: str                      # same manifest hash ingest_civitai mints

    @property
    def provider(self) -> str:
        return "civitai"

    @property
    def source_ref(self) -> str:
        return str(self.version_id)

    def bank_files(self) -> list[tuple[str, int, str]]:
        out: list[tuple[str, int, str]] = []
        for f in self.files:
            sha = str(f.get("sha256") or "").strip().lower()
            if not sha:
                return []
            out.append((str(f.get("name")), int(f.get("size_bytes") or 0), f"sha256:{sha}"))
        return sorted(out)

    def bank_extra(self) -> dict[str, str]:
        model = self.payload.get("model") if isinstance(self.payload.get("model"), dict) else {}
        return {
            "base_model": str(self.payload.get("baseModel") or ""),
            "model_type": str((model or {}).get("type") or ""),
        }


def _civitai_manifest_revision(file_names: list[str]) -> str:
    """The clone-side civitai 'revision': a hash over the downloaded file
    listing (civitai has no commit sha). MUST stay identical between
    plan_civitai and ingest_civitai so bank hits match full-clone runs."""
    import hashlib

    manifest = json.dumps(
        [{"name": n} for n in sorted(file_names)], sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def plan_huggingface(
    source_ref: str,
    *,
    revision: str | None = None,
    dtype_preference: list[str] | None = None,
    gguf_quant: str | None = None,
    hf_token: str | None = None,
) -> HFSourcePlan:
    """Resolve + classify one HF repo from metadata alone (no weight bytes)."""
    install_hf_http_timeouts()
    repo_id, sha = resolve_hf_identity(source_ref, revision=revision, hf_token=hf_token)
    rev = sha or (str(revision).strip() if revision else None)
    paths, sizes, side, content_ids = _hf_classification_inputs(repo_id, rev, hf_token)
    classification = classify_repo(
        paths,
        sizes=sizes,
        config_json=side.get("config_json"),
        safetensors_metadata=side.get("safetensors_metadata"),
        readme_tags=side.get("readme_tags") or (),
        dtype_pref=tuple(dtype_preference or ("bf16", "fp16", "fp32")),
        gguf_quant=gguf_quant,
    )
    return HFSourcePlan(
        repo_id=repo_id, revision=sha, paths=paths, sizes=sizes, side=side,
        classification=classification, content_ids=content_ids,
    )


def plan_civitai(
    model_version_id: int, *, civitai_api_key: str | None = None,
    gguf_quant: str | None = None,
) -> CivitaiSourcePlan:
    """Fetch one civitai model version's metadata (no downloads)."""
    from gen_worker.models.download import _civitai_select_files, fetch_civitai_model_version

    version_id = int(model_version_id or 0)
    if version_id <= 0:
        raise ValueError("civitai_model_version_id is required")
    payload = fetch_civitai_model_version(version_id, api_key=(civitai_api_key or ""))
    files = _civitai_select_files(payload, gguf_quant=gguf_quant)
    return CivitaiSourcePlan(
        version_id=version_id,
        payload=payload,
        files=files,
        revision=_civitai_manifest_revision([str(f.get("name")) for f in files]),
    )


def _snapshot_download_with_retries(
    repo_id: str,
    revision: str | None,
    dest_dir: Path,
    *,
    allow_patterns: list[str],
    hf_token: str | None,
    progress: ProgressFn | None,
    total_hint: Optional[int],
) -> None:
    """Bounded, resumable ``snapshot_download`` (gw#456): every socket has a
    timeout floor (:mod:`gen_worker.net`), the stall watchdog reports byte
    progress and aborts no-progress downloads, and transient network failures
    retry (hf_hub resumes ``.incomplete`` files via Range). Exhausted retries
    raise :class:`CloneDownloadError`."""
    snapshot_download = hf().snapshot_download
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    from gen_worker.models.download import (
        _HF_DOWNLOAD_MAX_SECONDS,
        _HF_DOWNLOAD_STALL_TIMEOUT_S,
        _run_with_stall_watchdog,
    )

    attempts = _download_attempts()
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _run_with_stall_watchdog(
                lambda: snapshot_download(
                    repo_id,
                    revision=revision,
                    local_dir=str(dest_dir),
                    allow_patterns=allow_patterns,
                    token=(hf_token or None),
                ),
                label=f"clone {repo_id}@{revision or 'main'}",
                progress_root=dest_dir,
                progress_callback=progress,
                total_hint=total_hint,
                stall_timeout=_HF_DOWNLOAD_STALL_TIMEOUT_S,
                wall_clock_max=_HF_DOWNLOAD_MAX_SECONDS,
            )
            return
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError,
                EntryNotFoundError, ValueError, TypeError):
            raise  # permanent — retrying cannot help
        except Exception as exc:
            # Transport errors: httpx exceptions, HfHubHTTPError (5xx/429),
            # raw socket OSErrors, DownloadStalledError — all bounded-retried.
            last = exc
            if attempt < attempts:
                logger.warning(
                    "clone download %s attempt %d/%d failed (%s: %s); retrying (resumable)",
                    repo_id, attempt, attempts, type(exc).__name__, exc)
                time.sleep(min(10.0, 2.0 * attempt))
    raise CloneDownloadError(
        f"clone download of {repo_id} failed after {attempts} attempt(s): "
        f"{type(last).__name__}: {last}") from last


def ingest_huggingface(
    source_ref: str,
    dest_dir: Path,
    *,
    revision: str | None = None,
    dtype_preference: list[str] | None = None,
    gguf_quant: str | None = None,
    hf_token: str | None = None,
    progress: ProgressFn | None = None,
    plan: HFSourcePlan | None = None,
) -> IngestedSource:
    """Classify + selectively download one HF repo into ``dest_dir``.

    ``plan`` (from :func:`plan_huggingface`) skips re-doing the metadata
    calls when the caller already planned this source."""
    install_hf_http_timeouts()
    if plan is None:
        plan = plan_huggingface(
            source_ref, revision=revision, dtype_preference=dtype_preference,
            gguf_quant=gguf_quant, hf_token=hf_token)
    repo_id, sha = plan.repo_id, plan.revision
    rev = sha or (str(revision).strip() if revision else None)
    paths, sizes = plan.paths, plan.sizes
    classification = plan.classification

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    selected_bytes = sum(int(sizes.get(p, 0)) for p in classification.allow_patterns)
    if progress is not None:
        progress(0, selected_bytes or None)
    _snapshot_download_with_retries(
        repo_id, rev, dest_dir,
        allow_patterns=list(classification.allow_patterns),
        hf_token=hf_token,
        progress=progress,
        total_hint=selected_bytes or None,
    )
    if progress is not None:
        progress(selected_bytes, selected_bytes or None)

    layout_info = detect_huggingface_source_layout(repo_dir=dest_dir, files=paths)
    library = classification.runtime_library
    library_name = {
        "diffusers-single-file": "diffusers",
        "diffusers-lora": "diffusers",
        "llama-cpp": "llama.cpp",
        # No hub-recognized loader library for TRELLIS-style pipeline trees:
        # publish with library_name unset (validator opt-out, e2e #112).
        "trellis2": "",
    }.get(library, library)
    repo_spec = {
        "kind": "adapter" if library in {"peft", "diffusers-lora"} else "model",
        "library_name": library_name,
        "model_family": str(layout_info.model_family or ""),
        "class_name": str(classification.attrs.get("architecture") or ""),
    }
    attrs = dict(classification.attrs)
    on_disk_dtype = _detect_snapshot_dtype(dest_dir)
    if on_disk_dtype:
        attrs["dtype"] = on_disk_dtype
    attrs.setdefault("runtime_library", library)
    metadata = {
        "source_provider": "huggingface",
        "source_repo": repo_id,
        "source_revision": sha,
        "ingest_strategy": classification.strategy,
        "runtime_library": library,
        "source_layout": str(layout_info.source_layout),
        "model_family": str(layout_info.model_family),
        "model_family_variant": str(layout_info.model_family_variant),
        "selected_count": str(len(classification.allow_patterns)),
        "selected_bytes": str(selected_bytes),
        "source_file_count": str(len(paths)),
    }
    if library == "diffusers-single-file" and _is_multi_weight_bundle(dest_dir):
        # Multi-component bundle (distinct component single-files, e.g.
        # chatterbox t3/s3gen/ve): no library loads it as one artifact —
        # opt out of the layout contract exactly like the civitai branch
        # (empty library_name skips finalize-side validation; e2e #112).
        repo_spec["library_name"] = ""
        metadata["multi_weight_bundle"] = "true"
    return IngestedSource(
        provider="huggingface",
        source_ref=repo_id,
        source_revision=sha,
        dir=dest_dir,
        layout=str(layout_info.source_layout),
        model_family=str(layout_info.model_family),
        model_family_variant=str(layout_info.model_family_variant),
        classification=classification,
        attrs=attrs,
        metadata=metadata,
        repo_spec=repo_spec,
    )


_SHARD_NAME_RE = None


def _is_multi_weight_names(names: Iterable[str]) -> bool:
    """Pure-name core of :func:`_is_multi_weight_bundle`: True when the
    TOP-LEVEL weight file names form more than one logical weight set
    (HF-convention shards collapse into one). Callers pass basenames only."""
    global _SHARD_NAME_RE
    import re

    if _SHARD_NAME_RE is None:
        _SHARD_NAME_RE = re.compile(r"^(.+)-(\d+)-of-(\d+)\.(safetensors|gguf)$")
    logical: set[tuple[str, str]] = set()
    for name in names:
        if not (name.endswith(".safetensors") or name.endswith(".gguf")):
            continue
        m = _SHARD_NAME_RE.match(name)
        logical.add((m.group(1), m.group(3)) if m else (name, ""))
    return len(logical) > 1


def _is_multi_weight_bundle(dest_dir: Path) -> bool:
    """True when a snapshot carries MULTIPLE distinct top-level weight files
    that are not one HF-convention shard set — e.g. Anima/Ernie civitai
    bundles shipping DiT + text-encoder + VAE as separate single-files.
    tensorhub's diffusers/single-file layout contract rightly rejects these
    (multiple_files_for_single_file_layout, found live e2e #112); they must
    publish with library_name unset (validator opt-out)."""
    return _is_multi_weight_names(
        p.name for p in dest_dir.iterdir() if p.is_file())


def _resolve_civitai_family(base_family: str, layout_family: str) -> str:
    """Civitai's structured baseModel is authoritative over filename-token
    inference: creator filenames routinely carry other arch names (e.g.
    'gonzalomoXLFluxPony_v70PhotoXLDMD.safetensors' is an SDXL 1.0 build
    whose name poisoned the family to flux and broke the repackage — found
    live, e2e tracker #112). Filename hints are the fallback."""
    base_family = str(base_family or "").strip()
    layout_family = str(layout_family or "").strip()
    if base_family and base_family != "other":
        return base_family
    if layout_family not in ("", "unknown"):
        return layout_family
    return base_family


def _local_gguf_quant(path: Path) -> str:
    """Quant token ("q5_k_m") from a local gguf header's general.file_type."""
    try:
        from gguf import GGUFReader
        from gguf.constants import LlamaFileType

        reader = GGUFReader(str(path), "r")
        field = reader.fields.get("general.file_type")
        if field is None:
            return ""
        ft = int(field.contents())
        name = LlamaFileType(ft).name
        return name.removeprefix("MOSTLY_").lower()
    except Exception:
        return ""


def ingest_civitai(
    model_version_id: int,
    dest_dir: Path,
    *,
    civitai_api_key: str | None = None,
    progress: ProgressFn | None = None,
    gguf_quant: str | None = None,
) -> IngestedSource:
    """Fetch one civitai model version into ``dest_dir`` (bounded provider API)."""
    from gen_worker.models.download import download_civitai, fetch_civitai_model_version

    version_id = int(model_version_id or 0)
    if version_id <= 0:
        raise ValueError("civitai_model_version_id is required")
    payload = fetch_civitai_model_version(version_id, api_key=(civitai_api_key or ""))

    dest_dir = Path(dest_dir)
    download_civitai(version_id, dest_dir, api_key=(civitai_api_key or ""),
                     progress=progress, gguf_quant=gguf_quant)

    files = sorted(p.name for p in dest_dir.iterdir() if p.is_file() and p.name != ".civitai.json")
    layout_info = detect_huggingface_source_layout(repo_dir=dest_dir, files=files)

    base_model_raw = str(payload.get("baseModel") or "").strip()
    try:
        from .base_model_families import civitai_to_family

        base_family = civitai_to_family(base_model_raw) or ""
    except Exception:
        base_family = ""
    model_family = _resolve_civitai_family(
        base_family, str(layout_info.model_family or ""))

    model_id = int(payload.get("modelId") or 0)
    air = str(payload.get("air") or "")
    attrs = {
        "runtime_library": "diffusers-single-file",
        "base_model_family": base_family,
        "base_model_civitai_baseModel": base_model_raw,
        "lineage_source": "civitai_baseModel" if base_model_raw else "unknown",
        "file_layout": layout_info.source_layout if layout_info.source_layout != "unknown" else "singlefile",
    }
    on_disk_dtype = _detect_snapshot_dtype(dest_dir)
    if on_disk_dtype:
        attrs["dtype"] = on_disk_dtype
    model_kind = str((payload.get("model") or {}).get("type") or "").strip().lower() \
        if isinstance(payload.get("model"), dict) else ""
    if model_kind in {"lora", "locon", "lycoris", "dora"}:
        attrs["runtime_library"] = "diffusers-lora"

    # th#611: gguf-only civitai versions publish AS-IS (single unshardable
    # artifact; the hub classifies family + flavor from the header). Without
    # a classification the clone falls into the safetensors repackage path
    # and dies with "no safetensors entry for repackage".
    classification: RepoClassification | None = None
    gguf_names = [f for f in files if f.lower().endswith(".gguf")]
    st_names = [f for f in files if f.lower().endswith(".safetensors")]
    if gguf_names and not st_names:
        quant = _local_gguf_quant(dest_dir / gguf_names[0])
        attrs["dtype"] = f"gguf:{quant}" if quant else "gguf"
        attrs["file_type"] = "gguf"
        attrs["file_layout"] = "singlefile"
        classification = RepoClassification(
            strategy="gguf",
            runtime_library="diffusers-single-file",
            allow_patterns=list(files),
            attrs=dict(attrs),
            detection_reason=f"civitai gguf-only version ({len(gguf_names)} gguf)",
        )
    metadata = {
        "source_provider": "civitai",
        "source_repo": str(version_id),
        "source_kind": "civitai_model_version",
        "civitai_model_version_id": str(version_id),
        "civitai_model_id": str(model_id),
        "civitai_base_model": base_model_raw,
        "civitai_air": air,
        "source_layout": str(layout_info.source_layout),
        "model_family": model_family or "unknown",
        "model_family_variant": str(layout_info.model_family_variant),
        "source_file_count": str(len(files)),
    }
    revision = _civitai_manifest_revision(files)
    repo_spec = {
        "kind": "adapter" if attrs["runtime_library"] == "diffusers-lora" else "model",
        "library_name": "diffusers",
        "model_family": model_family or "",
    }
    if attrs["runtime_library"] != "diffusers-lora" and _is_multi_weight_bundle(dest_dir):
        # Multi-component bundle (distinct DiT/TE/VAE files): no library can
        # load it as one artifact — opt out of the layout contract (an empty
        # library_name skips finalize-side validation; found live, e2e #112).
        repo_spec["library_name"] = ""
        metadata["multi_weight_bundle"] = "true"
    return IngestedSource(
        provider="civitai",
        source_ref=str(version_id),
        source_revision=revision,
        dir=dest_dir,
        layout=str(layout_info.source_layout),
        model_family=model_family or "unknown",
        model_family_variant=str(layout_info.model_family_variant),
        attrs=attrs,
        metadata=metadata,
        repo_spec=repo_spec,
        classification=classification,
    )


__all__ = [
    "CivitaiSourcePlan",
    "CloneDownloadError",
    "HFSourcePlan",
    "IngestedSource",
    "ingest_civitai",
    "ingest_huggingface",
    "plan_civitai",
    "plan_huggingface",
    "resolve_hf_identity",
]
