"""Hub-API ingest: materialize a source model snapshot locally.

HuggingFace: ``HfApi.list_repo_files`` → :mod:`cozy_convert.classifier` →
``snapshot_download(allow_patterns=...)``. Civitai: the provider-bounded
fetch from ``gen_worker.models.download`` plus clone-side metadata
(baseModel lineage, kohya hints). No arbitrary-URL sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .classifier import RepoClassification, classify_repo
from .layout import detect_huggingface_source_layout

ProgressFn = Callable[[int, Optional[int]], None]

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
    from huggingface_hub import HfApi

    repo_id = str(source_ref or "").strip()
    if repo_id.count("/") != 1:
        raise ValueError(f"huggingface source ref must be org/name, got {source_ref!r}")
    info = HfApi(token=(hf_token or None)).repo_info(
        repo_id, revision=(str(revision).strip() or None) if revision else None)
    return repo_id, str(getattr(info, "sha", "") or "")


def _hf_classification_inputs(
    repo_id: str,
    revision: str | None,
    hf_token: str | None,
) -> tuple[list[str], dict[str, int], dict[str, Any]]:
    """One list_repo_tree walk: paths, sizes, and small side signals."""
    from huggingface_hub import HfApi

    api = HfApi(token=(hf_token or None))
    paths: list[str] = []
    sizes: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id, revision=revision, recursive=True):
        path = str(getattr(entry, "path", "") or "")
        size = getattr(entry, "size", None)
        if not path or size is None:
            continue  # skip directory rows
        paths.append(path)
        sizes[path] = int(size or 0)

    side: dict[str, Any] = {}
    if "config.json" in paths:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(repo_id, "config.json", revision=revision,
                                    token=(hf_token or None))
            side["config_json"] = json.loads(Path(local).read_text(encoding="utf-8"))
        except Exception:
            side["config_json"] = None

    root_st = [p for p in paths if "/" not in p and p.lower().endswith(".safetensors")]
    if root_st and "model_index.json" not in paths and "adapter_config.json" not in paths:
        # LoRA sniff — remote safetensors __metadata__ via the hub API.
        try:
            from huggingface_hub import get_safetensors_metadata

            st_md = get_safetensors_metadata(repo_id, revision=revision, token=(hf_token or None))
            for fmeta in (getattr(st_md, "files_metadata", None) or {}).values():
                md = getattr(fmeta, "metadata", None)
                if md:
                    side["safetensors_metadata"] = {str(k): str(v) for k, v in md.items()}
                    break
        except Exception:
            pass
        try:
            from huggingface_hub import ModelCard

            card = ModelCard.load(repo_id, token=(hf_token or None))
            tags = getattr(card.data, "tags", None) or []
            side["readme_tags"] = [str(t) for t in tags]
        except Exception:
            pass
    return paths, sizes, side


def ingest_huggingface(
    source_ref: str,
    dest_dir: Path,
    *,
    revision: str | None = None,
    dtype_preference: list[str] | None = None,
    gguf_quant: str | None = None,
    hf_token: str | None = None,
    progress: ProgressFn | None = None,
) -> IngestedSource:
    """Classify + selectively download one HF repo into ``dest_dir``."""
    from huggingface_hub import snapshot_download

    repo_id, sha = resolve_hf_identity(source_ref, revision=revision, hf_token=hf_token)
    rev = sha or (str(revision).strip() if revision else None)
    paths, sizes, side = _hf_classification_inputs(repo_id, rev, hf_token)
    classification = classify_repo(
        paths,
        sizes=sizes,
        config_json=side.get("config_json"),
        safetensors_metadata=side.get("safetensors_metadata"),
        readme_tags=side.get("readme_tags") or (),
        dtype_pref=tuple(dtype_preference or ("bf16", "fp16", "fp32")),
        gguf_quant=gguf_quant,
    )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    selected_bytes = sum(int(sizes.get(p, 0)) for p in classification.allow_patterns)
    if progress is not None:
        progress(0, selected_bytes or None)
    snapshot_download(
        repo_id,
        revision=rev,
        local_dir=str(dest_dir),
        allow_patterns=list(classification.allow_patterns),
        token=(hf_token or None),
    )
    if progress is not None:
        progress(selected_bytes, selected_bytes or None)

    layout_info = detect_huggingface_source_layout(repo_dir=dest_dir, files=paths)
    library = classification.runtime_library
    library_name = {
        "diffusers-single-file": "diffusers",
        "diffusers-lora": "diffusers",
        "llama-cpp": "llama.cpp",
    }.get(library, library)
    repo_spec = {
        "kind": "adapter" if library in {"peft", "diffusers-lora"} else "model",
        "library_name": library_name,
        "model_family": str(layout_info.model_family or ""),
        "class_name": str(classification.attrs.get("architecture") or ""),
    }
    attrs = dict(classification.attrs)
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


def ingest_civitai(
    model_version_id: int,
    dest_dir: Path,
    *,
    civitai_api_key: str | None = None,
    progress: ProgressFn | None = None,
) -> IngestedSource:
    """Fetch one civitai model version into ``dest_dir`` (bounded provider API)."""
    from gen_worker.models.download import download_civitai, fetch_civitai_model_version

    version_id = int(model_version_id or 0)
    if version_id <= 0:
        raise ValueError("civitai_model_version_id is required")
    payload = fetch_civitai_model_version(version_id, api_key=(civitai_api_key or ""))

    dest_dir = Path(dest_dir)
    download_civitai(version_id, dest_dir, api_key=(civitai_api_key or ""), progress=progress)

    files = sorted(p.name for p in dest_dir.iterdir() if p.is_file() and p.name != ".civitai.json")
    layout_info = detect_huggingface_source_layout(repo_dir=dest_dir, files=files)

    base_model_raw = str(payload.get("baseModel") or "").strip()
    try:
        from .base_model_families import civitai_to_family

        base_family = civitai_to_family(base_model_raw) or ""
    except Exception:
        base_family = ""
    model_family = str(layout_info.model_family or "").strip()
    if model_family in ("", "unknown") and base_family:
        model_family = base_family

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
    manifest = json.dumps(
        [{"name": n} for n in files], sort_keys=True, separators=(",", ":"))
    import hashlib

    revision = "sha256:" + hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    repo_spec = {
        "kind": "adapter" if attrs["runtime_library"] == "diffusers-lora" else "model",
        "library_name": "diffusers",
        "model_family": model_family or "",
    }
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
    )


__all__ = [
    "IngestedSource",
    "ingest_huggingface",
    "ingest_civitai",
    "resolve_hf_identity",
]
