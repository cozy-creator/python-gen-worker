from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, cast

import requests

from .refs import HuggingFaceRef
from .hf_selection import HFSelectionPolicy, finalize_diffusers_download, plan_diffusers_download


@dataclass(frozen=True)
class HuggingFaceDownloadResult:
    local_dir: Path


def _default_weight_precisions() -> list[str]:
    # Default: only reduced-precision weights.
    return ["fp16", "bf16"]


class HuggingFaceHubDownloader:
    """
    Download Hugging Face repos using the official huggingface_hub cache/layout.

    This intentionally delegates caching/resume/locking behavior to huggingface_hub.
    """

    def __init__(self, hf_home: Optional[str] = None, hf_token: Optional[str] = None) -> None:
        # Caller passes hf_home/hf_token from gen_worker.config.Settings.
        # No env fallback — Settings is the single source of truth (issue #253).
        self.hf_home = (hf_home or "").strip() or None
        self.hf_token = (hf_token or "").strip() or None

    def download(self, ref: HuggingFaceRef) -> HuggingFaceDownloadResult:
        try:
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required for hf: model refs. Install gen-worker with the Hugging Face extra."
            ) from e
        hf_hub_download_fn = cast(Callable[..., str], hf_hub_download)
        snapshot_download_fn = cast(Callable[..., str], snapshot_download)
        hf_hub_url_fn: Optional[Callable[..., str]] = None
        try:
            from huggingface_hub import hf_hub_url as _hf_hub_url
            hf_hub_url_fn = cast(Callable[..., str], _hf_hub_url)
        except Exception:
            pass

        # The wire format is bare ref + typed provider — HF refs reach
        # this loader without any prefix. No defensive strip needed.
        repo_id = (ref.repo_id or "").strip()
        if not repo_id:
            raise ValueError("empty hf repo_id")

        kwargs: dict[str, Any] = {"resume_download": True}
        if self.hf_home:
            kwargs["cache_dir"] = self.hf_home
        if self.hf_token:
            kwargs["token"] = self.hf_token

        # Prefer minimal downloads for diffusers-style repos by default.
        if True:
            # Non-configurable safety guard: refuse to download extremely large file sets by default.
            max_total_bytes = 30_000_000_000  # 30GB

            policy = HFSelectionPolicy(
                components_override=None,
                include_optional_components=False,
                weight_precisions=_default_weight_precisions(),
                allow_root_json=False,
            )

            # Best-effort local completeness check: if we already have a local snapshot folder that
            # contains all required files, skip network calls and downloads.
            local_snapshot = _try_get_local_snapshot_dir(
                snapshot_download=snapshot_download_fn,
                repo_id=repo_id,
                revision=ref.revision,
                cache_dir=self.hf_home,
                token=self.hf_token,
            )
            if local_snapshot is not None:
                local_files = _walk_relative_files(local_snapshot)
                model_index = _try_load_local_model_index(local_snapshot)
                if model_index is not None:
                    plan = plan_diffusers_download(model_index=model_index, repo_files=sorted(local_files), policy=policy)
                    needed = finalize_diffusers_download(
                        plan=plan,
                        repo_files=sorted(local_files),
                        weight_index_json_by_file=_load_local_weight_indexes(local_snapshot, plan.required_weight_index_files),
                    )
                    if needed.issubset(local_files) and not _has_incomplete_markers(local_snapshot):
                        return HuggingFaceDownloadResult(local_dir=local_snapshot)

            api = HfApi(token=self.hf_token)
            repo_files: Sequence[str] = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=ref.revision)

            # Best-effort: get file sizes from list_repo_tree when available.
            repo_file_sizes: dict[str, int] = {}
            if hasattr(api, "list_repo_tree"):
                try:
                    tree = api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=ref.revision, recursive=True)
                    for ent in tree:
                        # huggingface_hub RepoFile has .path and .size
                        p = getattr(ent, "path", None)
                        sz = getattr(ent, "size", None)
                        if isinstance(p, str) and isinstance(sz, int):
                            repo_file_sizes[p] = sz
                except Exception:
                    repo_file_sizes = {}

            # Fetch model_index.json if present; otherwise infer components from repo structure.
            model_index = _try_fetch_model_index_json(
                hf_hub_download=hf_hub_download_fn,
                repo_id=repo_id,
                revision=ref.revision,
                cache_dir=self.hf_home,
                token=self.hf_token,
            )
            if model_index is None and policy.components_override is None:
                inferred = _infer_diffusers_components_from_repo_files(repo_files)
                policy = HFSelectionPolicy(
                    components_override=inferred,
                    include_optional_components=policy.include_optional_components,
                    weight_precisions=policy.weight_precisions,
                    allow_root_json=policy.allow_root_json,
                )
                model_index = {"_class_name": "Unknown"}

            if model_index is None:
                raise RuntimeError(
                    f"huggingface ref {ref.repo_id!r} is missing model_index.json and no diffusers-like components could be inferred."
                )

            # Prefetch all sharded-weight index JSONs (small) so we can choose the best weight set per component.
            idx_json_by_file: dict[str, dict] = {}
            for pth in repo_files:
                if not pth.lower().endswith(".safetensors.index.json"):
                    continue
                try:
                    p = hf_hub_download_fn(
                        repo_id=repo_id,
                        revision=ref.revision,
                        filename=pth,
                        cache_dir=self.hf_home,
                        token=self.hf_token,
                    )
                    idx_json_by_file[pth] = json.loads(Path(p).read_text("utf-8"))
                except Exception:
                    continue

            session = requests.Session()
            dtype_cache: dict[str, Optional[set[str]]] = {}

            def probe_safetensors_dtypes(rel_path: str) -> Optional[set[str]]:
                if rel_path in dtype_cache:
                    return dtype_cache[rel_path]
                if hf_hub_url_fn is None:
                    dtype_cache[rel_path] = None
                    return None
                if not rel_path.lower().endswith(".safetensors"):
                    dtype_cache[rel_path] = None
                    return None

                url = hf_hub_url_fn(repo_id=repo_id, filename=rel_path, repo_type="model", revision=ref.revision)
                headers = {"Range": "bytes=0-7"}
                if self.hf_token:
                    headers["Authorization"] = f"Bearer {self.hf_token}"

                try:
                    r = session.get(url, headers=headers, allow_redirects=True, timeout=60)
                    r.raise_for_status()
                    if len(r.content) != 8:
                        dtype_cache[rel_path] = None
                        return None
                    (header_len,) = struct.unpack("<Q", r.content)
                    if header_len <= 0 or header_len > (32 << 20):
                        dtype_cache[rel_path] = None
                        return None

                    headers2 = {"Range": f"bytes=8-{8 + header_len - 1}"}
                    if self.hf_token:
                        headers2["Authorization"] = f"Bearer {self.hf_token}"
                    r2 = session.get(url, headers=headers2, allow_redirects=True, timeout=60)
                    r2.raise_for_status()
                    raw = json.loads(r2.content.decode("utf-8"))
                    dtypes: set[str] = set()
                    for name, meta in raw.items():
                        if name == "__metadata__":
                            continue
                        if not isinstance(meta, dict):
                            continue
                        dt = meta.get("dtype")
                        if isinstance(dt, str) and dt.strip():
                            dtypes.add(dt.strip())
                    dtype_cache[rel_path] = dtypes or None
                    return dtype_cache[rel_path]
                except Exception:
                    dtype_cache[rel_path] = None
                    return None

            plan = plan_diffusers_download(
                model_index=model_index,
                repo_files=repo_files,
                policy=policy,
                weight_index_json_by_file=idx_json_by_file,
                repo_file_sizes=repo_file_sizes,
                probe_safetensors_dtypes=probe_safetensors_dtypes,
            )

            selected_files = finalize_diffusers_download(plan=plan, repo_files=repo_files, weight_index_json_by_file=idx_json_by_file)

            if repo_file_sizes:
                total = sum(int(repo_file_sizes.get(p, 0) or 0) for p in selected_files)
                if total > max_total_bytes:
                    raise RuntimeError(
                        f"refusing to download an excessively large Hugging Face repo selection: {total} bytes "
                        f"(limit {max_total_bytes} bytes)."
                    )

            # Deterministic order helps debugging and keeps behavior stable.
            kwargs["allow_patterns"] = sorted(selected_files)

        local = snapshot_download_fn(repo_id=repo_id, revision=ref.revision, **kwargs)
        return HuggingFaceDownloadResult(local_dir=Path(local))


def _try_fetch_model_index_json(
    *,
    hf_hub_download: Callable[..., str],
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    token: str | None,
) -> Optional[dict]:
    try:
        index_path = hf_hub_download(
            repo_id=repo_id,
            revision=revision,
            filename="model_index.json",
            cache_dir=cache_dir,
            token=token,
        )
    except Exception:
        return None
    try:
        return json.loads(Path(index_path).read_text("utf-8"))
    except Exception:
        return None


def _infer_diffusers_components_from_repo_files(repo_files: Sequence[str]) -> list[str]:
    known = [
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
    ]
    present: list[str] = []
    for c in known:
        if any(p.startswith(f"{c}/") for p in repo_files):
            present.append(c)
    # Require at least one heavyweight component to avoid selecting random repos.
    if not any(c in present for c in ("transformer", "unet")):
        return []
    return present


def _try_get_local_snapshot_dir(
    *,
    snapshot_download: Callable[..., str],
    repo_id: str,
    revision: str | None,
    cache_dir: str | None,
    token: str | None,
) -> Optional[Path]:
    try:
        p = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            resume_download=True,
            cache_dir=cache_dir,
            token=token,
        )
        return Path(p)
    except Exception:
        return None


def _walk_relative_files(root: Path) -> set[str]:
    out: set[str] = set()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        out.add(rel)
    return out


def _try_load_local_model_index(root: Path) -> Optional[dict]:
    p = root / "model_index.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return None


def _load_local_weight_indexes(root: Path, idx_paths: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rel in idx_paths:
        p = root / rel
        if not p.exists():
            continue
        try:
            out[rel] = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
    return out


def _has_incomplete_markers(root: Path) -> bool:
    # huggingface_hub uses *.incomplete markers for partial files.
    return any(p.is_file() and p.name.endswith(".incomplete") for p in root.rglob("*.incomplete"))
