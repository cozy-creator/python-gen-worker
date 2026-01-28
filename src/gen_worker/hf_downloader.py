from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .model_refs import HuggingFaceRef


@dataclass(frozen=True)
class HuggingFaceDownloadResult:
    local_dir: Path


def _csv_env(name: str) -> Optional[List[str]]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _normalize_components(cs: Iterable[str]) -> List[str]:
    out: List[str] = []
    for c in cs:
        c = (c or "").strip().strip("/")
        if not c:
            continue
        out.append(c)
    # de-dupe but keep stable order
    seen: Set[str] = set()
    deduped: List[str] = []
    for c in out:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


def _default_weight_precisions() -> List[str]:
    # Default: only reduced-precision weights.
    return ["fp16", "bf16"]


def _precisions_from_env() -> List[str]:
    ps = _csv_env("COZY_HF_WEIGHT_PRECISIONS")
    if not ps:
        return _default_weight_precisions()
    return [p.lower().strip() for p in ps if p.strip()]


def _token_from_env() -> Optional[str]:
    # huggingface_hub typically uses HF_TOKEN, but keep compatibility with common envs.
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        v = (os.getenv(k) or "").strip()
        if v:
            return v
    return None


class HuggingFaceHubDownloader:
    """
    Download Hugging Face repos using the official huggingface_hub cache/layout.

    This intentionally delegates caching/resume/locking behavior to huggingface_hub.
    """

    def __init__(self, hf_home: Optional[str] = None, hf_token: Optional[str] = None) -> None:
        self.hf_home = (hf_home or os.getenv("HF_HOME") or "").strip() or None
        self.hf_token = (hf_token or _token_from_env() or "").strip() or None

    def download(self, ref: HuggingFaceRef) -> HuggingFaceDownloadResult:
        kwargs = {"resume_download": True}
        if self.hf_home:
            kwargs["cache_dir"] = self.hf_home
        if self.hf_token:
            kwargs["token"] = self.hf_token

        # Prefer minimal downloads for diffusers-style repos by default.
        # This can be overridden by setting COZY_HF_FULL_REPO_DOWNLOAD=1.
        full_repo = (os.getenv("COZY_HF_FULL_REPO_DOWNLOAD") or "").strip() in ("1", "true", "yes")
        if not full_repo:
            allow = self._build_allow_patterns(ref, HfApi(token=self.hf_token))
            kwargs["allow_patterns"] = allow

        local = snapshot_download(repo_id=ref.repo_id, revision=ref.revision, **kwargs)
        return HuggingFaceDownloadResult(local_dir=Path(local))

    def _build_allow_patterns(self, ref: HuggingFaceRef, api) -> List[str]:
        """
        Build allow_patterns for snapshot_download so we avoid pulling huge legacy files.

        Defaults (can be overridden via env vars):
        - Only reduced-precision safetensors weights (fp16/bf16).
        - Only the components needed for inference.
        - Never download repo-root .ckpt/.bin weights unless full repo download is enabled.
        """
        from huggingface_hub import hf_hub_download  # type: ignore

        # Download only model_index.json first (small), then derive components.
        index_path = hf_hub_download(
            repo_id=ref.repo_id,
            revision=ref.revision,
            filename="model_index.json",
            cache_dir=self.hf_home,
            token=self.hf_token,
        )
        model_index = json.loads(Path(index_path).read_text("utf-8"))

        env_components = _csv_env("COZY_HF_COMPONENTS")
        include_optional = (os.getenv("COZY_HF_INCLUDE_OPTIONAL_COMPONENTS") or "").strip() in (
            "1",
            "true",
            "yes",
        )
        deny_by_default = {"safety_checker", "feature_extractor"}
        precisions = _precisions_from_env()

        if env_components is not None:
            components = _normalize_components(env_components)
        else:
            keys = [k for k in model_index.keys() if isinstance(k, str) and not k.startswith("_")]
            if not include_optional:
                keys = [k for k in keys if k not in deny_by_default]
            components = _normalize_components(keys)

        # Validate component folders exist (without downloading) and that required weights exist.
        repo_files: Sequence[str] = api.list_repo_files(repo_id=ref.repo_id, repo_type="model", revision=ref.revision)
        file_set = set(repo_files)

        present_components: List[str] = []
        for c in components:
            if any(p.startswith(f"{c}/") for p in repo_files):
                present_components.append(c)

        if not present_components:
            raise RuntimeError(
                f"hf:{ref.repo_id} does not look like a diffusers pipeline repo (no component folders found). "
                "Set COZY_HF_FULL_REPO_DOWNLOAD=1 to force a full snapshot_download."
            )

        allow: List[str] = ["model_index.json"]

        # Some repos have additional small root json needed for loading.
        if (os.getenv("COZY_HF_ALLOW_ROOT_JSON") or "").strip() in ("1", "true", "yes"):
            allow.append("*.json")

        small_tree_components = {"tokenizer", "tokenizer_2", "scheduler"}

        def _has_precision_safetensors(c: str) -> bool:
            for p in repo_files:
                if not p.startswith(f"{c}/"):
                    continue
                if not p.endswith(".safetensors"):
                    continue
                if any(prec in p.lower() for prec in precisions):
                    return True
            return False

        for c in present_components:
            # Always include config if present.
            if f"{c}/config.json" in file_set:
                allow.append(f"{c}/config.json")

            if c in small_tree_components:
                allow.append(f"{c}/*")
                continue

            # Reduced precision safetensors only by default.
            for prec in precisions:
                allow.append(f"{c}/*{prec}*.safetensors")

            # Hard fail early if we would end up with no weights for a required component.
            if not _has_precision_safetensors(c):
                raise RuntimeError(
                    f"hf:{ref.repo_id} missing required reduced-precision safetensors for component '{c}'. "
                    f"Available files include: {[p for p in repo_files if p.startswith(c + '/') and p.endswith('.safetensors')][:10]} "
                    "You can override with COZY_HF_WEIGHT_PRECISIONS=fp16,bf16,fp32 or set COZY_HF_FULL_REPO_DOWNLOAD=1."
                )

        return allow
