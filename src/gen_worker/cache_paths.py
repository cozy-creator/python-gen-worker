from __future__ import annotations

import os
from pathlib import Path


DEFAULT_TENSORHUB_CACHE_DIR = "~/.cache/tensorhub"
DEFAULT_WORKER_LOCAL_MODEL_CACHE_DIR = "/tmp/tensorhub/local-model-cache"


def _expand_dir(raw: str) -> Path:
    s = (raw or "").strip() or DEFAULT_TENSORHUB_CACHE_DIR
    return Path(os.path.expanduser(s))


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory."""
    return _expand_dir(os.getenv("TENSORHUB_CACHE_DIR", DEFAULT_TENSORHUB_CACHE_DIR))


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas."""
    return tensorhub_cache_dir() / "cas"


def worker_model_cache_dir() -> Path:
    """Effective model cache root used by worker download/materialization code."""
    return tensorhub_cas_dir()


def worker_local_model_cache_dir_default() -> Path:
    """Default local (non-NFS) cache for NFS->local localization."""
    return Path(DEFAULT_WORKER_LOCAL_MODEL_CACHE_DIR)
