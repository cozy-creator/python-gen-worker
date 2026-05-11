from __future__ import annotations

from pathlib import Path


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"
DEFAULT_WORKER_LOCAL_MODEL_CACHE_DIR = "/tmp/tensorhub/local-model-cache"


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory."""
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas."""
    return tensorhub_cache_dir() / "cas"


def worker_local_model_cache_dir_default() -> Path:
    """Default local (non-NFS) cache for NFS->local localization."""
    return Path(DEFAULT_WORKER_LOCAL_MODEL_CACHE_DIR)
