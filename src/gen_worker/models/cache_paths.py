from __future__ import annotations

import os
from pathlib import Path


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"
DEFAULT_WORKER_LOCAL_MODEL_CACHE_DIR = "/tmp/tensorhub/local-model-cache"


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory.

    Honors the ``TENSORHUB_CACHE_DIR`` environment variable when set, so the
    cozy local runner can point the CAS at a persistent ``~/.cache/tensorhub``
    (weights survive reboots) instead of ``/tmp``. Falls back to the ``/tmp``
    default when unset, preserving existing worker/orchestrator behavior.
    """
    env = os.environ.get("TENSORHUB_CACHE_DIR")
    if env and env.strip():
        return Path(env).expanduser()
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas."""
    return tensorhub_cache_dir() / "cas"


def worker_local_model_cache_dir_default() -> Path:
    """Default local (non-NFS) cache for NFS->local localization."""
    return Path(DEFAULT_WORKER_LOCAL_MODEL_CACHE_DIR)
