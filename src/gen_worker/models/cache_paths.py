from __future__ import annotations

from pathlib import Path

from ..config import get_settings


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory.

    Honors the ``TENSORHUB_CACHE_DIR`` environment variable when set, so the
    cozy local runner can point the CAS at a persistent ``~/.cache/tensorhub``
    (weights survive reboots) instead of ``/tmp``. Falls back to the ``/tmp``
    default when unset, preserving existing worker/orchestrator behavior.
    """
    env = get_settings().tensorhub_cache_dir.strip()
    if env:
        return Path(env).expanduser()
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas."""
    return tensorhub_cache_dir() / "cas"
