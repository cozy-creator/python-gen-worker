from __future__ import annotations

from pathlib import Path

from ..config import get_settings


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory — the worker's CAS root.

    Honors the ``TENSORHUB_CACHE_DIR`` environment variable when set. This is
    the ONE knob for where the CAS lives: the cozy-local runner points it at
    a persistent ``~/.cache/tensorhub`` (weights survive reboots); tensorhub
    points it at a mounted RunPod endpoint volume (`/tensorhub-endpoint-cache`)
    when the pod has one attached, so the ordinary R2 prefetch that
    materializes blobs here on the first boot warms the volume for later
    same-endpoint pods too (th#850) — no separate shared/read-through tier.
    Falls back to the ``/tmp`` default when unset. The CAS implementation
    itself is deliberately backend-agnostic: nothing branches on what's
    mounted here.
    """
    env = get_settings().tensorhub_cache_dir.strip()
    if env:
        return Path(env).expanduser()
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas."""
    return tensorhub_cache_dir() / "cas"
