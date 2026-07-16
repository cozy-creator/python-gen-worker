from __future__ import annotations

import os
from pathlib import Path

from ..config import get_settings


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"

# RunPod network volumes are attached at pod creation. Tensorhub reserves this
# mount solely for an endpoint-owned immutable CAS tier; ordinary pod volumes
# continue using their existing path. This is a provider contract, not an
# operator-tunable cache knob.
ENDPOINT_SHARED_CACHE_MOUNT = Path("/tensorhub-endpoint-cache")


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


def endpoint_shared_cas_dir() -> Path | None:
    """Return the endpoint-owned shared CAS only when it is really mounted.

    A plain directory in an image or on the container disk must never be
    mistaken for the endpoint-isolated provider volume.
    """
    if os.path.ismount(ENDPOINT_SHARED_CACHE_MOUNT):
        return ENDPOINT_SHARED_CACHE_MOUNT / "tensorhub-cas-v1"
    return None
