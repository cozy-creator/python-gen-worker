from __future__ import annotations

import os
from pathlib import Path

from ..config import get_settings


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory — the worker's CAS root.

    Honors the ``TENSORHUB_CACHE_DIR`` environment variable when set. This is
    the ONE knob for where the CAS lives: the cozy-local runner points it at
    a persistent ``~/.cache/tensorhub`` (weights survive reboots). The CAS
    root ALWAYS stays on local/pod-local storage — a managed, bounded LRU
    tier (th#850 managed-tier ruling, gw#599: supersedes the earlier
    CAS-root-on-volume shape). A mounted RunPod endpoint volume, when
    attached, is a FILL SOURCE consulted before R2 (see
    ``tensorhub_fill_source_dir``) — it is never the CAS root itself. Falls
    back to the ``/tmp`` default when unset. The CAS implementation itself is
    deliberately backend-agnostic: nothing branches on what's mounted here.
    """
    env = get_settings().tensorhub_cache_dir.strip()
    if env:
        return Path(env).expanduser()
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas. Always local/pod storage."""
    return tensorhub_cache_dir() / "cas"


def tensorhub_fill_source_dir() -> Path | None:
    """Endpoint-scoped datacenter-warm fill source (th#850 managed-tier
    ruling), or ``None`` when no volume is attached.

    Honors ``TENSORHUB_FILL_SOURCE_DIR``, set by tensorhub only when this
    pod's endpoint has a RunPod network volume attached. Guarded by
    ``os.path.ismount`` — a plain directory baked into the image or left on
    the container disk must never be mistaken for the real per-endpoint
    volume (same defensive pattern the removed ``endpoint_shared_cas_dir``
    used). This is FILL SOURCE #1 in the CAS layer's fetch order (volume,
    then R2); it is never the CAS root. cozy-local and any pod without a
    volume leave this unset, which is the degenerate case: fetch goes
    straight to R2, byte-identical to pre-th#850 behavior.
    """
    env = get_settings().tensorhub_fill_source_dir.strip()
    if not env:
        return None
    path = Path(env).expanduser()
    if not os.path.ismount(path):
        return None
    return path
