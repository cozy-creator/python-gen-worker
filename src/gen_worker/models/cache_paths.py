from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from gen_worker._vendor.tensorfs import LocalCAS

from ..config import Settings, current_or

if TYPE_CHECKING:
    from gen_worker._vendor.torchcg import Engine

_STANDALONE = Settings()


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"


def tensorhub_cache_dir() -> Path:
    """TensorHub cache root directory — the worker's CAS root.

    Honors the ``TENSORHUB_CACHE_DIR`` environment variable when set. This is
    the ONE knob for where the CAS lives: the cozy-local runner points it at
    a persistent ``~/.cache/tensorhub`` (weights survive reboots). The CAS
    root ALWAYS stays on local/pod-local storage — a managed, bounded LRU
    tier, never on a volume. A mounted RunPod endpoint volume, when
    attached, is a FILL SOURCE consulted before R2 (see
    ``tensorhub_fill_source_dir``) — it is never the CAS root itself. Falls
    back to the ``/tmp`` default when unset. The CAS implementation itself is
    deliberately backend-agnostic: nothing branches on what's mounted here.
    """
    configured = current_or(_STANDALONE).tensorhub_cache_dir.strip()
    if configured:
        return Path(configured).expanduser()
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas. Always local/pod storage."""
    return tensorhub_cache_dir() / "cas"


def open_worker_cas(root: Path | None = None) -> LocalCAS:
    """Open the worker's one tensorfs store.

    Production callers omit ``root`` and therefore share
    :func:`tensorhub_cas_dir`. The override exists for an explicitly scoped
    model store (local CLI and tests); consumers must pass that same root on
    every path rather than inventing a private CAS subdirectory.
    """

    return LocalCAS(tensorhub_cas_dir() if root is None else Path(root))


def open_worker_engine(root: Path | None = None) -> Engine:
    """Open TCG on the worker's one canonical tensorfs store.

    Compile, import, resolve, and runner construction all cross this factory so
    no caller can silently introduce a second compiled-graph store.  The import
    stays lazy because model-only commands do not require TCG at startup.
    """
    from gen_worker._vendor.torchcg import Engine

    return Engine(open_worker_cas(root))


def tensorhub_fill_source_dir() -> Path | None:
    """Endpoint-scoped datacenter-warm fill source, or ``None`` when no volume
    is attached.

    Honors ``TENSORHUB_FILL_SOURCE_DIR``, set by tensorhub only when this
    pod's endpoint has a RunPod network volume attached. Guarded by
    ``os.path.ismount`` — a plain directory baked into the image or left on
    the container disk must never be mistaken for the real per-endpoint
    volume. This is FILL SOURCE #1 in the CAS layer's fetch order (volume,
    then R2); it is never the CAS root. cozy-local and any pod without a
    volume leave this unset, which is the degenerate case: fetch goes straight
    to R2.
    """
    configured = current_or(_STANDALONE).tensorhub_fill_source_dir.strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not os.path.ismount(path):
        return None
    return path
