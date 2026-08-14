from __future__ import annotations

import os
from pathlib import Path

from hashrepo import LocalCAS

from ..config import Settings, current_or

_STANDALONE = Settings()


TENSORHUB_CACHE_DIR = "/tmp/tensorhub-cache"
_CAS_DIRECTORIES = ("objects", "refs", "locks", "tmp")


def _mkdir_traversable_if_missing(path: Path) -> None:
    """Create exactly one CAS-owned directory with child-traversable mode.

    ``mkdir(mode=...)`` is still masked by the process umask. The explicit
    ``fchmod`` therefore applies only when this call won creation; pre-existing
    configured parents and stores retain their operator-owned permissions.
    """

    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise NotADirectoryError(path)
        return
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fchmod(descriptor, 0o755)
    finally:
        os.close(descriptor)


def _prepare_cas_directories(root: Path) -> None:
    missing: list[Path] = []
    cursor = root
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(cursor)
    for directory in reversed(missing):
        _mkdir_traversable_if_missing(directory)
    for name in _CAS_DIRECTORIES:
        _mkdir_traversable_if_missing(root / name)


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
    """Open the worker's one HashRepo store.

    Production callers omit ``root`` and therefore share
    :func:`tensorhub_cas_dir`. The override exists for an explicitly scoped
    model store (local CLI and tests); consumers must pass that same root on
    every path rather than inventing a private CAS subdirectory.
    """

    cas_root = tensorhub_cas_dir() if root is None else Path(root)
    _prepare_cas_directories(cas_root)
    return LocalCAS(cas_root)


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
