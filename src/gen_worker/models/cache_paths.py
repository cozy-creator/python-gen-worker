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


#: Where the IMAGE build bakes this release's serialized ExportedPrograms.
#: tensorhub's `image.DeriveCASImagePath`, one spelling, and the two repos
#: must not drift — a mismatch here is not an error, it is a permanent silent
#: cache miss, which is exactly the defect pgw#1462 part 2 exists to close.
BAKED_PROGRAM_CAS_DIR = "/app/.tensorhub/derive-cas"


def baked_program_cas_dir() -> Path | None:
    """The image's read-only exported-program CAS, or None when there is none.

    THE POD'S OWN CAS IS NOT THIS DIRECTORY, and that is the whole bug.
    `tensorhub_cas_dir()` is `<TENSORHUB_CACHE_DIR>/cas` — `/tmp/tensorhub-cache/cas`
    by default — while the builder bakes the release's graph blobs into
    `/app/.tensorhub/derive-cas`. th#2162 concluded that no hub route was needed
    because *"pgw's miner already falls through to its local CAS by content
    address"*; the fallthrough is real but it has never looked in the baked
    directory, so those blobs have been unreachable since the day they were
    first baked.

    Settings-driven so a non-container run (e2e, bare-metal dev, cozy-local)
    points it somewhere real, exactly as `endpoint_lock_path` does. An empty
    setting DISABLES the tier rather than guessing.
    """

    configured = current_or(_STANDALONE).baked_program_cas_root.strip()
    root = Path(configured).expanduser() if configured else Path(BAKED_PROGRAM_CAS_DIR)
    # Absence is the ordinary case on any image whose build used a committed
    # endpoint.lock (no derive step ran, so nothing was baked). It is not a
    # failure and must not read as one.
    return root if root.is_dir() else None


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
    from gen_worker._vendor.torchcg.engine import Engine

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
