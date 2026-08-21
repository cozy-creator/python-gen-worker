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
    """TensorHub cache root directory — the worker's CAS root."""
    configured = current_or(_STANDALONE).tensorhub_cache_dir.strip()
    if configured:
        return Path(configured).expanduser()
    return Path(TENSORHUB_CACHE_DIR)


def tensorhub_cas_dir() -> Path:
    """Worker CAS root: <TENSORHUB_CACHE_DIR>/cas."""
    return tensorhub_cache_dir() / "cas"


# Where the image build bakes this release's serialized ExportedPrograms — tensorhub's image.DeriveCASImagePath, one spelling; a drift between the two repos is not an error, it is a permanent silent cache miss.
BAKED_PROGRAM_CAS_DIR = "/app/.tensorhub/derive-cas"


def baked_program_cas_dir() -> Path | None:
    """The image's read-only exported-program CAS, or None when there is none."""

    configured = current_or(_STANDALONE).baked_program_cas_root.strip()
    root = Path(configured).expanduser() if configured else Path(BAKED_PROGRAM_CAS_DIR)
    return root if root.is_dir() else None


def open_worker_cas(root: Path | None = None) -> LocalCAS:
    """Open the worker's one tensorfs store."""

    return LocalCAS(tensorhub_cas_dir() if root is None else Path(root))


def open_worker_engine(root: Path | None = None) -> Engine:
    """Open TCG on the worker's one canonical tensorfs store."""
    from gen_worker._vendor.torchcg.engine import Engine

    return Engine(open_worker_cas(root))


def tensorhub_fill_source_dir() -> Path | None:
    """Endpoint-scoped datacenter-warm fill source, or ``None`` when no volume is attached."""
    configured = current_or(_STANDALONE).tensorhub_fill_source_dir.strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not os.path.ismount(path):
        return None
    return path
