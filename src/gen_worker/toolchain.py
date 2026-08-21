"""The compiler-stack facts a mint records, and this worker's own version."""

from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
from typing import Dict, Tuple

from . import dist_records, env_seal

logger = logging.getLogger(__name__)


def gen_worker_version() -> str:
    """This install's published version, or ``""`` when it cannot be read."""
    try:
        from importlib.metadata import version

        return str(version("gen-worker"))
    except Exception:  # noqa: BLE001 — a missing dist is not a worker failure
        return ""


@functools.lru_cache(maxsize=8192)
def closure_file_digest(path: str, mtime_ns: int, size: int) -> str:
    """Content digest of one source file, keyed on (path, mtime, size) so a repeated read of an unchanged file never re-hashes it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


@functools.lru_cache(None)
def toolchain_digest() -> Tuple[Tuple[str, str], ...]:
    out: Dict[str, str] = {
        "settings_declaration": env_seal.declaration_digest(),
        "loaded_libs": env_seal.loaded_libs_digest(),
    }
    wanted = ("torch", "triton")
    for name, record in dist_records.record_texts().items():
        if name in wanted or name.startswith("nvidia-"):
            out[name] = hashlib.sha256(record.encode()).hexdigest()[:16]
    try:
        import triton

        bin_dir = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
        if bin_dir.is_dir():
            for tool in sorted(bin_dir.iterdir()):
                if tool.is_file():
                    out[f"bin:{tool.name}"] = hashlib.sha256(
                        tool.read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001 — a toolchain fact is never fatal
        logger.debug("toolchain_digest: cuda tool hash failed", exc_info=True)
    return tuple(sorted(out.items()))


__all__ = ["closure_file_digest", "gen_worker_version", "toolchain_digest"]
