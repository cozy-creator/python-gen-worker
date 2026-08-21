"""Admission: bytes on disk become CAS objects and a v1 manifest entry."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .._vendor.tensorfs.local import LocalCAS
from .._vendor.tensorfs.manifest import Chunk, FileEntry
from .._vendor.tensorfs.refs import CASRef
from .planner import plan_chunks


def ingest_file(
    cas: LocalCAS,
    source: str | Path,
    *,
    manifest_path: str | None = None,
) -> FileEntry:
    """Admit one file on the planner grid and return its manifest entry."""

    path = Path(source)
    initial = path.stat()
    whole = hashlib.sha256()
    chunks: list[Chunk] = []
    copied = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        chunk_lengths = plan_chunks(handle, initial.st_size)
        if not chunk_lengths:
            return FileEntry(
                manifest_path or path.name,
                initial.st_size,
                cas.put_file(path, size=initial.st_size),
            )
        handle.seek(0)
        for length in chunk_lengths:
            data = handle.read(length)
            if len(data) != length:
                raise OSError(f"{path} ended while it was being ingested")
            copied += len(data)
            whole.update(data)
            digest = cas.put_bytes(data)
            chunks.append(Chunk(digest, len(data)))
        if handle.read(1):
            raise OSError(f"{path} grew while it was being ingested")
        after = os.fstat(handle.fileno())
    if (
        copied != initial.st_size
        or before.st_size != initial.st_size
        or after.st_size != initial.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise OSError(f"{path} changed while it was being ingested")
    return FileEntry(
        manifest_path or path.name,
        copied,
        CASRef(whole.hexdigest()),
        tuple(chunks),
    )
