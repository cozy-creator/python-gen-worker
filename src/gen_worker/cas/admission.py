"""Admission: bytes on disk become CAS objects and a v1 manifest entry.

Two admissions, and the difference is whether the store keeps a COPY.

``ingest_file`` writes every object into the store — that is what a local
cache wants, because the file it came from is about to disappear.

``stage_file`` writes nothing. It plans the same objects and hashes the same
bytes, and returns WHERE each object already lives: a byte range of the file
the producer just wrote. A publish only needs to READ its objects, so a tree
being published needs no second copy of itself on the same disk (pgw#1666 —
a 50 GB mirror paid its whole download and cast and then died on ENOSPC in
`publish_v2`, staging a copy of bytes that were already there).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .._vendor.tensorfs.local import LocalCAS
from .._vendor.tensorfs.manifest import Chunk, FileEntry
from .._vendor.tensorfs.refs import CASRef
from .planner import plan_chunks

_READ_BLOCK = 1 << 20


@dataclass(frozen=True, slots=True)
class StagedObject:
    """One CAS object's bytes, named where they already are on disk."""

    digest: CASRef
    path: Path
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class StagedFile:
    """One file's manifest entry and the objects it is made of."""

    entry: FileEntry
    objects: tuple[StagedObject, ...]


def stage_file(
    source: str | Path,
    *,
    manifest_path: str | None = None,
) -> StagedFile:
    """Plan one file into CAS objects WITHOUT writing a byte.

    Same grid, same digests and same identity guard as ``ingest_file``; the
    objects are byte ranges of ``source`` rather than copies of it.
    """

    path = Path(source)
    initial = path.stat()
    whole = hashlib.sha256()
    chunks: list[Chunk] = []
    objects: list[StagedObject] = []
    read = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        chunk_lengths = plan_chunks(handle, initial.st_size)
        handle.seek(0)
        for length in chunk_lengths or (initial.st_size,):
            piece = hashlib.sha256()
            remaining = length
            while remaining > 0:
                data = handle.read(min(remaining, _READ_BLOCK))
                if not data:
                    raise OSError(f"{path} ended while it was being staged")
                remaining -= len(data)
                whole.update(data)
                piece.update(data)
            objects.append(
                StagedObject(CASRef(piece.hexdigest()), path, read, length))
            if chunk_lengths:
                chunks.append(Chunk(CASRef(piece.hexdigest()), length))
            read += length
        if handle.read(1):
            raise OSError(f"{path} grew while it was being staged")
        after = os.fstat(handle.fileno())
    if (
        read != initial.st_size
        or before.st_size != initial.st_size
        or after.st_size != initial.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise OSError(f"{path} changed while it was being staged")
    return StagedFile(
        FileEntry(
            manifest_path or path.name,
            read,
            CASRef(whole.hexdigest()),
            tuple(chunks),
        ),
        tuple(objects),
    )


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
