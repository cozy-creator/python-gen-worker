"""Whole-directory admission."""

from __future__ import annotations

from pathlib import Path

from gen_worker._vendor.tensorfs.local import LocalCAS
from gen_worker._vendor.tensorfs.manifest import FileEntry, RepositoryManifest
from gen_worker.cas import ingest_file


def ingest_repository(cas: LocalCAS, source: str | Path) -> RepositoryManifest:
    """Admit every regular file below a directory into one manifest."""

    root = Path(source)
    if not root.is_dir():
        raise NotADirectoryError(root)
    entries: list[FileEntry] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"repository contains a symlink: {path.relative_to(root)}")
        if path.is_dir():
            continue
        if not path.is_file():
            relative = path.relative_to(root)
            raise ValueError(f"repository contains a non-regular file: {relative}")
        entries.append(
            ingest_file(cas, path, manifest_path=path.relative_to(root).as_posix())
        )
    return RepositoryManifest(tuple(entries))
