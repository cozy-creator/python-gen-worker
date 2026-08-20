"""Whole-directory admission. A TEST FIXTURE, and it lives here because of it.

`LocalCAS.ingest_repository` used to be production surface — it was one of the
five methods the pre-genesis `8bafdfbb` snapshot was pinned for. pgw#1575 asked
what actually calls it and the answer was: nothing in `src/`, ever. Production
admits bytes one file at a time (`publish_v2` -> `gen_worker.cas.ingest_file`)
or adopts them from a hub grant with the hub's own manifest
(`transfer/grants.py`). Admitting a directory tree is what a TEST does to build
a real store to read back out of.

So it is here rather than in `src/gen_worker/cas/`, where
`scripts/lint_unreached_surface.py` would be right to call it stranded.
"""

from __future__ import annotations

from pathlib import Path

from gen_worker._vendor.tensorfs.local import LocalCAS
from gen_worker._vendor.tensorfs.manifest import FileEntry, RepositoryManifest
from gen_worker.cas import ingest_file


def ingest_repository(cas: LocalCAS, source: str | Path) -> RepositoryManifest:
    """Admit every regular file below a directory into one manifest.

    V1 manifests describe files, not symlinks or empty directories. Refusing
    those entry types keeps projection portable and prevents a source-tree link
    from escaping the repository root.
    """

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
