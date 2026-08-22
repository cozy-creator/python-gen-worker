"""Convert Tensorhub's ref-keyed snapshot wire map into worker objects.

``ModelBinding.manifest_digest`` never had a sender and is tombstoned by
th#2208.  A snapshot map key is therefore the canonical ref on both sides of
the wire; there is no second spelling to reconcile here.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping
from .models.refs import WireRef


def resolved_repo_from_snapshot(snap: Any) -> Any:
    """``pb.Snapshot`` -> the typed ``WorkerResolvedRepo`` the download layer speaks: the ONE wire-boundary conversion."""
    from .models.hub_client import (
        WorkerResolvedChunk,
        WorkerResolvedRepo,
        WorkerResolvedRepoFile,
    )

    return WorkerResolvedRepo(
        snapshot_digest=snap.digest,
        files=[
            WorkerResolvedRepoFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                url=f.url or None,
                digest=f.digest or "",
                chunks=tuple(
                    WorkerResolvedChunk(
                        sha256=(c.sha256 or "").strip().lower(),
                        url=c.url,
                        length=int(c.len),
                    )
                    for c in f.chunks
                ),
            )
            for f in snap.files
        ],
    )


def snapshot_from_resolved_repo(resolved: Any) -> Any:
    """``WorkerResolvedRepo`` -> ``pb.Snapshot``: the INVERSE, and it lives here for the reason this module exists — one place the two spellings meet."""
    from .pb import worker_scheduler_pb2 as pb

    return pb.Snapshot(
        digest=resolved.snapshot_digest,
        files=[
            pb.SnapshotFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                url=f.url or "",
                digest=f.digest or "",
                chunks=[
                    pb.ChunkRef(
                        sha256=c.sha256,
                        url=c.url,
                        len=int(c.length),
                    )
                    for c in (f.chunks or ())
                ],
            )
            for f in resolved.files
        ],
    )


def resolved_repos(
    wire: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return every ref with its typed ``WorkerResolvedRepo``."""
    return {
        str(ref): resolved_repo_from_snapshot(snap)
        for ref, snap in wire.items()
    }


def index_snapshots(
    wire: Mapping[str, Any],
) -> Dict[WireRef, Any]:
    """Wire snapshot map -> the ref-keyed view the worker materializes from."""
    return {WireRef(ref): snapshot for ref, snapshot in wire.items()}


__all__ = [
    "index_snapshots",
    "resolved_repo_from_snapshot",
    "resolved_repos",
]
