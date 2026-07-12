"""Local-only GGUF composed snapshots (cl#27, GGUF-DESIGN consumption half).

A ``#gguf-<qtype>`` flavor checkpoint is denoiser-only by contract (th#611:
exactly one ``.gguf`` weight plus sidecars). Serving it needs the base tag's
OTHER components (encoders/VAE/scheduler/configs) — without paying for the
base's bf16 denoiser shards, which the gguf replaces. This module resolves
BOTH manifests, drops the base's denoiser weights, and materializes ONE
composed snapshot dir through the ordinary CAS downloader (blob-level dedupe
with any sibling snapshot is automatic). The loading layer detects the
result via :data:`GGUF_MARKER`.

Production never reaches this path: the orchestrator's resolver refuses
``#gguf-*`` (gguf_local_only) and HelloAck picks never carry it — the only
callers are the local CLI resolve paths (run/serve/prefetch).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .hub_client import WorkerResolvedRepo
from .ladder import gguf_qtype, is_denoiser_weight_path
from .refs import TensorhubRef

logger = logging.getLogger(__name__)

# Written into the composed snapshot dir after materialization; the loading
# layer's lane detection reads it (with a structural fallback for a dir that
# lost the marker mid-crash).
GGUF_MARKER = ".cozy-gguf.json"


def composed_digest(flavor_digest: str, base_digest: str) -> str:
    """Snapshot-dir key for one (flavor, base) composition — distinct from
    both plain digests so a denoiser-only flavor snapshot and a full base
    snapshot never collide with the composed tree."""
    return f"{flavor_digest}.gguf.{base_digest[:16]}"


def compose_resolved(
    base: WorkerResolvedRepo, flavor: WorkerResolvedRepo
) -> Tuple[WorkerResolvedRepo, str]:
    """Merge the base manifest (minus denoiser weights) with the flavor
    manifest. Returns ``(composed, gguf_relpath)``."""
    ggufs = [f for f in flavor.files if f.path.lower().endswith(".gguf")]
    if len(ggufs) != 1:
        raise ValueError(
            f"gguf flavor manifest has {len(ggufs)} .gguf files, want exactly 1 "
            "(denoiser-only layout contract)"
        )
    files = [f for f in base.files if not is_denoiser_weight_path(f.path)]
    if not any(f.path.strip().lstrip("/") == "model_index.json" for f in files):
        raise ValueError(
            "gguf composition needs a diffusers-tree base (no model_index.json "
            "in the base manifest)"
        )
    have = {f.path for f in files}
    files.extend(f for f in flavor.files if f.path not in have)
    composed = WorkerResolvedRepo(
        snapshot_digest=composed_digest(flavor.snapshot_digest, base.snapshot_digest),
        files=files,
    )
    return composed, ggufs[0].path


def write_marker(snap_dir: Path, *, flavor: str, gguf_relpath: str) -> None:
    marker = {
        "flavor": flavor,
        "qtype": gguf_qtype(flavor),
        "gguf_path": gguf_relpath,
    }
    (Path(snap_dir) / GGUF_MARKER).write_text(json.dumps(marker), encoding="utf-8")


def read_marker(snap_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(snap_dir) / GGUF_MARKER
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def fetch_gguf_snapshot(
    thref: TensorhubRef,
    *,
    cache_dir: Path,
    emit: Callable[[Dict[str, Any]], None],
    resolve: Optional[Callable[[TensorhubRef], WorkerResolvedRepo]] = None,
) -> str:
    """Resolve + download the composed snapshot for a ``#gguf-*`` tensorhub
    ref. Mirrors ``_fetch_tensorhub_snapshot``'s contract (progress events,
    one re-resolve retry on presigned-URL expiry); returns the snapshot dir.
    """
    import asyncio
    import time

    from .cozy_snapshot import ensure_snapshot_async
    from .errors import UrlExpiredError

    if resolve is not None:
        resolver = resolve
    else:
        from .hub_client import resolve_repo

        resolver = resolve_repo

    if not gguf_qtype(str(thref.flavor or "")):
        raise ValueError(f"not a servable gguf flavor ref: {thref.canonical()!r}")
    base_ref = dataclasses.replace(thref, flavor=None)
    canonical = thref.canonical()

    def _resolve_composed() -> Tuple[WorkerResolvedRepo, str]:
        return compose_resolved(resolver(base_ref), resolver(thref))

    emit({"kind": "model_fetch.started", "ref": canonical, "provider": "tensorhub"})
    composed, gguf_rel = _resolve_composed()

    snap_dir = Path(cache_dir) / "snapshots" / composed.snapshot_digest
    if snap_dir.exists():
        if read_marker(snap_dir) is None:
            write_marker(snap_dir, flavor=str(thref.flavor), gguf_relpath=gguf_rel)
        emit({"kind": "model_fetch.completed", "ref": canonical,
              "provider": "tensorhub", "local_dir": str(snap_dir)})
        return str(snap_dir)

    last_at = [0.0]

    def _progress(done: int, total: Optional[int]) -> None:
        now = time.monotonic()
        if now - last_at[0] < 1.0 and (total is None or done < total):
            return
        last_at[0] = now
        emit({"kind": "model_fetch.progress", "ref": canonical,
              "provider": "tensorhub", "done_bytes": int(done),
              "total_bytes": int(total) if total else None})

    async def _download(res: WorkerResolvedRepo) -> Path:
        return await ensure_snapshot_async(
            base_dir=Path(cache_dir), ref=thref, resolved=res, progress=_progress,
        )

    try:
        snap = asyncio.run(_download(composed))
    except UrlExpiredError:
        emit({"kind": "model_fetch.reresolve", "ref": canonical,
              "provider": "tensorhub", "reason": "url_expired"})
        composed, gguf_rel = _resolve_composed()
        snap = asyncio.run(_download(composed))
    write_marker(snap, flavor=str(thref.flavor), gguf_relpath=gguf_rel)
    emit({"kind": "model_fetch.completed", "ref": canonical,
          "provider": "tensorhub", "local_dir": str(snap)})
    return str(snap)


__all__ = [
    "GGUF_MARKER",
    "compose_resolved",
    "composed_digest",
    "fetch_gguf_snapshot",
    "read_marker",
    "write_marker",
]
