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
from .refs import TensorhubRef

logger = logging.getLogger(__name__)

# Written into the composed snapshot dir after materialization; the loading
# layer's lane detection reads it (with a structural fallback for a dir that
# lost the marker mid-crash).
GGUF_MARKER = ".cozy-gguf.json"

FP4_CORES_MIN_SM = 100
GGUF_TE_FP8_FACTOR = 0.5
GGUF_QUALITY_ORDER = (
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q5_1",
    "q5_0",
    "q4_k_m",
    "q4_k_s",
    "q4_1",
    "q4_0",
    "q3_k_m",
    "q3_k_s",
    "q2_k",
)


@dataclasses.dataclass(frozen=True)
class GgufPick:
    flavor: str
    estimated_vram_gb: float
    te_offload: bool = False


def gguf_qtype(flavor: str) -> str:
    token = str(flavor or "").strip().lower()
    if not token.startswith("gguf-"):
        return ""
    qtype = token.removeprefix("gguf-")
    return qtype if qtype in GGUF_QUALITY_ORDER else ""


def is_denoiser_weight_path(path: str) -> bool:
    p = str(path or "").strip().lstrip("/")
    head, _, rest = p.partition("/")
    return head in ("transformer", "unet") and bool(rest) and rest != "config.json"


def _is_text_encoder_weight(path: str) -> bool:
    head, _, rest = str(path or "").strip().lstrip("/").partition("/")
    return head.startswith("text_encoder") and bool(rest)


def select_gguf(
    resolved: WorkerResolvedRepo,
    *,
    gpu_sm: int,
    free_vram_gb: float,
    installed_libs: Tuple[str, ...] = (),
) -> Optional[GgufPick]:
    """Pick a local-only GGUF flavor after every native rung misses."""
    if gpu_sm <= 0 or gpu_sm >= FP4_CORES_MIN_SM:
        return None
    files = resolved.files
    if not any(f.path.strip().lstrip("/") == "model_index.json" for f in files):
        return None

    free = float(free_vram_gb)
    base_gb = float(resolved.size_bytes or sum(f.size_bytes for f in files)) / 1e9
    # The regular loader can keep the base resident or cast its denoiser to
    # fp8 at load. GGUF is the rung after those, never a preference over them.
    from .loading import FP8_STORAGE_FIT_FACTOR

    if base_gb <= free or base_gb * FP8_STORAGE_FIT_FACTOR <= free:
        return None

    from .ladder import placement_for_flavor, placement_from_metadata

    libs = frozenset(installed_libs)
    for row in resolved.sibling_flavors:
        if gguf_qtype(row.flavor) or row.size_bytes <= 0:
            continue
        placement = placement_from_metadata(row.placement) or placement_for_flavor(row.flavor)
        if placement is None:
            continue
        if placement.admits_sm(gpu_sm) and all(lib in libs for lib in placement.engines):
            if row.size_bytes / 1e9 <= free:
                return None

    non_denoiser = sum(f.size_bytes for f in files if not is_denoiser_weight_path(f.path))
    text_encoder = sum(f.size_bytes for f in files if _is_text_encoder_weight(f.path))
    overhead_gb = max(0, non_denoiser - text_encoder) / 1e9
    te_gb = text_encoder / 1e9
    rows = sorted(
        (
            (GGUF_QUALITY_ORDER.index(qtype), row)
            for row in resolved.sibling_flavors
            if (qtype := gguf_qtype(row.flavor)) and row.size_bytes > 0
        ),
        key=lambda item: (item[0], item[1].flavor),
    )
    for _, row in rows:
        gguf_gb = row.size_bytes / 1e9
        resident = gguf_gb + overhead_gb + GGUF_TE_FP8_FACTOR * te_gb
        if resident <= free:
            return GgufPick(row.flavor, resident)
        offloaded = gguf_gb + overhead_gb
        if offloaded <= free:
            return GgufPick(row.flavor, offloaded, te_offload=True)
    return None


def maybe_rebind_gguf(
    binding: Any,
    *,
    resolved: Optional[WorkerResolvedRepo] = None,
    gpu_sm: Optional[int] = None,
    free_vram_gb: Optional[float] = None,
    installed_libs: Optional[Tuple[str, ...]] = None,
) -> Any:
    """Fail-open local CLI fold; production resolution remains hub-owned."""
    if (
        getattr(binding, "source", "") != "tensorhub"
        or getattr(binding, "flavor", "")
        or getattr(binding, "storage_dtype", "")
        or getattr(binding, "components", ())
    ):
        return binding
    try:
        from ..api.binding import rebind_pick, wire_ref
        from .hub_client import resolve_repo
        from .hub_policy import detect_worker_capabilities
        from .memory import get_available_vram_gb
        from .refs import parse_model_ref

        thref = parse_model_ref(wire_ref(binding)).tensorhub
        if thref is None or thref.digest or thref.flavor:
            return binding
        caps = (
            detect_worker_capabilities()
            if gpu_sm is None or installed_libs is None
            else None
        )
        pick = select_gguf(
            resolved or resolve_repo(thref),
            gpu_sm=caps.gpu_sm if gpu_sm is None and caps is not None else int(gpu_sm or 0),
            free_vram_gb=(
                get_available_vram_gb() if free_vram_gb is None else free_vram_gb
            ),
            installed_libs=(
                tuple(caps.installed_libs)
                if installed_libs is None and caps is not None
                else tuple(installed_libs or ())
            ),
        )
        return rebind_pick(binding, flavor=pick.flavor) if pick else binding
    except Exception as exc:
        logger.debug("local GGUF selection failed open: %s", exc)
        return binding


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
    "FP4_CORES_MIN_SM",
    "GGUF_MARKER",
    "GGUF_QUALITY_ORDER",
    "GgufPick",
    "compose_resolved",
    "composed_digest",
    "fetch_gguf_snapshot",
    "gguf_qtype",
    "is_denoiser_weight_path",
    "maybe_rebind_gguf",
    "read_marker",
    "select_gguf",
    "write_marker",
]
