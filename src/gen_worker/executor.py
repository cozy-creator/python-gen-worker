"""Job execution: intake, GPU semaphore, deadline + cancellation watchdog,
sync-on-thread / async-on-loop, JobProgress deltas, result send, and the
worker-side model seam (ensure-local + setup injection + ModelOp handling).

One dispatch path for every endpoint kind. Everything runs on the single
asyncio loop; sync tenant code runs in threads via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import threading
import time
import typing
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


import msgspec

from .api.binding import wire_ref
from .api.errors import (
    ArtifactTransferError,
    CanceledError,
    RetryableError,
    ValidationError,
)
from .api.streaming import BatchItemDelta, Done, Error, IncrementalTokenDelta
from .api.types import Compute, MediaAsset
from .capability import HardwareUnmetError, InsufficientDiskError
from .input_assets import cleanup_input_assets, materialize_input_assets
from .models import disk_gc
from .models import residency as residency_mod
from .models.memory import estimate_cuda_resident_gb, get_available_ram_gb
from .models.cache_paths import tensorhub_cas_dir
from .models.download import ensure_local
from .models.errors import UrlExpiredError
from .models.residency import Residency
from .pb import worker_scheduler_pb2 as pb
from .registry import EndpointSpec
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)
from .utils import lora as lora_util

_CONTEXT_BY_KIND: Dict[str, type] = {
    "inference": RequestContext,
    "conversion": ConversionContext,
    "dataset": DatasetContext,
    "training": TrainingContext,
}

logger = logging.getLogger(__name__)

INLINE_RESULT_MAX_BYTES = 64 * 1024
_CANCEL_GRACE_S = 5.0
_STUCK_THREAD_RECYCLE_S = 30.0
_DOWNLOAD_RETRIES = 3
_PROGRESS_EVENT_MIN_INTERVAL_S = 5.0
_GiB = 1024 ** 3
# Disk headroom preserved beyond a download's known size (#370).
_DISK_GC_MARGIN_BYTES = 2 * _GiB
# Refs used within the grace window are not disk-GC candidates.
_DISK_GC_GRACE_S = 300.0

try:  # torch is optional at import time; the executor works without it.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# Credential material inside exception messages (auth headers, presigned-URL
# query params). Redacted in place — replacing the whole message with
# "internal error" made every download/publish failure undiagnosable from the
# hub (pods ship no logs; presigned URLs carry X-Amz-* params).
_REDACTIONS = (
    re.compile(r"Bearer\s+[^\s\"'&]+"),
    re.compile(r"(?:X-Amz-[A-Za-z0-9-]+|Signature)=[^&\s\"']*"),
)


def _sanitize(message: str) -> str:
    out = str(message or "").strip()
    for pat in _REDACTIONS:
        out = pat.sub("[redacted]", out)
    return out[:1024]


def _reserved_repo_info(payload: Any, field_name: str) -> Dict[str, Any]:
    """``payload.source`` / ``payload.destination`` as a plain dict ({} when
    absent). Producer payloads carry these reserved-name structs (#376)."""
    obj = getattr(payload, field_name, None)
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        out = msgspec.to_builtins(obj)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def _map_exception(exc: BaseException) -> Tuple[int, str]:
    """-> (JobStatus, safe_message)."""
    if isinstance(exc, (CanceledError, asyncio.CancelledError)):
        return pb.JOB_STATUS_CANCELED, "canceled"
    if isinstance(exc, (ValidationError, msgspec.ValidationError, msgspec.DecodeError, ValueError)):
        return pb.JOB_STATUS_INVALID, _sanitize(str(exc) or "invalid input")
    if isinstance(exc, RetryableError):
        return pb.JOB_STATUS_RETRYABLE, _sanitize(str(exc) or "retryable error")
    if isinstance(exc, ArtifactTransferError) and getattr(exc, "retryable", False):
        return pb.JOB_STATUS_RETRYABLE, _sanitize(str(exc) or "artifact transfer failed")
    if isinstance(exc, HardwareUnmetError):
        return pb.JOB_STATUS_RETRYABLE, _sanitize(str(exc) or "hardware unmet")
    if isinstance(exc, UrlExpiredError):
        # Hub-side URL staleness, not a client problem — retry re-mints URLs.
        return pb.JOB_STATUS_RETRYABLE, "model download url expired"
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return pb.JOB_STATUS_RETRYABLE, "out of memory"
    # Unexpected exception: keep it terse but NEVER opaque — "internal error"
    # made every novel worker-side failure undiagnosable from the hub (pods
    # ship no logs). Class name + sanitized first line is safe and decisive.
    detail = _sanitize(str(exc).splitlines()[0] if str(exc) else "")
    label = type(exc).__name__
    return pb.JOB_STATUS_FATAL, f"{label}: {detail}"[:512] if detail else label


def _output_media_seconds(output: Any) -> float:
    """Sum probed MEDIA seconds (e.g. ``VideoAsset.duration_s``) across the
    job output. Billing source for ``per_output_second`` (th#572)."""
    total = 0.0
    seen: set = set()
    stack = [output]
    while stack:
        item = stack.pop()
        if item is None or isinstance(item, (str, bytes, bytearray, int, float, bool)):
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, MediaAsset):
            d = getattr(item, "duration_s", None)
            if isinstance(d, (int, float)) and d > 0:
                total += float(d)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        elif isinstance(item, msgspec.Struct):
            stack.extend(getattr(item, f, None) for f in item.__struct_fields__)
    return total


# ---------------------------------------------------------------------------
# Model seam: models.download (ensure-local) + models.residency (tier map),
# with ModelEvent emission. Single-loop, per-ref asyncio locks — no
# check-then-create races.
# ---------------------------------------------------------------------------


def _snapshot_to_resolved(snap: pb.Snapshot) -> Dict[str, Any]:
    return {
        "snapshot_digest": snap.digest,
        "files": [
            {"path": f.path, "size_bytes": f.size_bytes, "blake3": f.blake3, "url": f.url}
            for f in snap.files
        ],
    }


def _model_op_error_vocab(exc: BaseException) -> str:
    """Contract §9 ModelEvent.error vocabulary for LOAD/UNLOAD failures."""
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return "oom"
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    return "load_failed"


def _is_corrupt_load_error(exc: BaseException) -> bool:
    """Errors a truncated/corrupt snapshot produces at weights-load time
    (gw#408). Broad on purpose: the digest re-verify gate downstream
    separates real corruption from code bugs — a verified-clean tree
    re-raises the original error instead of quarantining."""
    import errno as _errno
    import struct as _struct

    if isinstance(exc, OSError):
        # e.g. "Unable to load weights from checkpoint file" (raised as
        # OSError by transformers/diffusers), FileNotFoundError from a
        # half-built tree. Resource exhaustion is not corruption.
        return getattr(exc, "errno", None) not in (_errno.ENOSPC, _errno.ENOMEM)
    if isinstance(exc, _struct.error):
        return True
    return type(exc).__name__ in (
        "SafetensorError", "HeaderTooLarge", "MetadataIncompleteBuffer",
        "UnpicklingError", "JSONDecodeError",
    )


def _is_terminal_download_error(exc: BaseException) -> bool:
    if isinstance(exc, (UrlExpiredError, InsufficientDiskError)):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        # requests.HTTPError carries the code on .response, not the exception.
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    return isinstance(exc, (ValueError, KeyError))


_RESIDENCY_STATE_TO_PB = {
    residency_mod.ON_DISK: pb.MODEL_STATE_ON_DISK,
    residency_mod.IN_RAM: pb.MODEL_STATE_IN_RAM,
    residency_mod.IN_VRAM: pb.MODEL_STATE_IN_VRAM,
    residency_mod.EVICTED: pb.MODEL_STATE_EVICTED,
}

_TIER_TO_PB = {
    residency_mod.Tier.VRAM: pb.RESIDENCY_TIER_VRAM,
    residency_mod.Tier.RAM: pb.RESIDENCY_TIER_RAM,
    residency_mod.Tier.DISK: pb.RESIDENCY_TIER_DISK,
}


class ModelStore:
    """The worker's model seam: ensure-local with retries, the residency map,
    and disk retention (#370). All tier transitions flow through
    :class:`~gen_worker.models.residency.Residency`, whose events this store
    forwards as wire ``ModelEvent``s."""

    def __init__(
        self,
        emit: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        hf_home: str = "",
        hf_token: str = "",
        cache_dir: Optional[Path] = None,
        vram_budget_bytes: Optional[int] = None,
        disk_free_bytes_fn: Optional[Callable[[], int]] = None,
    ) -> None:
        self._emit = emit
        self._hf_home = hf_home or None
        self._hf_token = hf_token or None
        self._cache_dir = cache_dir or tensorhub_cas_dir()
        self.residency = Residency(
            on_event=self._on_residency_event, vram_budget_bytes=vram_budget_bytes,
        )
        self._locks: Dict[str, asyncio.Lock] = {}
        self._bindings: Dict[str, Any] = {}
        self.keep: set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._index = disk_gc.RefIndex(self._cache_dir)
        self._disk_free = disk_free_bytes_fn or self._default_disk_free
        # Refs whose on-disk snapshot passed integrity verification THIS boot
        # (gw#408): a cached snapshot is re-verified on first use per process
        # so pod-churn corruption can never be trusted forever.
        self._verified: set[str] = set()

    def _default_disk_free(self) -> int:
        p = Path(self._cache_dir)
        for candidate in (p, *p.parents):  # cache dir may not exist yet
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                continue
        return 0

    # ---- events ------------------------------------------------------------

    def bind_loop(self) -> None:
        """Capture the running loop so residency events raised from worker
        threads (demote/promote via to_thread) still reach the wire."""
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def _on_residency_event(self, ref: str, state: str, vram_bytes: int) -> None:
        pb_state = _RESIDENCY_STATE_TO_PB.get(state)
        if pb_state is None:
            return
        kw: Dict[str, Any] = {}
        if state == residency_mod.IN_VRAM:
            kw["vram_bytes"] = int(vram_bytes)
        coro = self._event(ref, pb_state, **kw)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is not None and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            else:
                coro.close()
            return
        loop.create_task(coro)

    async def _event(self, ref: str, state: int, **kw: Any) -> None:
        await self._emit(pb.WorkerMessage(model_event=pb.ModelEvent(ref=ref, state=state, **kw)))

    # ---- residency facade ----------------------------------------------------

    def residency_snapshot(self) -> List[pb.ModelResidency]:
        return [
            pb.ModelResidency(ref=ref, tier=_TIER_TO_PB[tier], vram_bytes=vram)
            for ref, tier, vram in self.residency.snapshot()
        ]

    def local_path(self, ref: str) -> Optional[Path]:
        return self.residency.local_path(ref)

    # ---- disk retention (#370) ------------------------------------------------

    def rescan_disk(self) -> None:
        """Boot-time truth: re-register still-present downloads from the
        persisted ref index so Hello.models and GC see what disk holds."""
        for ref, ent in self._index.entries().items():
            p = Path(str(ent.get("path") or ""))
            if p.exists():
                if self.residency.tier(ref) is None:
                    self.residency.track_disk(ref, p)
            else:
                self._index.remove(ref)

    def gc_disk(self, target_free_bytes: int, *, exclude: Tuple[str, ...] = ()) -> None:
        """Evict LRU disk-tier refs until free disk reaches the target.
        Non-keep refs go first (grace-honoring, then grace-ignoring); under
        keep-pressure the escape hatch evicts LRU `keep` refs too (contract §7
        — EVICTED is emitted so the hub re-downloads when demand returns).
        In-use / loaded refs are never touched."""
        for include_keep, honor_grace in (
            (False, True), (False, False), (True, True), (True, False),
        ):
            for ref in self._gc_candidates(include_keep, honor_grace, exclude):
                if self._disk_free() >= target_free_bytes:
                    return
                self._evict_disk_ref(ref)

    def _gc_candidates(
        self, include_keep: bool, honor_grace: bool, exclude: Tuple[str, ...]
    ) -> List[str]:
        now = time.time()
        out: List[Tuple[float, str]] = []
        for ref in self.residency.refs_in(residency_mod.Tier.DISK):
            if ref in exclude or self.residency.in_use(ref):
                continue
            if (ref in self.keep) != include_keep:
                continue
            last = self._index.last_used(ref)
            if honor_grace and (now - last) < _DISK_GC_GRACE_S:
                continue
            out.append((last, ref))
        out.sort()
        return [r for _, r in out]

    def _evict_disk_ref(self, ref: str) -> None:
        path = self.residency.local_path(ref) or self._index.path(ref)
        if not self.residency.evict(ref):  # refuses in-use entries; emits EVICTED
            return
        if path is not None:
            disk_gc.delete_ref_bytes(ref, path, self._cache_dir)
            disk_gc.sweep_orphan_blobs(self._cache_dir)
        self._index.remove(ref)

    async def _ensure_disk_headroom(self, ref: str, needed_bytes: int) -> None:
        target = int(needed_bytes) + _DISK_GC_MARGIN_BYTES
        if self._disk_free() >= target:
            return
        await asyncio.to_thread(self.gc_disk, target, exclude=(ref,))
        free = self._disk_free()
        if free < target:
            await self._event(ref, pb.MODEL_STATE_FAILED, error="insufficient_disk")
            raise InsufficientDiskError(
                f"need {needed_bytes} bytes for {ref}; {free} free after disk GC",
                available_bytes=free, required_bytes=needed_bytes,
                path=str(self._cache_dir),
            )

    # ---- ensure-local ----------------------------------------------------------

    def _lock(self, ref: str) -> asyncio.Lock:
        return self._locks.setdefault(ref, asyncio.Lock())

    def register_binding(self, ref: str, binding: Any) -> None:
        """Endpoint-spec binding for ``ref`` — supplies files/provider on
        download paths that only carry the bare ref (ModelOp, startup
        prefetch), so ``files=`` selections apply everywhere (#377)."""
        self._bindings.setdefault(ref, binding)

    async def ensure_local(
        self,
        ref: str,
        snapshot: Optional[pb.Snapshot] = None,
        *,
        binding: Any = None,
    ) -> Path:
        """Materialize `ref` on disk. Transient failures retry with backoff;
        terminal (4xx-class) failures raise immediately. Emits ModelEvents.
        ``binding`` (when known) supplies provider + file-selection metadata;
        bare-ref callers fall back to the registered endpoint binding."""
        self.bind_loop()
        if binding is None:
            binding = self._bindings.get(ref)
        async with self._lock(ref):
            cached = self.residency.local_path(ref)
            # A digest-carrying snapshot is authoritative: a cached
            # materialization of the SAME ref at a DIFFERENT digest is stale
            # (flavor re-published — e.g. compile-cache digest-change
            # re-adoption, e2e#117 live find #7) and must not short-circuit.
            want = ""
            if snapshot is not None and snapshot.digest:
                want = snapshot.digest.split(":", 1)[-1].strip().lower()
            if cached is not None and cached.exists() and (not want or cached.name == want):
                if ref in self._verified:
                    self._index.touch(ref)
                    return cached
                # First use this boot: verify before trusting (gw#408). A
                # pod-churn-truncated snapshot used to fatal every load until
                # a manual delete; now it is quarantined + re-materialized.
                ok, bad = await asyncio.to_thread(
                    self._verify_snapshot_tree, cached, snapshot
                )
                if ok:
                    self._verified.add(ref)
                    self._index.touch(ref)
                    return cached
                logger.error(
                    "snapshot for %s failed first-use verification "
                    "(%d bad files); quarantining and re-materializing",
                    ref, len(bad),
                )
                # Quarantine emits EVICTED; the re-download below emits
                # DOWNLOADING/ON_DISK (or FAILED on a terminal error) — the
                # hub sees the true story, not a spurious FAILED.
                await asyncio.to_thread(self._quarantine_snapshot, ref, cached, bad)
                # fall through to a fresh download below
            if snapshot is not None and snapshot.files:
                # Sizes are known up front for tensorhub snapshots: gate on
                # disk headroom, GC-ing LRU refs first (#370).
                await self._ensure_disk_headroom(
                    ref, sum(int(f.size_bytes) for f in snapshot.files)
                )
            last_progress = 0.0

            def _progress(done: int, total: Optional[int]) -> None:
                nonlocal last_progress
                now = time.monotonic()
                if now - last_progress < _PROGRESS_EVENT_MIN_INTERVAL_S:
                    return
                last_progress = now
                assert self._loop is not None
                asyncio.run_coroutine_threadsafe(
                    self._event(ref, pb.MODEL_STATE_DOWNLOADING,
                                bytes_done=int(done), bytes_total=int(total or 0)),
                    self._loop,
                )

            await self._event(ref, pb.MODEL_STATE_DOWNLOADING)
            delay = 1.0
            for attempt in range(1, _DOWNLOAD_RETRIES + 1):
                try:
                    resolved = None
                    if snapshot is not None and snapshot.digest:
                        resolved = _snapshot_to_resolved(snapshot)
                    path = await ensure_local(
                        ref,
                        provider=getattr(binding, "provider", None),
                        snapshot=resolved,
                        cache_dir=self._cache_dir,
                        hf_home=self._hf_home,
                        hf_token=self._hf_token,
                        allow_patterns=tuple(getattr(binding, "files", ()) or ()),
                        progress=_progress,
                    )
                    self.residency.track_disk(ref, path)
                    # tree_bytes stats every file — off-loop (gw#407: no
                    # multi-GB directory walks on the event loop).
                    size = await asyncio.to_thread(disk_gc.tree_bytes, path)
                    self._index.record(ref, path, size)
                    # Fresh downloads were digest-verified by the downloader.
                    self._verified.add(ref)
                    return path
                except Exception as exc:
                    terminal = _is_terminal_download_error(exc) or attempt >= _DOWNLOAD_RETRIES
                    if terminal:
                        await self._event(ref, pb.MODEL_STATE_FAILED,
                                          error=self._error_vocab(exc))
                        raise
                    logger.warning("download of %s failed (attempt %d): %s; retrying in %.1fs",
                                   ref, attempt, exc, delay)
                    await asyncio.sleep(delay)
                    delay *= 4
            raise RuntimeError("unreachable")

    # ---- snapshot integrity (gw#408) -------------------------------------------

    def _verify_snapshot_tree(
        self, path: Path, snapshot: Optional[pb.Snapshot]
    ) -> Tuple[bool, List[str]]:
        """Integrity of a materialized snapshot (worker thread; blocking IO).

        With a resolved manifest every regular file is checked against its
        declared size AND blake3 digest; files the manifest cannot cover
        (reassembled chunked originals, merged single-file checkpoints) plus
        manifest-less trees (hf/civitai) get the structural safetensors check
        (header parses + every declared tensor byte present). Returns
        ``(ok, bad_digests)`` — the digests name blobs to quarantine."""
        from .models.cozy_snapshot import _is_part_file, _is_parts_manifest, _norm_rel_path
        from .models.loading import safetensors_file_valid
        from .presigned_upload import blake3_hash_file

        p = Path(path)
        bad: List[str] = []
        covered: set[Path] = set()
        files = list(snapshot.files) if snapshot is not None else []
        if files and p.is_dir():
            for f in files:
                if _is_parts_manifest(f.path) or _is_part_file(f.path):
                    continue  # not materialized: parts live only in blobs/
                try:
                    dst = p / _norm_rel_path(f.path)
                except ValueError:
                    continue
                covered.add(dst)
                digest = (f.blake3 or "").strip().lower()
                try:
                    if not dst.exists():
                        raise ValueError("missing")
                    if f.size_bytes and dst.stat().st_size != int(f.size_bytes):
                        raise ValueError("size mismatch")
                    if digest and blake3_hash_file(dst).lower() != digest:
                        raise ValueError("blake3 mismatch")
                except (OSError, ValueError) as exc:
                    logger.warning("snapshot file %s/%s corrupt: %s", p.name, f.path, exc)
                    bad.append(digest or f.path)
        try:
            candidates = [p] if p.is_file() else sorted(p.rglob("*.safetensors"))
        except OSError:
            candidates = []
        for st in candidates:
            if st in covered or st.suffix != ".safetensors":
                continue
            if not safetensors_file_valid(st):
                logger.warning("snapshot file %s structurally invalid (truncated?)", st)
                bad.append(str(st.relative_to(p)) if st != p else st.name)
        return (not bad, bad)

    def _quarantine_snapshot(self, ref: str, path: Path, bad: List[str]) -> None:
        """Evict + delete a corrupt materialization AND the corrupt blobs it
        was built from, so re-materialization re-downloads instead of
        re-linking the same bad bytes. Emits EVICTED via residency."""
        from .models.cozy_snapshot import delete_blobs

        self._verified.discard(ref)
        self.residency.evict(ref, force=True)
        disk_gc.delete_ref_bytes(ref, Path(path), self._cache_dir)
        delete_blobs(self._cache_dir, [d for d in bad if "/" not in d and "." not in d])
        disk_gc.sweep_orphan_blobs(self._cache_dir)
        self._index.remove(ref)

    async def refetch_corrupt(
        self, ref: str, snapshot: Optional[pb.Snapshot] = None, *, binding: Any = None
    ) -> Optional[Path]:
        """Load-failure path (gw#408): a weights load failed with a
        corruption-shaped error — digest-verify the snapshot. A clean tree
        returns None (the failure is NOT corruption; caller re-raises); a
        dirty tree is quarantined and re-materialized, returning the fresh
        path for exactly one load retry."""
        path = self.residency.local_path(ref) or self._index.path(ref)
        if path is None:
            return None
        async with self._lock(ref):
            ok, bad = await asyncio.to_thread(self._verify_snapshot_tree, Path(path), snapshot)
            if ok:
                self._verified.add(ref)
                return None
            logger.error(
                "load failure traced to corrupt snapshot for %s (%d bad files); "
                "quarantining and re-materializing", ref, len(bad),
            )
            await asyncio.to_thread(self._quarantine_snapshot, ref, Path(path), bad)
        return await self.ensure_local(ref, snapshot, binding=binding)

    @staticmethod
    def _error_vocab(exc: BaseException) -> str:
        if isinstance(exc, UrlExpiredError):
            return "url_expired"
        if isinstance(exc, InsufficientDiskError):
            return "insufficient_disk"
        text = str(exc).lower()
        if "expired" in text or "403" in text:
            return "url_expired"
        if "digest" in text or "blake3" in text or "hash" in text:
            return "digest_mismatch"
        if "no space" in text or "disk" in text:
            return "insufficient_disk"
        return "download_failed"


# ---------------------------------------------------------------------------
# Endpoint instances (setup/warmup lifecycle)
# ---------------------------------------------------------------------------


@dataclass
class _ClassRecord:
    cls: type
    specs: List[EndpointSpec] = dc_field(default_factory=list)
    instance: Any = None
    server: Any = None  # ServerHandle for runtime="vllm"/"llama-server"
    ready: bool = False
    failed: Optional[str] = None
    lock: asyncio.Lock = dc_field(default_factory=asyncio.Lock)


@dataclass
class _Job:
    request_id: str
    attempt: int
    spec: Optional[EndpointSpec]
    ctx: Optional[RequestContext] = None
    task: Optional[asyncio.Task] = None
    exec_task: Optional[asyncio.Task] = None
    finished: bool = False
    superseded: bool = False
    admitted_at: float = dc_field(default_factory=time.monotonic)


class _GpuSlotLease:
    """Thread-safe handle for a job's GPU slot (#382).

    Blob uploads and result sends are network/CPU work; holding the GPU
    semaphore across them idles the GPU for longer than the model's own
    compute on turbo image models. The lease lets ``RequestContext`` release
    the slot from the handler thread while ``save_bytes`` waits on the
    network (re-acquiring before returning to tenant code), and lets the
    executor free the slot as soon as ``_execute`` returns — before
    result-blob upload and result send. Transitions are lock-guarded so a
    hold is released at most once.
    """

    __slots__ = ("_sem", "_loop", "_lock", "_held")

    def __init__(self, sem: asyncio.Semaphore, loop: asyncio.AbstractEventLoop) -> None:
        self._sem = sem
        self._loop = loop
        self._lock = threading.Lock()
        self._held = True

    def yield_slot(self) -> bool:
        """Release the slot if held (any thread). True iff this call released."""
        with self._lock:
            if not self._held:
                return False
            self._held = False
        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            self._sem.release()
        else:
            self._loop.call_soon_threadsafe(self._sem.release)
        return True

    def reacquire(self) -> None:
        """Blocking re-acquire from a handler thread."""
        asyncio.run_coroutine_threadsafe(self._sem.acquire(), self._loop).result()
        with self._lock:
            self._held = True


class Executor:
    def __init__(
        self,
        specs: List[EndpointSpec],
        send: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        settings: Any = None,
        store: Optional[ModelStore] = None,
        gpu_slots: int = 1,
        on_state_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self.specs: Dict[str, EndpointSpec] = {s.name: s for s in specs}
        self._send = send
        self._settings = settings
        self.store = store or ModelStore(send)
        for s in specs:
            for b in s.models.values():
                self.store.register_binding(wire_ref(b), b)
        self._gpu_semaphore = asyncio.Semaphore(max(1, gpu_slots))
        # Model loads/promotions serialize so allocator-delta measurements
        # and free-VRAM reads don't cross-contaminate (#369).
        self._load_lock = asyncio.Lock()
        # Parsed per-request LoRA state dicts, keyed by ref@digest (gw#393).
        self._adapter_cache = lora_util.AdapterCache()
        # Adapters attached to resident pipelines; requests toggle the active
        # set (gw#399). Demotion out of VRAM drops attachments.
        self._adapters = lora_util.AdapterResidency()
        self.store.residency.pre_demote = self._adapters.detach
        self._on_state_change = on_state_change or (lambda: None)
        self.file_base_url: str = ""
        self.draining = False
        self.jobs: Dict[Tuple[str, int], _Job] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        # Instance groups: specs sharing (cls, bindings) share one instance;
        # variant specs of the same class get separate instances. Function-
        # shaped endpoints (cls=None) have no instance at all.
        self._classes: Dict[Any, _ClassRecord] = {}
        for s in specs:
            if s.cls is None:
                continue
            rec = self._classes.setdefault(s.instance_key, _ClassRecord(cls=s.cls))
            rec.specs.append(s)
        # Hardware-gate failures: fn name -> (reason, detail, axes).
        self.unavailable: Dict[str, Tuple[str, str, Dict[str, str]]] = {}

    # ---- availability ----------------------------------------------------

    def gate_functions(self, gpu_info: Dict[str, Any]) -> None:
        """Run hardware gates; populate self.unavailable."""
        gpu_count = int(gpu_info.get("gpu_count") or 0)
        total_vram_gb = float(gpu_info.get("gpu_total_mem") or 0) / (1024 ** 3)
        detected_sm = str(gpu_info.get("gpu_sm") or "")
        detected_cc = (float(detected_sm) / 10.0) if detected_sm.isdigit() else None
        libs = {str(x) for x in (gpu_info.get("installed_libs") or [])}
        for name, spec in self.specs.items():
            r = spec.resources
            if spec.needs_gpu and gpu_count <= 0:
                self.unavailable[name] = (
                    "cuda_unavailable", "function requires CUDA but no GPU detected",
                    {"gpu_count": str(gpu_count)})
                continue
            if r.compute_capability is not None and detected_cc is not None \
                    and detected_cc < float(r.compute_capability):
                self.unavailable[name] = (
                    "compute_capability_unmet",
                    f"requires SM {r.compute_capability:.1f}, detected {detected_cc:.1f}",
                    {"detected_sm": f"{detected_cc:.1f}", "required_sm": f"{float(r.compute_capability):.1f}"})
                continue
            # Compare on the rounded GiB: a "24GB" card reports ~23.99GiB via
            # mem_get_info (driver/ECC reserve), which must not gate off
            # functions declaring vram_gb=24.
            if r.vram_gb is not None and total_vram_gb and round(total_vram_gb) < float(r.vram_gb):
                self.unavailable[name] = (
                    "insufficient_vram",
                    f"requires {r.vram_gb:.0f}GiB VRAM, detected {total_vram_gb:.0f}GiB",
                    {"required_vram_gb": f"{float(r.vram_gb):.0f}",
                     "detected_vram_gb": f"{total_vram_gb:.0f}"})
                continue
            missing = [lib for lib in (r.libraries or ()) if lib not in libs]
            if missing:
                import importlib.util
                missing = [m for m in missing if importlib.util.find_spec(m) is None]
            if missing:
                self.unavailable[name] = (
                    "missing_cuda_library", f"missing required libraries: {', '.join(missing)}",
                    {"missing": ",".join(missing)})

    def available_functions(self) -> List[str]:
        out = []
        for name, spec in self.specs.items():
            if name in self.unavailable or self.draining:
                continue
            if spec.cls is None:
                out.append(name)
                continue
            rec = self._classes[spec.instance_key]
            if rec.ready or (not spec.models and rec.failed is None):
                out.append(name)
        return sorted(out)

    def loading_functions(self) -> List[str]:
        avail = set(self.available_functions())
        return sorted(
            name for name, spec in self.specs.items()
            if name not in avail and name not in self.unavailable
            and spec.cls is not None
            and self._classes[spec.instance_key].failed is None
        )

    def in_flight_keys(self) -> List[Tuple[str, int]]:
        return [k for k, j in self.jobs.items() if not j.finished and not j.superseded]

    # ---- setup -------------------------------------------------------------

    async def ensure_setup(self, spec: EndpointSpec, snapshots: Optional[Dict[str, pb.Snapshot]] = None) -> Any:
        if spec.cls is None:
            return None  # function-shaped endpoint: no instance, no setup
        self.store.bind_loop()
        rec = self._classes[spec.instance_key]
        async with rec.lock:
            if rec.ready:
                await self._promote_setup_refs(spec)
                return rec.instance
            try:
                instance = await self._setup_locked(spec, rec, snapshots)
            except Exception as exc:
                # Honest failure (th#581): a function whose model download /
                # pipeline setup fails must surface a terminal per-function
                # error to the hub, not sit in loading_functions forever
                # while the worker reports READY.
                self._mark_setup_failed(rec, exc)
                raise
            if rec.failed is not None:
                # Recovery (hub-directed LOAD retry succeeded): lift the
                # per-function disable; the next StateDelta re-advertises.
                rec.failed = None
                for s in rec.specs:
                    self.unavailable.pop(s.name, None)
            rec.instance = instance
            rec.ready = True
            self._on_state_change()
            return instance

    def _mark_setup_failed(self, rec: _ClassRecord, exc: BaseException) -> None:
        if isinstance(exc, (InsufficientDiskError, RetryableError)):
            # Transient pressure (disk GC frees space / warm-tier RAM drains):
            # fail the op RETRYABLE, never disable the function.
            return
        if isinstance(exc, HardwareUnmetError):
            reason = getattr(exc, "reason", "hardware_unmet")
            axes = {str(k): str(v) for k, v in (exc.axes() or {}).items()}
        else:
            reason, axes = "setup_failed", {}
        detail = _sanitize(f"{type(exc).__name__}: {exc}")
        rec.failed = detail
        for s in rec.specs:
            self.unavailable[s.name] = (reason, detail, axes)
        self._on_state_change()

    async def _setup_locked(
        self, spec: EndpointSpec, rec: _ClassRecord,
        snapshots: Optional[Dict[str, pb.Snapshot]],
    ) -> Any:
        assert spec.cls is not None  # guarded by ensure_setup
        setup_slots = self._setup_slots(spec)
        paths: Dict[str, str] = {}
        for slot in setup_slots:
            binding = spec.models[slot]
            ref = wire_ref(binding)
            snap = (snapshots or {}).get(ref)
            path = await self.store.ensure_local(ref, snap, binding=binding)
            paths[slot] = str(path)
        compile_artifact = await self._fetch_compile_snapshot(spec, snapshots)
        # Loads serialize: concurrent setups would cross-contaminate each
        # other's allocator deltas and place_pipeline's free-VRAM reads.
        async with self._load_lock:
            await self._ensure_host_ram_for(spec, paths)
            await self._make_room_for(spec, setup_slots)
            instance = spec.cls()
            setup = getattr(instance, "setup", None)
            loaded: Dict[str, Tuple[Any, int]] = {}
            vram_before = self._vram_allocated()
            if spec.runtime:
                rec.server = await self._boot_engine_server(spec, paths)
            if callable(setup):
                kwargs, loaded = await self._injection_kwargs(
                    spec, setup, paths, server=rec.server,
                    compile_artifact=compile_artifact, snapshots=snapshots)
                if asyncio.iscoroutinefunction(setup):
                    await setup(**kwargs)
                else:
                    await asyncio.to_thread(setup, **kwargs)
            warmup = getattr(instance, "warmup", None)
            if callable(warmup):
                from . import compile_cache

                counters_before = compile_cache.inductor_counters()
                warm_t0 = time.monotonic()
                if asyncio.iscoroutinefunction(warmup):
                    await warmup()
                else:
                    await asyncio.to_thread(warmup)
                if spec.compile is not None:
                    stats = compile_cache.counters_delta(
                        counters_before, compile_cache.inductor_counters())
                    logger.info(
                        "compile-cache warmup %s: fxgraph hits=%d misses=%d "
                        "in %.1fs", spec.name,
                        stats.get("fxgraph_cache_hit", 0),
                        stats.get("fxgraph_cache_miss", 0),
                        time.monotonic() - warm_t0)
            vram_delta = max(0, self._vram_allocated() - vram_before)
            self._register_residency(spec, setup_slots, loaded, vram_delta)
        return instance

    def _register_residency(
        self,
        spec: EndpointSpec,
        setup_slots: List[str],
        loaded: Dict[str, Tuple[Any, int]],
        total_delta: int,
    ) -> None:
        """Honest per-ref residency after a setup (#369). Worker-constructed
        pipelines carry their own measured allocator delta AND the object
        (Residency owns it: demote/promote actually move memory). Refs the
        tenant loaded inside setup() split the residual delta — no object,
        so their VRAM is only reclaimable by record teardown."""
        res = self.store.residency
        per_ref: Dict[str, Tuple[Any, int]] = {}
        for slot in setup_slots:
            ref = wire_ref(spec.models[slot])
            obj, measured = loaded.get(slot, (None, 0))
            prev_obj, prev_bytes = per_ref.get(ref, (None, 0))
            per_ref[ref] = (obj or prev_obj, prev_bytes + measured)
        residual = max(0, total_delta - sum(b for _, b in per_ref.values()))
        opaque = [r for r, (obj, _) in per_ref.items() if obj is None]
        share = residual // len(opaque) if opaque else 0
        for ref, (obj, measured) in per_ref.items():
            vram = measured + (share if obj is None else 0)
            if vram > 0:
                res.track_vram(ref, obj, vram_bytes=vram)
            elif obj is not None and int(estimate_cuda_resident_gb(obj) * _GiB) > 0:
                res.track_vram(ref, obj)  # measured via cuda-resident estimate
            else:
                res.track_ram(ref, obj)   # CPU-only host / offloaded load

    async def _promote_setup_refs(self, spec: EndpointSpec) -> None:
        """RunJob/LOAD for a demoted (RAM-tier) instance: swap the pipelines
        back into VRAM instead of a cold reload (#371)."""
        res = self.store.residency
        refs = [wire_ref(spec.models[s]) for s in self._setup_slots(spec)]
        cuda_host = torch is not None and torch.cuda.is_available()
        if any(res.tier(r) is residency_mod.Tier.RAM for r in refs):
            async with self._load_lock:
                for ref in refs:
                    if res.tier(ref) is residency_mod.Tier.RAM:
                        ok = await asyncio.to_thread(res.promote, ref)
                        self._on_state_change()
                        if not ok and cuda_host and res.tier(ref) is residency_mod.Tier.RAM:
                            # Promote refused/rolled back (gw#409): fail the
                            # job RETRYABLE at promote time — never hand a
                            # handler a pipeline that fatals mid-denoise.
                            raise RetryableError(
                                f"promotion of {ref} to VRAM failed; retrying"
                            )
        for ref in refs:
            res.touch(ref)

    @staticmethod
    def _worker_loaded_slots(spec: EndpointSpec) -> set:
        """Setup slots the WORKER materializes in host RAM (class-typed
        annotations loaded via ``from_pretrained``). str/Path slots and
        engine runtimes (vllm/llama-server) stream weights themselves and
        must not be counted against the host-RAM admission gate."""
        if spec.cls is None or spec.runtime:
            return set()
        setup = getattr(spec.cls, "setup", None)
        if setup is None:
            return set()
        try:
            hints = typing.get_type_hints(setup)
        except Exception:
            return set()
        return {
            name for name, ann in hints.items()
            if isinstance(ann, type) and callable(getattr(ann, "from_pretrained", None))
        }

    async def _ensure_host_ram_for(self, spec: EndpointSpec, paths: Dict[str, str]) -> None:
        """Load-size-aware host-RAM admission (gw#407). ``from_pretrained``
        stages the full weight set in host RAM before placement; loading into
        a nearly-full host pushes it into reclaim-thrash that stalls the whole
        process — including gRPC keepalive acks — so the hub disconnects and
        requeues in a livelock (J17: 16 SDXL variants on a 31GB host). Free
        warm RAM-tier LRU pipelines first; when even that cannot cover the
        incoming bytes plus the floor, fail RETRYABLE instead of thrashing.

        Only worker-loaded (pipeline-typed) slots count: tenant-owned and
        engine-runtime slots do not stage full weight sets in host RAM."""
        slots = self._worker_loaded_slots(spec)
        if not paths or not slots:
            return
        incoming = 0
        for slot, p in paths.items():
            if slot in slots:
                incoming += await asyncio.to_thread(disk_gc.tree_bytes, Path(p))
        if incoming <= 0:
            return
        if await asyncio.to_thread(self.store.residency.make_room_ram, incoming):
            return
        avail = get_available_ram_gb()
        raise RetryableError(
            f"insufficient host RAM to load {spec.name}: "
            f"~{incoming / _GiB:.1f}GiB incoming, {avail:.1f}GiB available "
            "after releasing the warm tier"
        )

    async def _make_room_for(self, spec: EndpointSpec, setup_slots: List[str]) -> None:
        """Evict idle LRU pipelines before loading instead of degrading the
        new load down the offload ladder (#371). Estimate: per-ref vram_hint
        from a prior load, else the endpoint's declared vram_gb."""
        res = self.store.residency
        refs = [wire_ref(spec.models[s]) for s in setup_slots]
        needed = sum(res.vram_hint(r) for r in refs)
        if needed <= 0 and spec.resources.vram_gb:
            needed = int(float(spec.resources.vram_gb) * _GiB)
        if needed <= 0:
            return
        if await asyncio.to_thread(res.make_room, needed):
            self._on_state_change()
            return
        # Movable demotions weren't enough: tear down idle records holding
        # non-movable LRU victims (tenant-loaded refs).
        for ref in res.lru_vram_victims():
            rec = self._record_holding(ref)
            if rec is None or self._record_in_use(rec):
                continue
            await self._vacate_record(rec)
            if await asyncio.to_thread(res.make_room, needed):
                break
        self._on_state_change()

    @staticmethod
    def _setup_slots(spec: EndpointSpec) -> List[str]:
        """Model slots loaded once at setup time. Classes without setup()
        take their models per call via handler-parameter injection."""
        if spec.cls is None or not spec.models:
            return []
        if getattr(spec.cls, "setup", None) is None:
            return []
        return list(spec.models)

    async def _boot_engine_server(self, spec: EndpointSpec, paths: Dict[str, str]) -> Any:
        """Boot the runtime="vllm"/"llama-server" subprocess and health-wait."""
        from .runtimes.server import RUNTIME_FACTORIES

        factory = RUNTIME_FACTORIES[spec.runtime]  # validated at decoration
        if not paths:
            raise ValidationError(
                f"runtime={spec.runtime!r} on {spec.name!r} requires a model binding"
            )
        model_path = next(iter(paths.values()))
        proc = factory(model_path)
        return await asyncio.to_thread(proc.start)

    async def _fetch_compile_snapshot(
        self, spec: EndpointSpec, snapshots: Optional[Dict[str, pb.Snapshot]]
    ) -> Optional[Path]:
        """Hub-attached compiled-artifact snapshot for this endpoint's family
        (#569 boot-attach, opt-in hub-side): a TRT engine cell (#390,
        preferred — bigger measured win) or an inductor cache cell. Returns
        the local artifact path, or None — missing/unusable snapshots mean
        eager, never an error."""
        if spec.compile is None or not snapshots:
            return None
        from . import compile_cache, trt_engine

        family = getattr(spec.compile, "family", "") or ""
        candidates = [
            (ref, snap) for ref, snap in snapshots.items()
            if trt_engine.is_engine_ref(ref, family)
        ] + [
            (ref, snap) for ref, snap in snapshots.items()
            if compile_cache.is_cache_ref(ref, family)
        ]
        for ref, snap in candidates:
            try:
                local = await self.store.ensure_local(ref, snap)
                return compile_cache.find_artifact(local)
            except Exception as exc:
                logger.warning(
                    "compiled-artifact snapshot %s unusable (%s); trying next/eager", ref, exc
                )
        return None

    async def _injection_kwargs(
        self,
        spec: EndpointSpec,
        setup: Callable[..., Any],
        paths: Dict[str, str],
        *,
        server: Any = None,
        compile_artifact: Optional[Path] = None,
        snapshots: Optional[Dict[str, pb.Snapshot]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Tuple[Any, int]]]:
        """Typed injection: each slot receives exactly what its ``setup``
        annotation says — a ``str``/``Path`` local path, or a constructed
        pipeline for a class annotation exposing ``from_pretrained`` (built off
        the loop; the binding dtype is honored and the worker applies its
        placement/offload policy to the result). A parameter annotated
        ``ServerHandle`` receives the booted engine server.

        Returns ``(kwargs, loaded)`` where ``loaded`` maps each pipeline slot
        to ``(pipeline, allocator_delta)`` — the per-ref measurement residency
        books (#369; caller holds the load lock)."""
        from .runtimes.server import ServerHandle

        try:
            hints = typing.get_type_hints(setup)
        except Exception:
            hints = {}
        kwargs: Dict[str, Any] = {}
        loaded: Dict[str, Tuple[Any, int]] = {}
        if server is not None:
            for pname, ann in hints.items():
                if ann is ServerHandle:
                    kwargs[pname] = server
        for slot, path in paths.items():
            ann = hints.get(slot)
            if ann is None or ann is str:
                kwargs[slot] = path
            elif ann is Path:
                kwargs[slot] = Path(path)
            elif isinstance(ann, type) and callable(getattr(ann, "from_pretrained", None)):
                from .models.loading import load_from_pretrained
                from .models.memory import place_pipeline

                binding = spec.models.get(slot)
                dtype = str(getattr(binding, "dtype", "") or "")
                storage_dtype = str(getattr(binding, "storage_dtype", "") or "")
                before = self._vram_allocated()
                try:
                    pipe = await asyncio.to_thread(
                        load_from_pretrained, ann, path, dtype=dtype,
                        storage_dtype=storage_dtype,
                    )
                except Exception as exc:
                    # Corruption-shaped load failure (gw#408): digest-verify
                    # the snapshot; quarantine + re-materialize + retry ONCE
                    # when corruption is confirmed, re-raise otherwise.
                    fresh: Optional[Path] = None
                    if binding is not None and _is_corrupt_load_error(exc):
                        ref = wire_ref(binding)
                        fresh = await self.store.refetch_corrupt(
                            ref, (snapshots or {}).get(ref), binding=binding
                        )
                    if fresh is None:
                        raise
                    logger.warning(
                        "weights load for slot %r failed on a corrupt snapshot "
                        "(%s: %s); retrying once after re-materialization",
                        slot, type(exc).__name__, exc,
                    )
                    path = str(fresh)
                    paths[slot] = path
                    pipe = await asyncio.to_thread(
                        load_from_pretrained, ann, path, dtype=dtype,
                        storage_dtype=storage_dtype,
                    )
                # Worker-owned placement/offload policy: one decider for the
                # whole worker; endpoints never write device/offload code.
                await asyncio.to_thread(place_pipeline, pipe)
                if spec.compile is not None:
                    # Opt-in acceleration against a pre-built per-SKU artifact:
                    # a TRT engine (#390, refit with this pipeline's weights)
                    # or an inductor cache (#384). No verified artifact =>
                    # stays eager. ``compile_artifact`` is hub-attached (#569).
                    await asyncio.to_thread(
                        self._enable_compiled, pipe, spec.compile, compile_artifact,
                    )
                loaded[slot] = (pipe, max(0, self._vram_allocated() - before))
                kwargs[slot] = pipe
            else:
                kwargs[slot] = path
        return kwargs, loaded

    def _enable_compiled(self, pipe: Any, cfg: Any, artifact: Optional[Path]) -> bool:
        """Arm the best available compiled path for a freshly loaded pipeline:
        a TRT engine artifact swaps the module (fail-soft), anything else goes
        through the torch.compile cache policy (which also covers the no-
        artifact and ALLOW_COLD lanes)."""
        from . import compile_cache, trt_engine

        if artifact is not None:
            try:
                meta = trt_engine.unpack_metadata(Path(artifact))
            except Exception:
                meta = None
            if meta is not None and str(meta.get("kind") or "") == "trt-engine":
                if trt_engine.enable(pipe, cfg, self.store._cache_dir, artifact):
                    return True
                artifact = None  # unusable engine: fall through to eager policy
        return compile_cache.enable(pipe, cfg, self.store._cache_dir, artifact)

    @staticmethod
    def _vram_allocated() -> int:
        if torch is not None and torch.cuda.is_available():
            try:
                return int(torch.cuda.memory_allocated())
            except Exception:
                return 0
        return 0

    async def shutdown_instances(self) -> None:
        for rec in self._classes.values():
            inst, rec.instance, rec.ready = rec.instance, None, False
            shutdown = getattr(inst, "shutdown", None)
            if inst is not None and callable(shutdown):
                try:
                    if asyncio.iscoroutinefunction(shutdown):
                        await shutdown()
                    else:
                        await asyncio.to_thread(shutdown)
                except Exception:
                    logger.exception("shutdown() failed for %s", rec.cls.__name__)
            server, rec.server = rec.server, None
            if server is not None:
                await asyncio.to_thread(server.stop)

    # ---- ModelOp -----------------------------------------------------------

    async def handle_model_op(self, op: pb.ModelOp) -> None:
        self.store.bind_loop()
        ref = op.ref
        snap = op.snapshot if op.HasField("snapshot") else None
        try:
            if op.op == pb.MODEL_OP_KIND_DOWNLOAD:
                await self.store.ensure_local(ref, snap)
            elif op.op == pb.MODEL_OP_KIND_LOAD:
                await self.store.ensure_local(ref, snap)
                loaded = False
                snapshots = {ref: snap} if snap is not None else None
                for spec in self.specs.values():
                    if ref in (wire_ref(b) for b in spec.models.values()):
                        await self.ensure_setup(spec, snapshots)
                        loaded = True
                if not loaded:
                    # No endpoint binds this ref; nothing owns a VRAM load for it.
                    await self._send(pb.WorkerMessage(model_event=pb.ModelEvent(
                        ref=ref, state=pb.MODEL_STATE_FAILED, error="load_failed")))
            elif op.op == pb.MODEL_OP_KIND_UNLOAD:
                if self._ref_in_use(ref):
                    await self._send(pb.WorkerMessage(model_event=pb.ModelEvent(
                        ref=ref, state=pb.MODEL_STATE_FAILED, error="model_in_use")))
                    return
                await self._unload_ref(ref)
            elif op.op == pb.MODEL_OP_KIND_ADOPT_COMPILE_CACHE:
                await self._adopt_compile_cache(ref, snap)
        except Exception as exc:
            logger.warning("ModelOp %s on %s failed: %s", op.op, ref, exc)
            # ensure_local already emitted FAILED for download errors; emit for
            # load/unload paths that failed outside it. OOM must say "oom" —
            # it is the orchestrator's trigger to UNLOAD a resident model for
            # headroom (contract §9 vocabulary).
            if op.op != pb.MODEL_OP_KIND_DOWNLOAD:
                await self._send(pb.WorkerMessage(model_event=pb.ModelEvent(
                    ref=ref, state=pb.MODEL_STATE_FAILED,
                    error=_model_op_error_vocab(exc))))

    async def _adopt_compile_cache(self, ref: str, snap: Optional[pb.Snapshot]) -> None:
        """Hot adoption (th#567): download+verify a compiled artifact and
        re-wrap the already-resident modules in place — weights untouched, no
        reload, one warmup. Handles BOTH cell kinds on the same rails: an
        inductor cache (#384: seed dirs + torch.compile) and a TRT engine
        (#390: deserialize + refit with the resident weights + module swap).
        ANY failure => stay eager and report ``adopt_failed:<reason>``;
        adoption must never degrade service."""
        from . import compile_cache, trt_engine

        t0 = time.monotonic()

        async def fail(reason: str, detail: str = "") -> None:
            logger.warning("compile-cache adopt %s failed: %s %s", ref, reason, detail)
            await self._send(pb.WorkerMessage(model_event=pb.ModelEvent(
                ref=ref, state=pb.MODEL_STATE_FAILED,
                error=f"adopt_failed:{reason}")))

        family = compile_cache.family_from_ref(ref)
        is_trt = trt_engine.is_engine_ref(ref)
        if not family or not (is_trt or compile_cache.is_cache_ref(ref)):
            return await fail("bad_ref")
        candidates = [
            s for s in self.specs.values()
            if s.cls is not None and s.compile is not None
            and (getattr(s.compile, "family", "") or "") == family
            and s.name not in self.unavailable
        ]
        if not candidates:
            return await fail("no_endpoint")
        resident: Dict[int, Tuple[_ClassRecord, EndpointSpec]] = {}
        for s in candidates:
            rec = self._classes[s.instance_key]
            if rec.ready:
                resident.setdefault(id(rec), (rec, s))
        if not resident:
            return await fail("not_resident")
        if self.in_flight_keys():
            # The hub schedules adoption idle-only; defensive — never touch
            # a module while any job is in flight.
            return await fail("model_in_use")
        try:
            local = await self.store.ensure_local(ref, snap)
        except Exception as exc:
            return await fail("download", str(exc))
        artifact = compile_cache.find_artifact(local)
        if artifact is None:
            return await fail("artifact_missing")
        meta: Dict[str, Any] = {}
        if not is_trt:
            try:
                meta = await asyncio.to_thread(
                    compile_cache.seed_artifact, artifact, family, self.store._cache_dir
                )
            except compile_cache.AdoptError as exc:
                return await fail(exc.reason, str(exc))
            except Exception as exc:
                return await fail("artifact_invalid", str(exc))

        # Re-wrap + warmup under the GPU slot: a job landing mid-adoption
        # queues on the semaphore instead of racing the trace/refit.
        async with self._gpu_semaphore:
            adopted: List[Tuple[_ClassRecord, EndpointSpec]] = []
            wrapped_objs: List[Any] = []
            for rec, s in resident.values():
                applied = False
                for slot in self._setup_slots(s):
                    obj = self.store.residency.obj(wire_ref(s.models[slot]))
                    if obj is None:
                        continue
                    if is_trt:
                        try:
                            await asyncio.to_thread(
                                trt_engine.load_and_wrap, obj, s.compile,
                                artifact, self.store._cache_dir,
                            )
                            applied = True
                        except compile_cache.AdoptError as exc:
                            return await fail(exc.reason, str(exc))
                        except Exception as exc:
                            return await fail("artifact_invalid", str(exc))
                    else:
                        # Graph-key parity (gw#391): the producer's low-VRAM
                        # prep mode is traced into the cells — a drifted
                        # pipeline can only miss, so reject deterministically
                        # instead of paying a warmup to find out.
                        drift = compile_cache.mode_drift(meta, obj)
                        if drift:
                            return await fail("key_mismatch", drift)
                        # Re-adoption of a re-published cell: drop the previous
                        # wrap + dynamo's in-memory code so warmup re-traces
                        # against the freshly seeded caches (gw#391).
                        compile_cache.unwrap(obj)
                        if await asyncio.to_thread(
                            compile_cache.apply, obj, s.compile, cache_ready=True
                        ):
                            applied = True
                            wrapped_objs.append(obj)
                if applied:
                    adopted.append((rec, s))
            if not adopted:
                return await fail("no_target")

            async def rollback() -> None:
                for obj in wrapped_objs:
                    compile_cache.unwrap(obj)

            counters_before = compile_cache.inductor_counters()
            warm_t0 = time.monotonic()
            warmed = 0
            for rec, _s in adopted:
                warmup = getattr(rec.instance, "warmup", None)
                if not callable(warmup):
                    continue
                try:
                    if asyncio.iscoroutinefunction(warmup):
                        await warmup()
                    else:
                        await asyncio.to_thread(warmup)
                    warmed += 1
                except Exception as exc:
                    await rollback()
                    return await fail("warmup", f"{type(exc).__name__}: {exc}")
            warmup_s = round(time.monotonic() - warm_t0, 3)
            stats = compile_cache.counters_delta(
                counters_before, compile_cache.inductor_counters())
            hits = stats.get("fxgraph_cache_hit", 0)
            misses = stats.get("fxgraph_cache_miss", 0)

            if not is_trt:
                # Honest failure mode (gw#391): ADOPTED must mean the seeded
                # cell actually served the trace. No warmup = unprovable;
                # zero hits = silently eager. Both roll back to true eager.
                if not warmed:
                    await rollback()
                    return await fail(
                        "no_warmup",
                        "no adopted endpoint defines warmup(); cache hits unprovable")
                if hits <= 0:
                    await rollback()
                    return await fail("cache_miss", (
                        f"warmup observed 0 fxgraph cache hits "
                        f"(misses={misses}, warmup={warmup_s}s) — cell useless "
                        f"on this runtime, serving eager"))

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "compile-cache adopt %s: adopted in %dms (fxgraph hits=%d misses=%d, "
            "warmup %.1fs)", ref, duration_ms, hits, misses, warmup_s)
        if misses:
            logger.warning(
                "compile-cache adopt %s: %d fxgraph misses during warmup — "
                "cell covers the declared shapes only partially", ref, misses)
        await self._send(pb.WorkerMessage(model_event=pb.ModelEvent(
            ref=ref, state=pb.MODEL_STATE_ADOPTED, duration_ms=duration_ms,
            cache_hits=hits, cache_misses=misses, warmup_s=warmup_s)))

    def _ref_in_use(self, ref: str) -> bool:
        for job in self.jobs.values():
            if job.finished or job.superseded or job.spec is None:
                continue
            if ref in (wire_ref(b) for b in job.spec.models.values()):
                return True
        return False

    def _record_holding(self, ref: str) -> Optional[_ClassRecord]:
        for rec in self._classes.values():
            if not rec.ready:
                continue
            if any(ref in (wire_ref(b) for b in s.models.values()) for s in rec.specs):
                return rec
        return None

    def _record_in_use(self, rec: _ClassRecord) -> bool:
        for s in rec.specs:
            for b in s.models.values():
                ref = wire_ref(b)
                if self._ref_in_use(ref) or self.store.residency.in_use(ref):
                    return True
        return False

    async def _vacate_record(self, rec: _ClassRecord) -> None:
        """Tear an instance down and book every ref it held back to disk —
        registry state and instance state move together (#369)."""
        inst, rec.instance, rec.ready = rec.instance, None, False
        shutdown = getattr(inst, "shutdown", None)
        if inst is not None and callable(shutdown):
            try:
                if asyncio.iscoroutinefunction(shutdown):
                    await shutdown()
                else:
                    await asyncio.to_thread(shutdown)
            except Exception:
                logger.exception("shutdown() during vacate failed")
        del inst
        server, rec.server = rec.server, None
        if server is not None:
            await asyncio.to_thread(server.stop)
        if torch is not None and torch.cuda.is_available():
            try:
                await asyncio.to_thread(torch.cuda.empty_cache)
            except Exception:
                pass
        for s in rec.specs:
            for b in s.models.values():
                self.store.residency.release_to_disk(wire_ref(b))
        self._on_state_change()

    async def _unload_ref(self, ref: str) -> None:
        """Hub UNLOAD: free the ref's VRAM. Worker-owned pipelines demote to
        the warm RAM tier (instance stays ready; the next LOAD/RunJob promotes
        back in seconds); tenant-loaded refs require record teardown (#371)."""
        async with self._load_lock:
            res = self.store.residency
            if res.tier(ref) is residency_mod.Tier.VRAM:
                if await asyncio.to_thread(res.demote, ref):
                    self._on_state_change()
                    return
            rec = self._record_holding(ref)
            if rec is not None:
                await self._vacate_record(rec)
            else:
                res.release_to_disk(ref)
        self._on_state_change()

    # ---- job intake --------------------------------------------------------

    async def handle_run_job(self, run: pb.RunJob) -> None:
        key = (run.request_id, run.attempt)
        existing = self.jobs.get(key)
        if existing is not None and not existing.superseded:
            if not existing.finished:
                await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
                    request_id=run.request_id, attempt=run.attempt)))
            return
        # Same request, different attempt: abort the old attempt silently.
        for (rid, att), job in list(self.jobs.items()):
            if rid == run.request_id and att != run.attempt and not job.finished:
                job.superseded = True
                if job.ctx is not None:
                    job.ctx._cancel()
                if job.exec_task is not None:
                    job.exec_task.cancel()

        if self.draining:
            await self._send_result(run.request_id, run.attempt, pb.JOB_STATUS_RETRYABLE,
                                    safe_message="worker draining")
            return
        spec = self.specs.get(run.function_name)
        if spec is None:
            await self._send_result(run.request_id, run.attempt, pb.JOB_STATUS_INVALID,
                                    safe_message=f"unknown function {run.function_name!r}")
            return
        if run.function_name in self.unavailable:
            reason, detail, _ = self.unavailable[run.function_name]
            await self._send_result(run.request_id, run.attempt, pb.JOB_STATUS_RETRYABLE,
                                    safe_message=f"function unavailable: {reason}")
            return

        job = _Job(request_id=run.request_id, attempt=run.attempt, spec=spec)
        self.jobs[key] = job
        self._idle.clear()
        logger.info("job admitted %s attempt=%d", run.request_id, run.attempt)
        await self._send(pb.WorkerMessage(job_accepted=pb.JobAccepted(
            request_id=run.request_id, attempt=run.attempt)))
        job.task = asyncio.create_task(self._run_job(job, run), name=f"job-{run.request_id}")

    def handle_cancel(self, cancel: pb.CancelJob) -> None:
        job = self.jobs.get((cancel.request_id, cancel.attempt))
        if job is None or job.finished:
            return  # unknown pair or natural result already stands
        if job.ctx is not None:
            job.ctx._cancel()  # cooperative: sync handlers poll ctx
        if job.exec_task is not None and job.spec is not None and job.spec.is_async:
            job.exec_task.cancel()  # async handlers are cancelled on the loop

    async def wait_idle(self, timeout: Optional[float] = None) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def abort_all(self, safe_message: str = "worker draining") -> None:
        for job in list(self.jobs.values()):
            if job.finished or job.superseded:
                continue
            if job.ctx is not None:
                job.ctx._cancel()
            if job.exec_task is not None:
                job.exec_task.cancel()
            await self._finish(job, pb.JOB_STATUS_RETRYABLE, safe_message=safe_message)

    # ---- job execution -----------------------------------------------------

    async def _run_job(self, job: _Job, run: pb.RunJob) -> None:
        spec = job.spec
        assert spec is not None
        # Pin this job's model refs for its WHOLE lifetime (gw#409): the gap
        # between ensure_setup's promote and the execution-time pin let a
        # concurrent job's make_room demote a just-promoted pipeline. Refs
        # without entries yet are no-ops; the inner pin still covers adapters.
        with self.store.residency.executing(*(wire_ref(b) for b in spec.models.values())):
            await self._run_job_pinned(job, run)

    async def _run_job_pinned(self, job: _Job, run: pb.RunJob) -> None:
        spec = job.spec
        assert spec is not None
        concurrency_at_start = len(self.in_flight_keys()) - 1
        try:
            payload = msgspec.msgpack.decode(run.input_payload, type=spec.payload_type)
        except (msgspec.ValidationError, msgspec.DecodeError) as exc:
            await self._finish(job, pb.JOB_STATUS_INVALID, safe_message=_sanitize(str(exc)))
            return

        snapshots = dict(run.snapshots) if run.snapshots else {}
        compute = run.compute if run.HasField("compute") else None
        needs_gpu = (compute.accelerator == "cuda") if compute is not None else spec.needs_gpu
        gpu_index = int(compute.gpu_index) if compute is not None else 0
        timeout_ms = int(run.timeout_ms or 0) or int(spec.timeout_ms or 0)

        producer = spec.kind != "inference"
        source_info = _reserved_repo_info(payload, "source") if producer else {}
        destination_info = _reserved_repo_info(payload, "destination") if producer else {}

        ctx_cls = _CONTEXT_BY_KIND.get(spec.kind, RequestContext)
        ctx = ctx_cls(
            request_id=run.request_id,
            owner=run.tenant or None,
            invoker_id=run.invoker_id or None,
            timeout_ms=timeout_ms or None,
            file_api_base_url=self.file_base_url or None,
            worker_capability_token=run.capability_token or None,
            compute=Compute(
                accelerator=(compute.accelerator if compute is not None else
                             ("cuda" if spec.needs_gpu else "none")),
                vram_gb=int(compute.vram_gb) if compute is not None else 0,
                gpu_count=int(compute.gpu_count) if compute is not None else 0,
            ),
            models={b.slot: b.ref for b in run.models},
            loras={
                b.slot: tuple(
                    {"ref": ov.ref, "weight": float(ov.weight) or 1.0} for ov in b.loras
                )
                for b in run.models if b.loras
            },
            source_info=source_info,
            destination_info=destination_info,
            execution_hints=(
                {"output_format": "inline"} if run.output_mode == pb.OUTPUT_MODE_INLINE else {}
            ),
            hf_token=getattr(self._settings, "hf_token", "") or "",
        )
        job.ctx = ctx

        try:
            if source_info:
                await self._materialize_source(ctx, source_info, snapshots)
            # Typed media inputs: URL-ref Assets (hub-approved remote media)
            # are downloaded and given a local_path before the handler runs.
            await asyncio.to_thread(materialize_input_assets, payload, run.request_id)
            instance = await self.ensure_setup(spec, snapshots)
            kwargs = await self._handler_kwargs(spec, snapshots)
            adapters = await self._prepare_adapters(run, spec, snapshots)
        except asyncio.CancelledError:
            await self._finish(job, pb.JOB_STATUS_CANCELED, safe_message="canceled")
            return
        except Exception as exc:
            if isinstance(exc, HardwareUnmetError) and not isinstance(exc, InsufficientDiskError):
                # Self-disable the function on this worker; lifecycle emits
                # FnUnavailable and drops it from available_functions.
                # (Disk pressure is transient — GC frees space — so it only
                # fails the job RETRYABLE, never disables the function.)
                axes = exc.axes() if hasattr(exc, "axes") else {}
                self.unavailable[spec.name] = (
                    getattr(exc, "reason", "hardware_unmet"), _sanitize(str(exc)),
                    {str(k): str(v) for k, v in (axes or {}).items()},
                )
                self._on_state_change()
            status, msg = _map_exception(exc)
            logger.exception("setup/injection failed for %s", spec.name)
            await self._finish(job, status, safe_message=msg)
            return

        queue_ms = int((time.monotonic() - job.admitted_at) * 1000)
        lease: Optional[_GpuSlotLease] = None
        started = time.monotonic()
        try:
            if needs_gpu:
                await self._gpu_semaphore.acquire()
                lease = _GpuSlotLease(self._gpu_semaphore, asyncio.get_running_loop())
                ctx._gpu_slot_lease = lease
                if job.ctx.cancelled:
                    raise CanceledError("canceled")
            started = time.monotonic()
            # Pin-while-executing: the models (and adapter snapshots) this job
            # uses are not eviction candidates for its duration.
            exec_refs = [wire_ref(b) for b in spec.models.values()]
            adapter_refs = [a.ref for group in adapters.values() for a in group]
            with self.store.residency.executing(*exec_refs, *adapter_refs):
                active: List[Tuple[str, Any]] = []
                try:
                    for slot, prepared in adapters.items():
                        pipe = self._adapter_target(spec, slot)
                        ref = wire_ref(spec.models[slot])
                        await asyncio.to_thread(
                            self._adapters.activate, ref, pipe, prepared, run.request_id
                        )
                        active.append((ref, pipe))
                    # Explicit activation (gw#399): a slot this request names
                    # no adapters for must run bare even if a previous
                    # request's teardown failed and left adapters enabled.
                    for slot in spec.models:
                        if slot in adapters:
                            continue
                        ref = wire_ref(spec.models[slot])
                        if self._adapters.needs_deactivation(ref):
                            pipe = self.store.residency.obj(ref)
                            if pipe is not None:
                                await asyncio.to_thread(
                                    self._adapters.deactivate, ref, pipe, run.request_id
                                )
                    output = await self._execute(job, spec, instance, ctx, payload, kwargs,
                                                 timeout_ms=timeout_ms, gpu_index=gpu_index)
                finally:
                    # Guaranteed-inactive on every exit (OK / cancel /
                    # deadline / handler error); attachments stay resident.
                    for ref, pipe in active:
                        await asyncio.to_thread(
                            self._adapters.deactivate, ref, pipe, run.request_id
                        )
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index,
                                    output=output)
            # Handler GPU work is done — free the slot before result-blob
            # upload and result send so the next job's compute starts now.
            if lease is not None:
                lease.yield_slot()
            if spec.output_mode == "stream":
                await self._finish(job, pb.JOB_STATUS_OK, metrics=metrics)
            else:
                inline, blob_ref = await self._serialize_output(ctx, run, output)
                await self._finish(job, pb.JOB_STATUS_OK, inline=inline, blob_ref=blob_ref,
                                   metrics=metrics)
        except _DeadlineExceeded:
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index)
            await self._finish(job, pb.JOB_STATUS_FATAL, safe_message="deadline exceeded",
                               metrics=metrics)
        except BaseException as exc:
            status, msg = _map_exception(exc)
            if status == pb.JOB_STATUS_FATAL:
                logger.exception("handler %s failed", spec.name)
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index)
            await self._finish(job, status, safe_message=msg, metrics=metrics)
        finally:
            if lease is not None:
                lease.yield_slot()
            self._maybe_idle()

    async def _materialize_source(
        self, ctx: Any, info: Dict[str, Any], snapshots: Dict[str, pb.Snapshot]
    ) -> None:
        """Reserved-source contract (#376): materialize ``payload.source``
        locally before the handler runs. Same ModelStore path as model
        bindings — identical retry/classification and ModelEvent emission."""
        ref = str(info.get("ref") or "").strip()
        if not ref:
            raise ValidationError("payload.source.ref must be a non-empty repo ref")
        path = await self.store.ensure_local(ref, snapshots.get(ref))
        ctx._set_source_path(str(path))

    async def _handler_kwargs(
        self, spec: EndpointSpec, snapshots: Dict[str, pb.Snapshot]
    ) -> Dict[str, Any]:
        """Per-call model injection: handler parameters (after ctx, payload)
        whose names match model slots receive the local snapshot path."""
        try:
            sig = typing.get_type_hints(spec.method)
        except Exception:
            sig = {}
        import inspect as _inspect

        params = [
            p.name for p in _inspect.signature(spec.method).parameters.values()
            if p.name != "self"
        ][2:]
        setup_slots = set(self._setup_slots(spec))
        kwargs: Dict[str, Any] = {}
        for name in params:
            binding = spec.models.get(name)
            if binding is None or name in setup_slots:
                continue
            ref = wire_ref(binding)
            path = await self.store.ensure_local(ref, snapshots.get(ref), binding=binding)
            kwargs[name] = Path(path) if sig.get(name) is Path else str(path)
        return kwargs

    async def _prepare_adapters(
        self, run: pb.RunJob, spec: EndpointSpec, snapshots: Dict[str, pb.Snapshot]
    ) -> Dict[str, List[lora_util.PreparedAdapter]]:
        """Materialize + parse the job's per-slot LoRA overlays (gw#393).

        Downloads ride the normal ensure_local snapshot path (disk GC,
        ref-index, ModelEvents — so the hub learns adapter download bandwidth
        like any ref); parsed state dicts hit the digest-keyed RAM LRU.
        GPU-free: application happens later, under the job's GPU slot."""
        overlays = [(b.slot, list(b.loras)) for b in run.models if b.loras]
        if not overlays:
            return {}
        total = sum(len(loras) for _, loras in overlays)
        if total > lora_util.MAX_LORAS_PER_REQUEST:
            raise ValidationError(
                f"too many lora adapters: {total} "
                f"(max {lora_util.MAX_LORAS_PER_REQUEST} per request)"
            )
        out: Dict[str, List[lora_util.PreparedAdapter]] = {}
        for slot, loras in overlays:
            if slot not in spec.models:
                raise ValidationError(f"lora overlay names unknown model slot {slot!r}")
            prepared: List[lora_util.PreparedAdapter] = []
            for overlay in loras:
                ref = str(overlay.ref or "").strip()
                if not ref:
                    raise ValidationError(f"lora overlay on slot {slot!r} has an empty ref")
                weight = lora_util.validate_overlay_weight(overlay.weight, ref=ref)
                t0 = time.monotonic()
                path = await self.store.ensure_local(ref, snapshots.get(ref))
                ensure_ms = int((time.monotonic() - t0) * 1000)
                snap = snapshots.get(ref)
                digest = (snap.digest if snap is not None else "") or path.name
                cache_key = f"{ref}@{digest}"
                t1 = time.monotonic()
                state_dict = self._adapter_cache.get(cache_key)
                from_cache = state_dict is not None
                if state_dict is None:
                    file = lora_util.find_adapter_file(path, ref=ref)
                    state_dict = await asyncio.to_thread(
                        lora_util.load_adapter_state_dict, file, ref=ref
                    )
                    self._adapter_cache.put(cache_key, state_dict)
                parse_ms = int((time.monotonic() - t1) * 1000)
                prepared.append(lora_util.PreparedAdapter(
                    slot=slot, ref=ref, cache_key=cache_key,
                    name=lora_util.adapter_name(cache_key),
                    weight=weight, state_dict=state_dict,
                    from_cache=from_cache, ensure_ms=ensure_ms, parse_ms=parse_ms,
                ))
            out[slot] = prepared
        return out

    def _adapter_target(self, spec: EndpointSpec, slot: str) -> Any:
        """The worker-managed pipeline object adapters for ``slot`` apply to."""
        pipe = self.store.residency.obj(wire_ref(spec.models[slot]))
        if pipe is None:
            raise ValidationError(
                f"model slot {slot!r} has no worker-managed pipeline; "
                "lora overlays require a pipeline-injected setup slot"
            )
        return pipe

    async def _execute(
        self,
        job: _Job,
        spec: EndpointSpec,
        instance: Any,
        ctx: RequestContext,
        payload: Any,
        kwargs: Dict[str, Any],
        *,
        timeout_ms: int,
        gpu_index: int,
    ) -> Any:
        bound = spec.method if instance is None else getattr(instance, spec.attr_name)
        call_kwargs = {spec.ctx_param: ctx, spec.payload_param: payload, **kwargs}
        timeout_s = (timeout_ms / 1000.0) if timeout_ms > 0 else None

        loop = asyncio.get_running_loop()
        if spec.is_async_gen:
            coro = self._pump_async_gen(job, bound(**call_kwargs))
        elif spec.is_async:
            coro = bound(**call_kwargs)
        elif spec.output_mode == "stream":
            coro = asyncio.to_thread(self._pump_sync_gen, job, bound, call_kwargs, gpu_index, loop)
        else:
            coro = asyncio.to_thread(self._call_sync, bound, call_kwargs, gpu_index)

        job.exec_task = asyncio.ensure_future(coro)
        try:
            return await asyncio.wait_for(asyncio.shield(job.exec_task), timeout_s)
        except asyncio.TimeoutError:
            ctx._cancel()
            job.exec_task.cancel()
            if not spec.is_async:
                self._reap_stuck_thread(job)
            raise _DeadlineExceeded()
        except asyncio.CancelledError:
            # CancelJob path: the exec task was cancelled underneath us.
            raise CanceledError("canceled")

    @staticmethod
    def _call_sync(bound: Callable[..., Any], call_kwargs: Dict[str, Any], gpu_index: int) -> Any:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                pass
        return bound(**call_kwargs)

    def _reap_stuck_thread(self, job: _Job) -> None:
        """Deadline fired but the sync handler thread may not die. If it's
        still running after the recycle grace, exit so the pod is recycled."""

        async def _watch() -> None:
            assert job.exec_task is not None
            try:
                await asyncio.wait_for(asyncio.shield(job.exec_task), _STUCK_THREAD_RECYCLE_S)
            except asyncio.TimeoutError:
                logger.critical(
                    "handler thread for %s ignored deadline+cancel for %.0fs; "
                    "recycling worker process", job.request_id, _STUCK_THREAD_RECYCLE_S,
                )
                os._exit(70)
            except BaseException:
                pass  # thread finished (with error) — no recycle needed

        asyncio.create_task(_watch(), name=f"reap-{job.request_id}")

    # ---- streaming ---------------------------------------------------------

    def _encode_chunk(self, item: Any) -> Optional[Tuple[bytes, str]]:
        if isinstance(item, Done):
            return None
        if isinstance(item, Error):
            raise ValidationError(getattr(item, "message", "") or "stream error")
        if isinstance(item, IncrementalTokenDelta):
            return item.text.encode("utf-8"), "text/plain"
        if isinstance(item, BatchItemDelta):
            # First-class multi-item delta: msgpack keeps `chunk` binary.
            return msgspec.msgpack.encode(item), "application/x-batch-item+msgpack"
        return msgspec.json.encode(item), "application/json"

    async def _emit_progress(self, job: _Job, seq: int, data: bytes, content_type: str) -> None:
        await self._send(pb.WorkerMessage(job_progress=pb.JobProgress(
            request_id=job.request_id, attempt=job.attempt, seq=seq,
            data=data, content_type=content_type)))

    async def _pump_async_gen(self, job: _Job, agen: Any) -> None:
        seq = 0
        async for item in agen:
            if job.ctx is not None:
                job.ctx.raise_if_cancelled()
            enc = self._encode_chunk(item)
            if enc is None:
                break
            seq += 1
            await self._emit_progress(job, seq, enc[0], enc[1])

    def _pump_sync_gen(
        self,
        job: _Job,
        bound: Callable[..., Any],
        call_kwargs: Dict[str, Any],
        gpu_index: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_device(gpu_index)
            except Exception:
                pass
        seq = 0
        for item in bound(**call_kwargs):
            if job.ctx is not None:
                job.ctx.raise_if_cancelled()
            enc = self._encode_chunk(item)
            if enc is None:
                break
            seq += 1
            fut = asyncio.run_coroutine_threadsafe(
                self._emit_progress(job, seq, enc[0], enc[1]), loop
            )
            fut.result()  # backpressure: block the producer on queue overflow

    # ---- results -----------------------------------------------------------

    async def _serialize_output(
        self, ctx: RequestContext, run: pb.RunJob, output: Any
    ) -> Tuple[Optional[bytes], Optional[str]]:
        data = msgspec.msgpack.encode(output)
        if len(data) <= INLINE_RESULT_MAX_BYTES:
            return data, None
        try:
            asset = await asyncio.to_thread(
                ctx.save_bytes, f"results/{run.request_id}.msgpack", data
            )
            ref = getattr(asset, "ref", "") or ""
            if not ref:
                raise RuntimeError("upload returned no ref")
            return None, ref
        except Exception as exc:
            logger.warning("result blob upload failed for %s: %s", run.request_id, exc)
            raise RetryableError("output upload failed") from exc

    def _metrics(
        self, queue_ms: int, started: float, concurrency_at_start: int, gpu_index: int,
        output: Any = None,
    ) -> pb.JobMetrics:
        runtime_ms = int((time.monotonic() - started) * 1000)
        peak_rss = 0
        try:
            import psutil

            peak_rss = int(psutil.Process().memory_info().rss)
        except Exception:
            pass
        peak_vram = 0
        if torch is not None and torch.cuda.is_available():
            try:
                peak_vram = int(torch.cuda.max_memory_allocated(gpu_index))
            except Exception:
                pass
        return pb.JobMetrics(
            runtime_ms=runtime_ms, queue_ms=queue_ms, peak_rss_bytes=peak_rss,
            peak_vram_bytes=peak_vram, concurrency_at_start=max(0, concurrency_at_start),
            output_media_duration_s=_output_media_seconds(output),
        )

    async def _send_result(
        self,
        request_id: str,
        attempt: int,
        status: int,
        *,
        inline: Optional[bytes] = None,
        blob_ref: Optional[str] = None,
        safe_message: str = "",
        metrics: Optional[pb.JobMetrics] = None,
    ) -> None:
        result = pb.JobResult(request_id=request_id, attempt=attempt, status=status,
                              safe_message=safe_message)
        if inline is not None:
            result.inline = inline
        elif blob_ref:
            result.blob_ref = blob_ref
        if metrics is not None:
            result.metrics.CopyFrom(metrics)
        await self._send(pb.WorkerMessage(job_result=result))

    async def _finish(self, job: _Job, status: int, **kw: Any) -> None:
        if job.finished:
            return
        job.finished = True
        cleanup_input_assets(job.request_id)
        logger.info("job finished %s attempt=%d status=%s", job.request_id, job.attempt, status)
        if not job.superseded:
            await self._send_result(job.request_id, job.attempt, status, **kw)
        # Keep finished records so a RunJob retransmission doesn't re-execute;
        # prune oldest finished entries beyond a small window.
        finished = [k for k, j in self.jobs.items() if j.finished]
        if len(finished) > 1024:
            for k in finished[: len(finished) - 1024]:
                self.jobs.pop(k, None)
        self._maybe_idle()

    def _maybe_idle(self) -> None:
        if not self.in_flight_keys():
            self._idle.set()


class _DeadlineExceeded(Exception):
    pass
