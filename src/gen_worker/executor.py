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
from .api.types import Compute
from .capability import HardwareUnmetError
from .models import residency as residency_mod
from .models.cache_paths import tensorhub_cas_dir
from .models.download import ensure_local
from .models.residency import Residency
from .pb import worker_scheduler_pb2 as pb
from .registry import EndpointSpec
from .request_context import (
    ConversionContext,
    DatasetContext,
    RequestContext,
    TrainingContext,
)

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

try:  # torch is optional at import time; the executor works without it.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _sanitize(message: str) -> str:
    out = str(message or "").strip()
    for needle in ("Bearer ", "X-Amz-", "Signature="):
        if needle in out:
            return "internal error"
    return out[:1024]


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
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return pb.JOB_STATUS_RETRYABLE, "out of memory"
    return pb.JOB_STATUS_FATAL, "internal error"


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


def _is_terminal_download_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
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
    """The worker's model seam: ensure-local with retries + the residency map.
    All tier transitions flow through :class:`~gen_worker.models.residency.
    Residency`, whose events this store forwards as wire ``ModelEvent``s."""

    def __init__(
        self,
        emit: Callable[[pb.WorkerMessage], Awaitable[None]],
        *,
        hf_home: str = "",
        hf_token: str = "",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._emit = emit
        self._hf_home = hf_home or None
        self._hf_token = hf_token or None
        self._cache_dir = cache_dir or tensorhub_cas_dir()
        self.residency = Residency(on_event=self._on_residency_event)
        self._locks: Dict[str, asyncio.Lock] = {}
        self.keep: set[str] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ---- events ------------------------------------------------------------

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

    def mark_in_vram(self, ref: str, vram_bytes: int, obj: Any = None) -> None:
        self.residency.track_vram(ref, obj, vram_bytes=int(vram_bytes))

    def mark_in_ram(self, ref: str, obj: Any = None) -> None:
        self.residency.track_ram(ref, obj)

    def mark_unloaded(self, ref: str) -> None:
        self.residency.release_to_disk(ref)

    # ---- ensure-local ----------------------------------------------------------

    def _lock(self, ref: str) -> asyncio.Lock:
        return self._locks.setdefault(ref, asyncio.Lock())

    async def ensure_local(
        self,
        ref: str,
        snapshot: Optional[pb.Snapshot] = None,
        *,
        binding: Any = None,
    ) -> Path:
        """Materialize `ref` on disk. Transient failures retry with backoff;
        terminal (4xx-class) failures raise immediately. Emits ModelEvents.
        ``binding`` (when known) supplies provider + file-selection metadata."""
        async with self._lock(ref):
            cached = self.residency.local_path(ref)
            if cached is not None and cached.exists():
                return cached
            self._loop = asyncio.get_running_loop()
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

    @staticmethod
    def _error_vocab(exc: BaseException) -> str:
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
        self._gpu_semaphore = asyncio.Semaphore(max(1, gpu_slots))
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
            if r.vram_gb is not None and total_vram_gb and total_vram_gb < float(r.vram_gb):
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
        rec = self._classes[spec.instance_key]
        async with rec.lock:
            if rec.ready:
                return rec.instance
            setup_slots = self._setup_slots(spec)
            paths: Dict[str, str] = {}
            for slot in setup_slots:
                binding = spec.models[slot]
                ref = wire_ref(binding)
                snap = (snapshots or {}).get(ref)
                path = await self.store.ensure_local(ref, snap, binding=binding)
                paths[slot] = str(path)
            instance = spec.cls()
            setup = getattr(instance, "setup", None)
            vram_before = self._vram_allocated()
            if spec.runtime:
                rec.server = await self._boot_engine_server(spec, paths)
            if callable(setup):
                kwargs = await self._injection_kwargs(spec, setup, paths, server=rec.server)
                if asyncio.iscoroutinefunction(setup):
                    await setup(**kwargs)
                else:
                    await asyncio.to_thread(setup, **kwargs)
            warmup = getattr(instance, "warmup", None)
            if callable(warmup):
                if asyncio.iscoroutinefunction(warmup):
                    await warmup()
                else:
                    await asyncio.to_thread(warmup)
            # Measured VRAM (allocator delta across the load) — the number the
            # orchestrator's VRAM packer sees; never an estimate constant.
            vram_delta = max(0, self._vram_allocated() - vram_before)
            refs = [wire_ref(spec.models[s]) for s in setup_slots]
            for ref in refs:
                if vram_delta > 0:
                    self.store.mark_in_vram(ref, vram_delta if len(refs) == 1 else 0)
                else:
                    self.store.mark_in_ram(ref)
            rec.instance = instance
            rec.ready = True
            self._on_state_change()
            return instance

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

    async def _injection_kwargs(
        self,
        spec: EndpointSpec,
        setup: Callable[..., Any],
        paths: Dict[str, str],
        *,
        server: Any = None,
    ) -> Dict[str, Any]:
        """Typed injection: each slot receives exactly what its ``setup``
        annotation says — a ``str``/``Path`` local path, or a constructed
        pipeline for a class annotation exposing ``from_pretrained`` (built off
        the loop; the binding dtype is honored and the worker applies its
        placement/offload policy to the result). A parameter annotated
        ``ServerHandle`` receives the booted engine server."""
        from .runtimes.server import ServerHandle

        try:
            hints = typing.get_type_hints(setup)
        except Exception:
            hints = {}
        kwargs: Dict[str, Any] = {}
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
                pipe = await asyncio.to_thread(
                    load_from_pretrained, ann, path, dtype=dtype
                )
                # Worker-owned placement/offload policy: one decider for the
                # whole worker; endpoints never write device/offload code.
                await asyncio.to_thread(place_pipeline, pipe)
                kwargs[slot] = pipe
            else:
                kwargs[slot] = path
        return kwargs

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

    def _ref_in_use(self, ref: str) -> bool:
        for job in self.jobs.values():
            if job.finished or job.superseded or job.spec is None:
                continue
            if ref in (wire_ref(b) for b in job.spec.models.values()):
                return True
        return False

    async def _unload_ref(self, ref: str) -> None:
        """Demote a ref out of VRAM by tearing down the classes that hold it."""
        for rec in self._classes.values():
            holds = any(
                ref in (wire_ref(b) for b in s.models.values()) for s in rec.specs
            )
            if holds and rec.ready:
                inst, rec.instance, rec.ready = rec.instance, None, False
                shutdown = getattr(inst, "shutdown", None)
                if callable(shutdown):
                    try:
                        await asyncio.to_thread(shutdown)
                    except Exception:
                        logger.exception("shutdown() during UNLOAD failed")
                del inst
                server, rec.server = rec.server, None
                if server is not None:
                    await asyncio.to_thread(server.stop)
        if torch is not None and torch.cuda.is_available():
            try:
                await asyncio.to_thread(torch.cuda.empty_cache)
            except Exception:
                pass
        self.store.mark_unloaded(ref)
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
            execution_hints=(
                {"output_format": "inline"} if run.output_mode == pb.OUTPUT_MODE_INLINE else {}
            ),
            hf_token=getattr(self._settings, "hf_token", "") or "",
        )
        job.ctx = ctx

        try:
            instance = await self.ensure_setup(spec, snapshots)
            kwargs = await self._handler_kwargs(spec, snapshots)
        except asyncio.CancelledError:
            await self._finish(job, pb.JOB_STATUS_CANCELED, safe_message="canceled")
            return
        except Exception as exc:
            if isinstance(exc, HardwareUnmetError):
                # Self-disable the function on this worker; lifecycle emits
                # FnUnavailable and drops it from available_functions.
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
        acquired = False
        started = time.monotonic()
        try:
            if needs_gpu:
                await self._gpu_semaphore.acquire()
                acquired = True
                if job.ctx.cancelled:
                    raise CanceledError("canceled")
            started = time.monotonic()
            # Pin-while-executing: the models this job uses are not eviction
            # candidates for its duration (cross-pipeline eviction safety).
            exec_refs = [wire_ref(b) for b in spec.models.values()]
            with self.store.residency.executing(*exec_refs):
                output = await self._execute(job, spec, instance, ctx, payload, kwargs,
                                             timeout_ms=timeout_ms, gpu_index=gpu_index)
            metrics = self._metrics(queue_ms, started, concurrency_at_start, gpu_index)
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
            if acquired:
                self._gpu_semaphore.release()
            self._maybe_idle()

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
        self, queue_ms: int, started: float, concurrency_at_start: int, gpu_index: int
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
