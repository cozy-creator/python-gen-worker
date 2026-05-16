from __future__ import annotations

import grpc
import logging
import sys
import time
import json
import re
import urllib.request
import urllib.parse
import urllib.error
import random
import threading
import os
import signal
import traceback
import queue
import psutil
import importlib
import inspect
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import collections.abc as cabc
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Iterator, List, Tuple, Iterable, get_args, get_origin
import hashlib
import msgspec
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
import asyncio

import jwt
from ._worker_auth import _JWKSCache
from .config import Settings


def _normalize_grpc_addr(addr: str) -> tuple[str, bool]:
    """Normalize a scheduler address into (host:port, use_tls).

    Accepts bare `host:port`, or any of `grpc://`, `grpcs://`, `http://`,
    `https://` prefixes. Trailing `:443` implies TLS even without scheme.
    """
    a = (addr or "").strip()
    if not a:
        return "", False
    lower = a.lower()
    if lower.startswith("grpcs://"):
        return a[len("grpcs://"):].strip(), True
    if lower.startswith("grpc://"):
        return a[len("grpc://"):].strip(), False
    if lower.startswith("https://"):
        return a[len("https://"):].strip(), True
    if lower.startswith("http://"):
        return a[len("http://"):].strip(), False
    tls = a.endswith(":443")
    return a, tls
from ._worker_support import (
    _AuthInterceptor,
    _BatchedWorkerSpec,
    _ConversionWorkerSpec,
    _RequestSpec,
    _SerialWorkerSpec,
    _extract_checkpoint_id_from_result,
    _extract_resolved_compute,
    _extract_worker_capability_token,
    _parse_manifest_model_mapping,
    _workspace_scope_id,
    build_provider_index_from_manifest,
)
from .request_context import (
    RequestContext,
    _canonicalize_model_ref_string,
    _decode_unverified_jwt_claims,
    _encode_ref_for_url,
    _http_request,
    _infer_mime_type,
    _normalize_output_ref,
    _url_is_blocked,
)
# Use relative imports within the package
from .pb import worker_scheduler_pb2 as _pb
from .pb import worker_scheduler_pb2_grpc as _pb_grpc

pb: Any = _pb
pb_grpc: Any = _pb_grpc

WorkerSchedulerMessage = Any
WorkerEvent = Any
WorkerResources = Any
WorkerRegistration = Any
LoadModelCommand = Any
LoadModelResult = Any
UnloadModelResult = Any
JobExecutionRequest = Any
JobExecutionResult = Any
from .api.binding import Binding, CivitaiRepo, Dispatch, HFRepo, Repo
from .api.decorators import Resources
from .api.errors import (
    AuthError,
    CanceledError,
    FatalError,
    InputTooLargeError,
    OutputTooLargeError,
    RefCompatibilitySurprise,
    ResourceError,
    RetryableError,
    ValidationError,
)
from .capability import HardwareUnmetError

from .models.interface import ModelManagementInterface
from .models.downloader import ModelDownloader
from .models.ref_downloader import (
    ModelRefDownloader,
    lookup_provider_for_ref,
    reset_provider_by_ref,
    set_provider_by_ref,
)
from .models.refs import parse_model_ref
from .api.types import Asset, Tensors
from .models.cache import ModelCache
from .run_metrics_v1 import RunMetricsV1, best_effort_bytes_downloaded, best_effort_init_model_metrics, safe_json_bytes
from .models.cache_paths import tensorhub_cas_dir
from .wire_protocol import WIRE_PROTOCOL_MAJOR, WIRE_PROTOCOL_MINOR, wire_protocol_version_string
from dataclasses import dataclass as _injection_dataclass


async def _single_item_async_iter(value: Any):
    """Wrap a single value as a one-shot async iterator.

    Used by the BatchedWorker dispatch path when the tenant method
    returns a single awaitable (an `async def` that returns one delta
    instead of yielding many). Lets us share the same async-for emission
    loop for both shapes.
    """
    yield value


def _type_qualname(t: Any) -> str:
    """Return a class's fully-qualified ``module.qualname`` string."""
    mod = getattr(t, "__module__", "") or ""
    qn = getattr(t, "__qualname__", None) or getattr(t, "__name__", "") or ""
    if mod and qn:
        return f"{mod}.{qn}"
    return repr(t)


@_injection_dataclass(frozen=True)
class InjectionSpec:
    """Per-parameter binding spec attached to a discovered function.

    Replaces the 0.6.x ``InjectionSpec(param_name, param_type, model_ref)``
    shape — ``binding`` is now a :class:`Repo` or :class:`Dispatch` value
    from ``@inference(models={...})``.
    """

    param_name: str
    param_type: Any
    binding: Binding


def _wire_ref(repo: Repo) -> str:
    """Canonical wire ref for a Repo binding.

    The wire format is bare ref + explicit ``provider`` field — never
    prefixed. Caller emits ``provider`` alongside ``ref`` (see
    :func:`_binding_to_wire`). The implicit default on the consumer
    side is ``"tensorhub"`` when the provider field is absent on a payload.
    """
    return repo.ref


def _binding_to_wire(param_name: str, param_type: Any, binding: Binding) -> Dict[str, Any]:
    """Serialize an InjectionSpec to the wire shape consumed by the
    orchestrator's release resolver.
    """
    # Provider-aware match: Repo (cozy/tensorhub), HFRepo (huggingface),
    # CivitaiRepo (civitai). HFRepo/CivitaiRepo are Repo subclasses today,
    # so plain Repo would already match — the tuple form makes the typed
    # provider surface explicit at the call site (issue #10).
    if isinstance(binding, (Repo, HFRepo, CivitaiRepo)):
        out: Dict[str, Any] = {
            "kind": "fixed",
            "ref": _wire_ref(binding),
            "flavor": binding._flavor,
            "tag": binding._tag,
            "allow_override": binding._allow_override,
            "pipeline_classes": list(binding._pipeline_classes),
        }
        # Tensorhub is the implicit default — omit `provider` when it
        # would be tensorhub so the wire stays clean. Non-tensorhub
        # providers MUST emit the field explicitly.
        if binding.provider != "tensorhub":
            out["provider"] = binding.provider
        return {"param": param_name, "type": _type_qualname(param_type), "binding": out}
    if isinstance(binding, Dispatch):
        table: Dict[str, Dict[str, Any]] = {}
        for k, repo in binding.table.items():
            entry: Dict[str, Any] = {
                "ref": _wire_ref(repo),
                "flavor": repo._flavor,
                "tag": repo._tag,
            }
            if repo.provider != "tensorhub":
                entry["provider"] = repo.provider
            table[k] = entry
        return {
            "param": param_name,
            "type": _type_qualname(param_type),
            "binding": {
                "kind": "dispatch",
                "field": binding.field,
                "table": table,
                "allow_override": binding._allow_override,
                "pipeline_classes": list(binding._pipeline_classes),
            },
        }
    raise TypeError(f"unknown binding type: {type(binding).__name__}")


def _resolved_repo_id(ref: str, flavor: str = "", tag: str = "prod", provider: str = "tensorhub") -> str:
    """Construct a deterministic canonical identity for ``(provider, ref, tag, flavor)``.

    Used as the local cache key for orchestrator-stamped resolved checkpoints.
    The wire format carries ``provider`` as a typed field; for in-process
    cache-key purposes we serialize it with a ``provider::`` prefix (double
    colon, intentionally distinct from the legacy ``provider:`` single-colon
    wire prefix that no longer exists). ``cozy`` is the implicit default and
    is elided so existing cozy keys round-trip unchanged.
    """
    base = (ref or "").strip()
    if not base:
        return ""
    out = base
    if tag and tag != "prod":
        out = f"{out}:{tag}"
    if flavor:
        out = f"{out}#{flavor}"
    if provider and provider != "tensorhub":
        out = f"{provider}::{out}"
    return out


from .discovery.names import slugify_name
from .discovery.walk import find_endpoint_classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# Type variables for generic function signatures
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

HEARTBEAT_INTERVAL = 10  # seconds

_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024


class Worker:
    """Worker implementation that connects to the scheduler via gRPC."""

    def __init__(
        self,
        *,
        settings: Settings,
        user_module_names: List[str] = ["functions"],
        reconnect_delay: float = 5,
        max_reconnect_attempts: int = 0,  # 0 means infinite retries
        lb_only_retries: bool = False,
        model_manager: Optional[ModelManagementInterface] = None,
        downloader: Optional[ModelDownloader] = None,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a new worker.

        Args:
            settings: The canonical worker `Settings` loaded once at process
                start (`gen_worker.config.load_settings`). Provides
                orchestrator_public_addr, worker_id, worker_jwt,
                tensorhub_public_url, hf_token, hf_home, runpod_pod_id.
            user_module_names: List of Python module names containing
                user-defined @inference functions.
            reconnect_delay: Seconds to wait between reconnection attempts.
            max_reconnect_attempts: Max reconnect attempts (0 = infinite).
            model_manager: Optional model manager.
            downloader: Optional model downloader.
            manifest: Optional manifest dict (baked in at build time)
                containing models, resources, etc.
        """
        self._settings = settings
        scheduler_addr, use_tls = _normalize_grpc_addr(settings.orchestrator_public_addr)
        if not scheduler_addr:
            raise RuntimeError(
                "Settings.orchestrator_public_addr is required to start the worker"
            )
        self.scheduler_addr = scheduler_addr
        self.scheduler_addrs = self._normalize_scheduler_addrs(scheduler_addr, None)
        self._process_started_monotonic = time.monotonic()
        self._registered_event = threading.Event()
        self._registration_watchdog_thread: Optional[threading.Thread] = None
        self._startup_timeout_triggered = False
        self._register_timeout_s = 90
        self._warn_model_resolve_s = 30.0
        self._warn_model_load_s = 60.0
        self._warn_inference_s = 60.0
        self.user_module_names = user_module_names
        self.worker_jwt = settings.worker_jwt.strip()
        self._worker_claims = _decode_unverified_jwt_claims(self.worker_jwt) if self.worker_jwt else {}
        jwt_worker_id = str(self._worker_claims.get("sub") or "").strip()
        self.worker_id = settings.worker_id or jwt_worker_id or f"py-worker-{os.getpid()}"
        self.use_tls = use_tls
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.max_input_bytes = 0
        self.max_output_bytes = 0

        self._jwks_url = ""
        self._jwks_ttl_seconds = 300
        self._jwt_issuer = ""
        self._jwt_audience = ""
        self._jwks_cache: Optional[_JWKSCache] = None

        # Worker containers are treated as untrusted. Do not depend on
        # RELEASE_ID/OWNER env vars. Release + owner identity come from the
        # scheduler-issued JWT and per-job gRPC envelopes.
        self.release_id = str(self._worker_claims.get("release_id") or "").strip()
        self.owner = ""
        self.runpod_pod_id = settings.runpod_pod_id
        if not self.runpod_pod_id:
            logger.warning("Settings.runpod_pod_id is empty for this worker!")

        logger.info(f"RUNPOD_POD_ID: {self.runpod_pod_id}")

        self._request_specs: Dict[str, _RequestSpec] = {}
        # Transform-kind (@conversion) handlers, keyed by function
        # name. Dispatch shape is (request_context, payload_dict) → list[ProducedFlavor];
        # see gen_worker/conversion/dispatch.py for the contract.
        self._training_specs: Dict[str, Callable[..., Any]] = {}
        # #273: BatchedWorker (`@inference(runtime="sglang"|"vllm")`) handlers,
        # keyed by externally-addressable function name. Each spec carries the
        # singleton class instance + bound async method + payload/delta types.
        # The class instance lives once per worker process; setup(**models[, engine=...])
        # runs lazily on first dispatch via _start_one_batched_instance.
        self._batched_specs: Dict[str, _BatchedWorkerSpec] = {}
        # Map class-instance id -> (instance, engine, runtime) so drain can call
        # shutdown() exactly once per class even when multiple @inference.function
        # methods route through the same instance.
        self._batched_instances: List[Dict[str, Any]] = []
        # Per-request engine submission tracking. function_name -> request_id ->
        # asyncio task. Used by _handle_interrupt_request to call engine.abort().
        self._batched_inflight_lock = threading.Lock()
        self._batched_inflight: Dict[str, Dict[str, Any]] = {}
        # Single asyncio event loop hosting all BatchedWorker dispatch +
        # engine I/O. Started lazily in _ensure_batched_loop; runs on a
        # dedicated thread so the synchronous receive loop / job dispatch
        # thread can submit coroutines via run_coroutine_threadsafe.
        self._batched_loop: Optional[asyncio.AbstractEventLoop] = None
        self._batched_loop_thread: Optional[threading.Thread] = None
        # #322/#328: SerialWorker (`@inference` sync class) dispatch table.
        # Mirrors `_batched_specs` but for the synchronous archetype — one
        # request fully owns the GPU end-to-end, no engine, no asyncio loop.
        # The class instance is constructed once at discovery; `setup(**models)`
        # runs lazily on first dispatch (under a per-class lock) so we pay
        # the cold-load cost on the first job rather than at process boot.
        self._serial_class_specs: Dict[str, _SerialWorkerSpec] = {}
        # Per-class record: {cls_name, instance, endpoint_spec, started, started_lock}.
        # Mirrors `_batched_instances`; the drain path calls `instance.shutdown()`
        # exactly once per record.
        self._serial_class_instances: List[Dict[str, Any]] = []
        # #332: sync class endpoints for the conversion / training / dataset
        # kinds. Structurally identical to SerialWorker (sync setup → generate
        # → shutdown) but the result emission path uploads ProducedFlavors and
        # publishes a revision via `_finalize_produced_variants` rather than
        # returning an msgspec.Struct back to the caller.
        self._conversion_class_specs: Dict[str, _ConversionWorkerSpec] = {}
        # Per-class record for the conversion-kind classes; mirrors
        # `_serial_class_instances`. Reuses `_ensure_serial_class_started` /
        # `_find_serial_record` by appending to the same list, since the
        # lazy-setup contract is identical.
        # #324: per-function cross-request micro-batching aggregator. Populated
        # in `_register_endpoint_class_serial` when both `batch_window_ms` and
        # `max_batch` are declared on the @inference class. Auto-disabled at
        # setup-time when any attached cache wrapper has
        # `breaks_cross_request_batching=True` (e.g. TeaCache, nunchaku #597).
        # The serial dispatch wrapper inspects this map to route through the
        # aggregator vs invoke the bound method directly. Value is a
        # ``gen_worker.api.micro_batch.MicroBatchAggregator``; typed Any to
        # avoid pulling the lazy asyncio init into module load order.
        self._micro_batch_aggregators: Dict[str, Any] = {}
        self._active_requests: Dict[str, RequestContext] = {}
        self._active_requests_lock = threading.Lock()
        self._request_observations: Dict[str, Dict[str, Any]] = {}
        # request_id -> monotonic time when _handle_job_request first logged it.
        # Used by _outgoing_message_iterator to emit a [worker-timing] line
        # showing recv→handler_done→yielded deltas for latency profiling.
        # Popped on yield.
        self._request_recv_times: Dict[str, float] = {}
        self._request_handler_done_times: Dict[str, float] = {}
        # #321: batch context state removed alongside BatchExecutionRequest.
        self._drain_timeout_seconds = 0
        self._draining = False
        # Emit model.ready immediately on connect by default; downstream model-load
        # events still apply for workers that need to gate on GPU model pre-load.
        self._models_ready_on_connect: bool = False
        self._discovered_resources: Dict[str, Resources] = {} # Store resources per function
        self._function_schemas: Dict[str, Tuple[bytes, bytes, Optional[bytes], bytes]] = {}  # func_name -> (input_schema_json, output_schema_json, delta_schema_json, injection_json)
        # #321: runtime batching state removed alongside RuntimeBatchingConfigCommand.
        self._last_function_capabilities_hash = ""

        self._custom_runtime_cache: Dict[Tuple[str, str], Any] = {}  # (model_id, injected_type_qualname) -> runtime handle
        self._custom_runtime_locks: Dict[Tuple[str, str], threading.Lock] = {}

        # Local (non-NFS) cache for NFS->local snapshot localization.
        # Empty string disables localization entirely.
        self._local_model_cache_dir = ""
        self._local_model_cache: Optional[Any] = None
        self._local_model_cache_lock = threading.Lock()
        self._last_disk_inventory_hash: str = ""

        self._gpu_busy_lock = threading.Lock()
        # Busy is used by the scheduler as a cheap routing hint.
        # Use a refcount (not a boolean) so overlapping inference + model ops
        # cannot flip BUSY -> NOT BUSY prematurely.
        self._gpu_busy_refcount = 0
        # Detect GPU availability + count once at startup. We need both flags:
        # `_has_gpu` gates the status-reporting refcount; `_gpu_count` sizes
        # the BoundedSemaphore that actually serializes handler dispatch.
        self._has_gpu = False
        self._gpu_count = 0
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_count = int(torch.cuda.device_count() or 0)
                self._has_gpu = self._gpu_count > 0
        except ImportError:
            pass

        # GPU exclusivity primitive — one slot per physical GPU. SerialWorker /
        # Conversion / Training handlers acquire before running their handler
        # method; prefetched jobs block here until the current handler releases.
        # Result: sub-millisecond hot-handoff between sequential jobs on the
        # same GPU.
        # CRITICAL: only acquired by handlers whose function declares
        # accelerator=cuda*. For accelerator=none endpoints (marco-polo style),
        # the acquire is skipped entirely so trivial CPU endpoints scale to
        # regular-Python-webserver throughput (~5-10K req/sec per worker).
        # Sized to max(1, _gpu_count) so even worker-no-gpu boots get a real
        # primitive (handlers on a no-gpu worker that nevertheless declare
        # accelerator=cuda would block as if on a 1-GPU host — not a real
        # production scenario but a safe fallback).
        _sema_slots = max(1, int(self._gpu_count or 0) or 1)
        self._gpu_semaphore = threading.BoundedSemaphore(_sema_slots)

        # Track models currently in use by in-flight runs, so we can refuse
        # UnloadModelCommand that would thrash/kill active inference.
        self._active_model_use_lock = threading.Lock()
        self._active_model_use_counts: Dict[str, int] = {}

        self._channel: Optional[Any] = None
        self._stub: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._running = False
        self._stop_event = threading.Event()
        self._reconnect_count = 0
        self._outgoing_queue: queue.Queue[Any] = queue.Queue()
        self._leader_hint: Optional[str] = None
        # Cap redirect bounces. When a worker connects to the wrong
        # gen-orchestrator replica the orch responds with FAILED_PRECONDITION
        # not_leader:<addr> and closes the stream. The worker reconnects to the
        # advertised addr. If the redirect chain doesn't terminate within this
        # many hops something is wrong (stale lease addr, network partition,
        # misconfigured ORCHESTRATOR_PUBLIC_ADDR) and we should fail loud
        # rather than spinning forever.
        self._max_redirect_chain = 5
        self._redirect_chain_count = 0

        # Heartbeats ride on a SEPARATE gRPC channel so long-running tenant work
        # (safetensors.load_file on 7 GB, streaming_dtype_cast, large job_output
        # chunks) on the primary data stream can't starve them. The scheduler
        # only uses the heartbeat message's WorkerId to look up the existing
        # WorkerInfo and bump LastActiveAt, so the second stream is transparent
        # to the rest of the dispatch flow.
        self._heartbeat_channel: Optional[Any] = None
        self._heartbeat_stub: Optional[Any] = None
        self._heartbeat_stream: Optional[Any] = None
        self._heartbeat_outgoing_queue: queue.Queue[Any] = queue.Queue()

        self._receive_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_drain_thread: Optional[threading.Thread] = None
        self._job_queue: "queue.Queue[Tuple[Callable[..., Any], Tuple[Any, ...], str, float]]" = queue.Queue(maxsize=1024)
        self._job_executor = ThreadPoolExecutor(thread_name_prefix="gen-worker-job")
        # Started lazily from run() (and re-checked on every reconnect via
        # _ensure_job_dispatch_thread). Don't start here — self._running is
        # False during __init__, so the loop would exit immediately.
        self._job_dispatch_thread: Optional[threading.Thread] = None

        self._reconnect_delay_base = max(0, reconnect_delay)
        self._reconnect_delay_max = 60
        self._reconnect_jitter_seconds = 1.0
        self._lb_only_retries = lb_only_retries

        resolved_model_manager = model_manager
        # Auto-wire: if no explicit manager and diffusers is available, use the
        # built-in DiffusersModelManager so LoadModelCommand works out of the box.
        if resolved_model_manager is None:
            try:
                import diffusers  # noqa: F401
                from .pipeline.model_manager import DiffusersModelManager
                resolved_model_manager = DiffusersModelManager()
                logger.info("Auto-wired DiffusersModelManager (diffusers detected)")
            except ImportError:
                pass
        self._model_manager = resolved_model_manager
        self._downloader = downloader
        if self._downloader is None:
            self._downloader = ModelRefDownloader()
        self._supported_model_ids_from_scheduler: Optional[List[str]] = None  # allowlist from scheduler (repo refs)
        self._required_flavor_refs_from_scheduler: Optional[List[str]] = None  # warm-start pinned flavors
        self._release_allowed_model_ids: Optional[set[str]] = None
        # Immutable allowlist derived from the baked discovery manifest (tenant-declared scope).
        # Scheduler config may narrow this set but must never widen it.
        self._manifest_allowed_model_ids: Optional[set[str]] = None
        # Top-level FIXED model keyspace from baked discovery manifest (short_key -> model_ref).
        self._fixed_model_id_by_key: Dict[str, str] = {}
        # Per-function payload model keyspaces from baked discovery manifest.
        # function_name -> short_key -> model_ref
        self._payload_model_id_by_key_by_function: Dict[str, Dict[str, str]] = {}
        # DType specs tracked in the same shape as model id maps.
        self._fixed_model_spec_by_key: Dict[str, Dict[str, Any]] = {}
        self._payload_model_spec_by_key_by_function: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Orchestrator-resolved manifests received in EndpointConfig (startup prefetch baseline).
        # Keys are canonical bare tensorhub refs (e.g. "owner/repo@sha256:<digest>").
        self._resolved_repos_by_id_baseline: Dict[str, Any] = {}
        # Issue #17: provider index built from endpoint.lock at boot. Keys are
        # bare ref strings (matching what the orchestrator's wire format
        # sends), values are provider tags ("tensorhub" | "hf" | "civitai").
        # Consulted by ModelRefDownloader.download and the other
        # parse_model_ref call sites to recover provider for refs whose
        # wire-format string is a bare owner/repo with no provider prefix.
        # Refs absent from the index default to provider="tensorhub".
        self._provider_by_ref_index: Dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        # Orchestrator-reported availability state received with each
        # EndpointConfig. See  for the
        # FIXED vs PAYLOAD binding split.
        #   _disabled_functions_by_name: function_name -> metadata for
        #     functions whose FIXED refs failed terminally. Worker skips
        #     prefetch for refs used ONLY by these functions and omits the
        #     functions from its advertised spec.
        #   _payload_ref_availability_by_function: function_name ->
        #     short_key -> status dict. Worker consults this before
        #     dispatching a request; if the invocation names a short_key
        #     whose status is terminal-non-resolved it rejects locally.
        self._disabled_functions_by_name: Dict[str, Dict[str, Any]] = {}
        self._worker_local_unavailable_functions_by_name: Dict[str, Dict[str, Any]] = {}
        self._payload_ref_availability_by_function: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Per-request tracking of which caller-supplied refs the active
        # invocation bound via the orchestrator-stamped `resolved_models` map
        # (the 0.7.0 override channel — replaces the old Src.PAYLOAD_REF
        # signature pattern). Set by the dispatch path before the tenant
        # body runs; read by `_map_exception` to classify post-download
        # failures as `ref_compatibility_surprise`. Empty dict = no caller
        # refs on this request; the classifier returns False.
        self._current_payload_ref_keys: Dict[str, str] = {}
        self._prefetch_thread: Optional[threading.Thread] = None
        self._model_init_done_event = threading.Event() # To signal model init is complete

        # LRU model cache for tracking VRAM and disk-cached models
        self._model_cache = ModelCache()

        # Initialize model config from the manifest
        if manifest and isinstance(manifest, dict):
            # Issue #17: build the {bare_ref -> provider} index from the
            # baked endpoint.lock manifest. The downloader and other
            # parse_model_ref call sites consult this to route HF / civitai
            # bindings through the right code path; refs absent from the
            # index default to "tensorhub" at the consuming site.
            try:
                self._provider_by_ref_index = build_provider_index_from_manifest(manifest)
                if self._provider_by_ref_index:
                    logger.info(
                        "Built provider index from manifest entries=%d providers=%s",
                        len(self._provider_by_ref_index),
                        sorted(set(self._provider_by_ref_index.values())),
                    )
            except Exception as e:
                # Defensive — a malformed manifest entry must not block boot.
                # Missing entries fall through to the default tensorhub
                # provider so existing tensorhub builds keep working.
                logger.warning("Failed to build provider index from manifest: %s", e)
                self._provider_by_ref_index = {}

            global_models = manifest.get("models")
            if isinstance(global_models, dict):
                out_fixed, out_fixed_spec = _parse_manifest_model_mapping(global_models)
                if out_fixed:
                    self._fixed_model_id_by_key = out_fixed
                if out_fixed_spec:
                    self._fixed_model_spec_by_key = out_fixed_spec

            mbf = manifest.get("models_by_function")
            if isinstance(mbf, dict):
                for fn_name, mapping in mbf.items():
                    if not isinstance(mapping, dict):
                        continue
                    fn = str(fn_name).strip()
                    if not fn:
                        continue
                    keyspace, keyspace_spec = _parse_manifest_model_mapping(mapping)
                    if keyspace:
                        self._payload_model_id_by_key_by_function[fn] = keyspace
                    if keyspace_spec:
                        self._payload_model_spec_by_key_by_function[fn] = keyspace_spec

            # Compute union allowlist for prefetch/guardrails.
            allowed: set[str] = set()
            allowed.update(self._fixed_model_id_by_key.values())
            for m in self._payload_model_id_by_key_by_function.values():
                allowed.update(m.values())
            self._manifest_allowed_model_ids = allowed or None
            # Default enforcement scope to what the tenant declared (baked manifest),
            # until the scheduler narrows it further.
            self._release_allowed_model_ids = self._manifest_allowed_model_ids

            if self._fixed_model_id_by_key or self._payload_model_id_by_key_by_function:
                logger.info(
                    "Loaded model mappings from manifest (fixed=%d, functions=%d)",
                    len(self._fixed_model_id_by_key),
                    len(self._payload_model_id_by_key_by_function),
                )

        if self._model_manager:
            logger.info(f"ModelManager of type '{type(self._model_manager).__name__}' provided.")
            # If we have models from manifest, start pre-download in background
            if self._release_allowed_model_ids:
                self._supported_model_ids_from_scheduler = list(self._release_allowed_model_ids)
                logger.info(f"Starting pre-download of {len(self._supported_model_ids_from_scheduler)} models from manifest")
                model_init_thread = threading.Thread(
                    target=self._process_release_config_async_wrapper,
                    daemon=True,
                    name="ManifestModelInit"
                )
                model_init_thread.start()
            else:
                # No models to pre-download, mark init as done
                self._model_init_done_event.set()
        else:
            logger.info("No ModelManager provided. Worker operating in simple mode regarding models.")
            self._model_init_done_event.set() # No model init to wait for if no manager
        if self._downloader:
            logger.info(f"ModelDownloader of type '{type(self._downloader).__name__}' configured.")

        logger.info(
            f"Created worker: ID={self.worker_id}, Scheduler={scheduler_addr}, WireProtocol={wire_protocol_version_string()}"
        )
        self._emit_startup_phase(
            "boot",
            status="ok",
            emit_worker_event=False,
            scheduler_seed_count=len(self.scheduler_addrs),
        )

        # Discover functions before setting signals? Maybe after. Let's do it here.
        self._discover_and_register_functions()

        self._verify_worker_jwt()

        # The worker is usually a top-level process. When embedded in tests/dev
        # helpers, it might be constructed in a non-main thread (signal handlers
        # are only allowed in the main thread).
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_interrupt)
                signal.signal(signal.SIGTERM, self._handle_interrupt)
        except Exception:
            pass

    @staticmethod
    def _normalize_scheduler_addrs(primary: str, addrs: Optional[List[str]]) -> List[str]:
        unique: List[str] = []
        for addr in [primary] + (addrs or []):
            addr = (addr or "").strip()
            if addr and addr not in unique:
                unique.append(addr)
        return unique

    @staticmethod
    def _extract_leader_addr(details: Optional[str]) -> Optional[str]:
        if not details:
            return None
        if details.startswith("not_leader:"):
            leader = details.split("not_leader:", 1)[1].strip()
            return leader or None
        return None

    @staticmethod
    def _is_protocol_incompatibility(details: Optional[str]) -> bool:
        return bool(details and details.startswith("unsupported_worker_protocol:"))

    def _handle_protocol_incompatibility(self, details: str) -> None:
        logger.error(
            "Scheduler rejected worker wire protocol %s (%s). "
            "This worker cannot connect to the current orchestrator policy.",
            wire_protocol_version_string(),
            details,
        )
        self._emit_startup_phase(
            "scheduler_protocol_incompatible",
            status="error",
            level=logging.ERROR,
            emit_worker_event=False,
            wire_protocol=wire_protocol_version_string(),
            grpc_details=details,
        )
        self._running = False
        self._close_connection()
        self._stop_event.set()

    def _set_scheduler_addr(self, addr: str) -> None:
        addr = addr.strip()
        if not addr:
            return
        self.scheduler_addr = addr
        if not self._lb_only_retries and addr not in self.scheduler_addrs:
            self.scheduler_addrs.insert(0, addr)

    def _next_reconnect_delay(self, attempt: int) -> float:
        backoff = self._reconnect_delay_base * (2 ** max(attempt - 1, 0))
        if self._reconnect_delay_max > 0:
            backoff = min(backoff, self._reconnect_delay_max)
        jitter = random.uniform(0, self._reconnect_jitter_seconds) if self._reconnect_jitter_seconds > 0 else 0.0
        return backoff + jitter

    def _iter_scheduler_addrs(self) -> Iterator[str]:
        seen = set()
        for addr in self.scheduler_addrs:
            addr = addr.strip()
            if not addr or addr in seen:
                continue
            seen.add(addr)
            yield addr

    def _verify_worker_jwt(self) -> None:
        if not self.worker_jwt or not self._jwks_cache:
            return
        try:
            header = jwt.get_unverified_header(self.worker_jwt)
            kid = header.get("kid")
            key = self._jwks_cache.get_key(kid)
            if not key:
                raise ValueError("JWKS key not found for token")
            options: Any = {"verify_aud": bool(self._jwt_audience)}
            jwt.decode(
                self.worker_jwt,
                key=key,
                algorithms=["RS256"],
                audience=self._jwt_audience or None,
                issuer=self._jwt_issuer or None,
                options=options,
            )
            logger.info("Worker-connect JWT verified against scheduler JWKS.")
        except Exception as e:
            logger.error(f"Worker-connect JWT verification failed: {e}")
            raise

    def _looks_like_ref_compatibility_surprise(self, exc: BaseException) -> bool:
        """Classify load-time errors as ``ref_compatibility_surprise``.

        Runs when the current request carried a
        caller-supplied ref.

        Runs only when ``_current_payload_ref_keys`` is populated (meaning
        the orchestrator stamped a `resolved_models` override on this
        request and the dispatch path recorded which params the caller
        substituted). Without that flag we don't want to false-positive on
        requests that never involved a caller-supplied ref — an internal
        fine-tune or a fixed-binding endpoint's state_dict mismatch is NOT
        a compatibility surprise, it's a real bug.

        Detection patterns (sufficient, not exhaustive):

        - ``RuntimeError`` whose message starts with ``Error(s) in
          loading state_dict`` — the canonical PyTorch load failure when
          tensor names don't match.
        - ``KeyError`` on a diffusers component name (text_encoder,
          text_encoder_2, unet, vae, transformer, scheduler) — the repo's
          model_index declared a layout the pipeline class doesn't
          understand, or the tenant looked up a component that isn't
          there.
        - ``OSError`` / ``FileNotFoundError`` mentioning ``config.json``,
          ``model_index.json``, or a known diffusers component dir — the
          ref doesn't have the expected file layout despite having the
          right top-level class name.
        - The substrings ``"Cannot load"`` / ``"does not appear to be a
          valid"`` / ``"missing keys"`` / ``"unexpected keys"`` /
          ``"size mismatch for"`` anywhere in the message.
        """
        if not getattr(self, "_current_payload_ref_keys", None):
            return False
        msg = str(exc or "")
        lowered = msg.lower()
        # 1. State-dict key / shape mismatches — the reliable signal.
        if "error(s) in loading state_dict" in lowered:
            return True
        if "missing keys" in lowered or "unexpected keys" in lowered:
            return True
        if "size mismatch for" in lowered:
            return True
        # 2. Diffusers component KeyError during pipeline construction.
        if isinstance(exc, KeyError):
            comp_names = {
                "text_encoder", "text_encoder_2", "text_encoder_3",
                "unet", "transformer", "vae", "scheduler",
                "tokenizer", "tokenizer_2", "image_encoder", "prior",
            }
            # KeyError's str is the repr of the key — peel the quotes.
            key = msg.strip().strip("'\"")
            if key in comp_names:
                return True
        # 3. Missing config at from_pretrained.
        if isinstance(exc, (OSError, FileNotFoundError)):
            for tok in ("config.json", "model_index.json", "pytorch_model.bin",
                        "diffusion_pytorch_model", "safetensors"):
                if tok in lowered:
                    return True
        # 4. Free-form diffusers / transformers load errors.
        for tok in (
            "cannot load",
            "does not appear to be a valid",
            "cannot instantiate",
            "tried to load but failed",
        ):
            if tok in lowered:
                return True
        return False

    def _sanitize_safe_message(self, message: str) -> str:
        message = (message or "").strip()
        if not message:
            return "job failed"
        # Basic sanitization: remove obvious secrets/tokens/URLs/paths.
        # Keep this conservative/minimal; detailed diagnostics belong in logs.
        message = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer <redacted>", message)
        message = re.sub(r"https?://\S+", "<url>", message)
        message = re.sub(r"/(?:[A-Za-z0-9._\-]+/)*[A-Za-z0-9._\-]+", "<path>", message)
        return message[:500]

    def _map_exception(self, exc: BaseException) -> tuple[str, bool, str, str]:
        internal = f"{type(exc).__name__}: {str(exc)}".strip()
        if isinstance(exc, CanceledError):
            return "canceled", False, "canceled", internal
        if isinstance(exc, HardwareUnmetError):
            # Terminal: function self-disables on this worker. Not
            # retryable on the same worker; the orchestrator narrows future
            # dispatches away.
            return (
                "hardware_unmet",
                False,
                self._sanitize_safe_message(str(exc) or "hardware unmet"),
                internal,
            )
        # Explicit RefCompatibilitySurprise raised by tenant code or
        # auto-detected via the heuristic below. Subclasses ValidationError
        # so `isinstance` checks that expect "validation" still match, but
        # the classification is narrower: the caller's ref passed pre-
        # dispatch compat gates but failed at load time.
        if isinstance(exc, RefCompatibilitySurprise):
            return (
                "ref_compatibility_surprise",
                False,
                self._sanitize_safe_message(str(exc) or "ref compatibility failure"),
                internal,
            )
        # Heuristic: common load-time errors that almost always mean the
        # caller's ref doesn't match the function's expected shape. Runs
        # only when the current request carried a caller-supplied ref
        # (set via `_current_payload_ref_keys` in the dispatch path).
        if self._looks_like_ref_compatibility_surprise(exc):
            msg = str(exc) or ""
            if len(msg) > 240:
                msg = msg[:240] + "…"
            return (
                "ref_compatibility_surprise",
                False,
                self._sanitize_safe_message(msg or "ref compatibility failure"),
                internal,
            )
        if isinstance(exc, ValidationError) or isinstance(exc, ValueError):
            return "validation", False, self._sanitize_safe_message(str(exc) or "invalid input"), internal
        if isinstance(exc, RetryableError):
            return "retryable", True, self._sanitize_safe_message(str(exc) or "retryable error"), internal
        if isinstance(exc, ResourceError):
            return "resource", False, self._sanitize_safe_message(str(exc) or "resource exhausted"), internal
        if isinstance(exc, FatalError):
            return "fatal", False, self._sanitize_safe_message(str(exc) or "fatal error"), internal
        if isinstance(exc, AuthError):
            return "auth", False, self._sanitize_safe_message(str(exc) or "authentication failed"), internal
        # Torch OOM detection without importing torch at import time.
        if type(exc).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
            return "resource", False, "out of memory", internal
        return "internal", False, "internal error", internal

    def _emit_function_unavailable_signal(
        self,
        *,
        function_name: str,
        exc: HardwareUnmetError,
    ) -> None:
        """Emit WorkerFunctionUnavailableSignal for a hardware-gated self-disable.

        Also prunes the function from the worker's advertised spec (via
        ``_disabled_functions_by_name``) so an orchestrator reconnect
        does not re-dispatch the same failing function before the
        availability-snapshot update round-trips.
        """
        try:
            reason = getattr(exc, "reason", "") or "hardware_unmet"
            detail = str(exc)
            axes = exc.axes() if hasattr(exc, "axes") else {}
            signal = pb.WorkerFunctionUnavailableSignal(
                worker_id=str(self.worker_id or ""),
                release_id=str(self.release_id or ""),
                function_name=str(function_name or ""),
                reason=str(reason or ""),
                detail=self._sanitize_safe_message(detail)[:1024],
                detected_at_unix=int(time.time()),
                axes={str(k): str(v) for k, v in (axes or {}).items()},
            )
            self._send_message(pb.WorkerSchedulerMessage(worker_function_unavailable=signal))
            logger.warning(
                "function_unavailable_signal emitted: fn=%s reason=%s axes=%s",
                function_name,
                reason,
                axes,
            )
        except Exception:
            logger.exception("failed to emit WorkerFunctionUnavailableSignal for %s", function_name)

        # Locally mark this function disabled so subsequent dispatches on this
        # worker (before the orchestrator processes the upstream signal) are
        # rejected by the worker-side gate rather than being re-attempted.
        try:
            self._disabled_functions_by_name[str(function_name or "")] = {
                "function_name": str(function_name or ""),
                "reason": str(getattr(exc, "reason", "") or "hardware_unmet"),
                "detail": str(exc)[:1024],
                "self_disabled": True,
            }
        except Exception:
            pass

    def _emit_progress_event(self, event: Dict[str, Any]) -> None:
        try:
            request_id = event.get("request_id") or ""
            event_type = event.get("type") or ""
            payload = event.get("payload") or {}
            if "timestamp" not in payload:
                payload = dict(payload)
                payload["timestamp"] = event.get("timestamp", time.time())
            payload_json = json.dumps(payload).encode("utf-8")
            msg = pb.WorkerSchedulerMessage(
                worker_event=pb.WorkerEvent(
                    request_id=request_id,
                    event_type=event_type,
                    payload_json=payload_json,
                )
            )
            self._send_message(msg)
        except Exception:
            logger.exception("Failed to emit progress event")

    def _emit_worker_event_bytes(self, request_id: str, event_type: str, payload_json: bytes) -> None:
        """Best-effort worker->scheduler WorkerEvent emitter (must never fail a job)."""
        try:
            msg = pb.WorkerSchedulerMessage(
                worker_event=pb.WorkerEvent(
                    request_id=str(request_id or ""),
                    event_type=str(event_type or ""),
                    payload_json=bytes(payload_json or b"{}"),
                )
            )
            self._send_message(msg)
        except Exception:
            return

    def _emit_typed_worker_event(self, *, pb_type: str, msg_field: str, **fields: Any) -> bool:
        """Build and best-effort-send a typed `WorkerSchedulerMessage` oneof.

        Returns False when the proto class is unavailable (wire-version skew
        with the scheduler) or when the send fails. Never raises — this is a
        notification path; failures must not fail the job.
        """
        cls = getattr(pb, pb_type, None)
        if cls is None:
            return False
        try:
            msg = pb.WorkerSchedulerMessage(**{msg_field: cls(**fields)})
            self._send_message(msg)
            return True
        except Exception:
            return False

    def _emit_incremental_delta_typed(
        self,
        *,
        request_id: str,
        function_name: str,
        item_id: str,
        sequence: int,
        timestamp_unix_ms: int,
        delta_text: str,
        payload_json: bytes,
        audio_chunk: bytes = b"",
        audio_codec: str = "",
    ) -> bool:
        # #327: audio-token streaming. When the tenant yields a delta struct
        # with `audio_chunk: bytes` + `audio_codec: str` fields populated, the
        # bytes are routed to the typed proto slots instead of being
        # base64-encoded inside payload_json (saves ~33% wire size + the JSON
        # CPU tax). Text-shaped deltas leave audio_chunk empty.
        return self._emit_typed_worker_event(
            pb_type="IncrementalTokenDelta",
            msg_field="incremental_token_delta",
            request_id=str(request_id or ""),
            item_id=str(item_id or ""),
            function_name=str(function_name or ""),
            sequence=int(sequence),
            timestamp_unix_ms=int(timestamp_unix_ms),
            delta_text=str(delta_text or ""),
            payload_json=bytes(payload_json or b"{}"),
            audio_chunk=bytes(audio_chunk or b""),
            audio_codec=str(audio_codec or ""),
        )

    def _emit_incremental_done_typed(
        self,
        *,
        request_id: str,
        function_name: str,
        item_id: str,
        sequence: int,
        timestamp_unix_ms: int,
    ) -> bool:
        return self._emit_typed_worker_event(
            pb_type="IncrementalTokenStreamDone",
            msg_field="incremental_token_stream_done",
            request_id=str(request_id or ""),
            item_id=str(item_id or ""),
            function_name=str(function_name or ""),
            sequence=int(sequence),
            timestamp_unix_ms=int(timestamp_unix_ms),
        )

    def _emit_incremental_error_typed(
        self,
        *,
        request_id: str,
        function_name: str,
        item_id: str,
        sequence: int,
        timestamp_unix_ms: int,
        error_message: str,
    ) -> bool:
        return self._emit_typed_worker_event(
            pb_type="IncrementalTokenStreamError",
            msg_field="incremental_token_stream_error",
            request_id=str(request_id or ""),
            item_id=str(item_id or ""),
            function_name=str(function_name or ""),
            sequence=int(sequence),
            timestamp_unix_ms=int(timestamp_unix_ms),
            error_message=str(error_message or ""),
        )

    @staticmethod
    def _extract_audio_from_delta(delta_obj: Any) -> tuple[bytes, str]:
        """#327: AR-TTS audio-token streaming. If the tenant's yielded delta
        struct declares fields named `audio_chunk: bytes` and (optionally)
        `audio_codec: str`, peel them off so they can travel on the typed
        proto slots instead of being base64-encoded inside payload_json.
        Returns (audio_chunk_bytes, audio_codec_string); both empty when the
        delta is text-shaped.

        msgspec.Struct introspection is duck-typed: any object exposing the
        named attributes works (msgspec, dataclass, plain instance). bytes
        are passed through verbatim; str (in case the tenant hands base64
        through this path) is rejected — the contract is raw bytes only.
        """
        chunk_attr = getattr(delta_obj, "audio_chunk", None)
        codec_attr = getattr(delta_obj, "audio_codec", None)
        # msgspec.UNSET handling — tenants commonly default optional fields
        # to UNSET so they don't get serialized.
        try:
            if chunk_attr is msgspec.UNSET:
                chunk_attr = None
            if codec_attr is msgspec.UNSET:
                codec_attr = None
        except Exception:
            pass
        audio_bytes = b""
        if isinstance(chunk_attr, (bytes, bytearray, memoryview)):
            audio_bytes = bytes(chunk_attr)
        audio_codec = ""
        if isinstance(codec_attr, str):
            audio_codec = codec_attr
        return audio_bytes, audio_codec

    # #321: typed startup-phase enum on the wire (replaces stringly-typed
    # event_type dispatch). Map our internal phase strings (which still flow
    # into logs and ctx surfaces) to the proto enum so the orchestrator's
    # state machine receives a typed value.
    _STARTUP_PHASE_PROTO = {
        "booting":             1,  # WORKER_STARTUP_PHASE_BOOTING
        "models_downloading":  2,  # WORKER_STARTUP_PHASE_MODELS_DOWNLOADING
        "pipeline_loading":    3,  # WORKER_STARTUP_PHASE_PIPELINE_LOADING
        "ready":               4,  # WORKER_STARTUP_PHASE_READY
        "error":               5,  # WORKER_STARTUP_PHASE_ERROR
        # #322: warming = torch.compile graph capture / warmup() running.
        "warming":             6,  # WORKER_STARTUP_PHASE_WARMING
    }

    # ModelAvailabilityKind proto enum values (#321):
    _MODEL_AVAILABILITY_READY              = 1
    _MODEL_AVAILABILITY_DOWNLOAD_COMPLETED = 2
    _MODEL_AVAILABILITY_CACHED             = 3

    def _emit_model_ready(self, model_id: str, kind: int) -> None:
        """Emit a typed WorkerModelReadySignal (#321).

        Replaces the three event_type strings (model.ready, model.cached,
        model.download.completed) the worker used to send via worker_event.
        Orchestrator routes on the typed variant and uses the kind enum for
        the same downstream behavior (mark worker available; READY also
        appends to VRAM model list).
        """
        try:
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_model_ready=pb.WorkerModelReadySignal(
                        model_id=str(model_id or ""),
                        kind=kind,
                    )
                )
            )
        except Exception:
            pass

    def _emit_startup_phase(
        self,
        phase: str,
        *,
        status: str = "ok",
        level: int = logging.INFO,
        emit_worker_event: bool = True,
        **extra: Any,
    ) -> None:
        """
        Emit a structured startup phase marker to logs and (best-effort) the
        scheduler via the typed WorkerStartupPhaseSignal (#321).
        """
        phase_str = str(phase or "").strip()
        elapsed_ms = int(max(0.0, (time.monotonic() - float(getattr(self, "_process_started_monotonic", time.monotonic()))) * 1000.0))
        log_payload: Dict[str, Any] = {
            "phase": phase_str,
            "status": str(status or "ok"),
            "worker_id": str(getattr(self, "worker_id", "") or ""),
            "scheduler_addr": str(getattr(self, "scheduler_addr", "") or ""),
            "pid": int(os.getpid()),
            "uid": int(os.getuid()) if hasattr(os, "getuid") else None,
            "gid": int(os.getgid()) if hasattr(os, "getgid") else None,
            "cwd": str(os.getcwd()),
            "elapsed_ms": elapsed_ms,
        }
        log_payload.update({k: v for k, v in extra.items() if v is not None})
        try:
            logger.log(level, "worker.startup.phase %s", json.dumps(log_payload, separators=(",", ":"), sort_keys=True))
        except Exception:
            logger.log(level, "worker.startup.phase phase=%s status=%s", phase_str, status)
        if emit_worker_event:
            phase_enum = self._STARTUP_PHASE_PROTO.get(phase_str, 0)  # 0 = UNSPECIFIED; orchestrator ignores
            detail_parts = []
            for k, v in extra.items():
                if v is None:
                    continue
                detail_parts.append(f"{k}={v}")
            detail = " ".join(detail_parts)
            try:
                self._send_message(
                    pb.WorkerSchedulerMessage(
                        worker_startup_phase=pb.WorkerStartupPhaseSignal(
                            phase=phase_enum,
                            status=str(status or "ok"),
                            scheduler_addr=str(getattr(self, "scheduler_addr", "") or ""),
                            elapsed_ms=elapsed_ms,
                            detail=detail,
                        )
                    )
                )
            except Exception:
                pass

    def _emit_worker_fatal(self, phase: str, exc: BaseException, *, exit_code: int = 1) -> None:
        """
        Emit a structured fatal event that includes traceback metadata.
        """
        try:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            tb = traceback.format_exc()
        payload: Dict[str, Any] = {
            "phase": str(phase or "").strip() or "unknown",
            "exception_class": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": tb,
            "exit_code": int(exit_code),
            "worker_id": str(getattr(self, "worker_id", "") or ""),
            "scheduler_addr": str(getattr(self, "scheduler_addr", "") or ""),
            "elapsed_ms": int(max(0.0, (time.monotonic() - float(getattr(self, "_process_started_monotonic", time.monotonic()))) * 1000.0)),
        }
        try:
            logger.error("worker.fatal %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
        except Exception:
            logger.exception("worker.fatal: %s", exc)
        try:
            self._emit_worker_event_bytes("", "worker.fatal", safe_json_bytes(payload))
        except Exception:
            pass

    def _emit_request_event(self, request_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            self._emit_worker_event_bytes(request_id, event_type, safe_json_bytes(payload or {}))
        except Exception:
            pass

    def _start_task_phase_watchdog(
        self,
        *,
        request_id: str,
        phase: str,
        warn_after_s: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[threading.Timer]:
        """
        Start a soft watchdog for long-running request phases.

        Emits `request.<phase>.stuck` if the timer fires.
        """
        try:
            timeout_s = float(warn_after_s or 0.0)
        except Exception:
            timeout_s = 0.0
        if timeout_s <= 0:
            return None
        started_at = time.monotonic()
        base_payload = dict(payload or {})

        def _fire() -> None:
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            ev_payload = dict(base_payload)
            ev_payload["phase"] = str(phase or "")
            ev_payload["warn_after_s"] = timeout_s
            ev_payload["elapsed_ms"] = elapsed_ms
            self._emit_request_event(request_id, f"request.{phase}.stuck", ev_payload)
            logger.warning(
                "request phase stuck request_id=%s phase=%s elapsed_ms=%d warn_after_s=%.1f payload=%s",
                request_id,
                phase,
                elapsed_ms,
                timeout_s,
                ev_payload,
            )

        timer = threading.Timer(timeout_s, _fire)
        timer.daemon = True
        timer.start()
        return timer


    def _get_gpu_busy_status(self) -> bool:
        lock = getattr(self, "_gpu_busy_lock", None)
        if lock is None:
            return False
        with lock:
            return int(getattr(self, "_gpu_busy_refcount", 0) or 0) > 0

    def _gpu_busy_enter(self) -> None:
        if not bool(getattr(self, "_has_gpu", False)):
            return
        lock = getattr(self, "_gpu_busy_lock", None)
        if lock is None:
            return
        with lock:
            cur = int(getattr(self, "_gpu_busy_refcount", 0) or 0)
            self._gpu_busy_refcount = cur + 1

    def _gpu_busy_exit(self) -> None:
        if not bool(getattr(self, "_has_gpu", False)):
            return
        lock = getattr(self, "_gpu_busy_lock", None)
        if lock is None:
            return
        with lock:
            cur = int(getattr(self, "_gpu_busy_refcount", 0) or 0)
            if cur <= 1:
                self._gpu_busy_refcount = 0
            else:
                self._gpu_busy_refcount = cur - 1

    @staticmethod
    def _function_needs_gpu(spec: Any) -> bool:
        """True iff the function/class declares ``accelerator=cuda*``.

        Inspects the spec's resources block (function-level metadata, NOT
        the build profile). Returns False when the field is unset/unknown —
        we don't artificially serialize handlers whose declaration we
        cannot read.

        Used by ``_execute_serial_class_request`` / conversion / training
        dispatch paths to decide whether to acquire ``_gpu_semaphore``.
        ``accelerator=none`` endpoints (marco-polo class) skip the
        semaphore entirely and scale to ThreadPoolExecutor capacity.
        """
        if spec is None:
            return False
        resources = getattr(spec, "resources", None)
        if resources is None:
            return False
        accel = getattr(resources, "accelerator", None)
        if accel is None:
            return False
        try:
            return str(accel).strip().lower().startswith("cuda")
        except Exception:
            return False

    def _bind_gpu_for_request(self, ctx: Any) -> int:
        """Set ``CUDA_VISIBLE_DEVICES`` for this handler's GPU index.

        Returns the gpu_index this handler should bind to, or -1 if no
        GPU binding is needed (no compute attached, accelerator!=cuda).

        The orchestrator stamps ``ResolvedCompute.gpu_index`` (proto field
        9; added by the parallel gen-orchestrator concurrency rewrite) on
        each dispatch when a multi-GPU worker has 2 GPU jobs in flight.
        We use ``getattr`` with default 0 so this worker keeps booting
        cleanly against an older pb that lacks the field.

        Caveat: ``os.environ["CUDA_VISIBLE_DEVICES"]`` is process-wide.
        Once a handler has loaded a CUDA context, changing the env var
        later doesn't move it. For multi-GPU workers, handlers MUST
        either (a) call ``torch.cuda.set_device(ctx.compute.gpu_index)``
        explicitly, OR (b) trust ``CUDA_VISIBLE_DEVICES`` BEFORE any
        torch/cuda import or call inside the handler. We emit a debug
        log when the gpu_index changes between handlers so multi-GPU
        misconfigurations surface in operator logs.

        Call this AFTER the semaphore acquires (so two handlers can't
        race on CUDA_VISIBLE_DEVICES) and BEFORE the handler runs.
        """
        compute = getattr(ctx, "compute", None)
        if compute is None:
            return -1
        accelerator = str(getattr(compute, "accelerator", "") or "").lower()
        if not accelerator.startswith("cuda"):
            return -1
        gpu_index = int(getattr(compute, "gpu_index", 0) or 0)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        if prev is not None and prev != str(gpu_index):
            logger.debug(
                "gpu_index_changed_between_handlers from=%s to=%s — "
                "multi-GPU handlers should use torch.cuda.set_device() "
                "rather than relying on this env var",
                prev, gpu_index,
            )
        return gpu_index

    def _model_use_enter(self, canonical_model_id: str) -> None:
        mid = str(canonical_model_id or "").strip()
        if not mid:
            return
        lock = getattr(self, "_active_model_use_lock", None)
        if lock is None:
            return
        with lock:
            mp = getattr(self, "_active_model_use_counts", None)
            if mp is None:
                self._active_model_use_counts = {}
                mp = self._active_model_use_counts
            mp[mid] = int(mp.get(mid, 0) or 0) + 1

    def _model_use_exit(self, canonical_model_id: str) -> None:
        mid = str(canonical_model_id or "").strip()
        if not mid:
            return
        lock = getattr(self, "_active_model_use_lock", None)
        if lock is None:
            return
        with lock:
            mp = getattr(self, "_active_model_use_counts", None) or {}
            cur = int(mp.get(mid, 0) or 0) - 1
            if cur <= 0:
                mp.pop(mid, None)
            else:
                mp[mid] = cur
            try:
                self._active_model_use_counts = mp
            except Exception:
                pass

    def _is_model_in_use(self, canonical_model_id: str) -> bool:
        mid = str(canonical_model_id or "").strip()
        if not mid:
            return False
        lock = getattr(self, "_active_model_use_lock", None)
        if lock is None:
            return False
        with lock:
            mp = getattr(self, "_active_model_use_counts", None) or {}
            return int(mp.get(mid, 0) or 0) > 0


    def _discover_and_register_functions(self) -> None:
        """Discover and register handlers in user modules.

        Class-shape discovery (``@inference`` / ``@training`` / ``@dataset`` /
        ``@conversion`` on a class) flows through
        :func:`gen_worker.discovery.walk.find_endpoint_classes` — the same
        walker the build-time discovery uses. Keeping the two paths unified
        is the whole point of issue #12: when they drift, the worker boots
        fine but the orchestrator's dispatch fails with `Unknown function
        requested` (the conversion-endpoint outage of 2026-05-16).

        Legacy function-shape ``@inference`` / ``@conversion`` decorators are
        still scanned per-module so a stray decorator surfaces as a registered
        handler (or a loud error) instead of vanishing silently. Post-#322 the
        canonical shape is class-based.
        """
        logger.info(
            "Discovering worker handlers in modules: %s...",
            self.user_module_names,
        )
        discovered = 0

        # Class-shape: single walker call shared with build-time discovery.
        # The walker handles import failures, submodule walks, re-export
        # dedup, and third-party rejection — all the band-aids the
        # previous runtime path grew separately.
        found_classes = find_endpoint_classes(list(self.user_module_names))
        for found in found_classes:
            try:
                n = self._register_endpoint_class(found.cls, found.endpoint_spec)
            except Exception as exc:
                logger.exception(
                    "Skipping endpoint class '%s' from module '%s': %s",
                    found.qualname, found.module, exc,
                )
                continue
            discovered += n

        # Function-shape: per-module dict scan. Walks the same modules the
        # walker imported (they're now in sys.modules) so we don't re-import.
        for module_name in self.user_module_names:
            module = sys.modules.get(module_name)
            if module is None:
                # The walker logs the import failure already; nothing to scan.
                continue
            for _, obj in inspect.getmembers(module):
                if not inspect.isfunction(obj):
                    continue

                # Transform-kind endpoints: @conversion decorated.
                # The wrapper's (request_context, payload) signature isn't a
                # regular @inference shape; register via TrainingFunctionSpec
                # without running _inspect_request_spec.
                if getattr(obj, "_is_training_function", False) is True:
                    python_name = getattr(obj, "__name__", None)
                    if not python_name:
                        logger.error("Skipping unnamed @conversion in %s", module_name)
                        continue
                    # Slugify the Python name to match what orchestrator dispatches
                    # (matches the inference-function registration convention so
                    # function_name lookups agree at RPC time).
                    name = slugify_name(python_name)
                    if not name:
                        logger.error("@conversion '%s' in %s: function name cannot be normalized",
                                     python_name, module_name)
                        continue
                    if name in self._training_specs:
                        logger.warning("Handler name conflict for '%s'; skipping", name)
                        continue
                    resources = getattr(obj, "_worker_resources", None)
                    self._training_specs[name] = obj
                    if resources is not None:
                        self._discovered_resources[name] = resources
                    discovered += 1
                    logger.info("Registered training_function: '%s'", name)
                    continue

                if getattr(obj, "_is_inference_function", False) is True:
                    try:
                        spec = self._inspect_request_spec(obj)
                    except Exception as exc:
                        logger.error("Skipping function '%s': %s", getattr(obj, "__name__", "<unknown>"), exc)
                        continue
                    if spec.name in self._request_specs:
                        logger.warning("Handler name conflict for '%s'; skipping", spec.name)
                        continue
                    self._request_specs[spec.name] = spec
                    self._discovered_resources[spec.name] = spec.resources
                    self._function_schemas[spec.name] = (
                        spec.input_schema_json,
                        spec.output_schema_json,
                        spec.delta_schema_json,
                        spec.injection_json,
                    )
                    discovered += 1
                    logger.info("Registered function: '%s' (%s)", spec.name, spec.output_mode)
                    continue

        if discovered == 0:
            logger.error(
                "No worker handlers found in modules: %s. The worker will register "
                "with the orchestrator but reject every dispatched request with "
                "'Unknown function requested'. Common causes: (1) a module "
                "import failed (look for 'Could not import user module' above); "
                "(2) the @inference decorator was not applied to any class in "
                "the listed modules; (3) the @inference classes are defined in "
                "third-party packages (look for 'Skipping endpoint class' logs).",
                self.user_module_names,
            )
        else:
            logger.info("Discovery complete. Found %d handlers.", discovered)

    def _register_endpoint_class(self, cls: type, ep_spec: Any) -> int:
        """Register one ``@inference`` (or sibling) class endpoint.

        Returns the number of externally-addressable functions registered.

        Two archetypes (#322):
          - ``BatchedWorker`` — async class with ``runtime="sglang"|"vllm"``.
            Routes through the continuous-batching engine path.
          - ``SerialWorker`` — sync class. One request fully owns the GPU.
            Routes through the synchronous job-queue path (#328).
        """
        kind = str(getattr(ep_spec, "kind", "") or "")
        runtime = getattr(ep_spec, "runtime", None)
        archetype = str(getattr(cls, "__gen_worker_archetype__", "") or "")
        function_methods = list(getattr(cls, "__gen_worker_function_methods__", []) or [])
        # #273: @batched_inference decorator path. The new SDK shape sets a
        # PARALLEL marker (``__gen_worker_batched_inference_function_methods__``)
        # so the dispatcher can route LLM-serving classes without overloading
        # the @inference function-methods slot. Tenant owns the engine; SDK
        # only invokes the async-generator method and forwards yielded signals
        # over the existing IncrementalToken* wire primitives.
        batched_inference_methods = list(
            getattr(cls, "__gen_worker_batched_inference_function_methods__", []) or []
        )
        if batched_inference_methods:
            return self._register_endpoint_class_batched_inference(
                cls, ep_spec, batched_inference_methods
            )
        if not function_methods:
            logger.warning(
                "@%s class %r declares no @%s.function methods; skipping",
                kind, cls.__name__, kind,
            )
            return 0

        # SerialWorker path (#322/#328): sync `@inference` class without
        # `runtime=`. Dispatch through `_serial_class_specs`.
        if archetype == "SerialWorker" and kind == "inference" and runtime is None:
            return self._register_endpoint_class_serial(cls, ep_spec)

        # #332: non-inference sync class endpoints (conversion / training /
        # dataset). Structurally SerialWorker (sync setup → generate →
        # shutdown) but the result emission path uploads ProducedFlavors and
        # publishes a revision instead of returning a struct. Dispatch
        # through `_conversion_class_specs`.
        if (
            archetype == "SerialWorker"
            and kind in ("conversion", "training", "dataset")
            and runtime is None
        ):
            return self._register_endpoint_class_conversion(cls, ep_spec)

        if runtime is None or kind != "inference":
            # Async (BatchedWorker) classes with non-inference kinds aren't
            # supported — the continuous-batching engine path is keyed on
            # inference. Surface the rejection so operators don't think a
            # mis-declared class silently disappears.
            logger.warning(
                "Recognized %s class endpoint %r (archetype=%s runtime=%s) — "
                "unsupported combination; conversion / training / dataset "
                "endpoints must be sync (SerialWorker shape)",
                kind, cls.__name__, archetype, runtime,
            )
            return 0

        # Construct the singleton instance. Real ``setup(**models[, engine=…])``
        # is deferred to ``_start_one_batched_instance`` (lazy on first dispatch).
        try:
            instance = cls()
        except TypeError as exc:
            raise ValueError(
                f"@inference class {cls.__name__!r}: __init__ must take no "
                f"arguments (got TypeError {exc}). Move model loading into "
                "setup()."
            ) from exc

        registered = 0
        resources: Resources = getattr(ep_spec, "resources", None) or Resources()

        for method_name, _unbound, fn_spec in function_methods:
            method = getattr(instance, method_name, None)
            if method is None or not callable(method):
                logger.warning(
                    "@inference class %r: method %r missing on instance after "
                    "construction; skipping",
                    cls.__name__, method_name,
                )
                continue
            name = str(getattr(fn_spec, "name", method_name) or method_name)
            slug = slugify_name(name)
            if not slug:
                logger.error(
                    "@inference class %r: method %r yields empty slug; skipping",
                    cls.__name__, method_name,
                )
                continue
            if slug in self._request_specs or slug in self._training_specs or slug in self._batched_specs:
                logger.warning("Handler name conflict for '%s'; skipping", slug)
                continue

            try:
                payload_type, delta_type, ctx_param, payload_param = self._inspect_batched_method(
                    cls, getattr(cls, method_name)
                )
            except Exception as exc:
                logger.error(
                    "@inference class %r method %r: signature invalid: %s",
                    cls.__name__, method_name, exc,
                )
                continue

            input_schema = msgspec.json.schema(payload_type)
            delta_schema = msgspec.json.schema(delta_type)
            input_schema_json = json.dumps(
                input_schema, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            delta_schema_json = json.dumps(
                delta_schema, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")

            bspec = _BatchedWorkerSpec(
                name=slug,
                instance=instance,
                method=method,
                resources=resources,
                ctx_param=ctx_param,
                payload_param=payload_param,
                payload_type=payload_type,
                delta_type=delta_type,
                runtime=str(runtime),
                input_schema_json=input_schema_json,
                delta_schema_json=delta_schema_json,
                timeout_ms=getattr(fn_spec, "timeout_ms", None),
            )
            self._batched_specs[slug] = bspec
            self._discovered_resources[slug] = resources
            # Mirror schemas into _function_schemas so the existing
            # function-capability advertisement path picks them up. The
            # 4-tuple shape is (input, output, delta, injection); for
            # BatchedWorker we surface delta in both the output and
            # delta slots (no separate "single" output shape exists) and
            # carry an empty injection blob (no per-request injections).
            self._function_schemas[slug] = (
                input_schema_json,
                delta_schema_json,
                delta_schema_json,
                b"[]",
            )
            registered += 1
            logger.info(
                "Registered BatchedWorker function: '%s' (class=%s, runtime=%s)",
                slug, cls.__name__, runtime,
            )

        if registered > 0:
            self._batched_instances.append(
                {
                    "cls_name": cls.__name__,
                    "instance": instance,
                    "endpoint_spec": ep_spec,
                    "runtime": str(runtime),
                    "engine": None,  # populated lazily in _start_one_batched_instance (only when tenant declares engine=)
                    "started": False,
                }
            )
        return registered

    def _inspect_batched_method(
        self,
        cls: type,
        method: Callable[..., Any],
    ) -> Tuple[type, type, str, str]:
        """Validate a BatchedWorker method signature.

        Must look like::

            async def fn(self, ctx: RequestContext, payload: MyInput) -> AsyncIterator[MyOutput]:
                ...

        Returns ``(payload_type, delta_type, ctx_param_name, payload_param_name)``.
        Raises ValueError on shape errors.
        """
        if not (inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)):
            raise ValueError(
                "BatchedWorker @inference.function method must be async (`async def`); "
                "use SerialWorker (sync class) for one-request-at-a-time endpoints."
            )

        try:
            hints = typing.get_type_hints(
                method, globalns=getattr(method, "__globals__", None), include_extras=True
            )
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}") from exc

        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if len(params) < 3:
            raise ValueError(
                "BatchedWorker method must take (self, ctx: RequestContext, payload: <Struct>)"
            )
        # params[0] is self.
        ctx_p = params[1]
        payload_p = params[2]
        ctx_type = hints.get(ctx_p.name)
        if ctx_type is not RequestContext:
            raise ValueError(
                f"second arg must be `ctx: RequestContext` (got annotation {ctx_type!r})"
            )
        payload_type = hints.get(payload_p.name)
        if not (isinstance(payload_type, type) and issubclass(payload_type, msgspec.Struct)):
            raise ValueError(
                f"third arg must be a msgspec.Struct payload (got {payload_type!r})"
            )

        ret = hints.get("return")
        if ret is None:
            raise ValueError("missing return type annotation")
        origin = get_origin(ret)
        # AsyncIterator[X] / AsyncGenerator[X, None] / cabc.AsyncIterator[X].
        if origin in (cabc.AsyncIterator, cabc.AsyncGenerator):
            args = get_args(ret)
            if not args:
                raise ValueError("AsyncIterator return type must declare a yield type")
            dt = args[0]
        else:
            # Allow bare msgspec.Struct return + the body using `yield` —
            # async-generator functions report a return annotation of the
            # yield type sometimes (depends on how the tenant wrote it).
            dt = ret
        if not (isinstance(dt, type) and issubclass(dt, msgspec.Struct)):
            raise ValueError(
                f"BatchedWorker method must yield a msgspec.Struct (got {dt!r})"
            )
        return payload_type, dt, ctx_p.name, payload_p.name

    def _inspect_batched_inference_method(
        self,
        cls: type,
        method: Callable[..., Any],
    ) -> Tuple[type, str, str]:
        """Validate a @batched_inference.function method signature (#273).

        Method shape contract::

            async def fn(self, ctx, payload: MyInput) -> AsyncIterator[Signal]:
                yield IncrementalTokenDelta(...)
                yield Done()

        Where Signal is ``IncrementalTokenDelta | Done | Error`` (the typed
        signal union exported from ``gen_worker.api.streaming``). The
        return type annotation is OPTIONAL — the dispatcher does
        ``isinstance(item, _SIGNAL_TYPES)`` at runtime, so tenants can omit
        it. The annotation, when present, is allowed to be either a single
        Signal struct or a ``Union[IncrementalTokenDelta, Done, Error]`` /
        ``IncrementalTokenDelta | Done | Error``.

        Returns ``(payload_type, ctx_param_name, payload_param_name)``.
        Raises ``ValueError`` on shape violations.
        """
        if not inspect.isasyncgenfunction(method):
            # Should have been caught at decoration time, but guard here
            # in case the tenant rebound the method after decoration.
            raise ValueError(
                "@batched_inference.function method must be an async generator "
                "(``async def`` with ``yield``)."
            )

        try:
            hints = typing.get_type_hints(
                method,
                globalns=getattr(method, "__globals__", None),
                include_extras=True,
            )
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}") from exc

        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if len(params) < 3:
            raise ValueError(
                "@batched_inference.function method must take "
                "(self, ctx, payload: <Struct>)"
            )
        # params[0] is self.
        ctx_p = params[1]
        payload_p = params[2]
        ctx_type = hints.get(ctx_p.name)
        # ctx may or may not be annotated; if it is, the only legal type
        # is RequestContext (which the worker passes in at dispatch time).
        if ctx_type is not None and ctx_type is not RequestContext:
            raise ValueError(
                f"second arg must be `ctx` (RequestContext or unannotated); "
                f"got annotation {ctx_type!r}"
            )
        payload_type = hints.get(payload_p.name)
        if not (
            isinstance(payload_type, type) and issubclass(payload_type, msgspec.Struct)
        ):
            raise ValueError(
                f"third arg must be a msgspec.Struct payload (got {payload_type!r})"
            )

        # Return-type annotation is optional; if present, sanity-check it
        # admits typed token-stream signals. We don't enforce the exact
        # Union shape — runtime ``isinstance`` against _SIGNAL_TYPES is the
        # source of truth.
        return payload_type, ctx_p.name, payload_p.name

    def _register_endpoint_class_batched_inference(
        self,
        cls: type,
        ep_spec: Any,
        batched_inference_methods: list[tuple[str, Callable[..., Any], Any]],
    ) -> int:
        """Register one ``@batched_inference`` class endpoint (#273).

        Parallel to the existing BatchedWorker (``runtime="sglang"|"vllm"``)
        leg in ``_register_endpoint_class``, but for the new SDK shape where
        the tenant owns engine construction inside ``setup()``. The SDK only:
          * Invokes the tenant's async-generator method with (ctx, payload).
          * Forwards each yielded ``IncrementalTokenDelta`` / ``Done`` /
            ``Error`` signal over the existing IncrementalToken* wire
            envelopes.
          * Sets ``ctx.cancelled()`` on stream EOF / cancellation so the
            tenant's loop can break out cleanly.

        Engine integration (vLLM / SGLang / transformers) is OUT OF SCOPE
        for this slice; the tenant builds + drives the engine inside their
        setup() body for now.
        """
        from .api.streaming import _SIGNAL_TYPES  # local import to avoid cycle

        if not batched_inference_methods:
            logger.warning(
                "@batched_inference class %r declares no .function methods; "
                "skipping",
                cls.__name__,
            )
            return 0

        # Construct the singleton instance. Real ``setup(**models)`` is
        # deferred to ``_start_one_batched_instance`` (lazy on first
        # dispatch), same lifecycle as the existing BatchedWorker leg.
        try:
            instance = cls()
        except TypeError as exc:
            raise ValueError(
                f"@batched_inference class {cls.__name__!r}: __init__ must "
                f"take no arguments (got TypeError {exc}). Move engine "
                "construction into setup()."
            ) from exc

        registered = 0
        resources: Resources = getattr(ep_spec, "resources", None) or Resources()

        for method_name, _unbound, fn_spec in batched_inference_methods:
            method = getattr(instance, method_name, None)
            if method is None or not callable(method):
                logger.warning(
                    "@batched_inference class %r: method %r missing on "
                    "instance after construction; skipping",
                    cls.__name__, method_name,
                )
                continue
            name = str(getattr(fn_spec, "name", method_name) or method_name)
            slug = slugify_name(name)
            if not slug:
                logger.error(
                    "@batched_inference class %r: method %r yields empty "
                    "slug; skipping",
                    cls.__name__, method_name,
                )
                continue
            if (
                slug in self._request_specs
                or slug in self._training_specs
                or slug in self._batched_specs
                or slug in self._serial_class_specs
            ):
                logger.warning("Handler name conflict for '%s'; skipping", slug)
                continue

            try:
                payload_type, ctx_param, payload_param = (
                    self._inspect_batched_inference_method(
                        cls, getattr(cls, method_name)
                    )
                )
            except Exception as exc:
                logger.error(
                    "@batched_inference class %r method %r: signature "
                    "invalid: %s",
                    cls.__name__, method_name, exc,
                )
                continue

            # Use IncrementalTokenDelta as the canonical delta_type for
            # wire-schema advertisement. The dispatcher routes Done / Error
            # through their own emit paths (incremental_done_typed /
            # incremental_error_typed), not as deltas — so the schema only
            # describes what tenants send through the delta channel.
            from .api.streaming import IncrementalTokenDelta as _ITDStruct

            input_schema = msgspec.json.schema(payload_type)
            delta_schema = msgspec.json.schema(_ITDStruct)
            input_schema_json = json.dumps(
                input_schema, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            delta_schema_json = json.dumps(
                delta_schema, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")

            bspec = _BatchedWorkerSpec(
                name=slug,
                instance=instance,
                method=method,
                resources=resources,
                ctx_param=ctx_param,
                payload_param=payload_param,
                payload_type=payload_type,
                # delta_type=None semantic isn't representable in the
                # current dataclass (it requires a Struct type), so we
                # plug in the canonical IncrementalTokenDelta type here.
                # Runtime ``isinstance(item, spec.delta_type)`` is bypassed
                # in ``_execute_batched_request_async`` for tenants who
                # yield Done / Error too — we'll patch the dispatcher
                # below to recognize the signal-union shape.
                delta_type=_ITDStruct,
                runtime="tenant",  # marker: tenant owns the engine
                input_schema_json=input_schema_json,
                delta_schema_json=delta_schema_json,
                timeout_ms=getattr(fn_spec, "timeout_ms", None),
            )
            self._batched_specs[slug] = bspec
            self._discovered_resources[slug] = resources
            self._function_schemas[slug] = (
                input_schema_json,
                delta_schema_json,
                delta_schema_json,
                b"[]",
            )
            registered += 1
            logger.info(
                "Registered @batched_inference function: '%s' (class=%s, "
                "tenant-owned engine)",
                slug, cls.__name__,
            )

        if registered > 0:
            self._batched_instances.append(
                {
                    "cls_name": cls.__name__,
                    "instance": instance,
                    "endpoint_spec": ep_spec,
                    "runtime": "tenant",
                    "engine": None,  # tenant owns engine inside setup()
                    "started": False,
                    # Marker so dispatch helpers can short-circuit any
                    # SDK-engine probes (the tenant's setup() does it all).
                    "tenant_owned_engine": True,
                }
            )
        return registered

    def _register_endpoint_class_serial(self, cls: type, ep_spec: Any) -> int:
        """Register one synchronous ``@inference`` class as a SerialWorker (#322/#328).

        Mirrors ``_register_endpoint_class``'s BatchedWorker leg. Constructs the
        singleton class instance, inspects each ``@inference.function`` method
        for signature validity, and writes a ``_SerialWorkerSpec`` per method
        keyed by ``slugify_name(method_name)``. The actual ``setup(self, **models)``
        + optional ``warmup(self)`` calls are deferred to first dispatch so the
        worker can boot quickly and only pay the cold-load cost on the first
        request — see ``_start_one_serial_instance``.

        Sets ``TORCHINDUCTOR_CACHE_DIR`` env var BEFORE any tenant code runs
        (here at registration time) so that future ``torch.compile`` calls
        inside ``setup()`` see the persistent inductor cache.

        Returns the number of externally-addressable functions registered.
        """
        function_methods = list(getattr(cls, "__gen_worker_function_methods__", []) or [])
        kind = str(getattr(ep_spec, "kind", "") or "inference")

        # #322 cross-cutting hook: persistent torch.compile cache. The env var
        # must be set before any setup() (and certainly before torch.compile)
        # runs. Setting it here at discovery time also covers the case where
        # the tenant imports torch in their `setup()` body.
        self._configure_torchinductor_cache_dir()

        # Construct the singleton instance. setup() / warmup() are deferred to
        # first dispatch; here we just verify __init__ is no-arg.
        try:
            instance = cls()
        except TypeError as exc:
            raise ValueError(
                f"@{kind} class {cls.__name__!r}: __init__ must take no "
                f"arguments (got TypeError {exc}). Move model loading into "
                "setup()."
            ) from exc

        resources: Resources = getattr(ep_spec, "resources", None) or Resources()
        # #324: cross-request micro-batching declaration. Both kwargs must be
        # set for the aggregator to activate; either-None means "no batching".
        batch_window_ms = getattr(ep_spec, "batch_window_ms", None)
        max_batch = getattr(ep_spec, "max_batch", None)
        batching_declared = batch_window_ms is not None and max_batch is not None

        registered = 0

        for method_name, _unbound, fn_spec in function_methods:
            method = getattr(instance, method_name, None)
            if method is None or not callable(method):
                logger.warning(
                    "@inference class %r: method %r missing on instance after "
                    "construction; skipping",
                    cls.__name__, method_name,
                )
                continue
            name = str(getattr(fn_spec, "name", method_name) or method_name)
            slug = slugify_name(name)
            if not slug:
                logger.error(
                    "@inference class %r: method %r yields empty slug; skipping",
                    cls.__name__, method_name,
                )
                continue
            if (slug in self._request_specs
                    or slug in self._training_specs
                    or slug in self._batched_specs
                    or slug in self._serial_class_specs):
                logger.warning("Handler name conflict for '%s'; skipping", slug)
                continue

            try:
                (
                    payload_type,
                    output_type,
                    delta_type,
                    output_mode,
                    ctx_param,
                    payload_param,
                ) = self._inspect_serial_method(cls, getattr(cls, method_name))
            except Exception as exc:
                logger.error(
                    "@inference class %r method %r: signature invalid: %s",
                    cls.__name__, method_name, exc,
                )
                continue

            input_schema = msgspec.json.schema(payload_type)
            input_schema_json = json.dumps(
                input_schema, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            output_schema_json = b""
            delta_schema_json: Optional[bytes] = None
            if output_type is not None:
                output_schema = msgspec.json.schema(output_type)
                output_schema_json = json.dumps(
                    output_schema, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            if delta_type is not None:
                delta_schema = msgspec.json.schema(delta_type)
                delta_schema_json = json.dumps(
                    delta_schema, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")

            sspec = _SerialWorkerSpec(
                name=slug,
                instance=instance,
                method=method,
                resources=resources,
                ctx_param=ctx_param,
                payload_param=payload_param,
                payload_type=payload_type,
                output_mode=output_mode,
                output_type=output_type,
                delta_type=delta_type,
                input_schema_json=input_schema_json,
                output_schema_json=output_schema_json,
                delta_schema_json=delta_schema_json,
                timeout_ms=getattr(fn_spec, "timeout_ms", None),
            )
            self._serial_class_specs[slug] = sspec
            self._discovered_resources[slug] = resources
            # Mirror schemas into _function_schemas so the function-capability
            # advertisement path picks them up. Shape:
            # (input, output, delta, injection).
            self._function_schemas[slug] = (
                input_schema_json,
                output_schema_json or (delta_schema_json or b""),
                delta_schema_json,
                b"[]",
            )
            registered += 1
            logger.info(
                "Registered SerialWorker function: '%s' (class=%s, output_mode=%s)",
                slug, cls.__name__, output_mode,
            )

            # #324: optionally attach a cross-request micro-batching aggregator.
            # Creation is conditional on (a) both kwargs declared, AND (b) no
            # batch-breaking cache wrapper detected on the tenant instance at
            # registration time. setup() hasn't been called yet so caches added
            # there (the common case — TeaCache.apply() inside setup()) are
            # caught by a second scan in `_ensure_serial_class_started`.
            if batching_declared:
                from .api.micro_batch import (
                    MicroBatchAggregator,
                    should_disable_batching,
                )

                disable_reason = should_disable_batching(instance)
                if disable_reason:
                    logger.warning(
                        "MicroBatch aggregator NOT created for '%s' (class=%s): "
                        "%s. batch_window_ms=%s max_batch=%s will be ignored.",
                        slug, cls.__name__, disable_reason,
                        batch_window_ms, max_batch,
                    )
                else:
                    agg = MicroBatchAggregator(
                        function_name=slug,
                        batch_window_ms=int(batch_window_ms),
                        max_batch=int(max_batch),
                        call_fn=self._make_aggregator_call_fn(slug, sspec),
                    )
                    self._micro_batch_aggregators[slug] = agg
                    logger.info(
                        "MicroBatch aggregator created for '%s': window_ms=%d "
                        "max_batch=%d (drain coroutine starts on first request)",
                        slug, int(batch_window_ms), int(max_batch),
                    )

        if registered > 0:
            self._serial_class_instances.append(
                {
                    "cls_name": cls.__name__,
                    "instance": instance,
                    "endpoint_spec": ep_spec,
                    "archetype": "SerialWorker",
                    "started": False,
                    "started_lock": threading.Lock(),
                    "shutdown_done": False,
                }
            )
        return registered

    def _register_endpoint_class_conversion(self, cls: type, ep_spec: Any) -> int:
        """Register a sync class endpoint of kind conversion/training/dataset
        (#332).

        Structurally a SerialWorker — sync ``setup`` / ``shutdown`` plus one
        or more ``@conversion.function`` / ``@training.function`` /
        ``@dataset.function`` methods of shape::

            def generate(self, ctx: ConversionContext, payload: <Struct>)
                -> Iterator[ProducedFlavor]  # or list[ProducedFlavor]

        Differences from `_register_endpoint_class_serial`:

          * ``ctx`` is typed as ``ConversionContext`` (kind=conversion /
            training) or ``DatasetContext`` (kind=dataset) — both
            ``RequestContext`` subclasses; we capture the concrete class
            on the spec so dispatch instantiates the right one.
          * The return type is ``Iterator[ProducedFlavor]`` /
            ``list[ProducedFlavor]`` — NOT a single msgspec.Struct.
            ProducedFlavor itself IS a msgspec.Struct, so the same yield-
            type check used by the discovery validator works here.
          * Output emission goes through
            ``gen_worker.conversion.dispatch._finalize_produced_variants``
            (upload + publish_repo_revision / publish_dataset_revision),
            not through a wire-payload response.

        Lazy ``setup`` / ``shutdown`` reuses the same record list +
        helpers (`_serial_class_instances`, `_ensure_serial_class_started`,
        `_find_serial_record`) — the lifecycle contract is identical.
        """
        function_methods = list(getattr(cls, "__gen_worker_function_methods__", []) or [])
        kind = str(getattr(ep_spec, "kind", "") or "").strip()
        sub_kind = str(getattr(ep_spec, "sub_kind", "") or "").strip()

        # Pick the kind-specific RequestContext subclass once, at register
        # time. Done here so dispatch doesn't re-resolve per request.
        if kind == "dataset":
            from .request_context import DatasetContext
            ctx_class: Any = DatasetContext
        else:
            # conversion + training both get the ConversionContext mixin
            # (producer-contract publish + mktemp + open_output_writer).
            from .request_context import ConversionContext
            ctx_class = ConversionContext

        # Construct the singleton instance; setup runs lazily on first
        # dispatch via `_ensure_serial_class_started`.
        try:
            instance = cls()
        except TypeError as exc:
            raise ValueError(
                f"@{kind} class {cls.__name__!r}: __init__ must take no "
                f"arguments (got TypeError {exc}). Move model loading into "
                "setup()."
            ) from exc

        resources: Resources = getattr(ep_spec, "resources", None) or Resources()
        # `calibration` is part of the conversion contract for quantization
        # endpoints; it lives on the @conversion() decorator
        # historically. The class-shape decorator doesn't yet accept it,
        # but if it's attached (e.g. by a future decorator update or by
        # the tenant directly setting `__gen_worker_calibration__`) we
        # honor it.
        calibration_attr = getattr(cls, "__gen_worker_calibration__", None) or {}
        calibration_map: Dict[str, str] = {
            str(k): str(v) for k, v in (calibration_attr or {}).items()
        }

        registered = 0

        for method_name, _unbound, fn_spec in function_methods:
            method = getattr(instance, method_name, None)
            if method is None or not callable(method):
                logger.warning(
                    "@%s class %r: method %r missing on instance after "
                    "construction; skipping",
                    kind, cls.__name__, method_name,
                )
                continue
            name = str(getattr(fn_spec, "name", method_name) or method_name)
            slug = slugify_name(name)
            if not slug:
                logger.error(
                    "@%s class %r: method %r yields empty slug; skipping",
                    kind, cls.__name__, method_name,
                )
                continue
            if (
                slug in self._request_specs
                or slug in self._training_specs
                or slug in self._batched_specs
                or slug in self._serial_class_specs
                or slug in self._conversion_class_specs
            ):
                logger.warning("Handler name conflict for '%s'; skipping", slug)
                continue

            try:
                payload_type, ctx_param, payload_param = self._inspect_conversion_method(
                    cls, getattr(cls, method_name), ctx_class,
                )
            except Exception as exc:
                logger.error(
                    "@%s class %r method %r: signature invalid: %s",
                    kind, cls.__name__, method_name, exc,
                )
                continue

            input_schema = msgspec.json.schema(payload_type)
            input_schema_json = json.dumps(
                input_schema, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")

            cspec = _ConversionWorkerSpec(
                name=slug,
                instance=instance,
                method=method,
                resources=resources,
                ctx_param=ctx_param,
                payload_param=payload_param,
                payload_type=payload_type,
                endpoint_kind=kind,
                sub_kind=sub_kind,
                ctx_class=ctx_class,
                calibration=calibration_map,
                input_schema_json=input_schema_json,
                timeout_ms=getattr(fn_spec, "timeout_ms", None),
            )
            self._conversion_class_specs[slug] = cspec
            self._discovered_resources[slug] = resources
            self._function_schemas[slug] = (
                input_schema_json,
                b"",  # no structured output: ProducedFlavors are uploaded
                None,
                b"[]",
            )
            registered += 1
            logger.info(
                "Registered %s class function: '%s' (class=%s, sub_kind=%s)",
                kind, slug, cls.__name__, sub_kind or "<unset>",
            )

        if registered > 0:
            self._serial_class_instances.append(
                {
                    "cls_name": cls.__name__,
                    "instance": instance,
                    "endpoint_spec": ep_spec,
                    "archetype": "SerialWorker",
                    "started": False,
                    "started_lock": threading.Lock(),
                    "shutdown_done": False,
                }
            )
        return registered

    def _inspect_conversion_method(
        self,
        cls: type,
        method: Callable[..., Any],
        ctx_class: Any,
    ) -> Tuple[type, str, str]:
        """Validate a conversion / training / dataset class method signature
        (#332).

        Must look like::

            def fn(self, ctx: <ConversionContext|DatasetContext>,
                   payload: <msgspec.Struct>)
                -> Iterator[ProducedFlavor]  # or list[ProducedFlavor]

        Returns ``(payload_type, ctx_param_name, payload_param_name)``.

        Output type is fixed (``list[ProducedFlavor]`` / ``Iterator[…]``);
        we don't validate it strictly here — the dispatcher accepts any
        iterable yielding objects that quack like ``ProducedFlavor``.

        Async methods on a class otherwise classified as a conversion-kind
        SerialWorker are a validation error.
        """
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            raise ValueError(
                "conversion / training / dataset class methods must be sync "
                "(`def`); BatchedWorker (continuous batching) doesn't apply "
                "to these kinds."
            )

        try:
            hints = typing.get_type_hints(
                method, globalns=getattr(method, "__globals__", None), include_extras=True
            )
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}") from exc

        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if len(params) < 3:
            raise ValueError(
                "conversion-kind class method must take "
                "(self, ctx: ConversionContext, payload: <Struct>)"
            )
        ctx_p = params[1]
        payload_p = params[2]
        ctx_type = hints.get(ctx_p.name)
        # Accept the class itself or any subclass; ConversionContext and
        # DatasetContext both inherit from RequestContext but are the
        # tenant-facing types.
        if not (
            isinstance(ctx_type, type)
            and (issubclass(ctx_type, ctx_class) or issubclass(ctx_class, ctx_type))
        ):
            raise ValueError(
                f"second arg must be `ctx: {ctx_class.__name__}` (or a base "
                f"thereof); got {ctx_type!r}"
            )
        payload_type = hints.get(payload_p.name)
        if not (isinstance(payload_type, type) and issubclass(payload_type, msgspec.Struct)):
            raise ValueError(
                f"third arg must be a msgspec.Struct payload (got {payload_type!r})"
            )
        return payload_type, ctx_p.name, payload_p.name

    def _configure_torchinductor_cache_dir(self) -> None:
        """Ensure TORCHINDUCTOR_CACHE_DIR is set before any tenant setup()
        runs (#322 cross-cutting hooks).

        torch.compile caches the lowered fx graph + kernel code per-key under
        TORCHINDUCTOR_CACHE_DIR. Without persistence the worker re-compiles
        on every pod start — minutes per shape on cold cache. With persistence
        and a baked image layer, restarts are 5-15s per shape.

        Honors any env override (operator can point at a Runpod volume mount);
        only sets the default when unset.

        Idempotent: subsequent calls are no-ops once the directory exists.
        """
        existing = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "").strip()
        if existing:
            return
        default_dir = "/var/cache/gen-worker/inductor"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = default_dir
        try:
            Path(default_dir).mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as exc:
            # Falling back to a process-local tmpdir if /var/cache isn't
            # writable (e.g. dev mode running as unprivileged user). Inductor
            # will still cache; the cache just won't survive process restart.
            import tempfile
            fallback = tempfile.gettempdir() + "/gen-worker-inductor"
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = fallback
            try:
                Path(fallback).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            logger.warning(
                "TORCHINDUCTOR_CACHE_DIR fallback to %r (default %r not "
                "writable: %s). Compile cache will not persist across "
                "process restart.",
                fallback, default_dir, exc,
            )
        else:
            logger.info(
                "TORCHINDUCTOR_CACHE_DIR set to %s for torch.compile persistence",
                default_dir,
            )

    def _inspect_serial_method(
        self,
        cls: type,
        method: Callable[..., Any],
    ) -> Tuple[type, Optional[type], Optional[type], str, str, str]:
        """Validate a SerialWorker method signature.

        Must look like one of::

            def fn(self, ctx: RequestContext, payload: MyInput) -> MyOutput: ...
            def fn(self, ctx: RequestContext, payload: MyInput) -> Iterator[MyDelta]: ...

        Returns
        -------
        (payload_type, output_type, delta_type, output_mode, ctx_param_name, payload_param_name)

        - ``output_mode`` is ``"single"`` when the return type is a
          ``msgspec.Struct`` subclass, or ``"incremental"`` when it's an
          ``Iterator[Struct]`` / ``Iterable[Struct]``.
        - ``output_type`` is set in both modes (incremental mode uses the
          yielded delta type for both ``output_type`` and ``delta_type``).
        - ``delta_type`` is only set in incremental mode.

        Async methods on a class otherwise classified as SerialWorker are a
        validation error — they belong on the BatchedWorker leg.
        """
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            raise ValueError(
                "SerialWorker @inference.function method must be sync (`def`); "
                "use BatchedWorker (async class + runtime='sglang'|'vllm') for "
                "continuous-batching endpoints."
            )

        try:
            hints = typing.get_type_hints(
                method, globalns=getattr(method, "__globals__", None), include_extras=True
            )
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}") from exc

        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if len(params) < 3:
            raise ValueError(
                "SerialWorker method must take (self, ctx: RequestContext, payload: <Struct>)"
            )
        # params[0] is self.
        ctx_p = params[1]
        payload_p = params[2]
        ctx_type = hints.get(ctx_p.name)
        if ctx_type is not RequestContext:
            raise ValueError(
                f"second arg must be `ctx: RequestContext` (got annotation {ctx_type!r})"
            )
        payload_type = hints.get(payload_p.name)
        if not (isinstance(payload_type, type) and issubclass(payload_type, msgspec.Struct)):
            raise ValueError(
                f"third arg must be a msgspec.Struct payload (got {payload_type!r})"
            )

        ret = hints.get("return")
        if ret is None:
            raise ValueError("missing return type annotation")

        # Single-output: bare msgspec.Struct return.
        if isinstance(ret, type) and issubclass(ret, msgspec.Struct):
            return payload_type, ret, None, "single", ctx_p.name, payload_p.name

        # Incremental: Iterator[X] / Iterable[X].
        origin = get_origin(ret)
        if origin in (Iterator, Iterable, cabc.Iterator, cabc.Iterable):
            args = get_args(ret)
            if not args:
                raise ValueError("Iterator return type must declare a yield type")
            dt = args[0]
            if not (isinstance(dt, type) and issubclass(dt, msgspec.Struct)):
                raise ValueError(
                    f"SerialWorker method must yield a msgspec.Struct (got {dt!r})"
                )
            return payload_type, dt, dt, "incremental", ctx_p.name, payload_p.name

        raise ValueError(
            "SerialWorker method return type must be msgspec.Struct or "
            f"Iterator[msgspec.Struct] (got {ret!r})"
        )

    def _inspect_request_spec(self, func: Callable[..., Any]) -> _RequestSpec:
        python_name = func.__name__
        func_name = slugify_name(python_name)
        if not func_name:
            raise ValueError(f"{python_name}: function name cannot be normalized")
        resources: Resources = getattr(func, "_worker_resources", Resources())

        try:
            hints = typing.get_type_hints(func, globalns=func.__globals__, include_extras=True)
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}")

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("must accept ctx: RequestContext as first arg")

        ctx_name = params[0].name
        ctx_type = hints.get(ctx_name)
        if ctx_type is not RequestContext:
            raise ValueError("first argument must be ctx: RequestContext")

        # New 0.7.0 binding model: @inference(models={...}) attaches
        # __gen_worker_bindings__ keyed by parameter name. Treat any parameter
        # listed there as an injected binding; any msgspec.Struct parameter as
        # the payload. Everything else is a signature error.
        bindings_map: Dict[str, Binding] = dict(
            getattr(func, "__gen_worker_bindings__", None) or {}
        )

        injections: list[InjectionSpec] = []
        payload_type: Optional[type[msgspec.Struct]] = None
        payload_param: Optional[str] = None
        for p in params[1:]:
            ann = hints.get(p.name)
            if ann is None:
                raise ValueError(f"missing type annotation for param: {p.name}")
            if p.name in bindings_map:
                injections.append(
                    InjectionSpec(
                        param_name=p.name,
                        param_type=ann,
                        binding=bindings_map[p.name],
                    )
                )
                continue
            if isinstance(ann, type) and issubclass(ann, msgspec.Struct):
                if payload_type is not None:
                    raise ValueError("must accept exactly one msgspec.Struct payload arg")
                payload_type = ann
                payload_param = p.name
                continue
            raise ValueError(
                f"unsupported param type for {p.name!r}: {ann!r}. "
                "Inject models via @inference(models={...}); payload must be msgspec.Struct."
            )

        if payload_type is None or payload_param is None:
            raise ValueError("must accept exactly one msgspec.Struct payload arg")

        ret = hints.get("return")
        if ret is None:
            raise ValueError("missing return type annotation")

        output_mode = "single"
        output_type: Optional[type[msgspec.Struct]] = None
        delta_type: Optional[type[msgspec.Struct]] = None

        if isinstance(ret, type) and issubclass(ret, msgspec.Struct):
            output_type = ret
        else:
            origin = get_origin(ret)
            if origin in (Iterator, Iterable, cabc.Iterator, cabc.Iterable):
                args = get_args(ret)
                if len(args) != 1:
                    raise ValueError("incremental output return type must be Iterator[DeltaStruct]")
                dt = args[0]
                if not isinstance(dt, type) or not issubclass(dt, msgspec.Struct):
                    raise ValueError("delta type must be msgspec.Struct")
                output_mode = "incremental"
                delta_type = dt
                output_type = dt  # best-effort schema until proto adds delta schema field
            else:
                raise ValueError("return type must be msgspec.Struct or Iterator[msgspec.Struct]")

        input_schema = msgspec.json.schema(payload_type)
        output_schema = msgspec.json.schema(output_type)
        try:
            from .api.payload_constraints import apply_schema_constraints

            input_schema = apply_schema_constraints(input_schema, payload_type)
            output_schema = apply_schema_constraints(output_schema, output_type)
        except Exception:
            pass
        input_schema_json = json.dumps(input_schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
        output_schema_json = json.dumps(output_schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
        delta_schema_json = None
        if delta_type is not None:
            delta_schema = msgspec.json.schema(delta_type)
            try:
                from .api.payload_constraints import apply_schema_constraints

                delta_schema = apply_schema_constraints(delta_schema, delta_type)
            except Exception:
                pass
            delta_schema_json = json.dumps(delta_schema, separators=(",", ":"), sort_keys=True).encode("utf-8")

        injection_payload = [
            _binding_to_wire(inj.param_name, inj.param_type, inj.binding)
            for inj in injections
        ]
        injection_json = json.dumps(injection_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

        return _RequestSpec(
            name=func_name,
            func=func,
            resources=resources,
            ctx_param=ctx_name,
            payload_param=payload_param,
            payload_type=payload_type,
            output_mode=output_mode,
            output_type=output_type,
            delta_type=delta_type,
            injections=tuple(injections),
            input_schema_json=input_schema_json,
            output_schema_json=output_schema_json,
            delta_schema_json=delta_schema_json,
            injection_json=injection_json,
        )

    def _send_message(self, message: WorkerSchedulerMessage) -> None:
        """Add a message to the outgoing queue."""
        if self._running and not self._stop_event.is_set():
            try:
                self._outgoing_queue.put_nowait(message)
            except queue.Full:
                 logger.error("Outgoing message queue is full. Message dropped!")
        else:
            logger.warning("Attempted to send message while worker is stopping or stopped.")

    def _ensure_job_dispatch_thread(self) -> None:
        """Start the job-dispatch thread if it isn't running.

        Called from run() at boot and from _connect_once() on every reconnect
        so a thread that died for any reason gets resurrected before new
        requests can land on a dead _job_queue.
        """
        t = self._job_dispatch_thread
        if t is not None and t.is_alive():
            return
        self._job_dispatch_thread = threading.Thread(
            target=self._job_dispatch_loop,
            daemon=True,
            name="gen-worker-job-dispatch",
        )
        self._job_dispatch_thread.start()

    def _job_dispatch_loop(self) -> None:
        # Gate on self._running, NOT self._stop_event. _stop_event is set on
        # every reconnect by _handle_connection_error to wake the run-loop
        # and the recv-thread iterators — if this loop also exited on
        # _stop_event, the daemon thread would die on the first disconnect
        # and never come back, leaving subsequent reconnects with a live
        # receive thread enqueuing into _job_queue but nothing draining it.
        # _ensure_job_dispatch_thread() (called from _connect_once) restarts
        # this thread if it died for any reason.
        while self._running:
            try:
                target, args, request_id, enqueued_at = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._job_executor.submit(self._run_queued_job, target, args, request_id, enqueued_at)

    def _run_queued_job(
        self,
        target: Callable[..., Any],
        args: Tuple[Any, ...],
        request_id: str,
        enqueued_at: float,
    ) -> None:
        started_at = time.monotonic()
        with self._active_requests_lock:
            obs = self._request_observations.setdefault(request_id, {})
            obs["local_queue_ms"] = max(0, int((started_at - enqueued_at) * 1000))
            obs["started_at"] = started_at
        target(*args)

    def _materialize_assets(self, ctx: RequestContext, obj: Any) -> None:
        if isinstance(obj, (Asset, Tensors)):
            self._materialize_asset(ctx, obj)
            return
        if isinstance(obj, list):
            for it in obj:
                self._materialize_assets(ctx, it)
            return
        if isinstance(obj, dict):
            for it in obj.values():
                self._materialize_assets(ctx, it)
            return
        fields = getattr(obj, "__struct_fields__", None)
        if fields and isinstance(fields, (tuple, list)):
            for name in fields:
                try:
                    val = getattr(obj, name)
                except Exception as e:
                    logger.warning("_materialize_assets: could not access field %r on %s: %s", name, type(obj).__name__, e)
                    continue
                self._materialize_assets(ctx, val)

    def _auto_upload_output_assets(self, ctx: RequestContext, output_obj: Any) -> Any:
        """
        Auto-persist tenant-returned local assets.

        If a returned Asset/Tensors has local_path set, upload it through RequestContext.
        save_file()/save_checkpoint() and replace it with persisted metadata. This gives
        tenant code an optional shorthand: return local artifacts directly and let the
        worker persist them.
        """
        upload_idx = 0

        def _default_ref(local_path: str) -> str:
            nonlocal upload_idx
            leaf = os.path.basename(local_path) or "artifact.bin"
            ref = f"outputs/{ctx.request_id}/auto/{upload_idx:06d}-{leaf}"
            upload_idx += 1
            return _normalize_output_ref(ref)

        def _walk(v: Any) -> Any:
            if isinstance(v, Asset):
                local = str(getattr(v, "local_path", "") or "").strip()
                if not local:
                    return v
                ref = str(getattr(v, "ref", "") or "").strip() or _default_ref(local)
                saved: Asset | Tensors = ctx.save_file(ref, local)
                return saved

            if isinstance(v, Tensors):
                local = str(getattr(v, "local_path", "") or "").strip()
                if not local:
                    return v
                ref = str(getattr(v, "ref", "") or "").strip() or _default_ref(local)
                fmt = str(getattr(v, "format", "") or "").strip() or None
                saved = ctx.save_checkpoint(ref, local, format=fmt)
                return saved

            if isinstance(v, list):
                changed = False
                out: List[Any] = []
                for it in v:
                    ni = _walk(it)
                    out.append(ni)
                    if ni is not it:
                        changed = True
                if changed:
                    v[:] = out
                return v

            if isinstance(v, tuple):
                changed = False
                out_items: List[Any] = []
                for it in v:
                    ni = _walk(it)
                    out_items.append(ni)
                    if ni is not it:
                        changed = True
                return tuple(out_items) if changed else v

            if isinstance(v, dict):
                for k in list(v.keys()):
                    nv = _walk(v[k])
                    if nv is not v[k]:
                        v[k] = nv
                return v

            fields = getattr(v, "__struct_fields__", None)
            if fields and isinstance(fields, (tuple, list)):
                for name in fields:
                    try:
                        cur = getattr(v, name)
                    except Exception:
                        continue
                    nv = _walk(cur)
                    if nv is not cur:
                        try:
                            setattr(v, name, nv)
                        except Exception:
                            pass
            return v

        return _walk(output_obj)

    def _materialize_asset(self, ctx: RequestContext, asset: Asset | Tensors) -> None:
        if asset.local_path:
            return
        ref = (asset.ref or "").strip()
        if not ref:
            return
        if not (ref.startswith("http://") or ref.startswith("https://")):
            if mapped := ctx._materialized_input_url_for_ref(ref):
                ref = mapped
        base_dir = "/tmp/tensorhub/job"
        scope_id = _workspace_scope_id(ctx.request_id, getattr(ctx, "job_id", None))
        local_inputs_dir = os.path.join(base_dir, scope_id, "inputs")
        os.makedirs(local_inputs_dir, exist_ok=True)
        cache_dir = os.path.join(base_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        max_bytes = 200 * 1024 * 1024

        # External URL inputs — check shared cache first, download on miss.
        if ref.startswith("http://") or ref.startswith("https://"):
            if _url_is_blocked(ref):
                raise RuntimeError("input url blocked")
            download_token = (asset.download_token or "").strip() or None
            ext = os.path.splitext(urllib.parse.urlparse(ref).path)[1] or os.path.splitext(ref)[1]
            if not ext:
                try:
                    head_req = urllib.request.Request(ref, method="HEAD")
                    if download_token:
                        head_req.add_header("Authorization", f"Bearer {download_token}")
                    with urllib.request.urlopen(head_req, timeout=10) as head_resp:
                        cd = head_resp.headers.get("Content-Disposition") or ""
                        fname_match = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\r\n]+)', cd, re.IGNORECASE)
                        if fname_match:
                            ext = os.path.splitext(fname_match.group(1).strip())[1]
                except Exception:
                    pass
            cache_key_material = ref
            if download_token:
                token_hash = hashlib.sha256(download_token.encode("utf-8")).hexdigest()
                cache_key_material = f"{ref}\n{token_hash}"
            name_hash = hashlib.sha256(cache_key_material.encode("utf-8")).hexdigest()[:32]
            cache_path = os.path.join(cache_dir, f"{name_hash}{ext}")
            sidecar_path = cache_path + ".sha256"
            if not os.path.exists(cache_path):
                size, sha256_hex, mime = self._download_url_to_file(ref, cache_path, max_bytes, token=download_token)
                try:
                    with open(sidecar_path, "w") as _sf:
                        _sf.write(sha256_hex)
                except Exception:
                    pass
            else:
                size = os.path.getsize(cache_path)
                with open(cache_path, "rb") as f:
                    head = f.read(512)
                mime = _infer_mime_type(ref, head)
                try:
                    with open(sidecar_path) as _sf:
                        sha256_hex = _sf.read().strip()
                except FileNotFoundError:
                    # Sidecar missing for entries cached before this change — compute from file.
                    h = hashlib.sha256()
                    with open(cache_path, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                    sha256_hex = h.hexdigest()
                    try:
                        with open(sidecar_path, "w") as _sf:
                            _sf.write(sha256_hex)
                    except Exception:
                        pass
            local_path = os.path.join(local_inputs_dir, f"{name_hash}{ext}")
            if not os.path.exists(local_path):
                try:
                    os.link(cache_path, local_path)
                except Exception:
                    try:
                        import shutil
                        shutil.copyfile(cache_path, local_path)
                    except Exception:
                        local_path = cache_path
            asset.local_path = local_path
            if not asset.owner:
                asset.owner = self.owner
            if isinstance(asset, Asset):
                asset.mime_type = mime
            asset.size_bytes = size
            asset.sha256 = sha256_hex
            return

        # Non-URL refs require explicit file API credentials. Standard execution should
        # receive presigned input URLs from orchestrator and avoid this path.
        base = (getattr(ctx, "_file_api_base_url", None) or "").strip()
        token = (getattr(ctx, "_worker_capability_token", None) or "").strip()
        if not base or not token:
            raise RuntimeError("input ref was not materialized to URL and no file API credentials were provided")
        base = base.rstrip("/")
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"

        head_req = _http_request("HEAD", url, token, owner=ctx.owner)
        try:
            with urllib.request.urlopen(head_req, timeout=10) as resp:
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(f"failed to stat asset ({resp.status})")
                sha256_hex = (resp.headers.get("X-Cozy-SHA256") or "").strip()
                size_hdr = (resp.headers.get("X-Cozy-Size-Bytes") or "").strip()
                mime = (resp.headers.get("X-Cozy-Mime-Type") or "").strip()
        except urllib.error.HTTPError as e:
            code = getattr(e, "code", 0)
            if code in (401, 403):
                raise AuthError(f"file read unauthorized ({code}): check worker_capability_token validity") from e
            raise RuntimeError(f"failed to stat asset ({code or 'unknown'})") from e
        size = int(size_hdr) if size_hdr.isdigit() else 0
        if max_bytes > 0 and size > max_bytes:
            raise InputTooLargeError(size_bytes=size, max_bytes=max_bytes, source="input file")

        ext = os.path.splitext(ref)[1]
        if not ext and mime:
            guessed = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
            }.get(mime)
            ext = guessed or ""

        if not sha256_hex:
            sha256_hex = hashlib.sha256(ref.encode("utf-8")).hexdigest()
        cache_name = f"{sha256_hex[:32]}{ext}"
        cache_path = os.path.join(cache_dir, cache_name)

        if not os.path.exists(cache_path):
            get_req = _http_request("GET", url, token, owner=ctx.owner)
            try:
                with urllib.request.urlopen(get_req, timeout=30) as resp:
                    if resp.status < 200 or resp.status >= 300:
                        raise RuntimeError(f"failed to download asset ({resp.status})")
                    _size, _sha = self._stream_to_file(resp, cache_path, max_bytes)
                    if not size:
                        size = _size
                    if not sha256_hex:
                        sha256_hex = _sha
            except urllib.error.HTTPError as e:
                code = getattr(e, "code", 0)
                if code in (401, 403):
                    raise AuthError(f"file read unauthorized ({code}): check worker_capability_token validity") from e
                raise RuntimeError(f"failed to download asset ({code or 'unknown'})") from e

        local_path = os.path.join(local_inputs_dir, cache_name)
        if not os.path.exists(local_path):
            try:
                os.link(cache_path, local_path)
            except Exception:
                try:
                    import shutil

                    shutil.copyfile(cache_path, local_path)
                except Exception:
                    local_path = cache_path

        if not mime:
            with open(local_path, "rb") as f:
                head = f.read(512)
            mime = _infer_mime_type(ref, head)

        asset.local_path = local_path
        if not asset.owner:
            asset.owner = (ctx.owner or self.owner)
        if isinstance(asset, Asset):
            asset.mime_type = mime or None
        asset.size_bytes = size or None
        asset.sha256 = sha256_hex or None

    def _download_url_to_file(self, src: str, dst: str, max_bytes: int, token: Optional[str] = None) -> Tuple[int, str, Optional[str]]:
        attempts = 3
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < max(1, attempts):
            attempt += 1
            try:
                class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Optional[urllib.request.Request]:
                        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
                        if new_req is not None:
                            new_req.headers.pop("Authorization", None)
                            new_req.unredirected_hdrs.pop("Authorization", None)
                        return new_req

                client = urllib.request.build_opener(_StripAuthOnRedirect())
                # CivitAI download endpoints require the token as a query
                # parameter rather than an Authorization header.
                parsed = urllib.parse.urlparse(src)
                _is_civitai = parsed.netloc == "civitai.com" or (parsed.netloc or "").endswith(".civitai.com")
                if token and _is_civitai:
                    qs = parsed.query + ("&" if parsed.query else "") + urllib.parse.urlencode({"token": token})
                    request_src = urllib.parse.urlunparse(parsed._replace(query=qs))
                    logger.info("_download_url_to_file: appending token (last 4: ...%s) to civitai URL", token[-4:])
                else:
                    request_src = src
                req = urllib.request.Request(request_src, method="GET")
                # Use a non-Python User-Agent; Cloudflare returns 403 for the default "Python-urllib/x.y" UA.
                req.add_header("User-Agent", "curl/8.7.1")
                if token and not _is_civitai:
                    req.add_header("Authorization", f"Bearer {token}")
                    logger.info("_download_url_to_file: using token (last 4: ...%s) for %s", token[-4:], src)
                with client.open(req, timeout=30) as resp:
                    size, sha = self._stream_to_file(resp, dst, max_bytes)
                with open(dst, "rb") as f:
                    head = f.read(512)
                mime = _infer_mime_type(src, head)
                return size, sha, mime
            except Exception as e:
                last_err = e
                if attempt >= max(1, attempts):
                    break
                sleep_s = min(10.0, 0.5 * (2 ** (attempt - 1))) + random.random() * 0.2
                time.sleep(sleep_s)
        raise RuntimeError(f"failed to download url: {last_err}")

    def _stream_to_file(self, src: Any, dst: str, max_bytes: int) -> Tuple[int, str]:
        tmp = f"{dst}.tmp-{os.getpid()}-{threading.get_ident()}-{random.randint(0, 1_000_000)}"
        total = 0
        h = hashlib.sha256()
        try:
            with open(tmp, "wb") as out:
                while True:
                    chunk = src.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise InputTooLargeError(size_bytes=total, max_bytes=max_bytes, source="input file")
                    h.update(chunk)
                    out.write(chunk)
            os.replace(tmp, dst)
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
        return total, h.hexdigest()

    def _start_registration_watchdog(self) -> None:
        timeout_s = int(getattr(self, "_register_timeout_s", 0) or 0)
        if timeout_s <= 0:
            return
        if self._registration_watchdog_thread and self._registration_watchdog_thread.is_alive():
            return
        self._registration_watchdog_thread = threading.Thread(
            target=self._registration_watchdog_loop,
            daemon=True,
            name="WorkerRegistrationWatchdog",
        )
        self._registration_watchdog_thread.start()

    def _registration_watchdog_loop(self, timeout_s: Optional[float] = None) -> None:
        try:
            t = float(timeout_s if timeout_s is not None else (getattr(self, "_register_timeout_s", 0) or 0))
        except Exception:
            t = 0.0
        if t <= 0:
            return
        if self._registered_event.wait(timeout=t):
            return
        if not getattr(self, "_running", False) or getattr(self, "_stop_event", threading.Event()).is_set():
            return

        self._startup_timeout_triggered = True
        payload = {
            "phase": "startup_timeout_unregistered",
            "timeout_s": t,
            "worker_id": str(getattr(self, "worker_id", "") or ""),
            "scheduler_addr": str(getattr(self, "scheduler_addr", "") or ""),
            "elapsed_ms": int(max(0.0, (time.monotonic() - float(getattr(self, "_process_started_monotonic", time.monotonic()))) * 1000.0)),
        }
        self._emit_startup_phase(
            "startup_timeout_unregistered",
            status="error",
            level=logging.ERROR,
            timeout_s=t,
            emit_worker_event=False,
        )
        self._emit_worker_event_bytes("", "worker.startup_timeout_unregistered", safe_json_bytes(payload))
        logger.error(
            "Worker did not register with scheduler before timeout (timeout_s=%.1f addr=%s); stopping worker.",
            t,
            self.scheduler_addr,
        )
        try:
            self._close_connection()
        except Exception:
            pass
        try:
            self._running = False
        except Exception:
            pass
        try:
            self._stop_event.set()
        except Exception:
            pass

    def connect(self) -> bool:
        """Connect to the scheduler.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
        self._emit_startup_phase("scheduler_connecting", status="starting", emit_worker_event=False)
        attempted: set[str] = set()
        while True:
            addr = None
            if self._leader_hint and self._leader_hint not in attempted:
                addr = self._leader_hint
                self._leader_hint = None
            else:
                for candidate in self._iter_scheduler_addrs():
                    if candidate not in attempted:
                        addr = candidate
                        break
            if not addr:
                break
            attempted.add(addr)
            self._set_scheduler_addr(addr)
            if self._connect_once():
                return True
        return False

    def _connect_once(self) -> bool:
        try:
            if self.use_tls:
                # Default system CA bundle — adequate for orchestrator endpoints
                # reachable via a trusted public certificate. Pass custom creds
                # explicitly via the orchestrator deployment if a private CA is
                # involved.
                creds = grpc.ssl_channel_credentials()
                self._channel = grpc.secure_channel(self.scheduler_addr, creds)
            else:
                self._channel = grpc.insecure_channel(self.scheduler_addr)

            interceptors = []
            if self.worker_jwt:
                interceptors.append(_AuthInterceptor(self.worker_jwt))

            if interceptors:
                self._channel = grpc.intercept_channel(self._channel, *interceptors)

            self._stub = pb_grpc.SchedulerWorkerServiceStub(self._channel)
            self._reset_outgoing_queues()
            # Defense in depth: the job-dispatch thread is started in run() at
            # boot, but if it died mid-flight (e.g. an unhandled exception, or
            # a future code path setting _stop_event in a way that surprised
            # this loop's gate) the receive thread would silently enqueue into
            # a dead _job_queue. Resurrect here on every reconnect.
            self._ensure_job_dispatch_thread()

            # Start the bidirectional stream
            request_iterator = self._outgoing_message_iterator(self._outgoing_queue)
            self._stream = self._stub.ConnectWorker(request_iterator)

            logger.info(f"Attempting to connect to scheduler at {self.scheduler_addr}...")

            # Send initial registration immediately
            self._register_worker(is_heartbeat=False)
            self._registered_event.set()
            self._emit_startup_phase("registered", status="ok", scheduler_addr=self.scheduler_addr)

            # Dedicated heartbeat channel + stream. Opens a second
            # HTTP/2 connection to the scheduler so heartbeats aren't subject
            # to flow control or Python-level queue head-of-line blocking from
            # the primary data stream. Scheduler treats heartbeats on any
            # stream as "bump LastActiveAt for the WorkerId carried in the
            # message" so two parallel streams per worker are safe.
            if self.use_tls:
                hb_creds = grpc.ssl_channel_credentials()
                self._heartbeat_channel = grpc.secure_channel(self.scheduler_addr, hb_creds)
            else:
                self._heartbeat_channel = grpc.insecure_channel(self.scheduler_addr)
            if interceptors:
                self._heartbeat_channel = grpc.intercept_channel(self._heartbeat_channel, *interceptors)
            self._heartbeat_stub = pb_grpc.SchedulerWorkerServiceStub(self._heartbeat_channel)
            self._heartbeat_stream = self._heartbeat_stub.ConnectWorker(
                self._heartbeat_outgoing_iterator(self._heartbeat_outgoing_queue)
            )
            self._heartbeat_drain_thread = threading.Thread(target=self._heartbeat_drain_loop, daemon=True)
            self._heartbeat_drain_thread.start()

            # Start the receive loop in a separate thread *after* stream is initiated
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            logger.info(f"Successfully connected to scheduler at {self.scheduler_addr}")
            self._emit_startup_phase("ready", status="ok", scheduler_addr=self.scheduler_addr)
            # Reset redirect chain on a successful registration so a future
            # owner change (legitimate failover) starts with a fresh budget.
            self._redirect_chain_count = 0
            if self._models_ready_on_connect:
                # Conversion workers: signal ModelsReady immediately since there is
                # no GPU model to pre-load. Without this the scheduler's ModelsReady
                # gate would block all job dispatch to this worker indefinitely.
                self._emit_worker_event_bytes("", "model.ready", safe_json_bytes({}))
                logger.info("emitted model.ready (WORKER_MODELS_READY_ON_CONNECT=true)")
            self._reconnect_count = 0
            return True

        except grpc.RpcError as e:
            # Access code() and details() methods for RpcError
            code = e.code() if hasattr(e, 'code') and callable(e.code) else grpc.StatusCode.UNKNOWN
            details = e.details() if hasattr(e, 'details') and callable(e.details) else str(e)
            leader = self._extract_leader_addr(details)
            if code == grpc.StatusCode.FAILED_PRECONDITION and self._is_protocol_incompatibility(details):
                self._handle_protocol_incompatibility(str(details))
            elif code == grpc.StatusCode.FAILED_PRECONDITION and leader:
                self._redirect_chain_count += 1
                if self._redirect_chain_count > self._max_redirect_chain:
                    logger.error(
                        "Scheduler redirect chain exceeded %d hops (last leader=%s); aborting reconnect loop",
                        self._max_redirect_chain,
                        leader,
                    )
                    self._running = False
                    self._stop_event.set()
                else:
                    logger.warning(
                        f"Scheduler returned not_leader for {self.scheduler_addr}; redirecting to {leader} (hop %d/%d)",
                        self._redirect_chain_count, self._max_redirect_chain,
                    )
                    self._leader_hint = leader
                    self._set_scheduler_addr(leader)
            else:
                logger.error(f"Failed to connect to scheduler: {code} - {details}")
            self._emit_startup_phase(
                "scheduler_connect_failed",
                status="error",
                level=logging.WARNING,
                emit_worker_event=False,
                grpc_code=str(code),
                grpc_details=str(details),
            )
            self._close_connection()
            return False
        except Exception as e:
            logger.exception(f"Unexpected error connecting to scheduler: {e}")
            self._emit_startup_phase(
                "scheduler_connect_failed",
                status="error",
                level=logging.ERROR,
                emit_worker_event=False,
                error_type=type(e).__name__,
                error=str(e),
            )
            self._close_connection()
            return False

    def _reset_outgoing_queues(self) -> None:
        """Give a new gRPC stream fresh queues before re-registering.

        The previous stream iterator may still be unwinding after a reconnect
        signal. Reusing the same queue lets that stale iterator consume the new
        registration, leaving the replacement stream to send only later events.
        """
        self._outgoing_queue = queue.Queue()
        self._heartbeat_outgoing_queue = queue.Queue()

    def _outgoing_message_iterator(self, outbound_queue: queue.Queue[Any]) -> Iterator[WorkerSchedulerMessage]:
        """Yields messages from the outgoing queue to send to the scheduler."""
        while not self._stop_event.is_set():
            try:
                # Block for a short time to allow stopping gracefully
                message = outbound_queue.get(timeout=0.1)
                # [worker-timing] for marco-polo latency profiling: when a
                # job_result hits the wire, log recv→handler_done→yielded
                # deltas so we can compare against the orch's [invoke-timing]
                # wait phase and localize the 248ms gap.
                try:
                    if message.WhichOneof("msg") == "job_result":
                        rid = message.job_result.request_id
                        t_recv = self._request_recv_times.pop(rid, None)
                        t_handler = self._request_handler_done_times.pop(rid, None)
                        if t_recv is not None:
                            t_yield = time.monotonic()
                            handler_ms = (
                                int((t_handler - t_recv) * 1000)
                                if t_handler is not None else -1
                            )
                            yield_ms = int((t_yield - t_recv) * 1000)
                            queue_ms = (
                                int((t_yield - t_handler) * 1000)
                                if t_handler is not None else -1
                            )
                            logger.info(
                                "[worker-timing] rid=%s recv_to_handler_done=%dms "
                                "handler_done_to_yield=%dms recv_to_yield=%dms",
                                rid, handler_ms, queue_ms, yield_ms,
                            )
                except Exception:
                    pass
                yield message
                # self._outgoing_queue.task_done() # Not needed if not joining queue
            except queue.Empty:
                continue
            except Exception as e:
                 if not self._stop_event.is_set():
                     logger.exception(f"Error in outgoing message iterator: {e}")
                     self._handle_connection_error()
                     break # Exit iterator on error

    def _heartbeat_outgoing_iterator(self, outbound_queue: queue.Queue[Any]) -> Iterator[WorkerSchedulerMessage]:
        """Yields heartbeat messages onto the dedicated heartbeat stream."""
        while not self._stop_event.is_set():
            try:
                message = outbound_queue.get(timeout=0.1)
                yield message
            except queue.Empty:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.exception(f"Error in heartbeat outgoing iterator: {e}")
                    # Don't call _handle_connection_error — losing the heartbeat
                    # stream alone shouldn't tear down the primary data stream.
                    break

    def _heartbeat_drain_loop(self) -> None:
        """Silently drains any server→client messages on the heartbeat stream.

        The scheduler's ConnectWorker handler is bidirectional; it may push
        messages back to the worker after seeing a WorkerRegistration. Since
        this stream is dedicated to heartbeats, anything the scheduler sends
        us here is either duplicative (a dispatch that will also arrive on the
        primary stream) or irrelevant. We must keep reading, though — gRPC's
        flow control blocks the upstream sends if the downstream reader stops.
        """
        stream = self._heartbeat_stream
        if stream is None:
            return
        try:
            for _ in stream:
                if self._stop_event.is_set():
                    return
                # discard — primary receive loop handles real traffic
        except Exception:
            # Stream closed or errored; nothing to recover here. Main loop
            # will notice via its own health checks.
            return

    def _send_heartbeat_message(self, message: WorkerSchedulerMessage) -> None:
        """Add a message to the dedicated heartbeat queue."""
        if self._running and not self._stop_event.is_set():
            try:
                self._heartbeat_outgoing_queue.put_nowait(message)
            except queue.Full:
                logger.error("Heartbeat outgoing queue is full. Heartbeat dropped!")

    def _heartbeat_loop(self) -> None:
        """Periodically sends heartbeat messages on the dedicated heartbeat stream."""
        while not self._stop_event.wait(HEARTBEAT_INTERVAL):
            try:
                self._register_worker(is_heartbeat=True)
                logger.debug("Sent heartbeat to scheduler")
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error sending heartbeat: {e}")
                    self._handle_connection_error()
                    break # Stop heartbeating on error

    def _collect_gpu_and_memory_info(self) -> Dict[str, Any]:
        """Gather CPU/memory/GPU/cuda/torch metadata for WorkerResources.

        Honors `WORKER_FAKE_GPU_*` env overrides (for CI and local dev without
        a real GPU), then augments with `detect_worker_capabilities()` which
        fills in gpu_sm and installed_libs.
        """
        mem = psutil.virtual_memory()
        info: Dict[str, Any] = {
            "cpu_cores": os.cpu_count() or 0,
            "memory_bytes": mem.total,
            "gpu_count": 0,
            "gpu_total_mem": 0,
            "gpu_used_mem": 0,
            "gpu_free_mem": 0,
            "gpu_name": "",
            "gpu_driver": "",
            "gpu_sm": "",
            "cuda_version": "",
            "torch_version": "",
            "installed_libs": [],
        }

        if torch and torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            if info["gpu_count"] > 0:
                try:
                    props = torch.cuda.get_device_properties(0)
                    info["gpu_total_mem"] = props.total_memory
                    info["gpu_used_mem"] = torch.cuda.memory_allocated(0)
                    info["gpu_name"] = props.name
                    info["gpu_driver"] = torch.version.cuda or ""
                    try:
                        free_mem, total_mem = torch.cuda.mem_get_info(0)
                        info["gpu_total_mem"] = total_mem
                        info["gpu_used_mem"] = total_mem - free_mem
                        info["gpu_free_mem"] = free_mem
                    except Exception:
                        pass
                    logger.debug(
                        f"GPU: {props.name}, VRAM total={info['gpu_total_mem']}, used={info['gpu_used_mem']}, cuda={torch.version.cuda}"
                    )
                except Exception as gpu_err:
                    logger.warning(f"Could not get GPU properties: {gpu_err}")

        if torch is not None:
            if not info["torch_version"]:
                info["torch_version"] = getattr(torch, "__version__", "") or ""
            if not info["cuda_version"]:
                info["cuda_version"] = getattr(torch.version, "cuda", "") or ""

        try:
            from .models.hub_policy import detect_worker_capabilities

            caps = detect_worker_capabilities()
            info["installed_libs"] = list(caps.installed_libs or [])
            if caps.gpu_sm:
                info["gpu_sm"] = str(int(caps.gpu_sm))
            if not info["cuda_version"]:
                info["cuda_version"] = str(caps.cuda_version or "")
            if not info["torch_version"]:
                info["torch_version"] = str(caps.torch_version or "")
        except Exception:
            pass

        return info

    @staticmethod
    def _parse_compute_capability_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("min")
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            if "." in raw:
                return float(raw)
            iv = int(float(raw))
            # WorkerResources.gpu_sm is commonly "90"/"120"; normalize to
            # CUDA compute capability 9.0/12.0 so decorator requirements use
            # the same vocabulary as torch.cuda.get_device_capability().
            if iv >= 100:
                return float(iv) / 10.0
            if iv >= 20:
                return float(iv) / 10.0
            return float(iv)
        except Exception:
            return None

    def _function_host_availability(
        self,
        function_name: str,
        req: Optional[Resources],
        gpu_info: Dict[str, Any],
    ) -> tuple[bool, Dict[str, Any]]:
        cfg: Dict[str, Any] = dict(msgspec.to_builtins(req)) if req is not None else {}
        # Resources.__post_init__ already rejected 'cpu'/'gpu' aliases (#326);
        # by the time we see ``req`` here it's strictly 'cuda', 'none', '', or
        # absent. Lowercase + strip is still defensive against case-only typos
        # surviving msgspec serialization round-trips.
        accelerator = str(cfg.get("accelerator", "") or "").strip().lower()
        requires_gpu = bool(cfg.get("requires_gpu") is True or accelerator == "cuda")
        if cfg.get("min_compute_capability"):
            requires_gpu = True
        if cfg.get("min_vram_gb") is not None:
            requires_gpu = True

        gpu_count = int(gpu_info.get("gpu_count") or 0)
        axes: Dict[str, str] = {
            "accelerator": accelerator or ("cuda" if requires_gpu else "none"),
            "gpu_count": str(gpu_count),
        }
        if requires_gpu and gpu_count <= 0:
            return False, {
                "function_name": function_name,
                "reason": "cuda_unavailable",
                "detail": "function requires CUDA but this worker reported no GPU",
                "axes": axes,
            }

        min_cc = self._parse_compute_capability_value(cfg.get("min_compute_capability"))
        if min_cc is not None:
            detected_cc = self._parse_compute_capability_value(gpu_info.get("gpu_sm"))
            axes["required_compute_capability"] = f"{min_cc:.1f}"
            if detected_cc is not None:
                axes["detected_compute_capability"] = f"{detected_cc:.1f}"
            if detected_cc is None or detected_cc < min_cc:
                return False, {
                    "function_name": function_name,
                    "reason": "compute_capability_unmet",
                    "detail": f"function requires compute capability {min_cc:.1f}+",
                    "axes": axes,
                }

        min_vram_gb = cfg.get("min_vram_gb")
        if min_vram_gb is not None:
            try:
                required_gb = float(min_vram_gb)
            except Exception:
                required_gb = 0.0
            available_gb = float(int(gpu_info.get("gpu_total_mem") or 0)) / float(1024 ** 3)
            axes["required_vram_gb"] = f"{required_gb:.3f}"
            axes["detected_vram_gb"] = f"{available_gb:.3f}"
            if required_gb > 0 and available_gb < required_gb:
                return False, {
                    "function_name": function_name,
                    "reason": "insufficient_vram",
                    "detail": f"function requires {required_gb:g} GiB VRAM",
                    "axes": axes,
                }

        required_libs = cfg.get("required_libraries") or cfg.get("required_libs") or []
        if isinstance(required_libs, str):
            required_libs = [required_libs]
        required = {str(x).strip().lower() for x in required_libs if str(x).strip()}
        if required:
            installed = {
                str(x).strip().lower()
                for x in (gpu_info.get("installed_libs") or [])
                if str(x).strip()
            }
            missing = sorted(required - installed)
            if missing:
                axes["missing_libraries"] = ",".join(missing)
                return False, {
                    "function_name": function_name,
                    "reason": "missing_optional_library",
                    "detail": "function requires optional libraries not installed on this worker: " + ", ".join(missing),
                    "axes": axes,
                }

        return True, {}

    def _refresh_worker_local_function_availability(self, gpu_info: Dict[str, Any]) -> None:
        unavailable: Dict[str, Dict[str, Any]] = {}
        for fn_name, req in self._discovered_resources.items():
            ok, status = self._function_host_availability(fn_name, req, gpu_info)
            if not ok:
                unavailable[fn_name] = status
        self._worker_local_unavailable_functions_by_name = unavailable
        if unavailable:
            logger.info(
                "Worker-local unavailable functions: %s",
                sorted(unavailable.keys()),
            )

    def _collect_model_inventory(self) -> tuple[List[str], List[str], List[str], bool]:
        """Return `(vram_models, disk_models, downloading_models, supports_model_loading)`.

        Prefers `self._model_manager` for VRAM info; falls back to the newer
        `self._model_cache` when the legacy manager isn't wired. Disk and
        downloading-model lists come from the model cache when present.
        """
        vram_models: List[str] = []
        disk_models: List[str] = []
        downloading_models: List[str] = []
        supports_model_loading_flag = False

        if self._model_manager:
            vram_models = self._model_manager.get_vram_loaded_models()
            supports_model_loading_flag = True
        elif self._model_cache:
            vram_models = self._model_cache.get_vram_models()
            supports_model_loading_flag = True

        if self._model_cache:
            disk_models = self._model_cache.get_disk_models()
            stats = self._model_cache.get_stats()
            downloading_models = stats.downloading_models
            logger.debug(
                f"Model cache: vram={len(vram_models)}, disk={len(disk_models)}, "
                f"downloading={len(downloading_models)}"
            )

        return vram_models, disk_models, downloading_models, supports_model_loading_flag

    # #321: _build_function_schemas removed — WorkerResources.function_schemas
    # was never read by the orchestrator. Schemas are served by tensorhub HTTP.

    def _all_declared_function_names(self) -> List[str]:
        """Every function name the worker has discovered, irrespective of
        runnability. Source of truth for "which functions can possibly be
        loading?" — the loading set is a subset of this list.
        """
        return list(
            dict.fromkeys(
                list(self._request_specs.keys())
                + list(self._training_specs.keys())
                + list(self._batched_specs.keys())
                + list(self._serial_class_specs.keys())
            )
        )

    def _bound_model_refs_for_function(self, function_name: str) -> List[str]:
        """Canonicalized model refs bound to a single function via the
        baked manifest. A function's bound set = fixed (global) refs +
        per-function payload refs. The values are canonicalized to match
        the keying used by ``ModelCache``.
        """
        name = (function_name or "").strip()
        if not name:
            return []
        refs: set[str] = set()
        for v in (self._fixed_model_id_by_key or {}).values():
            s = str(v or "").strip()
            if not s:
                continue
            try:
                refs.add(_canonicalize_model_ref_string(s))
            except Exception:
                refs.add(s)
        per_fn = (self._payload_model_id_by_key_by_function or {}).get(name) or {}
        for v in per_fn.values():
            s = str(v or "").strip()
            if not s:
                continue
            try:
                refs.add(_canonicalize_model_ref_string(s))
            except Exception:
                refs.add(s)
        return sorted(refs)

    def _loading_function_names(self) -> List[str]:
        """Return the sorted list of function names whose bound model refs
        are still being downloaded (i.e. at least one bound ref is in the
        model cache's ``downloading_models`` set).

        Functions whose bound refs are all on disk (or that have no bound
        refs at all — e.g. marco-polo) are NOT loading. Functions that are
        also disabled or host-unavailable are excluded — those land in
        ``_function_unavailable_entry`` via the existing channels.
        """
        # Fast path: no model cache, no concept of "downloading".
        cache = getattr(self, "_model_cache", None)
        if cache is None:
            return []
        try:
            stats = cache.get_stats()
        except Exception:
            return []
        downloading = set(stats.downloading_models or [])
        if not downloading:
            return []
        out: List[str] = []
        for fn_name in self._all_declared_function_names():
            # Skip functions that are already gated out for a non-loading
            # reason (host-unavailable / scheduler-disabled). They aren't
            # "loading" — they're permanently or transiently unavailable.
            if (
                getattr(self, "_disabled_functions_by_name", {}).get(fn_name)
                or getattr(self, "_worker_local_unavailable_functions_by_name", {}).get(fn_name)
            ):
                continue
            bound = self._bound_model_refs_for_function(fn_name)
            if not bound:
                # Functions with no bound models (string-compare style) are
                # never "loading" — they're ready as soon as setup() returns.
                continue
            if any(ref in downloading for ref in bound):
                out.append(fn_name)
        out.sort()
        return out

    def _available_function_names(self) -> List[str]:
        names = self._all_declared_function_names()
        return [name for name in names if self.is_function_runnable(name)]

    def _register_worker(self, is_heartbeat: bool = False) -> None:
        """Create and send a registration/heartbeat message."""
        try:
            gpu_info = self._collect_gpu_and_memory_info()
            self._refresh_worker_local_function_availability(gpu_info)
            vram_models, disk_models, downloading_models, supports_model_loading_flag = (
                self._collect_model_inventory()
            )
            _ = downloading_models  # surfaced via `loading_functions` below;
            # the raw list is still consumed for logging in _collect_model_inventory.
            # Compute loading_functions BEFORE building available_functions so
            # the two sets are derived from the same model-cache snapshot
            # within one heartbeat tick (gen-orchestrator #341). Routing rule
            # on the scheduler side:
            #   eligible iff (fn in available_functions) AND (fn not in loading_functions)
            # available_functions already excludes loading functions because
            # _function_unavailable_entry consults the same loading set.
            loading_functions = self._loading_function_names()
            # #321: function_schemas, cpu_cores, memory_bytes,
            # gpu_memory_used_bytes, gpu_driver, cuda_version, torch_version,
            # supports_model_loading, tensorrt_version, onnxruntime_version,
            # build_timestamp were removed — orchestrator never read them.
            resources = pb.WorkerResources(
                # Worker identity is carried in the worker-connect JWT claims when auth is enabled.
                # When auth is disabled (dev mode), send worker_id directly so the orchestrator
                # can identify this worker without JWT claims.
                worker_id=self.worker_id,
                release_id=self.release_id,
                # owner is provided per-request via JobExecutionRequest.
                runpod_pod_id=self.runpod_pod_id,
                gpu_is_busy=self._get_gpu_busy_status(),
                gpu_count=gpu_info["gpu_count"],
                gpu_memory_bytes=gpu_info["gpu_total_mem"],
                gpu_memory_free_bytes=gpu_info["gpu_free_mem"],
                gpu_name=gpu_info["gpu_name"],
                gpu_sm=gpu_info["gpu_sm"],
                installed_libs=gpu_info["installed_libs"],
                available_functions=self._available_function_names(),
                loading_functions=loading_functions,
                vram_models=vram_models,   # Models in VRAM (hot)
                disk_models=disk_models,   # Models on disk (warm)
            )
            registration = pb.WorkerRegistration(
                resources=resources,
                is_heartbeat=is_heartbeat,
                protocol_major=WIRE_PROTOCOL_MAJOR,
                protocol_minor=WIRE_PROTOCOL_MINOR,
            )
            message = pb.WorkerSchedulerMessage(worker_registration=registration)
            # logger.info(f"DEBUG: Preparing to send registration. Resource object: {resources}")
            # logger.info(f"DEBUG: Value being sent for runpod_pod_id: '{resources.runpod_pod_id}'")
            if is_heartbeat:
                # Route heartbeats on the dedicated heartbeat stream.
                # The primary data stream can back up behind huge job_output
                # chunks or long GIL-held tenant code; the heartbeat stream is
                # kept idle so every 10 s tick gets through.
                self._send_heartbeat_message(message)
            else:
                self._send_message(message)

            # Best-effort: report disk inventory + volume identity for NFS-aware scheduling/debug.
            # This is intentionally redundant with WorkerResources.disk_models but adds the missing
            # "which shared volume is this?" dimension.
            try:
                vol = self._shared_disk_volume_info()
                if vol and disk_models:
                    import hashlib as _hashlib

                    # Only emit when inventory changes to avoid spamming.
                    h = _hashlib.sha256()
                    h.update((vol.get("disk_volume_key") or "").encode("utf-8"))
                    for mid in sorted(set(str(x) for x in disk_models)):
                        h.update(b"\0")
                        h.update(mid.encode("utf-8", errors="ignore"))
                    inv_hash = h.hexdigest()
                    if inv_hash != self._last_disk_inventory_hash:
                        self._last_disk_inventory_hash = inv_hash
                        payload = dict(vol)
                        payload["disk_models"] = sorted(set(str(x) for x in disk_models))
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(
                                    request_id="",
                                    event_type="models.disk_inventory",
                                    payload_json=json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
                                )
                            )
                        )
            except Exception:
                pass

            # Best-effort: publish function capability/profile hints so scheduler routing/planner
            # can consume endpoint-level batching and stage traits.
            try:
                self._emit_function_capabilities_event()
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to create or send registration/heartbeat: {e}")

    # #321: _runtime_batching_cfg_for_function + _handle_runtime_batching_config_cmd
    # removed. Proto messages (RuntimeBatchingConfigCommand/Result) were deleted
    # because the orchestrator-side producer never landed; the helper, state,
    # and the RequestContext.runtime_batching_config / preferred_batch_size /
    # prefetch_depth API went with them. Reintroduce when an SLA planner ships
    # a real producer.

    def _emit_function_capabilities_event(self) -> None:
        functions: List[Dict[str, Any]] = []
        all_names = dict.fromkeys(
            list(self._discovered_resources.keys())
            + list(self._request_specs.keys())
            + list(self._training_specs.keys())
        )
        for fn_name in all_names:
            req = self._discovered_resources.get(fn_name)
            caps: Dict[str, Any] = dict(msgspec.to_builtins(req)) if req is not None else {}
            spec = self._request_specs.get(fn_name)
            if spec is not None:
                caps["output_mode"] = spec.output_mode
            caps["function_name"] = fn_name
            unavailable = self._function_unavailable_entry(fn_name)
            caps["available"] = unavailable is None
            if unavailable is not None:
                caps["unavailable_reason"] = str(unavailable.get("reason", "") or "")
                caps["unavailable_detail"] = str(unavailable.get("detail", "") or "")
                caps["unavailable_axes"] = dict(unavailable.get("axes") or {})
            functions.append(caps)
        if not functions:
            return
        payload = {"functions": sorted(functions, key=lambda x: str(x.get("function_name") or ""))}
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sig = hashlib.sha256(raw).hexdigest()
        if sig == self._last_function_capabilities_hash:
            return
        self._last_function_capabilities_hash = sig
        # #321: typed signal. Previously emitted via worker_event with
        # event_type="worker.function_capabilities" — the orchestrator matched
        # "function_capabilities" (no prefix), so capabilities were silently
        # dropped. Typed dispatch eliminates the string-mismatch class of bug.
        self._send_message(
            pb.WorkerSchedulerMessage(
                worker_function_capabilities=pb.WorkerFunctionCapabilitiesSignal(
                    capabilities_json=raw,
                    signature=sig,
                )
            )
        )

    def run(self) -> None:
        """Run the worker, connecting to the scheduler and processing tasks."""
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        self._startup_timeout_triggered = False
        self._registered_event.clear()
        self._stop_event.clear()
        self._reconnect_count = 0 # Reset reconnect count on new run
        self._ensure_job_dispatch_thread()
        self._draining = False
        # Self-exit cap (gen-orchestrator #317): track the last time we
        # were successfully connected. After settings.worker_disconnected_timeout_s
        # of continuous failure to reconnect, exit cleanly so the container
        # is reaped. Stack-shutdown ergonomics without a separate
        # shutdown-coordination protocol.
        timeout_s = int(self._settings.worker_disconnected_timeout_s or 0)
        self._last_successful_connect_at = time.monotonic()
        self._start_registration_watchdog()

        try:
            while self._running and not self._stop_event.is_set():
                self._reconnect_count += 1
                logger.info(f"Connection attempt {self._reconnect_count}...")
                if self.connect():
                    # Successfully connected, wait for stop signal or disconnection
                    self._last_successful_connect_at = time.monotonic()
                    logger.info("Connection successful. Worker running.")
                    self._stop_event.wait() # Wait here until stopped or disconnected
                    logger.info("Worker run loop received stop/disconnect signal.")
                    # If stopped normally (self.stop() called), _running will be False
                    # If disconnected, connect() failed, threads stopped, _handle_connection_error called _stop_event.set()
                else:
                    # Connection failed
                    if self.max_reconnect_attempts > 0 and self._reconnect_count >= self.max_reconnect_attempts:
                        logger.error("Failed to connect after maximum attempts. Stopping worker.")
                        self._running = False # Ensure loop terminates
                        break

                    # Self-exit when disconnected too long. Bounded version
                    # of the previously-infinite reconnect loop — keeps the
                    # container from running forever if the orchestrator is
                    # gone for good (stack shutdown, machine power-off).
                    if timeout_s > 0:
                        elapsed = time.monotonic() - self._last_successful_connect_at
                        if elapsed >= timeout_s:
                            logger.error(
                                "Worker disconnected for %.1fs without a successful reconnect "
                                "(WORKER_DISCONNECTED_TIMEOUT_S=%d). Exiting so container is reaped.",
                                elapsed, timeout_s,
                            )
                            self._running = False
                            break

                    if self._running and not self._stop_event.is_set():
                        delay = self._next_reconnect_delay(self._reconnect_count)
                        logger.info(f"Connection attempt {self._reconnect_count} failed. Retrying in {delay:.2f} seconds...")
                        # Wait for delay, but break if stop event is set during wait
                        if self._stop_event.wait(delay):
                            logger.info("Stop requested during reconnect delay.")
                            break # Exit if stopped while waiting
                # After a failed attempt or disconnect, clear stop event for next retry
                if self._running:
                    self._stop_event.clear()
        except Exception as e:
            self._emit_worker_fatal("run_loop", e, exit_code=1)
            raise
        finally:
            # Cleanup after loop exits (either max attempts reached or manual stop)
            self.stop()
            if self._registration_watchdog_thread and self._registration_watchdog_thread.is_alive():
                self._registration_watchdog_thread.join(timeout=1.0)

        if self._startup_timeout_triggered:
            raise RuntimeError("startup_timeout_unregistered")

    def _handle_interrupt(self, sig: int, frame: Optional[Any]) -> None:
        """Handle interrupt signal (Ctrl+C)."""
        logger.info(f"Received signal {sig}, shutting down gracefully.")
        self.stop()

    def stop(self) -> None:
        """Stop the worker and clean up resources."""
        if not self._running and not self._stop_event.is_set(): # Check if already stopped or stopping
            # Avoid multiple stop calls piling up
            # logger.debug("Stop called but worker already stopped or stopping.")
            return

        logger.info("Stopping worker...")
        self._draining = True
        self._running = False # Signal loops to stop
        self._stop_event.set() # Wake up any waiting threads

        # #273: gracefully shut down BatchedWorker engines + instances
        # before tearing down the connection. shutdown_done flag guards
        # double-calls if _handle_worker_drain_cmd already invoked this.
        try:
            self._shutdown_batched_workers()
        except Exception:
            logger.exception("BatchedWorker shutdown during stop() raised; continuing")

        # #322/#328: same idempotent shutdown for SerialWorker classes.
        try:
            self._shutdown_serial_workers()
        except Exception:
            logger.exception("SerialWorker shutdown during stop() raised; continuing")

        # Cancel any active requests
        active_request_ids = []
        if self._drain_timeout_seconds > 0:
            deadline = time.time() + self._drain_timeout_seconds
            while time.time() < deadline:
                with self._active_requests_lock:
                    remaining = len(self._active_requests)
                if remaining == 0:
                    break
                time.sleep(0.2)

        with self._active_requests_lock:
            active_request_ids = list(self._active_requests.keys())
            for request_id in active_request_ids:
                ctx = self._active_requests.get(request_id)
                if ctx:
                    logger.debug(f"Cancelling active request {request_id} during stop.")
                    ctx.cancel()
            # Don't clear here, allow _execute_function to finish and remove

        # Wait for threads (give them a chance to finish)
        # Stop heartbeat first
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
             logger.debug("Joining heartbeat thread...")
             self._heartbeat_thread.join(timeout=1.0)

        # The outgoing iterator might be blocked on queue.get, stop_event wakes it

        # Close the gRPC connection (this might interrupt the receive loop)
        self._close_connection()

        # Wait for receive thread
        if self._receive_thread and self._receive_thread.is_alive():
            logger.debug("Joining receive thread...")
            self._receive_thread.join(timeout=2.0)

        # Wait for heartbeat-stream drain thread
        if self._heartbeat_drain_thread and self._heartbeat_drain_thread.is_alive():
            logger.debug("Joining heartbeat drain thread...")
            self._heartbeat_drain_thread.join(timeout=1.0)

        if self._registration_watchdog_thread and self._registration_watchdog_thread.is_alive():
            logger.debug("Joining registration watchdog thread...")
            self._registration_watchdog_thread.join(timeout=1.0)

        # Clear outgoing queue after threads are stopped
        logger.debug("Clearing outgoing message queue...")
        while not self._outgoing_queue.empty():
            try:
                self._outgoing_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Worker stopped.")
        # Reset stop event in case run() is called again
        self._stop_event.clear()

    def _close_connection(self) -> None:
        """Close the gRPC channel and reset state."""
        if self._stream:
             try:
                  # Attempt to cancel the stream from the client side
                  # This might help the server side release resources quicker
                  # Note: Behavior might vary depending on server implementation
                  if hasattr(self._stream, 'cancel') and callable(self._stream.cancel):
                     self._stream.cancel()
                     logger.debug("gRPC stream cancelled.")
             except Exception as e:
                  logger.warning(f"Error cancelling gRPC stream: {e}")
        self._stream = None

        if self._channel:
            try:
                self._channel.close()
                logger.debug("gRPC channel closed.")
            except Exception as e:
                 logger.error(f"Error closing gRPC channel: {e}")
        self._channel = None
        self._stub = None

        # Tear down the dedicated heartbeat channel too.
        if self._heartbeat_stream:
            try:
                if hasattr(self._heartbeat_stream, 'cancel') and callable(self._heartbeat_stream.cancel):
                    self._heartbeat_stream.cancel()
            except Exception as e:
                logger.warning(f"Error cancelling heartbeat stream: {e}")
        self._heartbeat_stream = None
        if self._heartbeat_channel:
            try:
                self._heartbeat_channel.close()
            except Exception as e:
                logger.warning(f"Error closing heartbeat channel: {e}")
        self._heartbeat_channel = None
        self._heartbeat_stub = None
        # Drain any leftover heartbeat messages so they don't resurface on reconnect.
        while not self._heartbeat_outgoing_queue.empty():
            try:
                self._heartbeat_outgoing_queue.get_nowait()
            except queue.Empty:
                break


    def _receive_loop(self) -> None:
        """Loop to receive messages from the scheduler via the stream."""
        logger.info("Receive loop started.")
        try:
            if not self._stream:
                 logger.error("Receive loop started without a valid stream.")
                 # Don't call _handle_connection_error here, connect should have failed
                 return

            for message in self._stream:
                # Check stop event *before* processing
                if self._stop_event.is_set():
                    logger.debug("Stop event set during iteration, exiting receive loop.")
                    break
                try:
                    self._process_message(message)
                except Exception as e:
                    # Log errors processing individual messages but continue loop
                    logger.exception(f"Error processing message: {e}")

        except grpc.RpcError as e:
            # RpcError indicates a problem with the gRPC connection itself
            code = e.code() if hasattr(e, 'code') and callable(e.code) else grpc.StatusCode.UNKNOWN
            details = e.details() if hasattr(e, 'details') and callable(e.details) else str(e)

            if self._stop_event.is_set():
                 # If stopping, cancellation is expected
                 if code == grpc.StatusCode.CANCELLED:
                     logger.info("gRPC stream cancelled gracefully during shutdown.")
                 else:
                     logger.warning(f"gRPC error during shutdown: {code} - {details}")
            elif code == grpc.StatusCode.FAILED_PRECONDITION:
                leader = self._extract_leader_addr(details)
                if self._is_protocol_incompatibility(details):
                    self._handle_protocol_incompatibility(str(details))
                elif leader:
                    self._redirect_chain_count += 1
                    if self._redirect_chain_count > self._max_redirect_chain:
                        logger.error(
                            "Scheduler redirect chain exceeded %d hops (last leader=%s); aborting reconnect loop",
                            self._max_redirect_chain,
                            leader,
                        )
                        self._running = False
                        self._stop_event.set()
                    else:
                        logger.warning(
                            f"Scheduler redirect received; reconnecting to leader at {leader} (hop %d/%d)",
                            self._redirect_chain_count, self._max_redirect_chain,
                        )
                        self._leader_hint = leader
                        self._set_scheduler_addr(leader)
                self._handle_connection_error()
            elif code == grpc.StatusCode.CANCELLED:
                logger.warning("gRPC stream unexpectedly cancelled by server or network.")
                self._handle_connection_error()
            elif code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL):
                 logger.warning(f"gRPC connection lost ({code}). Attempting reconnect.")
                 self._handle_connection_error()
            else:
                 logger.error(f"Unhandled gRPC error in receive loop: {code} - {details}")
                 self._handle_connection_error() # Attempt reconnect on unknown errors too
        except Exception as e:
            # Catch-all for non-gRPC errors in the loop
            if not self._stop_event.is_set():
                logger.exception(f"Unexpected error in receive loop: {e}")
                self._handle_connection_error() # Attempt reconnect
        finally:
             logger.info("Receive loop finished.")

    def _handle_connection_error(self) -> None:
         """Handles steps needed when a connection error occurs during run."""
         if self._running and not self._stop_event.is_set():
             logger.warning("Connection error detected. Signaling main loop to reconnect...")
             self._close_connection() # Ensure resources are closed before reconnect attempt
             self._stop_event.set() # Signal run loop to attempt reconnection
         # else: # Already stopping or stopped
             # logger.debug("Connection error detected but worker is already stopping.")


    def _process_message(self, message: WorkerSchedulerMessage) -> None:
        """Process a single message received from the scheduler."""
        msg_type = message.WhichOneof('msg')
        # logger.debug(f"Received message of type: {msg_type}")

        if msg_type == 'job_request':
            self._handle_job_request(message.job_request)
        # #321: batch_job_request removed.
        elif msg_type == 'load_model_cmd':
            self._handle_load_model_cmd(message.load_model_cmd)
        elif msg_type == 'download_model_cmd':
            # #339: orchestrator-driven cold-fetch for per-request model
            # affinity. Distinct from load_model_cmd (disk -> VRAM); this
            # is upstream -> disk.
            self._handle_download_model_cmd(message.download_model_cmd)
        elif msg_type == 'unload_model_cmd':
            self._handle_unload_model_cmd(message.unload_model_cmd)
        elif msg_type == 'interrupt_job_cmd':
            cmd = message.interrupt_job_cmd
            request_id = cmd.request_id
            item_ids = [str(x).strip() for x in list(getattr(cmd, "item_ids", []) or []) if str(x).strip()]
            cancel_queued_only = bool(getattr(cmd, "cancel_queued_only", False))
            self._handle_interrupt_request(request_id, item_ids=item_ids, cancel_queued_only=cancel_queued_only)
        elif msg_type == "worker_drain_cmd":
            self._handle_worker_drain_cmd(message.worker_drain_cmd)
        # #321: runtime_batching_config_cmd removed — producer never landed.
        # #321: inbound worker_event (JWT-rotation channel) removed — speculative;
        # rotation is done by force-disconnecting the worker so it reconnects with a new token.
        elif msg_type == 'endpoint_config':
            cfg = message.endpoint_config
            resolved_by_flavor = dict(getattr(cfg, "resolved_repos_by_ref", {}) or {})
            required_ref_list = list(getattr(cfg, "required_flavor_refs", []) or [])
            logger.info(
                "Received EndpointConfig (supported=%d required=%d resolved=%d)",
                len(cfg.supported_repo_refs),
                len(required_ref_list),
                len(resolved_by_flavor),
            )

            # Optional: authoritative model keyspace updates (runtime-editable).
            # - repo_ref_by_key: global fixed keyspace
            # - models_by_function: per-function payload keyspaces
            has_keyspace_update = False
            try:
                rbk = dict(getattr(cfg, "repo_ref_by_key", {}) or {})
                mbf = dict(getattr(cfg, "models_by_function", {}) or {})
                if rbk or mbf:
                    has_keyspace_update = True
                    new_fixed_by_key: Dict[str, str] = {}
                    new_fixed_spec_by_key: Dict[str, Dict[str, Any]] = {}
                    for k, v in rbk.items():
                        key = str(k).strip()
                        if not key:
                            continue
                        ref = _canonicalize_model_ref_string(str(v or ""))
                        if not ref:
                            continue
                        new_fixed_by_key[key] = ref
                        # Scheduler updates only carry the ref — preserve the
                        # attribute selectors (dtype/file_type/file_layout)
                        # that came from the baked endpoint.lock manifest so
                        # we don't lose them on every EndpointConfig tick.
                        prior = self._fixed_model_spec_by_key.get(key) or {}
                        new_fixed_spec_by_key[key] = {
                            "ref": ref,
                            "dtypes": list(prior.get("dtypes") or []),
                            "file_type": list(prior.get("file_type") or []),
                            "file_layout": list(prior.get("file_layout") or []),
                        }

                    new_payload_by_fn: Dict[str, Dict[str, str]] = {}
                    new_payload_spec_by_fn: Dict[str, Dict[str, Dict[str, Any]]] = {}
                    for fn_name, mbk in mbf.items():
                        if not fn_name or mbk is None:
                            continue
                        models = dict(getattr(mbk, "models", {}) or {})
                        if not models:
                            continue
                        out: Dict[str, str] = {}
                        out_spec: Dict[str, Dict[str, Any]] = {}
                        for k, spec in models.items():
                            key = str(k).strip()
                            if not key or spec is None:
                                continue
                            ref = _canonicalize_model_ref_string(str(getattr(spec, "ref", "") or ""))
                            if not ref:
                                continue
                            dtypes = []
                            try:
                                dtypes = [str(x).strip() for x in list(getattr(spec, "dtypes", []) or []) if str(x).strip()]
                            except Exception:
                                dtypes = []
                            out[key] = ref
                            out_spec[key] = {"ref": ref, "dtypes": dtypes}
                        if out:
                            new_payload_by_fn[str(fn_name)] = out
                        if out_spec:
                            new_payload_spec_by_fn[str(fn_name)] = out_spec
                    if new_fixed_by_key:
                        self._fixed_model_id_by_key = new_fixed_by_key
                        logger.info(
                            "EndpointConfig fixed model map updated keys=%s refs=%s",
                            sorted(new_fixed_by_key.keys()),
                            sorted(new_fixed_by_key.values()),
                        )
                    if new_fixed_spec_by_key:
                        self._fixed_model_spec_by_key = new_fixed_spec_by_key
                    if new_payload_by_fn:
                        self._payload_model_id_by_key_by_function = new_payload_by_fn
                        logger.info(
                            "EndpointConfig function model maps updated functions=%s",
                            sorted(new_payload_by_fn.keys()),
                        )
                    if new_payload_spec_by_fn:
                        self._payload_model_spec_by_key_by_function = new_payload_spec_by_fn
            except Exception:
                has_keyspace_update = False

            supported_from_sched = [
                _canonicalize_model_ref_string(str(v)) for v in list(cfg.supported_repo_refs)
            ]
            # Scheduler may narrow the set of models this worker should advertise/prefetch, but must
            # never widen beyond tenant-declared scope (baked manifest), unless it supplies an
            # explicit runtime model keyspace update.
            if self._manifest_allowed_model_ids is not None and not has_keyspace_update:
                supported_set = {s for s in supported_from_sched if s and s != "*"}
                if supported_set:
                    supported_set &= self._manifest_allowed_model_ids
                    self._supported_model_ids_from_scheduler = sorted(supported_set)
                else:
                    # Empty/"*" means "no extra restriction"; keep manifest scope.
                    self._supported_model_ids_from_scheduler = sorted(self._manifest_allowed_model_ids)
            else:
                self._supported_model_ids_from_scheduler = supported_from_sched

            # Baseline resolved manifests for Cozy model downloads (issue #66/#238).
            self._resolved_repos_by_id_baseline = self._canonicalize_resolved_repos_map(
                resolved_by_flavor
            )

            # Allowlist semantics:
            # - empty list means "no allowlist enforced" (allow all), for dev/seeded deployments
            # - "*" means wildcard allow-all (used in some seed manifests)
            allow = set(self._supported_model_ids_from_scheduler or [])
            if not allow or "*" in allow:
                if has_keyspace_update:
                    # When scheduler supplies an explicit keyspace update, prefer that as the
                    # runtime enforcement scope.
                    scope: set[str] = set()
                    scope.update((self._fixed_model_id_by_key or {}).values())
                    for m in (self._payload_model_id_by_key_by_function or {}).values():
                        scope.update(m.values())
                    self._release_allowed_model_ids = scope or None
                else:
                    # If we have a baked manifest scope, never allow widening beyond it.
                    self._release_allowed_model_ids = self._manifest_allowed_model_ids
            else:
                if self._manifest_allowed_model_ids is not None and not has_keyspace_update:
                    allow &= self._manifest_allowed_model_ids
                self._release_allowed_model_ids = allow or self._manifest_allowed_model_ids
            self._required_flavor_refs_from_scheduler = [
                _canonicalize_model_ref_string(str(v)) for v in required_ref_list
            ]

            # Honor orchestrator-reported degradation state:
            #   - disabled_functions: per-function entries whose FIXED refs
            #     failed terminally. Worker MUST skip prefetch for models
            #     that belong ONLY to disabled functions (nothing else
            #     needs them) AND must NOT advertise those functions in
            #     its registration spec so the orchestrator won't route
            #     invocations to this worker.
            #   - ref_availability_by_function: per-PAYLOAD-key status
            #     map on otherwise-runnable functions. Worker stores this
            #     on a per-function dispatch map so if a request with a
            #     broken key slips past the orchestrator gate (race
            #     during refresh), it can reject locally with the same
            #     424 shape.
            # Fixed-model disables and payload-ref availability have different
            # ownership: fixed keys are release config, payload refs are caller
            # supplied.
            self._disabled_functions_by_name = {}
            for df in list(getattr(cfg, "disabled_functions", []) or []):
                name = str(getattr(df, "function_name", "") or "").strip()
                if not name:
                    continue
                self._disabled_functions_by_name[name] = {
                    "function_name": name,
                    "model_key": str(getattr(df, "model_key", "") or ""),
                    "ref": str(getattr(df, "ref", "") or ""),
                    "reason": str(getattr(df, "reason", "") or ""),
                    "detail": str(getattr(df, "detail", "") or ""),
                    "detected_at_unix": int(getattr(df, "detected_at_unix", 0) or 0),
                }
            if self._disabled_functions_by_name:
                logger.info(
                    "EndpointConfig disabled_functions: %s",
                    sorted(self._disabled_functions_by_name.keys()),
                )

            self._payload_ref_availability_by_function = {}
            for fn_name, fn_avail in dict(
                getattr(cfg, "ref_availability_by_function", {}) or {}
            ).items():
                name = str(fn_name or "").strip()
                if not name or fn_avail is None:
                    continue
                by_key = dict(getattr(fn_avail, "by_model_key", {}) or {})
                out = {}
                for k, rs in by_key.items():
                    key = str(k or "").strip()
                    if not key or rs is None:
                        continue
                    out[key] = {
                        "ref": str(getattr(rs, "ref", "") or ""),
                        "status": str(getattr(rs, "status", "") or ""),
                        "reason": str(getattr(rs, "reason", "") or ""),
                        "detail": str(getattr(rs, "detail", "") or ""),
                        "last_checked_unix": int(getattr(rs, "last_checked_unix", 0) or 0),
                    }
                if out:
                    self._payload_ref_availability_by_function[name] = out

            # Filter out refs that belong ONLY to disabled functions from
            # the prefetch set. Refs referenced by at least one enabled
            # function still prefetch (they serve that enabled path).
            prefetch_refs = self._filter_prefetch_for_disabled_functions(
                self._required_flavor_refs_from_scheduler or []
            )

            # Start background prefetch regardless of model manager; disk readiness is useful even
            # for lightweight workers and enables cache-aware routing.
            self._start_startup_prefetch(prefetch_refs)

            if self._model_manager:
                # Legacy/model-manager-specific config hook (may load/prep models).
                self._model_init_done_event.clear() # Clear before starting new init
                model_init_thread = threading.Thread(target=self._process_release_config_async_wrapper, daemon=True)
                model_init_thread.start()
            else:
                self._model_init_done_event.set()
        elif msg_type is None:
             logger.warning("Received empty message from scheduler.")
        else:
            logger.warning(f"Received unhandled message type: {msg_type}")

    def _active_request_count(self) -> int:
        with self._active_requests_lock:
            return len(self._active_requests)

    def _emit_worker_drain_result(self, reason: str, status: str) -> None:
        active = self._active_request_count()
        self._send_message(
            pb.WorkerSchedulerMessage(
                worker_drain_result=pb.WorkerDrainResult(
                    worker_id=self.worker_id,
                    reason=reason,
                    status=status,
                    active_requests=active,
                    emitted_at_unix_ms=int(time.time() * 1000),
                )
            )
        )
        self._emit_worker_event_bytes(
            "",
            "worker.drain.status",
            safe_json_bytes({"reason": reason, "status": status, "active_requests": active}),
        )

    def _handle_worker_drain_cmd(self, cmd: Any) -> None:
        reason = str(getattr(cmd, "reason", "") or "scheduler_drain")
        deadline_ms = int(getattr(cmd, "deadline_unix_ms", 0) or 0)
        terminate_after_deadline = bool(getattr(cmd, "terminate_after_deadline", True))
        now_ms = int(time.time() * 1000)
        drain_seconds = max(0.0, (deadline_ms - now_ms) / 1000.0) if deadline_ms > 0 else float(self._drain_timeout_seconds or 0)

        logger.info(
            "Received worker drain command reason=%s deadline_ms=%s active_requests=%d",
            reason,
            deadline_ms,
            self._active_request_count(),
        )
        self._draining = True
        self._emit_worker_drain_result(reason, "draining")

        def _drain_then_stop() -> None:
            deadline = time.time() + drain_seconds
            while drain_seconds > 0 and time.time() < deadline:
                if self._active_request_count() == 0:
                    break
                time.sleep(0.2)
            final_status = "drained" if self._active_request_count() == 0 else "deadline_exceeded"
            # #273: shut down each BatchedWorker instance + engine exactly once.
            # Engines hold open subprocesses + GPU memory; not calling
            # shutdown leaks both. shutdown() runs on the BatchedWorker loop.
            self._shutdown_batched_workers()
            # #322/#328: idempotent SerialWorker shutdown so tenants can
            # release CUDA graphs / model handles cleanly on drain.
            try:
                self._shutdown_serial_workers()
            except Exception:
                logger.exception(
                    "SerialWorker shutdown during drain raised; continuing"
                )
            self._emit_worker_drain_result(reason, final_status)
            time.sleep(0.2)
            if terminate_after_deadline or final_status == "drained":
                self.stop()

        threading.Thread(target=_drain_then_stop, daemon=True, name="worker-drain").start()

    def _shutdown_batched_workers(self) -> None:
        """Call instance.shutdown() + engine.shutdown() on every registered
        BatchedWorker class. Safe to call multiple times; each rec is
        flagged after shutdown to avoid duplicate calls.
        """
        loop = self._batched_loop
        if loop is None or not loop.is_running():
            return
        for rec in self._batched_instances:
            if rec.get("shutdown_done"):
                continue
            instance = rec.get("instance")
            engine = rec.get("engine")

            async def _shutdown_one(_inst: Any, _eng: Any) -> None:
                if _inst is not None:
                    sd = getattr(_inst, "shutdown", None)
                    if callable(sd):
                        try:
                            res = sd()
                            if inspect.iscoroutine(res):
                                await res
                        except Exception:
                            logger.exception(
                                "BatchedWorker instance.shutdown raised (class=%s); continuing",
                                rec.get("cls_name"),
                            )
                if _eng is not None:
                    try:
                        await _eng.shutdown()
                    except Exception:
                        logger.exception(
                            "BatchedWorker engine.shutdown raised (class=%s); continuing",
                            rec.get("cls_name"),
                        )

            try:
                fut = asyncio.run_coroutine_threadsafe(
                    _shutdown_one(instance, engine), loop
                )
                fut.result(timeout=30.0)
            except Exception:
                logger.exception(
                    "BatchedWorker shutdown failed for class=%s; continuing",
                    rec.get("cls_name"),
                )
            finally:
                rec["shutdown_done"] = True
        # Stop the loop after all shutdowns. The loop thread is daemon;
        # call_soon_threadsafe + stop is the canonical clean way.
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

    @staticmethod
    def _canonicalize_resolved_repos_map(mp: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (mp or {}).items():
            raw = str(k or "").strip()
            if not raw:
                continue
            canon = _canonicalize_model_ref_string(raw)
            out[canon] = v
            # Also keep the raw key if different, to be tolerant of non-canonical senders.
            if raw != canon:
                out[raw] = v
            # For digest-based tensorhub refs (e.g. "owner/repo@blake3:<hex>"),
            # also add a tag-based alias (e.g. "owner/repo:latest") so that
            # lookups by tag in model_ref_downloader will find the resolved
            # entry. Tensorhub is the default provider — the alias keys are
            # bare (no prefix).
            try:
                # Issue #17: lookup the provider for this ref from the
                # endpoint.lock index. Refs absent from the index default to
                # "tensorhub" (matching the wire-format contract). HF /
                # civitai refs short-circuit the tag-alias logic below.
                prov = lookup_provider_for_ref(canon)
                parsed = parse_model_ref(canon, provider=prov)
                if parsed.provider == "tensorhub" and parsed.tensorhub is not None and parsed.tensorhub.digest:
                    tag_canon = f"{parsed.tensorhub.owner}/{parsed.tensorhub.repo}:{parsed.tensorhub.tag}"
                    if parsed.tensorhub.flavor:
                        tag_canon = f"{tag_canon}#{parsed.tensorhub.flavor}"
                    if tag_canon not in out:
                        out[tag_canon] = v
            except Exception:
                pass
        return out

    def _filter_prefetch_for_disabled_functions(self, refs: List[str]) -> List[str]:
        """Drop refs from the prefetch set when they belong ONLY to disabled
        functions (no enabled function references them anymore).

        Refs that are shared between a disabled function and at least one
        enabled function still prefetch — they serve the enabled path.
        """
        if not refs or not self._disabled_functions_by_name:
            return refs
        # Collect the ref set referenced by each disabled function. For now
        # we only have the top-level offending ref in DisabledFunction; the
        # full keyspace-by-function map lives on the release manifest (not
        # plumbed to the worker yet). So we do a simple filter: if a ref
        # appears in ANY disabled-function record AND we don't have
        # evidence of it being used by an enabled function, skip it.
        disabled_refs = {
            d.get("ref", "") for d in self._disabled_functions_by_name.values()
            if d.get("ref")
        }
        if not disabled_refs:
            return refs
        # Conservative: we don't have enough info to know which refs are
        # shared. For now, keep all refs; the orchestrator already narrows
        # required_flavor_refs on disable (via BuildEndpointConfig dropping
        # the entries that only resolve for disabled functions). This hook
        # exists as a forward-compat point. Future: plumb the full
        # fn->refs map from the release manifest to enable actual filter.
        logger.debug(
            "prefetch filter: disabled-function refs detected %s; retaining all refs pending keyspace plumbing",
            sorted(disabled_refs),
        )
        return refs

    def is_function_runnable(self, function_name: str) -> bool:
        """Return False when the orchestrator has reported the function as
        disabled (a FIXED ref failed terminally) or the local worker has
        detected that this host cannot satisfy the function's hardware/runtime
        requirements. Used by spec advertisement to omit non-runnable
        functions and by request dispatch as a second line of defense in case
        a request slipped past the orchestrator gate during a refresh race.
        """
        name = (function_name or "").strip()
        if not name:
            return True
        return self._function_unavailable_entry(name) is None

    def _function_unavailable_entry(self, function_name: str) -> Optional[Dict[str, Any]]:
        name = (function_name or "").strip()
        if not name:
            return None
        entry = getattr(self, "_disabled_functions_by_name", {}).get(name)
        if entry:
            return entry
        entry = getattr(self, "_worker_local_unavailable_functions_by_name", {}).get(name)
        if entry:
            return entry
        # gen-orchestrator #341: incremental function readiness. A function
        # whose bound model refs are still downloading is reported as
        # unavailable so:
        #   - is_function_runnable(fn) returns False (no local dispatch),
        #   - _available_function_names() omits it from
        #     WorkerResources.available_functions on the wire.
        # The parallel WorkerResources.loading_functions advertisement tells
        # the orchestrator to queue jobs for it rather than reject them.
        # Computed on-the-fly so heartbeat snapshots and dispatch-time gates
        # always agree with the current model-cache state.
        loading = self._currently_loading_function_set()
        if name in loading:
            return {
                "reason": "models_loading",
                "detail": "one or more bound model refs are still downloading",
                "function_name": name,
            }
        return None

    def _currently_loading_function_set(self) -> set[str]:
        """Return the set of function names currently loading (any bound
        model ref is in the ModelCache.downloading set). Wrapper around
        ``_loading_function_names`` for the dispatch-time set-membership
        check. Computed on every call so callers see the live state — the
        model cache transition (downloading → on-disk) happens on the
        prefetch worker thread without notifying this layer.
        """
        try:
            return set(self._loading_function_names())
        except Exception:
            return set()

    def payload_key_status(self, function_name: str, model_key: str) -> Optional[str]:
        """Return the status string for a PAYLOAD-bound short_key on a
        function, or None when no status is tracked. A terminal status
        other than "resolved" means the worker should reject the request
        locally before invoking the tenant function.
        """
        fn = (function_name or "").strip()
        key = (model_key or "").strip()
        if not fn or not key:
            return None
        byKey = self._payload_ref_availability_by_function.get(fn)
        if not byKey:
            return None
        entry = byKey.get(key)
        if not entry:
            return None
        s = entry.get("status", "")
        return str(s) if s else None

    def _start_startup_prefetch(self, model_ids: List[str]) -> None:
        model_ids = [str(m or "").strip() for m in (model_ids or []) if str(m or "").strip()]
        if not model_ids:
            return
        with self._prefetch_lock:
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                return
            self._prefetch_thread = threading.Thread(
                target=self._startup_prefetch_loop,
                args=(model_ids,),
                daemon=True,
                name="StartupModelPrefetch",
            )
            self._prefetch_thread.start()

    def _startup_prefetch_loop(self, model_ids: List[str]) -> None:
        """
        Download required models on startup without waiting for a run.

        This warms the disk cache and increments readiness via disk_models in the next heartbeat.
        """
        cache_dir = tensorhub_cas_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Concurrency cap (best-effort). download() itself is blocking, so we use worker threads.
        max_conc = 2
        try:
            max_conc = max(1, int(self._model_cache.get_max_concurrent_downloads()))
        except Exception:
            max_conc = 2

        q: "queue.Queue[str]" = queue.Queue()
        for mid in model_ids:
            q.put_nowait(mid)

        def worker() -> None:
            from .models.ref_downloader import reset_resolved_repos_by_id, set_resolved_repos_by_id

            while not self._stop_event.is_set():
                try:
                    model_id = q.get_nowait()
                except queue.Empty:
                    return

                canon = _canonicalize_model_ref_string(model_id)
                try:
                    started_at = time.monotonic()
                    # If Cozy snapshot already exists on disk, mark it ready without downloading.
                    existing = self._try_find_existing_cozy_snapshot_dir(canon, cache_dir)
                    if existing is not None:
                        self._model_cache.mark_cached_to_disk(canon, existing)
                        # #321: typed model-ready signals. The orchestrator
                        # collapses all three kinds into "mark available";
                        # we preserve the kind for triage visibility.
                        self._emit_model_ready(canon, self._MODEL_AVAILABILITY_DOWNLOAD_COMPLETED)
                        self._emit_model_ready(canon, self._MODEL_AVAILABILITY_READY)
                        self._emit_model_ready(canon, self._MODEL_AVAILABILITY_CACHED)
                        # Push a primary-stream heartbeat promptly (do not wait for
                        # the 10s heartbeat tick) so schedulers see disk inventory changes.
                        # Must be is_heartbeat=True: an is_heartbeat=False message makes the
                        # orchestrator treat this as an initial registration and re-send
                        # EndpointConfig, which would re-enter this loop indefinitely.
                        self._register_worker(is_heartbeat=True)
                        continue

                    self._model_cache.mark_downloading(canon, progress=0.0)
                    try:
                        payload = json.dumps({"model_id": canon}, separators=(",", ":"), sort_keys=True).encode("utf-8")
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id="", event_type="model.download.started", payload_json=payload)
                            )
                        )
                    except Exception:
                        pass

                    tok = set_resolved_repos_by_id(self._resolved_repos_by_id_baseline or None)
                    prov_tok = set_provider_by_ref(self._provider_by_ref_index or None)
                    try:
                        local_path = self._downloader.download(canon, str(cache_dir)) if self._downloader else ""
                    finally:
                        reset_provider_by_ref(prov_tok)
                        reset_resolved_repos_by_id(tok)

                    lp = Path(local_path) if local_path else None
                    if lp is None or not lp.exists():
                        raise RuntimeError(f"model download returned missing path: {local_path!r}")

                    self._model_cache.mark_cached_to_disk(canon, lp)
                    # #321: typed signals (replaces 3 worker_event emits).
                    self._emit_model_ready(canon, self._MODEL_AVAILABILITY_READY)
                    self._emit_model_ready(canon, self._MODEL_AVAILABILITY_CACHED)
                    self._emit_model_ready(canon, self._MODEL_AVAILABILITY_DOWNLOAD_COMPLETED)
                    # Push a primary-stream heartbeat promptly (do not wait for
                    # the 10s heartbeat tick) so schedulers see disk inventory changes.
                    # Must be is_heartbeat=True: see note at the cached-already branch above.
                    self._register_worker(is_heartbeat=True)
                except Exception as e:
                    try:
                        # Clear "downloading" state on failure so we don't report a stuck download forever.
                        if self._model_cache:
                            self._model_cache.unload_model(canon)
                    except Exception:
                        pass
                    # Best-effort: if resolved URLs expired, request refresh from scheduler.
                    msg = str(e)
                    if "403" in msg or "401" in msg:
                        try:
                            payload = json.dumps({"model_id": canon}, separators=(",", ":"), sort_keys=True).encode("utf-8")
                            self._send_message(
                                pb.WorkerSchedulerMessage(
                                    worker_event=pb.WorkerEvent(request_id="", event_type="model.url_refresh", payload_json=payload)
                                )
                            )
                        except Exception:
                            pass
                    try:
                        dur_ms = int((time.monotonic() - started_at) * 1000)
                        payload = json.dumps(
                            {"model_id": canon, "duration_ms": dur_ms, "error_type": type(e).__name__},
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id="", event_type="model.download.failed", payload_json=payload)
                            )
                        )
                    except Exception:
                        pass
                    logger.warning("Startup prefetch failed for %s: %s", canon, e)

        threads = [threading.Thread(target=worker, daemon=True, name=f"PrefetchWorker-{i}") for i in range(max_conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _prefs_for_canonical(self, canonical_ref: str) -> Dict[str, Any]:
        """Lookup the baked endpoint.lock attributes (dtype/file_type/file_layout)
        for the model whose canonical ref matches ``canonical_ref``. Returns the
        shape the ref_downloader's prefs contextvar expects.
        """
        prefs: Dict[str, Any] = {}
        # Both fixed and per-function model keyspaces can reference the same
        # ref, so walk all of them and take the first match.
        keyspace_specs: List[Dict[str, Dict[str, Any]]] = []
        if self._fixed_model_spec_by_key:
            keyspace_specs.append(self._fixed_model_spec_by_key)
        for per_fn in self._payload_model_spec_by_key_by_function.values():
            keyspace_specs.append(per_fn)
        for ks in keyspace_specs:
            for _key, spec in ks.items():
                spec_ref = str(spec.get("ref") or "").strip()
                if _canonicalize_model_ref_string(spec_ref) != canonical_ref:
                    continue
                dtypes = spec.get("dtypes") or []
                file_type = spec.get("file_type") or []
                file_layout = spec.get("file_layout") or []
                if dtypes:
                    prefs["dtypes"] = list(dtypes)
                if file_type:
                    prefs["file_types"] = list(file_type)
                if file_layout:
                    prefs["file_layouts"] = list(file_layout)
                if prefs:
                    return prefs
        return prefs

    def _provider_for_ref(self, ref: str, *, default: str = "tensorhub") -> str:
        """Issue #17: look up the provider for a bare ref from the
        endpoint.lock-derived provider index. Refs not in the index default
        to ``"tensorhub"`` — matching the wire-format contract (tensorhub
        is the implicit provider when none is specified on the wire).

        Safe to call from any thread, with or without the
        ``_provider_by_ref`` contextvar set: reads the instance dict
        directly so it works both during startup prefetch (contextvar not
        yet set) and during dispatch (contextvar set on the worker thread).
        """
        if not ref:
            return default
        index = getattr(self, "_provider_by_ref_index", None) or {}
        if not isinstance(index, dict) or not index:
            return default
        if ref in index:
            return index[ref]
        stripped = ref.strip()
        if stripped != ref and stripped in index:
            return index[stripped]
        return default

    def _try_find_existing_cozy_snapshot_dir(self, canonical_model_id: str, cache_dir: Path) -> Optional[Path]:
        # Only applies to cozy:@sha256 refs where the snapshot digest is known.
        # Issue #17: consult the provider index so HF refs short-circuit to
        # None instead of being mis-parsed as tensorhub. Reads the instance
        # index directly — this method may be called from threads where the
        # contextvar isn't set, so we can't rely on lookup_provider_for_ref.
        prov = self._provider_for_ref(canonical_model_id)
        try:
            parsed = parse_model_ref(canonical_model_id, provider=prov)
        except Exception:
            return None
        if parsed.provider != "tensorhub" or parsed.tensorhub is None:
            return None
        digest = (parsed.tensorhub.digest or "").strip()
        if not digest:
            # Tag refs are mutable; rely on downloads.
            return None
        snap_dir = cache_dir / "snapshots" / digest
        if snap_dir.exists():
            return snap_dir
        return None

    # #321: _handle_worker_event_from_scheduler removed — no orchestrator ever
    # sent a worker_event toward a worker. JWT rotation is done by minting a
    # new token, force-disconnecting the worker, and letting it reconnect with
    # the new JWT (drop+reconnect = same outcome with zero new protocol).

    def _process_release_config_async_wrapper(self) -> None:
        if not self._model_manager or self._supported_model_ids_from_scheduler is None:
            self._model_init_done_event.set()
            return
        
        loop = None
        try:
            # Get or create an event loop for this thread
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            from .models.ref_downloader import reset_resolved_repos_by_id, set_resolved_repos_by_id
            tok = set_resolved_repos_by_id(self._resolved_repos_by_id_baseline or None)
            prov_tok = set_provider_by_ref(self._provider_by_ref_index or None)
            try:
                loop.run_until_complete(
                    self._model_manager.process_supported_models_config(
                        self._supported_model_ids_from_scheduler,
                        self._downloader
                    )
                )
            finally:
                reset_provider_by_ref(prov_tok)
                reset_resolved_repos_by_id(tok)
            logger.info("Model configuration and downloads (if any) processed.")
        except Exception as e:
            logger.exception(f"Error during model_manager.process_supported_models_config: {e}")
        finally:
            if loop and not loop.is_running() and not loop.is_closed(): # Clean up loop if we created it
                loop.close()
            self._model_init_done_event.set() # Signal completion or failure

    def _handle_load_model_cmd(self, cmd: LoadModelCommand) -> None:
        model_id = _canonicalize_model_ref_string(str(getattr(cmd, "model_id", "") or "").strip())
        logger.info("Received LoadModelCommand for: %s", model_id)
        success = False
        error_msg = ""
        started_at = time.monotonic()

        try:
            payload = json.dumps({"model_id": model_id}, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(request_id="", event_type="model.load.started", payload_json=payload)
                )
            )
        except Exception:
            pass

        self._gpu_busy_enter()
        if not self._model_manager:
            error_msg = "LoadModelCommand: No model manager configured on worker."
            logger.error(error_msg)
        else:
            try:
                # Wait for initial model downloads if they haven't finished
                if not self._model_init_done_event.is_set():
                    logger.info(f"LoadModelCmd ({model_id}): Waiting for initial model setup...")
                    # Timeout for this wait, can be adjusted
                    if not self._model_init_done_event.wait(timeout=300.0): # 5 minutes
                         raise TimeoutError("Timeout waiting for model initialization before VRAM load.")

                logger.info(f"Model Memory Manager attempting to load '{model_id}' into VRAM...")
                # Set resolved cozy models context so downloads can use orchestrator-resolved URLs.
                from .models.ref_downloader import reset_resolved_repos_by_id, set_resolved_repos_by_id
                per_cmd = dict(getattr(cmd, "resolved_repos_by_id", {}) or {})
                baseline = getattr(self, "_resolved_repos_by_id_baseline", None) or {}
                merged = {**baseline, **per_cmd} if per_cmd else dict(baseline)
                tok = set_resolved_repos_by_id(merged or None)
                prov_tok = set_provider_by_ref(self._provider_by_ref_index or None)
                try:
                    # load_model_into_vram is async
                    success = asyncio.run(self._model_manager.load_model_into_vram(model_id))
                finally:
                    reset_provider_by_ref(prov_tok)
                    reset_resolved_repos_by_id(tok)
                if success: logger.info(f"Model '{model_id}' loaded to VRAM by Model Memory Manager.")
                else: error_msg = f"MMM.load_model_into_vram failed for '{model_id}'."; logger.error(error_msg)
            except Exception as e:
                msg = str(e)
                if "out of memory" in msg.lower():
                    error_msg = f"insufficient_vram: {msg}"
                else:
                    error_msg = f"Exception in mmm.load_model_into_vram for '{model_id}': {e}"
                logger.exception(error_msg)
        self._gpu_busy_exit()
        
        result = pb.LoadModelResult(model_id=model_id, success=success, error_message=error_msg)
        self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))

        # Update model cache tracking if successful
        if success and self._model_cache:
            # Estimate size (could be improved with actual model size tracking)
            estimated_size_gb = 0.0
            if self._model_manager and hasattr(self._model_manager, 'model_sizes'):
                estimated_size_gb = self._model_manager.model_sizes.get(model_id, 0.0)
            self._model_cache.mark_loaded_to_vram(model_id, None, estimated_size_gb)
            # Push an immediate heartbeat/registration update for faster scheduler convergence.
            self._register_worker(is_heartbeat=True)

        try:
            dur_ms = int((time.monotonic() - started_at) * 1000)
            ev_type = "model.load.completed" if success else "model.load.failed"
            payload = json.dumps(
                {"model_id": model_id, "duration_ms": dur_ms, "error_type": "" if success else "load_failed"},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(request_id="", event_type=ev_type, payload_json=payload)
                )
            )
        except Exception:
            pass

    def _handle_download_model_cmd(self, cmd: Any) -> None:
        """Handle orchestrator command to fetch a model ref onto local disk
        (gen-orchestrator #339 — per-request cold-fetch model affinity).

        Distinct from LoadModelCommand:
          - LoadModelCommand  = promote a model already on disk into VRAM.
          - DownloadModelCommand = fetch the bytes from upstream onto disk
            so the next dispatch can pick this worker via cache locality.

        The orchestrator clears its pending-download tracker when it sees
        WorkerModelReadySignal{kind=MODEL_AVAILABILITY_DOWNLOAD_COMPLETED}
        and/or the next heartbeat's `disk_models` includes the ref
        (handleWorkerModelReady in connect_worker.go). We emit both signals
        and force an immediate heartbeat so the scheduler converges within
        a few seconds rather than waiting for the next 10s tick.

        The download itself runs in a background thread because cold
        fetches can take minutes; blocking the message-receive thread that
        long would stall every other command (interrupts, load, drain).
        """
        raw_ref = str(getattr(cmd, "ref", "") or "").strip()
        raw_model_id = str(getattr(cmd, "model_id", "") or "").strip()
        # Per the proto: when `ref` is empty the worker MAY treat `model_id`
        # as the ref (legacy paths). We always canonicalize.
        ref_for_download = raw_ref or raw_model_id
        canonical = _canonicalize_model_ref_string(ref_for_download)
        # The orchestrator keys its pending-download tracker by the
        # `model_id` field on the LoadModelResult / WorkerModelReadySignal,
        # so the response must echo the model_id the orchestrator sent
        # (not the resolved ref) so its index lookup hits.
        echo_model_id = _canonicalize_model_ref_string(raw_model_id) or canonical
        logger.info(
            "Received DownloadModelCommand model_id=%s ref=%s (canonical=%s)",
            echo_model_id,
            raw_ref,
            canonical,
        )

        if not canonical:
            error_msg = "DownloadModelCommand: empty model_id/ref"
            logger.error(error_msg)
            try:
                result = pb.LoadModelResult(
                    model_id=echo_model_id,
                    success=False,
                    error_message=error_msg,
                )
                self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))
            except Exception:
                pass
            return

        # Run the actual download off the message-receive thread so a
        # multi-minute cold fetch doesn't block subsequent orchestrator
        # commands. Tests may invoke the runner inline by calling
        # `_run_download_model_cmd` directly.
        threading.Thread(
            target=self._run_download_model_cmd,
            args=(echo_model_id, canonical),
            daemon=True,
            name=f"download-model-{canonical[:48]}",
        ).start()

    def _run_download_model_cmd(self, echo_model_id: str, canonical_ref: str) -> None:
        """Body of DownloadModelCommand handler. Split out so tests can
        call it inline without spawning a thread.
        """
        from .models.ref_downloader import reset_resolved_repos_by_id, set_resolved_repos_by_id

        cache_dir = tensorhub_cas_dir()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        success = False
        error_msg = ""
        started_at = time.monotonic()

        # Announce start (best-effort, audit channel).
        try:
            payload = json.dumps({"model_id": canonical_ref}, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(
                        request_id="", event_type="model.download.started", payload_json=payload
                    )
                )
            )
        except Exception:
            pass

        try:
            # Short-circuit: if the snapshot is already on disk just refresh
            # the cache + emit the typed ready signal — no need to re-fetch.
            existing = None
            try:
                existing = self._try_find_existing_cozy_snapshot_dir(canonical_ref, cache_dir)
            except Exception:
                existing = None

            if existing is not None and self._model_cache:
                self._model_cache.mark_cached_to_disk(canonical_ref, existing)
                success = True
            elif self._downloader is None:
                error_msg = "DownloadModelCommand: no downloader configured on worker"
                logger.error(error_msg)
            else:
                if self._model_cache:
                    try:
                        self._model_cache.mark_downloading(canonical_ref, progress=0.0)
                    except Exception:
                        pass
                tok = set_resolved_repos_by_id(getattr(self, "_resolved_repos_by_id_baseline", None) or None)
                prov_tok = set_provider_by_ref(getattr(self, "_provider_by_ref_index", None) or None)
                try:
                    local_path = self._downloader.download(canonical_ref, str(cache_dir))
                finally:
                    reset_provider_by_ref(prov_tok)
                    reset_resolved_repos_by_id(tok)
                lp = Path(local_path) if local_path else None
                if lp is None or not lp.exists():
                    error_msg = f"download returned missing path: {local_path!r}"
                    logger.error("DownloadModelCommand %s: %s", canonical_ref, error_msg)
                    if self._model_cache:
                        try:
                            self._model_cache.unload_model(canonical_ref)
                        except Exception:
                            pass
                else:
                    if self._model_cache:
                        self._model_cache.mark_cached_to_disk(canonical_ref, lp)
                    success = True
        except Exception as e:
            error_msg = f"Exception in DownloadModelCommand for {canonical_ref!r}: {e}"
            logger.exception(error_msg)
            if self._model_cache:
                try:
                    self._model_cache.unload_model(canonical_ref)
                except Exception:
                    pass

        # Always send LoadModelResult so the orchestrator can clear its
        # pending tracker on both success and failure paths.
        try:
            result = pb.LoadModelResult(
                model_id=echo_model_id,
                success=success,
                error_message=error_msg,
            )
            self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))
        except Exception:
            logger.exception("DownloadModelCommand: failed to send LoadModelResult")

        if success:
            # #339: typed model-ready signal. handleWorkerModelReady reads
            # MODEL_AVAILABILITY_DOWNLOAD_COMPLETED to append to disk_models
            # immediately and clear pendingDownloadsByModel.
            self._emit_model_ready(echo_model_id, self._MODEL_AVAILABILITY_DOWNLOAD_COMPLETED)
            self._emit_model_ready(echo_model_id, self._MODEL_AVAILABILITY_CACHED)
            # Force an out-of-band heartbeat so disk_models on the next
            # wire-level tick reflects the new ref. is_heartbeat=True is
            # load-bearing — is_heartbeat=False would be treated as a
            # fresh registration and trigger an EndpointConfig re-send.
            try:
                self._register_worker(is_heartbeat=True)
            except Exception:
                logger.exception("DownloadModelCommand: force-heartbeat failed")

        # Audit-channel completion event for observability.
        try:
            dur_ms = int((time.monotonic() - started_at) * 1000)
            ev_type = "model.download.completed" if success else "model.download.failed"
            payload = json.dumps(
                {
                    "model_id": canonical_ref,
                    "duration_ms": dur_ms,
                    "error_type": "" if success else "download_failed",
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(request_id="", event_type=ev_type, payload_json=payload)
                )
            )
        except Exception:
            pass

    def _handle_unload_model_cmd(self, cmd: Any) -> None:
        """Handle orchestrator command to unload a model from VRAM."""
        model_id = _canonicalize_model_ref_string(str(getattr(cmd, "model_id", "") or "").strip())
        logger.info("Received UnloadModelCommand for: %s", model_id)
        success = False
        error_msg = ""
        started_at = time.monotonic()

        if self._is_model_in_use(model_id):
            error_msg = "model_in_use"
            logger.warning("Refusing UnloadModelCommand for in-use model_id=%s", model_id)
            result = pb.UnloadModelResult(model_id=model_id, success=False, error_message=error_msg)
            self._send_message(pb.WorkerSchedulerMessage(unload_model_result=result))
            try:
                payload = json.dumps({"model_id": model_id, "reason": "in_use"}, separators=(",", ":"), sort_keys=True).encode(
                    "utf-8"
                )
                self._send_message(
                    pb.WorkerSchedulerMessage(
                        worker_event=pb.WorkerEvent(request_id="", event_type="model.unload.failed", payload_json=payload)
                    )
                )
            except Exception:
                pass
            return

        try:
            payload = json.dumps({"model_id": model_id}, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(request_id="", event_type="model.unload.started", payload_json=payload)
                )
            )
        except Exception:
            pass

        self._gpu_busy_enter()
        try:
            # Try legacy model_manager first
            if self._model_manager and hasattr(self._model_manager, 'unload'):
                self._model_manager.unload(model_id)
                success = True
                logger.info(f"Model '{model_id}' unloaded via Model Manager.")

            # Also update model cache
            if self._model_cache:
                self._model_cache.unload_model(model_id)
                success = True
                logger.info(f"Model '{model_id}' removed from model cache.")

            if not self._model_manager and not self._model_cache:
                error_msg = "No model manager or cache configured on worker."
                logger.error(error_msg)

        except Exception as e:
            error_msg = f"Exception unloading model '{model_id}': {e}"
            logger.exception(error_msg)
            success = False
        finally:
            self._gpu_busy_exit()

        result = pb.UnloadModelResult(model_id=model_id, success=success, error_message=error_msg)
        self._send_message(pb.WorkerSchedulerMessage(unload_model_result=result))
        if success:
            # Push an immediate heartbeat/registration update for faster scheduler convergence.
            self._register_worker(is_heartbeat=True)

        try:
            dur_ms = int((time.monotonic() - started_at) * 1000)
            ev_type = "model.unload.completed" if success else "model.unload.failed"
            payload = json.dumps(
                {"model_id": model_id, "duration_ms": dur_ms, "error_type": "" if success else "unload_failed"},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(request_id="", event_type=ev_type, payload_json=payload)
                )
            )
        except Exception:
            pass

    def _handle_job_request(self, request: JobExecutionRequest) -> None:
        """Handle a request execution envelope from the scheduler."""
        request_id = request.request_id
        job_id = str(getattr(request, "job_id", "") or "").strip() or None
        function_name = request.function_name
        input_payload = request.input_payload
        required_model_id_for_exec = ""
        timeout_ms = int(getattr(request, "timeout_ms", 0) or 0)
        owner = str(getattr(request, "owner", "") or "") or (self.owner or "")
        invoker_id = str(getattr(request, "invoker_id", "") or "")
        file_base_url = str(getattr(request, "file_base_url", "") or "")
        worker_capability_token = _extract_worker_capability_token(request)
        resolved_repos_by_id = dict(getattr(request, "resolved_repos_by_id", {}) or {})
        # Tensorhub #232: resolved hardware spec populated by the orchestrator.
        # Surfaced to tenant code read-only via ctx.compute.
        compute = _extract_resolved_compute(request)
        parent_request_id = str(getattr(request, "parent_request_id", "") or "").strip() or None
        child_request_id = str(getattr(request, "child_request_id", "") or "").strip() or None
        item_id = str(getattr(request, "item_id", "") or "").strip() or None
        raw_item_index = getattr(request, "item_index", None)
        item_index: Optional[int] = None
        if raw_item_index is not None:
            try:
                item_index = int(raw_item_index)
            except Exception:
                item_index = None
        # #321: input URLs flow inside input_payload msgpack; no separate field.
        materialized_input_urls: Optional[Dict[str, str]] = None

        required_models_raw = list(getattr(request, "required_flavor_refs", []) or [])
        if required_models_raw:
            required_model_id_for_exec = str(required_models_raw[0] or "").strip()

        t_recv_monotonic = time.monotonic()
        # Stash on the active-requests observation so _outgoing_message_iterator
        # can compute a recv→yield delta when the result message ships.
        # Keyed by request_id; popped by _build_job_observation.
        self._request_recv_times[request_id] = t_recv_monotonic
        logger.info(f"Received Request: request_id={request_id}, function={function_name}, model='{required_model_id_for_exec or 'None'}'")
        self._emit_request_event(
            request_id,
            "request.received",
            {
                "function_name": function_name,
                "job_id": job_id or "",
                "required_flavor_refs_count": len(required_models_raw),
                "input_bytes": len(input_payload or b""),
            },
        )

        spec = self._request_specs.get(function_name)
        training_fn = self._training_specs.get(function_name) if spec is None else None
        # #273: BatchedWorker (`@inference(runtime="sglang"|"vllm")`) lookup.
        batched_spec = (
            self._batched_specs.get(function_name)
            if spec is None and training_fn is None
            else None
        )
        # #322/#328: SerialWorker (sync `@inference` class) lookup.
        serial_spec = (
            self._serial_class_specs.get(function_name)
            if spec is None and training_fn is None and batched_spec is None
            else None
        )
        # #332: conversion / training / dataset sync class lookup.
        conversion_spec = (
            self._conversion_class_specs.get(function_name)
            if spec is None and training_fn is None
            and batched_spec is None and serial_spec is None
            else None
        )
        if (
            spec is None
            and training_fn is None
            and batched_spec is None
            and serial_spec is None
            and conversion_spec is None
        ):
            error_msg = f"Unknown function requested: {function_name}"
            logger.error(error_msg)
            self._emit_request_event(
                request_id,
                "request.rejected",
                {"reason": "unknown_function", "function_name": function_name},
            )
            self._send_request_result(request_id, False, None, "internal", False, "internal error", error_msg)
            return
        unavailable = self._function_unavailable_entry(function_name)
        if unavailable is not None:
            reason = str(unavailable.get("reason", "") or "function_unavailable")
            detail = str(unavailable.get("detail", "") or "")
            error_msg = f"Function unavailable on this worker: {function_name} ({reason})"
            if detail:
                error_msg = f"{error_msg}: {detail}"
            logger.warning(error_msg)
            self._emit_request_event(
                request_id,
                "request.rejected",
                {"reason": reason, "function_name": function_name},
            )
            error_type = "hardware_unmet" if reason in {
                "cuda_unavailable",
                "compute_capability_unmet",
                "insufficient_vram",
                "missing_optional_library",
                "hardware_unmet",
            } else "function_unavailable"
            self._send_request_result(
                request_id,
                False,
                None,
                error_type,
                False,
                self._sanitize_safe_message(detail or "function unavailable on this worker"),
                error_msg,
            )
            return
        if self.max_input_bytes > 0 and len(input_payload) > self.max_input_bytes:
            error_msg = f"Input payload too large: {len(input_payload)} bytes (max {self.max_input_bytes})"
            logger.error(error_msg)
            self._emit_request_event(
                request_id,
                "request.rejected",
                {"reason": "input_too_large", "input_bytes": len(input_payload)},
            )
            self._send_request_result(request_id, False, None, "validation", False, "invalid input", error_msg)
            return
        if self._draining:
            error_msg = "Worker is draining; refusing new tasks"
            logger.warning(error_msg)
            self._emit_request_event(request_id, "request.rejected", {"reason": "worker_draining"})
            self._send_request_result(request_id, False, None, "retryable", True, "worker busy", error_msg)
            return

        # required_flavor_refs are pinned flavor refs chosen by the scheduler; the worker must not guess.
        required_models: List[str] = []
        for raw in required_models_raw:
            s = str(raw or "").strip()
            if not s:
                continue
            required_models.append(_canonicalize_model_ref_string(s))

        resource_req = self._discovered_resources.get(function_name)
        req_cfg: Dict[str, Any] = dict(msgspec.to_builtins(resource_req)) if resource_req is not None else {}
        execution_hints: Dict[str, Any] = {}
        kind = str(req_cfg.get("kind", "") or "").strip()
        if kind:
            execution_hints["kind"] = kind
        # For training tasks, extract the reserved-name fields
        # (`source` and `destination`) from the input payload so RequestContext
        # can expose them to tenant code via ctx.source, ctx.source_path,
        # ctx.destination, and so the repo-job upload scope can resolve. Falls
        # back to the legacy scalar `destination_repo` field when present.
        source_info_raw: Optional[Dict[str, Any]] = None
        destination_info_raw: Optional[Dict[str, Any]] = None
        # Widened past the original kind=="training" gate: clone_huggingface /
        # clone_civitai are @inference but still need destination_repo
        # + job_id in execution_hints so publish_repo_revision can open a repo
        # job scope at finalize time. Actual file uploads continue to route
        # through media (see save_checkpoint — we intentionally don't lift to
        # repo-cas here because tensorhub's session-open auth currently 403s
        # cap-token callers with `missing_session_id`).
        if input_payload:
            try:
                raw_input = msgspec.msgpack.decode(input_payload)
                if isinstance(raw_input, dict):
                    # New contract: payload.destination is a struct with {ref, tags}.
                    dest_obj = raw_input.get("destination")
                    if isinstance(dest_obj, dict):
                        destination_info_raw = dict(dest_obj)
                        dest_ref = str(dest_obj.get("ref") or "").strip()
                        if dest_ref:
                            execution_hints["destination_repo"] = dest_ref
                    else:
                        # Legacy scalar destination_repo field.
                        dest = str(raw_input.get("destination_repo") or "").strip()
                        if dest:
                            execution_hints["destination_repo"] = dest
                            destination_info_raw = {"ref": dest, "tags": []}
                    # New contract: payload.source is a struct with {ref, checkpoint_id?, attributes}.
                    src_obj = raw_input.get("source")
                    if isinstance(src_obj, dict):
                        source_info_raw = dict(src_obj)
            except Exception:
                pass
            if job_id and "job_id" not in execution_hints:
                execution_hints["job_id"] = job_id

        # Issue #1 (slim-request-context): pick the kind-specific subclass.
        # Inference functions get the bare RequestContext (no producer-contract
        # methods); @conversion handlers get ConversionContext (default)
        # or DatasetContext (kind=='dataset-generation*'). Trainer-class
        # endpoints route through a separate runtime (entrypoint.py) and don't
        # touch this code path. The handler's typed annotation
        # (`ctx: ConversionContext`) is the source of truth — we match it here.
        ctx_cls: Type[RequestContext] = RequestContext
        if training_fn is not None:
            training_spec = getattr(training_fn, "__training_spec__", None)
            spec_kind = str(getattr(training_spec, "kind", "") or "").strip()
            if spec_kind.startswith("dataset-generation"):
                from .request_context import DatasetContext
                ctx_cls = DatasetContext
            else:
                from .request_context import ConversionContext
                ctx_cls = ConversionContext
        elif conversion_spec is not None:
            # #332: class-shape conversion / training / dataset endpoints.
            # The register-time inspector resolved the ctx subclass already
            # and stored it on the spec — honor it.
            ctx_cls = conversion_spec.ctx_class or RequestContext

        ctx = ctx_cls(
            request_id,
            job_id=job_id,
            emitter=self._emit_progress_event,
            owner=owner or None,
            invoker_id=invoker_id or None,
            timeout_ms=timeout_ms if timeout_ms > 0 else None,
            file_api_base_url=(file_base_url or self._settings.tensorhub_public_url or None),
            worker_capability_token=worker_capability_token or None,
            materialized_input_urls=materialized_input_urls or None,
            local_output_dir=None,
            resolved_repos_by_id=resolved_repos_by_id or None,
            required_models=required_models or None,
            execution_hints=execution_hints or None,
            parent_request_id=parent_request_id,
            child_request_id=child_request_id,
            item_id=item_id,
            item_index=item_index,
            source_info=source_info_raw,
            destination_info=destination_info_raw,
            compute=compute,
            hf_token=self._settings.hf_token,
        )
        # Add to active requests *before* starting thread
        with self._active_requests_lock:
             # Double-check if request is already active (race condition mitigation)
             if request_id in self._active_requests:
                  error_msg = f"Task with request_id {request_id} is already active (race condition?)."
                  logger.error(error_msg)
                  self._emit_request_event(request_id, "request.rejected", {"reason": "duplicate_request_id"})
                  # #321: emit a terminal JobExecutionResult so the orchestrator
                  # sees this as a failure NOW instead of timing out the
                  # request. error_type=duplicate_request_id makes the failure
                  # mode searchable in request_events.
                  try:
                       self._send_message(
                            pb.WorkerSchedulerMessage(
                                 job_result=pb.JobExecutionResult(
                                      request_id=request_id,
                                      success=False,
                                      error_type="duplicate_request_id",
                                      retryable=False,
                                      safe_message="orchestrator dispatched a request_id that is already active on this worker",
                                 )
                            )
                       )
                  except Exception as send_err:
                       logger.error(
                            "failed to send duplicate_request_id result rid=%s: %s",
                            request_id, send_err,
                       )
                  return # Avoid starting duplicate thread
             active_count_at_start = len(self._active_requests)
             local_queued_count_at_start = self._job_queue.qsize()
             # #322/#273: surface which dispatch archetype handled this request
             # so the per-request TTFT/ITL fields downstream (in
             # `_request_observations`) can attribute timing to the correct
             # leg. legacy = `@inference` shape; batched = BatchedWorker;
             # serial = SerialWorker; training = `@conversion`.
             if batched_spec is not None:
                 archetype = "BatchedWorker"
             elif serial_spec is not None:
                 archetype = "SerialWorker"
             elif conversion_spec is not None:
                 archetype = "ConversionWorker"
             elif training_fn is not None:
                 archetype = "training_function"
             else:
                 archetype = "legacy_inference_function"
             self._active_requests[request_id] = ctx
             self._request_observations[request_id] = {
                 "release_id": self.release_id,
                 "function_name": function_name,
                 "provider": "runpod" if self.runpod_pod_id else "local",
                 "worker_id": self.worker_id,
                 "active_count_at_start": active_count_at_start,
                 "local_queued_count_at_start": local_queued_count_at_start,
                 "enqueued_at": time.monotonic(),
                 "archetype": archetype,
             }

        # Queue execution locally to keep the receive loop responsive and report
        # local queue delay back to the scheduler.
        # Training-function handlers take a different dispatch path — the
        # dispatch wrapper already owns msgspec-decode of the raw payload dict,
        # tenant-call, and ProducedFlavor upload via RequestContext.save_checkpoint.
        if training_fn is not None:
            target = self._execute_training_request
            args = (ctx, function_name, training_fn, input_payload)
        elif batched_spec is not None:
            # #273: BatchedWorker path. Schedule the async invocation on the
            # dedicated asyncio loop and return immediately; the loop drives
            # the engine + emits IncrementalTokenDelta typed wire events. The
            # job_queue task is the lightweight "submit" call — the heavy
            # work runs on the loop thread.
            target = self._submit_batched_request
            args = (ctx, batched_spec, input_payload)
        elif serial_spec is not None:
            # #322/#328: SerialWorker path. Sync `@inference` class — runs
            # on the job_executor thread pool (same pool as the legacy
            # function-shape `_execute_request` path).
            target = self._execute_serial_class_request
            args = (ctx, serial_spec, input_payload)
        elif conversion_spec is not None:
            # #332: conversion / training / dataset class-shape path.
            # Sync class that returns ProducedFlavors; library uploads +
            # publishes a revision via `_finalize_produced_variants`.
            target = self._execute_conversion_class_request
            args = (ctx, conversion_spec, input_payload)
        else:
            target = self._execute_request
            args = (ctx, spec, input_payload, request)
        try:
            self._job_queue.put_nowait((target, args, request_id, time.monotonic()))
        except queue.Full:
            error_msg = "Worker local queue is full"
            logger.warning("%s: request_id=%s", error_msg, request_id)
            self._emit_request_event(request_id, "request.rejected", {"reason": "local_queue_full"})
            with self._active_requests_lock:
                self._active_requests.pop(request_id, None)
                self._request_observations.pop(request_id, None)
            self._send_request_result(request_id, False, None, "worker_overloaded", True, "worker busy", error_msg)

    # #321: _handle_batch_job_request removed. BatchExecutionRequest was a
    # wire-level envelope that the worker unpacked and ran serially — no real
    # GPU batching. Real LLM batching needs continuous batching with shared
    # KV-cache; reintroduce with the right primitives then.

    def _handle_interrupt_request(self, request_id: str, *, item_ids: Optional[List[str]] = None, cancel_queued_only: bool = False) -> None:
        """Handle a request to interrupt/cancel an active request."""
        logger.info(
            "Received interrupt request for request_id=%s item_ids=%s cancel_queued_only=%s",
            request_id,
            item_ids or [],
            cancel_queued_only,
        )
        with self._active_requests_lock:
            ctx = self._active_requests.get(request_id)
            if ctx:
                ctx.cancel() # Set internal flag and event
            else:
                logger.warning(f"Could not interrupt request request_id={request_id}: Not found in active requests.")

        # #273: BatchedWorker — propagate cancel into engine.abort(request_id)
        # so the continuous-batching scheduler evicts the request and frees
        # KV blocks at the next iteration boundary. Idempotent — if the
        # request never reached the engine (e.g. cancel arrived before
        # _submit_batched_request scheduled the coroutine) this is a no-op.
        with self._batched_inflight_lock:
            inflight = self._batched_inflight.pop(request_id, None)
        if inflight is not None:
            engine = inflight.get("engine")
            loop = self._batched_loop
            if engine is not None and loop is not None and loop.is_running():
                try:
                    asyncio.run_coroutine_threadsafe(engine.abort(request_id), loop)
                except Exception:
                    logger.exception(
                        "Failed to schedule engine.abort for rid=%s", request_id
                    )

    # ========================================================================
    # #273 BatchedWorker dispatch path.
    # ========================================================================
    #
    # The BatchedWorker shape (decorator `@inference(runtime="sglang"|"vllm")`)
    # hosts a long-lived continuous-batching engine inside the worker
    # process. The engine drives many concurrent requests on a single
    # forward pass; new requests join the batch at iteration boundaries.
    # This worker code is the wire-level glue:
    #
    #   1. `_ensure_batched_loop` lazily spins up a dedicated asyncio loop
    #      on its own thread. All engine + tenant async code runs there.
    #
    #   2. `_start_one_batched_instance` is the per-class one-shot setup:
    #      construct the engine, await `engine.start(model_path)`, then
    #      `await instance.setup(engine=engine)` + `await instance.warmup()`.
    #      Memoized on the instance record; second call is a no-op.
    #
    #   3. `_submit_batched_request` runs on the job_queue worker thread.
    #      It schedules `_execute_batched_request_async` onto the loop,
    #      records the inflight tuple, and returns. The loop drives the
    #      tenant's async iterator, encodes each yielded delta to
    #      IncrementalTokenDelta, and emits Done/Error at termination.
    #
    #   4. `_handle_interrupt_request` cancels the asyncio task AND calls
    #      `engine.abort(rid)` so the engine evicts the request and the
    #      KV blocks free at the next batch iteration.
    #
    #   5. `_handle_worker_drain_cmd` awaits inflight engine work then
    #      calls `instance.shutdown()` exactly once per registered class.

    def _ensure_batched_loop(self) -> asyncio.AbstractEventLoop:
        """Lazy-init the BatchedWorker asyncio loop + its dedicated thread.

        Threadsafe; idempotent. Returns the loop instance once it's running.
        """
        if self._batched_loop is not None and self._batched_loop.is_running():
            return self._batched_loop

        loop_ready = threading.Event()
        loop_box: Dict[str, asyncio.AbstractEventLoop] = {}

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            loop_box["loop"] = loop
            asyncio.set_event_loop(loop)
            loop_ready.set()
            try:
                loop.run_forever()
            finally:
                # Close the loop once stopped. Pending tasks should be
                # cancelled before we get here.
                try:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(
            target=_run_loop,
            name="gen-worker-batched-loop",
            daemon=True,
        )
        thread.start()
        loop_ready.wait(timeout=5.0)
        loop = loop_box.get("loop")
        if loop is None:
            raise RuntimeError("BatchedWorker asyncio loop failed to start")
        self._batched_loop = loop
        self._batched_loop_thread = thread
        logger.info("BatchedWorker asyncio loop started on dedicated thread")
        return loop

    def _resolve_batched_model_path(self, ep_spec: Any) -> str:
        """Resolve the on-disk model_path the engine should load.

        Reads the @inference(models={"engine": <Repo>}) declaration on
        the class. v1 path: read the canonical ref, look up the tensorhub
        CAS snapshot dir if one already exists, otherwise pass the raw
        ref through (SGLang/vLLM both accept HuggingFace org/repo refs
        and will pull through their own HF caching).

        Future: tighten this to always materialize through the worker's
        downloader so the model_path is a local directory the engine can
        mmap. For #273 v1 we accept either local-dir or remote-ref since
        SGLang and vLLM both handle both shapes.
        """
        models = dict(getattr(ep_spec, "models", {}) or {})
        if not models:
            raise ValueError(
                "BatchedWorker class declared no models={} mapping — "
                "@inference(models={'engine': Repo('owner/repo')}) is required "
                "so the SDK can resolve a model_path for engine.start()."
            )
        # Prefer the "engine" key by convention; fall back to first entry.
        binding = models.get("engine") or next(iter(models.values()))
        ref = getattr(binding, "ref", None)
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(
                "BatchedWorker class models[...] binding has no .ref attribute"
            )
        ref = ref.strip()
        # If it's a tensorhub-style ref (`cozy:owner/repo[:tag][@digest]`),
        # try to resolve via the worker's existing snapshot machinery.
        try:
            canon = _canonicalize_model_ref_string(ref)
            cache_dir = tensorhub_cas_dir()
            local = self._try_find_existing_cozy_snapshot_dir(canon, cache_dir)
            if local is not None:
                return str(local)
        except Exception:
            # Best-effort — fall through to raw ref.
            pass
        # Wire-format refs are bare (issue wire-format-bare-refs-typed-provider);
        # no prefix-strip needed.
        return ref

    async def _start_one_batched_instance(self, rec: Dict[str, Any]) -> None:
        """One-shot per-class setup: SDK engine.start (when adopted) →
        instance.setup(**resolved_models[, engine=…]) → warmup.

        Setup contract for the BatchedWorker shape:
          1. Resolve every Repo-style binding declared on @inference(models={...})
             to a local snapshot dir (or ref string) via the same path SerialWorker
             uses (_resolve_serial_model_paths), so tenants get materialized model
             paths as kwargs matching their setup() signature.
          2. Match resolved kwargs against the tenant's setup() signature. If the
             tenant declared `engine` as a kwarg (or accepts **kwargs), construct
             the SDK engine wrapper and pass it. Otherwise the tenant owns engine
             construction inside setup() — skip make_engine entirely.
          3. Call `await setup(**filtered_kwargs)` then optional `warmup()`.
        """
        if rec.get("started"):
            return
        ep_spec = rec["endpoint_spec"]
        runtime = str(rec["runtime"])

        instance = rec["instance"]
        setup_fn = getattr(instance, "setup", None)
        if setup_fn is None:
            raise ValueError(
                f"BatchedWorker class {rec.get('cls_name')!r} missing setup() method"
            )

        # Inspect the tenant's setup signature so we only pass kwargs it declares.
        try:
            sig = inspect.signature(setup_fn)
        except (TypeError, ValueError):
            sig = None
        declared_kw: set[str] = set()
        accepts_var_kw = False
        if sig is not None:
            for pname, p in sig.parameters.items():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    accepts_var_kw = True
                elif p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    declared_kw.add(pname)

        # Resolve declared model bindings to local paths / refs. Reuses the
        # SerialWorker resolver — same mechanism (canonicalize → CAS lookup →
        # fall back to bare ref). Skips Dispatch bindings that resolve at
        # request time.
        resolved_models = self._resolve_serial_model_paths(ep_spec)

        # Build setup kwargs: each resolved binding goes in if the tenant
        # declared a kwarg by that name (or accepts **kwargs). Bindings the
        # tenant didn't ask for are silently dropped — they may have been
        # declared for a different lifecycle hook.
        setup_kwargs: Dict[str, Any] = {}
        for key, value in resolved_models.items():
            if accepts_var_kw or key in declared_kw:
                setup_kwargs[key] = value

        # SDK-managed engine: only construct + start it if the tenant opted in
        # by declaring `engine` as a setup kwarg (or accepts **kwargs). Tenants
        # like chatterbox-tts manage their own AsyncLLMEngine inside setup and
        # don't take `engine=` — for those, skip make_engine entirely.
        wants_sdk_engine = accepts_var_kw or ("engine" in declared_kw)
        engine: Any = None
        if wants_sdk_engine:
            from .engines import make_engine

            model_path = self._resolve_batched_model_path(ep_spec)
            engine = make_engine(runtime)
            rec["engine"] = engine
            logger.info(
                "BatchedWorker SDK engine starting: class=%s runtime=%s model_path=%s",
                rec.get("cls_name"), runtime, model_path,
            )
            await engine.start(model_path)
            setup_kwargs["engine"] = engine

        logger.info(
            "BatchedWorker setup starting: class=%s runtime=%s sdk_engine=%s setup_kwargs=%s",
            rec.get("cls_name"), runtime, wants_sdk_engine, sorted(setup_kwargs.keys()),
        )
        # Emit `warming` phase before setup() — model load / engine init is
        # the longest part of startup and the orchestrator surfaces this to
        # callers (#322 cross-cutting hooks).
        self._emit_startup_phase("warming", status="ok", scheduler_addr=self.scheduler_addr)

        await setup_fn(**setup_kwargs)

        # Optional warmup() — async or sync.
        warmup_fn = getattr(instance, "warmup", None)
        if callable(warmup_fn):
            res = warmup_fn()
            if inspect.iscoroutine(res):
                await res
        rec["started"] = True
        logger.info(
            "BatchedWorker setup complete: class=%s runtime=%s",
            rec.get("cls_name"), runtime,
        )

    def _submit_batched_request(
        self,
        ctx: RequestContext,
        spec: _BatchedWorkerSpec,
        input_payload: bytes,
    ) -> None:
        """Job-queue target: schedule a BatchedWorker job on the asyncio loop.

        Runs on the synchronous job-dispatch thread. We don't block here —
        coroutine work runs on the BatchedWorker loop. The active_requests
        bookkeeping for cleanup happens in the async wrapper.
        """
        request_id = ctx.request_id
        try:
            loop = self._ensure_batched_loop()
        except Exception as exc:
            logger.exception("Failed to start BatchedWorker loop: %s", exc)
            self._send_request_result(
                request_id, False, None, "internal", True,
                "batched_worker_loop_unavailable", str(exc),
            )
            return

        # Find the instance record for this spec so the async wrapper can
        # call setup() lazily on first dispatch.
        rec = None
        for r in self._batched_instances:
            if r.get("instance") is spec.instance:
                rec = r
                break
        if rec is None:
            err = f"BatchedWorker instance record missing for function={spec.name}"
            logger.error(err)
            self._send_request_result(
                request_id, False, None, "internal", False,
                "internal_error", err,
            )
            return

        coro = self._execute_batched_request_async(ctx, spec, rec, input_payload)
        future = asyncio.run_coroutine_threadsafe(coro, loop)

        # Track inflight so InterruptJobCommand can call engine.abort.
        with self._batched_inflight_lock:
            self._batched_inflight[request_id] = {
                "engine": rec.get("engine"),
                "future": future,
                "function_name": spec.name,
            }

    async def _execute_batched_request_async(
        self,
        ctx: RequestContext,
        spec: _BatchedWorkerSpec,
        rec: Dict[str, Any],
        input_payload: bytes,
    ) -> None:
        """Drive one BatchedWorker request to completion.

        Lifecycle on the loop thread:
          1. Lazy-start the engine + instance.setup the first time we see
             a request for this class.
          2. msgspec-decode the payload.
          3. Call the tenant's `async def fn(ctx, payload) -> AsyncIterator[Delta]`.
          4. async-iterate the generator; encode each yielded item to
             IncrementalTokenDelta and emit it on the wire.
          5. Emit IncrementalTokenStreamDone on clean termination or
             IncrementalTokenStreamError on exception.
          6. Send the terminal JobExecutionResult.
        """
        request_id = ctx.request_id
        success = False
        error_type = ""
        retryable = False
        safe_message = ""
        error_message = ""
        seq = 0
        last_item_id = "item-0"

        # Lazy-start the engine + instance on first dispatch. We hold the
        # _started flag on the instance record so concurrent first-dispatches
        # don't double-init. Use the loop's lock semantics (the loop is
        # single-threaded so a simple flag-check is safe; but if multiple
        # in-flight starts race, the second await on the started flag
        # serializes naturally).
        if not rec.get("started"):
            try:
                await self._start_one_batched_instance(rec)
            except Exception as exc:
                logger.exception(
                    "BatchedWorker startup failed (rid=%s class=%s): %s",
                    request_id, rec.get("cls_name"), exc,
                )
                self._send_request_result(
                    request_id, False, None, "internal", True,
                    self._sanitize_safe_message(str(exc) or "engine startup failed"),
                    repr(exc),
                )
                with self._batched_inflight_lock:
                    self._batched_inflight.pop(request_id, None)
                with self._active_requests_lock:
                    self._active_requests.pop(request_id, None)
                return

        # Ensure the inflight engine reference matches (in case start filled it).
        with self._batched_inflight_lock:
            inflight = self._batched_inflight.get(request_id)
            if inflight is not None:
                inflight["engine"] = rec.get("engine")

        self._emit_request_event(
            request_id,
            "request.started",
            {
                "function_name": spec.name,
                "runtime": spec.runtime,
                "input_bytes": len(input_payload or b""),
            },
        )

        try:
            if ctx.is_canceled():
                raise CanceledError("canceled")

            input_obj = msgspec.msgpack.decode(input_payload, type=spec.payload_type)

            # Build call kwargs honoring the method's declared param names.
            call_kwargs: Dict[str, Any] = {
                spec.ctx_param: ctx,
                spec.payload_param: input_obj,
            }
            t_infer0 = time.monotonic()
            iterator_obj = spec.method(**call_kwargs)

            if not hasattr(iterator_obj, "__aiter__"):
                # tenant returned an awaitable (single-shot async). Resolve
                # and treat the value as the sole yielded delta.
                if inspect.isawaitable(iterator_obj):
                    one = await iterator_obj
                    iterator_obj = _single_item_async_iter(one)
                else:
                    raise TypeError(
                        "BatchedWorker method must return an AsyncIterator; "
                        f"got {type(iterator_obj).__name__}"
                    )

            # #273: typed signal recognition for @batched_inference. The
            # tenant's async generator may yield IncrementalTokenDelta /
            # Done / Error from gen_worker.api.streaming directly; we
            # short-circuit Done → emit IncrementalTokenStreamDone and
            # Error → emit IncrementalTokenStreamError, then break.
            from .api.streaming import (
                Done as _SignalDone,
                Error as _SignalError,
                IncrementalTokenDelta as _SignalDelta,
            )

            tenant_signaled_done = False
            tenant_signaled_error: Optional[str] = None

            async for item in iterator_obj:
                if ctx.is_canceled():
                    raise CanceledError("canceled")

                # ---- typed signal short-circuit (#273) ----
                if isinstance(item, _SignalDone):
                    tenant_signaled_done = True
                    break
                if isinstance(item, _SignalError):
                    tenant_signaled_error = str(getattr(item, "message", "") or "")
                    break

                # IncrementalTokenDelta from gen_worker.api.streaming is
                # the canonical typed delta — keep going through the
                # legacy duck-typed serializer below (text → delta_text
                # slot). Other msgspec.Structs continue to work as before.
                if (
                    spec.delta_type is not None
                    and not isinstance(item, _SignalDelta)
                    and not isinstance(item, spec.delta_type)
                ):
                    raise TypeError(
                        f"delta item type {type(item)!r} != {spec.delta_type!r}"
                    )

                # #327: peel off audio_chunk + audio_codec from the struct
                # so they travel on the typed proto slots, not base64-encoded
                # inside payload_json.
                audio_chunk, audio_codec = self._extract_audio_from_delta(item)
                payload_dict = msgspec.to_builtins(item)
                if isinstance(payload_dict, dict) and audio_chunk:
                    payload_dict.pop("audio_chunk", None)
                    payload_dict.pop("audio_codec", None)
                raw = json.dumps(
                    payload_dict, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
                item_id = "item-0"
                delta_text = ""
                if isinstance(payload_dict, dict):
                    iid = payload_dict.get("item_id")
                    if isinstance(iid, str) and iid.strip():
                        item_id = iid.strip()
                    for key in (
                        "delta_text",
                        "delta",
                        "token",
                        "text",
                        "content",
                        "caption_delta",
                    ):
                        val = payload_dict.get(key)
                        if isinstance(val, str) and val.strip():
                            delta_text = val.strip()
                            break
                seq += 1
                last_item_id = item_id
                ts_ms = int(time.time() * 1000)
                if not self._emit_incremental_delta_typed(
                    request_id=request_id,
                    function_name=spec.name,
                    item_id=item_id,
                    sequence=seq,
                    timestamp_unix_ms=ts_ms,
                    delta_text=delta_text,
                    payload_json=raw,
                    audio_chunk=audio_chunk,
                    audio_codec=audio_codec,
                ):
                    logger.warning(
                        "IncrementalTokenDelta proto class unavailable; dropping "
                        "delta rid=%s seq=%d",
                        request_id, seq,
                    )

            if ctx.is_canceled():
                raise CanceledError("canceled")

            # #273: tenant yielded Error(...) — emit error + terminate.
            if tenant_signaled_error is not None:
                raise FatalError(tenant_signaled_error or "tenant signaled error")

            # Stream-end marker (typed). Emitted whether or not the tenant
            # yielded an explicit Done() — Done() is sugar for "I'm done"
            # but the dispatcher emits the terminal proto either way.
            done_ts_ms = int(time.time() * 1000)
            if not self._emit_incremental_done_typed(
                request_id=request_id,
                function_name=spec.name,
                item_id=last_item_id,
                sequence=seq + 1,
                timestamp_unix_ms=done_ts_ms,
            ):
                logger.warning(
                    "IncrementalTokenStreamDone proto class unavailable; cannot "
                    "signal stream end rid=%s",
                    request_id,
                )
            _ = tenant_signaled_done  # suppress unused-warning; kept for log/debug parity

            self._emit_request_event(
                request_id,
                "request.inference.completed",
                {
                    "function_name": spec.name,
                    "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                    "deltas": seq,
                },
            )
            success = True

        except Exception as e:
            logger.exception("BatchedWorker request %s failed: %s", request_id, e)
            error_type, retryable, safe_message, error_message = self._map_exception(e)
            self._emit_request_event(
                request_id,
                "request.failed",
                {
                    "function_name": spec.name,
                    "error_type": error_type,
                    "retryable": bool(retryable),
                    "safe_message": safe_message,
                },
            )
            try:
                if not self._emit_incremental_error_typed(
                    request_id=request_id,
                    function_name=spec.name,
                    item_id=last_item_id,
                    sequence=seq + 1,
                    timestamp_unix_ms=int(time.time() * 1000),
                    error_message=safe_message,
                ):
                    logger.warning(
                        "IncrementalTokenStreamError proto class unavailable; "
                        "cannot signal stream error rid=%s",
                        request_id,
                    )
            except Exception:
                pass
            # Best-effort engine.abort on any failure path so the engine
            # doesn't keep a half-finished generation alive.
            engine = rec.get("engine")
            if engine is not None:
                try:
                    await engine.abort(request_id)
                except Exception:
                    pass

        finally:
            with self._batched_inflight_lock:
                self._batched_inflight.pop(request_id, None)
            with self._active_requests_lock:
                self._active_requests.pop(request_id, None)
            self._send_request_result(
                request_id,
                success,
                b"" if success else None,
                error_type,
                bool(retryable),
                safe_message,
                error_message,
            )

    # ========================================================================
    # SerialWorker class-shape dispatch path (#322/#324/#328).
    # ========================================================================
    #
    # SerialWorker = sync `@inference` class. Each request fully owns the GPU
    # end-to-end. Compared to the legacy function-shape (`_execute_request`)
    # this path is lighter because models are bound at class-level
    # (loaded once via `setup(**models)`) rather than per-request injection.
    #
    # #324: when the class declares both `batch_window_ms` and `max_batch`,
    # concurrent requests are routed through a `MicroBatchAggregator` that
    # collects them in a time window and fires ONE batched forward call.
    # Auto-disabled if any attached cache wrapper has
    # `breaks_cross_request_batching=True` (TeaCache, nunchaku #597).

    def _make_aggregator_call_fn(
        self,
        slug: str,
        sspec: "_SerialWorkerSpec",
    ) -> Callable[[List[Any]], Any]:
        """Return a closure the MicroBatchAggregator invokes with a payload list.

        The closure calls the tenant's bound `@inference.function` method with
        the LIST of decoded payloads as the payload argument. The tenant is
        responsible for batching shape handling — see
        ``gen_worker.api.micro_batch`` for the contract.

        ``ctx`` is passed as ``None`` when batched: per-caller RequestContexts
        can't all flow through one call, and per-caller bookkeeping
        (cancellation, events) is handled outside the aggregator's call_fn.
        Tenants whose batched body needs ctx should either avoid the
        aggregator or test for ``ctx is None``.
        """
        method = sspec.method

        def _call(payloads: List[Any]) -> Any:
            return method(None, payloads)

        _call.__name__ = f"micro_batch_call_{slug}"
        return _call

    def _ensure_serial_class_started(self, rec: Dict[str, Any]) -> None:
        """Lazy-call ``instance.setup(**models)`` once per SerialWorker class (#322/#328).

        Runs under the per-record lock so concurrent first-requests serialize
        the cold-start. After setup, re-scans for batch-breaking cache wrappers
        and drops any aggregator that became invalid (the tenant's setup()
        body is where TeaCache typically attaches).

        Idempotent — second call is a no-op.

        Lifecycle:
          1. Confirm ``TORCHINDUCTOR_CACHE_DIR`` is set (defensive re-check —
             discovery time should have set it, but env scrubbing in between
             would otherwise silently disable persistent torch.compile cache).
          2. Resolve every ``ep_spec.models[key]`` binding to a local path
             (snapshot dir if cached, otherwise canonical ref string).
          3. Call ``instance.setup(**models)`` so tenants declaring
             ``def setup(self, **models)`` get the model_path-resolved kwargs.
          4. Emit the typed ``warming`` startup phase signal so operators
             can tell "30s into a 2min compile" apart from "just slow."
          5. Call ``instance.warmup()`` if defined; transition implicit to
             ``ready`` (orchestrator advertises ready once warmup returns).
          6. Re-scan for batch-breaking cache wrappers (tenants typically
             attach TeaCache inside setup()).
        """
        if rec.get("started"):
            return
        with rec["started_lock"]:
            if rec.get("started"):
                return
            instance = rec["instance"]
            ep_spec = rec.get("endpoint_spec")
            cls_name = rec.get("cls_name") or type(instance).__name__
            setup_fn = getattr(instance, "setup", None)
            if setup_fn is None:
                raise ValueError(
                    f"@inference SerialWorker class {cls_name!r}: missing "
                    "setup() method"
                )

            # #322 cross-cutting hook: persistent torch.compile cache. The env
            # var must be set BEFORE setup() runs. Idempotent — already set at
            # discovery time, but a defensive re-check covers env scrubbing
            # between discovery and first dispatch.
            self._configure_torchinductor_cache_dir()

            # #328 model binding → setup() kwargs. Resolve every binding in
            # ep_spec.models to a local path the loader can pick up. Tenants
            # whose setup signature is ``def setup(self, **models)`` load with
            # the resolved kwargs; classes whose setup is bare ``def setup(self)``
            # still work because models_kwargs is empty when ep_spec.models
            # is empty.
            models_kwargs = self._resolve_serial_model_paths(ep_spec)

            logger.info(
                "SerialWorker setup starting: class=%s models=%s",
                cls_name, sorted(models_kwargs.keys()),
            )
            try:
                setup_fn(**models_kwargs)
            except TypeError as exc:
                # Setup signature doesn't accept the resolved kwargs. Fall
                # back to bare setup() so classes that intentionally bind
                # models elsewhere still load. The next request may fail
                # if setup() actually needed those kwargs.
                logger.warning(
                    "@inference SerialWorker class %r: setup(**%s) raised "
                    "TypeError %r; retrying with bare setup()",
                    cls_name, sorted(models_kwargs.keys()), exc,
                )
                try:
                    setup_fn()
                except Exception as exc2:  # noqa: BLE001
                    logger.exception(
                        "@inference SerialWorker class %r: bare setup() also "
                        "raised; continuing — the next request may fail. err=%r",
                        cls_name, exc2,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "@inference SerialWorker class %r: setup() raised; "
                    "continuing — the next request may fail. err=%r",
                    cls_name, exc,
                )

            warmup_fn = getattr(instance, "warmup", None)
            if callable(warmup_fn):
                # Emit the typed ``warming`` startup phase (#322 — added to
                # WorkerStartupPhase enum, ordered between pipeline_loading
                # and ready). Operators see "warming" instead of "just slow"
                # while torch.compile flushes graphs.
                self._emit_startup_phase(
                    "warming",
                    status="ok",
                    scheduler_addr=getattr(self, "scheduler_addr", ""),
                    class_name=cls_name,
                )
                try:
                    warmup_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "@inference SerialWorker class %r: warmup() raised "
                        "%r; continuing", cls_name, exc,
                    )

            # #324: re-scan for batch-breaking cache wrappers now that setup()
            # has attached them. Drop any aggregator that became invalid.
            from .api.micro_batch import should_disable_batching

            disable_reason = should_disable_batching(instance)
            if disable_reason:
                dropped: List[str] = []
                for slug, sspec in list(self._serial_class_specs.items()):
                    if sspec.instance is instance and slug in self._micro_batch_aggregators:
                        agg = self._micro_batch_aggregators.pop(slug, None)
                        if agg is not None:
                            try:
                                agg.shutdown()
                            except Exception:
                                pass
                        dropped.append(slug)
                if dropped:
                    logger.warning(
                        "MicroBatch aggregator(s) auto-disabled after setup() "
                        "on class=%s: %s. Reason: %s",
                        cls_name, ", ".join(dropped), disable_reason,
                    )

            rec["started"] = True
            logger.info(
                "SerialWorker setup complete: class=%s functions=%d "
                "aggregators=%d",
                cls_name,
                sum(1 for s in self._serial_class_specs.values()
                    if s.instance is instance),
                sum(1 for slug, _ in self._micro_batch_aggregators.items()
                    if any(
                        self._serial_class_specs[slug].instance is instance
                        for slug in self._micro_batch_aggregators
                        if slug in self._serial_class_specs
                    )),
            )

    def _find_serial_record(self, instance: Any) -> Optional[Dict[str, Any]]:
        for rec in self._serial_class_instances:
            if rec.get("instance") is instance:
                return rec
        return None

    def _resolve_serial_model_paths(self, ep_spec: Any) -> Dict[str, Any]:
        """Resolve every binding in ``ep_spec.models`` to a local path
        suitable for ``instance.setup(**models)`` (#322/#328).

        Returns ``{kwarg_name: model_path}``. Each value is the local
        snapshot directory if the model is already in the tensorhub CAS
        cache, otherwise the canonicalized ref string (which most loaders
        — diffusers / transformers — accept directly).

        Bindings without a ``.ref`` attribute (Dispatch bindings — per-
        request lookup) are skipped here; tenants who use Dispatch
        bindings on SerialWorker resolve at request time via
        ``ctx.resolved_repos_by_id``.
        """
        out: Dict[str, Any] = {}
        if ep_spec is None:
            return out
        models = dict(getattr(ep_spec, "models", {}) or {})
        for key, binding in models.items():
            ref = getattr(binding, "ref", None)
            if not isinstance(ref, str) or not ref.strip():
                # Dispatch bindings (or anything without a static .ref)
                # are resolved at request time, not setup() time.
                continue
            bare_ref = ref.strip()
            try:
                canon = _canonicalize_model_ref_string(bare_ref)
                cache_dir = tensorhub_cas_dir()
                local = self._try_find_existing_cozy_snapshot_dir(canon, cache_dir)
                if local is not None:
                    out[key] = str(local)
                    continue
            except Exception:
                # Fall through to ref-as-string (loader resolves via own cache).
                pass
            # Wire-format refs are bare — provider is tracked separately,
            # no prefix-strip needed.
            out[key] = bare_ref
        return out

    def _shutdown_serial_workers(self) -> None:
        """Call ``instance.shutdown()`` on every registered SerialWorker class.

        Symmetric to ``_shutdown_batched_workers`` but synchronous — no
        asyncio loop to drive, just direct method calls on the singleton
        instance. Idempotent: each rec's shutdown_done flag guards against
        double-call (drain + stop both invoke this; we want shutdown()
        to run exactly once per class).
        """
        for rec in self._serial_class_instances:
            if rec.get("shutdown_done"):
                continue
            instance = rec.get("instance")
            sd = getattr(instance, "shutdown", None)
            if callable(sd):
                try:
                    res = sd()
                    # Tenants might write shutdown() as async by mistake on a
                    # sync class. We can't await here (we're sync). Log + close
                    # the coroutine to avoid the un-awaited-coroutine warning.
                    if inspect.iscoroutine(res):
                        logger.warning(
                            "SerialWorker class=%s shutdown() returned a "
                            "coroutine — sync shutdown expected on the "
                            "SerialWorker archetype. If your endpoint needs "
                            "async shutdown, use BatchedWorker (async class "
                            "+ runtime=) instead.",
                            rec.get("cls_name"),
                        )
                        try:
                            res.close()
                        except Exception:
                            pass
                except Exception:
                    logger.exception(
                        "SerialWorker instance.shutdown raised (class=%s); continuing",
                        rec.get("cls_name"),
                    )
            rec["shutdown_done"] = True

    def _execute_serial_class_request(
        self,
        ctx: RequestContext,
        sspec: "_SerialWorkerSpec",
        input_payload: bytes,
    ) -> None:
        """Run a SerialWorker class-shape request on the job-executor thread.

        Mirrors ``_execute_request`` but for the class-shape SerialWorker.
        Skipping the injection pipeline (models bind once at setup time);
        routing optionally through the cross-request micro-batching
        aggregator when one is registered for this function.
        """
        request_id = ctx.request_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""
        retryable = False
        success = False

        rm = RunMetricsV1(
            request_id=str(request_id or ""),
            function_name=str(sspec.name or ""),
            required_models=list(getattr(ctx, "required_models", []) or []),
            resolved_repos_by_id=getattr(ctx, "resolved_repos_by_id", None) or None,
        )
        try:
            setattr(ctx, "_run_metrics", rm)
        except Exception:
            pass
        rm.mark_compute_started()

        self._emit_request_event(
            request_id,
            "request.started",
            {
                "function_name": str(sspec.name or ""),
                "required_flavor_refs": list(getattr(ctx, "required_models", []) or []),
                "execution_hints": getattr(ctx, "execution_hints", {}),
                "archetype": "SerialWorker",
            },
        )

        # Stage 5 semaphore: only acquire when the function declares
        # accelerator=cuda*. accelerator=none endpoints (marco-polo) skip
        # entirely so they scale to ThreadPoolExecutor capacity.
        needs_gpu = self._function_needs_gpu(sspec)
        # [concurrency-debug] persistent diagnostic; emit entry/exit so we can
        # see ``active`` and the semaphore value during burst tests.
        try:
            logger.info(
                "[concurrency-debug] enter rid=%s active=%d sema_value=%s needs_gpu=%s",
                request_id,
                len(self._active_requests),
                getattr(self._gpu_semaphore, "_value", None),
                needs_gpu,
            )
        except Exception:
            pass
        if needs_gpu:
            self._gpu_semaphore.acquire()
        try:
            # Bind GPU AFTER the semaphore acquire so two handlers can't race
            # on CUDA_VISIBLE_DEVICES, and BEFORE the handler runs so a fresh
            # torch context starts on the right device.
            if needs_gpu:
                self._bind_gpu_for_request(ctx)
            self._gpu_busy_enter()
            t_infer0 = time.monotonic()
            try:
                if ctx.is_canceled():
                    raise CanceledError("canceled")

                # Lazy setup on first dispatch.
                rec = self._find_serial_record(sspec.instance)
                if rec is None:
                    raise RuntimeError(
                        f"SerialWorker instance for {sspec.name!r} not registered"
                    )
                self._ensure_serial_class_started(rec)

                if ctx.is_canceled():
                    raise CanceledError("canceled")

                input_obj = msgspec.msgpack.decode(input_payload, type=sspec.payload_type)

                # #324: if a micro-batch aggregator is registered for this slug,
                # route through it. Otherwise call the bound method directly.
                aggregator = self._micro_batch_aggregators.get(sspec.name)
                if aggregator is not None and sspec.output_mode == "single":
                    # Aggregator drains on the BatchedWorker asyncio loop —
                    # reusing it means one shared loop hosts every aggregator
                    # in the worker process (no per-function thread overhead).
                    loop = self._ensure_batched_loop()
                    if aggregator._loop is None:
                        aggregator.start(loop)
                    # `submit` returns a concurrent.futures.Future. Block this
                    # job-executor thread on it (this is one of pool of threads
                    # — blocking is fine).
                    cfut = aggregator.submit(request_id, input_obj)
                    result = cfut.result()
                else:
                    result = sspec.method(ctx, input_obj)

                if ctx.is_canceled():
                    raise CanceledError("canceled")

                if sspec.output_mode == "single":
                    if sspec.output_type is not None and not isinstance(result, sspec.output_type):
                        raise TypeError(
                            f"Function {sspec.name} returned {type(result)!r}, "
                            f"expected {sspec.output_type!r}"
                        )
                    output_payload = msgspec.msgpack.encode(msgspec.to_builtins(result))
                    if self.max_output_bytes > 0 and len(output_payload) > self.max_output_bytes:
                        raise OutputTooLargeError(
                            size_bytes=len(output_payload),
                            max_bytes=self.max_output_bytes,
                        )
                    success = True
                else:
                    # Incremental: iterate the sync generator and emit deltas.
                    max_delta_bytes = 65536
                    count = 0
                    last_item_id = "item-0"
                    if not isinstance(result, cabc.Iterable):
                        raise TypeError(
                            "incremental output functions must return an iterator/iterable"
                        )
                    for item in result:
                        if ctx.is_canceled():
                            raise CanceledError("canceled")
                        if sspec.delta_type is not None and not isinstance(item, sspec.delta_type):
                            raise TypeError(
                                f"delta item type {type(item)!r} != {sspec.delta_type!r}"
                            )
                        # #327: peel off audio_chunk + audio_codec before JSON
                        # serialization so audio bytes ride the typed proto slots.
                        audio_chunk, audio_codec = self._extract_audio_from_delta(item)
                        payload = msgspec.to_builtins(item)
                        if isinstance(payload, dict) and audio_chunk:
                            payload.pop("audio_chunk", None)
                            payload.pop("audio_codec", None)
                        raw = json.dumps(
                            payload, separators=(",", ":"), sort_keys=True
                        ).encode("utf-8")
                        if max_delta_bytes > 0 and len(raw) > max_delta_bytes:
                            raw = json.dumps(
                                {"truncated": True},
                                separators=(",", ":"), sort_keys=True,
                            ).encode("utf-8")
                        item_id = "item-0"
                        delta_text = ""
                        if isinstance(payload, dict):
                            iid = payload.get("item_id")
                            if isinstance(iid, str) and iid.strip():
                                item_id = iid.strip()
                            for key in (
                                "delta_text", "delta", "token", "text", "content",
                                "caption_delta",
                            ):
                                val = payload.get(key)
                                if isinstance(val, str) and val.strip():
                                    delta_text = val.strip()
                                    break
                        self._emit_incremental_delta_typed(
                            request_id=request_id,
                            function_name=sspec.name,
                            item_id=item_id,
                            sequence=count + 1,
                            timestamp_unix_ms=int(time.time() * 1000),
                            delta_text=delta_text,
                            payload_json=raw,
                            audio_chunk=audio_chunk,
                            audio_codec=audio_codec,
                        )
                        last_item_id = item_id
                        count += 1
                    self._emit_incremental_done_typed(
                        request_id=request_id,
                        function_name=sspec.name,
                        item_id=last_item_id,
                        sequence=count + 1,
                        timestamp_unix_ms=int(time.time() * 1000),
                    )
                    output_payload = b""
                    success = True

                try:
                    rm.inference_ms = int((time.monotonic() - t_infer0) * 1000)
                except Exception:
                    pass
                self._emit_request_event(
                    request_id,
                    "request.inference.completed",
                    {
                        "function_name": sspec.name,
                        "output_mode": sspec.output_mode,
                        "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                        "archetype": "SerialWorker",
                    },
                )
            except Exception as e:
                logger.exception("SerialWorker task %s failed: %s", request_id, e)
                error_type, retryable, safe_message, error_message = self._map_exception(e)
            finally:
                self._gpu_busy_exit()
                with self._active_requests_lock:
                    self._active_requests.pop(request_id, None)
                try:
                    logger.info(
                        "[concurrency-debug] exit rid=%s active=%d",
                        request_id, len(self._active_requests),
                    )
                except Exception:
                    pass
                self._request_handler_done_times[request_id] = time.monotonic()
                self._send_request_result(
                    request_id,
                    success,
                    output_payload if success else None,
                    error_type,
                    bool(retryable),
                    safe_message,
                    error_message,
                )
        finally:
            if needs_gpu:
                self._gpu_semaphore.release()

    def _execute_conversion_class_request(
        self,
        ctx: "RequestContext",
        cspec: "_ConversionWorkerSpec",
        input_payload: bytes,
    ) -> None:
        """Run a class-shape conversion / training / dataset request (#332).

        Mirrors ``_execute_training_request`` but for the class-shape
        endpoint contract:

          1. Lazy setup on first dispatch (reuses
             ``_ensure_serial_class_started`` — same lifecycle as the
             SerialWorker leg).
          2. Materialize the source snapshot when the payload carries a
             ``source.ref`` (reserved-name materialization done by
             ``_materialize_source_for_training``).
          3. Decode the wire payload into the method's payload type and
             call ``method(ctx, payload)``.
          4. Collect the returned/yielded ``ProducedFlavor`` items into a
             list (the class-shape supports either ``Iterator`` or
             ``list`` returns).
          5. Hand the variants to ``_finalize_produced_variants`` —
             upload each flavor's files, publish a revision via
             ``publish_repo_revision`` (or ``publish_dataset_revision``
             for the dataset kind).
          6. Apply ``destination.tags`` against the produced checkpoint
             (best-effort; non-fatal on failure).
          7. Send a structured success response (empty msgpack map — the
             durable outputs are the uploaded files + the job record).
        """
        request_id = ctx.request_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""
        retryable = False
        success = False

        function_name = str(cspec.name or "")

        rm = RunMetricsV1(
            request_id=str(request_id or ""),
            function_name=function_name,
            required_models=list(getattr(ctx, "required_models", []) or []),
            resolved_repos_by_id=getattr(ctx, "resolved_repos_by_id", None) or None,
        )
        try:
            setattr(ctx, "_run_metrics", rm)
        except Exception:
            pass
        rm.mark_compute_started()
        if rm.compute_started_at:
            self._emit_worker_event_bytes(
                request_id,
                "metrics.compute.started",
                safe_json_bytes({"at": rm.compute_started_at}),
            )

        self._emit_request_event(
            request_id,
            "request.started",
            {
                "function_name": function_name,
                "required_flavor_refs": list(getattr(ctx, "required_models", []) or []),
                "execution_hints": getattr(ctx, "execution_hints", {}),
                "archetype": "ConversionWorker",
                "endpoint_kind": cspec.endpoint_kind,
                "sub_kind": cspec.sub_kind,
            },
        )

        from .models.ref_downloader import (
            reset_resolved_repos_by_id,
            set_resolved_repos_by_id,
        )

        baseline = getattr(self, "_resolved_repos_by_id_baseline", None) or None
        resolved_map = getattr(ctx, "resolved_repos_by_id", None) or baseline
        resolved_tok = set_resolved_repos_by_id(resolved_map)
        # Issue #17: provide the provider index alongside the resolved-repos
        # map so any downloader.download() that fires during the tenant body
        # routes HF/civitai refs through the right branch.
        provider_tok = set_provider_by_ref(getattr(self, "_provider_by_ref_index", None) or None)

        # Stage 5 semaphore: same gating as SerialWorker dispatch.
        # ConversionWorker/Training spec wraps the tenant call below.
        needs_gpu = self._function_needs_gpu(cspec)
        if needs_gpu:
            self._gpu_semaphore.acquire()
            self._bind_gpu_for_request(ctx)

        self._gpu_busy_enter()
        t_infer0 = time.monotonic()
        self._emit_request_event(
            request_id,
            "request.training.started",
            {"function_name": function_name, "phase": cspec.endpoint_kind},
        )

        try:
            if ctx.is_canceled():
                raise CanceledError("canceled")

            # Lazy setup on first dispatch — reuses the SerialWorker
            # bootstrap helper (same lifecycle).
            rec = self._find_serial_record(cspec.instance)
            if rec is None:
                raise RuntimeError(
                    f"ConversionWorker instance for {cspec.name!r} not registered"
                )
            self._ensure_serial_class_started(rec)

            if ctx.is_canceled():
                raise CanceledError("canceled")

            # Reserved-name source materialization. Populates
            # ctx.source_path with the local snapshot dir so the tenant's
            # generate() body can construct a Source handle on it.
            src_info = getattr(ctx, "source", None) or {}
            if isinstance(src_info, dict) and str(src_info.get("ref") or "").strip():
                self._materialize_source_for_training(ctx, src_info)

            # Decode the payload into the tenant's msgspec.Struct type.
            input_obj = msgspec.msgpack.decode(
                input_payload, type=cspec.payload_type,
            )

            # Auto-enforce calibration policy for quantization endpoints —
            # mirrors the function-shape contract's check in
            # `gen_worker.conversion.dispatch._run`. When the tenant
            # declared `calibration={scheme: 'unsupported'}` and the
            # caller supplied a dataset for that scheme, reject before
            # the method runs.
            if cspec.calibration:
                try:
                    self._enforce_class_calibration_policy(
                        function_name, cspec.calibration, input_payload,
                    )
                except ValueError as exc:
                    raise exc

            logger.info(
                "[request_id=%s] calling %s class method %s",
                request_id, cspec.endpoint_kind, function_name,
            )
            result = cspec.method(ctx, input_obj)

            # Class-shape returns either a list[ProducedFlavor] or yields
            # them as an Iterator[ProducedFlavor]. Normalize to a list so
            # the finalizer can operate on a concrete collection.
            if result is None:
                variants: list = []
            elif isinstance(result, list):
                variants = result
            elif isinstance(result, cabc.Iterable):
                variants = list(result)
            else:
                raise TypeError(
                    f"{function_name}: must return list[ProducedFlavor] or "
                    f"Iterator[ProducedFlavor]; got {type(result).__name__}"
                )

            if ctx.is_canceled():
                raise CanceledError("canceled")

            logger.info(
                "[request_id=%s] %s class method %s returned (%d variants)",
                request_id, cspec.endpoint_kind, function_name, len(variants),
            )

            # Upload + publish. The finalizer routes between
            # publish_repo_revision (checkpoints) and
            # publish_dataset_revision (datasets) based on the granular
            # kind label — pass sub_kind so the existing logic keeps
            # working unchanged.
            from .conversion.dispatch import _finalize_produced_variants

            finalize_kind = cspec.sub_kind or (
                "dataset-generation" if cspec.endpoint_kind == "dataset" else ""
            )
            _finalize_produced_variants(ctx, variants, kind=finalize_kind)

            # Apply destination.tags to the produced checkpoint. Same
            # best-effort dance as `_execute_training_request`.
            destination_info = getattr(ctx, "destination", None) or {}
            if isinstance(destination_info, dict) and destination_info.get("tags"):
                checkpoint_id = _extract_checkpoint_id_from_result(variants)
                if checkpoint_id:
                    try:
                        self._apply_destination_tags(ctx, destination_info, checkpoint_id)
                    except Exception as exc:
                        logger.warning(
                            "[request_id=%s] tag apply raised (non-fatal): %s",
                            request_id, exc,
                        )

            output_payload = msgspec.msgpack.encode({})
            success = True

            try:
                rm.inference_ms = int((time.monotonic() - t_infer0) * 1000)
            except Exception:
                pass
            self._emit_request_event(
                request_id,
                "request.inference.completed",
                {
                    "function_name": function_name,
                    "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                    "archetype": "ConversionWorker",
                    "endpoint_kind": cspec.endpoint_kind,
                },
            )
        except CanceledError:
            error_type = "cancelled"
            retryable = False
            safe_message = "cancelled"
            error_message = "cancelled"
            self._emit_request_event(
                request_id,
                "request.cancelled",
                {"function_name": function_name},
            )
            success = False
        except Exception as exc:
            if isinstance(exc, HardwareUnmetError):
                error_type = "hardware_unmet"
                retryable = False
                safe_message = self._sanitize_safe_message(str(exc) or "hardware unmet")
                error_message = f"{type(exc).__name__}: {exc}"
                self._emit_function_unavailable_signal(
                    function_name=function_name, exc=exc,
                )
                logger.warning(
                    "[request_id=%s] %s class method %s self-disabled: %s",
                    request_id, cspec.endpoint_kind, function_name, exc,
                )
            else:
                logger.exception(
                    "[request_id=%s] %s class method %s failed: %s",
                    request_id, cspec.endpoint_kind, function_name, exc,
                )
                error_type, retryable, safe_message, error_message = self._map_exception(exc)
            self._emit_request_event(
                request_id,
                "request.training.failed",
                {
                    "function_name": function_name,
                    "phase": cspec.endpoint_kind,
                    "error_type": error_type,
                    "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                },
            )
            success = False
        finally:
            try:
                reset_provider_by_ref(provider_tok)
            except Exception:
                pass
            try:
                reset_resolved_repos_by_id(resolved_tok)
            except Exception:
                pass
            self._gpu_busy_exit()
            # Stage 5: release the GPU semaphore mirroring the acquire above.
            if needs_gpu:
                try:
                    self._gpu_semaphore.release()
                except ValueError:
                    # BoundedSemaphore raises if released too many times —
                    # defensive: log and swallow so we don't crash the
                    # job-executor thread on a future refactor mistake.
                    logger.exception("gpu_semaphore over-release on conversion path")
            rm.mark_compute_completed()
            if rm.compute_completed_at:
                self._emit_worker_event_bytes(
                    request_id,
                    "metrics.compute.completed",
                    safe_json_bytes({"at": rm.compute_completed_at}),
                )
            try:
                rm.finalize()
                for ev_type, event_payload in rm.canonical_events():
                    if ev_type in ("metrics.compute.started", "metrics.compute.completed"):
                        continue
                    self._emit_worker_event_bytes(
                        request_id, ev_type, safe_json_bytes(event_payload),
                    )
            except Exception:
                pass
            try:
                self._emit_worker_event_bytes(
                    request_id,
                    "metrics.job",
                    safe_json_bytes(rm.to_metrics_run_payload()),
                )
            except Exception:
                pass
            with self._active_requests_lock:
                obs = self._request_observations.setdefault(request_id, {})
                obs["peak_memory_bytes"] = int(getattr(rm, "peak_ram_bytes", 0) or 0)
                obs["peak_vram_bytes"] = int(getattr(rm, "peak_vram_bytes", 0) or 0)
            if success:
                self._emit_request_event(
                    request_id,
                    "request.completed",
                    {
                        "function_name": function_name,
                        "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                    },
                )
            self._send_request_result(
                request_id, success, output_payload, error_type,
                bool(retryable), safe_message, error_message,
            )
            with self._active_requests_lock:
                self._active_requests.pop(request_id, None)

    def _enforce_class_calibration_policy(
        self,
        function_name: str,
        calibration: Dict[str, str],
        input_payload: bytes,
    ) -> None:
        """Reject the request when a `calibration='unsupported'` scheme
        was paired with a dataset.

        Mirrors the function-shape check in
        ``gen_worker.conversion.dispatch._run``. Best-effort: we look at
        the raw payload dict (datasets + specs[].scheme), and raise
        ``ValueError`` when an unsupported pairing is detected. Any
        parse failure (malformed payload, missing fields) is treated as
        a no-op — the tenant's method will surface the validation error
        instead.
        """
        try:
            raw = msgspec.msgpack.decode(input_payload) if input_payload else {}
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        datasets = raw.get("datasets") or []
        if not datasets:
            return
        specs = raw.get("specs") or []
        if not isinstance(specs, list):
            return
        from .conversion.calibration import lookup_policy

        for s in specs:
            scheme = ""
            if isinstance(s, dict):
                scheme = str(s.get("scheme") or "").strip()
            if not scheme:
                continue
            policy = lookup_policy(calibration, scheme)
            if policy == "unsupported":
                raise ValueError(
                    f"{function_name}: scheme={scheme!r} does not accept a "
                    f"calibration dataset (calibration='unsupported'); "
                    f"remove the dataset or choose a calibrated scheme."
                )

    def _execute_request(
        self,
        ctx: RequestContext,
        spec: _RequestSpec,
        input_payload: bytes,
        request: Any = None,
    ) -> None:
        """Execute a discovered request handler and send result/events back."""
        request_id = ctx.request_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""  # internal/legacy
        retryable = False
        success = False

        # Metrics (best-effort): never fail a job.
        resolved_map = getattr(ctx, "resolved_repos_by_id", None) or None
        rm = RunMetricsV1(
            request_id=str(request_id or ""),
            function_name=str(spec.name or ""),
            required_models=list(getattr(ctx, "required_models", []) or []),
            resolved_repos_by_id=resolved_map,
        )
        # Attach to ctx so RequestContext.save_* and injection paths can accumulate.
        try:
            setattr(ctx, "_run_metrics", rm)
        except Exception:
            pass
        rm.mark_compute_started()
        if rm.compute_started_at:
            self._emit_worker_event_bytes(request_id, "metrics.compute.started", safe_json_bytes({"at": rm.compute_started_at}))

        # Tensorhub #232: auto-bind the dispatched hardware into per-job logs
        # so operators grep "dispatched_tier=A100-80" for incident response
        # without needing tenant-side logging cooperation. Best-effort; uses
        # the ctx.compute sentinel defaults when the orchestrator didn't
        # attach resolved_compute.
        _c = getattr(ctx, "compute", None)
        if _c is not None and (_c.accelerator or _c.vram_gb or _c.gpu_count or _c.gpu_tier):
            logger.info(
                "request.dispatched rid=%s fn=%s dispatched_accelerator=%s "
                "dispatched_tier=%s dispatched_vram_gb=%s dispatched_gpu_count=%s "
                "dispatched_memory_gb=%s dispatched_cpu_cores=%s",
                request_id,
                spec.name,
                _c.accelerator or "none",
                _c.gpu_tier or "",
                _c.vram_gb,
                _c.gpu_count,
                _c.memory_gb,
                _c.cpu_cores,
            )
            # One-shot hardware_match_check — cross-reference the orchestrator's
            # selected tier against the worker's actual GPU name. WARN on
            # mismatch; doesn't fail the job. Catches scheduler bugs where a
            # job lands on unintended hardware without surfacing failure-mode
            # noise to tenants.
            if _c.gpu_tier and _c.accelerator == "cuda":
                try:
                    # Use the module-level `torch` (None when torch is
                    # unavailable). A function-local `import torch` here
                    # shadows the module binding and turns later references
                    # (e.g. the peak-mem tracker) into UnboundLocalError.
                    if torch is not None and torch.cuda.is_available():
                        actual = torch.cuda.get_device_name(0)
                        # Tier strings are heuristic (e.g. "A100-80") — accept
                        # any SUBSTRING match against the driver-reported name
                        # (e.g. "NVIDIA A100-SXM4-80GB"). Canonicalize both
                        # sides by lowercasing + stripping non-alnum before
                        # the contains-check.
                        import re  # noqa: PLC0415
                        def _canon(s: str) -> str:
                            return re.sub(r"[^a-z0-9]+", "", s.lower())
                        canon_tier = _canon(_c.gpu_tier.split("-", 1)[0])  # "A100"
                        canon_actual = _canon(actual)
                        if canon_tier and canon_tier not in canon_actual:
                            logger.warning(
                                "hardware_match_check rid=%s tier=%s actual=%r — "
                                "scheduler dispatched to unexpected hardware; "
                                "job proceeds but investigate scheduler placement",
                                request_id, _c.gpu_tier, actual,
                            )
                except Exception:
                    # torch missing / cuda not available / any inspection
                    # error — skip the check entirely, don't fail the job.
                    pass

        self._emit_request_event(
            request_id,
            "request.started",
            {
                "function_name": str(spec.name or ""),
                "required_flavor_refs": list(getattr(ctx, "required_models", []) or []),
                "execution_hints": getattr(ctx, "execution_hints", {}),
            },
        )

        # Initialize model cache_state for required models using worker's cache hints.
        try:
            vram = self._model_cache.get_vram_models() if self._model_cache else []
            disk = self._model_cache.get_disk_models() if self._model_cache else []
            best_effort_init_model_metrics(
                rm,
                rm.required_models,
                vram_models=vram,
                disk_models=disk,
                cache_dir=tensorhub_cas_dir(),
            )
        except Exception:
            pass

        # Best-effort peak tracking
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # Stage 5 semaphore: gate on the function's declared accelerator.
        # Only serialize handlers whose function declares accelerator=cuda*.
        needs_gpu = self._function_needs_gpu(spec)
        if needs_gpu:
            self._gpu_semaphore.acquire()
            self._bind_gpu_for_request(ctx)
        # Refcounted BUSY so overlapping jobs/model ops can't flip BUSY -> NOT BUSY early.
        self._gpu_busy_enter()

        from .models.ref_downloader import (
            reset_resolved_repos_by_id,
            set_resolved_repos_by_id,
        )

        baseline = getattr(self, "_resolved_repos_by_id_baseline", None) or None
        resolved_map = getattr(ctx, "resolved_repos_by_id", None) or baseline
        resolved_tok = set_resolved_repos_by_id(resolved_map)
        # Issue #17: pair the provider index with the resolved-repos map so
        # ModelRefDownloader.download routes HF/civitai bindings correctly
        # when the tenant body triggers a model fetch.
        provider_tok = set_provider_by_ref(getattr(self, "_provider_by_ref_index", None) or None)

        models_in_use: set[str] = set()
        inference_watchdog: Optional[threading.Timer] = None
        try:
            if ctx.is_canceled():
                raise CanceledError("canceled")

            # Decode payload strictly.
            input_obj = msgspec.msgpack.decode(input_payload, type=spec.payload_type)
            # Optional post-decode constraints (e.g. clamping) declared on the payload type.
            try:
                from .api.payload_constraints import apply_payload_constraints

                _ = apply_payload_constraints(input_obj)
            except Exception:
                pass
            self._materialize_assets(ctx, input_obj)
            # Best-effort extract diffusion-ish numeric fields for metrics.job.
            try:
                def _get_num(name: str) -> Optional[float]:
                    try:
                        v = getattr(input_obj, name)
                    except Exception:
                        return None
                    if isinstance(v, bool):
                        return None
                    if isinstance(v, (int, float)):
                        return float(v)
                    return None

                steps = _get_num("num_inference_steps") or _get_num("steps")
                if steps is not None:
                    rm.steps = int(steps)
                guidance = _get_num("guidance_scale") or _get_num("guidance")
                if guidance is not None:
                    rm.guidance = float(guidance)
                w = _get_num("width")
                h = _get_num("height")
                if w is not None:
                    rm.width = int(w)
                if h is not None:
                    rm.height = int(h)
            except Exception:
                pass

            # Resolve injected args.
            call_kwargs: Dict[str, Any] = {}
            call_kwargs[spec.ctx_param] = ctx
            call_kwargs[spec.payload_param] = input_obj

            # 0.7.0 binding model: caller-supplied overrides arrive on the wire
            # as orchestrator-stamped `resolved_models[param_name]` (typed by
            # orchestrator after validating the caller's `_models` block).
            # Record them here so _map_exception can classify post-download
            # load failures as `ref_compatibility_surprise`.
            resolved_models = self._resolved_models_for_request(request)
            payload_ref_keys_this_request: Dict[str, str] = {}
            for inj in spec.injections:
                if inj.binding._allow_override and inj.param_name in resolved_models:
                    payload_ref_keys_this_request[inj.param_name] = str(
                        (resolved_models.get(inj.param_name) or {}).get("ref") or ""
                    )
            self._current_payload_ref_keys = payload_ref_keys_this_request

            for inj in spec.injections:
                resolve_t0 = time.monotonic()
                resolve_watchdog = self._start_task_phase_watchdog(
                    request_id=request_id,
                    phase="model_resolve",
                    warn_after_s=float(getattr(self, "_warn_model_resolve_s", 30.0)),
                    payload={"function_name": spec.name, "param_name": inj.param_name},
                )
                self._emit_request_event(
                    request_id,
                    "request.model_resolve.started",
                    {"function_name": spec.name, "param_name": inj.param_name},
                )
                model_id = ""
                model_key: Optional[str] = None
                canon_model_id = ""
                try:
                    model_id, model_key = self._resolve_model_id_for_injection(
                        spec.name, inj, payload=input_obj, resolved_models=resolved_models,
                    )
                    canon_model_id = _canonicalize_model_ref_string(str(model_id or "").strip()) if model_id else ""
                    self._emit_request_event(
                        request_id,
                        "request.model_resolve.completed",
                        {
                            "function_name": spec.name,
                            "param_name": inj.param_name,
                            "model_id": canon_model_id,
                            "model_key": model_key or "",
                            "duration_ms": int((time.monotonic() - resolve_t0) * 1000),
                        },
                    )
                except Exception as resolve_exc:
                    self._emit_request_event(
                        request_id,
                        "request.model_resolve.failed",
                        {
                            "function_name": spec.name,
                            "param_name": inj.param_name,
                            "error_type": type(resolve_exc).__name__,
                            "duration_ms": int((time.monotonic() - resolve_t0) * 1000),
                        },
                    )
                    raise
                finally:
                    if resolve_watchdog is not None:
                        resolve_watchdog.cancel()
                # NOTE: 0.7.0 dropped the per-`[models]` dtype preference path
                # (the [models] toml table was deleted). Dtype preferences flow
                # in via the orchestrator-resolved checkpoint metadata if needed.
                if canon_model_id and canon_model_id not in models_in_use:
                    self._model_use_enter(canon_model_id)
                    models_in_use.add(canon_model_id)
                load_t0 = time.monotonic()
                load_watchdog = self._start_task_phase_watchdog(
                    request_id=request_id,
                    phase="model_load",
                    warn_after_s=float(getattr(self, "_warn_model_load_s", 60.0)),
                    payload={
                        "function_name": spec.name,
                        "param_name": inj.param_name,
                        "model_id": canon_model_id,
                    },
                )
                self._emit_request_event(
                    request_id,
                    "request.model_load.started",
                    {
                        "function_name": spec.name,
                        "param_name": inj.param_name,
                        "model_id": canon_model_id,
                    },
                )
                try:
                    call_kwargs[inj.param_name] = self._resolve_injected_value(ctx, inj.param_type, model_id, inj)
                    logger.info(
                        "[request_id=%s] model load resolved: param=%s model=%s duration_ms=%d",
                        request_id, inj.param_name, canon_model_id,
                        int((time.monotonic() - load_t0) * 1000),
                    )
                    self._emit_request_event(
                        request_id,
                        "request.model_load.completed",
                        {
                            "function_name": spec.name,
                            "param_name": inj.param_name,
                            "model_id": canon_model_id,
                            "duration_ms": int((time.monotonic() - load_t0) * 1000),
                        },
                    )
                except Exception as load_exc:
                    self._emit_request_event(
                        request_id,
                        "request.model_load.failed",
                        {
                            "function_name": spec.name,
                            "param_name": inj.param_name,
                            "model_id": canon_model_id,
                            "error_type": type(load_exc).__name__,
                            "duration_ms": int((time.monotonic() - load_t0) * 1000),
                        },
                    )
                    raise
                finally:
                    if load_watchdog is not None:
                        load_watchdog.cancel()

            # Invoke.
            execution_hints = getattr(ctx, "execution_hints", {}) or {}
            execution_kind = str(execution_hints.get("kind", "") or "").strip().lower()
            if execution_kind not in {"inference", "training"}:
                execution_kind = "inference"
            logger.info(
                "[request_id=%s] all injections resolved, entering %s phase for function=%s canceled=%s",
                request_id, execution_kind, spec.name, ctx.is_canceled(),
            )
            t_infer0 = time.monotonic()
            inference_watchdog = self._start_task_phase_watchdog(
                request_id=request_id,
                phase=execution_kind,
                warn_after_s=float(getattr(self, "_warn_inference_s", 60.0)),
                payload={"function_name": spec.name, "output_mode": spec.output_mode, "phase": execution_kind},
            )
            self._emit_request_event(
                request_id,
                f"request.{execution_kind}.started",
                {"function_name": spec.name, "output_mode": spec.output_mode, "phase": execution_kind},
            )

            # Reserved-name source materialization for training jobs.
            # If the payload declared payload.source (a SourceRepo-shaped dict),
            # resolve it against tensorhub via the worker's downloader and
            # populate ctx.source_path with the local snapshot dir.
            if execution_kind == "training":
                src_info = getattr(ctx, "source", None) or {}
                if isinstance(src_info, dict) and str(src_info.get("ref") or "").strip():
                    try:
                        self._materialize_source_for_training(ctx, src_info)
                    except Exception as exc:
                        logger.exception(
                            "[request_id=%s] source materialization failed: %s",
                            request_id, exc,
                        )
                        raise

            logger.info("[request_id=%s] calling %s", request_id, spec.name)
            # OOM-retry: escalate offload on injected pipelines if the tenant call
            # raises torch.cuda.OutOfMemoryError. Only meaningful for non-generator
            # single-output functions — streaming/incremental runs can't be rewound.
            try:
                from .inference_memory import (
                    _escalate_pipeline_mode as _escalate_pipeline_mode,
                    flush_memory as _flush_memory,
                )
            except Exception:
                _escalate_pipeline_mode = None  # type: ignore[assignment]
                _flush_memory = None  # type: ignore[assignment]

            _injected_pipes: List[Any] = []
            try:
                for _inj in spec.injections:
                    v = call_kwargs.get(_inj.param_name)
                    if v is not None and hasattr(v, "__class__"):
                        _injected_pipes.append(v)
            except Exception:
                _injected_pipes = []

            _oom_types: tuple[type, ...] = ()
            try:
                if torch is not None:
                    _oom_types = (torch.cuda.OutOfMemoryError,)  # type: ignore[attr-defined]
            except Exception:
                _oom_types = ()

            _can_retry_oom = (
                spec.output_mode == "single"
                and not inspect.isasyncgenfunction(spec.func)
                and _escalate_pipeline_mode is not None
                and _oom_types
            )
            _max_oom_retries = 2
            _oom_attempt = 0
            while True:
                try:
                    if inspect.iscoroutinefunction(spec.func):
                        result = asyncio.run(spec.func(**call_kwargs))
                    elif inspect.isasyncgenfunction(spec.func):
                        result = spec.func(**call_kwargs)
                    else:
                        result = spec.func(**call_kwargs)
                    break
                except _oom_types as _oom_exc:  # type: ignore[misc]
                    _oom_attempt += 1
                    if _flush_memory is not None:
                        _flush_memory()
                    if not _can_retry_oom or _oom_attempt > _max_oom_retries:
                        logger.error(
                            "[request_id=%s] inference OOM (attempt %d); giving up",
                            request_id, _oom_attempt,
                        )
                        raise
                    escalated = False
                    for _pipe in _injected_pipes:
                        try:
                            if _escalate_pipeline_mode(_pipe, logger=logger, escalation=("vae_only", "model_offload", "group_offload", "sequential")):
                                escalated = True
                        except Exception:
                            pass
                    if not escalated:
                        logger.warning(
                            "[request_id=%s] inference OOM (attempt %d); no further escalation possible, re-raising",
                            request_id, _oom_attempt,
                        )
                        raise
                    logger.warning(
                        "[request_id=%s] inference OOM (attempt %d); retrying with escalated offload",
                        request_id, _oom_attempt,
                    )
                    try:
                        self._emit_worker_event_bytes(request_id, "inference.oom_retry", safe_json_bytes({
                            "function_name": spec.name,
                            "attempt": _oom_attempt,
                        }))
                    except Exception:
                        pass
            logger.info("[request_id=%s] %s returned, output_mode=%s", request_id, spec.name, spec.output_mode)

            if ctx.is_canceled():
                raise CanceledError("canceled")

            if spec.output_mode == "single":
                if spec.output_type is not None and not isinstance(result, spec.output_type):
                    raise TypeError(f"Function {spec.name} returned {type(result)!r}, expected {spec.output_type!r}")
                result = self._auto_upload_output_assets(ctx, result)
                output_payload = msgspec.msgpack.encode(msgspec.to_builtins(result))
                if self.max_output_bytes > 0 and len(output_payload) > self.max_output_bytes:
                    raise OutputTooLargeError(size_bytes=len(output_payload), max_bytes=self.max_output_bytes)
                success = True

                # Training: apply destination.tags to the produced
                # checkpoint. Non-fatal on failure — upload already succeeded.
                if execution_kind == "training":
                    dest_info = getattr(ctx, "destination", None) or {}
                    if isinstance(dest_info, dict) and dest_info.get("tags"):
                        checkpoint_id = _extract_checkpoint_id_from_result(result)
                        if checkpoint_id:
                            try:
                                self._apply_destination_tags(ctx, dest_info, checkpoint_id)
                            except Exception as exc:
                                logger.warning(
                                    "[request_id=%s] tag apply raised (non-fatal): %s",
                                    request_id, exc,
                                )
                        else:
                            logger.warning(
                                "[request_id=%s] destination.tags set but could not extract checkpoint_id from result",
                                request_id,
                            )
            else:
                # Incremental output: the function returns an iterator of delta structs.
                max_delta_bytes = 65536
                max_events = 0
                count = 0
                last_item_id = "item-0"

                def emit_delta(delta_obj: msgspec.Struct) -> None:
                    nonlocal count, last_item_id
                    # #327: route audio_chunk / audio_codec to the typed
                    # IncrementalTokenDelta proto slots instead of base64-
                    # in-payload_json.
                    audio_chunk, audio_codec = self._extract_audio_from_delta(delta_obj)
                    payload = msgspec.to_builtins(delta_obj)
                    if isinstance(payload, dict) and audio_chunk:
                        payload.pop("audio_chunk", None)
                        payload.pop("audio_codec", None)
                    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    if max_delta_bytes > 0 and len(raw) > max_delta_bytes:
                        raw = json.dumps({"truncated": True}, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    item_id = "item-0"
                    delta_text = ""
                    if isinstance(payload, dict):
                        iid = payload.get("item_id")
                        if isinstance(iid, str) and iid.strip():
                            item_id = iid.strip()
                        for key in ("delta_text", "delta", "token", "text", "content", "caption_delta"):
                            val = payload.get(key)
                            if isinstance(val, str) and val.strip():
                                delta_text = val.strip()
                                break
                    ts_ms = int(time.time() * 1000)
                    # #321: dual streaming path removed. The typed
                    # IncrementalToken* messages are the contract; a False
                    # return now means the proto class is missing on the wire
                    # (deployment skew with the scheduler) — drop the delta
                    # and log rather than silently routing through
                    # worker_event, which the orchestrator wouldn't typed-
                    # branch on anyway.
                    if not self._emit_incremental_delta_typed(
                        request_id=request_id,
                        function_name=spec.name,
                        item_id=item_id,
                        sequence=count + 1,
                        timestamp_unix_ms=ts_ms,
                        delta_text=delta_text,
                        payload_json=raw,
                        audio_chunk=audio_chunk,
                        audio_codec=audio_codec,
                    ):
                        logger.warning(
                            "IncrementalTokenDelta proto class unavailable; dropping delta rid=%s seq=%d",
                            request_id, count + 1,
                        )
                    last_item_id = item_id
                    count += 1

                iterator_obj = result

                async def consume_async() -> None:
                    nonlocal count
                    async for item in iterator_obj:
                        if ctx.is_canceled():
                            raise CanceledError("canceled")
                        if spec.delta_type is not None and not isinstance(item, spec.delta_type):
                            raise TypeError(f"delta item type {type(item)!r} != {spec.delta_type!r}")
                        emit_delta(item)
                        if max_events > 0 and count >= max_events:
                            break

                if hasattr(iterator_obj, "__aiter__"):
                    asyncio.run(consume_async())
                else:
                    if not isinstance(iterator_obj, cabc.Iterator) and not isinstance(iterator_obj, cabc.Iterable):
                        raise TypeError("incremental output functions must return an iterator/iterable")

                    for item in iterator_obj:
                        if ctx.is_canceled():
                            raise CanceledError("canceled")
                        if spec.delta_type is not None and not isinstance(item, spec.delta_type):
                            raise TypeError(f"delta item type {type(item)!r} != {spec.delta_type!r}")
                        emit_delta(item)
                        if max_events > 0 and count >= max_events:
                            break

                # Emit completion marker (#321: typed-only, no worker_event fallback).
                done_ts_ms = int(time.time() * 1000)
                if not self._emit_incremental_done_typed(
                    request_id=request_id,
                    function_name=spec.name,
                    item_id=last_item_id,
                    sequence=count + 1,
                    timestamp_unix_ms=done_ts_ms,
                ):
                    logger.warning(
                        "IncrementalTokenStreamDone proto class unavailable; cannot signal stream end rid=%s",
                        request_id,
                    )
                output_payload = b""
                success = True

            # inference_ms is best-effort and currently reflects the time spent in the user function
            # (including incremental consumption for streaming outputs).
            try:
                rm.inference_ms = int((time.monotonic() - t_infer0) * 1000)
            except Exception:
                pass
            if inference_watchdog is not None:
                inference_watchdog.cancel()
            self._emit_request_event(
                request_id,
                f"request.{execution_kind}.completed",
                {
                    "function_name": spec.name,
                    "output_mode": spec.output_mode,
                    "phase": execution_kind,
                    "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                },
            )

            logger.info("Task %s completed successfully. inference_ms=%d", request_id, int((time.monotonic() - t_infer0) * 1000))

        except Exception as e:
            logger.exception("Task %s failed: %s", request_id, e)
            error_type, retryable, safe_message, error_message = self._map_exception(e)
            if isinstance(e, HardwareUnmetError):
                # tell the orchestrator to stop dispatching this
                # function to this worker.
                self._emit_function_unavailable_signal(function_name=str(spec.name or ""), exc=e)
            if inference_watchdog is not None:
                inference_watchdog.cancel()
            self._emit_request_event(
                request_id,
                "request.failed",
                {
                    "function_name": spec.name,
                    "error_type": error_type,
                    "retryable": bool(retryable),
                    "safe_message": safe_message,
                },
            )
            if "t_infer0" in locals():
                self._emit_request_event(
                    request_id,
                    f"request.{execution_kind}.failed",
                    {
                        "function_name": spec.name,
                        "phase": execution_kind,
                        "error_type": error_type,
                        "duration_ms": int((time.monotonic() - float(locals().get("t_infer0", time.monotonic()))) * 1000),
                    },
                )
            if spec.output_mode == "incremental":
                try:
                    # #321: typed-only, no worker_event fallback.
                    if not self._emit_incremental_error_typed(
                        request_id=request_id,
                        function_name=spec.name,
                        item_id="item-0",
                        sequence=0,
                        timestamp_unix_ms=int(time.time() * 1000),
                        error_message=safe_message,
                    ):
                        logger.warning(
                            "IncrementalTokenStreamError proto class unavailable; cannot signal stream error rid=%s",
                            request_id,
                        )
                except Exception:
                    pass
            success = False
        finally:
            if inference_watchdog is not None:
                inference_watchdog.cancel()
            try:
                reset_provider_by_ref(provider_tok)
            except Exception:
                pass
            reset_resolved_repos_by_id(resolved_tok)

            for mid in models_in_use:
                try:
                    self._model_use_exit(mid)
                except Exception:
                    pass
            self._gpu_busy_exit()
            # Stage 5: release the GPU semaphore mirroring the acquire above.
            if needs_gpu:
                try:
                    self._gpu_semaphore.release()
                except ValueError:
                    logger.exception("gpu_semaphore over-release on inference path")
            # Clear per-request PAYLOAD_REF tracking so the classifier doesn't
            # mis-fire on the next request that didn't involve caller refs.
            self._current_payload_ref_keys = {}

            # Best-effort resource peaks.
            try:
                import resource as _resource  # linux/mac
                r = _resource.getrusage(_resource.RUSAGE_SELF)
                # Linux: ru_maxrss is KB. macOS: bytes. We only run linux in prod, but keep best-effort.
                rss = int(getattr(r, "ru_maxrss", 0) or 0)
                if rss > 0:
                    # Heuristic: treat small values as KB.
                    rm.peak_ram_bytes = rss * 1024 if rss < (1 << 40) else rss
            except Exception:
                pass
            if torch is not None:
                try:
                    if torch.cuda.is_available():
                        rm.peak_vram_bytes = int(torch.cuda.max_memory_allocated())
                except Exception:
                    pass

            rm.mark_compute_completed()
            if rm.compute_completed_at:
                self._emit_worker_event_bytes(request_id, "metrics.compute.completed", safe_json_bytes({"at": rm.compute_completed_at}))

                # Emit canonical metric events if values exist.
                try:
                    rm.finalize()
                    for ev_type, event_payload in rm.canonical_events():
                        # compute.* already emitted in real time above
                        if ev_type in ("metrics.compute.started", "metrics.compute.completed"):
                            continue
                        self._emit_worker_event_bytes(request_id, ev_type, safe_json_bytes(event_payload))
                except Exception:
                    pass
            # Emit extended debug payload at end (best-effort).
            try:
                self._emit_worker_event_bytes(request_id, "metrics.job", safe_json_bytes(rm.to_metrics_run_payload()))
            except Exception:
                pass
            with self._active_requests_lock:
                obs = self._request_observations.setdefault(request_id, {})
                obs["peak_memory_bytes"] = int(getattr(rm, "peak_ram_bytes", 0) or 0)
                obs["peak_vram_bytes"] = int(getattr(rm, "peak_vram_bytes", 0) or 0)
            if success:
                self._emit_request_event(
                    request_id,
                    "request.completed",
                    {
                        "function_name": spec.name,
                        "duration_ms": int((time.monotonic() - float(getattr(rm, "_t0_monotonic", time.monotonic()))) * 1000),
                    },
                )

            self._send_request_result(request_id, success, output_payload, error_type, bool(retryable), safe_message, error_message)

            # Issue #20: close any still-open upload sessions at request end.
            # Successful finalize removes the session from the cache so this
            # is a no-op for the happy path; aborts any lingering sessions
            # from partial failures (tenant function raised before finalize).
            try:
                ctx._close_upload_sessions(abort_open=True)
            except Exception:
                logger.debug("upload_session_close_on_request_end_failed", exc_info=True)

            with self._active_requests_lock:
                self._active_requests.pop(request_id, None)

    def _execute_training_request(
        self,
        ctx: RequestContext,
        function_name: str,
        training_fn: Callable[..., Any],
        input_payload: bytes,
    ) -> None:
        """Dispatch a @conversion handler.

        The dispatch wrapper (conversion/dispatch.py) owns tenant-call plumbing:
        msgspec-decoding each tenant parameter, building Source/Dataset helpers,
        invoking the tenant function, and uploading each returned
        ProducedFlavor via ctx.save_checkpoint. The worker's job here is the
        outer shell — metrics, events, source materialization, destination tag
        application, and the success/fail RPC response.
        """
        request_id = ctx.request_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""
        retryable = False
        success = False

        rm = RunMetricsV1(
            request_id=str(request_id or ""),
            function_name=str(function_name or ""),
            required_models=list(getattr(ctx, "required_models", []) or []),
            resolved_repos_by_id=getattr(ctx, "resolved_repos_by_id", None) or None,
        )
        try:
            setattr(ctx, "_run_metrics", rm)
        except Exception:
            pass
        rm.mark_compute_started()
        if rm.compute_started_at:
            self._emit_worker_event_bytes(request_id, "metrics.compute.started", safe_json_bytes({"at": rm.compute_started_at}))

        self._emit_request_event(
            request_id,
            "request.started",
            {
                "function_name": function_name,
                "required_flavor_refs": list(getattr(ctx, "required_models", []) or []),
                "execution_hints": getattr(ctx, "execution_hints", {}),
            },
        )

        # Stage 5 semaphore: training functions carry their Resources on
        # the `_worker_resources` attribute (no _SerialWorkerSpec object
        # exists for the legacy function-shape path). Build a minimal
        # accessor so `_function_needs_gpu` can inspect the same way.
        class _TrainingSpecShim:
            def __init__(self, res: Any) -> None:
                self.resources = res
        _tr_res = getattr(training_fn, "_worker_resources", None)
        needs_gpu = self._function_needs_gpu(_TrainingSpecShim(_tr_res))
        if needs_gpu:
            self._gpu_semaphore.acquire()
            self._bind_gpu_for_request(ctx)
        self._gpu_busy_enter()
        t_infer0 = time.monotonic()
        self._emit_request_event(
            request_id,
            "request.training.started",
            {"function_name": function_name, "phase": "training"},
        )

        # Populate the resolved_repos_by_id contextvar from this request's
        # ctx so the downloader (used by `_materialize_source_for_training`
        # and tenant code that calls Source.* helpers) can look up the
        # orchestrator-pre-resolved snapshot URLs. Without this, training
        # jobs fail at materialize-source time with "tensorhub ref ... not in
        # resolved_repos_by_id" even when the orchestrator did pre-resolve.
        # Mirrors the inference path's set at line ~4172 below.
        from .models.ref_downloader import (
            reset_resolved_repos_by_id,
            set_resolved_repos_by_id,
        )

        baseline = getattr(self, "_resolved_repos_by_id_baseline", None) or None
        resolved_map = getattr(ctx, "resolved_repos_by_id", None) or baseline
        resolved_tok = set_resolved_repos_by_id(resolved_map)
        # Issue #17: pair the provider index alongside the resolved-repos
        # set so model downloads inside the tenant body route to the
        # correct provider branch (HF / civitai / tensorhub).
        provider_tok = set_provider_by_ref(getattr(self, "_provider_by_ref_index", None) or None)

        try:
            if ctx.is_canceled():
                raise CanceledError("canceled")

            # Decode the raw payload dict. Training dispatch wrapper handles
            # msgspec.convert per tenant-declared field itself — we don't
            # pre-type the top-level dict.
            raw_payload: Any = msgspec.msgpack.decode(input_payload) if input_payload else {}

            # Reserved-name source materialization — populate ctx.source_path
            # with the local snapshot dir so the dispatch wrapper can build
            # Source helpers on top of it.
            src_info = getattr(ctx, "source", None) or {}
            if isinstance(src_info, dict) and str(src_info.get("ref") or "").strip():
                self._materialize_source_for_training(ctx, src_info)

            logger.info("[request_id=%s] calling training_function %s", request_id, function_name)
            result = training_fn(ctx, raw_payload)
            logger.info("[request_id=%s] training_function %s returned (%d variants)",
                        request_id, function_name, len(result) if isinstance(result, list) else -1)

            if ctx.is_canceled():
                raise CanceledError("canceled")

            # Apply destination.tags to the produced checkpoint. Dispatch wrapper
            # skips tag apply when RequestContext.apply_destination_tag isn't set;
            # handle it here against the tensorhub API instead.
            destination_info = getattr(ctx, "destination", None) or {}
            if isinstance(destination_info, dict) and destination_info.get("tags"):
                checkpoint_id = _extract_checkpoint_id_from_result(result)
                if checkpoint_id:
                    try:
                        self._apply_destination_tags(ctx, destination_info, checkpoint_id)
                    except Exception as exc:
                        logger.warning(
                            "[request_id=%s] tag apply raised (non-fatal): %s",
                            request_id, exc,
                        )

            # Synthesize a minimal success payload. Training functions don't
            # emit a canonical response body — uploads + the job record are the
            # durable outputs. Send an empty msgpack map so orchestrator sees
            # structured output.
            output_payload = msgspec.msgpack.encode({})
            success = True

        except CanceledError:
            error_type = "cancelled"
            retryable = False
            safe_message = "cancelled"
            error_message = "cancelled"
            self._emit_request_event(
                request_id,
                "request.cancelled",
                {"function_name": function_name},
            )
            success = False

        except Exception as exc:
            # HardwareUnmetError → function-level self-disable terminal.
            # Classify separately before the catch-all so (a) the error_type is
            # hardware_unmet (not internal), (b) the WorkerFunctionUnavailableSignal
            # goes upstream so the orchestrator narrows dispatch.
            if isinstance(exc, HardwareUnmetError):
                error_type = "hardware_unmet"
                retryable = False
                safe_message = self._sanitize_safe_message(str(exc) or "hardware unmet")
                error_message = f"{type(exc).__name__}: {exc}"
                self._emit_function_unavailable_signal(function_name=function_name, exc=exc)
                logger.warning(
                    "[request_id=%s] training_function %s self-disabled: %s",
                    request_id, function_name, exc,
                )
            else:
                error_type = "internal"
                retryable = False
                safe_message = f"{type(exc).__name__}: {exc}"
                error_message = safe_message
                logger.exception("[request_id=%s] training_function %s failed: %s",
                                 request_id, function_name, exc)
            self._emit_request_event(
                request_id,
                "request.failed",
                {
                    "function_name": function_name,
                    "error_type": error_type,
                    "retryable": retryable,
                    "safe_message": safe_message,
                },
            )
            self._emit_request_event(
                request_id,
                "request.training.failed",
                {
                    "function_name": function_name,
                    "phase": "training",
                    "error_type": error_type,
                    "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                },
            )
            success = False

        finally:
            try:
                reset_provider_by_ref(provider_tok)
            except Exception:
                pass
            try:
                reset_resolved_repos_by_id(resolved_tok)
            except Exception:
                pass
            self._gpu_busy_exit()
            # Stage 5: release the GPU semaphore mirroring the acquire above.
            if needs_gpu:
                try:
                    self._gpu_semaphore.release()
                except ValueError:
                    logger.exception("gpu_semaphore over-release on training path")
            rm.mark_compute_completed()
            if rm.compute_completed_at:
                self._emit_worker_event_bytes(request_id, "metrics.compute.completed", safe_json_bytes({"at": rm.compute_completed_at}))
            try:
                rm.finalize()
                for ev_type, event_payload in rm.canonical_events():
                    if ev_type in ("metrics.compute.started", "metrics.compute.completed"):
                        continue
                    self._emit_worker_event_bytes(request_id, ev_type, safe_json_bytes(event_payload))
            except Exception:
                pass
            try:
                self._emit_worker_event_bytes(request_id, "metrics.job", safe_json_bytes(rm.to_metrics_run_payload()))
            except Exception:
                pass
            with self._active_requests_lock:
                obs = self._request_observations.setdefault(request_id, {})
                obs["peak_memory_bytes"] = int(getattr(rm, "peak_ram_bytes", 0) or 0)
                obs["peak_vram_bytes"] = int(getattr(rm, "peak_vram_bytes", 0) or 0)
            if success:
                self._emit_request_event(
                    request_id,
                    "request.completed",
                    {
                        "function_name": function_name,
                        "duration_ms": int((time.monotonic() - t_infer0) * 1000),
                    },
                )
            self._send_request_result(request_id, success, output_payload, error_type, bool(retryable), safe_message, error_message)
            with self._active_requests_lock:
                self._active_requests.pop(request_id, None)

    def _get_local_model_cache(self) -> Optional[Any]:
        """
        Best-effort initialize a local (non-NFS) model cache for NFS->local localization.

        Returns None when disabled or misconfigured (e.g. local cache dir is itself on NFS).
        """
        d = (self._local_model_cache_dir or "").strip()
        if not d:
            return None
        with self._local_model_cache_lock:
            if self._local_model_cache is not None:
                return self._local_model_cache
            try:
                from .pipeline.mount_backend import mount_backend_for_path
                mb = mount_backend_for_path(d)
                if mb is not None and mb.is_nfs:
                    logger.warning(
                        "WORKER_LOCAL_MODEL_CACHE_DIR appears to be on NFS (%s, %s); disabling localization cache",
                        mb.fstype,
                        mb.mountpoint,
                    )
                    # Disable for the rest of the process lifetime.
                    self._local_model_cache_dir = ""
                    return None
            except Exception:
                pass

            try:
                from .pipeline.local_cache import LocalModelCache  # local import to avoid heavy import at worker init

                max_cache_gb = 100.0
                self._local_model_cache = LocalModelCache(d, max_cache_gb)
                logger.info("Local model cache enabled: %s (%.1fGB max)", d, max_cache_gb)
                return self._local_model_cache
            except Exception as e:
                logger.warning("Failed to init local model cache: %s", e)
                self._local_model_cache_dir = ""
                return None

    def _shared_disk_volume_info(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Best-effort identify the disk backend and volume identity for the shared model cache.

        - Does not include file paths or raw mount sources (those may leak internal details).
        - Provides a stable `disk_volume_key` that gen-orchestrator can use to group workers
          that share the same NFS volume.
        """
        try:
            from .pipeline.mount_backend import mount_backend_for_path, volume_key_for_path

            p = path or tensorhub_cas_dir()
            mb = mount_backend_for_path(p)
            if mb is None:
                return {}
            return {
                "disk_backend": "nfs" if mb.is_nfs else "local",
                "disk_fstype": mb.fstype,
                "disk_volume_key": volume_key_for_path(p),
            }
        except Exception:
            return {}

    def _materialize_source_for_training(self, ctx: RequestContext, source: Dict[str, Any]) -> None:
        """Materialize the source snapshot for a conversion/training job.

        Called just before tenant code runs when the job payload has a
        reserved-name `source` field. Resolves source.ref against tensorhub
        and populates ctx.source_path with the local snapshot directory.
        If source.checkpoint_id is set it pins a specific checkpoint.
        """
        ref = str(source.get("ref") or "").strip()
        if not ref:
            return
        if self._downloader is None:
            raise RuntimeError(
                "source materialization requires a model downloader "
                "(Settings.tensorhub_public_url must be set so the worker can resolve cozy refs)"
            )
        # Wire-format refs are bare tensorhub refs; the downloader treats
        # bare refs as tensorhub snapshots by default.
        canonical = _canonicalize_model_ref_string(ref)

        logger.info("materialize_source source=%s", source)

        cache_dir = str(tensorhub_cas_dir())
        local = self._downloader.download(canonical, cache_dir)

        if not local or not Path(local).exists():
            raise RuntimeError(f"source materialization returned missing path: {local!r}")

        ctx._set_source_path(local)
        logger.info(
            "[request_id=%s] conversion source materialized at %s (ref=%s)",
            ctx.request_id, local, ref,
        )

    def _apply_destination_tags(self, ctx: RequestContext, destination: Dict[str, Any], checkpoint_id: str) -> None:
        """Apply destination.tags to the newly-produced checkpoint.

        Called after the tenant function returns success and at least one
        variant upload has committed. For each tag in destination.tags, PUTs
        to tensorhub's `/repos/:owner/:repo/tags?tag=<tag>` route using the
        job's capability token. Tag-move failures log + surface as a job
        warning rather than failing the job (upload already succeeded).
        """
        if not isinstance(destination, dict):
            return
        tags = destination.get("tags") or []
        if not isinstance(tags, list) or not tags:
            return
        dest_ref = str(destination.get("ref") or "").strip()
        if not dest_ref:
            return
        if "/" not in dest_ref:
            logger.warning("[request_id=%s] destination.ref %r has no owner/repo shape; skipping tag apply", ctx.request_id, dest_ref)
            return
        owner, repo = dest_ref.split("/", 1)
        owner = owner.strip()
        repo = repo.strip()
        checkpoint_id = str(checkpoint_id or "").strip()
        if not owner or not repo or not checkpoint_id:
            logger.warning("[request_id=%s] tag apply skipped: owner=%r repo=%r checkpoint=%r", ctx.request_id, owner, repo, checkpoint_id)
            return

        base_url = (self._settings.tensorhub_public_url or "").strip()
        if not base_url:
            logger.warning("[request_id=%s] Settings.tensorhub_public_url unset; skipping destination tag apply", ctx.request_id)
            return
        base_url = base_url.rstrip("/")

        token = str(getattr(ctx, "_worker_capability_token", "") or "").strip()
        if not token:
            logger.warning("[request_id=%s] no worker_capability_token; skipping destination tag apply", ctx.request_id)
            return

        import urllib.parse
        import urllib.request

        for raw_tag in tags:
            tag = str(raw_tag or "").strip()
            if not tag:
                continue
            url = (
                f"{base_url}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
                f"{urllib.parse.quote(repo, safe='')}/tags?tag={urllib.parse.quote(tag, safe='')}"
            )
            body = json.dumps({"checkpoint_id": checkpoint_id}).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="PUT")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status >= 300:
                        logger.warning(
                            "[request_id=%s] tag apply %r returned HTTP %s (non-fatal)",
                            ctx.request_id, tag, resp.status,
                        )
                    else:
                        logger.info(
                            "[request_id=%s] tag %r moved to checkpoint %s on %s/%s",
                            ctx.request_id, tag, checkpoint_id[:16], owner, repo,
                        )
            except Exception as exc:
                logger.warning(
                    "[request_id=%s] tag apply %r failed (non-fatal): %s",
                    ctx.request_id, tag, exc,
                )

    def _resolve_injected_value(self, ctx: RequestContext, requested_type: Any, model_id: str, inj: InjectionSpec) -> Any:
        qn = _type_qualname(requested_type)
        rm: Optional[RunMetricsV1] = getattr(ctx, "_run_metrics", None)

        # diffusers pipeline injection via existing model manager (torch-only).
        if self._model_manager is not None:
            pipe = None
            try:
                # Prefer get_for_inference() for thread-safe pipeline access
                # (creates fresh scheduler to avoid concurrent access issues)
                if hasattr(self._model_manager, 'get_for_inference'):
                    pipe = self._model_manager.get_for_inference(model_id)
                elif hasattr(self._model_manager, 'get_active_pipeline'):
                    pipe = asyncio.run(self._model_manager.get_active_pipeline(model_id))
            except Exception:
                pipe = None
            if pipe is not None:
                if rm is not None and model_id:
                    try:
                        # Issue #17: route through the provider index so HF refs
                        # don't get mis-parsed as tensorhub here.
                        prov = self._provider_for_ref(str(model_id))
                        parsed = parse_model_ref(str(model_id), provider=prov)
                        canon = parsed.tensorhub.canonical() if parsed.provider == "tensorhub" and parsed.tensorhub is not None else str(model_id)
                        model_metrics = rm.models.get(canon)
                        rm.set_initial_model_state(
                            canon,
                            "hot_vram",
                            model_metrics.snapshot_digest if model_metrics is not None else None,
                        )
                    except Exception:
                        pass
                if isinstance(requested_type, type) and not isinstance(pipe, requested_type):
                    expected_qn = _type_qualname(requested_type)
                    got_qn = _type_qualname(type(pipe))
                    raise ValueError(
                        f"model injection type mismatch for {inj.param_name}: expected {expected_qn}, got {got_qn} (model_id={model_id})"
                    )
                return pipe

        # Transformers-style injection (AutoModel/AutoProcessor/etc) and any other
        # libraries with a `from_pretrained` factory.
        # We treat these as worker-owned cached handles: load once, reuse across invocations.
        if hasattr(requested_type, "from_pretrained") and callable(getattr(requested_type, "from_pretrained", None)):
            qn = _type_qualname(requested_type)
            key = (model_id, qn)
            lock = self._custom_runtime_locks.setdefault(key, threading.Lock())
            with lock:
                cached = self._custom_runtime_cache.get(key)
                if cached is not None:
                    return cached

                # Diffusers pipelines: prefer downloading via the worker's model downloader
                # and loading from a local directory, instead of letting diffusers /
                # huggingface_hub fetch the bare ref directly.
                #
                # This keeps tenant code inference-only while letting the platform handle
                # provider-aware caching + presigned-manifest resolution.
                obj = None
                is_diffusers_pipeline_type = False
                canon = ""
                try:
                    from diffusers import DiffusionPipeline
                except Exception:
                    DiffusionPipeline = None

                if (
                    obj is None
                    and DiffusionPipeline is not None
                    and isinstance(requested_type, type)
                    and issubclass(requested_type, DiffusionPipeline)
                ):
                    is_diffusers_pipeline_type = True
                    local = None
                    canon = str(model_id)
                    try:
                        # Issue #17: provider-aware decoding so HF refs reach
                        # the hf-branch caching path with the right canonical
                        # key (the bare repo_id rather than a tensorhub
                        # tag-derived alias).
                        prov = self._provider_for_ref(str(model_id))
                        parsed = parse_model_ref(str(model_id), provider=prov)
                        canon = parsed.tensorhub.canonical() if parsed.provider == "tensorhub" and parsed.tensorhub is not None else str(model_id)
                    except Exception:
                        canon = str(model_id)

                    # If this pipeline is already hot in VRAM, reuse it directly.
                    # This is crucial for multi-model functions (e.g. SDXL checkpoint router),
                    # otherwise the worker will steadily accumulate VRAM allocations and OOM.
                    try:
                        if self._model_cache is not None:
                            cached = self._model_cache.get_pipeline(canon)
                            if cached is not None:
                                if rm is not None and canon:
                                    try:
                                        model_metrics = rm.models.get(canon)
                                        rm.set_initial_model_state(
                                            canon,
                                            "hot_vram",
                                            model_metrics.snapshot_digest if model_metrics is not None else None,
                                        )
                                    except Exception:
                                        pass
                                if isinstance(requested_type, type) and not isinstance(cached, requested_type):
                                    expected_qn = _type_qualname(requested_type)
                                    got_qn = _type_qualname(type(cached))
                                    raise ValueError(
                                        f"model injection type mismatch for {inj.param_name}: expected {expected_qn}, got {got_qn} (model_id={model_id})"
                                    )
                                return cached
                    except Exception:
                        pass

                    try:
                        p = Path(model_id)
                        if p.exists():
                            local = p.as_posix()
                    except Exception:
                        local = None

                    if local is None:
                        if self._downloader is None:
                            raise ValueError("diffusers pipeline injection requires a model downloader")
                        cache_dir = str(tensorhub_cas_dir())
                        # Best-effort download timing.
                        t_dl0 = time.monotonic()
                        local = self._downloader.download(model_id, cache_dir)
                        t_dl_ms = int((time.monotonic() - t_dl0) * 1000)

                        warm = False
                        try:
                            warm = canon in set(self._model_cache.get_disk_models())
                        except Exception:
                            warm = False

                        # If we have an orchestrator-resolved manifest, estimate missing bytes.
                        resolved_entry = None
                        try:
                            resolved_entry = (getattr(ctx, "resolved_repos_by_id", None) or {}).get(canon)
                        except Exception:
                            resolved_entry = None
                        bytes_dl = None
                        if resolved_entry is not None:
                            bytes_dl = best_effort_bytes_downloaded(Path(cache_dir), resolved_entry)
                        if rm is not None:
                            try:
                                rm.add_fetch_time(canon, 0 if warm else t_dl_ms, 0 if warm else bytes_dl)
                            except Exception:
                                pass

                        # Ensure the shared cache has a durable snapshot under
                        # ${TENSORHUB_CACHE_DIR}/cas before any NFS->local
                        # localization happens, so future pods can warm-start.
                        try:
                            lp = Path(local or "")
                            if lp.exists():
                                try:
                                    self._model_cache.mark_cached_to_disk(canon, lp)
                                except Exception:
                                    pass
                                if not warm:
                                    # #321: typed signal.
                                    self._emit_model_ready(canon, self._MODEL_AVAILABILITY_CACHED)
                        except Exception:
                            pass
                    else:
                        # Model was provided as a local path; best-effort register it as disk-cached
                        # so VRAM eviction can keep a disk pointer for warm reloads.
                        try:
                            lp = Path(local or "")
                            if lp.exists() and self._model_cache is not None:
                                self._model_cache.mark_cached_to_disk(canon, lp)
                        except Exception:
                            pass

                    # Detect mount backend (NFS vs local) and localize snapshot to local disk if needed.
                    try:
                        from .pipeline.mount_backend import mount_backend_for_path

                        src_path = Path(local or "")
                        mb_src = mount_backend_for_path(src_path)
                        if mb_src is not None and rm is not None:
                            rm.set_model_disk_backend(
                                canon,
                                disk_fstype=mb_src.fstype,
                                disk_backend="nfs" if mb_src.is_nfs else "local",
                                localized=False,
                            )

                        if mb_src is not None and mb_src.is_nfs and src_path.exists() and src_path.is_dir():
                            lcache = self._get_local_model_cache()
                            if lcache is not None:
                                t_cp0 = time.monotonic()
                                cached_path, bytes_copied = lcache.cache_model_blocking_with_stats(canon, src_path)
                                cp_ms = int((time.monotonic() - t_cp0) * 1000)

                                # Update metrics with effective backend used for loading.
                                if rm is not None:
                                    try:
                                        mb_eff = mount_backend_for_path(cached_path)
                                        kw: Dict[str, Any] = {
                                            "disk_fstype": mb_eff.fstype if mb_eff is not None else None,
                                            "disk_backend": "local",
                                            "localized": True,
                                        }
                                        if bytes_copied is not None:
                                            kw["nfs_to_local_copy_ms"] = cp_ms
                                            kw["bytes_copied"] = bytes_copied
                                        rm.set_model_disk_backend(canon, **kw)
                                    except Exception:
                                        pass

                                local = str(cached_path)
                    except Exception:
                                pass

                    if local is None:
                        raise ValueError(f"diffusers pipeline local path is empty for model {model_id!r}")
                    local_path = Path(local)

                    kwargs: dict[str, Any] = {}

                    # Cozy pipeline YAML is authoritative; ensure diffusers can load even
                    # if the artifact only shipped cozy.pipeline.lock.yaml/yaml (no model_index.json).
                    try:
                        from gen_worker.pipeline.spec import (
                            cozy_custom_pipeline_arg,
                            ensure_diffusers_model_index_json,
                            load_cozy_pipeline_spec,
                        )

                        root = local_path
                        spec = load_cozy_pipeline_spec(root)
                        if spec is not None:
                            _ = ensure_diffusers_model_index_json(root)
                            try:
                                custom_pipeline = cozy_custom_pipeline_arg(root, spec)
                                if custom_pipeline:
                                    kwargs["custom_pipeline"] = custom_pipeline
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        from gen_worker.pipeline.loader import detect_diffusers_variant

                        variant = detect_diffusers_variant(local_path)
                        if variant is not None:
                            kwargs["variant"] = variant
                    except Exception:
                        pass

                    # Quantized weight-only inference requires explicit loader hints.
                    #
                    # Cozy Hub/orchestrator controls whether a function may select fp8 variants, but
                    # the worker must still pass an explicit quantization_config so diffusers loads
                    # quantized modules correctly (torchao-backed).
                    variant = str(kwargs.get("variant") or "").strip().lower()
                    if variant in ("fp8", "int8", "int4"):
                        # Ensure torchao is present (actual quantization implementation).
                        import importlib.util as _ilu

                        if _ilu.find_spec("torchao") is None:
                            raise ValueError(
                                f"{variant} diffusers variant selected, but torchao is not installed in this worker image"
                            )

                        try:
                            from diffusers.quantizers import PipelineQuantizationConfig
                            from diffusers import TorchAoConfig as DiffusersTorchAoConfig
                        except Exception as e:
                            raise ValueError(
                                f"{variant} diffusers variant selected, but diffusers torchao quantization hooks are unavailable"
                            ) from e

                        quant_kind = {
                            "fp8": "float8_weight_only",
                            "int8": "int8_weight_only",
                            "int4": "int4_weight_only",
                        }[variant]

                        root = local_path
                        quant_mapping: dict[str, Any] = {}
                        if (root / "transformer").exists():
                            quant_mapping["transformer"] = DiffusersTorchAoConfig(quant_kind)
                        if (root / "unet").exists():
                            quant_mapping["unet"] = DiffusersTorchAoConfig(quant_kind)
                        if not quant_mapping:
                            raise ValueError(
                                f"{variant} diffusers variant selected, but no quantizable component directories were found under {root}"
                            )
                        kwargs["quantization_config"] = PipelineQuantizationConfig(quant_mapping=quant_mapping)

                    # Choose a dtype that won't explode RAM on CPU. Prefer matching the variant.
                    try:
                        if torch is not None:
                            device_is_cuda = str(ctx.device).startswith("cuda") and torch.cuda.is_available()
                            variant = str(kwargs.get("variant") or "").strip().lower()
                            if device_is_cuda:
                                bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
                                if variant in ("fp8", "int8", "int4") and bf16_supported:
                                    kwargs["torch_dtype"] = torch.bfloat16
                                elif variant == "bf16" and bf16_supported:
                                    kwargs["torch_dtype"] = torch.bfloat16
                                else:
                                    # Fall back to model-name heuristic (catches flux, z-image, sd3, etc.)
                                    try:
                                        from gen_worker.pipeline.loader import get_torch_dtype as _get_torch_dtype
                                        kwargs["torch_dtype"] = _get_torch_dtype(None, canon or model_id)
                                    except Exception:
                                        kwargs["torch_dtype"] = torch.float16
                            elif variant == "fp16":
                                kwargs["torch_dtype"] = torch.float16
                            elif variant == "bf16":
                                kwargs["torch_dtype"] = torch.bfloat16
                            else:
                                kwargs["torch_dtype"] = torch.float32
                    except Exception:
                        pass

                    try:
                        t_pi0 = time.monotonic()
                        from_pretrained = getattr(requested_type, "from_pretrained")
                        obj = from_pretrained(local, **kwargs)
                        if rm is not None:
                            rm.add_pipeline_init_time(int((time.monotonic() - t_pi0) * 1000))
                    except OSError as e:
                        # Many Stable Diffusion repos include safety_checker/feature_extractor
                        # components that are often omitted from "minimal" downloads. Retry with
                        # those components disabled so the pipeline can still load.
                        msg = str(e).lower()
                        disable_optional_components = any(
                            s in msg
                            for s in (
                                "model.safetensors",
                                "pytorch_model.bin",
                                "preprocessor_config.json",
                                "feature_extractor",
                                "image processor",
                                "image_processor",
                                "safety_checker",
                            )
                        )
                        if disable_optional_components:
                            kwargs2 = dict(kwargs)
                            kwargs2.setdefault("safety_checker", None)
                            kwargs2.setdefault("feature_extractor", None)
                            kwargs2.setdefault("image_processor", None)
                            kwargs2.setdefault("requires_safety_checker", False)
                            t_pi0 = time.monotonic()
                            from_pretrained = getattr(requested_type, "from_pretrained")
                            obj = from_pretrained(local, **kwargs2)
                            if rm is not None:
                                rm.add_pipeline_init_time(int((time.monotonic() - t_pi0) * 1000))
                        else:
                            raise

                if obj is None:
                    t_pi0 = time.monotonic()
                    from_pretrained = getattr(requested_type, "from_pretrained")
                    model_source: str = str(model_id)
                    preload_kwargs: dict[str, Any] = {}
                    try:
                        p = Path(model_source)
                        if p.exists():
                            model_source = p.as_posix()
                        else:
                            # Issue #17: provider-aware parse so the right
                            # branch fires for HF / civitai bindings.
                            prov = self._provider_for_ref(model_source)
                            parsed = parse_model_ref(model_source, provider=prov)
                            if self._downloader is not None and parsed.provider in ("tensorhub", "hf"):
                                model_source = self._downloader.download(model_source, str(tensorhub_cas_dir()))
                            elif parsed.provider == "hf" and parsed.hf is not None:
                                # Fallback path when downloader is unavailable.
                                model_source = parsed.hf.repo_id
                                if parsed.hf.revision:
                                    preload_kwargs["revision"] = parsed.hf.revision
                    except Exception:
                        model_source = str(model_id)
                        preload_kwargs = {}

                    logger.info(
                        "Loading from_pretrained: source=%s type=%s kwargs=%s",
                        model_source, _type_qualname(requested_type), list(preload_kwargs.keys()),
                    )
                    obj = from_pretrained(model_source, **preload_kwargs)
                    logger.info(
                        "from_pretrained complete: source=%s (%.1fs)",
                        model_source, time.monotonic() - t_pi0,
                    )
                    if rm is not None:
                        rm.add_pipeline_init_time(int((time.monotonic() - t_pi0) * 1000))
                if isinstance(requested_type, type) and not isinstance(obj, requested_type):
                    expected_qn = _type_qualname(requested_type)
                    got_qn = _type_qualname(type(obj))
                    raise ValueError(
                        f"model injection type mismatch for {inj.param_name}: expected {expected_qn}, got {got_qn} (model_id={model_id})"
                    )
                # Best-effort move to worker device if supported.
                try:
                    if torch is not None and hasattr(obj, "to") and callable(getattr(obj, "to", None)):
                        # For diffusers pipelines, evict older VRAM pipelines before moving a new one
                        # to GPU to avoid transient OOMs during `.to("cuda")`.
                        try:
                            device_is_cuda = str(ctx.device).startswith("cuda") and torch.cuda.is_available()
                            if device_is_cuda and is_diffusers_pipeline_type and self._model_cache is not None and canon:
                                max_keep = 2
                                if max_keep > 0 and not self._model_cache.is_in_vram(canon):
                                    self._model_cache.evict_lru_vram_until_count(max_keep - 1)
                        except Exception:
                            pass

                        t_to0 = time.monotonic()
                        mem_before = 0
                        try:
                            if torch.cuda.is_available():
                                mem_before = int(torch.cuda.memory_allocated())
                        except Exception:
                            mem_before = 0

                        # Prefer moving with an explicit dtype to avoid mixed fp16/fp32 modules
                        # (can cause runtime errors like: input half + bias float).
                        torch_dtype = None
                        try:
                            torch_dtype = kwargs.get("torch_dtype") if isinstance(kwargs, dict) else None
                        except Exception:
                            torch_dtype = None

                        # Low-VRAM preflight: if the model won't fit in VRAM, skip
                        # moving to CUDA and apply model/group offload directly.
                        # Also handles escalating offload on OOM during .to().
                        from .inference_memory import (
                            apply_low_vram_config as _apply_low_vram_config,
                            estimate_pipeline_size_gb as _estimate_pipeline_size_gb,
                            flush_memory as _flush_memory,
                            get_available_vram_gb as _get_available_vram_gb,
                            get_total_vram_gb as _get_total_vram_gb,
                        )

                        device_is_cuda_target = str(ctx.device).startswith("cuda") and torch.cuda.is_available()
                        preflight_offload_mode: Optional[str] = None
                        if device_is_cuda_target and is_diffusers_pipeline_type:
                            model_gb = _estimate_pipeline_size_gb(obj)
                            total_vram = _get_total_vram_gb()
                            free_vram = _get_available_vram_gb()
                            safety_margin = 2.0
                            if model_gb > 0 and model_gb > max(0.0, free_vram - safety_margin):
                                logger.warning(
                                    "low_vram preflight: model=%s size=%.1fGB free_vram=%.1fGB total_vram=%.1fGB -> enabling offload before .to()",
                                    model_id, model_gb, free_vram, total_vram,
                                )
                                # Pick the right offload based on how tight we are.
                                if total_vram > 0 and total_vram <= 6.0:
                                    preflight_offload_mode = "group_offload"
                                else:
                                    preflight_offload_mode = "model_offload"

                        if preflight_offload_mode is not None:
                            applied = _apply_low_vram_config(
                                obj, mode=preflight_offload_mode, logger=logger,
                                model_size_gb=_estimate_pipeline_size_gb(obj),
                            )
                            try:
                                self._emit_worker_event_bytes("", "low_vram_mode_applied", json.dumps({
                                    "model_id": model_id,
                                    "stage": "preflight",
                                    "requested_mode": preflight_offload_mode,
                                    **{k: v for k, v in applied.items() if isinstance(v, (bool, str))},
                                }).encode("utf-8"))
                            except Exception:
                                pass
                            # Do NOT call .to(cuda) after offload hooks are installed.
                            if rm is not None:
                                rm.add_gpu_load_time(int((time.monotonic() - t_to0) * 1000))
                        else:
                            logger.info(
                                "Moving model to device=%s dtype=%s model=%s ...",
                                str(ctx.device), torch_dtype, model_id,
                            )
                            _move_attempts = 0
                            while True:
                                try:
                                    if torch_dtype is not None:
                                        obj = obj.to(str(ctx.device), dtype=torch_dtype)
                                    else:
                                        obj = obj.to(str(ctx.device))
                                    break
                                except TypeError:
                                    obj = obj.to(str(ctx.device))
                                    break
                                except torch.cuda.OutOfMemoryError as _oom:
                                    _move_attempts += 1
                                    _flush_memory()
                                    if _move_attempts >= 3 or not is_diffusers_pipeline_type:
                                        logger.error(
                                            "OOM moving %s to CUDA after %d attempt(s); giving up",
                                            model_id, _move_attempts,
                                        )
                                        raise
                                    # Escalate: 1st OOM -> model_offload; 2nd -> sequential.
                                    escalate_to = "model_offload" if _move_attempts == 1 else "sequential"
                                    logger.warning(
                                        "OOM moving %s to CUDA (attempt %d); escalating to %s",
                                        model_id, _move_attempts, escalate_to,
                                    )
                                    applied = _apply_low_vram_config(
                                        obj, mode=escalate_to, logger=logger,
                                    )
                                    try:
                                        self._emit_worker_event_bytes("", "low_vram_mode_applied", json.dumps({
                                            "model_id": model_id,
                                            "stage": "oom_escalation",
                                            "attempt": _move_attempts,
                                            "requested_mode": escalate_to,
                                            **{k: v for k, v in applied.items() if isinstance(v, (bool, str))},
                                        }).encode("utf-8"))
                                    except Exception:
                                        pass
                                    # Offload hooks are installed; skip .to(cuda) and exit loop.
                                    preflight_offload_mode = escalate_to
                                    break
                            logger.info(
                                "Model moved to device=%s successfully model=%s",
                                str(ctx.device), model_id,
                            )
                            if rm is not None:
                                rm.add_gpu_load_time(int((time.monotonic() - t_to0) * 1000))

                        # Baseline VAE/attention-slicing pass for diffusers pipelines
                        # that did not hit an offload path above.
                        if is_diffusers_pipeline_type and preflight_offload_mode is None:
                            try:
                                applied = _apply_low_vram_config(
                                    obj, mode="auto", logger=logger,
                                )
                                try:
                                    self._emit_worker_event_bytes("", "low_vram_mode_applied", json.dumps({
                                        "model_id": model_id,
                                        "stage": "baseline",
                                        **{k: v for k, v in applied.items() if isinstance(v, (bool, str))},
                                    }).encode("utf-8"))
                                except Exception:
                                    pass
                            except Exception as _lv_exc:
                                logger.debug("apply_low_vram_config(auto) failed: %s", _lv_exc)

                        # Cache diffusers pipelines in the worker's ModelCache (LRU + heartbeats)
                        # instead of the generic from_pretrained object cache, otherwise VRAM
                        # grows unbounded across many model_key requests.
                        try:
                            device_is_cuda = str(ctx.device).startswith("cuda") and torch.cuda.is_available()
                            if is_diffusers_pipeline_type and self._model_cache is not None and canon:
                                size_gb = 0.0
                                if device_is_cuda:
                                    try:
                                        mem_after = int(torch.cuda.memory_allocated())
                                        delta = max(0, mem_after - mem_before)
                                        size_gb = float(delta) / float(1024 ** 3)
                                    except Exception:
                                        size_gb = 0.0
                                    if size_gb <= 0.1:
                                        # Fallback heuristic when deltas are noisy.
                                        size_gb = 10.0
                                self._model_cache.mark_loaded_to_vram(canon, obj, size_gb)
                                logger.info(
                                    "pipeline injection resolved: model=%s size_gb=%.1f device=%s",
                                    canon, size_gb, str(ctx.device),
                                )
                                return obj
                        except Exception as _cache_exc:
                            logger.warning("model_cache mark_loaded_to_vram failed: %s", _cache_exc)
                except Exception as _to_exc:
                    logger.error("failed to move pipeline to device=%s: %s", str(ctx.device), _to_exc)
                    raise
                self._custom_runtime_cache[key] = obj
                return obj

        raise ValueError(f"no injection provider for type {qn} (model_id={model_id})")

    def _resolved_models_for_request(self, request: Any) -> Dict[str, Dict[str, Any]]:
        """Extract the orchestrator-stamped ``resolved_models`` map from a
        JobExecutionRequest envelope.

        The orchestrator validates caller-supplied ``_models`` overrides
        against each function's per-param binding (allow_override + pipeline
        class allowlist), drops the field from the payload, and stamps the
        resolved (ref, tag, flavor) triple onto the envelope. The worker
        consumes the map here.

        Returns ``{}`` when no overrides were stamped.
        """
        try:
            raw = getattr(request, "resolved_models", None)
        except Exception:
            return {}
        if not raw:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        # Two accepted shapes: protobuf map<string, ResolvedModel> or
        # plain Python dict (test fixtures). Both flow through msgspec.to_builtins
        # cleanly for the common case.
        try:
            iter_items = list(raw.items())
        except AttributeError:
            return {}
        for param_name, entry in iter_items:
            ref = ""
            tag = "prod"
            flavor = ""
            if isinstance(entry, dict):
                ref = str(entry.get("ref") or "").strip()
                tag = str(entry.get("tag") or "prod").strip() or "prod"
                flavor = str(entry.get("flavor") or "").strip()
            else:
                ref = str(getattr(entry, "ref", "") or "").strip()
                tag = str(getattr(entry, "tag", "") or "prod").strip() or "prod"
                flavor = str(getattr(entry, "flavor", "") or "").strip()
            if ref:
                out[str(param_name)] = {"ref": ref, "tag": tag, "flavor": flavor}
        return out

    def _resolve_model_id_for_injection(
        self,
        fn_name: str,
        inj: InjectionSpec,
        payload: msgspec.Struct,
        *,
        resolved_models: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> tuple[str, Optional[str]]:
        """Resolve the model id for a binding.

        Order:
          1. Orchestrator-stamped ``resolved_models[param_name]`` (caller
             override that has already passed the orchestrator's allowlist).
          2. Binding default — fixed ``(ref, flavor, tag)`` or dispatch table
             lookup on the discriminator field.
        """
        resolved_models = resolved_models or {}

        # 1. Caller override (already validated by the orchestrator).
        override = resolved_models.get(inj.param_name)
        if override:
            if not inj.binding._allow_override:
                # Defense-in-depth — orchestrator should already have rejected.
                raise ValueError(
                    f"function {fn_name!r} param {inj.param_name!r}: binding has no "
                    "allow_override declared, but resolved_models was stamped — orchestrator drift?"
                )
            ref = override.get("ref", "")
            flavor = override.get("flavor", "")
            tag = override.get("tag", "prod")
            model_id = _resolved_repo_id(ref, flavor=flavor, tag=tag)
            return model_id, None

        # 2. Binding default.
        binding = inj.binding
        # The wire format is bare ref + explicit provider field. Internal
        # canonical model_id keys discriminate by provider so the same
        # `owner/repo` on cozy vs. hf stays disjoint in the cache.
        if isinstance(binding, (Repo, HFRepo, CivitaiRepo)):
            model_id = _resolved_repo_id(
                binding.ref, flavor=binding._flavor, tag=binding._tag, provider=binding.provider
            )
            return model_id, None

        if isinstance(binding, Dispatch):
            # Discriminator field → table key → repo pick.
            try:
                chosen = getattr(payload, binding.field)
            except Exception:
                raise ValueError(
                    f"function {fn_name!r}: dispatch field {binding.field!r} missing on payload"
                ) from None
            if not isinstance(chosen, str):
                raise ValueError(
                    f"function {fn_name!r}: dispatch field {binding.field!r} must be a string, "
                    f"got {type(chosen).__name__}"
                )
            key = chosen.strip()
            pick = binding.table.get(key)
            if pick is None:
                allowed = sorted(binding.table.keys())
                raise ValueError(
                    f"function {fn_name!r}: dispatch key {key!r} not in table {allowed!r}"
                )
            model_id = _resolved_repo_id(
                pick.ref, flavor=pick._flavor, tag=pick._tag, provider=pick.provider
            )
            return model_id, key

        raise TypeError(f"unknown binding type: {type(binding).__name__}")

    def _enforce_model_allowlist(self, model_id: str, inj: InjectionSpec, *, allowed_ids: Optional[set[str]] = None) -> None:
        # Enforce BOTH:
        # - any per-function mapping restriction (allowed_ids), AND
        # - the release-level allowlist from the scheduler (defense-in-depth).
        if allowed_ids is not None and model_id not in allowed_ids:
            raise ValueError(f"model_id not allowed for function: {model_id!r} (injection param {inj.param_name})")
        if self._release_allowed_model_ids is not None and model_id not in self._release_allowed_model_ids:
            raise ValueError(f"model_id not allowed for release: {model_id!r} (injection param {inj.param_name})")

    def _send_request_result(
        self,
        request_id: str,
        success: bool,
        output_payload: Optional[bytes],
        error_type: str,
        retryable: bool,
        safe_message: str,
        error_message: str,
    ) -> None:
        """Send a request execution result back to the scheduler via the queue."""
        try:
            observation = self._build_job_observation(request_id, success, error_type)
            # #321: BatchExecutionItemResult branch removed alongside the
            # batch envelope; every request result is a typed JobExecutionResult.
            # error_message field removed too — safe_message is the canonical
            # client error string; error_type is the classification.
            result = pb.JobExecutionResult(
                request_id=request_id,
                success=success,
                output_payload=(output_payload or b'') if success else b'',
                error_type=error_type if not success else "",
                retryable=bool(retryable) if not success else False,
                safe_message=safe_message if not success else "",
                observation=observation,
            )
            self._send_message(pb.WorkerSchedulerMessage(job_result=result))
            logger.debug(f"Queued request result for request_id={request_id}, success={success}")
        except Exception as e:
             # This shouldn't generally fail unless message creation has issues
             logger.error(f"Failed to create or queue request result for request_id={request_id}: {e}")

    def _build_job_observation(self, request_id: str, success: bool, error_type: str) -> Any:
        with self._active_requests_lock:
            obs = dict(self._request_observations.pop(request_id, {}) or {})
        started_at = float(obs.get("started_at") or obs.get("enqueued_at") or time.monotonic())
        runtime_ms = max(0, int((time.monotonic() - started_at) * 1000))
        return pb.JobExecutionObservation(
            release_id=str(obs.get("release_id") or self.release_id or ""),
            function_name=str(obs.get("function_name") or ""),
            build_profile=str(obs.get("build_profile") or ""),
            image_digest=str(obs.get("image_digest") or ""),
            provider=str(obs.get("provider") or ("runpod" if self.runpod_pod_id else "local")),
            worker_id=str(obs.get("worker_id") or self.worker_id or ""),
            machine_class=str(obs.get("machine_class") or ""),
            status="succeeded" if success else "failed",
            error_type=str(error_type or ""),
            runtime_ms=runtime_ms,
            local_queue_ms=int(obs.get("local_queue_ms") or 0),
            peak_memory_bytes=int(obs.get("peak_memory_bytes") or 0),
            peak_vram_bytes=int(obs.get("peak_vram_bytes") or 0),
            active_count_at_start=int(obs.get("active_count_at_start") or 0),
            local_queued_count_at_start=int(obs.get("local_queued_count_at_start") or 0),
        )
