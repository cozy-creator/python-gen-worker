from __future__ import annotations

import grpc
import logging
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
from pathlib import Path
from dataclasses import dataclass
import collections.abc as cabc
from typing import Any, Callable, Dict, Optional, TypeVar, Iterator, List, Tuple, Iterable, get_args, get_origin
import hashlib
import msgspec
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
import asyncio

import jwt
from ._worker_auth import _JWKSCache
from .request_context import (
    RequestContext,
    _canonicalize_model_ref_string,
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
TaskExecutionRequest = Any
TaskExecutionResult = Any
from .api.decorators import ResourceRequirements
from .api.errors import AuthError, CanceledError, FatalError, ResourceError, RetryableError, ValidationError

from .models.interface import ModelManagementInterface
from .models.downloader import ModelDownloader
from .models.ref_downloader import ModelRefDownloader
from .models.refs import parse_model_ref
from .api.types import Asset, Tensors
from .models.cache import ModelCache
from .run_metrics_v1 import RunMetricsV1, best_effort_bytes_downloaded, best_effort_init_model_metrics, safe_json_bytes
from .models.cache_paths import worker_local_model_cache_dir_default, worker_model_cache_dir
from .wire_protocol import WIRE_PROTOCOL_MAJOR, WIRE_PROTOCOL_MINOR, wire_protocol_version_string
from .api.injection import (
    InjectionSpec,
    ModelRef,
    ModelRefSource,
    parse_injection,
    type_qualname,
)
from .discovery.names import slugify_function_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# Type variables for generic function signatures
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

# Generic type for action functions
ActionFunc = Callable[[Any, I], O]

HEARTBEAT_INTERVAL = 10  # seconds


def _workspace_scope_id(request_id: str, run_id: Optional[str]) -> str:
    rid = str(run_id or "").strip()
    if rid:
        return rid
    return str(request_id or "").strip()


def _extract_worker_capability_token(envelope: Any) -> str:
    return str(getattr(envelope, "worker_capability_token", "") or "").strip()


@dataclass(frozen=True)
class _TaskSpec:
    name: str
    func: Callable[..., Any]
    resources: ResourceRequirements
    ctx_param: str
    payload_param: str
    payload_type: type[msgspec.Struct]
    output_mode: str  # "single" | "incremental"
    output_type: Optional[type[msgspec.Struct]] = None
    delta_type: Optional[type[msgspec.Struct]] = None
    injections: Tuple[InjectionSpec, ...] = ()
    input_schema_json: bytes = b""
    output_schema_json: bytes = b""
    delta_schema_json: Optional[bytes] = None
    injection_json: bytes = b""


@dataclass(frozen=True)
class _WebsocketSpec:
    name: str
    func: Callable[..., Any]
    resources: ResourceRequirements
    ctx_param: str
    socket_param: str
    injections: Tuple[InjectionSpec, ...] = ()


class RealtimeSocket:
    """
    Worker-owned socket interface for realtime handlers (no FastAPI dependency).
    """

    async def send_bytes(self, data: bytes) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def send_json(self, obj: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def iter_bytes(self) -> typing.AsyncIterator[bytes]:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class _RealtimeSessionState:
    session_id: str
    spec: _WebsocketSpec
    ctx: RequestContext
    loop: asyncio.AbstractEventLoop
    in_q: "asyncio.Queue[Optional[bytes]]"
    closed: threading.Event


class _RealtimeSocketAdapter(RealtimeSocket):
    def __init__(self, worker: "Worker", session_id: str, loop: asyncio.AbstractEventLoop, in_q: "asyncio.Queue[Optional[bytes]]") -> None:
        self._worker = worker
        self._session_id = session_id
        self._loop = loop
        self._in_q = in_q
        self._closed = False

    async def send_bytes(self, data: bytes) -> None:
        if self._closed:
            return
        self._worker._send_message(
            pb.WorkerSchedulerMessage(
                realtime_frame=pb.RealtimeFrame(session_id=self._session_id, data=data, is_text=False)
            )
        )

    async def send_json(self, obj: Any) -> None:
        if self._closed:
            return
        data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self._worker._send_message(
            pb.WorkerSchedulerMessage(
                realtime_frame=pb.RealtimeFrame(session_id=self._session_id, data=data, is_text=True)
            )
        )

    async def iter_bytes(self) -> typing.AsyncIterator[bytes]:
        while True:
            item = await self._in_q.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._worker._send_message(
                pb.WorkerSchedulerMessage(
                    realtime_close_cmd=pb.RealtimeCloseCommand(session_id=self._session_id, reason="closed")
                )
            )
        except Exception:
            pass
# RequestContext, _JWKSCache, and related helpers live in their own modules.
# They are imported above via from ._worker_auth import ... and from .request_context import ...


# Define the interceptor class correctly
def _parse_manifest_model_mapping(mapping: Dict[str, Any]) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    ids: Dict[str, str] = {}
    specs: Dict[str, Dict[str, Any]] = {}
    for k, v in mapping.items():
        key = str(k).strip()
        if not key or not isinstance(v, dict):
            continue
        ref = _canonicalize_model_ref_string(str(v.get("ref") or "").strip())
        if not ref:
            continue
        dtypes = v.get("dtypes")
        ids[key] = ref
        specs[key] = {"ref": ref, "dtypes": [str(x) for x in dtypes if str(x).strip()] if isinstance(dtypes, list) else []}
    return ids, specs


class _AuthInterceptor(grpc.StreamStreamClientInterceptor):
    def __init__(self, token: str) -> None:
        self._token = token

    def intercept_stream_stream(self, continuation: Any, client_call_details: Any, request_iterator: Any) -> Any:
        metadata = list(client_call_details.metadata or [])
        metadata.append(('authorization', f'Bearer {self._token}'))
        new_details = client_call_details._replace(metadata=metadata)
        return continuation(new_details, request_iterator)

class Worker:
    """Worker implementation that connects to the scheduler via gRPC."""

    _JWT_ROTATE_EVENT_TYPES = ("worker.jwt.rotate", "worker_jwt.rotate")

    def __init__(
        self,
        scheduler_addr: str = "localhost:8080",
        scheduler_addrs: Optional[List[str]] = None,
        user_module_names: List[str] = ["functions"], # Add new parameter for user modules
        worker_id: Optional[str] = None,
        worker_jwt: str = "",
        use_tls: bool = False,
        reconnect_delay: float = 5,
        max_reconnect_attempts: int = 0,  # 0 means infinite retries
        lb_only_retries: bool = False,
        model_manager: Optional[ModelManagementInterface] = None, # Optional model manager
        downloader: Optional[ModelDownloader] = None,  # Optional model downloader
        manifest: Optional[Dict[str, Any]] = None,  # Optional manifest from build
    ) -> None:
        """Initialize a new worker.

        Args:
            scheduler_addr: Address of the scheduler service.
            scheduler_addrs: Optional list of seed scheduler addresses.
            user_module_names: List of Python module names containing user-defined @worker_function functions.
            worker_id: Unique ID for this worker (generated if not provided).
            worker_jwt: Worker-connect JWT (required).
            use_tls: Whether to use TLS for the connection.
            reconnect_delay: Seconds to wait between reconnection attempts.
            max_reconnect_attempts: Max reconnect attempts (0 = infinite).
            model_manager: Optional model manager.
            downloader: Optional model downloader.
            manifest: Optional manifest dict (baked in at build time) containing models, resources, etc.
        """
        self.scheduler_addr = scheduler_addr
        self.scheduler_addrs = self._normalize_scheduler_addrs(scheduler_addr, scheduler_addrs)
        self._process_started_monotonic = time.monotonic()
        self._registered_event = threading.Event()
        self._registration_watchdog_thread: Optional[threading.Thread] = None
        self._startup_timeout_triggered = False
        self._register_timeout_s = int(os.getenv("WORKER_REGISTER_TIMEOUT_S", "90") or "90")
        self._warn_model_resolve_s = float(os.getenv("WORKER_WARN_MODEL_RESOLVE_S", "30") or "30")
        self._warn_model_load_s = float(os.getenv("WORKER_WARN_MODEL_LOAD_S", "60") or "60")
        self._warn_inference_s = float(os.getenv("WORKER_WARN_INFERENCE_S", "60") or "60")
        self.user_module_names = user_module_names # Store module names
        self.worker_id = worker_id or f"py-worker-{os.getpid()}"
        self.worker_jwt = (worker_jwt or "").strip()
        if not self.worker_jwt:
            raise ValueError("WORKER_JWT is required (worker-connect JWT); refusing to run unauthenticated worker")
        self.use_tls = use_tls
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.max_input_bytes = int(os.getenv("WORKER_MAX_INPUT_BYTES", "0"))
        self.max_output_bytes = int(os.getenv("WORKER_MAX_OUTPUT_BYTES", "0"))

        self._jwks_url = os.getenv("SCHEDULER_JWKS_URL", "").strip()
        self._jwks_ttl_seconds = int(os.getenv("SCHEDULER_JWKS_TTL_SECONDS", "300"))
        self._jwt_issuer = os.getenv("SCHEDULER_JWT_ISSUER", "").strip()
        self._jwt_audience = os.getenv("SCHEDULER_JWT_AUDIENCE", "").strip()
        self._jwks_cache: Optional[_JWKSCache] = _JWKSCache(self._jwks_url, self._jwks_ttl_seconds) if self._jwks_url else None

        # Worker containers are treated as untrusted. Do not depend on RELEASE_ID/OWNER env vars.
        # Release + owner identity come from the scheduler-issued JWT and per-run gRPC envelopes.
        self.release_id = ""
        self.owner = ""
        self.runpod_pod_id = os.getenv("RUNPOD_POD_ID", "") # Read injected pod ID
        if not self.runpod_pod_id:
            logger.warning("RUNPOD_POD_ID environment variable not set for this worker!")

        logger.info(f"RUNPOD_POD_ID: {self.runpod_pod_id}")

        self._task_specs: Dict[str, _TaskSpec] = {}
        self._ws_specs: Dict[str, _WebsocketSpec] = {}
        self._active_tasks: Dict[str, RequestContext] = {}
        self._active_tasks_lock = threading.Lock()
        self._request_batch_context: Dict[str, Tuple[str, str]] = {}  # request_id -> (batch_id, item_id)
        self._request_batch_context_lock = threading.Lock()
        self.max_concurrency = int(os.getenv("WORKER_MAX_CONCURRENCY", "0"))
        self._drain_timeout_seconds = int(os.getenv("WORKER_DRAIN_TIMEOUT_SECONDS", "0"))
        self._draining = False
        self._discovered_resources: Dict[str, ResourceRequirements] = {} # Store resources per function
        self._function_schemas: Dict[str, Tuple[bytes, bytes, Optional[bytes], bytes]] = {}  # func_name -> (input_schema_json, output_schema_json, delta_schema_json, injection_json)
        self._runtime_batching_config_by_function: Dict[str, Dict[str, Any]] = {}
        self._runtime_batching_config_lock = threading.Lock()
        self._last_function_capabilities_hash = ""

        self._custom_runtime_cache: Dict[Tuple[str, str], Any] = {}  # (model_id, injected_type_qualname) -> runtime handle
        self._custom_runtime_locks: Dict[Tuple[str, str], threading.Lock] = {}

        # Local (non-NFS) cache for NFS->local snapshot localization.
        # Empty WORKER_LOCAL_MODEL_CACHE_DIR disables localization entirely.
        self._local_model_cache_dir = os.getenv(
            "WORKER_LOCAL_MODEL_CACHE_DIR",
            str(worker_local_model_cache_dir_default()),
        ).strip()
        self._local_model_cache: Optional[Any] = None
        self._local_model_cache_lock = threading.Lock()
        self._last_disk_inventory_hash: str = ""

        self._gpu_busy_lock = threading.Lock()
        # Busy is used by the scheduler as a cheap routing hint.
        # Use a refcount (not a boolean) so overlapping inference + model ops
        # cannot flip BUSY -> NOT BUSY prematurely.
        self._gpu_busy_refcount = 0
        self._is_gpu_busy = False  # legacy; derived from refcount in _get_gpu_busy_status()
        # Detect GPU availability once at startup
        self._has_gpu = False
        try:
            import torch
            self._has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            pass

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

        self._receive_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

        self._realtime_sessions: Dict[str, _RealtimeSessionState] = {}
        self._realtime_lock = threading.Lock()

        self._reconnect_delay_base = max(0, reconnect_delay)
        self._reconnect_delay_max = int(os.getenv("RECONNECT_MAX_DELAY", "60"))
        self._reconnect_jitter_seconds = float(os.getenv("RECONNECT_JITTER_SECONDS", "1.0"))
        self._lb_only_retries = lb_only_retries

        resolved_model_manager = model_manager
        if resolved_model_manager is None:
            model_manager_path = os.getenv("MODEL_MANAGER_CLASS", "").strip()
            if model_manager_path:
                try:
                    module_path, _, class_name = model_manager_path.partition(":")
                    if not module_path or not class_name:
                        raise ValueError("MODEL_MANAGER_CLASS must be in module:Class format")
                    module = importlib.import_module(module_path)
                    manager_cls = getattr(module, class_name)
                    resolved_model_manager = manager_cls()
                    logger.info(f"Loaded ModelManager from MODEL_MANAGER_CLASS={model_manager_path}")
                except Exception as e:
                    logger.exception(f"Failed to load MODEL_MANAGER_CLASS '{model_manager_path}': {e}")
        # Auto-wire: if no explicit manager and diffusers is available, use the
        # built-in DiffusersModelManager so LoadModelCommand works out of the box.
        if resolved_model_manager is None:
            try:
                import diffusers  # noqa: F401
                from .diffusers_model_manager import DiffusersModelManager
                resolved_model_manager = DiffusersModelManager()
                logger.info("Auto-wired DiffusersModelManager (diffusers detected)")
            except ImportError:
                pass
        self._model_manager = resolved_model_manager
        self._downloader = downloader
        if self._downloader is None:
            base_url = os.getenv("TENSORHUB_URL", "").strip()
            token = os.getenv("TENSORHUB_TOKEN", "").strip() or None
            self._downloader = ModelRefDownloader(
                cozy_base_url=base_url,
                cozy_token=token,
            )
        self._supported_model_ids_from_scheduler: Optional[List[str]] = None  # allowlist from scheduler (repo refs)
        self._required_variant_refs_from_scheduler: Optional[List[str]] = None  # warm-start pinned variants
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
        # Keys should be canonical model ref strings (e.g. "cozy:owner/repo@sha256:<digest>").
        self._resolved_cozy_models_by_id_baseline: Dict[str, Any] = {}
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._model_init_done_event = threading.Event() # To signal model init is complete

        # LRU model cache for tracking VRAM and disk-cached models
        self._model_cache = ModelCache()

        # Store manifest and initialize model config from it
        self._manifest = manifest
        if manifest and isinstance(manifest, dict):
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
        if isinstance(exc, CanceledError) or isinstance(exc, InterruptedError):
            return "canceled", False, "canceled", internal
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
        """Best-effort worker->scheduler WorkerEvent emitter (must never fail a run)."""
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
    ) -> bool:
        if not hasattr(pb, "IncrementalTokenDelta"):
            return False
        try:
            msg = pb.WorkerSchedulerMessage(
                incremental_token_delta=pb.IncrementalTokenDelta(
                    request_id=str(request_id or ""),
                    item_id=str(item_id or ""),
                    function_name=str(function_name or ""),
                    sequence=int(sequence),
                    timestamp_unix_ms=int(timestamp_unix_ms),
                    delta_text=str(delta_text or ""),
                    payload_json=bytes(payload_json or b"{}"),
                )
            )
            self._send_message(msg)
            return True
        except Exception:
            return False

    def _emit_incremental_done_typed(
        self,
        *,
        request_id: str,
        function_name: str,
        item_id: str,
        sequence: int,
        timestamp_unix_ms: int,
    ) -> bool:
        if not hasattr(pb, "IncrementalTokenStreamDone"):
            return False
        try:
            msg = pb.WorkerSchedulerMessage(
                incremental_token_stream_done=pb.IncrementalTokenStreamDone(
                    request_id=str(request_id or ""),
                    item_id=str(item_id or ""),
                    function_name=str(function_name or ""),
                    sequence=int(sequence),
                    timestamp_unix_ms=int(timestamp_unix_ms),
                )
            )
            self._send_message(msg)
            return True
        except Exception:
            return False

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
        if not hasattr(pb, "IncrementalTokenStreamError"):
            return False
        try:
            msg = pb.WorkerSchedulerMessage(
                incremental_token_stream_error=pb.IncrementalTokenStreamError(
                    request_id=str(request_id or ""),
                    item_id=str(item_id or ""),
                    function_name=str(function_name or ""),
                    sequence=int(sequence),
                    timestamp_unix_ms=int(timestamp_unix_ms),
                    error_message=str(error_message or ""),
                )
            )
            self._send_message(msg)
            return True
        except Exception:
            return False

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
        Emit a structured startup phase marker to logs and (best-effort) scheduler WorkerEvent.
        """
        payload: Dict[str, Any] = {
            "phase": str(phase or "").strip(),
            "status": str(status or "ok"),
            "worker_id": str(getattr(self, "worker_id", "") or ""),
            "scheduler_addr": str(getattr(self, "scheduler_addr", "") or ""),
            "pid": int(os.getpid()),
            "uid": int(os.getuid()) if hasattr(os, "getuid") else None,
            "gid": int(os.getgid()) if hasattr(os, "getgid") else None,
            "cwd": str(os.getcwd()),
            "elapsed_ms": int(max(0.0, (time.monotonic() - float(getattr(self, "_process_started_monotonic", time.monotonic()))) * 1000.0)),
        }
        payload.update({k: v for k, v in extra.items() if v is not None})
        try:
            logger.log(level, "worker.startup.phase %s", json.dumps(payload, separators=(",", ":"), sort_keys=True))
        except Exception:
            logger.log(level, "worker.startup.phase phase=%s status=%s", payload.get("phase"), payload.get("status"))
        if emit_worker_event:
            try:
                self._emit_worker_event_bytes("", "worker.startup.phase", safe_json_bytes(payload))
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

    def _emit_task_event(self, request_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
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
        Start a soft watchdog for long-running task phases.

        Emits `task.<phase>.stuck` if the timer fires.
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
            self._emit_task_event(request_id, f"task.{phase}.stuck", ev_payload)
            logger.warning(
                "task phase stuck request_id=%s phase=%s elapsed_ms=%d warn_after_s=%.1f payload=%s",
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


    def _set_gpu_busy_status(self, busy: bool, func_name_for_log: str = "") -> None:
        # Legacy setter kept for older call sites (none in-tree). Translate to refcount.
        if busy:
            self._gpu_busy_enter()
        else:
            self._gpu_busy_exit()


    def _get_gpu_busy_status(self) -> bool:
        lock = getattr(self, "_gpu_busy_lock", None)
        if lock is None:
            return bool(getattr(self, "_is_gpu_busy", False))
        with lock:
            ref = int(getattr(self, "_gpu_busy_refcount", 0) or 0)
            busy = ref > 0
            # Keep legacy boolean in sync for any external introspection.
            try:
                self._is_gpu_busy = busy
            except Exception:
                pass
            return busy

    def _gpu_busy_enter(self) -> None:
        if not bool(getattr(self, "_has_gpu", False)):
            return
        lock = getattr(self, "_gpu_busy_lock", None)
        if lock is None:
            return
        with lock:
            cur = int(getattr(self, "_gpu_busy_refcount", 0) or 0)
            self._gpu_busy_refcount = cur + 1
            self._is_gpu_busy = self._gpu_busy_refcount > 0

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
            self._is_gpu_busy = self._gpu_busy_refcount > 0

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
        """Discover and register functions marked with @worker_function / @worker_websocket."""
        logger.info("Discovering worker handlers in modules: %s...", self.user_module_names)
        discovered = 0

        for module_name in self.user_module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                logger.error("Could not import user module: %s", module_name)
                continue

            for _, obj in inspect.getmembers(module):
                if not inspect.isfunction(obj):
                    continue

                if getattr(obj, "_is_worker_function", False) is True:
                    try:
                        spec = self._inspect_task_spec(obj)
                    except Exception as exc:
                        logger.error("Skipping function '%s': %s", getattr(obj, "__name__", "<unknown>"), exc)
                        continue
                    if spec.name in self._task_specs or spec.name in self._ws_specs:
                        logger.warning("Handler name conflict for '%s'; skipping", spec.name)
                        continue
                    self._task_specs[spec.name] = spec
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

                if getattr(obj, "_is_worker_websocket", False) is True:
                    try:
                        ws_spec = self._inspect_websocket_spec(obj)
                    except Exception as exc:
                        logger.error("Skipping websocket '%s': %s", getattr(obj, "__name__", "<unknown>"), exc)
                        continue
                    if ws_spec.name in self._task_specs or ws_spec.name in self._ws_specs:
                        logger.warning("Handler name conflict for '%s'; skipping", ws_spec.name)
                        continue
                    self._ws_specs[ws_spec.name] = ws_spec
                    self._discovered_resources[ws_spec.name] = ws_spec.resources
                    discovered += 1
                    logger.info("Registered websocket: '%s'", ws_spec.name)

        if discovered == 0:
            logger.warning("No worker handlers found in modules: %s", self.user_module_names)
        else:
            logger.info("Discovery complete. Found %d handlers.", discovered)

    def _inspect_task_spec(self, func: Callable[..., Any]) -> _TaskSpec:
        python_name = func.__name__
        func_name = slugify_function_name(python_name)
        if not func_name:
            raise ValueError(f"{python_name}: function name cannot be normalized")
        resources: ResourceRequirements = getattr(func, "_worker_resources", ResourceRequirements())

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

        injections: list[InjectionSpec] = []
        payload_type: Optional[type[msgspec.Struct]] = None
        payload_param: Optional[str] = None
        for p in params[1:]:
            ann = hints.get(p.name)
            if ann is None:
                raise ValueError(f"missing type annotation for param: {p.name}")
            inj = parse_injection(ann)
            if inj is not None:
                base_t, model_ref = inj
                injections.append(InjectionSpec(param_name=p.name, param_type=base_t, model_ref=model_ref))
                continue
            if isinstance(ann, type) and issubclass(ann, msgspec.Struct):
                if payload_type is not None:
                    raise ValueError("must accept exactly one msgspec.Struct payload arg")
                payload_type = ann
                payload_param = p.name
                continue
            raise ValueError(f"unsupported param type (must be payload msgspec.Struct or Annotated injection): {p.name}={ann!r}")

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
            {
                "param": inj.param_name,
                "type": type_qualname(inj.param_type),
                "model_ref": {"source": inj.model_ref.source.value, "key": inj.model_ref.key},
            }
            for inj in injections
        ]
        injection_json = json.dumps(injection_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

        return _TaskSpec(
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

    def _inspect_websocket_spec(self, func: Callable[..., Any]) -> _WebsocketSpec:
        python_name = func.__name__
        func_name = slugify_function_name(python_name)
        if not func_name:
            raise ValueError(f"{python_name}: function name cannot be normalized")
        resources: ResourceRequirements = getattr(func, "_worker_resources", ResourceRequirements())
        if not inspect.iscoroutinefunction(func):
            raise ValueError("websocket handler must be async def")

        try:
            hints = typing.get_type_hints(func, globalns=func.__globals__, include_extras=True)
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}")

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if len(params) < 2:
            raise ValueError("websocket handler must accept (ctx: RequestContext, sock: RealtimeSocket, ...)")

        ctx_name = params[0].name
        if hints.get(ctx_name) is not RequestContext:
            raise ValueError("first argument must be ctx: RequestContext")

        # We do not enforce a concrete socket type here; it is worker-owned and may
        # be provided by the runtime. We only validate that the param exists.
        socket_name = params[1].name
        injections: list[InjectionSpec] = []
        for p in params[2:]:
            ann = hints.get(p.name)
            if ann is None:
                raise ValueError(f"missing type annotation for param: {p.name}")
            inj = parse_injection(ann)
            if inj is None:
                raise ValueError("websocket extra params must be Annotated injections")
            base_t, model_ref = inj
            if model_ref.source == ModelRefSource.PAYLOAD:
                raise ValueError("websocket handlers cannot use ModelRef(PAYLOAD, ...) (no payload for selection)")
            injections.append(InjectionSpec(param_name=p.name, param_type=base_t, model_ref=model_ref))

        return _WebsocketSpec(
            name=func_name,
            func=func,
            resources=resources,
            ctx_param=ctx_name,
            socket_param=socket_name,
            injections=tuple(injections),
        )

    def _infer_payload_type(
        self,
        func: Callable[..., Any],
        expects_pipeline: bool,
    ) -> Optional[type[msgspec.Struct]]:
        signature = inspect.signature(func)
        params = list(signature.parameters.values())
        expected_params = 3 if expects_pipeline else 2
        if len(params) != expected_params:
            logger.error(
                "Function '%s' has %d parameters but expected %d.",
                func.__name__,
                len(params),
                expected_params,
            )
            return None

        payload_param = params[-1]
        try:
            type_hints = typing.get_type_hints(func, globalns=func.__globals__)
        except Exception as exc:
            logger.error("Failed to resolve type hints for '%s': %s", func.__name__, exc)
            return None

        payload_type = type_hints.get(payload_param.name)
        if payload_type is None:
            logger.error("Function '%s' is missing a payload type annotation.", func.__name__)
            return None

        if not isinstance(payload_type, type) or not issubclass(payload_type, msgspec.Struct):
            logger.error(
                "Function '%s' payload type must be a msgspec.Struct, got %r.",
                func.__name__,
                payload_type,
            )
            return None

        return payload_type

    def _infer_return_type(self, func: Callable[..., Any]) -> Optional[type[msgspec.Struct]]:
        try:
            type_hints = typing.get_type_hints(func, globalns=func.__globals__)
        except Exception as exc:
            logger.error("Failed to resolve return type hints for '%s': %s", func.__name__, exc)
            return None

        return_type = type_hints.get("return")
        if return_type is None:
            logger.error("Function '%s' is missing a return type annotation.", func.__name__)
            return None

        if not isinstance(return_type, type) or not issubclass(return_type, msgspec.Struct):
            logger.error(
                "Function '%s' return type must be a msgspec.Struct, got %r.",
                func.__name__,
                return_type,
            )
            return None

        return return_type

    def _send_message(self, message: WorkerSchedulerMessage) -> None:
        """Add a message to the outgoing queue."""
        if self._running and not self._stop_event.is_set():
            try:
                self._outgoing_queue.put_nowait(message)
            except queue.Full:
                 logger.error("Outgoing message queue is full. Message dropped!")
        else:
            logger.warning("Attempted to send message while worker is stopping or stopped.")

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
            ref = f"runs/{ctx.request_id}/outputs/auto/{upload_idx:06d}-{leaf}"
            upload_idx += 1
            return _normalize_output_ref(ref)

        def _walk(v: Any) -> Any:
            if isinstance(v, Asset):
                local = str(getattr(v, "local_path", "") or "").strip()
                if not local:
                    return v
                ref = str(getattr(v, "ref", "") or "").strip() or _default_ref(local)
                saved = ctx.save_file(ref, local)
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
        base_dir = os.getenv("WORKER_RUN_DIR", "/tmp/tensorhub/run").rstrip("/")
        scope_id = _workspace_scope_id(ctx.request_id, getattr(ctx, "run_id", None))
        local_inputs_dir = os.path.join(base_dir, scope_id, "inputs")
        os.makedirs(local_inputs_dir, exist_ok=True)
        cache_dir = os.getenv("WORKER_CACHE_DIR", os.path.join(base_dir, "cache")).rstrip("/")
        os.makedirs(cache_dir, exist_ok=True)

        max_bytes = int(os.getenv("WORKER_MAX_INPUT_FILE_BYTES", str(200 * 1024 * 1024)))

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
            raise RuntimeError("input file too large")

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
        asset.mime_type = mime or None
        asset.size_bytes = size or None
        asset.sha256 = sha256_hex or None

    def _download_url_to_file(self, src: str, dst: str, max_bytes: int, token: Optional[str] = None) -> Tuple[int, str, Optional[str]]:
        attempts = int(os.getenv("WORKER_DOWNLOAD_RETRIES", "3"))
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
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise RuntimeError("input file too large")
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
                # TODO: Add proper credential loading if needed
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

            # Start the bidirectional stream
            request_iterator = self._outgoing_message_iterator()
            self._stream = self._stub.ConnectWorker(request_iterator)

            logger.info(f"Attempting to connect to scheduler at {self.scheduler_addr}...")

            # Send initial registration immediately
            self._register_worker(is_heartbeat=False)
            self._registered_event.set()
            self._emit_startup_phase("registered", status="ok", scheduler_addr=self.scheduler_addr)

            # Start the receive loop in a separate thread *after* stream is initiated
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            logger.info(f"Successfully connected to scheduler at {self.scheduler_addr}")
            self._emit_startup_phase("ready", status="ok", scheduler_addr=self.scheduler_addr)
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
                logger.warning(f"Scheduler returned not_leader for {self.scheduler_addr}; redirecting to {leader}")
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

    def _outgoing_message_iterator(self) -> Iterator[WorkerSchedulerMessage]:
        """Yields messages from the outgoing queue to send to the scheduler."""
        while not self._stop_event.is_set():
            try:
                # Block for a short time to allow stopping gracefully
                message = self._outgoing_queue.get(timeout=0.1)
                yield message
                # self._outgoing_queue.task_done() # Not needed if not joining queue
            except queue.Empty:
                continue
            except Exception as e:
                 if not self._stop_event.is_set():
                     logger.exception(f"Error in outgoing message iterator: {e}")
                     self._handle_connection_error()
                     break # Exit iterator on error

    def _heartbeat_loop(self) -> None:
        """Periodically sends heartbeat messages."""
        while not self._stop_event.wait(HEARTBEAT_INTERVAL):
            try:
                self._register_worker(is_heartbeat=True)
                logger.debug("Sent heartbeat to scheduler")
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error sending heartbeat: {e}")
                    self._handle_connection_error()
                    break # Stop heartbeating on error

    def _register_worker(self, is_heartbeat: bool = False) -> None:
        """Create and send a registration/heartbeat message."""
        try:
            mem = psutil.virtual_memory()
            cpu_cores = os.cpu_count() or 0

            gpu_count = 0
            gpu_total_mem = 0
            vram_models = []
            gpu_used_mem = 0
            gpu_free_mem = 0
            gpu_name = ""
            gpu_driver = ""

            if torch and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    try:
                        props = torch.cuda.get_device_properties(0)
                        gpu_total_mem = props.total_memory
                        gpu_used_mem = torch.cuda.memory_allocated(0)
                        gpu_name = props.name
                        gpu_driver = torch.version.cuda or ""
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info(0)
                            gpu_total_mem = total_mem
                            gpu_used_mem = total_mem - free_mem
                            gpu_free_mem = free_mem
                        except Exception:
                            pass
                        logger.debug(f"GPU: {props.name}, VRAM total={gpu_total_mem}, used={gpu_used_mem}, cuda={torch.version.cuda}")
                    except Exception as gpu_err:
                         logger.warning(f"Could not get GPU properties: {gpu_err}")

            fake_gpu_count = os.getenv("WORKER_FAKE_GPU_COUNT")
            if fake_gpu_count:
                try:
                    gpu_count = int(fake_gpu_count)
                    if gpu_count > 0:
                        fake_mem = int(os.getenv("WORKER_FAKE_GPU_MEMORY_BYTES", str(24 * 1024 * 1024 * 1024)))
                        gpu_total_mem = fake_mem
                        gpu_used_mem = 0
                        gpu_free_mem = fake_mem
                        gpu_name = os.getenv("WORKER_FAKE_GPU_NAME", "FakeGPU")
                        gpu_driver = os.getenv("WORKER_FAKE_GPU_DRIVER", "fake")
                except ValueError:
                    logger.warning("Invalid WORKER_FAKE_GPU_COUNT; ignoring fake GPU override.")

            supports_model_loading_flag = False
            disk_models: List[str] = []  # Models on disk, ready to load
            downloading_models: List[str] = []  # Models being downloaded

            if self._model_manager:
                vram_models = self._model_manager.get_vram_loaded_models()
                supports_model_loading_flag = True
            elif self._model_cache:
                # Use model cache for VRAM-loaded models if no legacy model_manager
                vram_models = self._model_cache.get_vram_models()
                supports_model_loading_flag = True

            # Get disk-cached and downloading models from model cache
            if self._model_cache:
                disk_models = self._model_cache.get_disk_models()
                stats = self._model_cache.get_stats()
                downloading_models = stats.downloading_models
                # Log model cache stats for debugging
                logger.debug(
                    f"Model cache: vram={len(vram_models)}, disk={len(disk_models)}, "
                    f"downloading={len(downloading_models)}"
                ) 

            cuda_version = os.getenv("WORKER_CUDA_VERSION", "").strip()
            torch_version = os.getenv("WORKER_TORCH_VERSION", "").strip()
            if torch is not None:
                if not torch_version:
                    torch_version = getattr(torch, "__version__", "") or ""
                if not cuda_version:
                    cuda_version = getattr(torch.version, "cuda", "") or ""
            if not cuda_version:
                cuda_version = os.getenv("CUDA_VERSION", "").strip() or os.getenv("NVIDIA_CUDA_VERSION", "").strip()

            gpu_sm = ""
            installed_libs: List[str] = []
            try:
                from .models.hub_policy import detect_worker_capabilities

                caps = detect_worker_capabilities()
                installed_libs = list(caps.installed_libs or [])
                if caps.gpu_sm:
                    gpu_sm = str(int(caps.gpu_sm))
                if not cuda_version:
                    cuda_version = str(caps.cuda_version or "")
                if not torch_version:
                    torch_version = str(caps.torch_version or "")
            except Exception:
                pass

            function_schemas = []
            for fname, (in_schema, out_schema, _delta_schema, inj_json) in self._function_schemas.items():
                try:
                    spec = self._task_specs.get(fname)
                    incremental = bool(spec and spec.output_mode == "incremental")
                    function_schemas.append(
                        pb.FunctionSchema(
                            name=fname,
                            input_schema_json=in_schema,
                            output_schema_json=out_schema,
                            injection_json=inj_json,
                            incremental_output=incremental,
                        )
                    )
                except Exception:
                    continue

            resources = pb.WorkerResources(
                # Worker identity is carried in the worker-connect JWT claims. These fields are ignored.
                worker_id="",
                release_id="",
                # owner is provided per-request via TaskExecutionRequest/RealtimeOpenCommand.
                runpod_pod_id=self.runpod_pod_id,
                gpu_is_busy=self._get_gpu_busy_status(),
                cpu_cores=cpu_cores,
                memory_bytes=mem.total,
                gpu_count=gpu_count,
                gpu_memory_bytes=gpu_total_mem,
                gpu_memory_used_bytes=gpu_used_mem,
                gpu_memory_free_bytes=gpu_free_mem,
                gpu_name=gpu_name,
                gpu_driver=gpu_driver,
                max_concurrency=self.max_concurrency,
                cuda_version=cuda_version,
                torch_version=torch_version,
                gpu_sm=gpu_sm,
                installed_libs=installed_libs,
                available_functions=list(dict.fromkeys(list(self._task_specs.keys()) + list(self._ws_specs.keys()))),
                vram_models=vram_models,   # Models in VRAM (hot)
                disk_models=disk_models,   # Models on disk (warm)
                supports_model_loading=supports_model_loading_flag,
                function_schemas=function_schemas,
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

    def _runtime_batching_cfg_for_function(self, function_name: str) -> Dict[str, Any]:
        fn = str(function_name or "").strip()
        if not fn:
            return {}
        with self._runtime_batching_config_lock:
            cfg = self._runtime_batching_config_by_function.get(fn)
            if not cfg:
                return {}
            return dict(cfg)

    def _handle_runtime_batching_config_cmd(self, cmd: Any) -> None:
        cfg = getattr(cmd, "config", None)
        function_name = str(getattr(cfg, "function_name", "") or "").strip()
        version = int(getattr(cfg, "version", 0) or 0)
        success = False
        error_message = ""
        if not function_name:
            error_message = "missing_function_name"
        elif function_name not in self._task_specs:
            error_message = f"unknown_function:{function_name}"
        else:
            normalized: Dict[str, Any] = {
                "function_name": function_name,
                "batch_size_target": max(1, int(getattr(cfg, "batch_size_target", 1) or 1)),
                "batch_size_min": max(1, int(getattr(cfg, "batch_size_min", 1) or 1)),
                "batch_size_max": max(1, int(getattr(cfg, "batch_size_max", 1) or 1)),
                "prefetch_depth": max(1, int(getattr(cfg, "prefetch_depth", 1) or 1)),
                "max_wait_ms": max(1, int(getattr(cfg, "max_wait_ms", 1) or 1)),
                "version": max(1, version),
            }
            if normalized["batch_size_max"] < normalized["batch_size_min"]:
                normalized["batch_size_max"] = normalized["batch_size_min"]
            if normalized["batch_size_target"] < normalized["batch_size_min"]:
                normalized["batch_size_target"] = normalized["batch_size_min"]
            if normalized["batch_size_target"] > normalized["batch_size_max"]:
                normalized["batch_size_target"] = normalized["batch_size_max"]

            with self._runtime_batching_config_lock:
                prev = self._runtime_batching_config_by_function.get(function_name)
                prev_version = int((prev or {}).get("version", 0) or 0)
                if version > 0 and version < prev_version:
                    normalized = dict(prev or {})
                else:
                    self._runtime_batching_config_by_function[function_name] = normalized
            success = True

            try:
                self._send_message(
                    pb.WorkerSchedulerMessage(
                        worker_event=pb.WorkerEvent(
                            request_id="",
                            event_type="worker.runtime_batching.updated",
                            payload_json=json.dumps(
                                {
                                    "function_name": function_name,
                                    "version": int(normalized.get("version", 0) or 0),
                                    "batch_size_target": int(normalized.get("batch_size_target", 1) or 1),
                                    "batch_size_min": int(normalized.get("batch_size_min", 1) or 1),
                                    "batch_size_max": int(normalized.get("batch_size_max", 1) or 1),
                                    "prefetch_depth": int(normalized.get("prefetch_depth", 1) or 1),
                                    "max_wait_ms": int(normalized.get("max_wait_ms", 1) or 1),
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("utf-8"),
                        )
                    )
                )
            except Exception:
                pass

        ack_version = max(1, version)
        self._send_message(
            pb.WorkerSchedulerMessage(
                runtime_batching_config_result=pb.RuntimeBatchingConfigResult(
                    function_name=function_name,
                    version=ack_version,
                    success=success,
                    error_message=error_message,
                )
            )
        )

    def _emit_function_capabilities_event(self) -> None:
        functions: List[Dict[str, Any]] = []
        for fn_name, req in self._discovered_resources.items():
            caps = dict(req.to_dict() if req else {})
            if not caps:
                continue
            spec = self._task_specs.get(fn_name)
            if spec is not None:
                caps["output_mode"] = spec.output_mode
            caps["function_name"] = fn_name
            functions.append(caps)
        if not functions:
            return
        payload = {"functions": sorted(functions, key=lambda x: str(x.get("function_name") or ""))}
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sig = hashlib.sha256(raw).hexdigest()
        if sig == self._last_function_capabilities_hash:
            return
        self._last_function_capabilities_hash = sig
        self._send_message(
            pb.WorkerSchedulerMessage(
                worker_event=pb.WorkerEvent(
                    request_id="",
                    event_type="worker.function_capabilities",
                    payload_json=raw,
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
        self._draining = False
        self._start_registration_watchdog()

        try:
            while self._running and not self._stop_event.is_set():
                self._reconnect_count += 1
                logger.info(f"Connection attempt {self._reconnect_count}...")
                if self.connect():
                    # Successfully connected, wait for stop signal or disconnection
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

        # Cancel any active tasks
        active_task_ids = []
        if self._drain_timeout_seconds > 0:
            deadline = time.time() + self._drain_timeout_seconds
            while time.time() < deadline:
                with self._active_tasks_lock:
                    remaining = len(self._active_tasks)
                if remaining == 0:
                    break
                time.sleep(0.2)

        with self._active_tasks_lock:
            active_task_ids = list(self._active_tasks.keys())
            for request_id in active_task_ids:
                ctx = self._active_tasks.get(request_id)
                if ctx:
                    logger.debug(f"Cancelling active task {request_id} during stop.")
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
                    logger.warning(f"Scheduler redirect received; reconnecting to leader at {leader}")
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
         """Handles actions needed when a connection error occurs during run."""
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

        if msg_type == 'run_request':
            self._handle_run_request(message.run_request)
        elif msg_type == 'batch_run_request':
            self._handle_batch_run_request(message.batch_run_request)
        elif msg_type == 'load_model_cmd':
            # TODO: Implement model loading logic
            # model_id = message.load_model_cmd.model_id
            # logger.warning(f"Received load_model_cmd for {model_id}, but not yet implemented.")
            # # Send result back (failure for now)
            # result = pb.LoadModelResult(model_id=model_id, success=False, error_message="Model loading not implemented")
            # self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))
            self._handle_load_model_cmd(message.load_model_cmd)
        elif msg_type == 'unload_model_cmd':
            self._handle_unload_model_cmd(message.unload_model_cmd)
        elif msg_type == 'interrupt_run_cmd':
            cmd = message.interrupt_run_cmd
            request_id = cmd.request_id
            item_ids = [str(x).strip() for x in list(getattr(cmd, "item_ids", []) or []) if str(x).strip()]
            cancel_queued_only = bool(getattr(cmd, "cancel_queued_only", False))
            self._handle_interrupt_request(request_id, item_ids=item_ids, cancel_queued_only=cancel_queued_only)
        elif msg_type == "runtime_batching_config_cmd":
            self._handle_runtime_batching_config_cmd(message.runtime_batching_config_cmd)
        elif msg_type == "realtime_open_cmd":
            self._handle_realtime_open_cmd(message.realtime_open_cmd)
        elif msg_type == "realtime_frame":
            self._handle_realtime_frame(message.realtime_frame)
        elif msg_type == "realtime_close_cmd":
            self._handle_realtime_close_cmd(message.realtime_close_cmd)
        elif msg_type == "worker_event":
            self._handle_worker_event_from_scheduler(message.worker_event)
        # Add handling for other message types if needed (e.g., config updates)
        elif msg_type == 'endpoint_config':
            cfg = message.endpoint_config
            resolved_by_variant = dict(getattr(cfg, "resolved_cozy_models_by_variant_ref", {}) or {})
            logger.info(
                "Received EndpointConfig (supported=%d required=%d resolved=%d)",
                len(cfg.supported_repo_refs),
                len(cfg.required_variant_refs),
                len(resolved_by_variant),
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
                        new_fixed_spec_by_key[key] = {"ref": ref, "dtypes": []}

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
                    if new_fixed_spec_by_key:
                        self._fixed_model_spec_by_key = new_fixed_spec_by_key
                    if new_payload_by_fn:
                        self._payload_model_id_by_key_by_function = new_payload_by_fn
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
            self._resolved_cozy_models_by_id_baseline = self._canonicalize_resolved_models_map(
                resolved_by_variant
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
            self._required_variant_refs_from_scheduler = [
                _canonicalize_model_ref_string(str(v)) for v in list(cfg.required_variant_refs)
            ]

            # Start background prefetch regardless of model manager; disk readiness is useful even
            # for lightweight workers and enables cache-aware routing.
            self._start_startup_prefetch(self._required_variant_refs_from_scheduler or [])

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

    @staticmethod
    def _canonicalize_resolved_models_map(mp: Dict[str, Any]) -> Dict[str, Any]:
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
            # For digest-based refs (e.g. "cozy:owner/repo@blake3:<hex>"), also add
            # a tag-based alias (e.g. "cozy:owner/repo:latest") so that lookups by
            # tag in model_ref_downloader will find the resolved entry.
            try:
                parsed = parse_model_ref(canon)
                if parsed.scheme == "cozy" and parsed.cozy is not None and parsed.cozy.digest:
                    tag_canon = f"cozy:{parsed.cozy.owner}/{parsed.cozy.repo}:{parsed.cozy.tag}"
                    if tag_canon not in out:
                        out[tag_canon] = v
            except Exception:
                pass
        return out

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
        cache_dir = worker_model_cache_dir()
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
            from .models.ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id

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
                        # Best-effort telemetry + convergence signals.
                        try:
                            payload = json.dumps(
                                {"model_id": canon, "cached": True, "duration_ms": 0},
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("utf-8")
                            self._send_message(
                                pb.WorkerSchedulerMessage(
                                    worker_event=pb.WorkerEvent(request_id="", event_type="model.download.completed", payload_json=payload)
                                )
                            )
                            self._send_message(
                                pb.WorkerSchedulerMessage(
                                    worker_event=pb.WorkerEvent(request_id="", event_type="model.ready", payload_json=json.dumps({"model_id": canon}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
                                )
                            )
                        except Exception:
                            pass
                        # Volume inventory signal (gen-orchestrator issue #236).
                        try:
                            payload = json.dumps(
                                {"model_variant_id": canon, **self._shared_disk_volume_info(existing)},
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("utf-8")
                            self._send_message(
                                pb.WorkerSchedulerMessage(
                                    worker_event=pb.WorkerEvent(request_id="", event_type="model.cached", payload_json=payload)
                                )
                            )
                        except Exception:
                            pass
                        # Push a registration update promptly (do not wait for the 10s heartbeat tick).
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

                    tok = set_resolved_cozy_models_by_id(self._resolved_cozy_models_by_id_baseline or None)
                    try:
                        local_path = self._downloader.download(canon, str(cache_dir)) if self._downloader else ""
                    finally:
                        reset_resolved_cozy_models_by_id(tok)

                    lp = Path(local_path) if local_path else None
                    if lp is None or not lp.exists():
                        raise RuntimeError(f"model download returned missing path: {local_path!r}")

                    self._model_cache.mark_cached_to_disk(canon, lp)
                    # Optional fast convergence signal.
                    try:
                        payload = json.dumps({"model_id": canon}, separators=(",", ":"), sort_keys=True).encode("utf-8")
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id="", event_type="model.ready", payload_json=payload)
                            )
                        )
                    except Exception:
                        pass
                    # Volume inventory signal (gen-orchestrator issue #236).
                    # model_variant_id should match what the scheduler uses in required_variant_refs.
                    try:
                        payload = json.dumps(
                            {"model_variant_id": canon, **self._shared_disk_volume_info(lp)},
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id="", event_type="model.cached", payload_json=payload)
                            )
                        )
                    except Exception:
                        pass
                    try:
                        dur_ms = int((time.monotonic() - started_at) * 1000)
                        payload = json.dumps(
                            {"model_id": canon, "cached": False, "duration_ms": dur_ms},
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id="", event_type="model.download.completed", payload_json=payload)
                            )
                        )
                    except Exception:
                        pass
                    # Push a registration update promptly (do not wait for the 10s heartbeat tick).
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

    def _try_find_existing_cozy_snapshot_dir(self, canonical_model_id: str, cache_dir: Path) -> Optional[Path]:
        # Only applies to cozy:@sha256 refs where the snapshot digest is known.
        try:
            parsed = parse_model_ref(canonical_model_id)
        except Exception:
            return None
        if parsed.scheme != "cozy" or parsed.cozy is None:
            return None
        digest = (parsed.cozy.digest or "").strip()
        if not digest:
            # Tag refs are mutable; rely on downloads.
            return None
        snap_dir = cache_dir / "snapshots" / digest
        if snap_dir.exists():
            return snap_dir
        return None

    def _handle_worker_event_from_scheduler(self, ev: WorkerEvent) -> None:
        # Direction is not enforced by the proto; we reserve WorkerEvent for low-frequency
        # scheduler->worker control signals too (e.g. WORKER_JWT rotation).
        try:
            event_type = str(getattr(ev, "event_type", "") or "").strip()
            if not event_type:
                return
            if event_type not in self._JWT_ROTATE_EVENT_TYPES:
                logger.info("Ignoring scheduler worker_event type=%r", event_type)
                return

            payload_raw = getattr(ev, "payload_json", b"") or b""
            payload: Dict[str, Any] = {}
            try:
                payload = json.loads(payload_raw.decode("utf-8")) if payload_raw else {}
            except Exception:
                payload = {}

            new_token = str(payload.get("worker_jwt") or payload.get("token") or "").strip()
            if not new_token:
                logger.warning("Received worker JWT rotation event without a token; ignoring")
                return

            # Only affects future reconnects. The current stream uses the interceptor created at connect time.
            self.worker_jwt = new_token
            logger.info("Stored rotated WORKER_JWT for next reconnect (len=%d).", len(new_token))
        except Exception as e:
            logger.warning("Failed to handle scheduler worker_event: %s", e)

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

            from .models.ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id
            tok = set_resolved_cozy_models_by_id(self._resolved_cozy_models_by_id_baseline or None)
            try:
                loop.run_until_complete(
                    self._model_manager.process_supported_models_config(
                        self._supported_model_ids_from_scheduler,
                        self._downloader
                    )
                )
            finally:
                reset_resolved_cozy_models_by_id(tok)
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
                from .models.ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id
                per_cmd = dict(getattr(cmd, "resolved_cozy_models_by_id", {}) or {})
                baseline = getattr(self, "_resolved_cozy_models_by_id_baseline", None) or {}
                merged = {**baseline, **per_cmd} if per_cmd else dict(baseline)
                tok = set_resolved_cozy_models_by_id(merged or None)
                try:
                    # load_model_into_vram is async
                    success = asyncio.run(self._model_manager.load_model_into_vram(model_id))
                finally:
                    reset_resolved_cozy_models_by_id(tok)
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

    def _handle_run_request(self, request: TaskExecutionRequest) -> None:
        """Handle a task execution request from the scheduler."""
        request_id = request.request_id
        run_id = str(getattr(request, "run_id", "") or "").strip() or None
        function_name = request.function_name
        input_payload = request.input_payload
        required_model_id_for_exec = ""
        timeout_ms = int(getattr(request, "timeout_ms", 0) or 0)
        owner = str(getattr(request, "owner", "") or "") or (self.owner or "")
        invoker_id = str(getattr(request, "invoker_id", "") or "")
        file_base_url = str(getattr(request, "file_base_url", "") or "")
        worker_capability_token = _extract_worker_capability_token(request)
        resolved_cozy_models_by_id = dict(getattr(request, "resolved_cozy_models_by_id", {}) or {})
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
        materialized_input_urls: Dict[str, str] = {}
        raw_urls_map = getattr(request, "input_ref_urls", None)
        if isinstance(raw_urls_map, cabc.Mapping):
            for k, v in raw_urls_map.items():
                ks = str(k or "").strip().lstrip("/")
                vs = str(v or "").strip()
                if ks and vs:
                    materialized_input_urls[ks] = vs
        raw_urls = getattr(request, "input_ref_urls_json", None)
        if isinstance(raw_urls, str) and raw_urls.strip():
            try:
                parsed = json.loads(raw_urls)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        ks = str(k or "").strip().lstrip("/")
                        vs = str(v or "").strip()
                        if ks and vs:
                            materialized_input_urls[ks] = vs
            except Exception:
                pass

        required_models_raw = list(getattr(request, "required_variant_refs", []) or [])
        if required_models_raw:
            required_model_id_for_exec = str(required_models_raw[0] or "").strip()

        logger.info(f"Received Task request: request_id={request_id}, function={function_name}, model='{required_model_id_for_exec or 'None'}'")
        self._emit_task_event(
            request_id,
            "task.received",
            {
                "function_name": function_name,
                "run_id": run_id or "",
                "required_variant_refs_count": len(required_models_raw),
                "input_bytes": len(input_payload or b""),
            },
        )

        spec = self._task_specs.get(function_name)
        if not spec:
            error_msg = f"Unknown function requested: {function_name}"
            logger.error(error_msg)
            self._emit_task_event(
                request_id,
                "task.rejected",
                {"reason": "unknown_function", "function_name": function_name},
            )
            self._send_task_result(request_id, False, None, "internal", False, "internal error", error_msg)
            return
        if self.max_input_bytes > 0 and len(input_payload) > self.max_input_bytes:
            error_msg = f"Input payload too large: {len(input_payload)} bytes (max {self.max_input_bytes})"
            logger.error(error_msg)
            self._emit_task_event(
                request_id,
                "task.rejected",
                {"reason": "input_too_large", "input_bytes": len(input_payload)},
            )
            self._send_task_result(request_id, False, None, "validation", False, "invalid input", error_msg)
            return
        if self._draining:
            error_msg = "Worker is draining; refusing new tasks"
            logger.warning(error_msg)
            self._emit_task_event(request_id, "task.rejected", {"reason": "worker_draining"})
            self._send_task_result(request_id, False, None, "retryable", True, "worker busy", error_msg)
            return

        # required_variant_refs are pinned variant refs chosen by the scheduler; the worker must not guess.
        required_models: List[str] = []
        for raw in required_models_raw:
            s = str(raw or "").strip()
            if not s:
                continue
            required_models.append(_canonicalize_model_ref_string(s))

        runtime_cfg = self._runtime_batching_cfg_for_function(function_name)
        resource_req = self._discovered_resources.get(function_name)
        req_cfg = dict(resource_req.to_dict() if resource_req else {})
        execution_hints: Dict[str, Any] = {}
        kind = str(req_cfg.get("kind", "") or "").strip()
        if kind:
            execution_hints["kind"] = kind
        if "memory_hint_mb" in req_cfg:
            try:
                execution_hints["memory_hint_mb"] = int(req_cfg.get("memory_hint_mb") or 0)
            except Exception:
                pass

        ctx = RequestContext(
            request_id,
            run_id=run_id,
            emitter=self._emit_progress_event,
            owner=owner or None,
            invoker_id=invoker_id or None,
            timeout_ms=timeout_ms if timeout_ms > 0 else None,
            file_api_base_url=file_base_url or None,
            worker_capability_token=worker_capability_token or None,
            materialized_input_urls=materialized_input_urls or None,
            local_output_dir=None,
            resolved_cozy_models_by_id=resolved_cozy_models_by_id or None,
            required_models=required_models or None,
            runtime_batching_config=runtime_cfg or None,
            execution_hints=execution_hints or None,
            parent_request_id=parent_request_id,
            child_request_id=child_request_id,
            item_id=item_id,
            item_index=item_index,
        )
        # Add to active tasks *before* starting thread
        with self._active_tasks_lock:
             # Double-check if task is already active (race condition mitigation)
             if request_id in self._active_tasks:
                  error_msg = f"Task with request_id {request_id} is already active (race condition?)."
                  logger.error(error_msg)
                  self._emit_task_event(request_id, "task.rejected", {"reason": "duplicate_request_id"})
                  return # Avoid starting duplicate thread
             if self.max_concurrency > 0 and len(self._active_tasks) >= self.max_concurrency:
                  error_msg = f"Worker concurrency limit reached ({self.max_concurrency})."
                  logger.error(error_msg)
                  self._emit_task_event(request_id, "task.rejected", {"reason": "max_concurrency_reached"})
                  self._send_task_result(request_id, False, None, "retryable", True, "worker busy", error_msg)
                  return
             self._active_tasks[request_id] = ctx

        # Execute function in a separate thread to avoid blocking the receive loop
        thread = threading.Thread(
            target=self._execute_task,
            args=(ctx, spec, input_payload),
            daemon=True,
        )
        thread.start()

    def _handle_batch_run_request(self, request: Any) -> None:
        batch_id = str(getattr(request, "batch_id", "") or "")
        batch_function_name = str(getattr(request, "function_name", "") or "")
        items = list(getattr(request, "items", []) or [])
        logger.info(
            "Received batch task request: batch_id=%s function=%s items=%d",
            batch_id or "(none)",
            batch_function_name or "(per-item)",
            len(items),
        )
        for item in items:
            if item is None:
                continue
            function_name = str(getattr(item, "function_name", "") or "") or batch_function_name
            request_id = str(getattr(item, "request_id", "") or "")
            item_id = str(getattr(item, "item_id", "") or "") or "item-000001"
            if not request_id or not function_name:
                continue
            with self._request_batch_context_lock:
                self._request_batch_context[request_id] = (batch_id, item_id)
            req = pb.TaskExecutionRequest(
                request_id=request_id,
                function_name=function_name,
                input_payload=bytes(getattr(item, "input_payload", b"") or b""),
                required_variant_refs=list(getattr(item, "required_variant_refs", []) or []),
                timeout_ms=int(getattr(item, "timeout_ms", 0) or 0),
                owner=str(getattr(item, "owner", "") or ""),
                invoker_id=str(getattr(item, "invoker_id", "") or ""),
                resolved_cozy_models_by_id=dict(getattr(item, "resolved_cozy_models_by_id", {}) or {}),
                parent_request_id=str(getattr(item, "parent_request_id", "") or ""),
                child_request_id=str(getattr(item, "child_request_id", "") or ""),
                item_id=item_id,
                item_index=int(getattr(item, "item_index", 0) or 0),
            )
            self._handle_run_request(req)

    def _handle_interrupt_request(self, request_id: str, *, item_ids: Optional[List[str]] = None, cancel_queued_only: bool = False) -> None:
        """Handle a request to interrupt/cancel a running task."""
        logger.info(
            "Received interrupt request for request_id=%s item_ids=%s cancel_queued_only=%s",
            request_id,
            item_ids or [],
            cancel_queued_only,
        )
        with self._active_tasks_lock:
            ctx = self._active_tasks.get(request_id)
            if ctx:
                ctx.cancel() # Set internal flag and event
            else:
                logger.warning(f"Could not interrupt task request_id={request_id}: Not found in active tasks.")

    def _handle_realtime_open_cmd(self, cmd: Any) -> None:
        session_id = str(getattr(cmd, "session_id", "") or "")
        function_name = str(getattr(cmd, "function_name", "") or "")
        if not session_id or not function_name:
            return
        spec = self._ws_specs.get(function_name)
        if spec is None:
            self._send_message(
                pb.WorkerSchedulerMessage(
                    realtime_close_cmd=pb.RealtimeCloseCommand(session_id=session_id, reason="unknown_function")
                )
            )
            return

        owner = str(getattr(cmd, "owner", "") or "") or (self.owner or "")
        invoker_id = str(getattr(cmd, "invoker_id", "") or "")
        timeout_ms = int(getattr(cmd, "timeout_ms", 0) or 0) or None
        file_base_url = str(getattr(cmd, "file_base_url", "") or "")
        worker_capability_token = _extract_worker_capability_token(cmd)
        materialized_input_urls: Dict[str, str] = {}
        raw_urls_map = getattr(cmd, "input_ref_urls", None)
        if isinstance(raw_urls_map, cabc.Mapping):
            for k, v in raw_urls_map.items():
                ks = str(k or "").strip().lstrip("/")
                vs = str(v or "").strip()
                if ks and vs:
                    materialized_input_urls[ks] = vs
        raw_urls = getattr(cmd, "input_ref_urls_json", None)
        if isinstance(raw_urls, str) and raw_urls.strip():
            try:
                parsed = json.loads(raw_urls)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        ks = str(k or "").strip().lstrip("/")
                        vs = str(v or "").strip()
                        if ks and vs:
                            materialized_input_urls[ks] = vs
            except Exception:
                pass
        ctx = RequestContext(
            session_id,
            emitter=self._emit_progress_event,
            owner=owner or None,
            invoker_id=invoker_id or None,
            timeout_ms=timeout_ms,
            file_api_base_url=file_base_url or None,
            worker_capability_token=worker_capability_token or None,
            materialized_input_urls=materialized_input_urls or None,
            resolved_cozy_models_by_id=getattr(self, "_resolved_cozy_models_by_id_baseline", None) or None,
        )

        max_frame = int(os.getenv("WORKER_MAX_WS_FRAME_BYTES", "0") or 0)

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            in_q: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=16)
            sock = _RealtimeSocketAdapter(self, session_id, loop, in_q)
            closed = threading.Event()
            st = _RealtimeSessionState(session_id=session_id, spec=spec, ctx=ctx, loop=loop, in_q=in_q, closed=closed)
            with self._realtime_lock:
                self._realtime_sessions[session_id] = st

            async def run_handler() -> None:
                # Build kwargs for handler.
                kwargs: Dict[str, Any] = {spec.ctx_param: ctx, spec.socket_param: sock}
                from .models.ref_downloader import (
                    reset_cozy_model_download_prefs_by_ref,
                    reset_resolved_cozy_models_by_id,
                    set_cozy_model_download_prefs_by_ref,
                    set_resolved_cozy_models_by_id,
                )
                baseline = getattr(self, "_resolved_cozy_models_by_id_baseline", None) or None
                resolved_tok = set_resolved_cozy_models_by_id(getattr(ctx, "resolved_cozy_models_by_id", None) or baseline)
                prefs_tok = set_cozy_model_download_prefs_by_ref({})
                try:
                    required_variant_refs = list(getattr(cmd, "required_variant_refs", []) or [])
                    for idx, inj in enumerate(spec.injections):
                        if idx >= len(required_variant_refs) or not str(required_variant_refs[idx]).strip():
                            raise ValueError(f"missing required_variant_refs for injection param: {inj.param_name}")
                        model_id = _canonicalize_model_ref_string(str(required_variant_refs[idx]).strip())
                        self._enforce_model_allowlist(model_id, inj)
                        kwargs[inj.param_name] = self._resolve_injected_value(ctx, inj.param_type, model_id, inj)
                    await spec.func(**kwargs)
                finally:
                    reset_resolved_cozy_models_by_id(resolved_tok)
                    reset_cozy_model_download_prefs_by_ref(prefs_tok)

            try:
                loop.run_until_complete(run_handler())
                self._send_message(
                    pb.WorkerSchedulerMessage(
                        realtime_close_cmd=pb.RealtimeCloseCommand(session_id=session_id, reason="completed")
                    )
                )
            except Exception as exc:
                self._send_message(
                    pb.WorkerSchedulerMessage(
                        realtime_close_cmd=pb.RealtimeCloseCommand(session_id=session_id, reason=f"error:{type(exc).__name__}")
                    )
                )
            finally:
                closed.set()
                try:
                    loop.call_soon_threadsafe(in_q.put_nowait, None)
                except Exception:
                    pass
                with self._realtime_lock:
                    self._realtime_sessions.pop(session_id, None)
                try:
                    loop.stop()
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        t = threading.Thread(target=runner, daemon=True)
        t.start()

    def _handle_realtime_frame(self, frame: Any) -> None:
        session_id = str(getattr(frame, "session_id", "") or "")
        data = bytes(getattr(frame, "data", b"") or b"")
        if not session_id:
            return

        max_frame = int(os.getenv("WORKER_MAX_WS_FRAME_BYTES", "0") or 0)
        if max_frame > 0 and len(data) > max_frame:
            self._send_message(
                pb.WorkerSchedulerMessage(
                    realtime_close_cmd=pb.RealtimeCloseCommand(session_id=session_id, reason="frame_too_large")
                )
            )
            return

        with self._realtime_lock:
            st = self._realtime_sessions.get(session_id)
        if st is None:
            return
        try:
            st.loop.call_soon_threadsafe(st.in_q.put_nowait, data)
        except Exception:
            pass

    def _handle_realtime_close_cmd(self, cmd: Any) -> None:
        session_id = str(getattr(cmd, "session_id", "") or "")
        if not session_id:
            return
        with self._realtime_lock:
            st = self._realtime_sessions.get(session_id)
        if st is None:
            return
        st.ctx.cancel()
        try:
            st.loop.call_soon_threadsafe(st.in_q.put_nowait, None)
        except Exception:
            pass

    def _execute_task(
        self,
        ctx: RequestContext,
        spec: _TaskSpec,
        input_payload: bytes,
    ) -> None:
        """Execute a discovered task handler and send result/events back."""
        request_id = ctx.request_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""  # internal/legacy
        retryable = False
        success = False

        # Metrics (best-effort): never fail a run.
        resolved_map = getattr(ctx, "resolved_cozy_models_by_id", None) or None
        rm = RunMetricsV1(
            request_id=str(request_id or ""),
            function_name=str(spec.name or ""),
            required_models=list(getattr(ctx, "required_models", []) or []),
            resolved_cozy_models_by_id=resolved_map,
        )
        # Attach to ctx so RequestContext.save_* and injection paths can accumulate.
        try:
            setattr(ctx, "_run_metrics", rm)
        except Exception:
            pass
        rm.mark_compute_started()
        if rm.compute_started_at:
            self._emit_worker_event_bytes(request_id, "metrics.compute.started", safe_json_bytes({"at": rm.compute_started_at}))
        self._emit_task_event(
            request_id,
            "task.started",
            {
                "function_name": str(spec.name or ""),
                "required_variant_refs": list(getattr(ctx, "required_models", []) or []),
                "runtime_batching_config": getattr(ctx, "runtime_batching_config", {}),
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
                cache_dir=worker_model_cache_dir(),
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

        # Refcounted BUSY so overlapping runs/model ops can't flip BUSY -> NOT BUSY early.
        self._gpu_busy_enter()

        from .models.ref_downloader import (
            reset_cozy_model_download_prefs_by_ref,
            reset_resolved_cozy_models_by_id,
            set_cozy_model_download_prefs_by_ref,
            set_resolved_cozy_models_by_id,
        )

        baseline = getattr(self, "_resolved_cozy_models_by_id_baseline", None) or None
        resolved_map = getattr(ctx, "resolved_cozy_models_by_id", None) or baseline
        resolved_tok = set_resolved_cozy_models_by_id(resolved_map)
        prefs_map: Dict[str, Any] = {}
        prefs_tok = set_cozy_model_download_prefs_by_ref(prefs_map)

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
            # Best-effort extract diffusion-ish numeric fields for metrics.run.
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

            for inj in spec.injections:
                resolve_t0 = time.monotonic()
                resolve_watchdog = self._start_task_phase_watchdog(
                    request_id=request_id,
                    phase="model_resolve",
                    warn_after_s=float(getattr(self, "_warn_model_resolve_s", 30.0)),
                    payload={"function_name": spec.name, "param_name": inj.param_name},
                )
                self._emit_task_event(
                    request_id,
                    "task.model_resolve.started",
                    {"function_name": spec.name, "param_name": inj.param_name},
                )
                model_id = ""
                model_key: Optional[str] = None
                canon_model_id = ""
                try:
                    model_id, model_key = self._resolve_model_id_for_injection(spec.name, inj, payload=input_obj)
                    canon_model_id = _canonicalize_model_ref_string(str(model_id or "").strip()) if model_id else ""
                    self._emit_task_event(
                        request_id,
                        "task.model_resolve.completed",
                        {
                            "function_name": spec.name,
                            "param_name": inj.param_name,
                            "model_id": canon_model_id,
                            "model_key": model_key or "",
                            "duration_ms": int((time.monotonic() - resolve_t0) * 1000),
                        },
                    )
                except Exception as resolve_exc:
                    self._emit_task_event(
                        request_id,
                        "task.model_resolve.failed",
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
                # Best-effort: attach dtype preferences from endpoint.toml-derived manifest mapping.
                if model_key:
                    try:
                        s = None
                        if inj.model_ref.source == ModelRefSource.FIXED:
                            s = self._fixed_model_spec_by_key.get(model_key)
                        elif inj.model_ref.source == ModelRefSource.PAYLOAD:
                            by_fn = self._payload_model_spec_by_key_by_function.get(spec.name) or {}
                            s = by_fn.get(model_key) if isinstance(by_fn, dict) else None
                        if isinstance(s, dict):
                            dts = s.get("dtypes")
                            if isinstance(dts, list) and canon_model_id:
                                prefs_map[canon_model_id] = {"dtypes": [str(x) for x in dts if str(x).strip()]}
                    except Exception:
                        pass
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
                self._emit_task_event(
                    request_id,
                    "task.model_load.started",
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
                    self._emit_task_event(
                        request_id,
                        "task.model_load.completed",
                        {
                            "function_name": spec.name,
                            "param_name": inj.param_name,
                            "model_id": canon_model_id,
                            "duration_ms": int((time.monotonic() - load_t0) * 1000),
                        },
                    )
                except Exception as load_exc:
                    self._emit_task_event(
                        request_id,
                        "task.model_load.failed",
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
            if execution_kind not in {"inference", "conversion", "training"}:
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
            self._emit_task_event(
                request_id,
                f"task.{execution_kind}.started",
                {"function_name": spec.name, "output_mode": spec.output_mode, "phase": execution_kind},
            )
            logger.info("[request_id=%s] calling %s", request_id, spec.name)
            if inspect.iscoroutinefunction(spec.func):
                result = asyncio.run(spec.func(**call_kwargs))
            elif inspect.isasyncgenfunction(spec.func):
                result = spec.func(**call_kwargs)
            else:
                result = spec.func(**call_kwargs)
            logger.info("[request_id=%s] %s returned, output_mode=%s", request_id, spec.name, spec.output_mode)

            if ctx.is_canceled():
                raise CanceledError("canceled")

            if spec.output_mode == "single":
                if spec.output_type is not None and not isinstance(result, spec.output_type):
                    raise TypeError(f"Function {spec.name} returned {type(result)!r}, expected {spec.output_type!r}")
                result = self._auto_upload_output_assets(ctx, result)
                output_payload = msgspec.msgpack.encode(msgspec.to_builtins(result))
                if self.max_output_bytes > 0 and len(output_payload) > self.max_output_bytes:
                    raise ValueError(f"Output payload too large: {len(output_payload)} bytes (max {self.max_output_bytes})")
                success = True
            else:
                # Incremental output: the function returns an iterator of delta structs.
                max_delta_bytes = int(os.getenv("WORKER_MAX_OUTPUT_DELTA_BYTES", "65536"))
                max_events = int(os.getenv("WORKER_MAX_OUTPUT_DELTA_EVENTS", "0"))
                count = 0
                last_item_id = "item-0"

                def emit_delta(delta_obj: msgspec.Struct) -> None:
                    nonlocal count, last_item_id
                    payload = msgspec.to_builtins(delta_obj)
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
                    emitted = self._emit_incremental_delta_typed(
                        request_id=request_id,
                        function_name=spec.name,
                        item_id=item_id,
                        sequence=count + 1,
                        timestamp_unix_ms=ts_ms,
                        delta_text=delta_text,
                        payload_json=raw,
                    )
                    if not emitted:
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id=request_id, event_type="output.delta", payload_json=raw)
                            )
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

                # Optionally emit completion marker.
                done_ts_ms = int(time.time() * 1000)
                emitted_done = self._emit_incremental_done_typed(
                    request_id=request_id,
                    function_name=spec.name,
                    item_id=last_item_id,
                    sequence=count + 1,
                    timestamp_unix_ms=done_ts_ms,
                )
                if not emitted_done:
                    self._send_message(
                        pb.WorkerSchedulerMessage(
                            worker_event=pb.WorkerEvent(request_id=request_id, event_type="output.completed", payload_json=b"{}")
                        )
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
            self._emit_task_event(
                request_id,
                f"task.{execution_kind}.completed",
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
            if inference_watchdog is not None:
                inference_watchdog.cancel()
            self._emit_task_event(
                request_id,
                "task.failed",
                {
                    "function_name": spec.name,
                    "error_type": error_type,
                    "retryable": bool(retryable),
                    "safe_message": safe_message,
                },
            )
            if "t_infer0" in locals():
                self._emit_task_event(
                    request_id,
                    f"task.{execution_kind}.failed",
                    {
                        "function_name": spec.name,
                        "phase": execution_kind,
                        "error_type": error_type,
                        "duration_ms": int((time.monotonic() - float(locals().get("t_infer0", time.monotonic()))) * 1000),
                    },
                )
            if spec.output_mode == "incremental":
                try:
                    payload = json.dumps({"error_type": error_type, "message": safe_message}, separators=(",", ":")).encode("utf-8")
                    emitted_err = self._emit_incremental_error_typed(
                        request_id=request_id,
                        function_name=spec.name,
                        item_id="item-0",
                        sequence=0,
                        timestamp_unix_ms=int(time.time() * 1000),
                        error_message=safe_message,
                    )
                    if not emitted_err:
                        self._send_message(
                            pb.WorkerSchedulerMessage(
                                worker_event=pb.WorkerEvent(request_id=request_id, event_type="output.error", payload_json=payload)
                            )
                        )
                except Exception:
                    pass
            success = False
        finally:
            if inference_watchdog is not None:
                inference_watchdog.cancel()
            reset_resolved_cozy_models_by_id(resolved_tok)
            reset_cozy_model_download_prefs_by_ref(prefs_tok)

            for mid in models_in_use:
                try:
                    self._model_use_exit(mid)
                except Exception:
                    pass
            self._gpu_busy_exit()

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
                self._emit_worker_event_bytes(request_id, "metrics.run", safe_json_bytes(rm.to_metrics_run_payload()))
            except Exception:
                pass
            if success:
                self._emit_task_event(
                    request_id,
                    "task.completed",
                    {
                        "function_name": spec.name,
                        "duration_ms": int((time.monotonic() - float(getattr(rm, "_t0_monotonic", time.monotonic()))) * 1000),
                    },
                )

            self._send_task_result(request_id, success, output_payload, error_type, bool(retryable), safe_message, error_message)

            with self._active_tasks_lock:
                self._active_tasks.pop(request_id, None)

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
                from .pipeline.loader import LocalModelCache  # local import to avoid heavy import at worker init

                max_cache_gb = float(os.environ.get("WORKER_LOCAL_CACHE_GB", "100"))
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

            p = path or worker_model_cache_dir()
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

    def _resolve_injected_value(self, ctx: RequestContext, requested_type: Any, model_id: str, inj: InjectionSpec) -> Any:
        qn = type_qualname(requested_type)
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
                        parsed = parse_model_ref(str(model_id))
                        canon = parsed.cozy.canonical() if parsed.scheme == "cozy" and parsed.cozy is not None else str(model_id)
                        model_metrics = rm.models.get(canon)
                        rm.set_initial_model_state(
                            canon,
                            "hot_vram",
                            model_metrics.snapshot_digest if model_metrics is not None else None,
                        )
                    except Exception:
                        pass
                if isinstance(requested_type, type) and not isinstance(pipe, requested_type):
                    expected_qn = type_qualname(requested_type)
                    got_qn = type_qualname(type(pipe))
                    raise ValueError(
                        f"model injection type mismatch for {inj.param_name}: expected {expected_qn}, got {got_qn} (model_id={model_id})"
                    )
                return pipe

        # Transformers-style injection (AutoModel/AutoProcessor/etc) and any other
        # libraries with a `from_pretrained` factory.
        # We treat these as worker-owned cached handles: load once, reuse across invocations.
        if hasattr(requested_type, "from_pretrained") and callable(getattr(requested_type, "from_pretrained", None)):
            qn = type_qualname(requested_type)
            key = (model_id, qn)
            lock = self._custom_runtime_locks.setdefault(key, threading.Lock())
            with lock:
                cached = self._custom_runtime_cache.get(key)
                if cached is not None:
                    return cached

                # Diffusers pipelines: prefer downloading via the worker's model downloader and
                # loading from a local directory, instead of letting diffusers/huggingface_hub
                # interpret model refs like "hf:owner/repo".
                #
                # This keeps tenant code inference-only while still supporting the platform's
                # model-ref schemes and caching behavior.
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
                        parsed = parse_model_ref(str(model_id))
                        canon = parsed.cozy.canonical() if parsed.scheme == "cozy" and parsed.cozy is not None else str(model_id)
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
                                    expected_qn = type_qualname(requested_type)
                                    got_qn = type_qualname(type(cached))
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
                        cache_dir = str(worker_model_cache_dir())
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
                            resolved_entry = (getattr(ctx, "resolved_cozy_models_by_id", None) or {}).get(canon)
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
                                    # Volume inventory signal (gen-orchestrator issue #236).
                                    try:
                                        payload = json.dumps(
                                            {"model_variant_id": canon, **self._shared_disk_volume_info(lp)},
                                            separators=(",", ":"),
                                            sort_keys=True,
                                        ).encode("utf-8")
                                        self._send_message(
                                            pb.WorkerSchedulerMessage(
                                                worker_event=pb.WorkerEvent(request_id="", event_type="model.cached", payload_json=payload)
                                            )
                                        )
                                    except Exception:
                                        pass
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
                            parsed = parse_model_ref(model_source)
                            if self._downloader is not None and parsed.scheme in ("cozy", "hf"):
                                model_source = self._downloader.download(model_source, str(worker_model_cache_dir()))
                            elif parsed.scheme == "hf" and parsed.hf is not None:
                                # Fallback path when downloader is unavailable.
                                model_source = parsed.hf.repo_id
                                if parsed.hf.revision:
                                    preload_kwargs["revision"] = parsed.hf.revision
                    except Exception:
                        model_source = str(model_id)
                        preload_kwargs = {}

                    logger.info(
                        "Loading from_pretrained: source=%s type=%s kwargs=%s",
                        model_source, type_qualname(requested_type), list(preload_kwargs.keys()),
                    )
                    obj = from_pretrained(model_source, **preload_kwargs)
                    logger.info(
                        "from_pretrained complete: source=%s (%.1fs)",
                        model_source, time.monotonic() - t_pi0,
                    )
                    if rm is not None:
                        rm.add_pipeline_init_time(int((time.monotonic() - t_pi0) * 1000))
                if isinstance(requested_type, type) and not isinstance(obj, requested_type):
                    expected_qn = type_qualname(requested_type)
                    got_qn = type_qualname(type(obj))
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
                                max_keep = int(os.getenv("WORKER_MAX_DIFFUSERS_VRAM_MODELS", "2") or "2")
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
                        logger.info(
                            "Moving model to device=%s dtype=%s model=%s ...",
                            str(ctx.device), torch_dtype, model_id,
                        )
                        try:
                            if torch_dtype is not None:
                                obj = obj.to(str(ctx.device), dtype=torch_dtype)
                            else:
                                obj = obj.to(str(ctx.device))
                        except TypeError:
                            # Some objects implement .to(device) but not dtype kwarg.
                            obj = obj.to(str(ctx.device))
                        logger.info(
                            "Model moved to device=%s successfully model=%s",
                            str(ctx.device), model_id,
                        )
                        if rm is not None:
                            rm.add_gpu_load_time(int((time.monotonic() - t_to0) * 1000))

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
                                        size_gb = float(os.getenv("WORKER_DIFFUSERS_VRAM_GB_FALLBACK", "10") or "10")
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

    def _resolve_model_id_for_injection(self, fn_name: str, inj: InjectionSpec, payload: msgspec.Struct) -> tuple[str, Optional[str]]:
        fixed_map = dict(getattr(self, "_fixed_model_id_by_key", {}) or {})
        payload_map = dict((getattr(self, "_payload_model_id_by_key_by_function", {}) or {}).get(fn_name) or {})
        allowed_ids: Optional[set[str]] = None
        local_allowed: set[str] = set()
        local_allowed.update(fixed_map.values())
        local_allowed.update(payload_map.values())
        if local_allowed:
            allowed_ids = local_allowed

        if inj.model_ref.source == ModelRefSource.FIXED:
            raw = inj.model_ref.key.strip()
            if not raw:
                raise ValueError(f"empty fixed ModelRef for injection param: {inj.param_name}")
            explicit_ref = _canonicalize_model_ref_string(str(inj.model_ref.ref or "").strip())
            if explicit_ref:
                raise ValueError(
                    f"function {fn_name!r} uses ModelRef(FIXED, {raw!r}) with inline ref; "
                    "declare fixed key mappings in endpoint.toml [models]"
                )
            if not fixed_map:
                raise ValueError(
                    "fixed model selection is not configured; expected top-level models in /app/.tensorhub/endpoint.lock"
                )
            if raw not in fixed_map:
                allowed = sorted(fixed_map.keys())
                head = allowed[:20]
                suffix = ""
                if len(allowed) > len(head):
                    suffix = f" (+{len(allowed) - len(head)} more)"
                raise ValueError(
                    f"unknown fixed model key {raw!r}; allowed keys: {head}{suffix}"
                )
            model_id = fixed_map[raw]
            self._enforce_model_allowlist(model_id, inj, allowed_ids=allowed_ids)
            return model_id, raw

        if inj.model_ref.source == ModelRefSource.PAYLOAD:
            field = inj.model_ref.key.strip()
            if not field:
                raise ValueError(f"empty payload ModelRef for injection param: {inj.param_name}")
            try:
                chosen = getattr(payload, field)
            except Exception:
                raise ValueError(f"missing payload field for model selection: {field!r}") from None
            if chosen is None:
                raise ValueError(f"payload field {field!r} is null; expected a model key")
            if not isinstance(chosen, str):
                raise ValueError(f"payload field {field!r} must be a string (model key), got {type(chosen)!r}")
            key = chosen.strip()
            if not key:
                raise ValueError(f"payload field {field!r} is empty; expected a model key")
            if not payload_map:
                raise ValueError(
                    f"payload model selection is not configured for function {fn_name!r}; "
                    f"expected models_by_function.{fn_name} in /app/.tensorhub/endpoint.lock"
                )
            if key not in payload_map:
                allowed = sorted(payload_map.keys())
                head = allowed[:20]
                suffix = ""
                if len(allowed) > len(head):
                    suffix = f" (+{len(allowed) - len(head)} more)"
                raise ValueError(
                    f"unknown model key {key!r} for function {fn_name!r} payload field {field!r}; "
                    f"allowed keys: {head}{suffix}"
                )
            model_id = payload_map[key]
            self._enforce_model_allowlist(model_id, inj, allowed_ids=allowed_ids)
            return model_id, key

        raise ValueError(f"unknown ModelRef source: {inj.model_ref.source!r}")

    def _enforce_model_allowlist(self, model_id: str, inj: InjectionSpec, *, allowed_ids: Optional[set[str]] = None) -> None:
        # Enforce BOTH:
        # - any per-function mapping restriction (allowed_ids), AND
        # - the release-level allowlist from the scheduler (defense-in-depth).
        if allowed_ids is not None and model_id not in allowed_ids:
            raise ValueError(f"model_id not allowed for function: {model_id!r} (injection param {inj.param_name})")
        if self._release_allowed_model_ids is not None and model_id not in self._release_allowed_model_ids:
            raise ValueError(f"model_id not allowed for release: {model_id!r} (injection param {inj.param_name})")

    def _send_task_result(
        self,
        request_id: str,
        success: bool,
        output_payload: Optional[bytes],
        error_type: str,
        retryable: bool,
        safe_message: str,
        error_message: str,
    ) -> None:
        """Send a task execution result back to the scheduler via the queue."""
        try:
            batch_ctx: Optional[Tuple[str, str]] = None
            with self._request_batch_context_lock:
                batch_ctx = self._request_batch_context.pop(request_id, None)
            if batch_ctx is not None:
                batch_id, item_id = batch_ctx
                item_result = pb.BatchExecutionItemResult(
                    request_id=request_id,
                    item_id=item_id or "item-000001",
                    success=success,
                    output_payload=(output_payload or b'') if success else b'',
                    error_message=error_message if not success else "",
                    error_type=error_type if not success else "",
                    retryable=bool(retryable) if not success else False,
                    safe_message=safe_message if not success else "",
                )
                msg = pb.WorkerSchedulerMessage(
                    batch_run_result=pb.BatchExecutionResult(
                        batch_id=batch_id or "",
                        items=[item_result],
                    )
                )
            else:
                result = pb.TaskExecutionResult(
                    request_id=request_id,
                    success=success,
                    output_payload=(output_payload or b'') if success else b'', # Default to b'' if None
                    error_message=error_message if not success else "",
                    error_type=error_type if not success else "",
                    retryable=bool(retryable) if not success else False,
                    safe_message=safe_message if not success else "",
                )
                msg = pb.WorkerSchedulerMessage(run_result=result)
            self._send_message(msg)
            logger.debug(f"Queued task result for request_id={request_id}, success={success}")
        except Exception as e:
             # This shouldn't generally fail unless message creation has issues
             logger.error(f"Failed to create or queue task result for request_id={request_id}: {e}")
