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
import queue
import psutil
import importlib
import inspect
import functools
import typing
import socket
import ipaddress
from pathlib import Path
from dataclasses import dataclass
import collections.abc as cabc
from typing import Any, Callable, Dict, Optional, TypeVar, Iterator, List, Tuple, Iterable, get_args, get_origin
from types import ModuleType
import hashlib
import msgspec
try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None
import asyncio

# JWT verification for worker auth (scheduler-issued)
import jwt
_jwt_algorithms: Optional[ModuleType]
try:
    import jwt.algorithms as _jwt_algorithms
except Exception:  # pragma: no cover - optional crypto backend
    _jwt_algorithms = None
RSAAlgorithm: Optional[Any] = getattr(_jwt_algorithms, "RSAAlgorithm", None) if _jwt_algorithms else None
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
from .decorators import ResourceRequirements # Import ResourceRequirements for type hints if needed
from .errors import AuthError, CanceledError, FatalError, ResourceError, RetryableError, ValidationError

from .model_interface import ModelManagementInterface
from .downloader import CozyHubDownloader, ModelDownloader
from .model_ref_downloader import ModelRefDownloader
from .model_refs import parse_model_ref
from .types import Asset
from .model_cache import ModelCache, ModelCacheStats, ModelLocation
from .run_metrics_v1 import RunMetricsV1, best_effort_bytes_downloaded, best_effort_init_model_metrics, safe_json_bytes
from .injection import (
    InjectionSpec,
    ModelRef,
    ModelRefSource,
    parse_injection,
    type_qualname,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# Type variables for generic function signatures
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

# Generic type for action functions
ActionFunc = Callable[[Any, I], O]

HEARTBEAT_INTERVAL = 10  # seconds


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
    ctx: ActionContext
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
def _encode_ref_for_url(ref: str) -> str:
    ref = ref.strip().lstrip("/")
    parts = [urllib.parse.quote(p, safe="") for p in ref.split("/") if p]
    return "/".join(parts)


def _infer_mime_type(ref: str, head: bytes) -> str:
    # Prefer magic bytes when available.
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if len(head) >= 12 and head[0:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    # Fall back to extension.
    import mimetypes

    guessed, _ = mimetypes.guess_type(ref)
    return guessed or "application/octet-stream"


def _default_output_prefix(run_id: str) -> str:
    return f"runs/{run_id}/outputs/"


def _require_file_api_base_url() -> str:
    base = os.getenv("FILE_API_BASE_URL", "").strip()
    if not base:
        base = os.getenv("ORCHESTRATOR_HTTP_URL", "").strip()
    if not base:
        base = os.getenv("COZY_HUB_URL", "").strip()
    if not base:
        raise RuntimeError("FILE_API_BASE_URL is required for file operations")
    return base.rstrip("/")


def _require_file_api_token() -> str:
    token = os.getenv("FILE_API_TOKEN", "").strip()
    if not token:
        token = os.getenv("COZY_HUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("FILE_API_TOKEN is required for file operations")
    return token


def _http_request(
    method: str,
    url: str,
    token: str,
    owner: Optional[str] = None,
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
) -> urllib.request.Request:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    owner = (owner or "").strip()
    if owner:
        req.add_header("X-Cozy-Owner", owner)
    if content_type:
        req.add_header("Content-Type", content_type)
    return req


def _is_private_ip_str(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return True
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _url_is_blocked(url_str: str) -> bool:
    try:
        u = urllib.parse.urlparse(url_str)
    except Exception:
        return True
    if u.scheme not in ("http", "https"):
        return True
    host = (u.hostname or "").strip()
    if not host:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = str(sockaddr[0])
        if _is_private_ip_str(ip_str):
            return True
    return False


def _canonicalize_model_ref_string(raw: str) -> str:
    """
    Best-effort normalization of Cozy/HF model ref strings for allowlisting and caching identity.

    If the string doesn't parse as a phase-1 model ref, return it unchanged.
    """
    s = (raw or "").strip()
    if not s:
        return s
    try:
        parsed = parse_model_ref(s)
        if parsed.scheme == "cozy" and parsed.cozy is not None:
            return parsed.cozy.canonical()
        if parsed.scheme == "hf" and parsed.hf is not None:
            return parsed.hf.canonical()
        return s
    except Exception:
        return s

class _JWKSCache:
    def __init__(self, url: str, ttl_seconds: int = 300) -> None:
        self._url = url
        self._ttl_seconds = max(ttl_seconds, 0)
        self._lock = threading.Lock()
        self._fetched_at = 0.0
        self._keys: Dict[str, Any] = {}

    def _fetch(self) -> None:
        if RSAAlgorithm is None:
            raise RuntimeError(
                "PyJWT RSA support is unavailable (missing cryptography). "
                "Install gen-worker with a JWT/RSA-capable build of PyJWT."
            )
        with urllib.request.urlopen(self._url, timeout=5) as resp:
            body = resp.read()
        payload = json.loads(body.decode("utf-8"))
        keys: Dict[str, Any] = {}
        for jwk in payload.get("keys", []):
            kid = jwk.get("kid")
            if not kid:
                continue
            try:
                keys[kid] = RSAAlgorithm.from_jwk(json.dumps(jwk))
            except Exception:
                continue
        self._keys = keys
        self._fetched_at = time.time()

    def _needs_refresh(self) -> bool:
        if not self._keys:
            return True
        if self._ttl_seconds <= 0:
            return False
        return (time.time() - self._fetched_at) > self._ttl_seconds

    def get_key(self, kid: Optional[str]) -> Optional[Any]:
        with self._lock:
            if self._needs_refresh():
                self._fetch()
            if kid and kid in self._keys:
                return self._keys[kid]
            # refresh on miss (rotation)
            self._fetch()
            if kid and kid in self._keys:
                return self._keys[kid]
            return None

class ActionContext:
    """Context object passed to action functions, allowing cancellation."""
    def __init__(
        self,
        run_id: str,
        emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        owner: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        file_api_base_url: Optional[str] = None,
        file_api_token: Optional[str] = None,
        local_output_dir: Optional[str] = None,
        resolved_cozy_models_by_id: Optional[Dict[str, Any]] = None,
        required_models: Optional[List[str]] = None,
    ) -> None:
        self._run_id = run_id
        self._owner = owner
        self._user_id = user_id
        self._timeout_ms = timeout_ms
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._file_api_token = (file_api_token or "").strip() or None
        self._local_output_dir = (local_output_dir or "").strip() or None
        self._resolved_cozy_models_by_id = resolved_cozy_models_by_id
        self._required_models = list(required_models or [])
        self._started_at = time.time()
        self._deadline: Optional[float] = None
        if timeout_ms is not None and timeout_ms > 0:
            self._deadline = self._started_at + (timeout_ms / 1000.0)
        self._canceled = False
        self._cancel_event = threading.Event()
        self._emitter = emitter

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def owner(self) -> Optional[str]:
        return self._owner

    @property
    def user_id(self) -> Optional[str]:
        return self._user_id

    @property
    def timeout_ms(self) -> Optional[int]:
        return self._timeout_ms

    @property
    def deadline(self) -> Optional[float]:
        return self._deadline

    @property
    def device(self) -> "torch.device":
        """Torch device for this worker runtime (e.g. cuda:0 or cpu).

        Tenant functions should prefer `ctx.device` over choosing a device
        themselves so the platform can standardize device selection.
        """
        if torch is None:
            raise RuntimeError("torch is not available in this runtime")

        forced = os.getenv("WORKER_TORCH_DEVICE") or os.getenv("WORKER_DEVICE")
        if forced:
            return torch.device(forced)

        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        return torch.device("cpu")

    def _get_file_api_base_url(self) -> str:
        if self._file_api_base_url:
            return self._file_api_base_url.rstrip("/")
        return _require_file_api_base_url()

    def _get_file_api_token(self) -> str:
        if self._file_api_token:
            return self._file_api_token
        return _require_file_api_token()

    def _resolve_local_output_path(self, ref: str) -> Optional[str]:
        """
        Dev-only local output backend.

        When local_output_dir is set, ActionContext.save_* will write outputs to disk
        instead of using Cozy Hub's file API.
        """
        base = (self._local_output_dir or "").strip()
        if not base:
            return None

        # Normalize and prevent path traversal.
        ref = (ref or "").strip().replace("\\", "/").lstrip("/")
        if not ref:
            raise ValueError("invalid ref")
        out = (Path(base).expanduser() / ref).resolve()
        root = Path(base).expanduser().resolve()
        if root not in out.parents and out != root:
            raise ValueError("path traversal")
        return str(out)


    @property
    def resolved_cozy_models_by_id(self) -> Optional[Dict[str, Any]]:
        return self._resolved_cozy_models_by_id

    @property
    def required_models(self) -> List[str]:
        return list(self._required_models)


    def time_remaining_s(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.time())

    def is_canceled(self) -> bool:
        """Check if the action was canceled."""
        return self._canceled

    def cancel(self) -> None:
        """Mark the action as canceled."""
        if not self._canceled:
            self._canceled = True
            self._cancel_event.set()
            logger.info(f"Action {self.run_id} marked for cancellation.")

    def done(self) -> threading.Event:
        """Returns an event that is set when the action is cancelled."""
        return self._cancel_event

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit a progress/event payload (best-effort)."""
        if not self._emitter:
            logger.debug(f"emit({event_type}) dropped: no emitter configured")
            return
        event = {
            "run_id": self._run_id,
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
        }
        self._emitter(event)

    def progress(self, progress: float, stage: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"progress": progress}
        if stage is not None:
            payload["stage"] = stage
        self.emit("job.progress", payload)

    def log(self, message: str, level: str = "info") -> None:
        self.emit("job.log", {"message": message, "level": level})

    def save_bytes(self, ref: str, data: bytes) -> Asset:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes expects bytes")
        data = bytes(data)
        max_bytes = int(os.getenv("WORKER_MAX_OUTPUT_FILE_BYTES", str(200 * 1024 * 1024)))
        if max_bytes > 0 and len(data) > max_bytes:
            raise ValueError("output file too large")
        ref = ref.strip().lstrip("/")
        if not ref.startswith(_default_output_prefix(self.run_id)):
            raise ValueError(f"ref must start with '{_default_output_prefix(self.run_id)}'")

        local_path = self._resolve_local_output_path(ref)
        if local_path:
            p = Path(local_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            return Asset(
                ref=ref,
                owner=self.owner,
                local_path=str(p),
                mime_type=None,
                size_bytes=len(data),
                sha256=sha,
            )

        base = self._get_file_api_base_url()
        token = self._get_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"
        # Default behavior is upsert: PUT to the tenant file store.
        req = _http_request(
            "PUT",
            url,
            token,
            owner=self.owner,
            body=data,
            content_type="application/octet-stream",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(f"file save failed ({resp.status})")
                try:
                    meta = json.loads(body.decode("utf-8"))
                except Exception:
                    meta = {}
        except urllib.error.HTTPError as e:
            code = getattr(e, 'code', 0)
            if code in (401, 403):
                raise AuthError(f"file save unauthorized ({code}): check file_token validity") from e
            raise RuntimeError(f"file save failed ({code or 'unknown'})") from e
        finally:
            rm = getattr(self, "_run_metrics", None)
            if rm is not None:
                try:
                    rm.add_upload_time(int((time.monotonic() - t0) * 1000))
                except Exception:
                    pass

        return Asset(
            ref=ref,
            owner=self.owner,
            local_path=None,
            mime_type=str(meta.get("mime_type") or "") or None,
            size_bytes=int(meta.get("size_bytes") or 0) or len(data),
            sha256=str(meta.get("sha256") or "") or None,
        )

    def save_file(self, ref: str, local_path: str) -> Asset:
        with open(local_path, "rb") as f:
            data = f.read()
        return self.save_bytes(ref, data)

    def save_bytes_create(self, ref: str, data: bytes) -> Asset:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes_create expects bytes")
        data = bytes(data)
        max_bytes = int(os.getenv("WORKER_MAX_OUTPUT_FILE_BYTES", str(200 * 1024 * 1024)))
        if max_bytes > 0 and len(data) > max_bytes:
            raise ValueError("output file too large")
        ref = ref.strip().lstrip("/")
        if not ref.startswith(_default_output_prefix(self.run_id)):
            raise ValueError(f"ref must start with '{_default_output_prefix(self.run_id)}'")

        local_path = self._resolve_local_output_path(ref)
        if local_path:
            p = Path(local_path)
            # Create semantics: fail if already exists.
            if p.exists():
                raise RuntimeError("output path already exists")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            return Asset(
                ref=ref,
                owner=self.owner,
                local_path=str(p),
                mime_type=None,
                size_bytes=len(data),
                sha256=sha,
            )

        base = self._get_file_api_base_url()
        token = self._get_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"
        req = _http_request(
            "POST",
            url,
            token,
            owner=self.owner,
            body=data,
            content_type="application/octet-stream",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(f"file save failed ({resp.status})")
                try:
                    meta = json.loads(body.decode("utf-8"))
                except Exception:
                    meta = {}
        except urllib.error.HTTPError as e:
            code = getattr(e, "code", 0)
            if code in (401, 403):
                raise AuthError(f"file save unauthorized ({code}): check file_token validity") from e
            if code == 409:
                raise RuntimeError("output path already exists") from e
            raise RuntimeError(f"file save failed ({code or 'unknown'})") from e
        finally:
            rm = getattr(self, "_run_metrics", None)
            if rm is not None:
                try:
                    rm.add_upload_time(int((time.monotonic() - t0) * 1000))
                except Exception:
                    pass

        return Asset(
            ref=ref,
            owner=self.owner,
            local_path=None,
            mime_type=str(meta.get("mime_type") or "") or None,
            size_bytes=int(meta.get("size_bytes") or 0) or len(data),
            sha256=str(meta.get("sha256") or "") or None,
        )

    def save_file_create(self, ref: str, local_path: str) -> Asset:
        with open(local_path, "rb") as f:
            data = f.read()
        return self.save_bytes_create(ref, data)

    def save_bytes_overwrite(self, ref: str, data: bytes) -> Asset:
        # Back-compat alias: overwrite is the default save_bytes behavior.
        return self.save_bytes(ref, data)

    def save_file_overwrite(self, ref: str, local_path: str) -> Asset:
        return self.save_file(ref, local_path)

# Define the interceptor class correctly
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
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 0,  # 0 means infinite retries
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
        self._active_tasks: Dict[str, ActionContext] = {}
        self._active_tasks_lock = threading.Lock()
        self._active_function_counts: Dict[str, int] = {}
        self.max_concurrency = int(os.getenv("WORKER_MAX_CONCURRENCY", "0"))
        self._drain_timeout_seconds = int(os.getenv("WORKER_DRAIN_TIMEOUT_SECONDS", "0"))
        self._draining = False
        self._discovered_resources: Dict[str, ResourceRequirements] = {} # Store resources per function
        self._function_schemas: Dict[str, Tuple[bytes, bytes, Optional[bytes], bytes]] = {}  # func_name -> (input_schema_json, output_schema_json, delta_schema_json, injection_json)

        self._custom_runtime_cache: Dict[Tuple[str, str], Any] = {}  # (model_id, injected_type_qualname) -> runtime handle
        self._custom_runtime_locks: Dict[Tuple[str, str], threading.Lock] = {}

        # Local (non-NFS) cache for NFS->local snapshot localization.
        # Empty WORKER_LOCAL_MODEL_CACHE_DIR disables localization entirely.
        self._local_model_cache_dir = os.getenv("WORKER_LOCAL_MODEL_CACHE_DIR", "/tmp/cozy/local-model-cache").strip()
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
        self._reconnect_jitter = float(os.getenv("RECONNECT_JITTER_SECONDS", "1.0"))

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
        self._model_manager = resolved_model_manager
        self._downloader = downloader
        if self._downloader is None:
            base_url = os.getenv("COZY_HUB_URL", "").strip()
            allow_api = str(os.getenv("WORKER_ALLOW_COZY_HUB_API_RESOLVE", "") or "").strip().lower() in ("1", "true", "t", "yes", "y")
            token = (os.getenv("COZY_HUB_TOKEN", "").strip() or None) if allow_api else None
            # Default to the composite model-ref downloader:
            # - Cozy snapshots using orchestrator-resolved URLs (no Cozy Hub API calls)
            # - Hugging Face refs via huggingface_hub when installed
            self._downloader = ModelRefDownloader(
                cozy_base_url=base_url,
                cozy_token=token,
                allow_cozy_hub_api_resolve=allow_api,
            )
        self._supported_model_ids_from_scheduler: Optional[List[str]] = None  # allowlist from scheduler (repo refs)
        self._required_variant_refs_from_scheduler: Optional[List[str]] = None  # warm-start pinned variants
        # Signature-driven model selection mapping and allowlist (provided by orchestrator).
        self._release_model_id_by_key: Dict[str, str] = {}
        self._release_allowed_model_ids: Optional[set[str]] = None
        # Per-function model keyspace from baked discovery manifest (short_key -> model_ref).
        self._model_id_by_key_by_function: Dict[str, Dict[str, str]] = {}
        # Orchestrator-resolved manifests received in DeploymentArtifactConfig (startup prefetch baseline).
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
            # New contract: per-function mapping (allows endpoint-specific model sets and dtype constraints).
            mbf = manifest.get("models_by_function")
            if isinstance(mbf, dict):
                for fn_name, mapping in mbf.items():
                    if not isinstance(mapping, dict):
                        continue
                    out: Dict[str, str] = {}
                    for k, v in mapping.items():
                        key = str(k).strip()
                        if not key:
                            continue
                        if isinstance(v, dict):
                            ref = str(v.get("ref") or "").strip()
                        else:
                            ref = str(v or "").strip()
                        if not ref:
                            continue
                        out[key] = _canonicalize_model_ref_string(ref)
                    if out:
                        self._model_id_by_key_by_function[str(fn_name)] = out

            # Legacy contract: flat mapping at manifest["models"] (string refs).
            legacy_models = manifest.get("models")
            if isinstance(legacy_models, dict):
                self._release_model_id_by_key = {
                    str(k): _canonicalize_model_ref_string(str(v)) for k, v in legacy_models.items()
                }

            # Compute union allowlist for prefetch/guardrails.
            allowed: set[str] = set()
            for m in self._model_id_by_key_by_function.values():
                allowed.update(m.values())
            if self._release_model_id_by_key:
                allowed.update(self._release_model_id_by_key.values())
            self._release_allowed_model_ids = allowed or None

            if self._model_id_by_key_by_function:
                logger.info(
                    "Loaded model mappings from manifest for %d functions",
                    len(self._model_id_by_key_by_function),
                )
            elif self._release_model_id_by_key:
                logger.info(
                    "Loaded %d models from legacy manifest mapping",
                    len(self._release_model_id_by_key),
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

        logger.info(f"Created worker: ID={self.worker_id}, Scheduler={scheduler_addr}")

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

    def _set_scheduler_addr(self, addr: str) -> None:
        addr = addr.strip()
        if not addr:
            return
        self.scheduler_addr = addr
        if addr not in self.scheduler_addrs:
            self.scheduler_addrs.insert(0, addr)

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
            options = {"verify_aud": bool(self._jwt_audience)}
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
            run_id = event.get("run_id") or ""
            event_type = event.get("type") or ""
            payload = event.get("payload") or {}
            if "timestamp" not in payload:
                payload = dict(payload)
                payload["timestamp"] = event.get("timestamp", time.time())
            payload_json = json.dumps(payload).encode("utf-8")
            msg = pb.WorkerSchedulerMessage(
                worker_event=pb.WorkerEvent(
                    run_id=run_id,
                    event_type=event_type,
                    payload_json=payload_json,
                )
            )
            self._send_message(msg)
        except Exception:
            logger.exception("Failed to emit progress event")

    def _emit_worker_event_bytes(self, run_id: str, event_type: str, payload_json: bytes) -> None:
        """Best-effort worker->scheduler WorkerEvent emitter (must never fail a run)."""
        try:
            msg = pb.WorkerSchedulerMessage(
                worker_event=pb.WorkerEvent(
                    run_id=str(run_id or ""),
                    event_type=str(event_type or ""),
                    payload_json=bytes(payload_json or b"{}"),
                )
            )
            self._send_message(msg)
        except Exception:
            return


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
        func_name = func.__name__
        resources: ResourceRequirements = getattr(func, "_worker_resources", ResourceRequirements())

        try:
            hints = typing.get_type_hints(func, globalns=func.__globals__, include_extras=True)
        except Exception as exc:
            raise ValueError(f"failed to resolve type hints: {exc}")

        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            raise ValueError("must accept ctx: ActionContext as first arg")

        ctx_name = params[0].name
        ctx_type = hints.get(ctx_name)
        if ctx_type is not ActionContext:
            raise ValueError("first argument must be ctx: ActionContext")

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
        input_schema_json = json.dumps(input_schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
        output_schema_json = json.dumps(output_schema, separators=(",", ":"), sort_keys=True).encode("utf-8")
        delta_schema_json = None
        if delta_type is not None:
            delta_schema_json = json.dumps(msgspec.json.schema(delta_type), separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )

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
        func_name = func.__name__
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
            raise ValueError("websocket handler must accept (ctx: ActionContext, sock: RealtimeSocket, ...)")

        ctx_name = params[0].name
        if hints.get(ctx_name) is not ActionContext:
            raise ValueError("first argument must be ctx: ActionContext")

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

    def _materialize_assets(self, ctx: ActionContext, obj: Any) -> None:
        if isinstance(obj, Asset):
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
                    self._materialize_assets(ctx, getattr(obj, name))
                except Exception:
                    continue

    def _materialize_asset(self, ctx: ActionContext, asset: Asset) -> None:
        if asset.local_path:
            return
        ref = (asset.ref or "").strip()
        if not ref:
            return
        run_id = ctx.run_id

        base_dir = os.getenv("WORKER_RUN_DIR", "/tmp/cozy").rstrip("/")
        local_inputs_dir = os.path.join(base_dir, run_id, "inputs")
        os.makedirs(local_inputs_dir, exist_ok=True)
        cache_dir = os.getenv("WORKER_CACHE_DIR", os.path.join(base_dir, "cache")).rstrip("/")
        os.makedirs(cache_dir, exist_ok=True)

        max_bytes = int(os.getenv("WORKER_MAX_INPUT_FILE_BYTES", str(200 * 1024 * 1024)))

        # External URL inputs (download directly into the run folder).
        if ref.startswith("http://") or ref.startswith("https://"):
            if _url_is_blocked(ref):
                raise RuntimeError("input url blocked")
            ext = os.path.splitext(urllib.parse.urlparse(ref).path)[1] or os.path.splitext(ref)[1]
            name_hash = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:32]
            local_path = os.path.join(local_inputs_dir, f"{name_hash}{ext}")
            size, sha256_hex, mime = self._download_url_to_file(ref, local_path, max_bytes)
            asset.local_path = local_path
            if not asset.owner:
                asset.owner = self.owner
            asset.mime_type = mime
            asset.size_bytes = size
            asset.sha256 = sha256_hex
            return

        # Cozy Hub file ref (owner scoped) - use orchestrator file API with HEAD+cache.
        base = ctx._get_file_api_base_url()
        token = ctx._get_file_api_token()
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
                raise AuthError(f"file read unauthorized ({code}): check file_token validity") from e
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
                    raise AuthError(f"file read unauthorized ({code}): check file_token validity") from e
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

    def _download_url_to_file(self, src: str, dst: str, max_bytes: int) -> Tuple[int, str, Optional[str]]:
        attempts = int(os.getenv("WORKER_DOWNLOAD_RETRIES", "3"))
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < max(1, attempts):
            attempt += 1
            try:
                client = urllib.request.build_opener()
                req = urllib.request.Request(src, method="GET")
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

    def connect(self) -> bool:
        """Connect to the scheduler.

        Returns:
            bool: True if connection was successful, False otherwise.
        """
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

            # Start the receive loop in a separate thread *after* stream is initiated
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            logger.info(f"Successfully connected to scheduler at {self.scheduler_addr}")
            self._reconnect_count = 0
            return True

        except grpc.RpcError as e:
            # Access code() and details() methods for RpcError
            code = e.code() if hasattr(e, 'code') and callable(e.code) else grpc.StatusCode.UNKNOWN
            details = e.details() if hasattr(e, 'details') and callable(e.details) else str(e)
            leader = self._extract_leader_addr(details)
            if code == grpc.StatusCode.FAILED_PRECONDITION and leader:
                logger.warning(f"Scheduler returned not_leader for {self.scheduler_addr}; redirecting to {leader}")
                self._leader_hint = leader
                self._set_scheduler_addr(leader)
            else:
                logger.error(f"Failed to connect to scheduler: {code} - {details}")
            self._close_connection()
            return False
        except Exception as e:
            logger.exception(f"Unexpected error connecting to scheduler: {e}")
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

            function_concurrency = {}
            for func_name, req in self._discovered_resources.items():
                if req and req.max_concurrency:
                    function_concurrency[func_name] = int(req.max_concurrency)

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
                from .cozy_hub_policy import detect_worker_capabilities

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
                function_concurrency=function_concurrency,
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
                is_heartbeat=is_heartbeat
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
                                    run_id="",
                                    event_type="models.disk_inventory",
                                    payload_json=json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
                                )
                            )
                        )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to create or send registration/heartbeat: {e}")

    def run(self) -> None:
        """Run the worker, connecting to the scheduler and processing tasks."""
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        self._stop_event.clear()
        self._reconnect_count = 0 # Reset reconnect count on new run
        self._draining = False

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
                    backoff = self._reconnect_delay_base * (2 ** max(self._reconnect_count - 1, 0))
                    if self._reconnect_delay_max > 0:
                        backoff = min(backoff, self._reconnect_delay_max)
                    jitter = random.uniform(0, self._reconnect_jitter) if self._reconnect_jitter > 0 else 0
                    delay = backoff + jitter
                    logger.info(f"Connection attempt {self._reconnect_count} failed. Retrying in {delay:.2f} seconds...")
                    # Wait for delay, but break if stop event is set during wait
                    if self._stop_event.wait(delay):
                        logger.info("Stop requested during reconnect delay.")
                        break # Exit if stopped while waiting
            # After a failed attempt or disconnect, clear stop event for next retry
            if self._running:
                 self._stop_event.clear()

        # Cleanup after loop exits (either max attempts reached or manual stop)
        self.stop()

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
            for run_id in active_task_ids:
                ctx = self._active_tasks.get(run_id)
                if ctx:
                    logger.debug(f"Cancelling active task {run_id} during stop.")
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
                if leader:
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
            run_id = message.interrupt_run_cmd.run_id
            self._handle_interrupt_request(run_id)
        elif msg_type == "realtime_open_cmd":
            self._handle_realtime_open_cmd(message.realtime_open_cmd)
        elif msg_type == "realtime_frame":
            self._handle_realtime_frame(message.realtime_frame)
        elif msg_type == "realtime_close_cmd":
            self._handle_realtime_close_cmd(message.realtime_close_cmd)
        elif msg_type == "worker_event":
            self._handle_worker_event_from_scheduler(message.worker_event)
        # Add handling for other message types if needed (e.g., config updates)
        elif msg_type == 'deployment_artifact_config':
            cfg = message.deployment_artifact_config
            resolved_by_variant = dict(getattr(cfg, "resolved_cozy_models_by_variant_ref", {}) or {})
            logger.info(
                "Received DeploymentArtifactConfig (supported=%d required=%d resolved=%d)",
                len(cfg.supported_repo_refs),
                len(cfg.required_variant_refs),
                len(resolved_by_variant),
            )
            self._supported_model_ids_from_scheduler = [
                _canonicalize_model_ref_string(str(v)) for v in list(cfg.supported_repo_refs)
            ]
            try:
                # Optional label->id mapping for signature-driven model selection.
                self._release_model_id_by_key = {
                    str(k): _canonicalize_model_ref_string(str(v))
                    for k, v in dict(cfg.repo_ref_by_key).items()
                }
            except Exception:
                self._release_model_id_by_key = {}

            # Baseline resolved manifests for Cozy model downloads (issue #66/#238).
            self._resolved_cozy_models_by_id_baseline = self._canonicalize_resolved_models_map(
                resolved_by_variant
            )

            self._release_allowed_model_ids = set(self._supported_model_ids_from_scheduler)
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
        cache_dir = Path(os.getenv("WORKER_MODEL_CACHE_DIR") or "/tmp/cozy/models")
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
            from .model_ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id

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
                                    worker_event=pb.WorkerEvent(run_id="", event_type="model.download.completed", payload_json=payload)
                                )
                            )
                            self._send_message(
                                pb.WorkerSchedulerMessage(
                                    worker_event=pb.WorkerEvent(run_id="", event_type="model.ready", payload_json=json.dumps({"model_id": canon}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
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
                                    worker_event=pb.WorkerEvent(run_id="", event_type="model.cached", payload_json=payload)
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
                                worker_event=pb.WorkerEvent(run_id="", event_type="model.download.started", payload_json=payload)
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
                                worker_event=pb.WorkerEvent(run_id="", event_type="model.ready", payload_json=payload)
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
                                worker_event=pb.WorkerEvent(run_id="", event_type="model.cached", payload_json=payload)
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
                                worker_event=pb.WorkerEvent(run_id="", event_type="model.download.completed", payload_json=payload)
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
                                    worker_event=pb.WorkerEvent(run_id="", event_type="model.url_refresh", payload_json=payload)
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
                                worker_event=pb.WorkerEvent(run_id="", event_type="model.download.failed", payload_json=payload)
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
        snap_dir = cache_dir / "cozy" / "snapshots" / digest
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

            from .model_ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id
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
                    worker_event=pb.WorkerEvent(run_id="", event_type="model.load.started", payload_json=payload)
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
                # load_model_into_vram is async
                success = asyncio.run(self._model_manager.load_model_into_vram(model_id))
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
                    worker_event=pb.WorkerEvent(run_id="", event_type=ev_type, payload_json=payload)
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
                        worker_event=pb.WorkerEvent(run_id="", event_type="model.unload.failed", payload_json=payload)
                    )
                )
            except Exception:
                pass
            return

        try:
            payload = json.dumps({"model_id": model_id}, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self._send_message(
                pb.WorkerSchedulerMessage(
                    worker_event=pb.WorkerEvent(run_id="", event_type="model.unload.started", payload_json=payload)
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
                    worker_event=pb.WorkerEvent(run_id="", event_type=ev_type, payload_json=payload)
                )
            )
        except Exception:
            pass

    def _handle_run_request(self, request: TaskExecutionRequest) -> None:
        """Handle a task execution request from the scheduler."""
        run_id = request.run_id
        function_name = request.function_name
        input_payload = request.input_payload
        required_model_id_for_exec = ""
        timeout_ms = int(getattr(request, "timeout_ms", 0) or 0)
        owner = str(getattr(request, "owner", "") or "") or (self.owner or "")
        user_id = str(getattr(request, "user_id", "") or "")
        file_base_url = str(getattr(request, "file_base_url", "") or "")
        file_token = str(getattr(request, "file_token", "") or "")
        resolved_cozy_models_by_id = dict(getattr(request, "resolved_cozy_models_by_id", {}) or {})

        required_models_raw = list(getattr(request, "required_variant_refs", []) or [])
        if required_models_raw:
            required_model_id_for_exec = str(required_models_raw[0] or "").strip()

        logger.info(f"Received Task request: run_id={run_id}, function={function_name}, model='{required_model_id_for_exec or 'None'}'")

        spec = self._task_specs.get(function_name)
        if not spec:
            error_msg = f"Unknown function requested: {function_name}"
            logger.error(error_msg)
            self._send_task_result(run_id, False, None, "internal", False, "internal error", error_msg)
            return
        if self.max_input_bytes > 0 and len(input_payload) > self.max_input_bytes:
            error_msg = f"Input payload too large: {len(input_payload)} bytes (max {self.max_input_bytes})"
            logger.error(error_msg)
            self._send_task_result(run_id, False, None, "validation", False, "invalid input", error_msg)
            return
        if self._draining:
            error_msg = "Worker is draining; refusing new tasks"
            logger.warning(error_msg)
            self._send_task_result(run_id, False, None, "retryable", True, "worker busy", error_msg)
            return

        # required_variant_refs are pinned variant refs chosen by the scheduler; the worker must not guess.
        required_models: List[str] = []
        for raw in required_models_raw:
            s = str(raw or "").strip()
            if not s:
                continue
            required_models.append(_canonicalize_model_ref_string(s))

        ctx = ActionContext(
            run_id,
            emitter=self._emit_progress_event,
            owner=owner or None,
            user_id=user_id or None,
            timeout_ms=timeout_ms if timeout_ms > 0 else None,
            file_api_base_url=file_base_url or None,
            file_api_token=file_token or None,
            local_output_dir=None,
            resolved_cozy_models_by_id=resolved_cozy_models_by_id or None,
            required_models=required_models or None,
        )
        # Add to active tasks *before* starting thread
        with self._active_tasks_lock:
             # Double-check if task is already active (race condition mitigation)
             if run_id in self._active_tasks:
                  error_msg = f"Task with run_id {run_id} is already active (race condition?)."
                  logger.error(error_msg)
                  return # Avoid starting duplicate thread
             if self.max_concurrency > 0 and len(self._active_tasks) >= self.max_concurrency:
                  error_msg = f"Worker concurrency limit reached ({self.max_concurrency})."
                  logger.error(error_msg)
                  self._send_task_result(run_id, False, None, "retryable", True, "worker busy", error_msg)
                  return
             resource_req = self._discovered_resources.get(function_name)
             func_limit = resource_req.max_concurrency if resource_req and resource_req.max_concurrency else 0
             if func_limit > 0 and self._active_function_counts.get(function_name, 0) >= func_limit:
                  error_msg = f"Function concurrency limit reached for {function_name} ({func_limit})."
                  logger.error(error_msg)
                  self._send_task_result(run_id, False, None, "retryable", True, "worker busy", error_msg)
                  return
             self._active_tasks[run_id] = ctx
             if func_limit > 0:
                  self._active_function_counts[function_name] = self._active_function_counts.get(function_name, 0) + 1

        # Execute function in a separate thread to avoid blocking the receive loop
        thread = threading.Thread(
            target=self._execute_task,
            args=(ctx, spec, input_payload),
            daemon=True,
        )
        thread.start()

    def _handle_interrupt_request(self, run_id: str) -> None:
        """Handle a request to interrupt/cancel a running task."""
        logger.info(f"Received interrupt request for run_id={run_id}")
        with self._active_tasks_lock:
            ctx = self._active_tasks.get(run_id)
            if ctx:
                ctx.cancel() # Set internal flag and event
            else:
                logger.warning(f"Could not interrupt task {run_id}: Not found in active tasks.")

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
        user_id = str(getattr(cmd, "user_id", "") or "")
        timeout_ms = int(getattr(cmd, "timeout_ms", 0) or 0) or None
        file_base_url = str(getattr(cmd, "file_base_url", "") or "")
        file_token = str(getattr(cmd, "file_token", "") or "")
        ctx = ActionContext(
            session_id,
            emitter=self._emit_progress_event,
            owner=owner or None,
            user_id=user_id or None,
            timeout_ms=timeout_ms,
            file_api_base_url=file_base_url or None,
            file_api_token=file_token or None,
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
                from .model_ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id
                baseline = getattr(self, "_resolved_cozy_models_by_id_baseline", None) or None
                resolved_tok = set_resolved_cozy_models_by_id(getattr(ctx, "resolved_cozy_models_by_id", None) or baseline)
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
        ctx: ActionContext,
        spec: _TaskSpec,
        input_payload: bytes,
    ) -> None:
        """Execute a discovered task handler and send result/events back."""
        run_id = ctx.run_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""  # internal/legacy
        retryable = False
        success = False

        # Metrics (best-effort): never fail a run.
        resolved_map = getattr(ctx, "resolved_cozy_models_by_id", None) or None
        rm = RunMetricsV1(
            run_id=str(run_id or ""),
            function_name=str(spec.name or ""),
            required_models=list(getattr(ctx, "required_models", []) or []),
            resolved_cozy_models_by_id=resolved_map,
        )
        # Attach to ctx so ActionContext.save_* and injection paths can accumulate.
        try:
            setattr(ctx, "_run_metrics", rm)
        except Exception:
            pass
        rm.mark_compute_started()
        if rm.compute_started_at:
            self._emit_worker_event_bytes(run_id, "metrics.compute.started", safe_json_bytes({"at": rm.compute_started_at}))

        # Initialize model cache_state for required models using worker's cache hints.
        try:
            vram = self._model_cache.get_vram_models() if self._model_cache else []
            disk = self._model_cache.get_disk_models() if self._model_cache else []
            best_effort_init_model_metrics(rm, rm.required_models, vram_models=vram, disk_models=disk, cache_dir=Path(os.getenv("WORKER_MODEL_CACHE_DIR", "/tmp/cozy/models")))
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

        from .model_ref_downloader import reset_resolved_cozy_models_by_id, set_resolved_cozy_models_by_id

        baseline = getattr(self, "_resolved_cozy_models_by_id_baseline", None) or None
        resolved_map = getattr(ctx, "resolved_cozy_models_by_id", None) or baseline
        resolved_tok = set_resolved_cozy_models_by_id(resolved_map)

        models_in_use: set[str] = set()
        try:
            if ctx.is_canceled():
                raise CanceledError("canceled")

            # Decode payload strictly.
            input_obj = msgspec.msgpack.decode(input_payload, type=spec.payload_type)
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
                model_id = self._resolve_model_id_for_injection(spec.name, inj, payload=input_obj)
                canon_model_id = _canonicalize_model_ref_string(str(model_id or "").strip()) if model_id else ""
                if canon_model_id and canon_model_id not in models_in_use:
                    self._model_use_enter(canon_model_id)
                    models_in_use.add(canon_model_id)
                call_kwargs[inj.param_name] = self._resolve_injected_value(ctx, inj.param_type, model_id, inj)

            # Invoke.
            t_infer0 = time.monotonic()
            if inspect.iscoroutinefunction(spec.func):
                result = asyncio.run(spec.func(**call_kwargs))
            elif inspect.isasyncgenfunction(spec.func):
                result = spec.func(**call_kwargs)
            else:
                result = spec.func(**call_kwargs)

            if ctx.is_canceled():
                raise CanceledError("canceled")

            if spec.output_mode == "single":
                if spec.output_type is not None and not isinstance(result, spec.output_type):
                    raise TypeError(f"Function {spec.name} returned {type(result)!r}, expected {spec.output_type!r}")
                output_payload = msgspec.msgpack.encode(msgspec.to_builtins(result))
                if self.max_output_bytes > 0 and len(output_payload) > self.max_output_bytes:
                    raise ValueError(f"Output payload too large: {len(output_payload)} bytes (max {self.max_output_bytes})")
                success = True
            else:
                # Incremental output: the function returns an iterator of delta structs.
                max_delta_bytes = int(os.getenv("WORKER_MAX_OUTPUT_DELTA_BYTES", "65536"))
                max_events = int(os.getenv("WORKER_MAX_OUTPUT_DELTA_EVENTS", "0"))
                count = 0

                def emit_delta(delta_obj: msgspec.Struct) -> None:
                    nonlocal count
                    payload = msgspec.to_builtins(delta_obj)
                    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    if max_delta_bytes > 0 and len(raw) > max_delta_bytes:
                        raw = json.dumps({"truncated": True}, separators=(",", ":"), sort_keys=True).encode("utf-8")
                    self._send_message(
                        pb.WorkerSchedulerMessage(
                            worker_event=pb.WorkerEvent(run_id=run_id, event_type="output.delta", payload_json=raw)
                        )
                    )
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
                self._send_message(
                    pb.WorkerSchedulerMessage(
                        worker_event=pb.WorkerEvent(run_id=run_id, event_type="output.completed", payload_json=b"{}")
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

            logger.info("Task %s completed successfully.", run_id)

        except Exception as e:
            error_type, retryable, safe_message, error_message = self._map_exception(e)
            if spec.output_mode == "incremental":
                try:
                    payload = json.dumps({"error_type": error_type, "message": safe_message}, separators=(",", ":")).encode("utf-8")
                    self._send_message(
                        pb.WorkerSchedulerMessage(
                            worker_event=pb.WorkerEvent(run_id=run_id, event_type="output.error", payload_json=payload)
                        )
                    )
                except Exception:
                    pass
            success = False
        finally:
            reset_resolved_cozy_models_by_id(resolved_tok)

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
                self._emit_worker_event_bytes(run_id, "metrics.compute.completed", safe_json_bytes({"at": rm.compute_completed_at}))

            # Emit canonical metric events if values exist.
            try:
                rm.finalize()
                for ev_type, payload in rm.canonical_events():
                    # compute.* already emitted in real time above
                    if ev_type in ("metrics.compute.started", "metrics.compute.completed"):
                        continue
                    self._emit_worker_event_bytes(run_id, ev_type, safe_json_bytes(payload))
            except Exception:
                pass
            # Emit extended debug payload at end (best-effort).
            try:
                self._emit_worker_event_bytes(run_id, "metrics.run", safe_json_bytes(rm.to_metrics_run_payload()))
            except Exception:
                pass

            self._send_task_result(run_id, success, output_payload, error_type, bool(retryable), safe_message, error_message)

            with self._active_tasks_lock:
                self._active_tasks.pop(run_id, None)
                func_limit = spec.resources.max_concurrency or 0
                if func_limit > 0:
                    current = self._active_function_counts.get(spec.name, 0) - 1
                    if current <= 0:
                        self._active_function_counts.pop(spec.name, None)
                    else:
                        self._active_function_counts[spec.name] = current

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
                from .mount_backend import mount_backend_for_path
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
                from .pipeline_loader import LocalModelCache  # local import to avoid heavy import at worker init

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
            from .mount_backend import mount_backend_for_path, volume_key_for_path

            p = path or Path(os.getenv("WORKER_MODEL_CACHE_DIR", "/tmp/cozy/models"))
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

    def _resolve_injected_value(self, ctx: ActionContext, requested_type: Any, model_id: str, inj: InjectionSpec) -> Any:
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
                        rm.set_initial_model_state(canon, "hot_vram", rm.models.get(canon, None).snapshot_digest if canon in rm.models else None)
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
                try:
                    from diffusers import DiffusionPipeline  # type: ignore
                except Exception:
                    DiffusionPipeline = None  # type: ignore

                if (
                    obj is None
                    and DiffusionPipeline is not None
                    and isinstance(requested_type, type)
                    and issubclass(requested_type, DiffusionPipeline)
                ):
                    local = None
                    canon = str(model_id)
                    try:
                        parsed = parse_model_ref(str(model_id))
                        canon = parsed.cozy.canonical() if parsed.scheme == "cozy" and parsed.cozy is not None else str(model_id)
                    except Exception:
                        canon = str(model_id)
                    try:
                        p = Path(model_id)
                        if p.exists():
                            local = p.as_posix()
                    except Exception:
                        local = None

                    if local is None:
                        if self._downloader is None:
                            raise ValueError("diffusers pipeline injection requires a model downloader")
                        cache_dir = os.getenv("WORKER_MODEL_CACHE_DIR", "/tmp/cozy/models")
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

                        # Ensure the shared cache has a durable snapshot under WORKER_MODEL_CACHE_DIR
                        # before any NFS->local localization happens, so future pods can warm-start.
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
                                                worker_event=pb.WorkerEvent(run_id="", event_type="model.cached", payload_json=payload)
                                            )
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    # Detect mount backend (NFS vs local) and localize snapshot to local disk if needed.
                    try:
                        from .mount_backend import mount_backend_for_path

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

                    kwargs: dict[str, Any] = {}

                    # Cozy pipeline YAML is authoritative; ensure diffusers can load even
                    # if the artifact only shipped cozy.pipeline.lock.yaml/yaml (no model_index.json).
                    try:
                        from gen_worker.cozy_pipeline_spec import (
                            cozy_custom_pipeline_arg,
                            ensure_diffusers_model_index_json,
                            load_cozy_pipeline_spec,
                        )

                        root = Path(local)
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
                        from gen_worker.pipeline_loader import detect_diffusers_variant  # type: ignore

                        variant = detect_diffusers_variant(Path(local))
                        if variant is not None:
                            kwargs["variant"] = variant
                    except Exception:
                        pass

                    # Choose a dtype that won't explode RAM on CPU. Prefer matching the variant.
                    try:
                        if torch is not None:
                            device_is_cuda = str(ctx.device).startswith("cuda") and torch.cuda.is_available()
                            variant = str(kwargs.get("variant") or "").strip().lower()
                            if device_is_cuda:
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
                        obj = requested_type.from_pretrained(local, **kwargs)
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
                            obj = requested_type.from_pretrained(local, **kwargs2)
                            if rm is not None:
                                rm.add_pipeline_init_time(int((time.monotonic() - t_pi0) * 1000))
                        else:
                            raise

                if obj is None:
                    t_pi0 = time.monotonic()
                    obj = requested_type.from_pretrained(model_id)
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
                        t_to0 = time.monotonic()
                        obj = obj.to(str(ctx.device))
                        if rm is not None:
                            rm.add_gpu_load_time(int((time.monotonic() - t_to0) * 1000))
                except Exception:
                    pass
                self._custom_runtime_cache[key] = obj
                return obj

        raise ValueError(f"no injection provider for type {qn} (model_id={model_id})")

    def _resolve_model_id_for_injection(self, fn_name: str, inj: InjectionSpec, payload: msgspec.Struct) -> str:
        # Prefer per-function model keyspace from baked discovery manifest.
        model_id_by_key = self._model_id_by_key_by_function.get(fn_name) or self._release_model_id_by_key
        allowed_ids: Optional[set[str]] = None
        if model_id_by_key:
            allowed_ids = set(model_id_by_key.values())

        if inj.model_ref.source in (ModelRefSource.FIXED, ModelRefSource.RELEASE):
            raw = inj.model_ref.key.strip()
            if not raw:
                raise ValueError(f"empty fixed ModelRef for injection param: {inj.param_name}")
            if raw in model_id_by_key:
                model_id = model_id_by_key[raw]
                self._enforce_model_allowlist(model_id, inj, allowed_ids=allowed_ids)
                return model_id
            # Allow direct model refs for FIXED injections (advanced usage).
            # Do not apply per-function allowlists when no short-key mapping was used.
            return raw

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
            # Payload-based selection must use a short-key from the baked discovery manifest.
            # Do not allow arbitrary refs in the payload by default, otherwise the
            # orchestrator cannot route cache-aware and the allowlist becomes harder
            # to reason about.
            if not model_id_by_key:
                raise ValueError(
                    "payload model selection is not configured: no short-key mapping is available "
                    "(expected /app/.cozy/manifest.json to provide models_by_function)"
                )
            if key not in model_id_by_key:
                allowed = sorted(model_id_by_key.keys())
                head = allowed[:20]
                suffix = ""
                if len(allowed) > len(head):
                    suffix = f" (+{len(allowed) - len(head)} more)"
                raise ValueError(
                    f"unknown model key {key!r} for payload field {field!r}; allowed keys: {head}{suffix}"
                )
            model_id = model_id_by_key[key]
            self._enforce_model_allowlist(model_id, inj, allowed_ids=allowed_ids)
            return model_id

        raise ValueError(f"unknown ModelRef source: {inj.model_ref.source!r}")

    def _enforce_model_allowlist(self, model_id: str, inj: InjectionSpec, *, allowed_ids: Optional[set[str]] = None) -> None:
        # Enforce BOTH:
        # - any per-function mapping restriction (allowed_ids), AND
        # - the release-level allowlist from the scheduler (defense-in-depth).
        if allowed_ids is not None and model_id not in allowed_ids:
            raise ValueError(f"model_id not allowed for endpoint: {model_id!r} (injection param {inj.param_name})")
        if self._release_allowed_model_ids is not None and model_id not in self._release_allowed_model_ids:
            raise ValueError(f"model_id not allowed for release: {model_id!r} (injection param {inj.param_name})")

    def _send_task_result(
        self,
        run_id: str,
        success: bool,
        output_payload: Optional[bytes],
        error_type: str,
        retryable: bool,
        safe_message: str,
        error_message: str,
    ) -> None:
        """Send a task execution result back to the scheduler via the queue."""
        try:
            result = pb.TaskExecutionResult(
                run_id=run_id,
                success=success,
                output_payload=(output_payload or b'') if success else b'', # Default to b'' if None
                error_message=error_message if not success else "",
                error_type=error_type if not success else "",
                retryable=bool(retryable) if not success else False,
                safe_message=safe_message if not success else "",
            )
            msg = pb.WorkerSchedulerMessage(run_result=result)
            self._send_message(msg)
            logger.debug(f"Queued task result for run_id={run_id}, success={success}")
        except Exception as e:
             # This shouldn't generally fail unless message creation has issues
             logger.error(f"Failed to create or queue task result for run_id={run_id}: {e}")
