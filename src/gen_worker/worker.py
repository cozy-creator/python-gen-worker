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
from typing import Any, Callable, Dict, Optional, TypeVar, Iterator, List, Tuple
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
from .errors import CanceledError, FatalError, ResourceError, RetryableError, ValidationError

from .model_interface import ModelManagementInterface
from .downloader import CozyHubDownloader, ModelDownloader
from .types import Asset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for logger

# Type variables for generic function signatures
I = TypeVar('I')  # Input type
O = TypeVar('O')  # Output type

# Generic type for action functions
ActionFunc = Callable[[Any, I], O]

HEARTBEAT_INTERVAL = 10  # seconds


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
    body: Optional[bytes] = None,
    content_type: Optional[str] = None,
) -> urllib.request.Request:
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    tenant_id = os.getenv("TENANT_ID", "").strip()
    if tenant_id:
        req.add_header("X-Cozy-Tenant-Id", tenant_id)
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
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self._run_id = run_id
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._timeout_ms = timeout_ms
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
    def tenant_id(self) -> Optional[str]:
        return self._tenant_id

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

        base = _require_file_api_base_url()
        token = _require_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"
        # Default behavior is upsert: PUT to the tenant file store.
        req = _http_request("PUT", url, token, body=data, content_type="application/octet-stream")
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
            raise RuntimeError(f"file save failed ({getattr(e, 'code', 'unknown')})") from e

        return Asset(
            ref=ref,
            tenant_id=self.tenant_id,
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

        base = _require_file_api_base_url()
        token = _require_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"
        req = _http_request("POST", url, token, body=data, content_type="application/octet-stream")
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
            if getattr(e, "code", None) == 409:
                raise RuntimeError("output path already exists") from e
            raise RuntimeError(f"file save failed ({getattr(e, 'code', 'unknown')})") from e

        return Asset(
            ref=ref,
            tenant_id=self.tenant_id,
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

    def __init__(
        self,
        scheduler_addr: str = "localhost:8080",
        scheduler_addrs: Optional[List[str]] = None,
        user_module_names: List[str] = ["functions"], # Add new parameter for user modules
        worker_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        use_tls: bool = False,
        reconnect_delay: int = 5,
        max_reconnect_attempts: int = 0,  # 0 means infinite retries
        model_manager: Optional[ModelManagementInterface] = None, # Optional model manager
        downloader: Optional[ModelDownloader] = None,  # Optional model downloader
    ) -> None:
        """Initialize a new worker.

        Args:
            scheduler_addr: Address of the scheduler service.
            scheduler_addrs: Optional list of seed scheduler addresses.
            user_module_names: List of Python module names containing user-defined @worker_function functions.
            worker_id: Unique ID for this worker (generated if not provided).
            auth_token: Optional authentication token.
            use_tls: Whether to use TLS for the connection.
            reconnect_delay: Seconds to wait between reconnection attempts.
            max_reconnect_attempts: Max reconnect attempts (0 = infinite).
            model_manager: Optional model manager.
            downloader: Optional model downloader.
        """
        self.scheduler_addr = scheduler_addr
        self.scheduler_addrs = self._normalize_scheduler_addrs(scheduler_addr, scheduler_addrs)
        self.user_module_names = user_module_names # Store module names
        self.worker_id = worker_id or f"py-worker-{os.getpid()}"
        self.auth_token = auth_token
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

        self.deployment_id = os.getenv("DEPLOYMENT_ID", "") # Read DEPLOYMENT_ID env var
        if not self.deployment_id:
            logger.warning("DEPLOYMENT_ID environment variable not set for this worker!")

        self.tenant_id = os.getenv("TENANT_ID", "")
        self.runpod_pod_id = os.getenv("RUNPOD_POD_ID", "") # Read injected pod ID
        if not self.runpod_pod_id:
            logger.warning("RUNPOD_POD_ID environment variable not set for this worker!")

        logger.info(f"RUNPOD_POD_ID: {self.runpod_pod_id}")

        self._actions: Dict[str, Callable[[ActionContext, Optional[Any], bytes], bytes]] = {}
        self._active_tasks: Dict[str, ActionContext] = {}
        self._active_tasks_lock = threading.Lock()
        self._active_function_counts: Dict[str, int] = {}
        self.max_concurrency = int(os.getenv("WORKER_MAX_CONCURRENCY", "0"))
        self._drain_timeout_seconds = int(os.getenv("WORKER_DRAIN_TIMEOUT_SECONDS", "0"))
        self._draining = False
        self._discovered_resources: Dict[str, ResourceRequirements] = {} # Store resources per function
        self._function_schemas: Dict[str, Tuple[bytes, bytes]] = {}  # func_name -> (input_schema_json, output_schema_json)

        self._gpu_busy_lock = threading.Lock()
        self._is_gpu_busy = False

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
            token = os.getenv("COZY_HUB_TOKEN", "").strip() or None
            if base_url:
                self._downloader = CozyHubDownloader(base_url, token=token)
        self._supported_model_ids_from_scheduler: Optional[List[str]] = None # To store IDs from scheduler
        self._model_init_done_event = threading.Event() # To signal model init is complete

        if self._model_manager:
            logger.info(f"ModelManager of type '{type(self._model_manager).__name__}' provided.")
        else:
            logger.info("No ModelManager provided. Worker operating in simple mode regarding models.")
            self._model_init_done_event.set() # No model init to wait for if no manager
        if self._downloader:
            logger.info(f"ModelDownloader of type '{type(self._downloader).__name__}' configured.")

        logger.info(f"Created worker: ID={self.worker_id}, DeploymentID={self.deployment_id or 'N/A'}, Scheduler={scheduler_addr}")

        # Discover functions before setting signals? Maybe after. Let's do it here.
        self._discover_and_register_functions()

        self._verify_auth_token()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

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

    def _verify_auth_token(self) -> None:
        if not self.auth_token or not self._jwks_cache:
            return
        try:
            header = jwt.get_unverified_header(self.auth_token)
            kid = header.get("kid")
            key = self._jwks_cache.get_key(kid)
            if not key:
                raise ValueError("JWKS key not found for token")
            options = {"verify_aud": bool(self._jwt_audience)}
            jwt.decode(
                self.auth_token,
                key=key,
                algorithms=["RS256"],
                audience=self._jwt_audience or None,
                issuer=self._jwt_issuer or None,
                options=options,
            )
            logger.info("Worker auth token verified against scheduler JWKS.")
        except Exception as e:
            logger.error(f"Worker auth token verification failed: {e}")
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


    def _set_gpu_busy_status(self, busy: bool, func_name_for_log: str = "") -> None:
        with self._gpu_busy_lock:
            if self._is_gpu_busy == busy:
                return
            self._is_gpu_busy = busy
        if func_name_for_log:
            logger.info(f"GPU status changed to {busy} due to function '{func_name_for_log}'.")
        else:
            logger.info(f"GPU status changed to {busy}.")


    def _get_gpu_busy_status(self) -> bool:
        with self._gpu_busy_lock:
            return self._is_gpu_busy


    def _discover_and_register_functions(self) -> None:
        """Discover and register functions marked with @worker_function."""
        logger.info(f"Discovering worker functions in modules: {self.user_module_names}...")
        discovered_count = 0
        for module_name in self.user_module_names:
            try:
                module = importlib.import_module(module_name)
                logger.debug(f"Inspecting module: {module_name}")
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and hasattr(obj, '_is_worker_function'):
                        if getattr(obj, '_is_worker_function') is True:
                            # Found a decorated function
                            original_func = obj # Keep reference to the actual decorated function
                            func_name = original_func.__name__ # Use the real function name

                            if func_name in self._actions:
                                logger.warning(f"Function '{func_name}' from module '{module_name}' conflicts with an already registered function. Skipping.")
                                continue

                            resources: ResourceRequirements = getattr(original_func, '_worker_resources', ResourceRequirements())
                            self._discovered_resources[func_name] = resources

                            expects_pipeline_flag = resources.expects_pipeline_arg
                            payload_type = self._infer_payload_type(original_func, expects_pipeline_flag)
                            return_type = self._infer_return_type(original_func)
                            if payload_type is None or return_type is None:
                                logger.error(
                                    "Skipping function '%s' due to invalid or missing payload type annotation.",
                                    func_name,
                                )
                                continue

                            try:
                                input_schema = msgspec.json.schema(payload_type)
                                output_schema = msgspec.json.schema(return_type)
                                self._function_schemas[func_name] = (
                                    json.dumps(input_schema, separators=(",", ":"), sort_keys=True).encode("utf-8"),
                                    json.dumps(output_schema, separators=(",", ":"), sort_keys=True).encode("utf-8"),
                                )
                            except Exception as exc:
                                logger.error("Failed to generate msgspec JSON schema for '%s': %s", func_name, exc)
                                continue

                            # Create the wrapper for gRPC/msgpack interaction
                            def create_wrapper(
                                captured_func: Callable[..., Any],
                                captured_name: str,
                                captured_payload_type: type[msgspec.Struct],
                                captured_return_type: type[msgspec.Struct],
                                func_expects_pipeline: bool = False,
                            ) -> Callable[[ActionContext, Optional[Any], bytes], bytes]:
                                @functools.wraps(captured_func) # Preserve metadata of original user func
                                def wrapper(ctx: ActionContext, pipeline_instance: Optional[Any], input_bytes: bytes) -> bytes:
                                    try:
                                        input_obj = msgspec.msgpack.decode(input_bytes, type=captured_payload_type)
                                        self._materialize_assets(ctx.run_id, input_obj)
                                        # Pass the context and deserialized input to the *original* user function
                                        if func_expects_pipeline: # Only pass pipeline if function expects it
                                            if pipeline_instance is None:
                                                err_msg = f"Function '{captured_name}' expected a pipeline argument, but None was provided by the Worker core."
                                                logger.error(err_msg)
                                                raise ValueError(err_msg)
                                            result = captured_func(ctx, pipeline_instance, input_obj)
                                        else:
                                            result = captured_func(ctx, input_obj) # For functions not needing a model

                                        if ctx.is_canceled():
                                             raise InterruptedError("Task was canceled during execution")
                                        if not isinstance(result, captured_return_type):
                                            raise TypeError(
                                                f"Function {captured_name} returned {type(result)!r}, "
                                                f"expected {captured_return_type!r}"
                                            )
                                        # Ensure result is bytes after msgspec msgpack serialization
                                        # Encode as msgpack *maps* (field names preserved) so
                                        # downstream services (scheduler/orchestrator) can
                                        # reliably discover nested Asset objects by key.
                                        packed_result = msgspec.msgpack.encode(msgspec.to_builtins(result))
                                        if not isinstance(packed_result, bytes):
                                            raise TypeError(
                                                f"Function {captured_name} did not return msgspec-serializable data resulting in bytes"
                                            )
                                        return packed_result
                                    except InterruptedError as ie: # Catch cancellation specifically
                                        logger.warning(f"Function {captured_name} run {ctx.run_id} was interrupted.")
                                        raise # Re-raise to be handled in _execute_function
                                    except Exception as e:
                                        logger.exception(f"Error during execution of function {captured_name} (run_id: {ctx.run_id})")
                                        raise # Re-raise to be caught by _execute_function
                                return wrapper

                            self._actions[func_name] = create_wrapper(
                                original_func,
                                func_name,
                                payload_type,
                                return_type,
                                func_expects_pipeline=expects_pipeline_flag,
                            )
                            logger.info(f"Registered function: '{func_name}' from module '{module_name}' with resources: {resources}")
                            discovered_count += 1

            except ImportError:
                logger.error(f"Could not import user module: {module_name}")
            except Exception as e:
                logger.exception(f"Error during discovery in module {module_name}: {e}")

        if discovered_count == 0:
             logger.warning(f"No functions decorated with @worker_function found in specified modules: {self.user_module_names}")
        else:
             logger.info(f"Discovery complete. Found {discovered_count} worker functions.")

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

    def _materialize_assets(self, run_id: str, obj: Any) -> None:
        if isinstance(obj, Asset):
            self._materialize_asset(run_id, obj)
            return
        if isinstance(obj, list):
            for it in obj:
                self._materialize_assets(run_id, it)
            return
        if isinstance(obj, dict):
            for it in obj.values():
                self._materialize_assets(run_id, it)
            return
        fields = getattr(obj, "__struct_fields__", None)
        if fields and isinstance(fields, (tuple, list)):
            for name in fields:
                try:
                    self._materialize_assets(run_id, getattr(obj, name))
                except Exception:
                    continue

    def _materialize_asset(self, run_id: str, asset: Asset) -> None:
        if asset.local_path:
            return
        ref = (asset.ref or "").strip()
        if not ref:
            return

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
            if not asset.tenant_id:
                asset.tenant_id = self.tenant_id
            asset.mime_type = mime
            asset.size_bytes = size
            asset.sha256 = sha256_hex
            return

        # Cozy Hub file ref (tenant scoped) - use orchestrator file API with HEAD+cache.
        base = _require_file_api_base_url()
        token = _require_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"

        head_req = _http_request("HEAD", url, token)
        with urllib.request.urlopen(head_req, timeout=10) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"failed to stat asset ({resp.status})")
            sha256_hex = (resp.headers.get("X-Cozy-SHA256") or "").strip()
            size_hdr = (resp.headers.get("X-Cozy-Size-Bytes") or "").strip()
            mime = (resp.headers.get("X-Cozy-Mime-Type") or "").strip()
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
            get_req = _http_request("GET", url, token)
            with urllib.request.urlopen(get_req, timeout=30) as resp:
                if resp.status < 200 or resp.status >= 300:
                    raise RuntimeError(f"failed to download asset ({resp.status})")
                _size, _sha = self._stream_to_file(resp, cache_path, max_bytes)
                if not size:
                    size = _size
                if not sha256_hex:
                    sha256_hex = _sha

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
        if not asset.tenant_id:
            asset.tenant_id = self.tenant_id
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
            if self.auth_token:
                interceptors.append(_AuthInterceptor(self.auth_token))

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
            # current_models = []
            if self._model_manager:
                vram_models = self._model_manager.get_vram_loaded_models()
                supports_model_loading_flag = True 

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

            function_schemas = []
            for fname, (in_schema, out_schema) in self._function_schemas.items():
                try:
                    function_schemas.append(
                        pb.FunctionSchema(
                            name=fname,
                            input_schema_json=in_schema,
                            output_schema_json=out_schema,
                        )
                    )
                except Exception:
                    continue

            resources = pb.WorkerResources(
                worker_id=self.worker_id,
                deployment_id=self.deployment_id,
                # tenant_id=self.tenant_id,
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
                available_functions=list(self._actions.keys()),
                available_models=vram_models,
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
            # TODO: Implement model unloading logic
            model_id = message.unload_model_cmd.model_id
            logger.warning(f"Received unload_model_cmd for {model_id}, but not yet implemented.")
            result = pb.UnloadModelResult(model_id=model_id, success=False, error_message="Model unloading not implemented")
            self._send_message(pb.WorkerSchedulerMessage(unload_model_result=result))
        elif msg_type == 'interrupt_run_cmd':
            run_id = message.interrupt_run_cmd.run_id
            self._handle_interrupt_request(run_id)
        # Add handling for other message types if needed (e.g., config updates)
        elif msg_type == 'deployment_model_config':
            if self._model_manager:
                logger.info(f"Received DeploymentModelConfig: {message.deployment_model_config.supported_model_ids}")
                self._supported_model_ids_from_scheduler = list(message.deployment_model_config.supported_model_ids)
                self._model_init_done_event.clear() # Clear before starting new init
                model_init_thread = threading.Thread(target=self._process_deployment_config_async_wrapper, daemon=True)
                model_init_thread.start()
            else:
                logger.info("Received DeploymentModelConfig, but no model manager configured. Ignoring.")
                self._model_init_done_event.set() # Signal completion as there's nothing to do
        elif msg_type is None:
             logger.warning("Received empty message from scheduler.")
        else:
            logger.warning(f"Received unhandled message type: {msg_type}")

    def _process_deployment_config_async_wrapper(self) -> None:
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

            loop.run_until_complete(
                self._model_manager.process_supported_models_config(
                    self._supported_model_ids_from_scheduler,
                    self._downloader
                )
            )
            logger.info("Model configuration and downloads (if any) processed.")
        except Exception as e:
            logger.exception(f"Error during model_manager.process_supported_models_config: {e}")
        finally:
            if loop and not loop.is_running() and not loop.is_closed(): # Clean up loop if we created it
                loop.close()
            self._model_init_done_event.set() # Signal completion or failure

    def _handle_load_model_cmd(self, cmd: LoadModelCommand) -> None:
        model_id = cmd.model_id
        logger.info(f"Received LoadModelCommand for: {model_id}")
        success = False; error_msg = ""
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
            except Exception as e: error_msg = f"Exception in mmm.load_model_into_vram for '{model_id}': {e}"; logger.exception(error_msg)
        
        result = pb.LoadModelResult(model_id=model_id, success=success, error_message=error_msg)
        self._send_message(pb.WorkerSchedulerMessage(load_model_result=result))


    def _handle_run_request(self, request: TaskExecutionRequest) -> None:
        """Handle a task execution request from the scheduler."""
        run_id = request.run_id
        function_name = request.function_name
        input_payload = request.input_payload
        required_model_id_for_exec = ""
        timeout_ms = int(getattr(request, "timeout_ms", 0) or 0)
        tenant_id = str(getattr(request, "tenant_id", "") or "") or (self.tenant_id or "")
        user_id = str(getattr(request, "user_id", "") or "")

        if request.required_models and len(request.required_models) > 0:
            required_model_id_for_exec = request.required_models[0]

        logger.info(f"Received Task request: run_id={run_id}, function={function_name}, model='{required_model_id_for_exec or 'None'}'")

        func_wrapper = self._actions.get(function_name)
        if not func_wrapper:
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

        ctx = ActionContext(
            run_id,
            emitter=self._emit_progress_event,
            tenant_id=tenant_id or None,
            user_id=user_id or None,
            timeout_ms=timeout_ms if timeout_ms > 0 else None,
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
            target=self._execute_function,
            args=(ctx, function_name, func_wrapper, input_payload, required_model_id_for_exec),
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

    def _execute_function(
        self,
        ctx: ActionContext,
        function_name: str,
        func_to_execute: Callable[[ActionContext, Optional[Any], bytes], bytes],
        input_payload: bytes,
        required_model_id: str,
    ) -> None:
        """Execute the registered function and send the result/error back."""
        run_id = ctx.run_id
        output_payload: Optional[bytes] = None
        error_type: str = ""
        safe_message: str = ""
        error_message: str = ""  # internal/legacy
        success = False

        # Determine if this function requires GPU and manage worker's GPU state
        func_requires_gpu = False
        resource_req = self._discovered_resources.get(function_name)
        if resource_req:
            func_requires_gpu = resource_req.requires_gpu
            func_expects_pipeline = resource_req.expects_pipeline_arg

        # Variable to track if this specific thread execution set the GPU busy
        this_thread_set_gpu_busy = False
        if func_requires_gpu:
            with self._gpu_busy_lock: # Lock to check and set self._is_gpu_busy atomically
                if not self._is_gpu_busy:
                    self._is_gpu_busy = True
                    this_thread_set_gpu_busy = True
                    logger.info(f"Worker GPU marked as BUSY by task {run_id} ({function_name}).")
                else:
                    logger.warning(f"Task {run_id} ({function_name}) requires GPU, but worker GPU was already marked busy. Proceeding...")

        active_pipeline_instance = None # To hold the pipeline for the user function
        try:
            if ctx.is_canceled():
                raise CanceledError("canceled")
            
            if func_expects_pipeline:
                if not required_model_id and resource_req and resource_req.model_name:
                    required_model_id = str(resource_req.model_name)
                if not required_model_id:
                    raise ValueError(f"Function '{function_name}' expects a pipeline argument, but no model ID was provided.")
                
                if not self._model_manager:
                    raise RuntimeError(f"Function '{function_name}' expects a pipeline argument, but no model manager configured on worker.")
                
                if not self._model_init_done_event.is_set():
                    logger.info(f"Task {run_id} ({function_name}) waiting for initial model setup...")
                    if not self._model_init_done_event.wait(timeout=300.0): # 5 min timeout
                        raise TimeoutError(f"Timeout waiting for model initialization for task {run_id}")
                    logger.info(f"Initial model setup complete. Proceeding for task {run_id}.")
                
                logger.info(f"Task {run_id} ({function_name}) getting active pipeline for model '{required_model_id}'...")
                # get_active_pipeline is async
                active_pipeline_instance = asyncio.run(self._model_manager.get_active_pipeline(required_model_id))
                if not active_pipeline_instance:
                    raise RuntimeError(f"ModelManager failed to provide active pipeline for '{required_model_id}' for task {run_id}.")
                
                logger.info(f"Task {run_id} ({function_name}) obtained pipeline for model '{required_model_id}'.")

            # Execute the function wrapper (which handles deserialization/serialization)
            output_payload = func_to_execute(ctx, active_pipeline_instance, input_payload)
            # Check for cancellation *during* execution (func should check ctx.is_canceled)
            if ctx.is_canceled():
                raise CanceledError("canceled")

            if output_payload is not None and self.max_output_bytes > 0:
                if len(output_payload) > self.max_output_bytes:
                    raise ValueError(f"Output payload too large: {len(output_payload)} bytes (max {self.max_output_bytes})")

            success = True
            logger.info(f"Task {run_id} completed successfully.")

        except Exception as e:
            error_type, retryable, safe_message, error_message = self._map_exception(e)
            if error_type == "canceled":
                logger.warning(f"Task {run_id} was canceled: {safe_message}")
            elif retryable:
                logger.error(f"Task {run_id} ({function_name}) retryable failure: {safe_message}")
            else:
                logger.exception(f"Error executing function for run_id={run_id}: {safe_message}")
            success = False
        finally:
            # Release the GPU if this thread set it busy
            if this_thread_set_gpu_busy:
                with self._gpu_busy_lock: # Lock to set self._is_gpu_busy
                    self._is_gpu_busy = False
                logger.info(f"Worker GPU marked as NOT BUSY by task {run_id} ({function_name}).")

            # Always send a result back, regardless of success, failure, or cancellation
            self._send_task_result(run_id, success, output_payload, error_type, retryable if not success else False, safe_message, error_message)
            # Remove from active tasks *after* sending result
            with self._active_tasks_lock:
                if run_id in self._active_tasks:
                    del self._active_tasks[run_id]
                resource_req = self._discovered_resources.get(function_name)
                func_limit = resource_req.max_concurrency if resource_req and resource_req.max_concurrency else 0
                if func_limit > 0:
                    current = self._active_function_counts.get(function_name, 0) - 1
                    if current <= 0:
                        self._active_function_counts.pop(function_name, None)
                    else:
                        self._active_function_counts[function_name] = current
                # else: # Might have been removed by stop() already
                     # logger.warning(f"Task {run_id} not found in active tasks during cleanup.")

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
