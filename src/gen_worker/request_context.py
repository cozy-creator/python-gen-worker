from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from .api.errors import AuthError
from .api.types import Asset
from .models.refs import parse_model_ref

logger = logging.getLogger(__name__)


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


def _default_output_prefix(request_id: str) -> str:
    return f"runs/{request_id}/outputs/"


def _normalize_output_ref(ref: str) -> str:
    out = str(ref or "").strip()
    if not out:
        raise ValueError("invalid ref")
    lower = out.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        raise ValueError("output ref must be a logical file ref, not a URL")
    return out.lstrip("/")


def _require_file_api_base_url() -> str:
    base = os.getenv("FILE_API_BASE_URL", "").strip()
    if not base:
        base = os.getenv("ORCHESTRATOR_HTTP_URL", "").strip()
    if not base:
        base = os.getenv("TENSORHUB_URL", "").strip()
    if not base:
        raise RuntimeError("FILE_API_BASE_URL is required for file operations")
    return base.rstrip("/")


def _require_file_api_token() -> str:
    token = os.getenv("FILE_API_TOKEN", "").strip()
    if not token:
        token = os.getenv("TENSORHUB_TOKEN", "").strip()
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


class RequestContext:
    """Context object passed to action functions, allowing cancellation."""

    def __init__(
        self,
        request_id: str,
        emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        owner: Optional[str] = None,
        invoker_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        file_api_base_url: Optional[str] = None,
        file_api_token: Optional[str] = None,
        materialized_input_urls: Optional[Dict[str, str]] = None,
        local_output_dir: Optional[str] = None,
        resolved_cozy_models_by_id: Optional[Dict[str, Any]] = None,
        required_models: Optional[List[str]] = None,
        runtime_batching_config: Optional[Dict[str, Any]] = None,
        stage_execution_hints: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
        child_request_id: Optional[str] = None,
        item_id: Optional[str] = None,
        item_index: Optional[int] = None,
        item_span: Optional[Dict[str, int]] = None,
    ) -> None:
        self._request_id = str(request_id or "").strip()
        self._owner = owner
        self._invoker_id = invoker_id
        self._timeout_ms = timeout_ms
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._file_api_token = (file_api_token or "").strip() or None
        self._materialized_input_urls = dict(materialized_input_urls or {})
        self._local_output_dir = (local_output_dir or "").strip() or None
        self._resolved_cozy_models_by_id = resolved_cozy_models_by_id
        self._required_models = list(required_models or [])
        self._runtime_batching_config = dict(runtime_batching_config or {})
        self._stage_execution_hints = dict(stage_execution_hints or {})
        self._parent_request_id = str(parent_request_id or "").strip() or None
        self._child_request_id = str(child_request_id or "").strip() or None
        self._item_id = str(item_id or "").strip() or None
        self._item_index = int(item_index) if item_index is not None else None
        self._item_span = dict(item_span or {})
        self._started_at = time.time()
        self._deadline: Optional[float] = None
        if timeout_ms is not None and timeout_ms > 0:
            self._deadline = self._started_at + (timeout_ms / 1000.0)
        self._canceled = False
        self._cancel_event = threading.Event()
        self._emitter = emitter

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def owner(self) -> Optional[str]:
        return self._owner

    @property
    def invoker_id(self) -> Optional[str]:
        return self._invoker_id

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

        When local_output_dir is set, RequestContext.save_* will write outputs to disk
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

    def _materialized_input_url_for_ref(self, ref: str) -> Optional[str]:
        raw = (ref or "").strip().lstrip("/")
        if not raw:
            return None
        out = str(self._materialized_input_urls.get(raw) or "").strip()
        if out:
            return out
        return None

    @property
    def resolved_cozy_models_by_id(self) -> Optional[Dict[str, Any]]:
        return self._resolved_cozy_models_by_id

    @property
    def required_models(self) -> List[str]:
        return list(self._required_models)

    @property
    def runtime_batching_config(self) -> Dict[str, Any]:
        return dict(self._runtime_batching_config)

    @property
    def stage_execution_hints(self) -> Dict[str, Any]:
        return dict(self._stage_execution_hints)

    @property
    def parent_request_id(self) -> Optional[str]:
        return self._parent_request_id

    @property
    def child_request_id(self) -> Optional[str]:
        return self._child_request_id

    @property
    def item_id(self) -> Optional[str]:
        return self._item_id

    @property
    def item_index(self) -> Optional[int]:
        return self._item_index

    @property
    def item_span(self) -> Dict[str, int]:
        return dict(self._item_span)

    def partition_context(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "request_id": self._request_id,
            "parent_request_id": self._parent_request_id,
            "child_request_id": self._child_request_id,
            "item_id": self._item_id,
            "item_index": self._item_index,
        }
        if self._item_span:
            out["item_span"] = dict(self._item_span)
        return out

    def item_output_ref(self, filename: str) -> str:
        """Build a canonical output ref scoped to this request item."""
        leaf = str(filename or "").strip().lstrip("/")
        if not leaf:
            raise ValueError("filename is required")
        item_key = self._item_id
        if not item_key and self._item_index is not None:
            item_key = f"item-{self._item_index:06d}"
        if not item_key:
            item_key = "item-000000"
        return f"runs/{self._request_id}/outputs/items/{item_key}/{leaf}"

    def preferred_batch_size(self, default: int = 1) -> int:
        cfg = self._runtime_batching_config
        target = int(cfg.get("batch_size_target", default) or default)
        mn = int(cfg.get("batch_size_min", 1) or 1)
        mx = int(cfg.get("batch_size_max", max(mn, target)) or max(mn, target))
        if mx < mn:
            mx = mn
        if target < mn:
            target = mn
        if target > mx:
            target = mx
        return max(1, target)

    def prefetch_depth(self, default: int = 1) -> int:
        cfg = self._runtime_batching_config
        v = int(cfg.get("prefetch_depth", default) or default)
        return max(1, v)

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
            logger.info(f"Action {self.request_id} marked for cancellation.")

    def done(self) -> threading.Event:
        """Returns an event that is set when the action is cancelled."""
        return self._cancel_event

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit a progress/event payload (best-effort)."""
        if not self._emitter:
            logger.debug(f"emit({event_type}) dropped: no emitter configured")
            return
        event = {
            "request_id": self._request_id,
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
        }
        self._emitter(event)

    def progress(self, progress: float, stage: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"progress": progress}
        if stage is not None:
            payload["stage"] = stage
        self.emit("request.progress", payload)

    def log(self, message: str, level: str = "info") -> None:
        self.emit("request.log", {"message": message, "level": level})

    def save_bytes(self, ref: str, data: bytes) -> Asset:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes expects bytes")
        data = bytes(data)
        max_bytes = int(os.getenv("WORKER_MAX_OUTPUT_FILE_BYTES", str(200 * 1024 * 1024)))
        if max_bytes > 0 and len(data) > max_bytes:
            raise ValueError("output file too large")
        ref = _normalize_output_ref(ref)

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
        ref = _normalize_output_ref(ref)

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
