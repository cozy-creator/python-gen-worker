from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import base64
import shutil
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from .api.errors import AuthError
from .api.types import Asset, Tensors
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


def _infer_tensors_format(ref_or_path: str) -> str:
    leaf = str(ref_or_path or "").strip().lower()
    if leaf.endswith(".safetensors"):
        return "safetensors"
    if leaf.endswith(".bin"):
        return "bin"
    if leaf.endswith(".pt"):
        return "pt"
    if leaf.endswith(".pth"):
        return "pth"
    if leaf.endswith(".ckpt"):
        return "ckpt"
    return "unknown"


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
    token = os.getenv("WORKER_CAPABILITY_TOKEN", "").strip()
    if not token:
        token = os.getenv("FILE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError("WORKER_CAPABILITY_TOKEN (or FILE_API_TOKEN) is required for file operations")
    return token


def _parse_owner_repo(value: str) -> tuple[str, str]:
    raw = str(value or "").strip().strip("/")
    if "/" not in raw:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    owner, repo = raw.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        raise ValueError("destination_repo must be in '<owner>/<repo>' format")
    return owner, repo


def _decode_unverified_jwt_claims(token: str) -> Dict[str, Any]:
    raw = str(token or "").strip()
    if raw.count(".") < 2:
        return {}
    try:
        parts = raw.split(".")
        payload_b64 = parts[1]
        pad = "=" * ((4 - (len(payload_b64) % 4)) % 4)
        payload = base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii"))
        parsed = json.loads(payload.decode("utf-8"))
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _normalize_repo_name(value: str) -> str:
    return str(value or "").strip().strip("/").lower()


def _assert_token_repo_scope_matches_destination(token: str, owner: str, repo: str) -> None:
    claims = _decode_unverified_jwt_claims(token)
    if not claims:
        raise ValueError("worker_capability_token must be a structured JWT")

    destination_owner = _normalize_repo_name(owner)
    destination_repo = _normalize_repo_name(repo)
    cap_kind = _normalize_repo_name(str(claims.get("cap_kind") or ""))
    if cap_kind != "worker_capability":
        raise ValueError("worker_capability_token must have cap_kind=worker_capability")
    actions = [str(a or "").strip() for a in list(claims.get("actions") or [])]
    if "revision:create" not in actions:
        raise ValueError("worker_capability_token missing required action 'revision:create'")
    claimed_owner = _normalize_repo_name(str(claims.get("owner") or claims.get("org") or ""))
    claimed_repo = _normalize_repo_name(str(claims.get("repo") or ""))

    if claimed_repo:
        try:
            claimed_repo_owner, claimed_repo_name = _parse_owner_repo(claimed_repo)
        except ValueError as e:
            raise ValueError("worker_capability_token has invalid repo scope claim") from e
        if _normalize_repo_name(claimed_repo_owner) != destination_owner or _normalize_repo_name(claimed_repo_name) != destination_repo:
            raise ValueError("destination_repo does not match worker_capability_token repo scope")
        if claimed_owner and claimed_owner != destination_owner:
            raise ValueError("destination_repo owner does not match worker_capability_token owner scope")
        return

    if claimed_owner and claimed_owner != destination_owner:
        raise ValueError("destination_repo owner does not match worker_capability_token owner scope")


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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class _RequestOutputStream:
    """Buffered output writer with finalize() -> Asset/Tensors.

    Current implementation writes chunks to a local temp file and delegates final
    persistence to RequestContext save_* methods. This keeps write() memory-bounded
    and enables future backend-native append/compose implementations behind the
    same interface.
    """

    def __init__(
        self,
        *,
        ctx: "RequestContext",
        ref: str,
        kind: str,  # "asset" | "checkpoint"
        format: Optional[str] = None,
        create: bool = False,
    ) -> None:
        self._ctx = ctx
        self._ref = _normalize_output_ref(ref)
        self._kind = str(kind or "asset").strip().lower()
        self._format = str(format or "").strip() or None
        self._create = bool(create)
        suffix = Path(self._ref).suffix or ".bin"
        fd, tmp = tempfile.mkstemp(prefix=f"gw-out-{ctx.request_id}-", suffix=suffix)
        os.close(fd)
        self._tmp_path = tmp
        self._fh = open(self._tmp_path, "wb")
        self._bytes_written = 0
        self._finalized = False
        self._result: Any = None

    @property
    def bytes_written(self) -> int:
        return int(self._bytes_written)

    @property
    def ref(self) -> str:
        return self._ref

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._finalized:
            raise RuntimeError("output stream already finalized")
        if isinstance(data, memoryview):
            data = data.tobytes()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("write expects bytes-like input")
        b = bytes(data)
        if not b:
            return 0
        n = self._fh.write(b)
        self._bytes_written += int(n)
        return int(n)

    def flush(self) -> None:
        if self._finalized:
            return
        self._fh.flush()

    def finalize(self) -> Any:
        if self._finalized:
            return self._result
        self._fh.flush()
        self._fh.close()
        try:
            if self._kind == "checkpoint":
                self._result = self._ctx.save_checkpoint(self._ref, self._tmp_path, format=self._format)
            else:
                if self._create:
                    self._result = self._ctx.save_file_create(self._ref, self._tmp_path)
                else:
                    self._result = self._ctx.save_file(self._ref, self._tmp_path)
            self._finalized = True
            return self._result
        finally:
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass

    def close(self) -> None:
        if self._finalized:
            return
        try:
            self._fh.close()
        finally:
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass

    def __enter__(self) -> "_RequestOutputStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            self.finalize()
        else:
            self.close()
        return False


class RequestContext:
    """Context object passed to action functions, allowing cancellation."""

    def __init__(
        self,
        request_id: str,
        run_id: Optional[str] = None,
        emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        owner: Optional[str] = None,
        invoker_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        file_api_base_url: Optional[str] = None,
        worker_capability_token: Optional[str] = None,
        file_api_token: Optional[str] = None,
        materialized_input_urls: Optional[Dict[str, str]] = None,
        local_output_dir: Optional[str] = None,
        resolved_cozy_models_by_id: Optional[Dict[str, Any]] = None,
        required_models: Optional[List[str]] = None,
        runtime_batching_config: Optional[Dict[str, Any]] = None,
        execution_hints: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
        child_request_id: Optional[str] = None,
        item_id: Optional[str] = None,
        item_index: Optional[int] = None,
        item_span: Optional[Dict[str, int]] = None,
    ) -> None:
        self._request_id = str(request_id or "").strip()
        self._run_id = str(run_id or "").strip() or None
        self._owner = owner
        self._invoker_id = invoker_id
        self._timeout_ms = timeout_ms
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._worker_capability_token = (worker_capability_token or file_api_token or "").strip() or None
        # Legacy private name retained for compatibility with older call paths.
        self._file_api_token = self._worker_capability_token
        self._materialized_input_urls = dict(materialized_input_urls or {})
        self._local_output_dir = (local_output_dir or "").strip() or None
        self._resolved_cozy_models_by_id = resolved_cozy_models_by_id
        self._required_models = list(required_models or [])
        self._runtime_batching_config = dict(runtime_batching_config or {})
        self._execution_hints = dict(execution_hints or {})
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
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def workspace_scope_id(self) -> str:
        return self._run_id or self._request_id

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

    def _get_worker_capability_token(self) -> str:
        if self._worker_capability_token:
            return self._worker_capability_token
        return _require_file_api_token()

    def _get_file_api_token(self) -> str:
        # Legacy compatibility alias.
        return self._get_worker_capability_token()

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
    def execution_hints(self) -> Dict[str, Any]:
        return dict(self._execution_hints)

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
            "run_id": self._run_id,
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
                raise AuthError(f"file save unauthorized ({code}): check worker_capability_token validity") from e
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
        ref = _normalize_output_ref(ref)
        src = str(local_path or "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        size = int(os.path.getsize(src))
        max_bytes = int(os.getenv("WORKER_MAX_OUTPUT_FILE_BYTES", str(200 * 1024 * 1024)))
        if max_bytes > 0 and size > max_bytes:
            raise ValueError("output file too large")

        local_out = self._resolve_local_output_path(ref)
        if local_out:
            dst = Path(local_out)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            sha = _sha256_file(str(dst))
            return Asset(
                ref=ref,
                owner=self.owner,
                local_path=str(dst),
                mime_type=None,
                size_bytes=size,
                sha256=sha,
            )

        base = self._get_file_api_base_url()
        token = self._get_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }
        owner = (self.owner or "").strip()
        if owner:
            headers["X-Cozy-Owner"] = owner
        t0 = time.monotonic()
        try:
            with open(src, "rb") as fin:
                resp = requests.put(url, headers=headers, data=fin, timeout=30)
            code = int(resp.status_code)
            if code in (401, 403):
                raise AuthError(f"file save unauthorized ({code}): check worker_capability_token validity")
            if code < 200 or code >= 300:
                raise RuntimeError(f"file save failed ({code})")
            try:
                meta = resp.json()
            except Exception:
                meta = {}
        except requests.RequestException as e:
            raise RuntimeError("file save failed (network_error)") from e
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
            size_bytes=int(meta.get("size_bytes") or 0) or size,
            sha256=str(meta.get("sha256") or "") or _sha256_file(src),
        )

    def save_checkpoint(self, ref: str, local_path: str, format: Optional[str] = None) -> Tensors:
        """Save checkpoint/model-weight bytes and return a first-class tensor artifact."""
        asset = self.save_file(ref, local_path)
        fmt = str(format or "").strip() or _infer_tensors_format(ref or local_path)
        return Tensors(
            ref=asset.ref,
            owner=asset.owner,
            local_path=asset.local_path,
            format=fmt,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
            download_token=asset.download_token,
        )

    def save_checkpoint_bytes(self, ref: str, data: bytes, format: Optional[str] = None) -> Tensors:
        """Save in-memory checkpoint/model-weight bytes."""
        asset = self.save_bytes(ref, data)
        fmt = str(format or "").strip() or _infer_tensors_format(ref)
        return Tensors(
            ref=asset.ref,
            owner=asset.owner,
            local_path=asset.local_path,
            format=fmt,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
            download_token=asset.download_token,
        )

    def open_output_stream(self, ref: str, *, create: bool = False) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to an Asset."""
        return _RequestOutputStream(ctx=self, ref=ref, kind="asset", create=create)

    def open_checkpoint_stream(
        self,
        ref: str,
        *,
        format: Optional[str] = None,
    ) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to Tensors."""
        return _RequestOutputStream(ctx=self, ref=ref, kind="checkpoint", format=format)

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
                raise AuthError(f"file save unauthorized ({code}): check worker_capability_token validity") from e
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
        ref = _normalize_output_ref(ref)
        src = str(local_path or "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        size = int(os.path.getsize(src))
        max_bytes = int(os.getenv("WORKER_MAX_OUTPUT_FILE_BYTES", str(200 * 1024 * 1024)))
        if max_bytes > 0 and size > max_bytes:
            raise ValueError("output file too large")

        local_out = self._resolve_local_output_path(ref)
        if local_out:
            dst = Path(local_out)
            if dst.exists():
                raise RuntimeError("output path already exists")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
            sha = _sha256_file(str(dst))
            return Asset(
                ref=ref,
                owner=self.owner,
                local_path=str(dst),
                mime_type=None,
                size_bytes=size,
                sha256=sha,
            )

        base = self._get_file_api_base_url()
        token = self._get_file_api_token()
        url = f"{base}/api/v1/file/{_encode_ref_for_url(ref)}"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }
        owner = (self.owner or "").strip()
        if owner:
            headers["X-Cozy-Owner"] = owner
        t0 = time.monotonic()
        try:
            with open(src, "rb") as fin:
                resp = requests.post(url, headers=headers, data=fin, timeout=30)
            code = int(resp.status_code)
            if code in (401, 403):
                raise AuthError(f"file save unauthorized ({code}): check worker_capability_token validity")
            if code == 409:
                raise RuntimeError("output path already exists")
            if code < 200 or code >= 300:
                raise RuntimeError(f"file save failed ({code})")
            try:
                meta = resp.json()
            except Exception:
                meta = {}
        except requests.RequestException as e:
            raise RuntimeError("file save failed (network_error)") from e
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
            size_bytes=int(meta.get("size_bytes") or 0) or size,
            sha256=str(meta.get("sha256") or "") or _sha256_file(src),
        )

    def save_bytes_overwrite(self, ref: str, data: bytes) -> Asset:
        # Back-compat alias: overwrite is the default save_bytes behavior.
        return self.save_bytes(ref, data)

    def save_file_overwrite(self, ref: str, local_path: str) -> Asset:
        return self.save_file(ref, local_path)

    def publish_repo_revision(
        self,
        *,
        destination_repo: str,
        artifact_refs: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_if_missing: bool = True,
    ) -> Dict[str, Any]:
        """Publish conversion lineage to Tensorhub using public HTTP APIs only.

        Uses worker capability auth and never touches DB/internal-only paths.
        """
        owner, repo = _parse_owner_repo(destination_repo)
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or self._file_api_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo)
        logger.info(
            "worker_publish_attempt request_id=%s run_id=%s owner=%s repo=%s",
            self.request_id,
            self.run_id or "",
            owner,
            repo,
        )
        if not base or not token:
            # Local/dev mode: no remote publish channel configured.
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "owner": owner, "repo": repo}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Cozy-Owner": owner,
        }

        def _request_json(method: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            url = f"{base}{path}"
            resp = requests.request(method=method, url=url, headers=headers, data=json.dumps(payload), timeout=30)
            code = int(resp.status_code)
            if code in (401, 403):
                raise AuthError(f"repo publish unauthorized ({code}): check worker_capability_token validity")
            if code < 200 or code >= 300:
                # Create may be idempotent and already exists.
                if path == "/api/v1/repos" and code in (400, 409):
                    return {"ok": True, "already_exists": True}
                raise RuntimeError(f"repo publish request failed ({code}) {path}: {resp.text[:256]}")
            try:
                parsed = resp.json()
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"ok": True}

        if create_if_missing:
            _request_json("POST", "/api/v1/repos", {"repo_name": repo})

        start = _request_json(
            "POST",
            f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/runs/start",
            {"kind": "conversion", "input_versions": []},
        )
        run_id = str(start.get("run_id") or "").strip()
        if run_id == "":
            raise RuntimeError("repo publish failed: missing run_id from start response")

        metrics = dict(metadata or {})
        refs = [str(r or "").strip() for r in list(artifact_refs or []) if str(r or "").strip()]
        if refs:
            metrics["artifact_refs"] = refs
        _request_json(
            "POST",
            f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/runs/{urllib.parse.quote(run_id, safe='')}/finalize",
            {"status": "succeeded", "metrics_json": metrics, "cost_json": {}, "output_versions": []},
        )

        logger.info(
            "worker_publish_succeeded request_id=%s run_id=%s owner=%s repo=%s published_run_id=%s",
            self.request_id,
            self.run_id or "",
            owner,
            repo,
            run_id,
        )

        return {"ok": True, "owner": owner, "repo": repo, "run_id": run_id}

    def read_repo_metadata(self, *, destination_repo: str) -> Dict[str, Any]:
        """Read repo-level metadata from Tensorhub public HTTP API."""
        owner, repo = _parse_owner_repo(destination_repo)
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or self._file_api_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo)
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "exists": False, "metadata": {}}

        url = f"{base}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/metadata"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Cozy-Owner": owner,
        }
        resp = requests.get(url, headers=headers, timeout=30)
        code = int(resp.status_code)
        if code in (401, 403):
            raise AuthError(f"repo metadata read unauthorized ({code}): check worker_capability_token validity")
        if code == 404:
            return {"ok": True, "exists": False, "metadata": {}}
        if code < 200 or code >= 300:
            raise RuntimeError(f"repo metadata read failed ({code}): {resp.text[:256]}")

        try:
            parsed = resp.json()
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        metadata = parsed.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        exists = bool(parsed.get("exists", True))
        return {"ok": True, "exists": exists, "metadata": metadata}

    def write_repo_metadata(self, *, destination_repo: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Write repo-level metadata via Tensorhub public HTTP API."""
        owner, repo = _parse_owner_repo(destination_repo)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or self._file_api_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo)
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "owner": owner, "repo": repo}

        url = f"{base}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/metadata"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Cozy-Owner": owner,
        }
        resp = requests.put(url, headers=headers, data=json.dumps({"metadata": metadata}), timeout=30)
        code = int(resp.status_code)
        if code in (401, 403):
            raise AuthError(f"repo metadata write unauthorized ({code}): check worker_capability_token validity")
        if code < 200 or code >= 300:
            raise RuntimeError(f"repo metadata write failed ({code}): {resp.text[:256]}")

        try:
            parsed = resp.json()
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        returned = parsed.get("metadata")
        if not isinstance(returned, dict):
            returned = metadata
        return {"ok": True, "owner": owner, "repo": repo, "metadata": returned}
