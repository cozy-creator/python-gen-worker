from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import base64
import re
import shutil
import socket
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import requests
from blake3 import blake3

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from .api.errors import AuthError, OutputTooLargeError
from .api.types import Asset, Tensors
from .models.refs import parse_model_ref

logger = logging.getLogger(__name__)

_PUBLIC_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_STALE_MIRROR_CLAIM_ERROR_CODES = {"source_version_not_found", "source_variants_not_found"}
_MAX_OUTPUT_FILE_BYTES = 20 * 1024 * 1024 * 1024  # 20 GiB hard cap per file.
_FILE_API_HTTP_TIMEOUT_S = 60
_FILE_API_STREAM_CHUNK_TIMEOUT_S = 120
_FILE_API_STREAM_FINALIZE_TIMEOUT_S = 600
_FILE_API_STREAM_REPLAY_TIMEOUT_S = 600
_FILE_API_STREAM_ABORT_TIMEOUT_S = 15


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
    return f"jobs/{request_id}/outputs/"


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


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _require_file_api_base_url() -> str:
    base = os.getenv("FILE_API_BASE_URL", "").strip()
    if not base:
        base = os.getenv("ORCHESTRATOR_HTTP_URL", "").strip()
    if not base:
        base = os.getenv("TENSORHUB_URL", "").strip()
    if not base:
        raise RuntimeError("FILE_API_BASE_URL is required for file operations")
    return base.rstrip("/")


def _require_worker_capability_token() -> str:
    token = os.getenv("WORKER_CAPABILITY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("WORKER_CAPABILITY_TOKEN is required for file operations")
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


def _parse_owner_repo_with_optional_tag(value: str) -> tuple[str, str, str]:
    raw = str(value or "").strip().strip("/")
    tag = ""
    if ":" in raw:
        raw, tag = raw.rsplit(":", 1)
        tag = str(tag or "").strip().lower()
    owner, repo = _parse_owner_repo(raw)
    return owner, repo, tag


def _normalize_destination_repo_tags(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in list(values or []):
        tag = str(item or "").strip().lower()
        if not tag:
            continue
        if not _PUBLIC_TAG_RE.match(tag):
            raise ValueError("destination_repo_tags contains an invalid tag")
        if tag == "latest":
            raise ValueError("destination_repo_tags must not include latest")
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    out.sort()
    return out


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


def _error_code_from_exception(exc: Exception, *, fallback: str = "unknown") -> str:
    raw = str(exc or "").strip()
    if not raw:
        return fallback
    parts = [p.strip().lower() for p in raw.split(":") if p and p.strip()]
    if not parts:
        return fallback
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def _utc_timestamp_rfc3339() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _enforce_output_file_size_limit(size_bytes: int) -> None:
    size = int(size_bytes)
    if size < 0:
        raise ValueError("size_bytes must be non-negative")
    if size > _MAX_OUTPUT_FILE_BYTES:
        raise OutputTooLargeError(size_bytes=size, max_bytes=_MAX_OUTPUT_FILE_BYTES)


def _assert_token_repo_scope_matches_destination(
    token: str,
    owner: str,
    repo: str,
    *,
    required_permissions: Optional[List[str]] = None,
) -> None:
    claims = _decode_unverified_jwt_claims(token)
    if not claims:
        raise ValueError("worker_capability_token must be a structured JWT")

    destination_owner = _normalize_repo_name(owner)
    destination_repo = _normalize_repo_name(repo)
    cap_kind = _normalize_repo_name(str(claims.get("cap_kind") or ""))
    if cap_kind != "worker_capability":
        raise ValueError("worker_capability_token must have cap_kind=worker_capability")
    if required_permissions is None:
        needed = ["repo-version:create"]
    else:
        needed = [str(p or "").strip() for p in list(required_permissions) if str(p or "").strip()]
    repos_read = [str(v or "").strip() for v in list(claims.get("tensor_repos_read") or [])]
    repos_update_legacy = [str(v or "").strip() for v in list(claims.get("tensor_repos_update") or [])]
    repos_version_create = [str(v or "").strip() for v in list(claims.get("tensor_repos_version_create") or [])]
    repos_variant_create = [str(v or "").strip() for v in list(claims.get("tensor_repos_variant_create") or [])]
    if not repos_version_create:
        repos_version_create = list(repos_update_legacy)
    if not repos_variant_create:
        repos_variant_create = list(repos_update_legacy)
    create_claim = claims.get("tensor_repo_create")
    create_policy = create_claim if isinstance(create_claim, dict) else {}
    create_owner = _normalize_repo_name(str(create_policy.get("owner") or ""))
    create_allowed_names = [_normalize_repo_name(str(v or "")) for v in list(create_policy.get("allowed_names") or [])]
    create_allow_any_name = bool(create_policy.get("allow_any_name"))

    def _repo_match(values: List[str]) -> bool:
        for raw in values:
            try:
                scoped_owner, scoped_repo = _parse_owner_repo(_normalize_repo_name(raw))
            except ValueError:
                continue
            if _normalize_repo_name(scoped_owner) == destination_owner and _normalize_repo_name(scoped_repo) == destination_repo:
                return True
        return False

    for permission in needed:
        if permission == "tensor-repo:read":
            if _repo_match(repos_read) or _repo_match(repos_version_create) or _repo_match(repos_variant_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token read scope")
        if permission == "repo-version:create":
            if _repo_match(repos_version_create) or _repo_match(repos_variant_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token repo-version:create scope")
        if permission == "repo-variant:create":
            if _repo_match(repos_variant_create) or _repo_match(repos_version_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token repo-variant:create scope")
        if permission == "tensor-repo:update":
            # Legacy alias.
            if _repo_match(repos_version_create) or _repo_match(repos_variant_create):
                continue
            raise ValueError("destination_repo does not match worker_capability_token update scope")
        if permission == "tensor-repo:create":
            if create_owner != destination_owner:
                raise ValueError("destination_repo owner does not match worker_capability_token create scope")
            if create_allow_any_name:
                continue
            if destination_repo in create_allowed_names:
                continue
            raise ValueError("destination_repo is not in worker_capability_token create allow-list")
        raise ValueError(f"unsupported required permission '{permission}'")


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
    """Chunk-writable output writer with finalize() -> Asset/Tensors.

    All writes are buffered to a temp file. On finalize(), the file is hashed
    with BLAKE3, then uploaded via presigned S3 multipart URLs obtained from
    TensorHub.
    """

    def __init__(
        self,
        *,
        ctx: "RequestContext",
        ref: str,
        kind: str,  # "asset" | "checkpoint"
        format: Optional[str] = None,
        create: bool = False,
        expected_size_bytes: Optional[int] = None,
    ) -> None:
        from .presigned_upload import blake3_hash_file, presigned_upload_file

        self._ctx = ctx
        self._ref = _normalize_output_ref(ref)
        self._kind = str(kind or "asset").strip().lower()
        self._format = str(format or "").strip() or None
        self._create = bool(create)
        self._expected_size_bytes = int(expected_size_bytes or 0)
        if self._expected_size_bytes < 0:
            self._expected_size_bytes = 0
        if self._expected_size_bytes > 0:
            _enforce_output_file_size_limit(self._expected_size_bytes)
        self._stream_remote = bool(self._ctx._should_stream_output_to_file_api(self._ref))
        self._sha = hashlib.sha256()
        self._blake3_hasher = blake3()
        self._bytes_written = 0
        self._bytes_uploaded = 0
        self._chunks_written = 0
        self._chunks_uploaded = 0
        self._stream_error_class: Optional[str] = None
        self._progress_lock = threading.Lock()
        self._stream_mode = "presigned" if self._stream_remote else "local_fallback"
        self._started_mono = time.monotonic()
        try:
            interval = float(os.getenv("WORKER_STREAM_PROGRESS_INTERVAL_S", "0.20") or "0.20")
        except Exception:
            interval = 0.20
        self._progress_interval_s = max(0.0, interval)
        self._last_progress_emit_mono = self._started_mono
        self._last_progress_mono = self._started_mono
        self._last_progress_uploaded = 0
        self._retry_attempts = max(1, int(os.getenv("WORKER_STREAM_UPLOAD_RETRY_ATTEMPTS", "5") or "5"))
        self._retry_backoff_ms = max(0, int(os.getenv("WORKER_STREAM_UPLOAD_RETRY_BACKOFF_MS", "500") or "500"))
        self._session_id: Optional[str] = None
        self._uploader_meta: Dict[str, Any] = {}
        self._repo_job_scope = self._ctx._repo_job_upload_scope() if self._kind == "checkpoint" else None
        self._finalized = False
        self._result: Any = None
        self._abort_remote: bool = False

        # Always buffer to temp file.
        suffix = Path(self._ref).suffix or ".bin"
        prefix = f"gw-out-{ctx.request_id}-"
        fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        self._tmp_path = tmp
        self._fh: Optional[Any] = open(self._tmp_path, "wb")

    @property
    def bytes_written(self) -> int:
        return int(self._bytes_written)

    @property
    def bytes_uploaded(self) -> int:
        return int(self._bytes_uploaded) if self._bytes_uploaded > 0 else int(self._bytes_written)

    @property
    def stream_mode(self) -> str:
        return self._stream_mode

    @property
    def elapsed_s(self) -> float:
        return float(max(time.monotonic() - self._started_mono, 0.0))

    @property
    def average_upload_bps(self) -> float:
        elapsed = max(self.elapsed_s, 1e-6)
        return float(self.bytes_uploaded) / elapsed

    @property
    def ref(self) -> str:
        return self._ref

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._finalized:
            raise RuntimeError("output stream already finalized")
        if self._ctx.is_canceled():
            self.close()
            raise InterruptedError("canceled")
        if isinstance(data, memoryview):
            data = data.tobytes()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("write expects bytes-like input")
        b = bytes(data)
        if not b:
            return 0
        _enforce_output_file_size_limit(self._bytes_written + len(b))
        assert self._fh is not None
        n = self._fh.write(b)
        self._sha.update(b[:n])
        self._blake3_hasher.update(b[:n])
        self._bytes_written += int(n)
        if n > 0:
            self._chunks_written += 1
        self._maybe_emit_progress(stage="stream_write")
        return int(n)

    def flush(self) -> None:
        if self._finalized:
            return
        if self._fh is not None:
            self._fh.flush()

    def finalize(self) -> Any:
        if self._finalized:
            return self._result
        if self._ctx.is_canceled():
            self.close()
            raise InterruptedError("canceled")

        assert self._fh is not None
        assert self._tmp_path is not None
        self._fh.flush()
        self._fh.close()
        self._fh = None

        try:
            if self._stream_remote:
                finalize_t0 = time.monotonic()
                self._result = self._finalize_presigned_upload()
                self._finalized = True
                self._maybe_emit_progress(
                    stage="stream_finalized",
                    force=True,
                    extra={"finalize_elapsed_s": float(max(time.monotonic() - finalize_t0, 0.0))},
                )
                return self._result
            else:
                raw: Asset | Tensors
                if self._kind == "checkpoint":
                    raw = self._ctx.save_checkpoint(self._ref, self._tmp_path, format=self._format)
                else:
                    if self._create:
                        raw = self._ctx.save_file_create(self._ref, self._tmp_path)
                    else:
                        raw = self._ctx.save_file(self._ref, self._tmp_path)
                self._result = self._with_stream_mode(raw)
                self._finalized = True
                self._maybe_emit_progress(stage="stream_finalized", force=True)
                return self._result
        finally:
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass

    def close(self) -> None:
        if self._finalized:
            return
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        if self._tmp_path:
            try:
                os.remove(self._tmp_path)
            except Exception:
                pass
        self._maybe_emit_progress(stage="stream_aborted", force=True)
        self._finalized = True

    def _signal_remote_done(self) -> None:
        pass

    def _abort_due_to_cancel(self) -> None:
        if self._abort_remote:
            return
        self._abort_remote = True
        self._signal_remote_done()
        self._maybe_emit_progress(stage="stream_canceled", force=True)

    def _with_stream_mode(self, value: Any) -> Any:
        if isinstance(value, Asset):
            return Asset(
                ref=value.ref,
                owner=value.owner,
                local_path=value.local_path,
                mime_type=value.mime_type,
                size_bytes=value.size_bytes,
                sha256=value.sha256,
                download_token=value.download_token,
                stream_mode=self.stream_mode,
            )
        if isinstance(value, Tensors):
            return Tensors(
                ref=value.ref,
                owner=value.owner,
                local_path=value.local_path,
                format=value.format,
                size_bytes=value.size_bytes,
                sha256=value.sha256,
                blake3=value.blake3,
                blob_digest=value.blob_digest,
                blob_domain=value.blob_domain,
                blob_path=value.blob_path,
                snapshot_digest=value.snapshot_digest,
                download_token=value.download_token,
                stream_mode=self.stream_mode,
            )
        return value

    def _maybe_emit_progress(
        self,
        *,
        stage: str,
        force: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.monotonic()
        with self._progress_lock:
            if not force and self._progress_interval_s > 0.0:
                if (now - self._last_progress_emit_mono) < self._progress_interval_s:
                    return
            bytes_written = int(self._bytes_written)
            bytes_uploaded = int(self._bytes_uploaded if self._stream_remote else self._bytes_written)
            chunks_written = int(self._chunks_written)
            chunks_uploaded = int(self._chunks_uploaded if self._stream_remote else self._chunks_written)
            error_class = str(self._stream_error_class or "").strip()
            elapsed = max(now - self._started_mono, 1e-6)
            delta_elapsed = max(now - self._last_progress_mono, 1e-6)
            delta_uploaded = max(0, bytes_uploaded - int(self._last_progress_uploaded))
            inst_bps = float(delta_uploaded) / delta_elapsed
            avg_bps = float(bytes_uploaded) / elapsed
            self._last_progress_emit_mono = now
            self._last_progress_mono = now
            self._last_progress_uploaded = bytes_uploaded
        payload: Dict[str, Any] = {
            "stage": stage,
            "ref": self._ref,
            "stream_mode": self.stream_mode,
            "bytes_written": bytes_written,
            "bytes_uploaded": bytes_uploaded,
            "chunks_written": chunks_written,
            "chunks_uploaded": chunks_uploaded,
            "upload_bps": float(avg_bps),
            "inst_upload_bps": float(inst_bps),
            "elapsed_s": float(elapsed),
        }
        if error_class:
            payload["error_class"] = error_class
        if extra:
            payload.update(dict(extra))
        self._ctx.emit("request.upload_progress", payload)

    def _finalize_presigned_upload(self) -> Any:
        """Hash the buffered temp file, then upload via presigned S3 multipart."""
        from .presigned_upload import blake3_hash_file, presigned_upload_file

        assert self._tmp_path is not None
        file_size = os.path.getsize(self._tmp_path)
        if file_size <= 0:
            raise RuntimeError("file save failed (empty file)")

        # Use the rolling hash if available (was computed during writes).
        blake3_hex = self._blake3_hasher.hexdigest()

        # Build auth headers.
        base = self._ctx._get_file_api_base_url()
        token = self._ctx._get_worker_capability_token()
        headers: Dict[str, str] = {"Authorization": f"Bearer {token}"}
        owner = (self._ctx.owner or "").strip()
        if owner:
            headers["X-Cozy-Owner"] = owner

        # Build endpoint and create payload.
        create_payload: Dict[str, Any] = {}
        req_id = str(self._ctx.request_id or "").strip()
        if req_id:
            create_payload["request_id"] = req_id

        if self._repo_job_scope is None:
            # Media upload.
            create_payload["ref"] = self._ref
            job_id = str(self._ctx.job_id or "").strip()
            if job_id:
                create_payload["job_id"] = job_id
            endpoint_path = "/api/v1/media/uploads"
        else:
            # Repo-CAS upload.
            repo_owner, repo, job_id = self._repo_job_scope
            create_payload["path"] = self._ref
            endpoint_path = (
                f"/api/v1/repos/{urllib.parse.quote(repo_owner, safe='')}/"
                f"{urllib.parse.quote(repo, safe='')}/jobs/{urllib.parse.quote(job_id, safe='')}/uploads"
            )

        def _progress_cb(parts_done: int, total_parts: int, bytes_up: int) -> None:
            with self._progress_lock:
                self._bytes_uploaded = bytes_up
                self._chunks_uploaded = parts_done
            self._maybe_emit_progress(stage="stream_upload")

        result = presigned_upload_file(
            file_path=self._tmp_path,
            base_url=base,
            endpoint_path=endpoint_path,
            headers=headers,
            create_payload=create_payload,
            blake3_hex=blake3_hex,
            size_bytes=file_size,
            retry_attempts=self._retry_attempts,
            retry_backoff_ms=self._retry_backoff_ms,
            on_progress=_progress_cb,
            cancel_check=self._ctx.is_canceled,
        )

        self._uploader_meta = result.meta
        with self._progress_lock:
            self._bytes_uploaded = file_size

        # Build return type from metadata.
        meta = dict(result.meta)
        size = int(meta.get("size_bytes") or file_size)
        sha = str(meta.get("sha256") or "").strip() or self._sha.hexdigest()
        asset = Asset(
            ref=self._ref,
            owner=self._ctx.owner,
            local_path=None,
            mime_type=str(meta.get("mime_type") or "") or None,
            size_bytes=size,
            sha256=sha,
            stream_mode=self.stream_mode,
        )
        if self._kind == "checkpoint":
            fmt = str(self._format or "").strip() or _infer_tensors_format(self._ref)
            return Tensors(
                ref=asset.ref,
                owner=asset.owner,
                local_path=asset.local_path,
                format=fmt,
                size_bytes=asset.size_bytes,
                sha256=asset.sha256,
                blake3=str(meta.get("blake3") or blake3_hex).strip() or None,
                blob_digest=str(meta.get("blob_digest") or "").strip() or None,
                blob_domain=str(meta.get("blob_domain") or "").strip() or None,
                blob_path=str(meta.get("blob_path") or "").strip() or None,
                snapshot_digest=str(meta.get("snapshot_digest") or "").strip() or None,
                download_token=asset.download_token,
                stream_mode=self.stream_mode,
            )
        return asset

    @staticmethod
    def _classify_error(exc: BaseException) -> str:
        msg = str(exc or "").lower()
        if isinstance(exc, InterruptedError):
            return "canceled"
        if isinstance(exc, AuthError):
            return "auth_error"
        if "network_error" in msg:
            return "network_error"
        if "already exists" in msg:
            return "conflict"
        if "unauthorized" in msg:
            return "auth_error"
        if "file save failed (" in msg:
            left = msg.split("file save failed (", 1)[1]
            code = left.split(")", 1)[0].strip()
            if code.isdigit() and code.startswith("5"):
                return "server_error"
            if code.isdigit() and code.startswith("4"):
                return "client_error"
        return "unknown_error"

    def __enter__(self) -> "_RequestOutputStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if exc_type is None:
            self.finalize()
        else:
            self.close()
        return False


class RequestContext:
    """Context object passed to request handlers, allowing cancellation."""

    def __init__(
        self,
        request_id: str,
        job_id: Optional[str] = None,
        emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        owner: Optional[str] = None,
        invoker_id: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        file_api_base_url: Optional[str] = None,
        worker_capability_token: Optional[str] = None,
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
        self._job_id = str(job_id or "").strip() or None
        self._owner = owner
        self._invoker_id = invoker_id
        self._timeout_ms = timeout_ms
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._worker_capability_token = (worker_capability_token or "").strip() or None
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
        self._cached_repo_job_scope: Optional[tuple[str, str, str]] = None

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def job_id(self) -> Optional[str]:
        return self._job_id

    @property
    def workspace_scope_id(self) -> str:
        return self._job_id or self._request_id

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
        return _require_worker_capability_token()

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

    def _should_stream_output_to_file_api(self, ref: str) -> bool:
        try:
            if self._resolve_local_output_path(ref):
                return False
        except Exception:
            return False
        try:
            _ = self._get_file_api_base_url()
            _ = self._get_worker_capability_token()
        except Exception:
            return False
        return True

    def _materialized_input_url_for_ref(self, ref: str) -> Optional[str]:
        raw = (ref or "").strip().lstrip("/")
        if not raw:
            return None
        out = str(self._materialized_input_urls.get(raw) or "").strip()
        if out:
            return out
        return None

    def _repo_job_upload_scope(self) -> Optional[tuple[str, str, str]]:
        """Return (owner, repo, job_id) for repo-CAS uploads, or None.

        Pure getter — no HTTP calls or side effects. TensorHub auto-creates
        the repo and lineage record on first upload when the capability token
        is valid.
        """
        if self._cached_repo_job_scope is not None:
            return self._cached_repo_job_scope

        hints = dict(self._execution_hints or {})
        kind = str(hints.get("kind", "") or "").strip().lower()
        if kind not in {"conversion", "training"}:
            return None
        destination_repo = str(
            hints.get("destination_repo")
            or hints.get("repo")
            or hints.get("output_repo")
            or ""
        ).strip()
        if destination_repo == "":
            return None
        job_id = str(
            hints.get("job_id")
            or hints.get("conversion_job_id")
            or hints.get("training_job_id")
            or self._job_id
            or ""
        ).strip()
        if job_id == "":
            return None
        try:
            owner, repo = _parse_owner_repo(destination_repo)
        except Exception:
            return None

        result = (owner, repo, job_id)
        self._cached_repo_job_scope = result
        return result

    def _tensor_upload_execution_kind(self) -> str:
        hints = dict(self._execution_hints or {})
        return str(hints.get("kind", "") or "").strip().lower()

    def _require_repo_job_scope_for_tensors(self, ref: str) -> None:
        """
        For training/conversion checkpoints, remote tensor uploads must be job-scoped
        repo-cas writes. This prevents silent fallback to user-files/media uploads.
        """
        kind = self._tensor_upload_execution_kind()
        if kind not in {"conversion", "training"}:
            return
        try:
            if self._resolve_local_output_path(ref):
                return
        except Exception:
            pass
        if self._repo_job_upload_scope() is None:
            raise RuntimeError(
                "tensor upload requires repo job scope (execution_hints.kind with destination_repo and job_id)"
            )

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
            "job_id": self._job_id,
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
        return f"jobs/{self._request_id}/outputs/items/{item_key}/{leaf}"

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
        """Check if the request was canceled."""
        return self._canceled

    def cancel(self) -> None:
        """Mark the request as canceled."""
        if not self._canceled:
            self._canceled = True
            self._cancel_event.set()
            logger.info(f"Action {self.request_id} marked for cancellation.")

    def done(self) -> threading.Event:
        """Returns an event that is set when the request is cancelled."""
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
        _enforce_output_file_size_limit(len(data))
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
        stream = self.open_output_stream(ref, create=False, expected_size_bytes=len(data))
        stream.write(data)
        out = stream.finalize()
        if isinstance(out, Asset):
            return out
        raise RuntimeError("file save failed (invalid_asset_response)")

    def save_file(self, ref: str, local_path: str) -> Asset:
        ref = _normalize_output_ref(ref)
        src = str(local_path or "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        size = int(os.path.getsize(src))
        _enforce_output_file_size_limit(size)

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
        stream = self.open_output_stream(ref, create=False, expected_size_bytes=size)
        with open(src, "rb") as fin:
            while True:
                chunk = fin.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        out = stream.finalize()
        if isinstance(out, Asset):
            return out
        raise RuntimeError("file save failed (invalid_asset_response)")

    def save_checkpoint(self, ref: str, local_path: str, format: Optional[str] = None) -> Tensors:
        """Save checkpoint/model-weight bytes and return a first-class tensor artifact."""
        ref = _normalize_output_ref(ref)
        self._require_repo_job_scope_for_tensors(ref)
        src = str(local_path or "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        size = int(os.path.getsize(src))
        _enforce_output_file_size_limit(size)
        fmt = str(format or "").strip() or _infer_tensors_format(ref or local_path)

        # For conversion/training job-scoped writes, force checkpoint-stream path so
        # tensor uploads use repo-cas job endpoints rather than media upload routes.
        if self._repo_job_upload_scope() is not None and self._should_stream_output_to_file_api(ref):
            stream = self.open_checkpoint_stream(ref, format=fmt, expected_size_bytes=size)
            with open(src, "rb") as fin:
                while True:
                    chunk = fin.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            out = stream.finalize()
            if isinstance(out, Tensors):
                return out
            raise RuntimeError("file save failed (invalid_tensors_response)")

        asset = self.save_file(ref, src)
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
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_checkpoint_bytes expects bytes")
        payload = bytes(data)
        _enforce_output_file_size_limit(len(payload))
        ref = _normalize_output_ref(ref)
        self._require_repo_job_scope_for_tensors(ref)
        fmt = str(format or "").strip() or _infer_tensors_format(ref)

        if self._repo_job_upload_scope() is not None and self._should_stream_output_to_file_api(ref):
            stream = self.open_checkpoint_stream(ref, format=fmt, expected_size_bytes=len(payload))
            stream.write(payload)
            out = stream.finalize()
            if isinstance(out, Tensors):
                return out
            raise RuntimeError("file save failed (invalid_tensors_response)")

        asset = self.save_bytes(ref, payload)
        return Tensors(
            ref=asset.ref,
            owner=asset.owner,
            local_path=asset.local_path,
            format=fmt,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
            download_token=asset.download_token,
        )

    def open_output_stream(
        self,
        ref: str,
        *,
        create: bool = False,
        expected_size_bytes: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to an Asset."""
        return _RequestOutputStream(
            ctx=self,
            ref=ref,
            kind="asset",
            create=create,
            expected_size_bytes=expected_size_bytes,
        )

    def open_checkpoint_stream(
        self,
        ref: str,
        *,
        format: Optional[str] = None,
        expected_size_bytes: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to Tensors."""
        ref = _normalize_output_ref(ref)
        self._require_repo_job_scope_for_tensors(ref)
        return _RequestOutputStream(
            ctx=self,
            ref=ref,
            kind="checkpoint",
            format=format,
            expected_size_bytes=expected_size_bytes,
        )

    def save_bytes_create(self, ref: str, data: bytes) -> Asset:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_bytes_create expects bytes")
        data = bytes(data)
        _enforce_output_file_size_limit(len(data))
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
        stream = self.open_output_stream(ref, create=True, expected_size_bytes=len(data))
        stream.write(data)
        out = stream.finalize()
        if isinstance(out, Asset):
            return out
        raise RuntimeError("file save failed (invalid_asset_response)")

    def save_file_create(self, ref: str, local_path: str) -> Asset:
        ref = _normalize_output_ref(ref)
        src = str(local_path or "").strip()
        if not src:
            raise ValueError("local_path is required")
        if not os.path.exists(src):
            raise FileNotFoundError(src)

        size = int(os.path.getsize(src))
        _enforce_output_file_size_limit(size)

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
        stream = self.open_output_stream(ref, create=True, expected_size_bytes=size)
        with open(src, "rb") as fin:
            while True:
                chunk = fin.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        out = stream.finalize()
        if isinstance(out, Asset):
            return out
        raise RuntimeError("file save failed (invalid_asset_response)")

    def save_bytes_overwrite(self, ref: str, data: bytes) -> Asset:
        # Back-compat alias: overwrite is the default save_bytes behavior.
        return self.save_bytes(ref, data)

    def save_file_overwrite(self, ref: str, local_path: str) -> Asset:
        return self.save_file(ref, local_path)

    def publish_repo_revision(
        self,
        *,
        destination_repo: str,
        artifact_refs: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_if_missing: bool = True,
        destination_repo_tags: Optional[List[str]] = None,
        source_repo: Optional[str] = None,
        source_version_id: Optional[str] = None,
        target_version_id: Optional[str] = None,
        snapshot_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Publish conversion lineage to Tensorhub using public HTTP APIs only.

        Uses worker capability auth and never touches DB/internal-only paths.

        artifact_refs: list of artifact references. Each item can be:
          - a dict with keys: digest (required), path, size_bytes, domain
            (from presigned upload complete response: blob_digest, blob_path, etc.)
          - a string file ref (legacy, used as path only — no digest validation)
        """
        owner, repo = _parse_owner_repo(destination_repo)
        source_owner = ""
        source_name = ""
        if source_repo and str(source_repo).strip():
            try:
                source_owner, source_name, _ = _parse_owner_repo_with_optional_tag(str(source_repo))
            except ValueError:
                source_owner, source_name = "", ""
        normalized_source_version_id = str(source_version_id or "").strip().lower()
        normalized_target_version_id = str(target_version_id or "").strip().lower()
        normalized_tags = _normalize_destination_repo_tags(destination_repo_tags)

        version_mode = "new_version"
        if (
            normalized_source_version_id
            and source_owner.strip().lower() == owner.strip().lower()
            and source_name.strip().lower() == repo.strip().lower()
        ):
            version_mode = "same_version_variant"

        input_versions: List[str] = []
        if normalized_source_version_id:
            input_versions.append(normalized_source_version_id)
        output_versions: List[str] = []
        if normalized_target_version_id:
            output_versions.append(normalized_target_version_id)
        if version_mode == "same_version_variant" and normalized_source_version_id:
            output_versions = [normalized_source_version_id]

        transform_spec: Dict[str, Any] = {}
        md = dict(metadata or {})
        if source_repo and str(source_repo).strip():
            transform_spec["source_repo"] = str(source_repo).strip()
        for key in ("source_provider", "source_ref", "source_revision", "target_layout", "save_formats"):
            value = md.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
            transform_spec[key] = value
        publish_intent: Dict[str, Any] = {
            "repo": f"{owner.strip().lower()}/{repo.strip().lower()}",
            "version_mode": version_mode,
            "transform_spec": transform_spec,
            "tag_policy": {
                "destination_repo_tags": [],
                "manage_latest": True,
            },
        }
        if normalized_source_version_id:
            publish_intent["source_version_id"] = normalized_source_version_id
        if normalized_tags:
            if output_versions:
                publish_intent["tag_policy"] = {
                    "destination_repo_tags": normalized_tags,
                    "manage_latest": True,
                }
            else:
                logger.warning(
                    "worker_publish_tags_skipped request_id=%s job_id=%s owner=%s repo=%s reason=missing_target_version tags=%s",
                    self.request_id,
                    self.job_id or "",
                    owner,
                    repo,
                    ",".join(normalized_tags),
                )
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if token:
            required_permissions = ["repo-version:create"]
            if create_if_missing:
                required_permissions.insert(0, "tensor-repo:create")
            _assert_token_repo_scope_matches_destination(token, owner, repo, required_permissions=required_permissions)
        logger.info(
            "worker_publish_attempt request_id=%s job_id=%s owner=%s repo=%s",
            self.request_id,
            self.job_id or "",
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

        def _request_json(method: str, path: str, payload: Dict[str, Any], *, allow_404: bool = False) -> Dict[str, Any]:
            url = f"{base}{path}"
            resp = requests.request(method=method, url=url, headers=headers, data=json.dumps(payload), timeout=30)
            code = int(resp.status_code)
            if code in (401, 403):
                detail = ""
                try:
                    detail = str((resp.text or "").strip())
                except Exception:
                    detail = ""
                if detail != "":
                    raise AuthError(
                        f"repo publish unauthorized ({code}): check worker_capability_token validity; response={detail[:256]}"
                    )
                raise AuthError(f"repo publish unauthorized ({code}): check worker_capability_token validity")
            if allow_404 and code == 404:
                return {"ok": False, "not_found": True}
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

        # TensorHub auto-creates the repo and lineage record on the upload path,
        # so no explicit repo creation or conversion-jobs/start call is needed.
        # The job_id comes from the capability token (via execution_hints).
        scope = self._repo_job_upload_scope()
        if scope is not None:
            _, _, job_id = scope
        else:
            job_id = ""
        if job_id == "":
            raise RuntimeError("repo publish failed: no job_id in execution_hints (missing destination_repo or job scope)")

        metrics = dict(md)
        commit_output_variants = []
        raw_output_variants = md.get("output_variants")
        if isinstance(raw_output_variants, list):
            for item in raw_output_variants:
                if isinstance(item, dict):
                    commit_output_variants.append(dict(item))

        # Wire artifact_refs into output_variants if not already present.
        parsed_artifacts = []
        for ref in (artifact_refs or []):
            if isinstance(ref, dict):
                art: Dict[str, Any] = {}
                digest = str(ref.get("digest") or ref.get("blob_digest") or "").strip()
                if digest:
                    art["digest"] = digest
                path_val = str(ref.get("path") or ref.get("blob_path") or "").strip()
                if path_val:
                    art["path"] = path_val
                size_val = ref.get("size_bytes")
                if size_val is not None:
                    art["size_bytes"] = int(size_val)
                domain_val = str(ref.get("domain") or ref.get("blob_domain") or "private").strip()
                art["domain"] = domain_val
                if art.get("digest"):
                    parsed_artifacts.append(art)
            elif isinstance(ref, str) and ref.strip():
                # Legacy: string ref used as path hint only.
                parsed_artifacts.append({"path": ref.strip()})

        if parsed_artifacts and commit_output_variants:
            for variant in commit_output_variants:
                if not variant.get("artifacts"):
                    variant["artifacts"] = parsed_artifacts
        # Only include output_variants when they have the required fields
        # (version_id, variant_label, etc.). Bare artifact-only variants will
        # fail TensorHub validation. When we have no structured variants, the
        # commit relies on output_versions + publish_intent instead.

        commit_payload = {
            "job_id": str(self.request_id or job_id),
            "run_kind": "conversion",
            "status": "succeeded",
            "commit_idempotency_key": f"worker:{str(self.request_id or '').strip() or 'request'}:{job_id}:v1",
            "metrics_json": metrics,
            "cost_json": {},
            "output_versions": output_versions,
            "output_variants": commit_output_variants,
            "publish_intent": publish_intent,
        }
        if isinstance(snapshot_manifest, dict) and snapshot_manifest:
            commit_payload["snapshot_manifest"] = snapshot_manifest
        logger.info(
            "worker_publish_commit request_id=%s job_id=%s owner=%s repo=%s output_versions=%d output_variants=%d variant_labels=%s snapshot_manifest=%s",
            self.request_id,
            job_id,
            owner,
            repo,
            len(output_versions),
            len(commit_output_variants),
            [str(v.get("variant_label") or "").strip() for v in commit_output_variants if isinstance(v, dict)],
            bool(commit_payload.get("snapshot_manifest")),
        )
        commit_result = _request_json(
            "POST",
            f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/jobs/{urllib.parse.quote(job_id, safe='')}/commit",
            commit_payload,
            allow_404=True,
        )
        if bool(commit_result.get("not_found")):
            # Back-compat fallback for older Tensorhub deployments.
            finalize_payload = {
                "status": "succeeded",
                "metrics_json": metrics,
                "cost_json": {},
                "output_versions": output_versions,
                "publish_intent": publish_intent,
            }
            commit_result = _request_json(
                "POST",
                f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/jobs/{urllib.parse.quote(job_id, safe='')}/finalize",
                finalize_payload,
            )
        commit_output_versions = [
            str(v or "").strip().lower()
            for v in list((commit_result or {}).get("output_versions") or [])
            if str(v or "").strip()
        ]
        logger.info(
            "worker_publish_commit_result request_id=%s job_id=%s owner=%s repo=%s output_versions=%d output_variant_count=%s",
            self.request_id,
            job_id,
            owner,
            repo,
            len(commit_output_versions),
            commit_result.get("output_variant_count"),
        )

        logger.info(
            "worker_publish_succeeded request_id=%s job_id=%s owner=%s repo=%s published_job_id=%s",
            self.request_id,
            self.job_id or "",
            owner,
            repo,
            job_id,
        )

        out: Dict[str, Any] = {"ok": True, "owner": owner, "repo": repo, "job_id": job_id, "job_id": job_id}
        if commit_output_versions:
            out["output_versions"] = commit_output_versions
        elif output_versions:
            out["output_versions"] = output_versions
        if normalized_tags:
            out["destination_repo_tags"] = normalized_tags
        return out

    def read_repo_metadata(self, *, destination_repo: str) -> Dict[str, Any]:
        """Read repo-level metadata from Tensorhub public HTTP API."""
        owner, repo = _parse_owner_repo(destination_repo)
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo, required_permissions=["tensor-repo:read"])
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
        token = (self._worker_capability_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo, required_permissions=["repo-version:create"])
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

    def search_metadata_claims(
        self,
        *,
        scope: str = "version",
        identity_hash: Optional[str] = None,
        metadata_contains: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        cursor: int = 0,
    ) -> Dict[str, Any]:
        """Search metadata claims by writer identity (derived server-side from token)."""
        normalized_scope = str(scope or "").strip().lower() or "version"
        if normalized_scope not in {"repo", "version"}:
            raise ValueError("scope must be 'repo' or 'version'")
        digest = str(identity_hash or "").strip().lower()
        if digest and ":" not in digest:
            raise ValueError("identity_hash must be a digest ref")
        if metadata_contains is not None and not isinstance(metadata_contains, dict):
            raise ValueError("metadata_contains must be an object")
        if not digest and not metadata_contains:
            raise ValueError("identity_hash or metadata_contains is required")
        if limit <= 0:
            limit = 50
        if limit > 200:
            limit = 200
        if cursor < 0:
            cursor = 0

        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "items": []}

        payload: Dict[str, Any] = {
            "scope": normalized_scope,
            "limit": int(limit),
            "cursor": int(cursor),
        }
        if digest:
            payload["identity_hash"] = digest
        if metadata_contains:
            payload["metadata_contains"] = metadata_contains

        url = f"{base}/api/v1/metadata/claims/search"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        code = int(resp.status_code)
        if code in (401, 403):
            raise AuthError(f"metadata claims search unauthorized ({code}): check worker_capability_token validity")
        if code < 200 or code >= 300:
            err_code = ""
            err_msg = ""
            try:
                body = resp.json()
            except Exception:
                body = {}
            if isinstance(body, dict):
                raw = body.get("error")
                if isinstance(raw, dict):
                    err_code = str(raw.get("code") or "").strip()
                    err_msg = str(raw.get("message") or "").strip()
                elif isinstance(raw, str):
                    err_code = raw.strip()
                if not err_msg:
                    err_msg = str(body.get("message") or "").strip()
            detail = err_code or f"status_{code}"
            if err_msg:
                raise RuntimeError(f"metadata_claim_search_failed:{detail}:{err_msg}")
            raise RuntimeError(f"metadata_claim_search_failed:{detail}")
        try:
            parsed = resp.json()
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        items = parsed.get("items")
        if not isinstance(items, list):
            items = []
        return {
            "ok": True,
            "scope": str(parsed.get("scope") or normalized_scope),
            "limit": int(parsed.get("limit") or limit),
            "cursor": int(parsed.get("cursor") or cursor),
            "next_cursor": int(parsed.get("next_cursor") or (cursor + len(items))),
            "items": items,
        }

    def upsert_version_metadata_claim(
        self,
        *,
        destination_repo: str,
        version_id: str,
        identity_hash: Optional[str],
        metadata_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create/update a version metadata claim for this worker/user writer identity."""
        owner, repo = _parse_owner_repo(destination_repo)
        normalized_version_id = str(version_id or "").strip().lower()
        if ":" not in normalized_version_id:
            raise ValueError("version_id must be a digest ref")
        if not isinstance(metadata_json, dict):
            raise ValueError("metadata_json must be an object")
        digest = str(identity_hash or "").strip().lower()
        if digest and ":" not in digest:
            raise ValueError("identity_hash must be a digest ref")

        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo, required_permissions=["repo-version:create"])
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "owner": owner, "repo": repo}

        payload: Dict[str, Any] = {"metadata_json": metadata_json}
        if digest:
            payload["identity_hash"] = digest
        url = (
            f"{base}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(repo, safe='')}/versions/{urllib.parse.quote(normalized_version_id, safe='')}/metadata/claims"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Cozy-Owner": owner,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        code = int(resp.status_code)
        if code in (401, 403):
            raise AuthError(f"version metadata claim write unauthorized ({code}): check worker_capability_token validity")
        if code < 200 or code >= 300:
            err_code = ""
            try:
                body = resp.json()
            except Exception:
                body = {}
            if isinstance(body, dict):
                raw = body.get("error")
                if isinstance(raw, dict):
                    err_code = str(raw.get("code") or "").strip()
                elif isinstance(raw, str):
                    err_code = raw.strip()
            raise RuntimeError(f"metadata_claim_invalid:{err_code or code}")

        try:
            parsed = resp.json()
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return {"ok": True, **parsed}

    def mirror_dedupe_or_run(
        self,
        *,
        source_identity: Dict[str, Any],
        destination_repo: str,
        destination_repo_tags: Optional[List[str]] = None,
        on_miss: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Perform mirror dedupe orchestration with a single high-level API.

        Flow:
        1) Search claims for canonical source identity.
        2) Attempt copy-by-reference from matching claim candidates.
        3) Invalidate stale claims when source lineage no longer exists.
        4) Run `on_miss` callback when no reusable lineage is found.
        5) Write/update destination version claim after hit or miss publish.
        """

        if not isinstance(source_identity, dict):
            raise ValueError("source_identity must be an object")

        provider = str(source_identity.get("provider") or "").strip().lower()
        source_ref = str(source_identity.get("source_ref") or "").strip()
        source_revision = str(source_identity.get("source_revision") or "").strip()
        identity_hash = str(source_identity.get("identity_hash") or "").strip().lower()
        dedupe_supported = bool(source_identity.get("dedupe_supported", True))

        if provider == "":
            raise ValueError("source_identity.provider is required")
        if source_ref == "":
            raise ValueError("source_identity.source_ref is required")
        if identity_hash == "" or ":" not in identity_hash:
            raise ValueError("source_identity.identity_hash must be a digest ref")

        destination_owner, destination_name = _parse_owner_repo(destination_repo)
        normalized_destination = f"{destination_owner}/{destination_name}"
        normalized_tags = _normalize_destination_repo_tags(destination_repo_tags)

        metadata_contains = {
            "source": {
                "provider": provider,
                "source_ref": source_ref,
                "source_revision": source_revision,
            }
        }

        invalidated_claim_ids: List[int] = []
        warnings: List[Dict[str, Any]] = []

        def _record_warning(*, stage: str, code: str, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
            item: Dict[str, Any] = {
                "stage": str(stage or "").strip() or "unknown",
                "code": str(code or "").strip().lower() or "unknown",
                "message": str(message or "").strip(),
            }
            if payload:
                item.update(payload)
            warnings.append(item)

        def _build_claim_metadata(*, version_id: str, primary_ref: str, primary_format: str) -> Dict[str, Any]:
            return {
                "source": {
                    "provider": provider,
                    "source_ref": source_ref,
                    "source_revision": source_revision,
                },
                "result": {
                    "destination_repo": normalized_destination,
                    "version_id": version_id,
                    "primary_artifact_ref": primary_ref,
                    "primary_artifact_format": primary_format,
                },
                "request_id": str(self.request_id or "").strip(),
                "written_at": _utc_timestamp_rfc3339(),
            }

        def _write_claim(*, version_id: str, primary_ref: str, primary_format: str) -> bool:
            normalized_version = str(version_id or "").strip().lower()
            if normalized_version == "" or ":" not in normalized_version:
                return False
            try:
                self.upsert_version_metadata_claim(
                    destination_repo=normalized_destination,
                    version_id=normalized_version,
                    identity_hash=identity_hash,
                    metadata_json=_build_claim_metadata(
                        version_id=normalized_version,
                        primary_ref=str(primary_ref or "").strip(),
                        primary_format=str(primary_format or "").strip(),
                    ),
                )
                return True
            except Exception as exc:
                _record_warning(
                    stage="claim_write",
                    code=_error_code_from_exception(exc, fallback="metadata_claim_write_failed"),
                    message=str(exc),
                    payload={"version_id": normalized_version},
                )
                return False

        if dedupe_supported:
            try:
                search_out = self.search_metadata_claims(
                    scope="version",
                    identity_hash=identity_hash,
                    metadata_contains=metadata_contains,
                    limit=20,
                    cursor=0,
                )
            except Exception as exc:
                code = _error_code_from_exception(exc, fallback="metadata_claim_search_failed")
                raise RuntimeError(f"mirror_dedupe_search_failed:{code}") from exc

            if not bool((search_out or {}).get("skipped")):
                for raw in list((search_out or {}).get("items") or []):
                    if not isinstance(raw, dict):
                        continue
                    source_owner = str(raw.get("owner") or "").strip().lower()
                    source_repo = str(raw.get("repo") or "").strip().lower()
                    source_version_id = str(raw.get("version_id") or "").strip().lower()
                    if source_owner == "" or source_repo == "" or source_version_id == "":
                        continue

                    claim_id_raw = raw.get("claim_id")
                    claim_id = int(claim_id_raw) if isinstance(claim_id_raw, int) else 0
                    source_repo_ref = f"{source_owner}/{source_repo}"

                    try:
                        copy_out = self.copy_repo_by_reference(
                            source_repo=source_repo_ref,
                            source_version_id=source_version_id,
                            destination_repo=normalized_destination,
                            destination_repo_tags=normalized_tags,
                            claim_id=claim_id if claim_id > 0 else None,
                        )
                    except Exception as exc:
                        code = _error_code_from_exception(exc, fallback="mirror_copy_failed")
                        _record_warning(
                            stage="copy_by_reference",
                            code=code,
                            message=str(exc),
                            payload={
                                "source_repo": source_repo_ref,
                                "source_version_id": source_version_id,
                            },
                        )
                        if code in _STALE_MIRROR_CLAIM_ERROR_CODES and claim_id > 0:
                            try:
                                self.delete_version_metadata_claim(
                                    destination_repo=source_repo_ref,
                                    version_id=source_version_id,
                                    claim_id=claim_id,
                                )
                                invalidated_claim_ids.append(claim_id)
                            except Exception as delete_exc:
                                _record_warning(
                                    stage="claim_invalidation",
                                    code=_error_code_from_exception(delete_exc, fallback="metadata_claim_delete_failed"),
                                    message=str(delete_exc),
                                    payload={
                                        "claim_id": claim_id,
                                        "source_repo": source_repo_ref,
                                        "source_version_id": source_version_id,
                                    },
                                )
                        continue

                    copied_version_id = str((copy_out or {}).get("copied_version_id") or source_version_id).strip().lower()
                    _meta_raw = raw.get("metadata_json")
                    metadata_json: Dict[Any, Any] = _meta_raw if isinstance(_meta_raw, dict) else {}
                    _result_raw = metadata_json.get("result")
                    result_json: Dict[Any, Any] = _result_raw if isinstance(_result_raw, dict) else {}
                    primary_ref = str(result_json.get("primary_artifact_ref") or "").strip()
                    primary_format = str(result_json.get("primary_artifact_format") or "").strip()
                    claim_written = _write_claim(
                        version_id=copied_version_id,
                        primary_ref=primary_ref,
                        primary_format=primary_format,
                    )
                    return {
                        "ok": True,
                        "result_code": "dedupe_copy_hit",
                        "hit": True,
                        "source_provider": provider,
                        "source_ref": source_ref,
                        "source_revision": source_revision,
                        "identity_hash": identity_hash,
                        "copied_from_repo": source_repo_ref,
                        "copied_from_version_id": source_version_id,
                        "copied_version_id": copied_version_id,
                        "primary_artifact_ref": primary_ref,
                        "primary_artifact_format": primary_format,
                        "claim_written": claim_written,
                        "invalidated_claim_ids": invalidated_claim_ids,
                        "warnings": warnings,
                    }

        miss_result: Dict[str, Any] = {}
        if callable(on_miss):
            try:
                maybe = on_miss()
                if isinstance(maybe, dict):
                    miss_result = maybe
            except Exception as exc:
                code = _error_code_from_exception(exc, fallback="on_miss_failed")
                raise RuntimeError(f"mirror_dedupe_on_miss_failed:{code}") from exc

        published_version_id = str(
            miss_result.get("version_id")
            or miss_result.get("published_version_id")
            or ""
        ).strip().lower()
        primary_ref = str(
            miss_result.get("primary_artifact_ref")
            or miss_result.get("primary_ref")
            or ""
        ).strip()
        primary_format = str(
            miss_result.get("primary_artifact_format")
            or miss_result.get("primary_format")
            or ""
        ).strip()

        claim_written = False
        if published_version_id:
            claim_written = _write_claim(
                version_id=published_version_id,
                primary_ref=primary_ref,
                primary_format=primary_format,
            )

        return {
            "ok": True,
            "result_code": "dedupe_miss",
            "hit": False,
            "source_provider": provider,
            "source_ref": source_ref,
            "source_revision": source_revision,
            "identity_hash": identity_hash,
            "published_version_id": published_version_id,
            "primary_artifact_ref": primary_ref,
            "primary_artifact_format": primary_format,
            "claim_written": claim_written,
            "invalidated_claim_ids": invalidated_claim_ids,
            "warnings": warnings,
            "miss_result": miss_result,
        }

    def delete_version_metadata_claim(
        self,
        *,
        destination_repo: str,
        version_id: str,
        claim_id: int,
    ) -> Dict[str, Any]:
        """Delete a version metadata claim by numeric claim id."""
        owner, repo = _parse_owner_repo(destination_repo)
        normalized_version_id = str(version_id or "").strip().lower()
        if ":" not in normalized_version_id:
            raise ValueError("version_id must be a digest ref")
        if int(claim_id) <= 0:
            raise ValueError("claim_id must be > 0")

        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo, required_permissions=["repo-version:create"])
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "owner": owner, "repo": repo}

        url = (
            f"{base}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(repo, safe='')}/versions/{urllib.parse.quote(normalized_version_id, safe='')}/"
            f"metadata/claims/{int(claim_id)}"
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Cozy-Owner": owner,
        }
        resp = requests.delete(url, headers=headers, timeout=30)
        code = int(resp.status_code)
        if code in (401, 403):
            raise AuthError(f"version metadata claim delete unauthorized ({code}): check worker_capability_token validity")
        if code == 404:
            return {"ok": True, "deleted": False}
        if code < 200 or code >= 300:
            raise RuntimeError(f"metadata_claim_delete_failed:{code}")
        return {"ok": True, "deleted": True, "claim_id": int(claim_id)}

    def copy_repo_by_reference(
        self,
        *,
        source_repo: str,
        source_version_id: str,
        destination_repo: str,
        destination_repo_tags: Optional[List[str]] = None,
        claim_id: Optional[int] = None,
        release_visibility: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Copy an existing repo/version snapshot into another repo without re-upload."""
        source_owner, source_name = _parse_owner_repo(source_repo)
        destination_owner, destination_name = _parse_owner_repo(destination_repo)
        normalized_version_id = str(source_version_id or "").strip().lower()
        if ":" not in normalized_version_id:
            raise ValueError("source_version_id must be a digest ref")
        normalized_tags = _normalize_destination_repo_tags(destination_repo_tags)
        visibility = str(release_visibility or "").strip().lower()
        if visibility not in {"", "public", "private"}:
            raise ValueError("release_visibility must be 'public' or 'private'")
        claim_id_value = int(claim_id or 0)
        if claim_id_value < 0:
            raise ValueError("claim_id must be >= 0")

        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, destination_owner, destination_name, required_permissions=["repo-version:create"])
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel"}

        payload: Dict[str, Any] = {
            "source_owner": source_owner,
            "source_repo": source_name,
            "source_version_id": normalized_version_id,
            "destination_owner": destination_owner,
            "destination_repo": destination_name,
            "destination_repo_tags": normalized_tags,
        }
        if visibility:
            payload["release_visibility"] = visibility
        if claim_id_value > 0:
            payload["claim_id"] = claim_id_value
        url = f"{base}/api/v1/repos/copy-by-reference"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Cozy-Owner": destination_owner,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        code = int(resp.status_code)
        if code in (401, 403):
            raise AuthError(f"repo copy-by-reference unauthorized ({code}): check worker_capability_token validity")
        if code < 200 or code >= 300:
            err_code = ""
            err_msg = ""
            try:
                body = resp.json()
            except Exception:
                body = {}
            if isinstance(body, dict):
                raw = body.get("error")
                if isinstance(raw, dict):
                    err_code = str(raw.get("code") or "").strip()
                    err_msg = str(raw.get("message") or "").strip()
                elif isinstance(raw, str):
                    err_code = raw.strip()
                if not err_msg:
                    err_msg = str(body.get("message") or "").strip()
            detail = err_code or f"status_{code}"
            if err_msg:
                raise RuntimeError(f"mirror_copy_forbidden:{detail}:{err_msg}")
            raise RuntimeError(f"mirror_copy_forbidden:{detail}")
        try:
            parsed = resp.json()
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return {"ok": True, **parsed}
