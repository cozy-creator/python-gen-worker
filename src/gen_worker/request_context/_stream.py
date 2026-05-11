"""Chunk-writable output stream used by RequestContext.

Pulled out of the monolithic request_context module. Forward-ref to
`RequestContext` is string-typed (via `from __future__ import annotations`)
to avoid a circular import between this module and `__init__.py`.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import tempfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from blake3 import blake3

from ..api.errors import AuthError
from ..api.types import Asset, Tensors
from ._helpers import _enforce_output_file_size_limit, _infer_mime_type, _infer_tensors_format, _normalize_output_ref

if TYPE_CHECKING:
    from . import RequestContext


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
        # Lineage metadata (checkpoint uploads only).
        produced_by_kind: Optional[str] = None,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
        output_kind: Optional[str] = None,
        target_dtype: Optional[str] = None,
        flavor: Optional[str] = None,
    ) -> None:
        from ..presigned_upload import blake3_hash_file, presigned_upload_file

        self._ctx = ctx
        self._ref = _normalize_output_ref(ref)
        self._kind = str(kind or "asset").strip().lower()
        self._format = str(format or "").strip() or None
        self._create = bool(create)
        self._expected_size_bytes = int(expected_size_bytes or 0)
        # Lineage: carried onto the /complete payload for repo-cas uploads.
        self._lineage_produced_by_kind = (str(produced_by_kind or "").strip() or None)
        self._lineage_step_number = step_number if isinstance(step_number, int) else None
        self._lineage_epoch_number = epoch_number if isinstance(epoch_number, int) else None
        self._lineage_output_kind = (str(output_kind or "").strip() or None)
        self._lineage_target_dtype = (str(target_dtype or "").strip() or None)
        self._lineage_flavor = (str(flavor or "").strip() or None)
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
        # Route all output streams (checkpoint AND asset) through the
        # repo-CAS upload session when the job carries a destination_repo
        # scope. Without this, clone/conversion jobs splay their non-tensor
        # auxiliary files (README, config.json, etc.) onto /api/v1/media/:owner/uploads,
        # which the worker cap-token does not authorize.
        # Inference jobs have no destination_repo, so scope stays None and
        # the media route remains the default for those.
        self._repo_job_scope = self._ctx._repo_job_upload_scope()
        self._finalized = False
        self._result: Any = None

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
                if self._kind == "checkpoint":
                    raw = self._ctx.save_checkpoint(
                        self._ref,
                        self._tmp_path,
                        format=self._format,
                        produced_by_kind=self._lineage_produced_by_kind,
                        step_number=self._lineage_step_number,
                        epoch_number=self._lineage_epoch_number,
                        output_kind=self._lineage_output_kind,
                        target_dtype=self._lineage_target_dtype,
                        flavor=self._lineage_flavor,
                    )
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
                blake3=value.blake3,
                media_id=value.media_id,
                url=value.url,
                url_expires_at=value.url_expires_at,
                receipt_jws=value.receipt_jws,
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
        from ..presigned_upload import blake3_hash_file, presigned_upload_file

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
        content_type = self._infer_content_type()
        if content_type and content_type != "application/octet-stream":
            create_payload["content_type"] = content_type

        if self._repo_job_scope is None:
            # Media upload.
            create_payload["ref"] = self._ref
            job_id = str(self._ctx.job_id or "").strip()
            if job_id:
                create_payload["job_id"] = job_id
            # Owner is now an explicit URL segment (mirrors /repos/:owner/...,
            # /endpoints/:owner/..., /datasets/:owner/...). The capability
            # token still carries owner binding; tensorhub enforces the URL
            # owner matches the token-bound owner on each request.
            if not owner:
                raise RuntimeError(
                    "file save failed (missing owner): media uploads require ctx.owner"
                )
            owner_seg = urllib.parse.quote(owner, safe="")
            endpoint_path = f"/api/v1/media/{owner_seg}/uploads"
        else:
            # Repo-CAS upload — issue #20 session-scoped URL shape. The
            # session is opened lazily by the ctx-level manager on first use
            # and cached so subsequent uploads to the same repo reuse it.
            repo_owner, repo, _job_id = self._repo_job_scope
            create_payload["path"] = self._ref
            session_id = self._ctx._checkpoint_session_id(repo_owner, repo)
            endpoint_path = (
                f"/api/v1/repos/{urllib.parse.quote(repo_owner, safe='')}/"
                f"{urllib.parse.quote(repo, safe='')}/upload-sessions/"
                f"{urllib.parse.quote(session_id, safe='')}/uploads"
            )

        def _progress_cb(parts_done: int, total_parts: int, bytes_up: int) -> None:
            with self._progress_lock:
                self._bytes_uploaded = bytes_up
                self._chunks_uploaded = parts_done
            self._maybe_emit_progress(stage="stream_upload")

        # Repo-CAS uploads carry lineage metadata onto /complete so tensorhub
        # can populate checkpoint_lineage. Media uploads have no lineage.
        complete_extra: Optional[Dict[str, Any]] = None
        if self._repo_job_scope is not None:
            extra: Dict[str, Any] = {}
            if self._lineage_produced_by_kind:
                extra["produced_by_kind"] = self._lineage_produced_by_kind
            if self._lineage_step_number is not None:
                extra["step_number"] = int(self._lineage_step_number)
            if self._lineage_epoch_number is not None:
                extra["epoch_number"] = int(self._lineage_epoch_number)
            if self._lineage_output_kind:
                extra["output_kind"] = self._lineage_output_kind
            if self._lineage_target_dtype:
                extra["target_dtype"] = self._lineage_target_dtype
            if self._lineage_flavor:
                extra["flavor"] = self._lineage_flavor
            if extra:
                complete_extra = extra

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
            complete_extra=complete_extra,
        )

        self._uploader_meta = result.meta
        with self._progress_lock:
            self._bytes_uploaded = file_size

        # Build return type from metadata.
        meta = dict(result.meta)
        size = int(meta.get("size_bytes") or file_size)
        sha = str(meta.get("sha256") or "").strip() or self._sha.hexdigest()
        final_ref = str(meta.get("ref") or self._ref).strip() or self._ref
        asset = Asset(
            ref=final_ref,
            owner=self._ctx.owner,
            local_path=None,
            mime_type=str(meta.get("mime_type") or "") or None,
            size_bytes=size,
            sha256=sha,
            blake3=str(meta.get("blake3") or blake3_hex).strip() or None,
            media_id=str(meta.get("media_id") or "").strip() or None,
            url=str(meta.get("url") or "").strip() or None,
            url_expires_at=str(meta.get("url_expires_at") or "").strip() or None,
            receipt_jws=str(meta.get("receipt_jws") or "").strip() or None,
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

    def _infer_content_type(self) -> str:
        assert self._tmp_path is not None
        head = b""
        try:
            with open(self._tmp_path, "rb") as f:
                head = f.read(512)
        except Exception:
            head = b""
        return _infer_mime_type(self._ref, head)

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

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            self.finalize()
        else:
            self.close()
        return False
