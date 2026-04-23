from __future__ import annotations

import hashlib
import json
import logging
import os
import base64
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import requests

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from ..api.errors import AuthError
from ..api.types import Asset, Compute, Tensors


def _default_compute() -> Compute:
    """Sentinel Compute used when the orchestrator didn't attach resolved_compute.

    Tenants can safely read ``ctx.compute.vram_gb`` etc. without None-checks;
    zero / empty values are the "not specified" signal.
    """
    return Compute()

logger = logging.getLogger(__name__)

# Helpers, constants, and JWT/SSRF utilities live in _helpers.py. They are
# re-exported here so existing `from gen_worker.request_context import _foo`
# call sites (worker.py, trainer/runtime.py, tests) keep working.
from ._helpers import (
    _HINT_KEYS_DESTINATION_REPO,
    _HINT_KEYS_EXECUTION_KIND,
    _HINT_KEYS_JOB_ID,
    _MAX_OUTPUT_FILE_BYTES,
    _FILE_API_HTTP_TIMEOUT_S,
    _FILE_API_STREAM_ABORT_TIMEOUT_S,
    _FILE_API_STREAM_CHUNK_TIMEOUT_S,
    _FILE_API_STREAM_FINALIZE_TIMEOUT_S,
    _FILE_API_STREAM_REPLAY_TIMEOUT_S,
    _PUBLIC_TAG_RE,
    _STALE_MIRROR_CLAIM_ERROR_CODES,
    _assert_token_repo_scope_matches_destination,
    _canonicalize_model_ref_string,
    _decode_unverified_jwt_claims,
    _default_output_prefix,
    _encode_ref_for_url,
    _enforce_output_file_size_limit,
    _env_bool,
    _env_int,
    _error_code_from_exception,
    _http_request,
    _infer_mime_type,
    _infer_tensors_format,
    _is_private_ip_str,
    _normalize_destination_repo_tags,
    _normalize_output_ref,
    _normalize_repo_name,
    _parse_owner_repo,
    _parse_owner_repo_with_optional_tag,
    _require_file_api_base_url,
    _require_worker_capability_token,
    _resolve_hint_first_string,
    _sha256_file,
    _url_is_blocked,
    _utc_timestamp_rfc3339,
)


from ._stream import _RequestOutputStream

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
        source_info: Optional[Dict[str, Any]] = None,
        destination_info: Optional[Dict[str, Any]] = None,
        compute: Optional["Compute"] = None,
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
        # Reserved-name conversion/training contract attributes. Populated by
        # Worker._handle_job_request before invoking tenant code when the
        # endpoint is kind=conversion|training and the payload declares the
        # reserved `source`/`destination` struct fields. See e2e progress.json
        # issue #5.
        self._source_info = dict(source_info or {})
        self._destination_info = dict(destination_info or {})
        self._source_path: Optional[str] = None
        # Resolved hardware for this invocation (tensorhub #232). Populated by
        # Worker._handle_job_request from JobExecutionRequest.resolved_compute.
        # Sentinel defaults when unset — tenants can safely read fields without
        # None-checks.
        self._compute: "Compute" = compute if compute is not None else _default_compute()

        # Upload-session manager (issue #20). Lazy-created on first
        # ctx.save_file / ctx.save_checkpoint / ctx.save_output_stream use.
        # Tenant-invisible; library-internal machinery for the session
        # lifecycle (open → per-file uploads → finalize).
        self._upload_sessions = None  # type: Optional["_UploadSessionManager"]

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
    def compute(self) -> Compute:
        """Resolved hardware for this invocation (tensorhub #232).

        Read-only. For inference this equals the endpoint's ``[resources]``.
        For training this is the endpoint resources merged with the invoker's
        ``compute`` overrides from the wire payload. Fields default to zero /
        empty strings when the orchestrator hasn't attached resolved_compute
        (older protobuf, inference-only callpath, etc) — tenants can branch
        on ``ctx.compute.gpu_count`` etc without None-checks.
        """
        return self._compute

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

    def _upload_session_manager(self):
        """Lazy-instantiate the upload session manager for this request.

        Library-internal (issue #20). Tenants don't touch this — they use
        ctx.save_file / save_checkpoint / save_output_stream / save_checkpoint_release
        which route through the manager implicitly.
        """
        if self._upload_sessions is None:
            from ._upload_session import _UploadSessionManager
            base = self._get_file_api_base_url()

            def _hdrs() -> Dict[str, str]:
                token = self._get_worker_capability_token()
                h: Dict[str, str] = {"Authorization": f"Bearer {token}"}
                if self._owner:
                    h["X-Cozy-Owner"] = self._owner
                return h

            self._upload_sessions = _UploadSessionManager(
                base_url=base,
                headers_provider=_hdrs,
                job_id=self._job_id,
            )
        return self._upload_sessions

    def _checkpoint_session_id(self, repo_owner: str, repo_name: str) -> str:
        """Return the session_id for a checkpoint upload session on this
        destination repo, opening it lazily on first call. Cached per ctx —
        subsequent uploads to the same repo reuse the session."""
        mgr = self._upload_session_manager()
        return mgr.session_id_for(
            "checkpoint",
            {"repo_owner": repo_owner, "repo_name": repo_name},
        )

    def _close_upload_sessions(self, *, abort_open: bool = True) -> None:
        """Called at request end (by worker.py). Aborts any still-open
        sessions — a finalize call removes the session from the cache so
        this is a no-op for successfully-finalized requests."""
        if self._upload_sessions is not None:
            self._upload_sessions.close_all(abort_open=abort_open)
            self._upload_sessions = None

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
            logger.debug("_should_stream_output_to_file_api: local path resolve failed for ref=%r", ref, exc_info=True)
            return False
        try:
            _ = self._get_file_api_base_url()
            _ = self._get_worker_capability_token()
        except Exception:
            logger.debug("_should_stream_output_to_file_api: file_api base or capability token unavailable", exc_info=True)
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

        # Scope resolves whenever destination_repo + job_id are present.
        # Previously gated on kind=="training", which broke publish for
        # @inference_function clone jobs that still emit checkpoints.
        hints = dict(self._execution_hints or {})
        destination_repo = _resolve_hint_first_string(hints, keys=_HINT_KEYS_DESTINATION_REPO)
        if destination_repo == "":
            return None
        job_id = _resolve_hint_first_string(hints, keys=_HINT_KEYS_JOB_ID, fallback=self._job_id)
        if job_id == "":
            return None
        try:
            owner, repo = _parse_owner_repo(destination_repo)
        except Exception:
            logger.debug("_repo_job_upload_scope: destination_repo=%r did not parse as owner/repo", destination_repo, exc_info=True)
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
        if kind != "training":
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

    # Reserved-name conversion/training contract. `source` and `destination`
    # come from the job payload's reserved fields (the Worker extracts and
    # populates them before invoking tenant code). `source_path` is populated
    # by the library after it materializes the source snapshot to local disk.
    # Tenant code reads these; writes happen only inside the library.
    @property
    def source(self) -> Dict[str, Any]:
        """Echoed source descriptor: {ref, variant_id?, attributes}. Empty dict
        if this isn't a conversion/training job or the payload didn't supply
        the reserved `source` field."""
        return dict(self._source_info)

    @property
    def source_path(self) -> Optional[str]:
        """Local path to the materialized source snapshot. Populated by the
        library after resolve+download; None before materialization or when
        this isn't a conversion/training job."""
        return self._source_path

    @property
    def destination(self) -> Dict[str, Any]:
        """Echoed destination descriptor: {ref, tags}. Empty dict if this isn't
        a conversion/training job or the payload didn't supply the reserved
        `destination` field."""
        return dict(self._destination_info)

    def _set_source_path(self, path: str) -> None:
        """Library-internal: called by Worker after the source snapshot has
        been materialized locally. Tenant code must not call this."""
        self._source_path = str(path) if path else None

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

    def save_checkpoint(
        self,
        ref: str,
        local_path: str,
        format: Optional[str] = None,
        *,
        produced_by_kind: Optional[str] = None,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
        output_kind: Optional[str] = None,
        target_dtype: Optional[str] = None,
        variant_label: Optional[str] = None,
    ) -> Tensors:
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

        # Job-scoped writes go through the repo-CAS upload-session stream so
        # the returned Tensors carries a blake3 digest + blob_digest + the
        # session tracks completed_files for the finalize manifest. (Earlier
        # media-route fallback was a workaround for tensorhub's
        # `handleOpenUploadSession` rejecting cap-token callers — that server
        # bug is fixed now; media-route is no longer needed, and it also
        # broke `_build_snapshot_manifest` because save_file returns an Asset
        # without blake3 and the clone pipeline's manifest builder requires a
        # blake3 digest per entry.)
        if self._repo_job_upload_scope() is not None and self._should_stream_output_to_file_api(ref):
            stream = self.open_checkpoint_stream(
                ref,
                format=fmt,
                expected_size_bytes=size,
                produced_by_kind=produced_by_kind,
                step_number=step_number,
                epoch_number=epoch_number,
                output_kind=output_kind,
                target_dtype=target_dtype,
                variant_label=variant_label,
            )
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

    def save_checkpoint_bytes(
        self,
        ref: str,
        data: bytes,
        format: Optional[str] = None,
        *,
        produced_by_kind: Optional[str] = None,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
        output_kind: Optional[str] = None,
        target_dtype: Optional[str] = None,
        variant_label: Optional[str] = None,
    ) -> Tensors:
        """Save in-memory checkpoint/model-weight bytes."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("save_checkpoint_bytes expects bytes")
        payload = bytes(data)
        _enforce_output_file_size_limit(len(payload))
        ref = _normalize_output_ref(ref)
        self._require_repo_job_scope_for_tensors(ref)
        fmt = str(format or "").strip() or _infer_tensors_format(ref)

        # Job-scoped writes go through the repo-CAS upload-session stream;
        # see save_checkpoint for the rationale.
        if self._repo_job_upload_scope() is not None and self._should_stream_output_to_file_api(ref):
            stream = self.open_checkpoint_stream(
                ref,
                format=fmt,
                expected_size_bytes=len(payload),
                produced_by_kind=produced_by_kind,
                step_number=step_number,
                epoch_number=epoch_number,
                output_kind=output_kind,
                target_dtype=target_dtype,
                variant_label=variant_label,
            )
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
        """Open a chunk-writable output stream that finalizes to an Asset.

        Public API name (issue #20): prefer ``ctx.save_output_stream(...)``
        — same behavior, the preferred name for consistency with other
        save_* methods.
        """
        return _RequestOutputStream(
            ctx=self,
            ref=ref,
            kind="asset",
            create=create,
            expected_size_bytes=expected_size_bytes,
        )

    def save_output_stream(
        self,
        ref: str,
        *,
        create: bool = False,
        expected_size_bytes: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to an Asset.

        Issue #20: preferred name for streaming writes, consistent with
        other ``ctx.save_*`` methods. Alias for
        ``ctx.open_output_stream(...)``.
        """
        return self.open_output_stream(
            ref, create=create, expected_size_bytes=expected_size_bytes,
        )

    def open_checkpoint_stream(
        self,
        ref: str,
        *,
        format: Optional[str] = None,
        expected_size_bytes: Optional[int] = None,
        produced_by_kind: Optional[str] = None,
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
        output_kind: Optional[str] = None,
        target_dtype: Optional[str] = None,
        variant_label: Optional[str] = None,
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
            produced_by_kind=produced_by_kind,
            step_number=step_number,
            epoch_number=epoch_number,
            output_kind=output_kind,
            target_dtype=target_dtype,
            variant_label=variant_label,
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

    # ------------------------------------------------------------------
    # Visibility controls (issue #20). `publish` means "make publicly
    # available" — the new semantics after the terminology cleanup.
    # ------------------------------------------------------------------

    def publish_checkpoint(self, destination_repo: str, checkpoint_id: str) -> Dict[str, Any]:
        """Flip a finalized checkpoint's visibility to 'public'. Idempotent.

        `destination_repo` is 'owner/repo'; `checkpoint_id` is the content-addressed
        digest returned by finalize. Hits POST /api/v1/repos/:owner/:repo/checkpoints/:id/publish.
        """
        owner, repo = _parse_owner_repo(destination_repo)
        cid = str(checkpoint_id or "").strip()
        if not cid:
            raise ValueError("checkpoint_id is required")
        base = self._get_file_api_base_url()
        token = self._get_worker_capability_token()
        url = (
            f"{base}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(repo, safe='')}/checkpoints/{urllib.parse.quote(cid, safe='')}/publish"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if self._owner:
            headers["X-Cozy-Owner"] = self._owner
        resp = requests.post(url, headers=headers, data=b"{}", timeout=30)
        if resp.status_code in (401, 403):
            raise AuthError(f"publish_checkpoint unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"publish_checkpoint failed ({resp.status_code}): {resp.text[:256]}")
        return resp.json() if resp.text else {"ok": True}

    def unpublish_checkpoint(self, destination_repo: str, checkpoint_id: str) -> Dict[str, Any]:
        """Flip a public checkpoint back to 'private'. Idempotent."""
        owner, repo = _parse_owner_repo(destination_repo)
        cid = str(checkpoint_id or "").strip()
        if not cid:
            raise ValueError("checkpoint_id is required")
        base = self._get_file_api_base_url()
        token = self._get_worker_capability_token()
        url = (
            f"{base}/api/v1/repos/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(repo, safe='')}/checkpoints/{urllib.parse.quote(cid, safe='')}/unpublish"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if self._owner:
            headers["X-Cozy-Owner"] = self._owner
        resp = requests.post(url, headers=headers, data=b"{}", timeout=30)
        if resp.status_code in (401, 403):
            raise AuthError(f"unpublish_checkpoint unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"unpublish_checkpoint failed ({resp.status_code}): {resp.text[:256]}")
        return resp.json() if resp.text else {"ok": True}

    # Issue #21: per-primitive publish/unpublish for datasets, endpoints,
    # endpoint source-code releases, and media. All route to matching
    # tensorhub endpoints with uniform semantics (empty body; idempotent;
    # returns ok + new visibility).

    def _visibility_flip(self, op: str, url: str) -> Dict[str, Any]:
        token = self._get_worker_capability_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if self._owner:
            headers["X-Cozy-Owner"] = self._owner
        resp = requests.post(url, headers=headers, data=b"{}", timeout=30)
        if resp.status_code in (401, 403):
            raise AuthError(f"{op} unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"{op} failed ({resp.status_code}): {resp.text[:256]}")
        return resp.json() if resp.text else {"ok": True}

    def publish_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Flip a dataset's visibility to 'public'. Idempotent."""
        did = str(dataset_id or "").strip()
        if not did:
            raise ValueError("dataset_id is required")
        base = self._get_file_api_base_url()
        url = f"{base}/api/v1/datasets/{urllib.parse.quote(did, safe='')}/publish"
        return self._visibility_flip("publish_dataset", url)

    def unpublish_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Flip a dataset's visibility back to 'private'. Idempotent."""
        did = str(dataset_id or "").strip()
        if not did:
            raise ValueError("dataset_id is required")
        base = self._get_file_api_base_url()
        url = f"{base}/api/v1/datasets/{urllib.parse.quote(did, safe='')}/unpublish"
        return self._visibility_flip("unpublish_dataset", url)

    def publish_endpoint(self, owner: str, endpoint_name: str) -> Dict[str, Any]:
        """Flip an endpoint's endpoint-level visibility to 'public' (anyone can call/list). Idempotent."""
        own = str(owner or "").strip()
        name = str(endpoint_name or "").strip()
        if not own or not name:
            raise ValueError("owner and endpoint_name are required")
        base = self._get_file_api_base_url()
        url = (
            f"{base}/api/v1/endpoints/{urllib.parse.quote(own, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}/publish"
        )
        return self._visibility_flip("publish_endpoint", url)

    def unpublish_endpoint(self, owner: str, endpoint_name: str) -> Dict[str, Any]:
        """Flip an endpoint's endpoint-level visibility back to 'private'. Idempotent."""
        own = str(owner or "").strip()
        name = str(endpoint_name or "").strip()
        if not own or not name:
            raise ValueError("owner and endpoint_name are required")
        base = self._get_file_api_base_url()
        url = (
            f"{base}/api/v1/endpoints/{urllib.parse.quote(own, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}/unpublish"
        )
        return self._visibility_flip("unpublish_endpoint", url)

    def publish_endpoint_release(
        self, owner: str, endpoint_name: str, release_id: str
    ) -> Dict[str, Any]:
        """Flip an endpoint release's source-code visibility to 'public' (bundle publicly readable). Idempotent.

        Orthogonal to publish_endpoint — an endpoint can be publicly callable
        while individual release sources stay private, or vice versa.
        """
        own = str(owner or "").strip()
        name = str(endpoint_name or "").strip()
        rid = str(release_id or "").strip()
        if not own or not name or not rid:
            raise ValueError("owner, endpoint_name, and release_id are required")
        base = self._get_file_api_base_url()
        url = (
            f"{base}/api/v1/endpoints/{urllib.parse.quote(own, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}/releases/"
            f"{urllib.parse.quote(rid, safe='')}/publish"
        )
        return self._visibility_flip("publish_endpoint_release", url)

    def unpublish_endpoint_release(
        self, owner: str, endpoint_name: str, release_id: str
    ) -> Dict[str, Any]:
        """Flip an endpoint release's source-code visibility back to 'private'. Idempotent."""
        own = str(owner or "").strip()
        name = str(endpoint_name or "").strip()
        rid = str(release_id or "").strip()
        if not own or not name or not rid:
            raise ValueError("owner, endpoint_name, and release_id are required")
        base = self._get_file_api_base_url()
        url = (
            f"{base}/api/v1/endpoints/{urllib.parse.quote(own, safe='')}/"
            f"{urllib.parse.quote(name, safe='')}/releases/"
            f"{urllib.parse.quote(rid, safe='')}/unpublish"
        )
        return self._visibility_flip("unpublish_endpoint_release", url)

    def publish_media(self, media_id: str) -> Dict[str, Any]:
        """Flip a media asset's visibility to 'public'. Idempotent."""
        mid = str(media_id or "").strip()
        if not mid:
            raise ValueError("media_id is required")
        base = self._get_file_api_base_url()
        url = f"{base}/api/v1/media/{urllib.parse.quote(mid, safe='')}/publish"
        return self._visibility_flip("publish_media", url)

    def unpublish_media(self, media_id: str) -> Dict[str, Any]:
        """Flip a media asset's visibility back to 'private'. Idempotent."""
        mid = str(media_id or "").strip()
        if not mid:
            raise ValueError("media_id is required")
        base = self._get_file_api_base_url()
        url = f"{base}/api/v1/media/{urllib.parse.quote(mid, safe='')}/unpublish"
        return self._visibility_flip("unpublish_media", url)

    def finalize_checkpoints(
        self,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Issue #20: preferred name for the catalog-commit operation.

        Forwards to ``publish_repo_revision`` (which now routes to
        ``POST /upload-sessions/:session_id/finalize`` internally via the
        session manager). Kept as an alias so tenant code can adopt the
        new terminology at their pace; ``publish_repo_revision`` remains
        functional for back-compat within this module.
        """
        return self.publish_repo_revision(**kwargs)

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
        # HARD-CUT issue #14: the new /publish endpoint shape takes a list of
        # checkpoints + per-checkpoint lineage + a tags map. Existing callers
        # (clone_pipeline.py) still pass the legacy args; we translate them
        # into the new shape here so the wire format is always current.
        relationship_kind: str = "import",
        release_visibility: str = "private",
        auto_create_external_parent: bool = True,
        merge_with_existing: bool = True,
    ) -> Dict[str, Any]:
        """Publish checkpoints + lineage to Tensorhub via the worker-cap /publish endpoint.

        Every entry in metadata['output_variants'] becomes one row in
        tensorhub.repo_checkpoints (checkpoint_id IS the snapshot digest).
        `source_repo` + `source_version_id` (if set) produce one lineage
        edge per checkpoint with relationship_kind. Tags in
        `destination_repo_tags` all get pointed at the first checkpoint in
        the list (typical case: one output_spec with tag='prod').

        For imports (clone_huggingface), pass source_repo as
        'external-sources/upstream' and source_version_id as the external
        reference like 'hf:black-forest-labs/FLUX.2-klein-4B'; set
        `relationship_kind='import'` + `auto_create_external_parent=True`.
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
                    "worker_finalize_tags_skipped request_id=%s job_id=%s owner=%s repo=%s reason=missing_target_version tags=%s",
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
            "worker_finalize_attempt request_id=%s job_id=%s owner=%s repo=%s",
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
                if "artifacts" not in variant:
                    variant["artifacts"] = parsed_artifacts
        # Only include output_variants when they have the required fields
        # (version_id, variant_label, etc.). Bare artifact-only variants will
        # fail TensorHub validation. When we have no structured variants, the
        # commit relies on output_versions + publish_intent instead.

        # ----- Build the /publish request body (issue #14 shape). -----
        _ = publish_intent
        _ = metrics

        manifest_entries: List[Dict[str, Any]] = []
        if isinstance(snapshot_manifest, dict):
            raw_entries = snapshot_manifest.get("entries")
            if isinstance(raw_entries, list):
                manifest_entries = [e for e in raw_entries if isinstance(e, dict)]

        # Each commit_output_variants entry becomes one checkpoint row. The
        # entry's snapshot_digest IS the checkpoint_id (content-addressed).
        publish_checkpoints: List[Dict[str, Any]] = []
        parent_repo_ref = ""
        if source_repo and str(source_repo).strip():
            parent_repo_ref = str(source_repo).strip()
        parent_checkpoint_id = normalized_source_version_id

        for v in commit_output_variants:
            dtype = str(v.get("quantization") or "").strip()
            file_layout = str(v.get("file_layout") or "").strip()
            file_type = str(v.get("file_type") or "").strip()
            # Mirror the structured fields into the attributes jsonb so
            # endpoint.toml-style `[models] ref + attributes = { dtype = [...] }`
            # selectors still hit a GIN-indexed match in repo_checkpoints.
            attrs: Dict[str, str] = {}
            label = str(v.get("variant_label") or "").strip()
            if label:
                attrs["variant_label"] = label
            if dtype:
                attrs["dtype"] = dtype
            if file_layout:
                attrs["file_layout"] = file_layout
            if file_type:
                attrs["file_type"] = file_type
            lineage: List[Dict[str, Any]] = []
            if parent_repo_ref and parent_checkpoint_id:
                lineage.append({
                    "parent_repo":          parent_repo_ref,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "relationship_kind":    relationship_kind or "import",
                })
            # checkpoint_id is left empty — the server computes it from
            # sha256(canonical(snapshot_manifest.entries)) and echoes the
            # value back. Per-variant snapshot_manifest is taken from the
            # caller's v["snapshot_manifest"] when present; otherwise falls
            # back to the shared top-level manifest_entries (clone path).
            v_manifest = v.get("snapshot_manifest")
            if isinstance(v_manifest, list):
                per_variant_entries = [e for e in v_manifest if isinstance(e, dict)]
            else:
                per_variant_entries = manifest_entries
            supplied_checkpoint_id = str(v.get("snapshot_digest") or "").strip()
            # Issue #22: kind / library / dtype / file_layout / file_type
            # dropped from the finalize body. Server infers them from the
            # uploaded files; client-supplied values are ignored there.
            publish_checkpoints.append({
                "checkpoint_id":     supplied_checkpoint_id,
                "snapshot_manifest": per_variant_entries,
                "attributes":        attrs,
                "display_label":     label,
                "size_bytes":        int(v.get("size_bytes") or 0),
                "lineage":           lineage,
            })

        # Fallback: no commit_output_variants supplied (e.g., a simple one-off
        # publish). Emit exactly one checkpoint from target_version_id +
        # aggregate snapshot_manifest.
        if not publish_checkpoints and normalized_target_version_id:
            lineage: List[Dict[str, Any]] = []
            if parent_repo_ref and parent_checkpoint_id:
                lineage.append({
                    "parent_repo":          parent_repo_ref,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "relationship_kind":    relationship_kind or "import",
                })
            # Issue #22: server-authoritative metadata; no kind/library/dtype
            # /file_layout/file_type in the body.
            publish_checkpoints.append({
                "checkpoint_id":     normalized_target_version_id,
                "snapshot_manifest": manifest_entries,
                "attributes":        {},
                "display_label":     "",
                "size_bytes":        0,
                "lineage":           lineage,
            })

        tags_map: Dict[str, str] = {}
        if publish_checkpoints and normalized_tags:
            primary_checkpoint_id = publish_checkpoints[0]["checkpoint_id"]
            for t in normalized_tags:
                if str(t or "").strip():
                    tags_map[str(t).strip()] = primary_checkpoint_id

        if not publish_checkpoints:
            logger.warning(
                "worker_finalize_skipped request_id=%s job_id=%s owner=%s repo=%s reason=no_checkpoints",
                self.request_id, self.job_id or "", owner, repo,
            )
        else:
            publish_req_body = {
                "release_visibility":          release_visibility or "private",
                "tags":                        tags_map,
                "checkpoints":                 publish_checkpoints,
                "auto_create_external_parent": bool(auto_create_external_parent),
                "merge_with_existing":         bool(merge_with_existing),
            }
            # Issue #20: finalize hits the session-scoped URL. The session
            # was opened lazily on the first save_* call to this repo and
            # cached in the ctx-level manager.
            try:
                session_id = self._checkpoint_session_id(owner, repo)
                _request_json(
                    "POST",
                    f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/upload-sessions/{urllib.parse.quote(session_id, safe='')}/finalize",
                    publish_req_body,
                )
                # Finalize succeeded — drop the session from the manager's
                # cache so close_all at request end doesn't try to abort it.
                mgr = self._upload_session_manager()
                sess = mgr.get("checkpoint", {"repo_owner": owner, "repo_name": repo})
                if sess is not None:
                    # Finalize endpoint already returned; just forget cache.
                    mgr._sessions.pop(sess.scope_key, None)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning(
                    "worker_finalize_catalog_failed request_id=%s job_id=%s owner=%s repo=%s error=%r",
                    self.request_id, self.job_id or "", owner, repo, exc,
                )
                raise

        published_ids = [c["checkpoint_id"] for c in publish_checkpoints]
        logger.info(
            "worker_finalize_succeeded request_id=%s job_id=%s owner=%s repo=%s checkpoints=%d primary=%s",
            self.request_id, self.job_id or "", owner, repo,
            len(published_ids), published_ids[0] if published_ids else "",
        )

        out: Dict[str, Any] = {
            "ok":                 True,
            "owner":              owner,
            "repo":               repo,
            "job_id":             job_id,
            "checkpoint_ids":     published_ids,
            "output_versions":    published_ids,
            "release_visibility": release_visibility or "private",
        }
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
