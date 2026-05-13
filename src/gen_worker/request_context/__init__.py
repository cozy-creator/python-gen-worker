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
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

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
    _assert_token_repo_scope_matches_destination,
    _canonicalize_model_ref_string,
    _decode_unverified_jwt_claims,
    _default_output_prefix,
    _encode_ref_for_url,
    _enforce_output_file_size_limit,
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
    _require_worker_capability_token,
    _resolve_hint_first_string,
    _sha256_file,
    _url_is_blocked,
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
        resolved_repos_by_id: Optional[Dict[str, Any]] = None,
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
        hf_token: str = "",
    ) -> None:
        self._request_id = str(request_id or "").strip()
        self._job_id = str(job_id or "").strip() or None
        self._owner = owner
        self._invoker_id = invoker_id
        self._timeout_ms = timeout_ms
        self._file_api_base_url = (file_api_base_url or "").strip() or None
        self._worker_capability_token = (worker_capability_token or "").strip() or None
        self._hf_token = (hf_token or "").strip()
        self._materialized_input_urls = dict(materialized_input_urls or {})
        self._local_output_dir = (local_output_dir or "").strip() or None
        self._resolved_repos_by_id = resolved_repos_by_id
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
        # reserved `source`/`destination` struct fields.
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

        # Repo fields declared by the ingest pipeline before the first upload
        # session opens. Empty values are omitted and tensorhub keeps/inherits
        # existing repo values.
        self._repo_spec: Dict[str, str] = {}

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

        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")

        return torch.device("cpu")

    @property
    def hf_token(self) -> str:
        """HuggingFace API token, sourced from gen_worker.config.Settings.

        Tenant code calling into `gen_worker.clone` / `gen_worker.conversion`
        helpers passes `hf_token=ctx.hf_token` so the library never reads env
        on the tenant's behalf. Empty string when no token is configured —
        helpers fall back to unauthenticated calls (works for public HF repos).
        """
        return self._hf_token

    def _get_file_api_base_url(self) -> str:
        if not self._file_api_base_url:
            raise RuntimeError(
                "file API base URL is not configured for this request — "
                "Worker did not propagate Settings.tensorhub_public_url"
            )
        return self._file_api_base_url.rstrip("/")

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

            def _repo_spec_provider() -> Dict[str, str]:
                return dict(self._repo_spec)

            self._upload_sessions = _UploadSessionManager(
                base_url=base,
                headers_provider=_hdrs,
                job_id=self._job_id,
                repo_spec_provider=_repo_spec_provider,
            )
        return self._upload_sessions

    def set_repo_spec(
        self,
        *,
        kind: str = "",
        library_name: str = "",
        model_family: str = "",
        class_name: str = "",
        adapter_for: str = "",
    ) -> None:
        """Set destination repo fields for upload sessions opened from this ctx.

        Must be called before the first save_checkpoint / save_file call for
        the destination repo because these fields are sent when the upload
        session opens.
        """
        spec = {
            "kind": kind,
            "library_name": library_name,
            "model_family": model_family,
            "class_name": class_name,
            "adapter_for": adapter_for,
        }
        self._repo_spec = {
            k: str(v or "").strip()
            for k, v in spec.items()
            if str(v or "").strip()
        }

    def _checkpoint_revision_id(self, repo_owner: str, repo_name: str) -> str:
        """Return the revision_id for a checkpoint upload session on this
        destination repo, opening it lazily on first call. Cached per ctx —
        subsequent uploads to the same repo reuse the session.

        Tensorhub still calls the open-session token `session_id` over the
        wire; we surface it locally as `revision_id` because the resource
        it materializes is one repo revision. The two names refer to the
        same opaque value."""
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
    def resolved_repos_by_id(self) -> Optional[Dict[str, Any]]:
        return self._resolved_repos_by_id

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
        """Echoed source descriptor: {ref, checkpoint_id?, attributes}. Empty dict
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
        return f"outputs/{self._request_id}/items/{item_key}/{leaf}"

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

    # Inline-bytes threshold: when the client requested
    # `Prefer: bytes=inline` AND the payload is at or below this many
    # bytes, skip the tensorhub upload and return the bytes directly
    # on the Asset (see Asset.bytes docstring). Default ~1 MiB matches
    # the orchestrator-side default ORCHESTRATOR_OUTPUT_INLINE_MAX_BYTES.
    _SAVE_BYTES_INLINE_THRESHOLD = 4 * 1024 * 1024

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

        # Inline path: client signaled `Prefer: bytes=inline` and the
        # payload fits under the inline threshold. Skip the tensorhub
        # upload entirely — return raw bytes on the Asset and let the
        # orchestrator pass them through to the client. msgpack on the
        # wire keeps the bytes raw (no base64 inflation); JSON clients
        # get them base64-encoded by Go's encoding/json on the way out.
        output_format = str(
            (self._execution_hints or {}).get("output_format", "")
        ).strip().lower()
        if output_format == "inline" and len(data) <= self._SAVE_BYTES_INLINE_THRESHOLD:
            return Asset(
                ref=ref,
                owner=self.owner,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                inline_bytes=data,
            )

        stream = self.open_output_stream(ref, create=False, expected_size_bytes=len(data))
        stream.write(data)
        out = stream.finalize()
        if isinstance(out, Asset):
            return out
        raise RuntimeError("file save failed (invalid_asset_response)")

    def save_image(
        self,
        image: Any,
        ref: Optional[str] = None,
        *,
        format: str = "webp",
        quality: int = 95,
        lossless: bool = False,
    ) -> Asset:
        fmt = str(format or "webp").strip().lower()
        if fmt in {"jpg", "jpeg"}:
            pil_format = "JPEG"
            ext = ".jpg"
        elif fmt == "png":
            pil_format = "PNG"
            ext = ".png"
        elif fmt == "webp":
            pil_format = "WEBP"
            ext = ".webp"
        else:
            raise ValueError("unsupported image format")

        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/image{ext}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += ext

        img = image
        if pil_format == "JPEG" and getattr(img, "mode", "") in {"RGBA", "LA", "P"}:
            img = img.convert("RGB")

        buf = BytesIO()
        save_kwargs: Dict[str, Any] = {}
        if pil_format in {"JPEG", "WEBP"}:
            save_kwargs["quality"] = max(1, min(int(quality), 100))
        if pil_format == "WEBP":
            save_kwargs["lossless"] = bool(lossless)
            save_kwargs["method"] = 6
        img.save(buf, format=pil_format, **save_kwargs)
        return self.save_bytes(ref, buf.getvalue())

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
        flavor: Optional[str] = None,
        attributes: Optional[dict] = None,
    ) -> Tensors:
        """Save checkpoint/model-weight bytes and return a first-class tensor artifact.

        ``attributes`` is a free-form provenance map (e.g. quantization
        library + scheme + group_size + calibration dataset id). Stamped
        onto the output flavor for downstream lineage queries. Threaded
        to the upload session's ``/complete`` payload when streaming via
        repo-CAS; ignored for the legacy media-route fallback path.
        """
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
                flavor=flavor,
                attributes=attributes,
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
        flavor: Optional[str] = None,
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
                flavor=flavor,
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
        flavor: Optional[str] = None,
        attributes: Optional[dict] = None,
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
            flavor=flavor,
            attributes=attributes,
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
        ``POST /revisions/:revision_id/finalize`` internally via the
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
        # HARD-CUT issue #14: the /publish endpoint takes checkpoint_flavors
        # plus per-checkpoint lineage and a tags map.
        relationship_kind: str = "import",
        auto_create_external_parent: bool = True,
    ) -> Dict[str, Any]:
        """Publish checkpoints + lineage to Tensorhub via the worker-cap /publish endpoint.

        Every entry in metadata['checkpoint_flavors'] becomes one concrete
        checkpoint attached to the destination tag group in Tensorhub.
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
        commit_checkpoint_flavors = []
        raw_checkpoint_flavors = md.get("checkpoint_flavors")
        if isinstance(raw_checkpoint_flavors, list):
            for item in raw_checkpoint_flavors:
                if isinstance(item, dict):
                    commit_checkpoint_flavors.append(dict(item))

        # Wire artifact_refs into checkpoint_flavors if not already present.
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
                parsed_artifacts.append({"path": ref.strip()})

        if parsed_artifacts and commit_checkpoint_flavors:
            for flavor_item in commit_checkpoint_flavors:
                if "artifacts" not in flavor_item:
                    flavor_item["artifacts"] = parsed_artifacts

        # ----- Build the /publish request body (issue #14 shape). -----
        _ = publish_intent
        _ = metrics

        manifest_entries: List[Dict[str, Any]] = []
        if isinstance(snapshot_manifest, dict):
            raw_entries = snapshot_manifest.get("entries")
            if isinstance(raw_entries, list):
                manifest_entries = [e for e in raw_entries if isinstance(e, dict)]

        # Each commit_checkpoint_flavors entry becomes one checkpoint row. The
        # server computes checkpoint_id from the per-flavor snapshot_manifest.
        publish_checkpoints: List[Dict[str, Any]] = []
        parent_repo_ref = ""
        if source_repo and str(source_repo).strip():
            parent_repo_ref = str(source_repo).strip()
        parent_checkpoint_id = normalized_source_version_id

        for v in commit_checkpoint_flavors:
            # The server computes checkpoint_id from the per-flavor
            # snapshot_manifest. Checkpoint rows carry only concrete fields and
            # flavor pointers; no arbitrary checkpoint attributes are sent.
            flavor = str(v.get("flavor") or "").strip()
            raw_flavors = v.get("flavors")
            flavors: List[str] = []
            if isinstance(raw_flavors, list):
                for raw_flavor in raw_flavors:
                    item = str(raw_flavor or "").strip()
                    if item and item not in flavors:
                        flavors.append(item)
            if flavor and flavor not in flavors:
                flavors.insert(0, flavor)
            label = str(v.get("display_label") or flavor).strip()
            lineage: List[Dict[str, Any]] = []
            if parent_repo_ref and parent_checkpoint_id:
                edge: Dict[str, Any] = {
                    "parent_repo":          parent_repo_ref,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "relationship_kind":    relationship_kind or "import",
                }
                # Issue #258 task 3: forward per-edge metadata when the
                # caller attached it (e.g. quantization_method +
                # quantization_library for quantization edges). The
                # publish handler in tensorhub validates the shape via
                # ValidateQuantizationLineageMetadata for relationship_kind
                # == "quantization" and rejects with 400 if required
                # fields are missing.
                raw_meta = v.get("lineage_metadata")
                if isinstance(raw_meta, dict) and raw_meta:
                    edge["metadata"] = raw_meta
                lineage.append(edge)
            # Per-flavor snapshot_manifest is taken from the caller's
            # v["snapshot_manifest"] when present; otherwise falls back to
            # the shared top-level manifest_entries (clone path).
            v_manifest = v.get("snapshot_manifest")
            if isinstance(v_manifest, list):
                per_flavor_entries = [e for e in v_manifest if isinstance(e, dict)]
            else:
                per_flavor_entries = manifest_entries
            # Explicit deletions list — paths from the prior :latest checkpoint
            # that the caller wants removed from this revision. The server
            # always merges with :latest, so this is the only way to express
            # overwrite semantics (e.g. clone's overwrite_repo path enumerates
            # the prior manifest and lists every path here).
            v_deletions_raw = v.get("deletions")
            per_flavor_deletions: List[str] = []
            if isinstance(v_deletions_raw, list):
                for d in v_deletions_raw:
                    if isinstance(d, str):
                        s = d.strip()
                        if s:
                            per_flavor_deletions.append(s)
            ck_entry: Dict[str, Any] = {
                "snapshot_manifest": per_flavor_entries,
                "display_label":     label,
                "flavor":            flavor,
                "flavors":           flavors,
                "lineage":           lineage,
            }
            if per_flavor_deletions:
                ck_entry["deletions"] = per_flavor_deletions
            # Per-checkpoint attributes — tensorhub stores these as
            # `repo_checkpoints.attributes` jsonb. Carries intrinsic facts
            # like `size_facts` used by the orchestrator's VRAM-aware
            # placement (gen-orchestrator #320). Tensorhub side must accept
            # the field in the publish payload to persist it.
            raw_attrs = v.get("attributes")
            if isinstance(raw_attrs, dict) and raw_attrs:
                ck_entry["attributes"] = dict(raw_attrs)
            publish_checkpoints.append(ck_entry)

        primary_default_flavor = ""
        if publish_checkpoints:
            primary_default_flavor = str(publish_checkpoints[0].get("flavor") or "").strip()
            if not primary_default_flavor:
                raw_primary_flavors = publish_checkpoints[0].get("flavors")
                if isinstance(raw_primary_flavors, list) and raw_primary_flavors:
                    primary_default_flavor = str(raw_primary_flavors[0] or "").strip()
        # Top-level tags list (renamed from tag_groups). Each entry points
        # at a flavor; the server selects which checkpoint that flavor
        # resolves to. `default_checkpoint_id` is no longer sent —
        # `default_flavor` carries the same information.
        tags: List[Dict[str, Any]] = []
        for tag in normalized_tags:
            clean_tag = str(tag or "").strip()
            if not clean_tag or clean_tag.lower() == "latest":
                continue
            entry: Dict[str, Any] = {"tag": clean_tag}
            if primary_default_flavor:
                entry["default_flavor"] = primary_default_flavor
            tags.append(entry)

        if not publish_checkpoints:
            logger.warning(
                "worker_finalize_skipped request_id=%s job_id=%s owner=%s repo=%s reason=no_checkpoints",
                self.request_id, self.job_id or "", owner, repo,
            )
        else:
            # `merge_with_existing` is no longer sent: the server always
            # merges with :latest. Overwrite semantics are expressed by
            # the caller populating per-checkpoint `deletions` lists; this
            # method passes them through unchanged.
            publish_req_body = {
                "tags":                        tags,
                "checkpoint_flavors":          publish_checkpoints,
                "auto_create_external_parent": bool(auto_create_external_parent),
            }
            # Issue #20: finalize hits the session-scoped URL. The session
            # was opened lazily on the first save_* call to this repo and
            # cached in the ctx-level manager.
            finalize_resp: Dict[str, Any] = {}
            try:
                revision_id = self._checkpoint_revision_id(owner, repo)
                finalize_resp = _request_json(
                    "POST",
                    f"/api/v1/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(repo, safe='')}/revisions/{urllib.parse.quote(revision_id, safe='')}/finalize",
                    publish_req_body,
                ) or {}
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

            # extract server-computed checkpoint_ids from the
            # finalize response so callers (and the post-finalize tag
            # promotion below) can reference them.
            resp_checkpoints = finalize_resp.get("checkpoints")
            if isinstance(resp_checkpoints, list):
                for idx, rc in enumerate(resp_checkpoints):
                    if idx < len(publish_checkpoints) and isinstance(rc, dict):
                        cid = str(rc.get("checkpoint_id") or "").strip()
                        if cid:
                            publish_checkpoints[idx]["checkpoint_id"] = cid

        published_ids = [str(c.get("checkpoint_id") or "") for c in publish_checkpoints]
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
        }
        if normalized_tags:
            out["destination_repo_tags"] = normalized_tags
        return out

    def publish_dataset_revision(
        self,
        *,
        destination_dataset: str,
        features_json: Dict[str, Any],
        row_artifacts_json: Optional[Dict[str, Any]] = None,
        snapshot_manifest: Optional[List[Dict[str, Any]]] = None,
        visibility: str = "private",
        kind: str = "",
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Publish a dataset revision into ``tensorhub.datasets``.

        Parallel to ``publish_repo_revision`` but writes to the datasets
        subsystem instead of ``repo_checkpoints``. The flow:

        1. Resolve ``destination_dataset`` (owner/name) against tensorhub.
        2. If the dataset row doesn't exist: ``POST /api/v1/datasets`` with
           ``{owner, name, visibility, schema: features_json}``.
        3. Otherwise: ``PATCH /api/v1/datasets/:id`` to update the schema
           + row_artifacts_json.

        The individual file bytes are expected to already be in CAS via
        prior ``save_checkpoint`` calls — this method just records the
        dataset-level metadata pointing at those blobs. The server
        cross-references by blob digest at materialize time.

        Args:
            destination_dataset: ``owner/name`` or ``owner/name:tag`` ref.
            features_json: HF-style features schema, e.g.
                ``{"prompt": {"_type": "Value", "dtype": "string"}, ...}``.
            row_artifacts_json: Optional mapping of row IDs → artifact
                refs for datasets that reference external image blobs.
            snapshot_manifest: Optional list of ``{path, digest, size_bytes}``
                entries — the parquet shards + any sidecar files that
                comprise this dataset revision. Used for provenance /
                content-identity tracking (naming-based versioning means
                the dataset row is mutable, but the manifest captures what
                content was active at publish time).
            visibility: ``"private"`` (default) or ``"public"``.
            kind: Free-form kind string (``"prompt_corpus"`` / ``"eval_set"``).
                Stored in features_json.__cozy_kind__ for now until
                tensorhub grows a dedicated kind column.
            dataset_info: Full ``dataset_info.json`` payload to record
                as tenant metadata.

        Returns:
            ``{ok: True, dataset_id: str, owner: str, name: str, existed: bool}``.

        Raises ``RuntimeError`` on HTTP failure. Callers in
        ``_finalize_produced_variants`` wrap with try/except so a failed
        dataset publish doesn't crash the job; the blob uploads already
        landed by that point.
        """
        owner, name = _parse_owner_repo(destination_dataset)
        base = (self._file_api_base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError("publish_dataset_revision: no file_api_base_url")
        token = self._get_worker_capability_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Cozy-Owner": owner,
        }

        # Stash the kind (+ dataset_info if provided) inside features_json
        # using a reserved `__cozy_*__` key so it survives through the
        # server's features_json passthrough. Once tensorhub adds a
        # dedicated `kind` column, migrate these to top-level fields.
        raw_schema = dict(features_json or {})
        if isinstance(raw_schema.get("features"), dict):
            features_payload = dict(raw_schema)
        else:
            features_payload = {"features": raw_schema}
        if kind:
            features_payload["__cozy_kind__"] = kind
        if dataset_info:
            features_payload["__cozy_dataset_info__"] = dataset_info
        if snapshot_manifest:
            features_payload["__cozy_snapshot_manifest__"] = snapshot_manifest

        # Step 1: look up any existing dataset by (owner, name). tensorhub
        # currently lists by owner and filters client-side; this is O(N)
        # but N is small (typically <100 datasets per org in practice).
        list_url = (
            f"{base}/api/v1/datasets?owner={urllib.parse.quote(owner, safe='')}"
        )
        list_resp = requests.get(list_url, headers=headers, timeout=30)
        existing_id = ""
        if 200 <= list_resp.status_code < 300:
            try:
                items = list_resp.json().get("items") or []
                for it in items:
                    if str(it.get("name") or "").lower() == name.lower():
                        existing_id = str(it.get("id") or "")
                        break
            except Exception:
                pass

        if not existing_id:
            # Step 2a: create.
            create_url = f"{base}/api/v1/datasets"
            create_body = {
                "owner": owner,
                "name": name,
                "visibility": visibility,
                "schema": features_payload,
            }
            resp = requests.post(
                create_url,
                headers=headers,
                data=json.dumps(create_body).encode("utf-8"),
                timeout=30,
            )
            if resp.status_code in (401, 403):
                raise AuthError(f"dataset create unauthorized ({resp.status_code})")
            if resp.status_code < 200 or resp.status_code >= 300:
                raise RuntimeError(
                    f"dataset create failed ({resp.status_code}): {resp.text[:256]}"
                )
            data = resp.json() if resp.text else {}
            dataset_id = str(data.get("id") or "")
            return {
                "ok": True,
                "dataset_id": dataset_id,
                "owner": owner,
                "name": name,
                "existed": False,
            }

        # Step 2b: update via PATCH.
        patch_url = f"{base}/api/v1/datasets/{urllib.parse.quote(existing_id, safe='')}"
        patch_body: Dict[str, Any] = {
            "schema": features_payload,
        }
        if row_artifacts_json is not None:
            patch_body["row_artifacts"] = row_artifacts_json
        if visibility in ("private", "public"):
            patch_body["visibility"] = visibility
        resp = requests.patch(
            patch_url,
            headers=headers,
            data=json.dumps(patch_body).encode("utf-8"),
            timeout=30,
        )
        if resp.status_code in (401, 403):
            raise AuthError(f"dataset patch unauthorized ({resp.status_code})")
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(
                f"dataset patch failed ({resp.status_code}): {resp.text[:256]}"
            )
        return {
            "ok": True,
            "dataset_id": existing_id,
            "owner": owner,
            "name": name,
            "existed": True,
        }

    def resolve_dataset(self, ref: str) -> str:
        """Download a dataset by ref into local cache; return the root path.

        Paired with ``publish_dataset_revision``. Flow:

        1. Parse ``ref`` (``owner/name``).
        2. ``GET /api/v1/datasets?owner=<owner>`` → find the dataset row by name.
        3. Read the embedded ``__cozy_snapshot_manifest__`` in features_json
           (set by ``publish_dataset_revision``) to get the list of
           ``{path, digest, size_bytes}`` entries.
        4. For each entry: resolve the blob digest via the repo-CAS download
           machinery (since ``_finalize_dataset_variants`` routed bytes there),
           write into a local cache dir at ``<cache>/datasets/<owner>/<name>/<rel_path>``.
        5. Return the cache dir.

        The full "download via tensorhub's dataset_blobs + dataset CAS"
        flow is more work on the server side (tasks #45.2/.3); for now we
        reuse the repo-CAS path because that's where the publish tenants
        actually uploaded the bytes.

        Raises ``RuntimeError`` when the dataset isn't found, the manifest
        is missing, or any download fails. Callers in `_build_datasets`
        upgrade the error to a helpful message.
        """
        owner, name = _parse_owner_repo(ref)
        base = (self._file_api_base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError(f"resolve_dataset({ref!r}): no file_api_base_url")
        token = self._get_worker_capability_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Cozy-Owner": owner,
        }

        # Step 1-2: look up the row.
        list_url = f"{base}/api/v1/datasets?owner={urllib.parse.quote(owner, safe='')}"
        list_resp = requests.get(list_url, headers=headers, timeout=30)
        if list_resp.status_code in (401, 403):
            raise AuthError(f"dataset lookup unauthorized ({list_resp.status_code})")
        if list_resp.status_code < 200 or list_resp.status_code >= 300:
            raise RuntimeError(f"dataset lookup failed ({list_resp.status_code}): {list_resp.text[:256]}")
        items = list_resp.json().get("items") or []
        row: Optional[Dict[str, Any]] = None
        for it in items:
            if str(it.get("name") or "").lower() == name.lower():
                row = it
                break
        if row is None:
            raise RuntimeError(f"resolve_dataset({ref!r}): dataset not found for owner={owner} name={name}")

        # Step 3: parse embedded snapshot_manifest.
        features = row.get("features") or row.get("schema") or {}
        if not isinstance(features, dict):
            features = {}
        manifest = features.get("__cozy_snapshot_manifest__") or []
        if not isinstance(manifest, list) or not manifest:
            raise RuntimeError(
                f"resolve_dataset({ref!r}): no __cozy_snapshot_manifest__ in features_json. "
                f"Dataset exists but wasn't published via publish_dataset_revision."
            )

        # Step 4: write each blob to a stable cache path. Use the sha-based
        # cache layout so identical content across datasets dedupes.
        import tempfile
        cache_root = Path(tempfile.gettempdir()) / "gen_worker_datasets"
        target_root = cache_root / owner / name
        target_root.mkdir(parents=True, exist_ok=True)

        for entry in manifest:
            if not isinstance(entry, dict):
                continue
            rel_path = str(entry.get("path") or "").strip()
            digest = str(entry.get("digest") or "").strip()
            if not rel_path or not digest:
                continue
            dest_file = target_root / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if dest_file.exists() and dest_file.stat().st_size > 0:
                continue  # already cached
            # Download via the existing blob-fetch path (repo-CAS digest).
            self._download_blob_by_digest(digest, dest_file)

        return str(target_root)

    def _download_blob_by_digest(self, digest: str, dest: Path) -> None:
        """Fetch a blob by ``<algo>:<hex>`` digest to ``dest``.

        Uses the repo-CAS by-digest read endpoint — works for any blob
        uploaded via ``save_checkpoint`` regardless of whether it's a
        checkpoint file or a dataset file. The server indexes all CAS
        content by blake3 digest; callers that know the digest can fetch
        without needing to know which subsystem the blob belongs to.
        """
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = self._get_worker_capability_token()
        # Normalize digest format for URL.
        digest_norm = digest if ":" in digest else f"blake3:{digest}"
        url = f"{base}/api/v1/blobs/{urllib.parse.quote(digest_norm, safe=':')}/content"
        headers = {"Authorization": f"Bearer {token}"}
        with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
            if resp.status_code in (401, 403):
                raise AuthError(f"blob fetch unauthorized ({resp.status_code}) digest={digest}")
            if resp.status_code == 404:
                raise RuntimeError(f"blob fetch 404 for digest={digest}")
            if resp.status_code < 200 or resp.status_code >= 300:
                raise RuntimeError(f"blob fetch failed ({resp.status_code}) digest={digest}: {resp.text[:256]}")
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

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
