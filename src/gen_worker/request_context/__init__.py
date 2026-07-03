from __future__ import annotations

import hashlib
import json
import logging
import os
import base64
import random
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional

if TYPE_CHECKING:  # heavy deps stay import-time-free; methods import lazily
    import torch

from ..api.errors import AuthError
from ..api.types import Asset, Compute, Tensors


def _default_compute() -> Compute:
    """Sentinel Compute used when the orchestrator didn't attach resolved_compute.

    Tenants can safely read ``ctx.compute.vram_gb`` etc. without None-checks;
    zero / empty values are the "not specified" signal.
    """
    return Compute()


def _copy_context_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _copy_context_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_context_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_context_metadata(v) for v in value)
    return value


logger = logging.getLogger(__name__)

_REPO_REVISION_FINALIZE_REQUEST_TIMEOUT_S = 30
_REPO_REVISION_FINALIZE_POLL_MAX_S = 30 * 60
_REPO_REVISION_FINALIZE_POLL_INITIAL_S = 1.0
_REPO_REVISION_FINALIZE_POLL_MAX_DELAY_S = 10.0

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
    _encode_ref_for_url,
    _enforce_output_file_size_limit,
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
        execution_hints: Optional[Dict[str, Any]] = None,
        parent_request_id: Optional[str] = None,
        child_request_id: Optional[str] = None,
        item_id: Optional[str] = None,
        item_index: Optional[int] = None,
        item_span: Optional[Dict[str, int]] = None,
        source_info: Optional[Dict[str, Any]] = None,
        destination_info: Optional[Dict[str, Any]] = None,
        compute: Optional["Compute"] = None,
        models: Optional[Dict[str, Any]] = None,
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
        self._models = _copy_context_metadata(models or {})

        # Upload-session manager (issue #20). Lazy-created on first
        # ctx.save_file / ctx.save_checkpoint / ctx.save_output_stream use.
        # Tenant-invisible; library-internal machinery for the session
        # lifecycle (open → per-file uploads → finalize).
        self._upload_sessions = None  # type: Optional["_UploadSessionManager"]
        # Guards lazy-init of `_upload_sessions`. Per-file upload threads
        # can race on the first save_* call when the manager hasn't been
        # instantiated yet.
        # The manager itself is already thread-safe internally.
        self._upload_sessions_lock = threading.Lock()

        # Capability-budget gate (issue #269 back-pressure). Lazy-built from
        # the worker_capability_token's max_total_bytes + max_bytes_per_file
        # claims on first upload. The pool's per-file fan-out can over-commit
        # if multiple 30+ GiB shards run in parallel; the gate blocks new
        # reservations until in-flight bytes fit the aggregate budget.
        self._upload_budget_gate = None  # type: Optional["BudgetGate"]
        self._upload_budget_gate_lock = threading.Lock()

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
    def models(self) -> Dict[str, Any]:
        """Resolved model bindings for this invocation.

        Read-only metadata resolved by the orchestrator from endpoint defaults
        plus request-time ``_models`` overrides. The tenant input payload never
        contains the reserved ``_models`` envelope; use this property when a
        handler needs to inspect the effective base refs or LoRA attachments.
        """
        return _copy_context_metadata(self._models)

    @property
    def device(self) -> "torch.device":
        """Torch device for this worker runtime (e.g. cuda:0 or cpu).

        Tenant functions should prefer `ctx.device` over choosing a device
        themselves so the platform can standardize device selection.
        """
        try:
            import torch
        except Exception:
            raise RuntimeError("torch is not available in this runtime") from None

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

    def _get_upload_budget_gate(self):
        """Lazy-construct the capability-budget gate from the JWT claims.

        Pure pass-through when the token has no budget claims (dev/test
        paths). See ``_concurrent_upload.BudgetGate`` for semantics.
        """
        if self._upload_budget_gate is None:
            with self._upload_budget_gate_lock:
                if self._upload_budget_gate is None:
                    from ._concurrent_upload import budget_gate_from_capability_jwt
                    token = self._get_worker_capability_token() or ""
                    self._upload_budget_gate = budget_gate_from_capability_jwt(token)
        return self._upload_budget_gate

    def _upload_session_manager(self):
        """Lazy-instantiate the upload session manager for this request.

        Library-internal (issue #20). Tenants don't touch this — they use
        ctx.save_file / save_checkpoint / save_output_stream / save_checkpoint_release
        which route through the manager implicitly.
        """
        # Double-checked locking so concurrent save_* threads (issue
        # #269) don't each materialize a manager and lose session
        # caching to each other.
        if self._upload_sessions is None:
            with self._upload_sessions_lock:
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
        adapter_for_checkpoint_group: str = "",
        adapter_for_family: str = "",
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
            "adapter_for_checkpoint_group": adapter_for_checkpoint_group,
            "adapter_for_family": adapter_for_family,
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
        # @inference clone jobs that still emit checkpoints.
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
    def resolved_repos_by_id(self) -> Optional[dict[str, Any]]:
        return self._resolved_repos_by_id

    @property
    def required_models(self) -> list[str]:
        return list(self._required_models)

    @property
    def execution_hints(self) -> dict[str, Any]:
        return dict(self._execution_hints)

    # Reserved-name conversion/training contract. `source` and `destination`
    # come from the job payload's reserved fields (the Worker extracts and
    # populates them before invoking tenant code). `source_path` is populated
    # by the library after it materializes the source snapshot to local disk.
    # Tenant code reads these; writes happen only inside the library.
    @property
    def source(self) -> dict[str, Any]:
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
    def destination(self) -> dict[str, Any]:
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
    def item_span(self) -> dict[str, int]:
        return dict(self._item_span)

    # #321: preferred_batch_size() / prefetch_depth() removed alongside
    # RuntimeBatchingConfigCommand — they only ever read state set by the
    # orchestrator's runtime override, and that producer never landed.

    def time_remaining_s(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.time())

    def is_canceled(self) -> bool:
        """Check if the request was canceled."""
        return self._canceled

    def raise_if_canceled(self, message: str = "request canceled") -> None:
        """Raise ``CanceledError(message)`` if this request has been canceled. No-op otherwise.

        Canonical cancellation idiom — call inside long-running loops.
        """
        if self.is_canceled():
            from ..api.errors import CanceledError
            raise CanceledError(message)

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

    def save_file(self, ref: str, local_path: str | os.PathLike[str]) -> Asset:
        ref = _normalize_output_ref(ref)
        src = str(os.fspath(local_path) if local_path else "").strip()
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
        # Reserve aggregate-bytes budget (issue #269 back-pressure) — held
        # until the upload completes. Reentrant: nested save_file from
        # inside save_checkpoint's non-streaming branch is a no-op for
        # the same thread.
        with self._get_upload_budget_gate().reserve(size):
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
        local_path: str | os.PathLike[str],
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
        library + scheme + group_size + calibration dataset id). Conversion
        dispatchers attach it to the final ``checkpoint_flavors[]`` publish
        payload; per-file repo-CAS ``/complete`` remains parts-only.
        """
        ref = _normalize_output_ref(ref)
        self._require_repo_job_scope_for_tensors(ref)
        src = str(os.fspath(local_path) if local_path else "").strip()
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
        # Reserve aggregate-bytes budget (issue #269 back-pressure). Held
        # across either branch (streaming or save_file fallthrough). Save_file
        # is reentrancy-aware so its inner reserve() is a no-op.
        with self._get_upload_budget_gate().reserve(size):
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

    def save_file_create(self, ref: str, local_path: str | os.PathLike[str]) -> Asset:
        ref = _normalize_output_ref(ref)
        src = str(os.fspath(local_path) if local_path else "").strip()
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
        # Reserve aggregate-bytes budget (issue #269 back-pressure).
        with self._get_upload_budget_gate().reserve(size):
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

    # Issue #1 (slim-request-context): admin-plane visibility toggles
    # (publish_checkpoint / publish_dataset / publish_endpoint /
    # publish_endpoint_release / publish_media + their unpublish_ counterparts)
    # were deleted as a hard cut. They were not used by any worker-author
    # endpoint; visibility flips belong in cozyctl / the tensorhub UI, not on
    # a per-request object.

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
    ) -> dict[str, Any]:
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
        reference. The external-source identifier is an internal tensorhub
        schema encoding (separate from the model-ref wire format) that
        prefixes the upstream provider, e.g. 'hf:black-forest-labs/FLUX.2-klein-4B'
        for huggingface or 'civitai:<id>' for civitai. Set
        `relationship_kind='import'` + `auto_create_external_parent=True`.
        """
        import requests
        owner, repo = _parse_owner_repo(destination_repo)
        normalized_source_version_id = str(source_version_id or "").strip().lower()
        normalized_tags = _normalize_destination_repo_tags(destination_repo_tags)

        md = dict(metadata or {})
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
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "owner": owner, "name": repo}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Cozy-Owner": owner,
        }

        def _parse_retry_after(raw: Any) -> Optional[float]:
            try:
                value = float(str(raw or "").strip())
            except Exception:
                return None
            if value <= 0:
                return None
            return min(value, _REPO_REVISION_FINALIZE_POLL_MAX_DELAY_S)

        def _response_json(resp: requests.Response) -> Dict[str, Any]:
            try:
                parsed = resp.json()
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"ok": True} if not resp.text else {"ok": False, "raw": resp.text}

        def _raise_publish_http_error(code: int, path: str, text: str) -> None:
            if code in (401, 403):
                detail = ""
                try:
                    detail = str((text or "").strip())
                except Exception:
                    detail = ""
                if detail != "":
                    raise AuthError(
                        f"repo publish unauthorized ({code}): check worker_capability_token validity; response={detail[:256]}"
                    )
                raise AuthError(f"repo publish unauthorized ({code}): check worker_capability_token validity")
            raise RuntimeError(f"repo publish request failed ({code}) {path}: {text[:512]}")

        def _poll_repo_revision_finalize(status_path: str, initial_delay_s: Optional[float] = None) -> Dict[str, Any]:
            delay_s = initial_delay_s or _REPO_REVISION_FINALIZE_POLL_INITIAL_S
            deadline = time.monotonic() + _REPO_REVISION_FINALIZE_POLL_MAX_S
            while True:
                if self._cancel_event.is_set():
                    raise RuntimeError("repo finalize canceled while polling")
                sleep_s = max(0.0, min(delay_s, _REPO_REVISION_FINALIZE_POLL_MAX_DELAY_S))
                if sleep_s > 0:
                    time.sleep(sleep_s + random.uniform(0, min(0.25, sleep_s * 0.1)))
                url = f"{base}{status_path}"
                resp = requests.get(url, headers=headers, timeout=_REPO_REVISION_FINALIZE_REQUEST_TIMEOUT_S)
                code = int(resp.status_code)
                if code in (401, 403):
                    _raise_publish_http_error(code, status_path, resp.text or "")
                if code < 200 or code >= 300:
                    raise RuntimeError(f"repo finalize status poll failed ({code}) {status_path}: {resp.text[:512]}")
                parsed = _response_json(resp)
                state = str(parsed.get("state") or "").strip().lower()
                finalize_status = str(parsed.get("finalize_status") or "").strip().lower()
                if state == "finalized" or finalize_status == "succeeded":
                    result = parsed.get("finalize_result")
                    if isinstance(result, dict):
                        return result
                    return parsed
                if state == "failed" or finalize_status == "failed":
                    err = parsed.get("finalize_error")
                    if isinstance(err, dict):
                        err_msg = json.dumps(err, sort_keys=True)
                    else:
                        err_msg = str(err or parsed)
                    raise RuntimeError(f"repo finalize failed: {err_msg[:512]}")
                retry_after = _parse_retry_after(parsed.get("retry_after_seconds"))
                if retry_after is not None:
                    delay_s = retry_after
                else:
                    delay_s = min(delay_s * 1.6, _REPO_REVISION_FINALIZE_POLL_MAX_DELAY_S)
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"repo finalize timed out after {_REPO_REVISION_FINALIZE_POLL_MAX_S}s polling {status_path}"
                    )

        def _status_path_from_location(path: str, location: str) -> str:
            loc = str(location or "").strip()
            if not loc:
                return path[:-len("/finalize")] if path.endswith("/finalize") else path
            parsed = urllib.parse.urlparse(loc)
            if parsed.scheme or parsed.netloc:
                status_path = parsed.path or "/"
                if parsed.query:
                    status_path = f"{status_path}?{parsed.query}"
                return status_path
            return loc

        def _request_repo_revision_finalize(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            deadline = time.monotonic() + _REPO_REVISION_FINALIZE_POLL_MAX_S
            body = json.dumps(payload)
            delay_s = _REPO_REVISION_FINALIZE_POLL_INITIAL_S
            while True:
                url = f"{base}{path}"
                try:
                    resp = requests.post(
                        url,
                        headers=headers,
                        data=body,
                        timeout=_REPO_REVISION_FINALIZE_REQUEST_TIMEOUT_S,
                    )
                except requests.RequestException as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"repo finalize failed (network): {exc}") from exc
                    logger.warning(
                        "worker_finalize_post_retry request_id=%s job_id=%s path=%s error=%r",
                        self.request_id,
                        self.job_id or "",
                        path,
                        exc,
                    )
                    time.sleep(delay_s + random.uniform(0, min(0.25, delay_s * 0.1)))
                    delay_s = min(delay_s * 1.6, _REPO_REVISION_FINALIZE_POLL_MAX_DELAY_S)
                    continue

                code = int(resp.status_code)
                if code == 202:
                    status_path = _status_path_from_location(path, resp.headers.get("Location", ""))
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                    if retry_after is None:
                        try:
                            retry_after = _parse_retry_after(_response_json(resp).get("retry_after_seconds"))
                        except Exception:
                            retry_after = None
                    return _poll_repo_revision_finalize(status_path, retry_after)
                if code in (401, 403):
                    _raise_publish_http_error(code, path, resp.text or "")
                if code >= 500 and time.monotonic() < deadline:
                    logger.warning(
                        "worker_finalize_post_retry request_id=%s job_id=%s path=%s status=%s response=%s",
                        self.request_id,
                        self.job_id or "",
                        path,
                        code,
                        (resp.text or "")[:256],
                    )
                    time.sleep(delay_s + random.uniform(0, min(0.25, delay_s * 0.1)))
                    delay_s = min(delay_s * 1.6, _REPO_REVISION_FINALIZE_POLL_MAX_DELAY_S)
                    continue
                if code < 200 or code >= 300:
                    _raise_publish_http_error(code, path, resp.text or "")
                return _response_json(resp)

        def _request_json(method: str, path: str, payload: Dict[str, Any], *, allow_404: bool = False) -> Dict[str, Any]:
            if method.upper() == "POST" and path.endswith("/finalize"):
                return _request_repo_revision_finalize(path, payload)
            url = f"{base}{path}"
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            code = int(resp.status_code)
            if code in (401, 403):
                _raise_publish_http_error(code, path, resp.text or "")
            if allow_404 and code == 404:
                return {"ok": False, "not_found": True}
            if code < 200 or code >= 300:
                # Create may be idempotent and already exists.
                if path == "/api/v1/repos" and code in (400, 409):
                    return {"ok": True, "already_exists": True}
                _raise_publish_http_error(code, path, resp.text or "")
            return _response_json(resp)

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
            label = str(v.get("display_label") or "").strip()
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
                "flavor":            flavor,
                "flavors":           flavors,
                "lineage":           lineage,
            }
            if label:
                ck_entry["display_label"] = label
            for axis_key in ("dtype", "file_layout", "file_type"):
                axis_value = str(v.get(axis_key) or "").strip()
                if axis_value:
                    ck_entry[axis_key] = axis_value
            if per_flavor_deletions:
                ck_entry["deletions"] = per_flavor_deletions
            # Per-checkpoint metadata — tensorhub stores this as
            # `checkpoints.metadata` jsonb. Carries intrinsic facts like
            # `size_facts` used by the orchestrator's VRAM-aware placement.
            raw_metadata = v.get("metadata")
            if isinstance(raw_metadata, dict) and raw_metadata:
                ck_entry["metadata"] = dict(raw_metadata)
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
            "name":               repo,
            "job_id":             job_id,
            "checkpoint_ids":     published_ids,
            "output_versions":    published_ids,
        }
        if normalized_tags:
            out["destination_repo_tags"] = normalized_tags
        return out


# ---------------------------------------------------------------------------
# Issue #1 (slim-request-context): kind-specific subclasses.
#
# RequestContext is the per-inference base. Conversion, dataset-producing,
# and trainer endpoints get richer subclasses that carry the
# producer-contract RPCs (publish_repo_revision, publish_dataset_revision,
# resolve_dataset, read/write_repo_metadata, materialize_blob).
#
# ConversionContext / DatasetContext / TrainingContext share `_PublisherMixin`
# for the producer-contract HTTP helpers (repo metadata read/write, blob
# fetch + materialization by digest). publish_repo_revision lives on the base
# RequestContext — producer-style @inference handlers (clone) call it too.
# ---------------------------------------------------------------------------


class _PublisherMixin:
    """Producer-contract helpers shared by ConversionContext, DatasetContext
    and TrainingContext: repo metadata read/write, blob fetch by digest, and
    ``materialize_blob``. Always combined with ``RequestContext`` via multiple
    inheritance (so ``self`` has ``_file_api_base_url`` / ``_owner`` /
    ``_get_worker_capability_token``).

    Not a public surface: tenants should never import this directly.
    """

    def _download_blob_by_digest(self, digest: str, dest: Path) -> None:
        """Fetch a blob by ``<algo>:<hex>`` digest to ``dest``.

        Uses the repo-CAS by-digest read endpoint — works for any blob
        uploaded via ``save_checkpoint`` regardless of whether it's a
        checkpoint file or a dataset file. The server indexes all CAS
        content by blake3 digest; callers that know the digest can fetch
        without needing to know which subsystem the blob belongs to.
        """
        import requests
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

    def read_repo_metadata(self, *, destination_repo: str) -> dict[str, Any]:
        """Read repo-level metadata from Tensorhub public HTTP API."""
        import requests
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

    def write_repo_metadata(self, *, destination_repo: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Write repo-level metadata via Tensorhub public HTTP API."""
        import requests
        owner, repo = _parse_owner_repo(destination_repo)
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        base = (self._file_api_base_url or "").strip().rstrip("/")
        token = (self._worker_capability_token or "").strip()
        if token:
            _assert_token_repo_scope_matches_destination(token, owner, repo, required_permissions=["repo-version:create"])
        if not base or not token:
            return {"ok": False, "skipped": True, "reason": "missing_worker_capability_channel", "owner": owner, "name": repo}

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
        return {"ok": True, "owner": owner, "name": repo, "metadata": returned}


    def materialize_blob(self, digest: str, dest: "str | os.PathLike[str]") -> Path:
        """Fetch a blob by ``<algo>:<hex>`` content-addressed digest.

        Returns the ``Path`` the blob was written to. Replacement for the
        private ``_download_blob_by_digest`` — exposed publicly so tenants
        that handle a digest directly (e.g. consuming a snapshot manifest
        emitted by an earlier conversion) can pull the bytes themselves.
        """
        dest_path = Path(os.fspath(dest))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._download_blob_by_digest(digest, dest_path)
        return dest_path


class ConversionContext(_PublisherMixin, RequestContext):
    """RequestContext for ``@conversion(sub_kind="format-conversion")``
    and similar conversion endpoints.

    Carries the producer-contract RPCs needed to publish new repo revisions
    and read/write repo metadata, plus the conversion-helper surface
    (``mktemp``, ``checkpoint_dir``, ``open_output_writer``,
    ``copy_unconverted_components``, ``cancelled``).

    Inference handlers receive ``RequestContext`` instead — they never need
    these methods.
    """

    def __init__(self, *args: Any, source: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Conversion-wrapper state. ``source`` is the resolved input model
        # (``gen_worker.conversion.source.Source``) for tenants that operate
        # on a checkpoint; dataset-generation tenants pass ``source=None``.
        self._source = source
        self._mktemp_root: Optional[Path] = None
        self._open_writers: list[Any] = []

    # ----- producer-contract RPCs -------------------------------------

    # ----- conversion-helper wrapper API ------------------------------
    #
    # Previously lived in ``gen_worker.conversion.context.ConversionContext``
    # as a wrapper around RequestContext. Subclassed in (since both classes
    # share the same name and the wrapper has no per-method state that
    # RequestContext doesn't already track).

    @property
    def cancelled(self) -> bool:
        """Return True if the scheduler has signaled cancellation.

        Same semantics as ``is_canceled()`` — kept under the British
        spelling for back-compat with conversion-endpoint code that read
        ``ctx.cancelled`` before the slim-context refactor.
        """
        return self.is_canceled()

    def mktemp(self) -> Path:
        """Return a job-scoped scratch directory. Contents are NOT persisted.

        Auto-cleaned at job end. Each call returns a fresh subdir so tenants
        can use it as ``out_dir`` for ``model.save_pretrained(ctx.mktemp())``
        without collision.
        """
        if self._mktemp_root is None:
            self._mktemp_root = Path(
                tempfile.mkdtemp(
                    prefix=f"txform-{self.request_id or 'x'}-",
                    dir=tempfile.gettempdir(),
                )
            )
        return Path(tempfile.mkdtemp(dir=str(self._mktemp_root)))

    def checkpoint_dir(self, *, key: str) -> Path:
        """Return a PERSISTENT scratch dir keyed by (job_id, key).

        Survives worker restart — intended for transformers.Trainer.output_dir
        so ``resume_from_checkpoint=True`` can pick up where a preempted job
        left off.
        """
        job_id = self.job_id or self.request_id or "x"
        base = Path(tempfile.gettempdir()) / "txform-persistent" / str(job_id)
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        dir_path = base / safe_key
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def open_output_writer(self) -> Any:
        """Return a fresh ``StreamingWriter`` for one output variant.

        Call once per entry in the tenant's specs list. The returned writer
        is scoped to a unique subdirectory under ``mktemp()``; tenants don't
        pick paths themselves.
        """
        from ..conversion.writer import StreamingWriter

        out_dir = self.mktemp()
        w = StreamingWriter(source=self._source, out_dir=out_dir)
        self._open_writers.append(w)
        return w

    def copy_unconverted_components(
        self,
        source: Any,
        out_dir: "str | os.PathLike[str]",
        *,
        skip: Any = (),
    ) -> None:
        """Copy components from ``source`` -> ``out_dir`` that the tenant didn't produce.

        For tenants that use ``source.as_hf_model() + model.save_pretrained()``
        and want source-layout passthrough for non-touched components. The
        ``skip`` iterable names components the tenant HAS produced (and thus
        should not be overwritten).
        """
        skip_set = set(skip)
        out_dir = Path(os.fspath(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        for comp_name, comp in source.components.items():
            if comp_name in skip_set:
                continue
            dst = out_dir / comp_name
            if dst.exists():
                continue
            shutil.copytree(str(comp.path), str(dst))


class DatasetContext(_PublisherMixin, RequestContext):
    """RequestContext for dataset-producing endpoints
    (``@dataset``).

    Adds ``publish_dataset_revision`` + ``resolve_dataset``.
    """

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
    ) -> dict[str, Any]:
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
        import requests
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
        import requests
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


class TrainingContext(_PublisherMixin, RequestContext):
    """RequestContext for trainer-class endpoints (transformers.Trainer +
    friends).

    Repo-metadata RPCs come from ``_PublisherMixin``.
    ``save_checkpoint`` lives on the base ``RequestContext`` (gated by
    ``_require_repo_job_scope_for_tensors``) because the internal upload
    paths in ``conversion/dispatch.py`` and ``worker.py`` call it via
    ``getattr`` regardless of which subclass the request used.
    """
