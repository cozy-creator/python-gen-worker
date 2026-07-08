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
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional

if TYPE_CHECKING:  # heavy deps stay import-time-free; methods import lazily
    import torch

from ..api.errors import AuthError
from ..api.types import (
    Asset,
    AudioAsset,
    Compute,
    ImageAsset,
    Tensors,
    VideoAsset,
)


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


def _as_asset(asset: Asset, cls: type) -> Any:
    """Re-type a plain Asset as a media Asset subclass (same fields)."""
    kw = {f: getattr(asset, f) for f in asset.__struct_fields__}
    return cls(**kw)


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
    _canonicalize_model_ref_string,
    _decode_unverified_jwt_claims,
    _encode_ref_for_url,
    _enforce_output_file_size_limit,
    _http_request,
    _infer_mime_type,
    _infer_tensors_format,
    _is_private_ip_str,
    _normalize_output_ref,
    _normalize_repo_name,
    _parse_owner_repo,
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
        loras: Optional[Dict[str, Any]] = None,
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
        self._loras = _copy_context_metadata(loras or {})

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

        # GPU-slot lease (#382). Set by the executor for GPU jobs; lets
        # blocking uploads release the GPU slot while they wait on the
        # network. None for CPU jobs and local (CLI) runs.
        self._gpu_slot_lease: Optional[Any] = None

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def deadline(self) -> Optional[float]:
        """Absolute unix-time deadline, or None when the request is unbounded."""
        return self._deadline

    @property
    def models(self) -> Dict[str, Any]:
        """Resolved model refs for this invocation, keyed by slot name."""
        return _copy_context_metadata(self._models)

    @property
    def loras(self) -> Dict[str, Any]:
        """Per-request LoRA overlays riding each model slot (gw#393):
        slot name -> tuple of ``{"ref", "weight"}``. Empty for adapter-free
        requests. The worker applies/removes the adapters around the handler
        call; this surface is read-only metadata."""
        return _copy_context_metadata(self._loras)

    @property
    def device(self) -> "torch.device":
        """Torch device for this worker runtime (e.g. cuda:0 or cpu)."""
        try:
            import torch
        except Exception:
            raise RuntimeError("torch is not available in this runtime") from None
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    def generator(self, seed: Optional[int] = None) -> "torch.Generator":
        """A ``torch.Generator`` on ``ctx.device``, seeded when ``seed`` is set."""
        import torch

        gen = torch.Generator(device=self.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        return gen

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

    # #321: preferred_batch_size() / prefetch_depth() removed alongside
    # RuntimeBatchingConfigCommand — they only ever read state set by the
    # orchestrator's runtime override, and that producer never landed.

    def time_remaining(self) -> Optional[float]:
        """Seconds until the deadline; None when unbounded."""
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.time())

    @property
    def cancelled(self) -> bool:
        """True once the request has been cancelled."""
        return self._canceled

    def raise_if_cancelled(self, message: str = "request cancelled") -> None:
        """Raise ``CanceledError(message)`` if cancelled. No-op otherwise.

        The one cancellation idiom — call inside long-running loops.
        """
        if self._canceled:
            from ..api.errors import CanceledError
            raise CanceledError(message)

    def _cancel(self) -> None:
        """Worker-internal: mark the request as cancelled."""
        if not self._canceled:
            self._canceled = True
            self._cancel_event.set()
            logger.info("request %s marked for cancellation.", self.request_id)

    @contextmanager
    def _gpu_slot_yielded(self):
        """Worker-internal: release the job's GPU slot for the duration of
        blocking non-GPU I/O (blob upload), re-acquiring before returning to
        tenant code (#382). No-op when there is no lease (CPU jobs, local
        runs) or the slot is already yielded (executor freed it post-handler).

        If the job was cancelled while yielded (deadline / CancelJob), the
        re-acquired slot is released again immediately: the executor's final
        release already saw ``held == False`` and skipped, so the balance
        stays exact and the freed slot isn't captured by a dying job.
        """
        lease = self._gpu_slot_lease
        if lease is None or not lease.yield_slot():
            yield
            return
        try:
            yield
        finally:
            lease.reacquire()
            if self._canceled:
                lease.yield_slot()

    def _emit_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Worker-internal: emit a progress/event payload (best-effort)."""
        if not self._emitter:
            logger.debug("emit(%s) dropped: no emitter configured", event_type)
            return
        self._emitter({
            "request_id": self._request_id,
            "type": event_type,
            "payload": payload or {},
            "timestamp": time.time(),
        })

    def progress(self, progress: float, stage: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"progress": progress}
        if stage is not None:
            payload["stage"] = stage
        self._emit_event("request.progress", payload)

    def log(self, message: str, level: str = "info") -> None:
        self._emit_event("request.log", {"message": message, "level": level})

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
                owner=self._owner,
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
                owner=self._owner,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                inline_bytes=data,
            )

        stream = self._open_output_stream(ref, create=False, expected_size_bytes=len(data))
        with self._gpu_slot_yielded():
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
    ) -> ImageAsset:
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
            # method=4 (default): method=6 costs ~2.6x the encode CPU for ~4%
            # smaller files (#382 measurements).
            save_kwargs["method"] = 4
        img.save(buf, format=pil_format, **save_kwargs)
        return _as_asset(self.save_bytes(ref, buf.getvalue()), ImageAsset)

    def save_audio(
        self,
        audio: Any,
        ref: Optional[str] = None,
        *,
        sample_rate: int = 44100,
        format: str = "wav",
    ) -> AudioAsset:
        """Encode + save audio; returns a typed :class:`AudioAsset`.

        ``audio`` is a numpy array (frames[, channels]) or a torch tensor;
        raw ``bytes`` are stored as-is (assumed already encoded).
        """
        fmt = str(format or "wav").strip().lower()
        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/audio.{fmt}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += f".{fmt}"
        if isinstance(audio, (bytes, bytearray)):
            data = bytes(audio)
        else:
            try:
                import numpy as np
                import soundfile as sf
            except ImportError as exc:
                from ..api.errors import ValidationError

                raise ValidationError(
                    "save_audio needs the audio extra: pip install 'gen-worker[audio]'"
                ) from exc
            arr = audio
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T  # (channels, frames) -> (frames, channels)
            buf = BytesIO()
            sf.write(buf, arr, int(sample_rate), format=fmt.upper())
            data = buf.getvalue()
        return _as_asset(self.save_bytes(ref, data), AudioAsset)

    def save_video(
        self,
        video: "bytes | str | os.PathLike[str]",
        ref: Optional[str] = None,
        *,
        format: str = "mp4",
    ) -> VideoAsset:
        """Save an encoded video (bytes or a local file path); returns a
        typed :class:`VideoAsset` with probed container metadata
        (duration_s/fps/width/height/has_audio/sample_rate, best-effort)."""
        fmt = str(format or "mp4").strip().lower()
        if ref is None or str(ref).strip() == "":
            ref = f"outputs/{self.request_id}/video.{fmt}"
        else:
            ref = _normalize_output_ref(str(ref))
            if Path(ref).suffix == "":
                ref += f".{fmt}"
        if isinstance(video, (bytes, bytearray)):
            asset = _as_asset(self.save_bytes(ref, bytes(video)), VideoAsset)
        else:
            asset = _as_asset(self.save_file(ref, video), VideoAsset)
        try:
            from ..io import probe_video

            for key, value in probe_video(
                bytes(video) if isinstance(video, (bytes, bytearray)) else video
            ).items():
                setattr(asset, key, value)
        except Exception:
            logger.debug("save_video: metadata probe failed", exc_info=True)
        return asset


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
                owner=self._owner,
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
            stream = self._open_output_stream(ref, create=False, expected_size_bytes=size)
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

    def _open_output_stream(
        self,
        ref: str,
        *,
        create: bool = False,
        expected_size_bytes: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Library-internal: chunk-writable output stream finalizing to an Asset."""
        return _RequestOutputStream(
            ctx=self,
            ref=ref,
            kind="asset",
            create=create,
            expected_size_bytes=expected_size_bytes,
        )

    def _save_file_create(self, ref: str, local_path: str | os.PathLike[str]) -> Asset:
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
                owner=self._owner,
                local_path=str(dst),
                mime_type=None,
                size_bytes=size,
                sha256=sha,
            )
        # Reserve aggregate-bytes budget (issue #269 back-pressure).
        with self._get_upload_budget_gate().reserve(size):
            stream = self._open_output_stream(ref, create=True, expected_size_bytes=size)
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



# ---------------------------------------------------------------------------
# Issue #1 (slim-request-context): kind-specific subclasses.
#
# RequestContext is the per-inference base. Conversion, dataset-producing,
# and trainer endpoints get richer subclasses that carry the
# producer-contract RPCs (publish_dataset_revision, resolve_dataset,
# read/write_repo_metadata, materialize_blob).
#
# ConversionContext / DatasetContext / TrainingContext share `_PublisherMixin`
# for the producer-contract HTTP helpers (repo metadata read/write, blob
# fetch + materialization by digest). Checkpoint publishing is NOT here:
# producer endpoints call cozy_convert.publish_flavors (the /commits path).
# ---------------------------------------------------------------------------


class _PublisherMixin:
    """Producer-contract helpers shared by ConversionContext, DatasetContext
    and TrainingContext: repo metadata read/write, blob fetch by digest, and
    ``materialize_blob``. Always combined with ``RequestContext`` via multiple
    inheritance (so ``self`` has ``_file_api_base_url`` / ``_owner`` /
    ``_get_worker_capability_token``).

    Not a public surface: tenants should never import this directly.
    """

    @property
    def hf_token(self) -> str:
        """HuggingFace API token for cozy_convert / conversion helpers.

        Empty string when unconfigured — helpers fall back to
        unauthenticated calls (public repos work)."""
        return self._hf_token

    @property
    def compute(self) -> Compute:
        """Resolved hardware for this invocation (read-only)."""
        return self._compute

    # Reserved-name conversion/training contract. `source` and `destination`
    # come from the job payload's reserved fields; `source_path` is populated
    # by the library after it materializes the source snapshot locally.
    @property
    def source(self) -> dict[str, Any]:
        return dict(self._source_info)

    @property
    def source_path(self) -> Optional[str]:
        return self._source_path

    @property
    def destination(self) -> dict[str, Any]:
        return dict(self._destination_info)

    def _set_source_path(self, path: str) -> None:
        """Library-internal: called after source materialization."""
        self._source_path = str(path) if path else None

    def set_repo_spec(
        self,
        *,
        kind: str = "",
        library_name: str = "",
        model_family: str = "",
        class_name: str = "",
        adapter_for_family: str = "",
    ) -> None:
        """Set destination repo fields for upload sessions opened from this ctx.

        Must be called before the first save_checkpoint / save_file call for
        the destination repo (the fields are sent when the session opens).
        """
        spec = {
            "kind": kind,
            "library_name": library_name,
            "model_family": model_family,
            "class_name": class_name,
            "adapter_for_family": adapter_for_family,
        }
        self._repo_spec = {
            k: str(v or "").strip()
            for k, v in spec.items()
            if str(v or "").strip()
        }

    def open_output_stream(
        self,
        ref: str,
        *,
        create: bool = False,
        expected_size_bytes: Optional[int] = None,
    ) -> _RequestOutputStream:
        """Open a chunk-writable output stream that finalizes to an Asset."""
        return self._open_output_stream(
            ref, create=create, expected_size_bytes=expected_size_bytes
        )

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
    (``mktemp``, ``checkpoint_dir``, ``copy_unconverted_components``,
    ``cancelled``). The ETL itself (ingest / cast / quant / clone / writers)
    lives in the ``cozy_convert`` package — this class only carries what the
    worker API needs.

    Inference handlers receive ``RequestContext`` instead — they never need
    these methods.
    """

    def __init__(self, *args: Any, source: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``source`` is the resolved input model handle (a cozy_convert
        # ``Source``) for tenants that operate on a checkpoint; None otherwise.
        self._source = source
        self._mktemp_root: Optional[Path] = None

    # ----- producer-contract RPCs -------------------------------------

    # ----- conversion-helper wrapper API ------------------------------
    #
    # Previously lived in ``gen_worker.conversion.context.ConversionContext``
    # as a wrapper around RequestContext. Subclassed in (since both classes
    # share the same name and the wrapper has no per-method state that
    # RequestContext doesn't already track).

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
        job_id = self._job_id or self.request_id or "x"
        base = Path(tempfile.gettempdir()) / "txform-persistent" / str(job_id)
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        dir_path = base / safe_key
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

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

        Writes to the datasets
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
