"""Chunk-writable output stream used by RequestContext.

Pulled out of the monolithic request_context module. Forward-ref to
`RequestContext` is string-typed (via `from __future__ import annotations`)
to avoid a circular import between this module and `__init__.py`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import resource
import threading
import time
import tempfile
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from blake3 import blake3

logger = logging.getLogger(__name__)

from ..api.errors import CanceledError
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
        step_number: Optional[int] = None,
        epoch_number: Optional[int] = None,
    ) -> None:

        self._ctx = ctx
        self._ref = _normalize_output_ref(ref)
        self._kind = str(kind or "asset").strip().lower()
        self._format = str(format or "").strip() or None
        self._create = bool(create)
        self._expected_size_bytes = int(expected_size_bytes or 0)
        self._lineage_step_number = step_number if isinstance(step_number, int) else None
        self._lineage_epoch_number = epoch_number if isinstance(epoch_number, int) else None
        if self._expected_size_bytes < 0:
            self._expected_size_bytes = 0
        if self._expected_size_bytes > 0:
            _enforce_output_file_size_limit(self._expected_size_bytes)
        self._stream_remote = bool(self._ctx._should_stream_output_to_file_api(self._ref))
        self._sha = hashlib.sha256()
        # Fan BLAKE3 across cores; `AUTO` (== -1) lets the impl pick. The
        # hasher itself stays thread-confined to the writer; AUTO only
        # affects internal parallelism within a single update() call.
        self._blake3_hasher = blake3(max_threads=blake3.AUTO)
        self._bytes_written = 0
        self._bytes_uploaded = 0
        self._chunks_written = 0
        self._chunks_uploaded = 0
        self._stream_error_class: Optional[str] = None
        self._progress_lock = threading.Lock()
        self._stream_mode = "presigned" if self._stream_remote else "local_fallback"
        self._started_mono = time.monotonic()
        self._progress_interval_s = 0.20
        self._last_progress_emit_mono = self._started_mono
        self._last_progress_mono = self._started_mono
        self._last_progress_uploaded = 0
        self._session_id: Optional[str] = None
        # Split routing by artifact kind (gw#453): checkpoint streams publish
        # via the /commits API when the job carries a destination_repo scope
        # (job-bound create_checkpoint grant, multi-GB per-file caps);
        # asset streams (sample images, media outputs) always ride the media
        # route — that's what the upload_media grant authorizes and caps.
        # Inference jobs have no destination_repo, so scope stays None.
        self._repo_job_scope = (
            self._ctx._repo_job_upload_scope() if self._kind == "checkpoint" else None
        )
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
    def ref(self) -> str:
        return self._ref

    def write(self, data: bytes | bytearray | memoryview) -> int:
        if self._finalized:
            raise RuntimeError("output stream already finalized")
        if self._ctx.cancelled:
            self.close()
            raise CanceledError("canceled")
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
        if self._ctx.cancelled:
            self.close()
            raise CanceledError("canceled")

        assert self._fh is not None
        assert self._tmp_path is not None
        self._fh.flush()
        self._fh.close()
        self._fh = None

        try:
            if self._stream_remote:
                finalize_t0 = time.monotonic()
                try:
                    if self._repo_job_scope is not None:
                        self._result = self._finalize_checkpoint_commit()
                    else:
                        self._result = self._finalize_presigned_upload()
                except CanceledError:
                    raise
                except Exception as exc:
                    self._emit_upload_failed(exc)
                    raise
                self._finalized = True
                self._maybe_emit_progress(
                    stage="stream_finalized",
                    force=True,
                    extra={"finalize_elapsed_s": float(max(time.monotonic() - finalize_t0, 0.0))},
                )
                return self._result
            else:
                if self._kind == "checkpoint":
                    raw = getattr(self._ctx, "save_checkpoint")(
                        self._ref,
                        self._tmp_path,
                        format=self._format,
                        step_number=self._lineage_step_number,
                        epoch_number=self._lineage_epoch_number,
                    )
                else:
                    raw = self._ctx.save_file(
                        self._ref, self._tmp_path, create=self._create
                    )
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
        self._ctx._emit_event("request.upload_progress", payload)

    def _emit_upload_failed(self, exc: Exception) -> None:
        """Typed upload-failure report (gw#471): the phantom-route breakage
        was invisible for a dozen runs because failures only hit worker logs.
        Emitted BEFORE the exception propagates; non-fatal callers (sample
        harvest) emit-and-continue, fatal paths emit-then-raise."""
        payload: Dict[str, Any] = {
            "code": "artifact_upload_failed",
            "kind": "checkpoint" if self._kind == "checkpoint" else "sample",
            "ref": self._ref,
            "error": str(exc)[:500],
            "attempt": 1,
        }
        if self._lineage_step_number is not None:
            payload["step_number"] = int(self._lineage_step_number)
        self._ctx._emit_event("request.warning", payload)

    def _finalize_checkpoint_commit(self) -> Tensors:
        """Publish the buffered checkpoint file as ONE tensorhub commit.

        gw#471: the hub's write API is commit-only (th#514/#515) — operations
        are declared up front with blake3 + size, then presigned parts are
        PUT and the revision finalized. The blake3 is already rolled during
        write(), so the temp file commits directly through the same client
        conversion's publish_flavors uses daily (gen_worker.convert.hub).
        One save_checkpoint == one commit == one finalized repo revision;
        the repo is auto-created server-side under the job's create_repo
        grant on first publish."""
        from ..convert.hub import CommitFile, HubClient

        assert self._tmp_path is not None
        assert self._repo_job_scope is not None
        file_size = os.path.getsize(self._tmp_path)
        if file_size <= 0:
            raise RuntimeError("file save failed (empty file)")
        blake3_hex = self._blake3_hasher.hexdigest()

        repo_owner, repo, _job_id = self._repo_job_scope
        ctx = self._ctx
        client = HubClient(
            base_url=ctx._get_file_api_base_url(),
            token=ctx._get_worker_capability_token(),
            owner=(ctx._owner or "").strip(),
        )
        # Worker-addable provenance stamp fields only (th#606).
        provenance: Dict[str, Any] = {}
        if self._lineage_step_number and self._lineage_step_number > 0:
            provenance["step_number"] = int(self._lineage_step_number)
        if self._lineage_epoch_number and self._lineage_epoch_number > 0:
            provenance["epoch_number"] = int(self._lineage_epoch_number)

        def _part_progress(parts_done: int, total_parts: int, bytes_up: int) -> None:
            with self._progress_lock:
                self._bytes_uploaded = bytes_up
                self._chunks_uploaded = parts_done
            self._maybe_emit_progress(stage="stream_upload")

        result = client.commit(
            destination_repo=f"{repo_owner}/{repo}",
            files=[CommitFile(
                path=self._ref,
                local_path=Path(self._tmp_path),
                size_bytes=int(file_size),
                blake3=blake3_hex,
            )],
            mode="merge",
            message=f"checkpoint {self._ref}",
            provenance=provenance,
            part_progress=_part_progress,
        )
        with self._progress_lock:
            self._bytes_uploaded = int(file_size)

        final = result.response if isinstance(result.response, dict) else {}
        ckpt = final.get("checkpoint") if isinstance(final.get("checkpoint"), dict) else {}
        fmt = str(self._format or "").strip() or _infer_tensors_format(self._ref)
        return Tensors(
            ref=self._ref,
            owner=ctx._owner,
            local_path=None,
            format=fmt,
            size_bytes=int(file_size),
            sha256=self._sha.hexdigest(),
            blake3=blake3_hex,
            blob_digest=f"blake3:{blake3_hex}",
            snapshot_digest=str((ckpt or {}).get("snapshot_digest") or "").strip() or None,
            stream_mode=self.stream_mode,
        )

    def _finalize_presigned_upload(self) -> Any:
        """Hash the buffered temp file, then upload to the MEDIA route via
        presigned S3 multipart. Checkpoint saves with a repo-job scope go
        through `_finalize_checkpoint_commit` instead (gw#453 routing)."""
        from ..presigned_upload import presigned_upload_file

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

        # Build endpoint and create payload.
        create_payload: Dict[str, Any] = {}
        req_id = str(self._ctx.request_id or "").strip()
        if req_id:
            create_payload["request_id"] = req_id
        content_type = self._infer_content_type()
        if content_type and content_type != "application/octet-stream":
            create_payload["content_type"] = content_type

        # Media upload. The URL owner segment MUST be the owner the
        # capability token's upload_media grant is bound to (the token's
        # `tenant` claim: the canonical invoking-org uuid). The
        # dispatch-stamped ctx.owner can be a slug or a destination-repo
        # owner resolving to a DIFFERENT org — tensorhub then finds no
        # matching grant and 403s (J19 run34 sample images). Inference
        # outputs work exactly because URL owner == token-bound owner.
        create_payload["ref"] = self._ref
        job_id = str(self._ctx._job_id or "").strip()
        if job_id:
            create_payload["job_id"] = job_id
        owner = self._ctx._media_upload_owner()
        if not owner:
            raise RuntimeError(
                "file save failed (missing owner): media uploads require ctx.owner"
            )
        headers["X-Cozy-Owner"] = owner
        owner_seg = urllib.parse.quote(owner, safe="")
        endpoint_path = f"/api/v1/media/{owner_seg}/uploads"

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
            on_progress=_progress_cb,
            cancel_check=lambda: self._ctx.cancelled,
            complete_extra=None,
        )

        # Issue #269: sample peak RSS to verify the streaming refactor is
        # actually keeping us bounded. ru_maxrss is in KiB on Linux,
        # bytes on macOS; we report both numbers so neither platform is
        # ambiguous. With per-file streaming, this should stay well under
        # 1 GiB even for 5 GB shards.
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            peak_kib = int(ru.ru_maxrss)  # Linux: KiB; macOS: bytes
            logger.debug(
                "upload_stream peak_resident=%d_kib ref=%s file_size=%d",
                peak_kib, self._ref, int(file_size),
            )
        except Exception:
            pass

        with self._progress_lock:
            self._bytes_uploaded = file_size

        # Build return type from metadata.
        meta = dict(result.meta)
        size = int(meta.get("size_bytes") or file_size)
        sha = str(meta.get("sha256") or "").strip() or self._sha.hexdigest()
        final_ref = str(meta.get("ref") or self._ref).strip() or self._ref
        asset = Asset(
            ref=final_ref,
            owner=self._ctx._owner,
            local_path=None,
            mime_type=str(meta.get("mime_type") or "") or None,
            size_bytes=size,
            sha256=sha,
            blake3=str(meta.get("blake3") or blake3_hex).strip() or None,
            media_id=str(meta.get("media_id") or "").strip() or None,
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

    def __enter__(self) -> "_RequestOutputStream":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.close()
