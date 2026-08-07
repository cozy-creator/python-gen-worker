"""SDK-backed S3/R2 transfer helpers for trusted Tensorhub workers."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .api.errors import ArtifactTransferError
from .models.chunk_cas import DigestMismatch, verify_file_digest
from .models.cozy_cas import fsync_dir, fsync_file
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
import boto3

_MULTIPART_CHUNK_BYTES = 64 * 1024 * 1024
_MULTIPART_MAX_WORKERS = 10
# pgw#1005 A: each OUTER attempt re-uploads every part from zero, on top of
# botocore's own per-part `max_attempts: 10` below — 4 × 10 was up to forty
# attempts per part and four full re-transfers of the object. Two outer
# attempts still cover the case botocore cannot (a client whose credentials or
# connection pool are the problem) without multiplying the transfer budget.
_SDK_TRANSFER_ATTEMPTS = 2
_SDK_UPLOAD_FILE_BUDGET = 2
_sdk_upload_slots = threading.BoundedSemaphore(_SDK_UPLOAD_FILE_BUDGET)


@dataclass(frozen=True)
class S3TransferGrant:
    endpoint_url: str
    bucket: str
    key: str
    access_key_id: str
    secret_access_key: str
    session_token: str = ""
    region: str = "auto"
    expires_at: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "S3TransferGrant":
        def _first(*keys: str) -> str:
            for key in keys:
                value = str(raw.get(key) or "").strip()
                if value:
                    return value
            return ""

        grant = cls(
            endpoint_url=_first("endpoint_url", "endpointUrl"),
            bucket=_first("bucket", "bucket_name", "bucketName"),
            key=_first("key", "object_key", "objectKey"),
            access_key_id=_first("access_key_id", "accessKeyId"),
            secret_access_key=_first("secret_access_key", "secretAccessKey"),
            session_token=_first("session_token", "sessionToken"),
            region=_first("region") or "auto",
            expires_at=_first("expires_at", "expiresAt"),
        )
        missing = [
            name
            for name, value in {
                "endpoint_url": grant.endpoint_url,
                "bucket": grant.bucket,
                "key": grant.key,
                "access_key_id": grant.access_key_id,
                "secret_access_key": grant.secret_access_key,
            }.items()
            if not value
        ]
        if missing:
            raise ArtifactTransferError(
                "tensorhub transfer grant is missing required fields: " + ", ".join(missing),
                provider="tensorhub",
                phase="grant",
                retryable=False,
            )
        return grant


@dataclass(frozen=True)
class S3TransferResult:
    bucket: str
    key: str
    size_bytes: int
    #: LEGACY, upload side only: the bare blake3 the caller declared. The
    #: DOWNLOAD side no longer produces one — it reports ``digest``, which is
    #: algorithm-tagged. This field dies with the media/dataset declare
    #: grammar (th#1303 S1, gated on the user-media + dataset-cas backfills).
    blake3: str = ""
    #: Algorithm-tagged whole-object digest ("sha256:<hex>").
    digest: str = ""
    etag: str = ""


def _grant_expired(grant: S3TransferGrant, *, margin_s: float = 60.0) -> bool:
    """pgw#1005 A: ``S3TransferGrant.expires_at`` was parsed and never read
    again. A scoped credential that is already dead turns a multi-GB upload
    into forty authentication failures; ask the caller to re-mint instead."""
    raw = str(grant.expires_at or "").strip()
    if not raw:
        return False
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc).timestamp() + margin_s >= parsed.timestamp()


def upload_file_with_grant(
    *,
    file_path: str | Path,
    grant: S3TransferGrant,
    blake3_hex: str,
    size_bytes: int,
    on_progress: Optional[Any] = None,
) -> S3TransferResult:
    """Upload one object through a scoped S3 credential, returning a digest
    this function PROVED (pgw#1005 A).

    It used to return ``blake3=blake3_hex`` — the value the caller passed in,
    having verified nothing — and that claim was forwarded verbatim into the
    ``/complete`` body. Its download twin was fixed and documented long ago
    (``download_file_with_grant`` calls ``verify_file_digest``); the upload
    side never got the same treatment, so the only real check was the hub
    re-hashing at ``/complete``.

    Now the local bytes are hashed and compared BEFORE the transfer, and the
    digest reported is the computed one. That is deliberately a check of what
    we SENT, not of what landed: this path is boto3 multipart, where R2 does
    not enforce ``x-amz-checksum-sha256`` on ``UploadPart``, so no write-time
    store enforcement is available (measured — it IS available on
    ``PutObject``, which is why the chunk-CAS path keeps per-chunk presigns
    and gets the strongest guarantee in the platform). What the hub's
    ``/complete`` re-hash then catches is corruption in flight; what this
    catches is the thing the re-hash cannot distinguish from it — a caller
    whose claim never described these bytes in the first place.

    ``etag`` stays "": s3transfer owns the multipart completion and does not
    hand one back. Reading it would cost a HEAD per object for a value nothing
    consumes.
    """
    path = Path(file_path)
    actual_size = int(path.stat().st_size)
    if actual_size != int(size_bytes):
        raise ArtifactTransferError(
            f"local file size changed before upload: expected {size_bytes}, got {actual_size}",
            provider="tensorhub",
            phase="sdk_upload",
            retryable=False,
        )
    if _grant_expired(grant):
        raise ArtifactTransferError(
            f"transfer grant for {grant.key} expires at {grant.expires_at}; "
            "refusing to start an upload that cannot finish — re-mint the grant",
            provider="tensorhub",
            phase="grant",
            retryable=True,
        )
    # Lazy: `presigned_upload` imports this module (inside a function) for the
    # grant path, and the hash helper is not worth a fourth implementation.
    from .presigned_upload import blake3_hash_file

    verified = blake3_hash_file(path)
    claimed = str(blake3_hex or "").strip().lower()
    if claimed and claimed != verified:
        raise ArtifactTransferError(
            f"local bytes for {path.name} hash to {verified[:16]}…, caller claimed "
            f"{claimed[:16]}… — refusing to upload under a digest they do not have",
            provider="tensorhub",
            phase="sdk_upload",
            retryable=False,
        )

    last_exc: Exception | None = None
    for attempt in range(1, _SDK_TRANSFER_ATTEMPTS + 1):
        client = _s3_client(grant)
        try:
            with _sdk_upload_slot():
                client.upload_file(
                    str(path),
                    grant.bucket,
                    grant.key,
                    Config=_transfer_config(max_concurrency=_sdk_workers_for_attempt(attempt)),
                    Callback=_BotoTransferProgress(actual_size, on_progress) if on_progress else None,
                )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= _SDK_TRANSFER_ATTEMPTS:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    if last_exc is not None:
        raise ArtifactTransferError(
            f"tensorhub SDK upload failed: {last_exc}",
            provider="tensorhub",
            phase="sdk_upload",
            retryable=True,
            cause_type=type(last_exc).__name__,
        ) from last_exc

    return S3TransferResult(
        bucket=grant.bucket, key=grant.key, size_bytes=actual_size, blake3=verified)


def download_file_with_grant(
    *,
    grant: S3TransferGrant,
    dest_path: str | Path,
    expected_digest: str,
    expected_size_bytes: int | None = None,
) -> S3TransferResult:
    """th#1303 S1: ``expected_digest`` is ALGORITHM-TAGGED and MANDATORY.

    It used to be ``expected_blake3: str = ""`` — every downloaded object was
    blake3-hashed unconditionally and then compared only ``if expected_blake3``,
    so a v2 caller paid for a full hash and got NO check from it. Now the hash
    dispatches on the ref's own tag and an absent digest is a refusal.
    """
    if not str(expected_digest or "").strip():
        raise ArtifactTransferError(
            "refusing grant download with no expected digest",
            provider="tensorhub",
            phase="sdk_download",
            retryable=False,
        )
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # pgw#938: G compute children share this destination directory.  A fixed
    # ``dest.tmp`` lets one process rename/unlink another process's verified
    # download.  Each writer gets its own same-directory temp, preserving the
    # atomic-finalize contract while making concurrent finalizers independent.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.", suffix=".tmp", dir=str(dest.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        client = _s3_client(grant)
        try:
            client.download_file(
                grant.bucket, grant.key, str(tmp), Config=_transfer_config()
            )
        except Exception as exc:
            raise ArtifactTransferError(
                f"tensorhub SDK download failed: {exc}",
                provider="tensorhub",
                phase="sdk_download",
                retryable=True,
                cause_type=type(exc).__name__,
            ) from exc

        size = int(tmp.stat().st_size)
        if expected_size_bytes is not None and size != int(expected_size_bytes):
            raise ArtifactTransferError(
                f"downloaded object size mismatch: expected {expected_size_bytes}, got {size}",
                provider="tensorhub",
                phase="sdk_download",
                retryable=False,
            )
        try:
            verify_file_digest(tmp, expected_digest)
        except (DigestMismatch, ValueError) as exc:
            raise ArtifactTransferError(
                f"downloaded object digest mismatch: {exc}",
                provider="tensorhub",
                phase="sdk_download",
                retryable=False,
            ) from exc
        # Durable atomic finalize (gw#408): see cozy_cas — data must hit stable
        # storage before the rename, or a pod hard-kill persists a truncated blob.
        fsync_file(tmp)
        os.replace(tmp, dest)
        fsync_dir(dest.parent)
        return S3TransferResult(
            bucket=grant.bucket,
            key=grant.key,
            size_bytes=size,
            digest=expected_digest,
        )
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class _sdk_upload_slot:
    def __enter__(self) -> None:
        _sdk_upload_slots.acquire()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        _sdk_upload_slots.release()


class _BotoTransferProgress:
    def __init__(self, total_bytes: int, on_progress: Any) -> None:
        self._total = max(int(total_bytes), 0)
        self._on_progress = on_progress
        self._seen = 0
        self._lock = threading.Lock()

    def __call__(self, delta: int) -> None:
        with self._lock:
            self._seen = min(self._seen + int(delta), self._total)
            seen = self._seen
        self._on_progress(1 if seen >= self._total else 0, 1, seen)


def _s3_client(grant: S3TransferGrant) -> Any:

    return boto3.client(
        "s3",
        endpoint_url=grant.endpoint_url,
        region_name=grant.region or "auto",
        aws_access_key_id=grant.access_key_id,
        aws_secret_access_key=grant.secret_access_key,
        aws_session_token=grant.session_token or None,
        config=Config(
            signature_version="s3v4",
            retries={"mode": "standard", "max_attempts": 10},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            tcp_keepalive=True,
        ),
    )


def _sdk_workers_for_attempt(attempt: int) -> int:
    if attempt <= 1:
        return _MULTIPART_MAX_WORKERS
    if attempt == 2:
        return max(1, _MULTIPART_MAX_WORKERS // 2)
    return 1


def _transfer_config(*, max_concurrency: int = _MULTIPART_MAX_WORKERS) -> Any:

    return TransferConfig(
        multipart_threshold=_MULTIPART_CHUNK_BYTES,
        multipart_chunksize=_MULTIPART_CHUNK_BYTES,
        max_concurrency=max(1, int(max_concurrency)),
        use_threads=True,
    )
