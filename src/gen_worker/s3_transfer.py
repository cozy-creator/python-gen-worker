"""SDK-backed S3/R2 transfer helpers for trusted Tensorhub workers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .api.errors import ArtifactTransferError
from .presigned_upload import blake3_hash_file

_MULTIPART_CHUNK_BYTES = 64 * 1024 * 1024
_MULTIPART_MAX_WORKERS = 10
_SDK_TRANSFER_ATTEMPTS = 4
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
    blake3: str
    etag: str = ""


def upload_file_with_grant(
    *,
    file_path: str | Path,
    grant: S3TransferGrant,
    blake3_hex: str,
    size_bytes: int,
    on_progress: Optional[Any] = None,
) -> S3TransferResult:
    path = Path(file_path)
    actual_size = int(path.stat().st_size)
    if actual_size != int(size_bytes):
        raise ArtifactTransferError(
            f"local file size changed before upload: expected {size_bytes}, got {actual_size}",
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

    return S3TransferResult(bucket=grant.bucket, key=grant.key, size_bytes=actual_size, blake3=blake3_hex)


def download_file_with_grant(
    *,
    grant: S3TransferGrant,
    dest_path: str | Path,
    expected_blake3: str = "",
    expected_size_bytes: int | None = None,
) -> S3TransferResult:
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    client = _s3_client(grant)
    try:
        client.download_file(grant.bucket, grant.key, str(tmp), Config=_transfer_config())
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
        tmp.unlink(missing_ok=True)
        raise ArtifactTransferError(
            f"downloaded object size mismatch: expected {expected_size_bytes}, got {size}",
            provider="tensorhub",
            phase="sdk_download",
            retryable=False,
        )
    digest = blake3_hash_file(tmp)
    if expected_blake3 and digest.lower() != expected_blake3.lower():
        tmp.unlink(missing_ok=True)
        raise ArtifactTransferError(
            "downloaded object BLAKE3 mismatch",
            provider="tensorhub",
            phase="sdk_download",
            retryable=False,
        )
    # Durable atomic finalize (gw#408): see cozy_cas — data must hit stable
    # storage before the rename, or a pod hard-kill persists a truncated blob.
    from .models.cozy_cas import fsync_dir, fsync_file

    fsync_file(tmp)
    os.replace(tmp, dest)
    fsync_dir(dest.parent)
    return S3TransferResult(bucket=grant.bucket, key=grant.key, size_bytes=size, blake3=digest)


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
    import boto3
    from botocore.config import Config

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
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=_MULTIPART_CHUNK_BYTES,
        multipart_chunksize=_MULTIPART_CHUNK_BYTES,
        max_concurrency=max(1, int(max_concurrency)),
        use_threads=True,
    )
