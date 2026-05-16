"""SDK-backed S3/R2 transfer helpers for trusted Tensorhub workers."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .api.errors import ArtifactTransferError
from .presigned_upload import blake3_hash_file

_MULTIPART_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_CONCURRENCY = 8


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

    client = _s3_client(grant)
    seen = 0
    lock = threading.Lock()

    def _progress(delta: int) -> None:
        nonlocal seen
        if not on_progress:
            return
        with lock:
            seen += int(delta)
            current = min(seen, actual_size)
        on_progress(1 if current >= actual_size else 0, 1, current)

    try:
        client.upload_file(
            str(path),
            grant.bucket,
            grant.key,
            Config=_transfer_config(),
            Callback=_progress if on_progress else None,
        )
        try:
            head = client.head_object(Bucket=grant.bucket, Key=grant.key)
            etag = str(head.get("ETag") or "").strip()
        except Exception:
            etag = ""
    except Exception as exc:
        raise ArtifactTransferError(
            f"tensorhub SDK upload failed: {exc}",
            provider="tensorhub",
            phase="sdk_upload",
            retryable=True,
            cause_type=type(exc).__name__,
        ) from exc

    return S3TransferResult(bucket=grant.bucket, key=grant.key, size_bytes=actual_size, blake3=blake3_hex, etag=etag)


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
    os.replace(tmp, dest)
    return S3TransferResult(bucket=grant.bucket, key=grant.key, size_bytes=size, blake3=digest)


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
            retries={"mode": "standard", "max_attempts": 5},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _transfer_config() -> Any:
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=_MULTIPART_CHUNK_BYTES,
        multipart_chunksize=_MULTIPART_CHUNK_BYTES,
        max_concurrency=_MAX_CONCURRENCY,
        use_threads=True,
    )
