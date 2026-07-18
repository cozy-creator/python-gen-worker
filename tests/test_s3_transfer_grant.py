"""Unit tests for the SDK-backed Tensorhub/R2 transfer path (issue #19).

`s3_transfer.py` is the trusted-worker upload/download path that replaced
worker model-weight presigned multipart: Tensorhub hands the worker a scoped
temporary R2 credential grant, and the worker transfers bytes through
boto3/s3transfer. These tests drive the REAL `s3_transfer` logic (grant
parsing, BLAKE3 + size validation, retry/backoff with shrinking concurrency,
the process-wide upload budget, atomic download materialization) with only the
boto3 client itself faked — so no network/R2 is required.

The existing budget test (`test_concurrency_semaphore` / the presigned
`test_upload_transport_real_socket`) covers the OLD presigned PUT path; this
file covers the NEW SDK grant path that is now the only model-weight transport.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from gen_worker import s3_transfer
from gen_worker.api.errors import ArtifactTransferError
from gen_worker.presigned_upload import blake3_hash_file


# --------------------------------------------------------------------------
# S3TransferGrant.from_mapping
# --------------------------------------------------------------------------

def _full_grant_mapping_snake() -> dict:
    return {
        "endpoint_url": "https://acct.r2.cloudflarestorage.com",
        "bucket": "tensor-cas",
        "key": "blobs/ab/abcd",
        "access_key_id": "AKIA_TEST",
        "secret_access_key": "secret_test",
        "session_token": "sess_test",
        "region": "weur",
        "expires_at": "2026-01-01T00:00:00Z",
    }


@pytest.mark.parametrize(
    "raw,region",
    [
        (_full_grant_mapping_snake(), "weur"),
        # Tensorhub may serialize either casing; both must parse. Region
        # defaults to "auto" when not supplied.
        (
            {
                "endpointUrl": "https://acct.r2.cloudflarestorage.com",
                "bucket": "tensor-cas",
                "objectKey": "blobs/ab/abcd",
                "accessKeyId": "AKIA_TEST",
                "secretAccessKey": "secret_test",
                "sessionToken": "sess_test",
            },
            "auto",
        ),
    ],
    ids=["snake_case", "camel_case"],
)
def test_grant_from_mapping_casings(raw: dict, region: str) -> None:
    g = s3_transfer.S3TransferGrant.from_mapping(raw)
    assert g.bucket == "tensor-cas"
    assert g.key == "blobs/ab/abcd"
    assert g.access_key_id == "AKIA_TEST"
    assert g.session_token == "sess_test"
    assert g.region == region


@pytest.mark.parametrize("missing", ["endpoint_url", "bucket", "key", "access_key_id", "secret_access_key"])
def test_grant_from_mapping_missing_required_raises(missing: str) -> None:
    raw = _full_grant_mapping_snake()
    del raw[missing]
    with pytest.raises(ArtifactTransferError) as ei:
        s3_transfer.S3TransferGrant.from_mapping(raw)
    assert ei.value.phase == "grant"
    assert ei.value.retryable is False


# --------------------------------------------------------------------------
# Fake boto3 client
# --------------------------------------------------------------------------

class _FakeClient:
    """Records upload/download calls; configurable failure behavior."""

    def __init__(self, *, fail_times: int = 0, download_bytes: bytes | None = None,
                 concurrency_probe: "_ConcurrencyProbe | None" = None) -> None:
        self.fail_times = fail_times
        self.download_bytes = download_bytes
        self.concurrency_probe = concurrency_probe
        self.upload_calls: list[dict] = []
        self.closed = False

    def upload_file(self, filename, bucket, key, Config=None, Callback=None):  # noqa: N803
        self.upload_calls.append({"filename": filename, "bucket": bucket, "key": key,
                                  "max_concurrency": getattr(Config, "max_concurrency", None)})
        if self.concurrency_probe is not None:
            self.concurrency_probe.enter()
            try:
                time.sleep(0.05)
            finally:
                self.concurrency_probe.exit()
        if len(self.upload_calls) <= self.fail_times:
            raise RuntimeError("simulated R2 PUT failure")
        if Callback is not None:
            Callback(1)

    def download_file(self, bucket, key, filename, Config=None):  # noqa: N803
        Path(filename).write_bytes(self.download_bytes or b"")

    def close(self) -> None:
        self.closed = True


class _ConcurrencyProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.max_seen = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep retry tests fast; the budget test installs its own sleep via the probe.
    monkeypatch.setattr(s3_transfer.time, "sleep", lambda *_a, **_k: None)


# --------------------------------------------------------------------------
# upload_file_with_grant
# --------------------------------------------------------------------------

def _grant() -> s3_transfer.S3TransferGrant:
    return s3_transfer.S3TransferGrant.from_mapping(_full_grant_mapping_snake())


def test_upload_success_passes_through_blake3_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "shard.safetensors"
    f.write_bytes(b"x" * 4096)
    fake = _FakeClient()
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    res = s3_transfer.upload_file_with_grant(
        file_path=f, grant=_grant(), blake3_hex="deadbeef", size_bytes=4096)

    assert res.size_bytes == 4096
    assert res.blake3 == "deadbeef"
    assert res.bucket == "tensor-cas"
    assert len(fake.upload_calls) == 1
    assert fake.closed is True  # client is always closed


def test_upload_size_mismatch_is_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "shard.bin"
    f.write_bytes(b"x" * 100)
    fake = _FakeClient()
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    with pytest.raises(ArtifactTransferError) as ei:
        s3_transfer.upload_file_with_grant(
            file_path=f, grant=_grant(), blake3_hex="x", size_bytes=999)
    assert ei.value.retryable is False
    # Must fail before touching the network.
    assert fake.upload_calls == []


def test_upload_retries_then_succeeds_with_shrinking_concurrency(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "shard.bin"
    f.write_bytes(b"y" * 2048)
    fake = _FakeClient(fail_times=2)  # attempts 1,2 fail; attempt 3 succeeds
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    res = s3_transfer.upload_file_with_grant(
        file_path=f, grant=_grant(), blake3_hex="h", size_bytes=2048)

    assert res.blake3 == "h"
    assert len(fake.upload_calls) == 3
    # Concurrency must shrink across attempts (10 -> 5 -> 1) so a retry storm
    # backs off instead of hammering R2 (issue #19).
    seen = [c["max_concurrency"] for c in fake.upload_calls]
    assert seen == sorted(seen, reverse=True)
    assert seen[-1] == 1


def test_upload_exhausts_attempts_raises_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "shard.bin"
    f.write_bytes(b"z" * 512)
    fake = _FakeClient(fail_times=999)
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    with pytest.raises(ArtifactTransferError) as ei:
        s3_transfer.upload_file_with_grant(
            file_path=f, grant=_grant(), blake3_hex="h", size_bytes=512)
    assert ei.value.retryable is True
    assert ei.value.cause_type == "RuntimeError"
    assert len(fake.upload_calls) == s3_transfer._SDK_TRANSFER_ATTEMPTS


def test_sdk_upload_budget_caps_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The process-wide budget must prevent multiplicative file concurrency:
    no more than _SDK_UPLOAD_FILE_BUDGET uploads run inside the client at once,
    even when many threads upload simultaneously (issue #19 acceptance)."""
    probe = _ConcurrencyProbe()
    fake = _FakeClient(concurrency_probe=probe)
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    files = []
    for i in range(8):
        f = tmp_path / f"f{i}.bin"
        f.write_bytes(b"a" * 256)
        files.append(f)

    def _one(p: Path) -> None:
        s3_transfer.upload_file_with_grant(file_path=p, grant=_grant(), blake3_hex="h", size_bytes=256)

    threads = [threading.Thread(target=_one, args=(p,)) for p in files]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert probe.max_seen <= s3_transfer._SDK_UPLOAD_FILE_BUDGET
    assert len(fake.upload_calls) == 8  # all eventually ran


# --------------------------------------------------------------------------
# download_file_with_grant
# --------------------------------------------------------------------------

def test_download_validates_blake3_and_materializes_atomically(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"weights-bytes" * 1000
    src = tmp_path / "expected.bin"
    src.write_bytes(payload)
    expected_hash = blake3_hash_file(src)

    fake = _FakeClient(download_bytes=payload)
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    dest = tmp_path / "out" / "model.bin"
    res = s3_transfer.download_file_with_grant(
        grant=_grant(), dest_path=dest, expected_blake3=expected_hash, expected_size_bytes=len(payload))

    assert dest.exists()
    assert dest.read_bytes() == payload
    assert res.blake3.lower() == expected_hash.lower()
    # No leftover temp file.
    assert not (tmp_path / "out" / "model.bin.tmp").exists()


def test_download_blake3_mismatch_raises_and_cleans_up(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(download_bytes=b"corrupted")
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    dest = tmp_path / "model.bin"
    with pytest.raises(ArtifactTransferError) as ei:
        s3_transfer.download_file_with_grant(
            grant=_grant(), dest_path=dest, expected_blake3="0" * 64)
    assert ei.value.retryable is False
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_download_size_mismatch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(download_bytes=b"short")
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda grant: fake)

    dest = tmp_path / "model.bin"
    with pytest.raises(ArtifactTransferError):
        s3_transfer.download_file_with_grant(
            grant=_grant(), dest_path=dest, expected_size_bytes=99999)
    assert not dest.exists()


# --------------------------------------------------------------------------
# _s3_client R2 compatibility config (issue #19: no unsupported checksum headers)
# --------------------------------------------------------------------------

def test_s3_client_uses_r2_safe_checksum_config() -> None:
    client = s3_transfer._s3_client(_grant())
    cfg = client.meta.config
    # R2 rejects AWS's default per-request checksum trailers; we must only send
    # them when required and only validate responses when required.
    assert cfg.request_checksum_calculation == "when_required"
    assert cfg.response_checksum_validation == "when_required"
    assert cfg.signature_version == "s3v4"
