"""pgw#1005 A: the SDK upload path must not return a digest it never verified.

`upload_file_with_grant` returned `blake3=blake3_hex` — the value the caller
passed in, having verified nothing — and that claim was forwarded verbatim into
the `/complete` body. Its download twin was fixed and documented long ago
(`download_file_with_grant` calls `verify_file_digest`); the upload side never
got the same treatment, so whether a corrupt SDK upload was caught rested
entirely on the hub re-hashing, and the only client-side pre-flight was a size
check.

Also here: `S3TransferGrant.expires_at` was parsed and never read again.

Every refusal below fires BEFORE any S3 client is constructed, so this test
touches no network; the accept path uses a stub client.
"""

from __future__ import annotations

import datetime as dt

import pytest

from gen_worker import s3_transfer
from gen_worker.api.errors import ArtifactTransferError
from gen_worker.presigned_upload import blake3_hash_file
from gen_worker.s3_transfer import S3TransferGrant, upload_file_with_grant


def rfc3339(delta_s: float) -> str:
    return (dt.datetime.now(dt.timezone.utc)
            + dt.timedelta(seconds=delta_s)).isoformat().replace("+00:00", "Z")


def grant(**over) -> S3TransferGrant:
    raw = {
        "bucket": "repo-cas", "key": "staging/obj",
        "endpoint_url": "https://example.invalid", "region": "auto",
        "access_key_id": "k", "secret_access_key": "s", "session_token": "t",
    }
    raw.update(over)
    return S3TransferGrant.from_mapping(raw)


class _StubClient:
    def __init__(self) -> None:
        self.uploads = []

    def upload_file(self, path, bucket, key, Config=None, Callback=None) -> None:
        self.uploads.append((path, bucket, key))

    def close(self) -> None:
        pass


@pytest.fixture()
def stub(monkeypatch):
    client = _StubClient()
    monkeypatch.setattr(s3_transfer, "_s3_client", lambda g: client)
    return client


def test_the_returned_digest_is_COMPUTED_not_echoed(tmp_path, stub):
    p = tmp_path / "obj.bin"
    p.write_bytes(b"the real bytes" * 100)
    truth = blake3_hash_file(p)

    res = upload_file_with_grant(
        file_path=p, grant=grant(), blake3_hex=truth, size_bytes=p.stat().st_size)

    assert res.blake3 == truth
    assert res.size_bytes == p.stat().st_size
    assert stub.uploads and stub.uploads[0][2] == "staging/obj"


def test_a_claim_the_bytes_do_not_have_is_REFUSED_before_a_byte_moves(tmp_path, stub):
    p = tmp_path / "obj.bin"
    p.write_bytes(b"actual content")

    with pytest.raises(ArtifactTransferError) as err:
        upload_file_with_grant(
            file_path=p, grant=grant(), blake3_hex="ff" * 32,
            size_bytes=p.stat().st_size)

    assert err.value.retryable is False
    assert "refusing to upload under a digest they do not have" in str(err.value)
    assert stub.uploads == [], "nothing may be sent under an unproven claim"


def test_a_caller_with_no_claim_gets_the_computed_one(tmp_path, stub):
    p = tmp_path / "obj.bin"
    p.write_bytes(b"no claim here")
    res = upload_file_with_grant(
        file_path=p, grant=grant(), blake3_hex="", size_bytes=p.stat().st_size)
    assert res.blake3 == blake3_hash_file(p)


def test_a_size_that_changed_under_us_is_still_refused_first(tmp_path, stub):
    p = tmp_path / "obj.bin"
    p.write_bytes(b"short")
    with pytest.raises(ArtifactTransferError) as err:
        upload_file_with_grant(
            file_path=p, grant=grant(), blake3_hex="", size_bytes=999)
    assert "size changed" in str(err.value)
    assert stub.uploads == []


def test_an_EXPIRED_grant_is_refused_as_RETRYABLE_rather_than_attempted(tmp_path, stub):
    """`expires_at` was parsed and never read. A dead scoped credential turns a
    multi-GB upload into a pile of auth failures; the honest answer is "re-mint
    and come back"."""
    p = tmp_path / "obj.bin"
    p.write_bytes(b"payload")

    with pytest.raises(ArtifactTransferError) as err:
        upload_file_with_grant(
            file_path=p, grant=grant(expires_at=rfc3339(-5)), blake3_hex="",
            size_bytes=p.stat().st_size)

    assert err.value.retryable is True
    assert "re-mint the grant" in str(err.value)
    assert stub.uploads == []


def test_a_live_or_unnamed_expiry_uploads_normally(tmp_path, stub):
    p = tmp_path / "obj.bin"
    p.write_bytes(b"payload")
    size = p.stat().st_size
    upload_file_with_grant(file_path=p, grant=grant(expires_at=rfc3339(3600)),
                           blake3_hex="", size_bytes=size)
    upload_file_with_grant(file_path=p, grant=grant(), blake3_hex="",
                           size_bytes=size)
    # An unparseable expiry is "named nothing", never "expired".
    upload_file_with_grant(file_path=p, grant=grant(expires_at="soon"),
                           blake3_hex="", size_bytes=size)
    assert len(stub.uploads) == 3


def test_the_outer_retry_no_longer_multiplies_botocores(tmp_path, monkeypatch):
    """4 outer × 10 botocore was up to forty attempts per part and four full
    re-transfers of the object, because each outer attempt re-uploads from
    zero."""
    assert s3_transfer._SDK_TRANSFER_ATTEMPTS == 2

    attempts = []

    class _Flaky(_StubClient):
        def upload_file(self, *a, **kw):
            attempts.append(1)
            raise RuntimeError("connection reset")

    monkeypatch.setattr(s3_transfer, "_s3_client", lambda g: _Flaky())
    monkeypatch.setattr(s3_transfer.time, "sleep", lambda s: None)
    p = tmp_path / "obj.bin"
    p.write_bytes(b"payload")

    with pytest.raises(ArtifactTransferError) as err:
        upload_file_with_grant(file_path=p, grant=grant(), blake3_hex="",
                               size_bytes=p.stat().st_size)

    assert err.value.retryable is True
    assert len(attempts) == 2
