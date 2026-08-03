"""th#1303 S1: the S3-grant download path refuses an undigestable object.

`s3_transfer.download_file_with_grant` had NO test coverage anywhere in this
repo — found by enumerating the arms S1 deleted and checking each for a guard,
rather than by checking that the guards written happened to pass. It is a
VERIFICATION path, and under th#1346 (R2 computes sha256 on PUT only, and
server-side CopyObject drops it, so a promoted CAS object carries no
store-asserted digest) it is load-bearing: the worker-side check is the only
thing standing behind those bytes.

Before S1 it took `expected_blake3: str = ""`, hashed EVERY downloaded object
with blake3 unconditionally, and then compared only `if expected_blake3:` — so
a v2 caller paid for a full hash and got no check from it, and a caller passing
nothing got no check at all. Both refusals below fire before any S3 client is
constructed, so this test touches no network.
"""

from __future__ import annotations

import pytest

from gen_worker.api.errors import ArtifactTransferError
from gen_worker.s3_transfer import S3TransferGrant, download_file_with_grant


def _grant() -> S3TransferGrant:
    return S3TransferGrant.from_mapping({
        "bucket": "repo-cas", "key": "blobs/sha256/aa/bb/" + "ab" * 32,
        "endpoint_url": "https://example.invalid", "region": "auto",
        "access_key_id": "k", "secret_access_key": "s", "session_token": "t",
    })


def test_grant_download_refuses_an_absent_digest(tmp_path):
    """The vacuous guard, at the grant transport. Deleting the mandatory
    precondition turns this red — and would restore a path that publishes
    bytes nothing checked."""
    with pytest.raises(ArtifactTransferError, match="no expected digest"):
        download_file_with_grant(
            grant=_grant(), dest_path=tmp_path / "x.bin", expected_digest="")
    assert not (tmp_path / "x.bin").exists()


def test_grant_download_refuses_a_whitespace_only_digest(tmp_path):
    """`"   "` is truthy in Python, so a bare `if not expected_digest` would
    let it through and then fail deep inside the hasher with a parse error
    instead of a refusal. The precondition strips before testing."""
    with pytest.raises(ArtifactTransferError, match="no expected digest"):
        download_file_with_grant(
            grant=_grant(), dest_path=tmp_path / "x.bin", expected_digest="   ")


def test_grant_download_signature_requires_the_digest(tmp_path):
    """`expected_digest` is a REQUIRED keyword, not one defaulting to "".

    The default was the whole defect: every existing call site kept compiling
    when the meaning changed. Reinstating `= ""` turns this red.
    """
    import inspect

    sig = inspect.signature(download_file_with_grant)
    param = sig.parameters["expected_digest"]
    assert param.default is inspect.Parameter.empty, (
        "expected_digest must have no default — a defaulted digest is how a "
        "call site silently opts out of verification"
    )
