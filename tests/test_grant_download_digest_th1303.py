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

import hashlib
import multiprocessing
import os
from pathlib import Path

import pytest

from gen_worker.api.errors import ArtifactTransferError
from gen_worker import s3_transfer
from gen_worker.s3_transfer import S3TransferGrant, download_file_with_grant


def _grant() -> S3TransferGrant:
    return S3TransferGrant.from_mapping({
        "bucket": "repo-cas", "key": "blobs/sha256/aa/bb/" + "ab" * 32,
        "endpoint_url": "https://example.invalid", "region": "auto",
        "access_key_id": "k", "secret_access_key": "s", "session_token": "t",
    })


class _BarrierClient:
    def __init__(self, barrier, paths, data: bytes) -> None:
        self.barrier = barrier
        self.paths = paths
        self.data = data

    def download_file(self, _bucket, _key, filename, Config=None) -> None:
        Path(filename).write_bytes(self.data)
        self.paths.put(filename)
        self.barrier.wait(timeout=10.0)


def _concurrent_download(grant, dest, digest, size, results) -> None:
    try:
        download_file_with_grant(
            grant=grant,
            dest_path=dest,
            expected_digest=digest,
            expected_size_bytes=size,
        )
    except BaseException as exc:  # pragma: no cover - reported to parent
        results.put(f"{type(exc).__name__}: {exc}")
    else:
        results.put("")


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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="cross-process race is POSIX-only")
def test_two_processes_finalize_one_grant_download_without_sharing_a_temp(
    tmp_path, monkeypatch,
):
    """pgw#938: two G children may receive the same residency broadcast.

    Both processes enter the real download/verify/fsync/replace path together.
    Their transfer client records the target it was handed; a fixed
    ``dest.tmp`` deterministically fails the distinct-path assertion even when
    scheduling happens to hide the FileNotFoundError race.
    """
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    paths = ctx.Queue()
    results = ctx.Queue()
    data = b"pgw938-concurrent-grant-download" * 1024
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        s3_transfer, "_s3_client", lambda _grant: _BarrierClient(barrier, paths, data)
    )
    dest = tmp_path / "blob.bin"
    procs = [
        ctx.Process(
            target=_concurrent_download,
            args=(_grant(), dest, digest, len(data), results),
        )
        for _ in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(30.0)
        assert proc.exitcode == 0

    errors = [results.get(timeout=2.0) for _ in procs]
    assert errors == ["", ""]
    writer_paths = [paths.get(timeout=2.0) for _ in procs]
    assert len(set(writer_paths)) == 2, writer_paths
    assert dest.read_bytes() == data
    assert not list(tmp_path.glob(".blob.bin.*.tmp"))
