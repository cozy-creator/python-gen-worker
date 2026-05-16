from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from blake3 import blake3

import gen_worker.models.cozy_snapshot_v2 as cozy_snapshot_v2
from gen_worker.models.cozy_snapshot_v2 import CozySnapshotV2Downloader, _coerce_resolved_model
from gen_worker.models.hub_client import WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef
from gen_worker.s3_transfer import S3TransferResult


def test_coerces_resolved_repo_file_with_transfer_grant() -> None:
    digest = "a" * 64
    resolved = {
        "snapshot_digest": "blake3:" + "b" * 64,
        "entries": [
            {
                "path": "model.safetensors",
                "size_bytes": 123,
                "blake3": digest,
                "transfer_grant": {
                    "endpoint_url": "https://account.r2.cloudflarestorage.com",
                    "bucket": "repo-cas",
                    "key": "repo-cas/blake3/aa/a",
                    "access_key_id": "access",
                    "secret_access_key": "secret",
                    "session_token": "session",
                },
            }
        ],
    }

    repo = _coerce_resolved_model(TensorhubRef(owner="cozy", repo="model"), resolved)

    assert repo.files[0].url is None
    assert repo.files[0].transfer_grant is not None
    assert repo.files[0].transfer_grant["bucket"] == "repo-cas"


def test_ensure_blobs_uses_sdk_download_grant(monkeypatch: Any, tmp_path: Path) -> None:
    content = b"abcdef"
    digest = blake3(content).hexdigest()
    calls: list[dict[str, Any]] = []

    def fake_download_file_with_grant(**kwargs: Any) -> S3TransferResult:
        calls.append(kwargs)
        dest = Path(kwargs["dest_path"])
        dest.write_bytes(content)
        grant = kwargs["grant"]
        return S3TransferResult(
            bucket=grant.bucket,
            key=grant.key,
            size_bytes=len(content),
            blake3=digest,
        )

    monkeypatch.setattr(cozy_snapshot_v2, "download_file_with_grant", fake_download_file_with_grant)

    file = WorkerResolvedRepoFile(
        path="model.safetensors",
        size_bytes=len(content),
        blake3=digest,
        url=None,
        transfer_grant={
            "endpoint_url": "https://account.r2.cloudflarestorage.com",
            "bucket": "repo-cas",
            "key": "repo-cas/blake3/ab/c",
            "access_key_id": "access",
            "secret_access_key": "secret",
            "session_token": "session",
        },
    )

    asyncio.run(CozySnapshotV2Downloader()._ensure_blobs(tmp_path / "blobs", [file]))

    assert len(calls) == 1
    assert calls[0]["expected_size_bytes"] == len(content)
    assert calls[0]["expected_blake3"] == digest
    assert (tmp_path / "blobs" / "blake3" / digest[:2] / digest[2:4] / digest).read_bytes() == content
