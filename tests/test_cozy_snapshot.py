"""Tensorhub CAS snapshot build coordination — digest-poisoning fix (#358).

Before the fix, a FAILED snapshot build left its entry (set event + stale
exception) parked in the builder registry forever: every later request for the
same digest took the waiter path and re-raised the old exception instead of
retrying. Now a failed build is evicted, so the next request rebuilds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker.models.cozy_snapshot import ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef

_DIGEST = "ab" * 32


def _resolved() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest=_DIGEST,
        files=[WorkerResolvedRepoFile(
            path="model.safetensors",
            size_bytes=5,
            blake3="cd" * 32,
            url="http://example.invalid/blob",
        )],
    )


def test_failed_build_is_evicted_and_retry_succeeds(tmp_path: Path, monkeypatch) -> None:
    calls = {"n": 0}

    async def _flaky_download(url: str, dst: Path, expected_size: int, expected_blake3: str, on_bytes=None) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient blob failure")
        dst.write_bytes(b"12345")

    monkeypatch.setattr(snap_mod, "_download_one_file", _flaky_download)
    ref = TensorhubRef(owner="e2e", repo="tiny")

    with pytest.raises(RuntimeError, match="transient blob failure"):
        asyncio.run(ensure_snapshot_async(base_dir=tmp_path, ref=ref, resolved=_resolved()))

    # The digest must not be poisoned: no parked entry, so the retry REBUILDS
    # (pre-fix this re-raised the stale exception without calling the leaf).
    assert _DIGEST not in snap_mod._SNAP_ENTRIES

    out = asyncio.run(ensure_snapshot_async(base_dir=tmp_path, ref=ref, resolved=_resolved()))
    assert calls["n"] == 2
    assert (out / "model.safetensors").read_bytes() == b"12345"

    # Third call is a pure cache hit — the leaf is not touched again.
    out2 = asyncio.run(ensure_snapshot_async(base_dir=tmp_path, ref=ref, resolved=_resolved()))
    assert out2 == out
    assert calls["n"] == 2
