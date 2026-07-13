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
from gen_worker.models.cozy_snapshot import ensure_snapshot_async, snapshot_dir_key
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


def _resolved_pipeline() -> WorkerResolvedRepo:
    """A diffusers-shaped snapshot: root model_index.json + two components."""
    return WorkerResolvedRepo(
        snapshot_digest=_DIGEST,
        files=[
            WorkerResolvedRepoFile(
                path="model_index.json", size_bytes=2, blake3="a1" * 32,
                url="http://example.invalid/model_index.json",
            ),
            WorkerResolvedRepoFile(
                path="vae/diffusion_pytorch_model.safetensors", size_bytes=3, blake3="b2" * 32,
                url="http://example.invalid/vae",
            ),
            WorkerResolvedRepoFile(
                path="unet/diffusion_pytorch_model.safetensors", size_bytes=4, blake3="c3" * 32,
                url="http://example.invalid/unet",
            ),
        ],
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


# --- components= scoped snapshots (pgw#505) ----------------------------------


def _install_fake_downloader(monkeypatch) -> None:
    async def _fake_download(url: str, dst: Path, expected_size: int, expected_blake3: str, on_bytes=None) -> None:
        dst.write_bytes(b"x" * expected_size)

    monkeypatch.setattr(snap_mod, "_download_one_file", _fake_download)


def test_components_narrows_materialized_files_and_keys_directory_separately(
    tmp_path: Path, monkeypatch,
) -> None:
    _install_fake_downloader(monkeypatch)
    ref = TensorhubRef(owner="e2e", repo="sdxl-full")

    out = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path, ref=ref, resolved=_resolved_pipeline(), components=("vae",),
    ))

    assert out.name == snapshot_dir_key(_DIGEST, ("vae",))
    assert out.name != _DIGEST  # never aliases the full-repo directory name
    assert (out / "model_index.json").exists()   # root config always kept
    assert (out / "vae" / "diffusion_pytorch_model.safetensors").exists()
    assert not (out / "unet").exists()            # narrowed away


def test_components_matching_nothing_raises(tmp_path: Path, monkeypatch) -> None:
    _install_fake_downloader(monkeypatch)
    ref = TensorhubRef(owner="e2e", repo="sdxl-full")

    with pytest.raises(ValueError, match="components="):
        asyncio.run(ensure_snapshot_async(
            base_dir=tmp_path, ref=ref, resolved=_resolved_pipeline(), components=("nope",),
        ))


def test_full_and_component_scoped_fetches_never_collide(tmp_path: Path, monkeypatch) -> None:
    """A component-scoped fetch and the whole-repo fetch of the SAME digest
    must materialize under DIFFERENT directories — sharing one would let a
    partial tree be mistaken for (or silently completed into) the full one."""
    _install_fake_downloader(monkeypatch)
    ref = TensorhubRef(owner="e2e", repo="sdxl-full")

    partial = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path, ref=ref, resolved=_resolved_pipeline(), components=("vae",),
    ))
    full = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path, ref=ref, resolved=_resolved_pipeline(),
    ))

    assert partial != full
    assert not (partial / "unet").exists()
    assert (full / "unet").exists() and (full / "vae").exists()
