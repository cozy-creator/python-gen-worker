"""Mirror-first serving (tensorhub #557): an orchestrator-shipped snapshot is
authoritative for ANY provider. An hf/civitai binding ref that arrives with a
resolved snapshot (its platform mirror, presigned R2) must download through the
tensorhub-CAS snapshot machinery and never touch the upstream registry; refs
without a snapshot keep their provider-direct path."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import gen_worker.models.cozy_snapshot as snap_mod
import gen_worker.models.download as dl_mod
from gen_worker.models.download import ensure_local

_DIGEST = "12" * 32


def _resolved() -> dict:
    return {
        "snapshot_digest": _DIGEST,
        "files": [{
            "path": "model_index.json",
            "size_bytes": 4,
            "blake3": "cd" * 32,
            "url": "http://r2.invalid/presigned",
        }],
    }


@pytest.fixture()
def fake_blob(monkeypatch):
    """Stub the CAS blob leaf; poison provider-direct paths so any upstream
    contact fails the test loudly."""
    async def _fake_download(url: str, dst: Path, expected_size: int, expected_blake3: str, on_bytes=None) -> None:
        assert url == "http://r2.invalid/presigned"
        dst.write_bytes(b"mock")

    monkeypatch.setattr(snap_mod, "_download_one_file", _fake_download)

    def _upstream_forbidden(*a, **kw):
        raise AssertionError("upstream registry contacted despite resolved snapshot")

    monkeypatch.setattr(dl_mod, "download_hf", _upstream_forbidden)
    monkeypatch.setattr(dl_mod, "download_civitai", _upstream_forbidden)
    snap_mod._SNAP_ENTRIES.clear()


def test_hf_ref_with_snapshot_downloads_from_cas(tmp_path: Path, fake_blob) -> None:
    out = asyncio.run(ensure_local(
        "black-forest-labs/FLUX.1-dev",
        provider="hf",
        snapshot=_resolved(),
        cache_dir=tmp_path,
    ))
    assert (out / "model_index.json").read_bytes() == b"mock"
    assert out.name == _DIGEST  # digest-addressed snapshot tree


def test_civitai_ref_with_snapshot_downloads_from_cas(tmp_path: Path, fake_blob) -> None:
    out = asyncio.run(ensure_local(
        "993999",
        provider="civitai",
        snapshot=_resolved(),
        cache_dir=tmp_path,
    ))
    assert (out / "model_index.json").read_bytes() == b"mock"


def test_hf_ref_without_snapshot_stays_provider_direct(tmp_path: Path, monkeypatch) -> None:
    sentinel = tmp_path / "hf-direct"

    def _fake_hf(parsed, **kw):
        sentinel.mkdir()
        return sentinel

    monkeypatch.setattr(dl_mod, "download_hf", _fake_hf)
    out = asyncio.run(ensure_local(
        "owner/repo",
        provider="hf",
        snapshot=None,
        cache_dir=tmp_path,
    ))
    assert out == sentinel


def test_tensorhub_ref_without_snapshot_is_retryable(tmp_path: Path) -> None:
    from gen_worker.api.errors import RetryableError

    with pytest.raises(RetryableError):
        asyncio.run(ensure_local("acme/checkpoint", provider="tensorhub", cache_dir=tmp_path))
