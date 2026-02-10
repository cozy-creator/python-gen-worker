from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from gen_worker.mount_backend import MountBackend
from gen_worker.pipeline_loader import PipelineLoader


def test_pipeline_loader_localizes_nfs_snapshot_to_local_cache(tmp_path: Path) -> None:
    models_dir = tmp_path / "workspace-cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    local_cache_dir = tmp_path / "local-cache"

    model_id = "modelA"
    src = models_dir / model_id
    src.mkdir(parents=True)
    (src / "weights.bin").write_bytes(b"abc")

    def fake_mount_backend_for_path(p: object) -> MountBackend:
        s = str(p)
        if "workspace-cache" in s:
            return MountBackend(mountpoint="/workspace", fstype="nfs4", source="10.0.0.1:/vol")
        return MountBackend(mountpoint="/tmp", fstype="ext4", source="/dev/nvme0n1p1")

    with patch("gen_worker.mount_backend.mount_backend_for_path", side_effect=fake_mount_backend_for_path):
        loader = PipelineLoader(models_dir=str(models_dir), local_cache_dir=str(local_cache_dir), downloader=None)
        out = asyncio.run(loader.ensure_model_available(model_id))

    assert out.exists()
    assert out != src
    assert (out / "weights.bin").read_bytes() == b"abc"
    # Localization is a copy, not a move.
    assert (src / "weights.bin").read_bytes() == b"abc"


def test_pipeline_loader_downloads_into_worker_model_cache_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    class DummyDownloader:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def download(self, model_id: str, dest_root: str) -> str:
            self.calls.append((model_id, dest_root))
            p = Path(dest_root) / model_id
            p.mkdir(parents=True, exist_ok=True)
            (p / "ok.txt").write_text("ok", encoding="utf-8")
            return str(p)

    dl = DummyDownloader()
    loader = PipelineLoader(models_dir=str(models_dir), local_cache_dir="", downloader=dl)
    out = asyncio.run(loader.ensure_model_available("m1"))

    assert dl.calls
    assert dl.calls[0][1] == str(models_dir)
    assert (out / "ok.txt").read_text(encoding="utf-8") == "ok"

