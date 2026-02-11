from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import entrypoint


def test_preflight_cache_dirs_fails_without_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    monkeypatch.setenv("WORKER_MODEL_CACHE_DIR", str(primary))
    monkeypatch.delenv("WORKER_CACHE_DIR_FALLBACK", raising=False)
    monkeypatch.delenv("WORKER_LOCAL_MODEL_CACHE_DIR", raising=False)

    def _fail(_: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(entrypoint, "_probe_cache_path_writable", _fail)

    with pytest.raises(RuntimeError, match="WORKER_MODEL_CACHE_DIR"):
        entrypoint._preflight_cache_dirs()


def test_preflight_cache_dirs_uses_fallback_when_primary_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    monkeypatch.setenv("WORKER_MODEL_CACHE_DIR", str(primary))
    monkeypatch.setenv("WORKER_CACHE_DIR_FALLBACK", str(fallback))
    monkeypatch.delenv("WORKER_LOCAL_MODEL_CACHE_DIR", raising=False)

    def _probe(path: Path) -> None:
        if path == primary:
            raise PermissionError("permission denied")
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".probe"
        probe.write_bytes(b"ok")
        probe.unlink()

    monkeypatch.setattr(entrypoint, "_probe_cache_path_writable", _probe)

    cfg = entrypoint._preflight_cache_dirs()
    assert cfg["model_cache_dir"] == str(fallback)
    assert cfg["local_model_cache_dir"] == ""
    assert (fallback).exists()

