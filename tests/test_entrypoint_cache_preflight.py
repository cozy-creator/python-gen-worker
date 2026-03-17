from __future__ import annotations

from pathlib import Path

import pytest

from gen_worker import entrypoint


def test_preflight_cache_dirs_fails_without_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(tmp_path / "primary-root"))
    monkeypatch.delenv("WORKER_LOCAL_MODEL_CACHE_DIR", raising=False)

    def _fail(_: Path) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(entrypoint, "_probe_cache_path_writable", _fail)

    with pytest.raises(RuntimeError, match="tensorhub CAS path"):
        entrypoint._preflight_cache_dirs()


def test_preflight_cache_dirs_uses_tensorhub_cache_dir_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "cache-root"
    primary = root / "cas"
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", str(root))
    monkeypatch.delenv("WORKER_LOCAL_MODEL_CACHE_DIR", raising=False)

    def _probe(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".probe"
        probe.write_bytes(b"ok")
        probe.unlink()

    monkeypatch.setattr(entrypoint, "_probe_cache_path_writable", _probe)

    cfg = entrypoint._preflight_cache_dirs()
    assert cfg["model_cache_dir"] == str(primary)
    assert cfg["local_model_cache_dir"] == ""
    assert primary.exists()
