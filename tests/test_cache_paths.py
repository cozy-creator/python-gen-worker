from __future__ import annotations

from pathlib import Path

from gen_worker.cache_paths import tensorhub_cache_dir, worker_model_cache_dir


def test_tensorhub_cache_dir_defaults_to_home_cache(monkeypatch) -> None:
    monkeypatch.delenv("TENSORHUB_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", "/home/tester")
    assert tensorhub_cache_dir() == Path("/home/tester/.cache/tensorhub")
    assert worker_model_cache_dir() == Path("/home/tester/.cache/tensorhub/cas")


def test_worker_model_cache_dir_uses_tensorhub_cache_dir_when_set(monkeypatch) -> None:
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", "/var/cache/tensorhub")
    assert worker_model_cache_dir() == Path("/var/cache/tensorhub/cas")


def test_worker_model_cache_dir_ignores_legacy_worker_model_cache_dir_override(monkeypatch) -> None:
    monkeypatch.setenv("TENSORHUB_CACHE_DIR", "/var/cache/tensorhub")
    monkeypatch.setenv("WORKER_MODEL_CACHE_DIR", "/mnt/models")
    assert worker_model_cache_dir() == Path("/var/cache/tensorhub/cas")
