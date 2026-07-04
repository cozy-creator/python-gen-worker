"""Settings loader: env -> field mapping for the non-container seams."""

from gen_worker.config import load_settings


def test_endpoint_lock_path_default() -> None:
    assert load_settings().endpoint_lock_path == "/app/.tensorhub/endpoint.lock"


def test_endpoint_lock_path_env(monkeypatch) -> None:
    monkeypatch.setenv("ENDPOINT_LOCK_PATH", "/tmp/e2e/endpoint.lock")
    assert load_settings().endpoint_lock_path == "/tmp/e2e/endpoint.lock"
