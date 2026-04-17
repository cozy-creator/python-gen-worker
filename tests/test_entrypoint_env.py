from __future__ import annotations

from gen_worker.entrypoint import _scheduler_public_addr_from_env


def test_scheduler_public_addr_prefers_new_env(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_ORCHESTRATOR_GRPC_ADDR", "new-host:8080")
    assert _scheduler_public_addr_from_env() == "new-host:8080"


def test_scheduler_public_addr_ignores_legacy_env(monkeypatch) -> None:
    monkeypatch.delenv("PUBLIC_ORCHESTRATOR_GRPC_ADDR", raising=False)
    monkeypatch.setenv("SCHEDULER_PUBLIC_ADDR", "old-host:8080")
    assert _scheduler_public_addr_from_env() == ""
