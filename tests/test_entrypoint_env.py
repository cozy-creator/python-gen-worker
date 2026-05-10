from __future__ import annotations

from gen_worker.entrypoint import _scheduler_public_addr_from_env


def test_scheduler_public_addr_reads_orchestrator_env(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_ORCHESTRATOR_GRPC_ADDR", "new-host:8080")
    assert _scheduler_public_addr_from_env() == "new-host:8080"


def test_scheduler_public_addr_requires_orchestrator_env(monkeypatch) -> None:
    monkeypatch.delenv("PUBLIC_ORCHESTRATOR_GRPC_ADDR", raising=False)
    assert _scheduler_public_addr_from_env() == ""
