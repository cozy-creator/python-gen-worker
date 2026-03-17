from __future__ import annotations

import pytest

from gen_worker import entrypoint
from gen_worker.trainer import runtime as trainer_runtime


def test_entrypoint_routes_to_trainer_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_MODE", "trainer")
    monkeypatch.setattr(trainer_runtime, "run_training_runtime_from_env", lambda: 13)
    assert entrypoint._run_main() == 13


def test_entrypoint_rejects_invalid_worker_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_MODE", "wat")
    assert entrypoint._run_main() == 1
