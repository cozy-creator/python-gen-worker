import os

from gen_worker.worker import Worker


def test_worker_does_not_use_release_id_env(monkeypatch):
    monkeypatch.setenv("RELEASE_ID", "rel-123")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")

    w = Worker(user_module_names=[], worker_jwt="dummy-worker-jwt")
    # Worker identity comes from the scheduler-issued WORKER_JWT claims, not from env.
    assert w.release_id == ""
