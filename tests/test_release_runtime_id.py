import os

from gen_worker.worker import Worker


def test_worker_uses_release_id_env(monkeypatch):
    monkeypatch.setenv("RELEASE_ID", "rel-123")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")

    w = Worker(user_module_names=[])
    assert w.release_id == "rel-123"
