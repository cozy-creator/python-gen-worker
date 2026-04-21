import base64
import json
import os

from gen_worker.worker import Worker


def _fake_jwt(claims: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}."


def test_worker_does_not_use_release_id_env(monkeypatch):
    monkeypatch.setenv("RELEASE_ID", "rel-123")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")

    w = Worker(user_module_names=[], worker_jwt="dummy-worker-jwt")
    # Worker identity comes from the scheduler-issued WORKER_JWT claims, not from env.
    assert w.release_id == ""


def test_worker_uses_worker_jwt_claims_for_registration_fallback(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-1")
    token = _fake_jwt({"sub": "worker-rel-1-abc", "release_id": "rel-1"})

    w = Worker(user_module_names=[], worker_jwt=token)

    assert w.worker_id == "worker-rel-1-abc"
    assert w.release_id == "rel-1"
