from __future__ import annotations

import json
from typing import Any

import pytest

import gen_worker.request_context as request_context_module
from gen_worker.request_context import ConversionContext, RequestContext


class _Response:
    def __init__(self, status_code: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self) -> dict[str, Any]:
        return self._body


class _NoopUploadSessionManager:
    def get(self, kind: str, scope: dict[str, Any]) -> None:
        return None


def _context(monkeypatch: pytest.MonkeyPatch) -> ConversionContext:
    monkeypatch.setattr(request_context_module, "_assert_token_repo_scope_matches_destination", lambda *a, **k: None)
    monkeypatch.setattr(request_context_module.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(request_context_module.random, "uniform", lambda *_a, **_k: 0.0)

    # Issue #1: publish_repo_revision moved off RequestContext and onto
    # ConversionContext. Tests construct the subclass directly.
    ctx = ConversionContext(
        request_id="req-1",
        job_id="00000000-0000-0000-0000-000000000001",
        file_api_base_url="http://tensorhub",
        worker_capability_token="cap-token",
    )
    ctx._repo_job_upload_scope = lambda: ("owner", "repo", "00000000-0000-0000-0000-000000000001")  # type: ignore[method-assign]
    ctx._checkpoint_revision_id = lambda _owner, _repo: "11111111-1111-1111-1111-111111111111"  # type: ignore[method-assign]
    ctx._upload_session_manager = lambda: _NoopUploadSessionManager()  # type: ignore[method-assign]
    return ctx


def test_request_context_models_returns_copy() -> None:
    ctx = RequestContext(
        request_id="req-1",
        models={
            "model": {
                "ref": "black-forest-labs/FLUX.2-klein-4B",
                "loras": [{"ref": "alice/cozy-knit-lora", "weight": 0.8}],
            }
        },
    )

    first = ctx.models
    first["model"]["ref"] = "mutated"
    first["model"]["loras"][0]["weight"] = 99

    assert ctx.models["model"]["ref"] == "black-forest-labs/FLUX.2-klein-4B"
    assert ctx.models["model"]["loras"][0]["weight"] == 0.8


def _publish(ctx: ConversionContext) -> dict[str, Any]:
    return ctx.publish_repo_revision(
        destination_repo="owner/repo",
        metadata={
            "checkpoint_flavors": [
                {
                    "flavor": "bf16",
                    "snapshot_manifest": [
                        {"path": "model.safetensors", "blake3": "abc", "size_bytes": 1},
                    ],
                }
            ]
        },
        destination_repo_tags=["prod"],
    )


def test_publish_repo_revision_polls_after_accepted_finalize(monkeypatch: pytest.MonkeyPatch) -> None:
    posts: list[str] = []
    gets: list[str] = []

    def fake_post(url: str, **kwargs: Any) -> _Response:
        posts.append(url)
        return _Response(
            202,
            {"state": "finalizing", "retry_after_seconds": 0.01},
            headers={
                "Location": "/api/v1/repos/owner/repo/revisions/11111111-1111-1111-1111-111111111111",
                "Retry-After": "0.01",
            },
        )

    def fake_get(url: str, **kwargs: Any) -> _Response:
        gets.append(url)
        return _Response(
            200,
            {
                "state": "finalized",
                "finalize_status": "succeeded",
                "finalize_result": {
                    "ok": True,
                    "checkpoints": [{"checkpoint_id": "blake3:final"}],
                },
            },
        )

    monkeypatch.setattr(request_context_module.requests, "post", fake_post)
    monkeypatch.setattr(request_context_module.requests, "get", fake_get)

    result = _publish(_context(monkeypatch))

    assert posts == [
        "http://tensorhub/api/v1/repos/owner/repo/revisions/11111111-1111-1111-1111-111111111111/finalize"
    ]
    assert gets == [
        "http://tensorhub/api/v1/repos/owner/repo/revisions/11111111-1111-1111-1111-111111111111"
    ]
    assert result["checkpoint_ids"] == ["blake3:final"]


def test_publish_repo_revision_forwards_checkpoint_artifact_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    posted_body: dict[str, Any] = {}

    def fake_post(_url: str, **kwargs: Any) -> _Response:
        posted_body.update(json.loads(kwargs.get("data") or "{}"))
        return _Response(200, {"ok": True, "checkpoints": [{"checkpoint_id": "blake3:final"}]})

    monkeypatch.setattr(request_context_module.requests, "post", fake_post)

    ctx = _context(monkeypatch)
    result = ctx.publish_repo_revision(
        destination_repo="owner/repo",
        metadata={
            "checkpoint_flavors": [
                {
                    "flavor": "bf16",
                    "snapshot_manifest": [
                        {"path": "model.safetensors", "blake3": "abc", "size_bytes": 1},
                    ],
                    "dtype": "bf16",
                    "file_layout": "diffusers",
                    "file_type": "safetensors",
                }
            ]
        },
        destination_repo_tags=["prod"],
    )

    checkpoint = posted_body["checkpoint_flavors"][0]
    assert result["checkpoint_ids"] == ["blake3:final"]
    assert checkpoint["dtype"] == "bf16"
    assert checkpoint["file_layout"] == "diffusers"
    assert checkpoint["file_type"] == "safetensors"
    assert "display_label" not in checkpoint


def test_publish_repo_revision_retries_finalize_post_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_post(url: str, **kwargs: Any) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise request_context_module.requests.ReadTimeout("no response")
        return _Response(
            202,
            {"state": "finalizing"},
            headers={"Location": "/api/v1/repos/owner/repo/revisions/11111111-1111-1111-1111-111111111111"},
        )

    def fake_get(url: str, **kwargs: Any) -> _Response:
        return _Response(
            200,
            {
                "state": "finalized",
                "finalize_result": {
                    "ok": True,
                    "checkpoints": [{"checkpoint_id": "blake3:after-timeout"}],
                },
            },
        )

    monkeypatch.setattr(request_context_module.requests, "post", fake_post)
    monkeypatch.setattr(request_context_module.requests, "get", fake_get)

    result = _publish(_context(monkeypatch))

    assert calls == 2
    assert result["checkpoint_ids"] == ["blake3:after-timeout"]


def test_publish_repo_revision_surfaces_failed_finalize_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        request_context_module.requests,
        "post",
        lambda *a, **k: _Response(
            202,
            {"state": "finalizing"},
            headers={"Location": "/api/v1/repos/owner/repo/revisions/11111111-1111-1111-1111-111111111111"},
        ),
    )
    monkeypatch.setattr(
        request_context_module.requests,
        "get",
        lambda *a, **k: _Response(
            200,
            {
                "state": "failed",
                "finalize_error": {"error": "service_unavailable", "message": "promote blob to CAS"},
            },
        ),
    )

    with pytest.raises(RuntimeError, match="service_unavailable"):
        _publish(_context(monkeypatch))
