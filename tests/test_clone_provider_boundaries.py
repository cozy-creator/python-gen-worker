from __future__ import annotations

from types import SimpleNamespace

import pytest

import gen_worker.clone as clone
from gen_worker.clone._shared import ingest_from_source


def test_from_civitai_rejects_source_url() -> None:
    payload = SimpleNamespace(source_url="https://civitai.com/models/123")
    with pytest.raises(ValueError, match="source_url is not supported"):
        clone.from_civitai(SimpleNamespace(), payload)


def test_from_civitai_requires_model_version_id() -> None:
    payload = SimpleNamespace(civitai_model_version_id=0)
    with pytest.raises(ValueError, match="civitai_model_version_id is required"):
        clone.from_civitai(SimpleNamespace(), payload)


def test_from_civitai_uses_model_version_id(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_clone(ctx, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(clone, "_run_clone", fake_run_clone)

    out = clone.from_civitai(
        SimpleNamespace(),
        SimpleNamespace(civitai_model_version_id=123, civitai_file_id=456),
    )

    assert out == {"ok": True}
    assert calls[0]["provider"] == "civitai"
    assert calls[0]["source_ref"] == "123"
    assert calls[0]["source_version_id"] == "123"
    assert calls[0]["source_metadata_overrides"] == {"civitai_file_id": "456"}


def test_ingest_from_source_rejects_arbitrary_url() -> None:
    with pytest.raises(ValueError, match="arbitrary URL model sources are not supported"):
        ingest_from_source(
            SimpleNamespace(request_id="req-1", hf_token=""),
            provider="huggingface",
            source_ref="https://example.com/model.safetensors",
            source_revision=None,
            output_ref=None,
        )


def test_from_huggingface_forwards_hf_token(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_clone(ctx, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(clone, "_run_clone", fake_run_clone)

    clone.from_huggingface(
        SimpleNamespace(),
        SimpleNamespace(huggingface_repo="org/model"),
        hf_token="invoker-token",
    )

    assert calls[0]["hf_token"] == "invoker-token"


def test_from_civitai_forwards_api_key(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_clone(ctx, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(clone, "_run_clone", fake_run_clone)

    clone.from_civitai(
        SimpleNamespace(),
        SimpleNamespace(civitai_model_version_id=123, civitai_file_id=0),
        civitai_api_key="invoker-key",
    )

    assert calls[0]["civitai_api_key"] == "invoker-key"


def test_ingest_from_source_hf_token_precedence(monkeypatch) -> None:
    """Per-request hf_token wins over ctx.hf_token (endpoint env)."""
    from gen_worker.clone import _shared

    seen: dict[str, object] = {}

    def fake_download(source_repo, repo_dir, **kwargs):
        seen["hf_token"] = kwargs.get("hf_token")
        # Bail before any real upload work — we only assert token threading.
        raise RuntimeError("stop-after-download-call")

    monkeypatch.setattr(_shared, "download_huggingface_repo_files", fake_download)

    with pytest.raises(RuntimeError, match="stop-after-download-call"):
        ingest_from_source(
            SimpleNamespace(request_id="req-1", hf_token="endpoint-env-token"),
            provider="huggingface",
            source_ref="org/model",
            source_revision=None,
            output_ref=None,
            hf_token="invoker-token",
        )

    assert seen["hf_token"] == "invoker-token"


def test_ingest_from_source_hf_token_falls_back_to_ctx(monkeypatch) -> None:
    """Unset per-request hf_token falls back to ctx.hf_token (endpoint env)."""
    from gen_worker.clone import _shared

    seen: dict[str, object] = {}

    def fake_download(source_repo, repo_dir, **kwargs):
        seen["hf_token"] = kwargs.get("hf_token")
        raise RuntimeError("stop-after-download-call")

    monkeypatch.setattr(_shared, "download_huggingface_repo_files", fake_download)

    with pytest.raises(RuntimeError, match="stop-after-download-call"):
        ingest_from_source(
            SimpleNamespace(request_id="req-1", hf_token="endpoint-env-token"),
            provider="huggingface",
            source_ref="org/model",
            source_revision=None,
            output_ref=None,
            hf_token=None,
        )

    assert seen["hf_token"] == "endpoint-env-token"


def test_ingest_from_source_civitai_api_key_threaded(monkeypatch) -> None:
    """Per-request civitai_api_key reaches download_civitai_model_version_files."""
    from gen_worker.clone import _shared

    seen: dict[str, object] = {}

    def fake_download(model_version_id, repo_dir, **kwargs):
        seen["civitai_api_key"] = kwargs.get("civitai_api_key")
        raise RuntimeError("stop-after-download-call")

    monkeypatch.setattr(_shared, "download_civitai_model_version_files", fake_download)

    with pytest.raises(RuntimeError, match="stop-after-download-call"):
        ingest_from_source(
            SimpleNamespace(request_id="req-1", hf_token=""),
            provider="civitai",
            source_ref="123",
            source_revision=None,
            civitai_model_version_id=123,
            output_ref=None,
            civitai_api_key="invoker-key",
        )

    assert seen["civitai_api_key"] == "invoker-key"
