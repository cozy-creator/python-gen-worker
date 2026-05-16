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
