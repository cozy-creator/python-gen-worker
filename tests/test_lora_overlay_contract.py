from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker import Repo, ValidationError
from gen_worker.worker import InjectionSpec, Worker


class _Pipeline:
    def __init__(self) -> None:
        self.active_specs: list[Any] = []
        self.cleaned = False

    def load_lora_weights(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_adapters(self, *args: Any, **kwargs: Any) -> None:
        return None

    def unload_lora_weights(self) -> None:
        self.cleaned = True


def _worker() -> Worker:
    w = Worker.__new__(Worker)
    w._downloader = None
    return w


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(request_id="req-123")


def test_resolved_models_preserves_lora_overlay_contract() -> None:
    request = SimpleNamespace(
        resolved_models={
            "pipeline": {
                "ref": "base/model",
                "provider": "hf",
                "loras": [{"ref": "alice/style:prod", "weight": 0.75}],
            }
        }
    )

    out = _worker()._resolved_models_for_request(request)

    assert out["pipeline"]["ref"] == "base/model"
    assert out["pipeline"]["loras"] == [{"ref": "alice/style:prod", "weight": 0.75}]


def test_lora_overlays_require_allow_lora() -> None:
    inj = InjectionSpec("pipeline", object, Repo("base/model"))
    resolved = {"pipeline": {"loras": [{"ref": "alice/style"}]}}

    with pytest.raises(ValueError, match="did not declare allow_lora"):
        _worker()._normalize_lora_overlays(inj, resolved)


def test_lora_overlay_runtime_must_support_diffusers_adapter_api(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    lora_file = tmp_path / "adapter_model.safetensors"
    lora_file.write_bytes(b"not used")
    inj = InjectionSpec("pipeline", object, Repo("base/model").allow_lora())
    resolved = {"pipeline": {"loras": [{"ref": lora_file.as_posix()}]}}

    with pytest.raises(ValueError, match="lacks load_lora_weights"):
        with _worker()._lora_overlay_context(_ctx(), [inj], {"pipeline": object()}, resolved):
            pass


def test_lora_overlay_context_generates_internal_names_serializes_and_cleans_up(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lora_file = tmp_path / "adapter_model.safetensors"
    lora_file.write_bytes(b"not used")
    pipeline = _Pipeline()
    inj = InjectionSpec("pipeline", object, Repo("base/model").allow_lora())
    resolved = {"pipeline": {"loras": [{"ref": lora_file.as_posix(), "weight": 0.5}]}}

    @contextmanager
    def fake_load_loras(pipe: _Pipeline, specs: list[Any], request_id: str):
        pipe.active_specs = specs
        try:
            yield
        finally:
            pipe.unload_lora_weights()
            pipe.active_specs = []

    monkeypatch.setattr("gen_worker.utils.lora.load_loras", fake_load_loras)
    w = _worker()

    with w._lora_overlay_context(_ctx(), [inj], {"pipeline": pipeline}, resolved):
        assert pipeline.active_specs[0].weight == 0.5
        assert pipeline.active_specs[0].adapter_name == "cozy_lora_req-123_pipeline_0"
        assert w._pipeline_lora_lock(pipeline).acquire(blocking=False) is False

    assert pipeline.active_specs == []
    assert pipeline.cleaned is True


def test_lora_target_module_validation_accepts_matching_pipeline_keys(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lora_file = tmp_path / "adapter_model.safetensors"
    lora_file.write_bytes(b"not used")
    pipeline = _Pipeline()
    pipeline.state_dict = lambda: {"unet.attn.to_q.weight": object()}  # type: ignore[method-assign]
    inj = InjectionSpec("pipeline", object, Repo("base/model").allow_lora())
    resolved = {"pipeline": {"loras": [{"ref": lora_file.as_posix()}]}}

    @contextmanager
    def fake_load_loras(pipe: _Pipeline, specs: list[Any], request_id: str):
        yield

    w = _worker()
    monkeypatch.setattr(w, "_read_safetensors_keys", lambda path: ["unet.attn.to_q.lora_down.weight", "unet.attn.to_q.lora_up.weight"])
    monkeypatch.setattr("gen_worker.utils.lora.load_loras", fake_load_loras)

    with w._lora_overlay_context(_ctx(), [inj], {"pipeline": pipeline}, resolved):
        pass


def test_lora_target_module_validation_rejects_mismatched_pipeline_keys(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lora_file = tmp_path / "adapter_model.safetensors"
    lora_file.write_bytes(b"not used")
    pipeline = _Pipeline()
    pipeline.state_dict = lambda: {"unet.other.weight": object()}  # type: ignore[method-assign]
    inj = InjectionSpec("pipeline", object, Repo("base/model").allow_lora())
    resolved = {"pipeline": {"loras": [{"ref": lora_file.as_posix()}]}}
    w = _worker()
    monkeypatch.setattr(w, "_read_safetensors_keys", lambda path: ["unet.attn.to_q.lora_down.weight"])

    with pytest.raises(ValidationError, match="targets modules not present"):
        with w._lora_overlay_context(_ctx(), [inj], {"pipeline": pipeline}, resolved):
            pass
