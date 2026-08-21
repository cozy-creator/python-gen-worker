from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker.convert.clone import run_clone
from gen_worker.convert.ingest import IngestedSource

from fake_hub import _FakeHub


class _Ctx:
    def __init__(self, server: Any) -> None:
        self._file_api_base_url = f"http://127.0.0.1:{server.server_port}"
        self._worker_capability_token = "cap-token"
        self.owner = "tensorhub"
        self.request_id = "req-1008"
        self.destination = {"repo": "tensorhub/fallback"}


def _source(dest_dir: Path) -> IngestedSource:
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "config.json").write_text('{"architectures": ["FakeBackbone"]}')
    (dest_dir / "model.safetensors").write_bytes(b"\x00" * 64)
    return IngestedSource(
        provider="huggingface", source_ref="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        source_revision="5be7df96", dir=dest_dir, layout="single-file",
        model_family="wan", model_family_variant="wan22",
        classification=SimpleNamespace(strategy="transformers"),
        attrs={"dtype": "fp32", "file_layout": "single-file"},
        metadata={"source_provider": "huggingface"},
        repo_spec={"kind": "model", "library_name": "transformers"},
    )


def _run(fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
         **kw: Any) -> Any:
    _FakeHub.state["finalize_calls"] = 1
    monkeypatch.setenv("COZY_CONVERT_WORKDIR", str(tmp_path / "work"))
    src = _source(tmp_path / "source")
    monkeypatch.setattr("gen_worker.convert.clone.ingest_huggingface",
                        lambda source_ref, dest_dir, **kwargs: src)
    return run_clone(
        _Ctx(fake_hub), provider="huggingface",
        source_ref="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        destination_repo="tensorhub/wan22-t2v-a14b",
        destination_release="r1",
        outputs=[{"dtype": "fp32", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
        **kw,
    )


def test_declared_facts_reach_the_publish_declare(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run(fake_hub, tmp_path, monkeypatch,
                  objective="flow", distilled=False, overwrite_repo=False)

    assert not result.failed_flavors, result.failed_flavors
    req = _FakeHub.state["publish_request"]
    assert req["release"] == "r1"
    assert req["objective"] == "flow"
    assert req["distilled"] is False
    assert req["mode"] == "merge"


def test_undeclared_objective_stays_off_the_wire(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run(fake_hub, tmp_path, monkeypatch)

    assert not result.failed_flavors, result.failed_flavors
    req = _FakeHub.state["publish_request"]
    assert "objective" not in req
    assert req["distilled"] is False
