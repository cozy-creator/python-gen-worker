"""pgw#1009: the CLONE producer restates classification on the declare.

th#1411 refuses a v2 publish into a repo whose live rows carry classification
unless the request restates `objective` / `distilled`. `publish_flavors` has
done that since pgw#654; `run_clone`'s publish call site never did — it spent
the caller's facts only on `apply_objective_scheduler_config` (the scheduler
stamp inside the produced tree) and dropped them before the wire.

Live cost: master's `tensorhub/wan22-t2v-a14b` mirror is missing
`transformer_2`, and the repair — one more `clone-huggingface` leg — is
impossible, because every re-clone into that already-classified repo dies at
declare with `classification_required`. Every mirror the catalog already
serves is in the same position.

Revert-turns-red: drop `objective=`/`distilled=` from the `publish_v2(...)`
call in `clone.py` and both assertions below fail.

    pytest tests/convert/test_clone_classification_pgw1008.py -q
"""

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
        destination_repo_tags=["prod"],
        outputs=[{"dtype": "fp32", "file_layout": "multi-file",
                  "file_type": "safetensors"}],
        **kw,
    )


def test_declared_facts_reach_the_publish_declare(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ie#609 repair shape: a merge leg into a classified mirror."""
    result = _run(fake_hub, tmp_path, monkeypatch,
                  objective="flow", distilled=False, overwrite_repo=False)

    assert not result.failed_flavors, result.failed_flavors
    req = _FakeHub.state["publish_request"]
    assert req["objective"] == "flow"
    assert req["distilled"] is False
    assert req["mode"] == "merge"


def test_undeclared_objective_stays_off_the_wire(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unstated objective is still unstated — a clone must not author
    `epsilon` (or an empty string) nobody declared, so the field is omitted and
    th#1411's refusal is what tells the caller to state it.

    `distilled` is different by construction: `CloneHuggingFaceInput.distilled`
    is a plain `bool` defaulting to False, so an undeclared clone DOES state
    `distilled: false` — the honest reading of a straight upstream mirror, and
    the same first-hand declaration `publish_flavors` makes."""
    result = _run(fake_hub, tmp_path, monkeypatch)

    assert not result.failed_flavors, result.failed_flavors
    req = _FakeHub.state["publish_request"]
    assert "objective" not in req
    assert req["distilled"] is False
