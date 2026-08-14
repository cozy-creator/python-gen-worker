from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gen_worker import compiled_graph_store


KEY = "cg-key-v1-" + "1" * 56
OTHER_KEY = "cg-key-v1-" + "2" * 56


def _record(key: str) -> dict[str, object]:
    return {
        "format": 1,
        "compiled_graph_key": key,
        "family": "micro",
        "arm_token": "",
        "bytes": 7,
        "stored_at": 1.0,
        "manifest": "sha256:" + "3" * 64,
        "verdict": compiled_graph_store.VERDICT_ADMITTED,
        "sink": compiled_graph_store.SINK_NONE,
    }


def test_sidecar_key_must_equal_its_directory_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = compiled_graph_store.sidecar_dir(KEY, tmp_path)
    path.mkdir(parents=True)
    (path / compiled_graph_store.RECORD_NAME).write_text(
        json.dumps(_record(OTHER_KEY))
    )
    monkeypatch.setattr(
        compiled_graph_store,
        "_engine",
        lambda _root=None: pytest.fail("wrong sidecar reached TCG resolve"),
    )

    assert compiled_graph_store.lookup(KEY, tmp_path) is None
    assert compiled_graph_store.stored_graphs(tmp_path) == []


def test_load_runner_refuses_a_runner_for_another_exact_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = SimpleNamespace(key=KEY, metadata={})
    wrong_runner = SimpleNamespace(key=OTHER_KEY)

    class FakeEngine:
        def resolve(self, key: str, destination: Path) -> Any:
            assert key == KEY
            assert destination.name == KEY
            return graph

        def runner(self, key: str, destination: Path) -> Any:
            assert key == KEY
            assert destination.name == KEY
            return wrong_runner

    monkeypatch.setattr(
        compiled_graph_store,
        "_engine",
        lambda _root=None: FakeEngine(),
    )

    assert compiled_graph_store.load_runner(KEY, tmp_path) is None


def test_malformed_key_never_reaches_tcg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiled_graph_store,
        "_engine",
        lambda _root=None: pytest.fail("malformed key reached TCG"),
    )
    assert compiled_graph_store.load_runner("cell-not-a-key", tmp_path) is None
