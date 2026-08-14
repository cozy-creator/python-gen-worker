"""Producer-side proof for the compiled-graph protocol hard cut (th#1947)."""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

from gen_worker import aot_serve, cell_resolve, fleet_cells, receipts
from gen_worker.models import provision
from gen_worker.procsplit import actions


def test_only_the_compiled_graph_worker_routes_are_authorized() -> None:
    expected = {
        "compiled_graphs.resolve": "/v1/worker/compiled-graphs/resolve",
        "compiled_graphs.publish_intent":
            "/v1/worker/compiled-graphs/publish-intent",
        "compiled_graphs.publish_complete":
            "/v1/worker/compiled-graphs/publish-complete",
    }
    for name, path in expected.items():
        assert actions.ACTIONS[name].path.fullmatch(path)

    assert not any(name.startswith("cells.") for name in actions.ACTIONS)
    assert not any(
        action.path.fullmatch("/v1/worker/cells/receipt")
        or action.path.fullmatch("/v1/worker/cells/revocations")
        for action in actions.ACTIONS.values()
    )
    assert actions.PUBLISH_ACTIONS == frozenset({
        "compiled_graphs.publish_intent",
        "compiled_graphs.publish_complete",
    })


def test_resolve_and_publish_speak_compiled_graph_key_only() -> None:
    key = "cg-key-v1-" + "a" * 56
    assert cell_resolve.RESOLVE_PATH == \
        "/v1/worker/compiled-graphs/resolve"
    assert "STATUS_AMBIGUOUS" not in cell_resolve.__all__
    assert not hasattr(cell_resolve, "STATUS_AMBIGUOUS")
    assert dataclasses.fields(cell_resolve.ResolvedCompiledGraph)[1].name == \
        "compiled_graph_key"
    assert fleet_cells.PublishEntry(key, {}).wire() == {
        "compiled_graph_key": key,
        "identity_axes": {},
        "mint_duration_ms": 0,
    }
    assert not hasattr(fleet_cells, "PUBLISH_STATUS_CONDEMNED")


def test_the_embedded_v1_receipt_is_the_only_receipt_seam() -> None:
    assert receipts.RECEIPT_VERSION == "compiled-graph-receipt-v1"
    assert dataclasses.fields(receipts.Receipt)[2].name == "compiled_graph_key"
    assert not hasattr(receipts, "RECEIPT_PATH")
    assert not hasattr(receipts, "REVOCATIONS_PATH")


def test_worker_has_no_legacy_artifact_or_path_arm_surface() -> None:
    retired = (
        "ArtifactRunner",
        "arm_entry",
        "enable",
        "entry_metadata",
        "pack",
        "stage_artifact",
        "stamp_entry",
        "unpack",
        "unpack_metadata",
    )
    assert all(not hasattr(aot_serve, name) for name in retired)
    source = inspect.getsource(aot_serve)
    assert "gen_worker.cell_key" not in source
    assert "from . import cell_key" not in source

    parameters = inspect.signature(provision.enable_compiled).parameters
    assert "compiled_graph_key" in parameters
    assert "artifact" not in parameters


def test_arm_state_fence_goes_red_when_marker_precedes_tcg_bind(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    tests = tmp_path / "tests"
    src.mkdir()
    tests.mkdir()
    (src / "aot_serve.py").write_text(
        "def arm_compiled_graph(pipeline, cfg, compiled_graph_key):\n"
        "    loaded = compiled_graph_store.load_runner(compiled_graph_key)\n"
        "    marker = _marker(pipeline)\n"
        "    loaded.runner.bind({}, device='cpu')\n"
        "    wrap_module(pipeline, marker, {})\n",
        encoding="utf-8",
    )
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("", encoding="utf-8")
    lint = Path(__file__).resolve().parents[1] / "scripts" / \
        "lint_arm_state_feeders.py"

    result = subprocess.run(
        [
            sys.executable,
            str(lint),
            "--src", str(src),
            "--tests", str(tests),
            "--allowlist", str(allowlist),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "bind must complete before _marker/wrap_module" in result.stderr
