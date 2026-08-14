"""Producer-side proof for the compiled-graph protocol hard cut (th#1947)."""

from __future__ import annotations

import dataclasses

from gen_worker import cell_resolve, fleet_cells, receipts
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
