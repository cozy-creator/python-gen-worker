"""Publish derives identity only through torch-compiled-graphs."""

from __future__ import annotations

from typing import Any

import pytest
from torch_compiled_graphs.identity import from_artifact_metadata

from gen_worker import fleet_cells


def _metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "kind": "aot-inductor",
        "graph_class": {"class_hash": "0123456789abcdef"},
        "sm": "sm_89",
        "toolchain": {"torch": "2.13.0", "triton": "3.5.0"},
    }
    meta["compiled_graph_key"] = from_artifact_metadata(meta).value
    return meta


def test_publish_entry_is_exactly_the_three_tcg_axes() -> None:
    meta = _metadata()

    entry = fleet_cells.intent_entry(meta, mint_duration_ms=17)

    identity = from_artifact_metadata(meta)
    assert entry.compiled_graph_key == identity.value
    assert entry.identity_axes == identity.as_dict()
    assert set(entry.identity_axes) == {"graph", "sm", "toolchain"}
    assert entry.mint_duration_ms == 17


def test_publish_refuses_a_stamp_that_disagrees_with_tcg() -> None:
    meta = _metadata()
    meta["compiled_graph_key"] = "cg-key-v1-" + "f" * 56

    with pytest.raises(fleet_cells.CellPublishRefused, match="disagrees"):
        fleet_cells.intent_entry(meta)


def test_pod_axes_come_from_the_live_worker_not_artifact_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = _metadata()
    entry = fleet_cells.intent_entry(meta)
    publisher = fleet_cells.CellPublisher(
        base_url="https://hub.invalid",
        worker_jwt=lambda: "worker-token",
        image_digest="sha256:image",
    )
    sent: dict[str, Any] = {}
    monkeypatch.setattr(
        fleet_cells.cc,
        "runtime_key",
        lambda: {"sku": "l4"},
    )
    monkeypatch.setattr(
        fleet_cells.cc,
        "gen_worker_version",
        lambda: "0.116.0",
    )

    def post(_path: str, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        sent.update(payload)
        return {
            "repo": "root/family-sdxl",
            "answers": [{
                "compiled_graph_key": entry.compiled_graph_key,
                "status": "granted",
                "capability_token": "grant",
            }],
        }

    monkeypatch.setattr(publisher, "_post", post)
    publisher.publish_intent("sdxl", [entry])

    assert sent["axes"] == {
        "sku": "l4",
        "image_digest": "sha256:image",
        "gen_worker": "0.116.0",
    }
    assert "sku" not in meta and "gen_worker" not in meta
