from __future__ import annotations

import pytest

from gen_worker.worker import RequestContext


def test_partition_context_exposes_lineage_and_item_fields() -> None:
    ctx = RequestContext(
        "req-1",
        parent_request_id="parent-1",
        child_request_id="child-1",
        item_id="item-42",
        item_index=42,
        item_span={"start": 40, "end": 44},
    )
    got = ctx.partition_context()
    assert got == {
        "request_id": "req-1",
        "parent_request_id": "parent-1",
        "child_request_id": "child-1",
        "item_id": "item-42",
        "item_index": 42,
        "item_span": {"start": 40, "end": 44},
    }


def test_item_output_ref_prefers_item_id_when_present() -> None:
    ctx = RequestContext("req-2", item_id="item-custom", item_index=7)
    ref = ctx.item_output_ref("/result.json")
    assert ref == "runs/req-2/outputs/items/item-custom/result.json"


def test_item_output_ref_falls_back_to_index_then_default() -> None:
    with_index = RequestContext("req-3", item_index=7)
    assert with_index.item_output_ref("out.bin") == "runs/req-3/outputs/items/item-000007/out.bin"

    without_index = RequestContext("req-4")
    assert without_index.item_output_ref("out.bin") == "runs/req-4/outputs/items/item-000000/out.bin"

    with pytest.raises(ValueError):
        without_index.item_output_ref("")
