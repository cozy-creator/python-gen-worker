"""Shared precision-ladder conformance vectors (th#697).

tests/testdata/precision_ladder_vectors.json is vendored byte-identically in
tensorhub (internal/orchestrator/precision/testdata/) — the Go resolver runs
the SAME file. Any ladder semantic change edits both copies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker.models.ladder import (
    FlavorRow,
    LadderModel,
    placement_from_metadata,
    resolve,
)

_VECTORS = json.loads(
    (Path(__file__).parent / "testdata" / "precision_ladder_vectors.json").read_text()
)["vectors"]


def _model(spec: dict) -> LadderModel:
    rows = tuple(
        FlavorRow(
            token=token,
            size_gb=float(row.get("size_gb", 0.0)),
            placement=placement_from_metadata(row.get("placement")),
        )
        for token, row in (spec.get("flavors") or {}).items()
    )
    return LadderModel(
        base_size_gb=float(spec.get("base_size_gb", 0.0)),
        fp8_cast_vram_gb=float(spec.get("fp8_cast_vram_gb", 0.0)),
        flavors=rows,
        castable=bool(spec.get("castable", True)),
    )


@pytest.mark.parametrize("vec", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_precision_ladder_vector(vec: dict) -> None:
    got = resolve(
        _model(vec["model"]),
        gpu_sm=int(vec["gpu_sm"]),
        free_vram_gb=float(vec["free_vram_gb"]),
        libs=vec.get("libs") or (),
        quality_floor=str(vec.get("quality_floor", "")),
        local=bool(vec.get("local", False)),
        allow_offload=bool(vec.get("allow_offload", True)),
    )
    expect = vec["expect"]
    if expect.get("refusal"):
        assert got.refusal == expect["refusal"], got
    else:
        assert got.refusal == "", got
        assert got.flavor == expect.get("flavor", ""), got
        assert got.cast == expect.get("cast", ""), got
        assert got.mode == expect.get("mode", "native"), got


def test_vector_count_guard() -> None:
    assert len(_VECTORS) >= 25
