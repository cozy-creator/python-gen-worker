"""Parent-side mint folding consumes the closed TCG artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gen_worker import aot_compile_pool, aot_mint


def _metadata(name: str, key: str, class_hash: str) -> dict[str, Any]:
    return {
        "compiled_graph_format": 1,
        "kind": "aot-inductor",
        "compiled_graph_key": key,
        "graph_class": {
            "name": name,
            "class_hash": class_hash,
            "graph": {},
        },
        "sm": "cpu-test",
        "toolchain": {"torch": "test"},
    }


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    packed: dict[str, aot_compile_pool.PackedGraphClass],
) -> aot_mint.MintResult:
    class _Pool:
        entry_seconds: dict[str, float] = {}
        peak_rss_bytes = 0

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def compile(self, *_args: Any, **_kwargs: Any) -> Any:
            return packed

    monkeypatch.setattr(aot_compile_pool, "EntryCompilePool", _Pool)
    monkeypatch.setattr(aot_mint, "_pool_facts", lambda _pool: {})
    monkeypatch.setattr(
        aot_mint,
        "canonicalize_packed_classes",
        lambda _blocks, _metas: {},
    )
    width = aot_compile_pool.entry_workers(
        max(1, len(packed)),
        limit=2,
        vcpus=16,
        available_bytes=64 * 1024**3,
        device_lock=True,
    )
    return aot_mint.mint_graph_classes(
        aot_compile_pool.EntryJob(
            function="generate",
            modules=("m",),
            out_dir="/tmp",
        ),
        workdir=Path("/tmp/pgw1270-tcg-parent"),
        width=width,
        spec=aot_mint.ExportSpec(family="test", target="unet"),
    )


def test_parent_preserves_closed_tcg_metadata_and_records_alias_outside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "cg-key-v1-" + "a" * 56
    rows = {
        name: aot_compile_pool.PackedGraphClass(
            name=name,
            key=key,
            artifact=f"/tmp/{key}.tar.gz",
            metadata=json.dumps(_metadata(name, key, "0123456789abcdef")),
        )
        for name in ("unet/a", "unet/b")
    }

    result = _drive(monkeypatch, rows)

    assert len(result.entries) == 1
    survivor = result.entries[0]
    assert survivor.key == key
    assert survivor.aliases == ("unet/b",)
    assert survivor.mint_phases
    assert set(survivor.metadata) == {
        "compiled_graph_format",
        "kind",
        "compiled_graph_key",
        "graph_class",
        "sm",
        "toolchain",
    }
    assert "aliases" not in survivor.metadata["graph_class"]
    assert "manifest_digest" not in survivor.metadata
    assert "mint_phases" not in survivor.metadata


def test_parent_refuses_a_child_key_not_repeated_by_tcg_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "cg-key-v1-" + "b" * 56
    meta = _metadata("unet/a", "cg-key-v1-" + "c" * 56, "fedcba9876543210")
    row = aot_compile_pool.PackedGraphClass(
        name="unet/a",
        key=key,
        artifact=f"/tmp/{key}.tar.gz",
        metadata=json.dumps(meta),
    )

    with pytest.raises(
        aot_mint.MintRefused,
        match="exact name and compiled_graph_key",
    ):
        _drive(monkeypatch, {row.name: row})
