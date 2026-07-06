"""Unit tests for the ONE streaming shard writer (cozy_convert.writer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from cozy_convert.writer import (
    MAX_SAFETENSORS_SHARD_BYTES,
    IncrementalSafetensorsWriter,
    materialize_pickle_to_safetensors,
    plan_shards,
    shard_safetensors_by_offset,
    streaming_dtype_cast,
    torch_dtype_to_st,
)


def test_max_shard_bytes_stays_clear_of_the_tunnel_timeout() -> None:
    """Regression guard for e2e tracker #110: tensorhub's per-upload
    /complete verifies a shard synchronously in one HTTP request, and a
    consistent ~300s ceiling in front of it (observed live; almost
    certainly the ngrok tunnel this stack rides on) deterministically
    kills any shard whose verify time exceeds it -- a 9.8GB shard failed
    identically on every one of 5 retries. This must stay meaningfully
    below the old 5GB (HF default) that caused it, not silently drift
    back up."""
    assert MAX_SAFETENSORS_SHARD_BYTES <= 2 * 1024 * 1024 * 1024


def _make_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return path


class TestPlanShards:
    def test_single_shard_when_under_cap(self) -> None:
        plan = plan_shards({"a": 10, "b": 20}, max_shard_bytes=100)
        assert plan.shard_names == ["model.safetensors"]
        assert plan.weight_map == {"a": "model.safetensors", "b": "model.safetensors"}
        assert plan.total_size == 30

    def test_multi_shard_first_fit_sorted(self) -> None:
        plan = plan_shards({"a": 60, "b": 60, "c": 60}, max_shard_bytes=100)
        assert len(plan.shard_names) == 3
        assert plan.shard_names[0] == "model-00001-of-00003.safetensors"
        assert sum(plan.shard_sizes.values()) == plan.total_size == 180

    def test_tensor_larger_than_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="tensor_exceeds_max_shard_bytes"):
            plan_shards({"big": 200}, max_shard_bytes=100)


class TestIncrementalWriter:
    def test_roundtrip_matches_save_file(self, tmp_path: Path) -> None:
        tensors = {
            "w": torch.randn(16, 8, dtype=torch.float32),
            "b": torch.randn(8, dtype=torch.float16),
            "i": torch.arange(10, dtype=torch.int64),
        }
        out = tmp_path / "inc.safetensors"
        with IncrementalSafetensorsWriter(out) as w:
            for name, t in tensors.items():
                w.add_tensor_metadata(name, dtype=torch_dtype_to_st(t.dtype), shape=list(t.shape))
            w.write_header()
            for name, t in tensors.items():
                w.write_tensor(name, t.contiguous().flatten().view(torch.uint8).numpy().tobytes())
        loaded = load_file(str(out))
        for name, t in tensors.items():
            assert torch.equal(loaded[name], t), name

    def test_out_of_order_write_rejected(self, tmp_path: Path) -> None:
        with IncrementalSafetensorsWriter(tmp_path / "x.safetensors") as w:
            w.add_tensor_metadata("a", dtype="F32", shape=[1])
            w.add_tensor_metadata("b", dtype="F32", shape=[1])
            w.write_header()
            with pytest.raises(RuntimeError, match="expected tensor 'a'"):
                w.write_tensor("b", b"\x00" * 4)


class TestStreamingDtypeCast:
    def test_fp32_to_fp16_values_and_nonfloat_passthrough(self, tmp_path: Path) -> None:
        src = _make_safetensors(tmp_path / "in.safetensors", {
            "w": torch.randn(32, 32, dtype=torch.float32),
            "idx": torch.arange(6, dtype=torch.int32),
        })
        result = streaming_dtype_cast(src, tmp_path / "out", target_dtype=torch.float16)
        assert result["tensor_count"] == 2
        assert result["converted_count"] == 1
        assert result["index_path"] is None
        loaded = load_file(str(result["output_paths"][0]))
        assert loaded["w"].dtype == torch.float16
        assert loaded["idx"].dtype == torch.int32
        orig = load_file(str(src))
        assert torch.allclose(loaded["w"].float(), orig["w"], atol=1e-3)

    def test_sharded_output_with_index(self, tmp_path: Path) -> None:
        tensors = {f"t{i}": torch.randn(64, 64, dtype=torch.float32) for i in range(4)}
        src = _make_safetensors(tmp_path / "in.safetensors", tensors)
        # each fp16 tensor = 8 KB; force multi-shard with a 20 KB cap
        result = streaming_dtype_cast(
            src, tmp_path / "out", target_dtype=torch.float16, shard_threshold=20 * 1024)
        assert len(result["output_paths"]) > 1
        index_path = result["index_path"]
        assert index_path is not None and index_path.exists()
        weight_map = json.loads(index_path.read_text())["weight_map"]
        merged: dict[str, torch.Tensor] = {}
        for shard in result["output_paths"]:
            merged.update(load_file(str(shard)))
        assert set(merged) == set(tensors) == set(weight_map)

    def test_sharded_input_via_index(self, tmp_path: Path) -> None:
        a = _make_safetensors(tmp_path / "m-00001-of-00002.safetensors",
                              {"a": torch.randn(4, 4)})
        _make_safetensors(tmp_path / "m-00002-of-00002.safetensors",
                          {"b": torch.randn(4, 4)})
        index = tmp_path / "m.safetensors.index.json"
        index.write_text(json.dumps({
            "metadata": {"total_size": 128},
            "weight_map": {"a": a.name, "b": "m-00002-of-00002.safetensors"},
        }))
        result = streaming_dtype_cast(index, tmp_path / "out", target_dtype=torch.bfloat16)
        merged = load_file(str(result["output_paths"][0]))
        assert set(merged) == {"a", "b"}
        assert merged["a"].dtype == torch.bfloat16


class TestShardByOffset:
    def test_raw_reshard_roundtrip(self, tmp_path: Path) -> None:
        tensors = {f"k{i}": torch.randn(128, dtype=torch.float32) for i in range(8)}
        src = _make_safetensors(tmp_path / "big.safetensors", tensors)
        shards, index, sizes = shard_safetensors_by_offset(
            src, tmp_path / "shards", max_shard_bytes=1500, shard_prefix="big")
        assert len(shards) > 1
        merged: dict[str, torch.Tensor] = {}
        for shard in shards:
            merged.update(load_file(str(shard)))
        for name, t in tensors.items():
            assert torch.equal(merged[name], t), name
        weight_map = json.loads(index.read_text())["weight_map"]
        assert set(weight_map) == set(tensors)


class TestPickleMaterialize:
    def test_pickle_to_safetensors(self, tmp_path: Path) -> None:
        state = {"layer.weight": torch.randn(4, 4), "layer.bias": torch.randn(4)}
        pkl = tmp_path / "model.ckpt"
        torch.save(state, str(pkl))
        out = materialize_pickle_to_safetensors(pkl, tmp_path / "work")
        loaded = load_file(str(out))
        assert torch.equal(loaded["layer.weight"], state["layer.weight"])
