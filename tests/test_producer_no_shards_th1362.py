from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
from safetensors import safe_open  # noqa: E402

from gen_worker.convert.writer import (  # noqa: E402
    ConversionImplementationError,
    assert_one_file_per_component,
    find_producer_shards,
    streaming_cast_snapshot,
)


def _tensors(seed: int, n: int) -> dict[str, "torch.Tensor"]:
    g = torch.Generator().manual_seed(seed)
    return {
        f"layer.{i}.weight": torch.randn(16, 16, generator=g, dtype=torch.float32)
        for i in range(n)
    }


def _sharded_component(comp: Path, prefix: str, tensors, per_shard: int) -> None:
    comp.mkdir(parents=True, exist_ok=True)
    names = list(tensors)
    groups = [names[i:i + per_shard] for i in range(0, len(names), per_shard)]
    weight_map = {}
    for i, group in enumerate(groups, start=1):
        member = f"{prefix}-{i:05d}-of-{len(groups):05d}.safetensors"
        safetensors_torch.save_file({k: tensors[k] for k in group},
                                    str(comp / member))
        for k in group:
            weight_map[k] = member
    (comp / f"{prefix}.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))


def test_a_real_cast_of_a_sharded_source_emits_one_file_per_component(tmp_path):
    """Read a SHARDED input, write an UNSHARDED output — both halves of the ruling in one run of the real streaming cast."""
    src = tmp_path / "src"
    unet = _tensors(1, 8)
    te = _tensors(2, 4)
    _sharded_component(src / "unet", "diffusion_pytorch_model", unet, per_shard=2)
    _sharded_component(src / "text_encoder", "model", te, per_shard=2)
    (src / "model_index.json").write_text('{"_class_name": "X"}')

    out = tmp_path / "out"
    streaming_cast_snapshot(src, out, target_dtype=torch.bfloat16,
                            file_layout="multi-file")

    assert find_producer_shards(out) == []
    for comp, prefix in (("unet", "diffusion_pytorch_model"), ("text_encoder", "model")):
        weights = sorted(p.name for p in (out / comp).glob("*.safetensors"))
        assert weights == [f"{prefix}.safetensors"], f"{comp}: {weights}"
        assert not list((out / comp).glob("*.index.json"))

    got = {}
    with safe_open(str(out / "unet" / "diffusion_pytorch_model.safetensors"),
                   framework="pt", device="cpu") as f:
        for k in f.keys():
            got[k] = f.get_tensor(k)
    assert set(got) == set(unet)
    for k, want in unet.items():
        assert torch.equal(got[k], want.to(torch.bfloat16))


def test_a_cast_of_an_unsharded_source_stays_one_file(tmp_path):
    src = tmp_path / "src"
    (src / "transformer").mkdir(parents=True)
    t = _tensors(3, 4)
    safetensors_torch.save_file(
        t, str(src / "transformer" / "diffusion_pytorch_model.safetensors"))
    (src / "model_index.json").write_text('{"_class_name": "X"}')

    out = tmp_path / "out"
    streaming_cast_snapshot(src, out, target_dtype=torch.float16,
                            file_layout="multi-file")
    assert find_producer_shards(out) == []
    assert sorted(p.name for p in (out / "transformer").glob("*.safetensors")) \
        == ["diffusion_pytorch_model.safetensors"]


def test_the_invariant_refuses_a_shard_set_a_producer_left_behind(tmp_path):
    """save_pretrained shards on its own, so a producer that forgets to say otherwise must be caught rather than trusted."""
    _sharded_component(tmp_path / "unet", "diffusion_pytorch_model",
                       _tensors(4, 4), per_shard=2)
    with pytest.raises(ConversionImplementationError,
                       match="sharded_producer_output"):
        assert_one_file_per_component(tmp_path, producer="test producer")


def test_the_invariant_refuses_a_bare_shard_member_with_no_index(tmp_path):
    comp = tmp_path / "unet"
    comp.mkdir()
    safetensors_torch.save_file(
        _tensors(5, 2),
        str(comp / "diffusion_pytorch_model-00001-of-00002.safetensors"))
    with pytest.raises(ConversionImplementationError,
                       match="sharded_producer_output"):
        assert_one_file_per_component(tmp_path, producer="test producer")


def test_the_invariant_passes_a_one_file_per_component_tree(tmp_path):
    for comp, prefix in (("unet", "diffusion_pytorch_model"),
                         ("text_encoder", "model"),
                         ("vae", "diffusion_pytorch_model")):
        (tmp_path / comp).mkdir()
        safetensors_torch.save_file(
            _tensors(6, 2), str(tmp_path / comp / f"{prefix}.safetensors"))
    (tmp_path / "model_index.json").write_text("{}")
    assert_one_file_per_component(tmp_path, producer="test producer")


def test_the_invariant_does_not_fire_on_dtype_variants(tmp_path):
    """Two unrelated weight files in one component are not a shard set — the check is narrow on purpose, so it cannot turn into a general 'one weight file' rule that refuses legitimate trees."""
    comp = tmp_path / "unet"
    comp.mkdir()
    safetensors_torch.save_file(
        _tensors(7, 2), str(comp / "diffusion_pytorch_model.safetensors"))
    safetensors_torch.save_file(
        _tensors(7, 2), str(comp / "diffusion_pytorch_model.fp16.safetensors"))
    assert_one_file_per_component(tmp_path, producer="test producer")


def test_a_single_produced_file_is_accepted(tmp_path):
    """The training-promote shape: one LoRA file, not a tree."""
    lora = tmp_path / "my-lora.safetensors"
    safetensors_torch.save_file(_tensors(8, 2), str(lora))
    assert_one_file_per_component(lora, producer="training promote")


def test_a_sharded_artifact_still_reads(tmp_path):
    """Read tolerance is PERMANENT."""
    tensors = _tensors(9, 4)
    _sharded_component(tmp_path / "unet", "diffusion_pytorch_model", tensors,
                       per_shard=2)
    index = tmp_path / "unet" / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    got = {}
    for member in sorted(set(weight_map.values())):
        with safe_open(str(index.parent / member), framework="pt", device="cpu") as f:
            for k in f.keys():
                got[k] = f.get_tensor(k)
    assert set(got) == set(tensors)
    for k, want in tensors.items():
        assert torch.equal(got[k], want)
