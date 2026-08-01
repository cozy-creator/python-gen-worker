"""th#1362 item 2: repos WE pull are de-sharded on the way in.

Drives the REAL ingest path — `build_flavor_tree` for the pure pass-through
mirror (dtype="source"), and `deshard_mirror_tree` for the tree walk — over
REAL safetensors files written by the real safetensors library. Nothing about
the bytes is faked, because the property under test is that the tensors come
out identical while the file layout does not.

The complementary half of the ruling is pinned here too: READ tolerance is
permanent, so a sharded tree must still load after all of this.

Run: pytest tests/test_deshard_mirror_th1362.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
from safetensors import safe_open  # noqa: E402

from gen_worker.convert.clone import (  # noqa: E402
    OutputSpec,
    build_flavor_tree,
    deshard_mirror_tree,
    tree_has_sharded_safetensors,
)
from gen_worker.convert.ingest import IngestedSource  # noqa: E402

from gen_worker.convert.writer import merge_safetensors_by_offset  # noqa: E402


def _tensors(seed: int, n: int = 4) -> dict[str, "torch.Tensor"]:
    g = torch.Generator().manual_seed(seed)
    return {
        f"block.{i}.weight": torch.randn(8, 8, generator=g, dtype=torch.float32).to(
            torch.bfloat16)
        for i in range(n)
    }


def _write_sharded(
    out_dir: Path, prefix: str, tensors: dict[str, "torch.Tensor"], per_shard: int,
    metadata: dict[str, str] | None = None,
) -> Path:
    """A real HF shard set: N member files plus the index that maps into them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(tensors)
    groups = [names[i:i + per_shard] for i in range(0, len(names), per_shard)]
    total = len(groups)
    weight_map: dict[str, str] = {}
    for i, group in enumerate(groups, start=1):
        member = f"{prefix}-{i:05d}-of-{total:05d}.safetensors"
        safetensors_torch.save_file(
            {k: tensors[k] for k in group}, str(out_dir / member),
            metadata=metadata or {"format": "pt"},
        )
        for k in group:
            weight_map[k] = member
    index = out_dir / f"{prefix}.safetensors.index.json"
    index.write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size()
                                       for t in tensors.values())},
        "weight_map": weight_map,
    }), encoding="utf-8")
    return index


def _load(path: Path) -> dict[str, "torch.Tensor"]:
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _load_sharded(index: Path) -> dict[str, "torch.Tensor"]:
    weight_map = json.loads(index.read_text())["weight_map"]
    out = {}
    for member in sorted(set(weight_map.values())):
        out.update(_load(index.parent / member))
    return out


# --------------------------------------------------------------------------

def test_a_shard_set_merges_to_one_file_with_identical_tensors(tmp_path):
    tensors = _tensors(1, 6)
    index = _write_sharded(tmp_path / "transformer", "diffusion_pytorch_model",
                           tensors, per_shard=2)
    assert len(list((tmp_path / "transformer").glob("*.safetensors"))) == 3

    n = deshard_mirror_tree(tmp_path)
    assert n == 1

    comp = tmp_path / "transformer"
    weights = sorted(p.name for p in comp.glob("*.safetensors"))
    assert weights == ["diffusion_pytorch_model.safetensors"]
    assert not index.exists()
    assert not list(comp.glob("*.index.json"))

    got = _load(comp / "diffusion_pytorch_model.safetensors")
    assert set(got) == set(tensors)
    for k, want in tensors.items():
        assert got[k].dtype == want.dtype and got[k].shape == want.shape
        assert torch.equal(got[k], want)


def test_metadata_survives_the_merge(tmp_path):
    tensors = _tensors(2, 4)
    _write_sharded(tmp_path / "unet", "diffusion_pytorch_model", tensors,
                   per_shard=2, metadata={"format": "pt", "origin": "upstream"})
    deshard_mirror_tree(tmp_path)
    with safe_open(str(tmp_path / "unet" / "diffusion_pytorch_model.safetensors"),
                   framework="pt", device="cpu") as f:
        md = f.metadata()
    assert md.get("origin") == "upstream"


def test_every_component_is_desharded_and_unsharded_ones_are_untouched(tmp_path):
    a = _tensors(3, 4)
    b = _tensors(4, 2)
    _write_sharded(tmp_path / "transformer", "diffusion_pytorch_model", a, per_shard=2)
    _write_sharded(tmp_path / "text_encoder", "model", b, per_shard=1)
    (tmp_path / "vae").mkdir()
    safetensors_torch.save_file(
        _tensors(5, 2), str(tmp_path / "vae" / "diffusion_pytorch_model.safetensors"))
    vae_before = (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").read_bytes()
    (tmp_path / "model_index.json").write_text('{"_class_name": "X"}')

    assert tree_has_sharded_safetensors(tmp_path)
    assert deshard_mirror_tree(tmp_path) == 2
    assert not tree_has_sharded_safetensors(tmp_path)

    for comp in ("transformer", "text_encoder", "vae"):
        weights = sorted(p.name for p in (tmp_path / comp).glob("*.safetensors"))
        assert len(weights) == 1, f"{comp} is not one file per component: {weights}"
    # A component that was never sharded is not rewritten.
    assert (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").read_bytes() \
        == vae_before
    assert (tmp_path / "model_index.json").exists()


def test_an_index_that_lies_about_its_bytes_is_refused_at_ingest(tmp_path):
    """The klein-4b bug class: the index names a tensor the shards do not hold.
    Catching it here, at ingest, is the point — it used to surface as an
    unloadable published checkpoint on a GPU pod."""
    tensors = _tensors(6, 4)
    index = _write_sharded(tmp_path / "text_encoder", "model", tensors, per_shard=2)
    payload = json.loads(index.read_text())
    payload["weight_map"]["block.99.weight"] = "model-00001-of-00002.safetensors"
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="index names"):
        deshard_mirror_tree(tmp_path)


def test_a_missing_shard_member_is_refused(tmp_path):
    tensors = _tensors(7, 4)
    _write_sharded(tmp_path / "unet", "diffusion_pytorch_model", tensors, per_shard=2)
    (tmp_path / "unet" / "diffusion_pytorch_model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing shard"):
        deshard_mirror_tree(tmp_path)


def test_shards_that_both_define_a_tensor_are_refused(tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    t = _tensors(8, 2)
    safetensors_torch.save_file(t, str(a))
    safetensors_torch.save_file(t, str(b))
    with pytest.raises(ValueError, match="both define"):
        merge_safetensors_by_offset([a, b], tmp_path / "merged.safetensors")


def test_sharded_reads_still_work_after_the_ruling(tmp_path):
    """Read tolerance is PERMANENT, not transitional: nothing in this change
    may make a sharded tree unreadable."""
    tensors = _tensors(9, 4)
    index = _write_sharded(tmp_path / "unet", "diffusion_pytorch_model", tensors,
                           per_shard=2)
    got = _load_sharded(index)
    assert set(got) == set(tensors)
    for k, want in tensors.items():
        assert torch.equal(got[k], want)


# --------------------------------------------------------------------------
# The real mirror path
# --------------------------------------------------------------------------

def _source(tmp_path: Path, layout: str = "diffusers") -> IngestedSource:
    return IngestedSource(
        provider="huggingface", source_ref="acme/thing", source_revision="deadbeef",
        dir=tmp_path, layout=layout, model_family="sdxl", model_family_variant="sdxl",
        attrs={"dtype": "bf16", "file_layout": layout, "file_type": "safetensors"},
    )


def test_pure_passthrough_mirror_is_desharded(tmp_path):
    """dtype="source" is the purest mirror there is — it rewrites no tensor
    values at all — and the ruling says it de-shards anyway, so that the corpus
    we own has ONE shape."""
    src = tmp_path / "src"
    tensors = _tensors(10, 6)
    _write_sharded(src / "transformer", "diffusion_pytorch_model", tensors, per_shard=3)
    (src / "model_index.json").write_text('{"_class_name": "X"}')

    out = tmp_path / "out"
    tree, attrs = build_flavor_tree(
        _source(src), OutputSpec(dtype="source", file_layout="diffusers",
                                 file_type="safetensors"), out)

    assert attrs["dtype"] == "bf16"
    weights = sorted(p.name for p in (tree / "transformer").glob("*.safetensors"))
    assert weights == ["diffusion_pytorch_model.safetensors"]
    assert not list((tree / "transformer").glob("*.index.json"))
    got = _load(tree / "transformer" / "diffusion_pytorch_model.safetensors")
    for k, want in tensors.items():
        assert torch.equal(got[k], want)
    # The source is someone else's tree and must be left exactly as ingested.
    assert (src / "transformer" /
            "diffusion_pytorch_model.safetensors.index.json").exists()


def test_dtype_matching_mirror_is_desharded(tmp_path):
    """The other pass-through: requested dtype already equals the source's, so
    no cast runs. It must still come out one-file-per-component."""
    src = tmp_path / "src"
    tensors = _tensors(11, 4)
    _write_sharded(src / "transformer", "diffusion_pytorch_model", tensors, per_shard=2)
    (src / "model_index.json").write_text('{"_class_name": "X"}')

    out = tmp_path / "out"
    tree, _ = build_flavor_tree(
        _source(src), OutputSpec(dtype="bf16", file_layout="diffusers",
                                 file_type="safetensors"), out)
    weights = sorted(p.name for p in (tree / "transformer").glob("*.safetensors"))
    assert weights == ["diffusion_pytorch_model.safetensors"]
    got = _load(tree / "transformer" / "diffusion_pytorch_model.safetensors")
    for k, want in tensors.items():
        assert torch.equal(got[k], want)
