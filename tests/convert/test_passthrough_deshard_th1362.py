"""th#1362: a produced flavor's PASSTHROUGH components are one file too.

The gap this closes, found live on te#137: `build_svdq_flavor_tree` marries our
4-bit denoiser to the base checkpoint's other components by hardlinking them
through `copy_non_weight_files`. qwen-image's text_encoder was mirrored before
th#1362 item 2 landed, so it is still a 9-member shard set — and it rode
straight into the produced tree, where `publish_flavors`' producer invariant
refused it:

    ConversionImplementationError: sharded_producer_output:
    publish_flavors[tensorhub/qwen-image] emitted a shard set (10 file(s), e.g.
    text_encoder/model-00001-of-00009.safetensors ...)

The guard was RIGHT — a flavor is our artifact whole, not just the component we
computed. What was missing is the normalization: item 2 wired de-shard into
clone.py's mirror arms only, so the married-tree producers passed legacy shards
through untouched. It now happens at the door every passthrough weight enters a
produced tree.

Real trees, real safetensors, real `publish_flavors` over HTTP to the fake hub —
the assertion is on the FILE LIST THAT GOES ON THE WIRE, not on an internal call.

    pytest tests/convert/test_passthrough_deshard_th1362.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
from safetensors import safe_open  # noqa: E402

from gen_worker.convert.produced import ProducedFlavor  # noqa: E402
from gen_worker.convert.publish import publish_flavors  # noqa: E402
from gen_worker.convert.svdq import build_svdq_flavor_tree  # noqa: E402
from gen_worker.convert.writer import ConversionImplementationError  # noqa: E402

from fake_hub import _FakeHub  # noqa: E402


class _Ctx:
    def __init__(self, base_url: str) -> None:
        self._file_api_base_url = base_url
        self._worker_capability_token = "cap-token"
        self.owner = "cozy"
        self.lines: list[str] = []

    def log(self, message: str, **fields: Any) -> None:
        self.lines.append(message)


def _tensors(seed: int, n: int) -> dict[str, "torch.Tensor"]:
    g = torch.Generator().manual_seed(seed)
    return {f"enc.{i}.weight": torch.randn(8, 8, generator=g) for i in range(n)}


def _write_sharded(comp: Path, prefix: str, tensors, per_shard: int) -> None:
    comp.mkdir(parents=True, exist_ok=True)
    names = list(tensors)
    groups = [names[i:i + per_shard] for i in range(0, len(names), per_shard)]
    weight_map: dict[str, str] = {}
    for i, group in enumerate(groups, start=1):
        member = f"{prefix}-{i:05d}-of-{len(groups):05d}.safetensors"
        safetensors_torch.save_file({k: tensors[k] for k in group},
                                    str(comp / member))
        for k in group:
            weight_map[k] = member
    (comp / f"{prefix}.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))


def _svdq_file(path: Path) -> Path:
    """A real nunchaku single-file checkpoint — self-describing metadata is
    what `detect_svdq_artifact` keys on, so this must be genuine."""
    safetensors_torch.save_file(
        _tensors(90, 2), str(path),
        metadata={
            "model_class": "NunchakuQwenImageTransformer2DModel",
            "quantization_config": json.dumps(
                {"method": "svdquant", "weight": {"dtype": "nvfp4"}, "rank": 128}),
        })
    return path


def _base_tree(root: Path, *, sharded_text_encoder: bool) -> dict[str, "torch.Tensor"]:
    """The qwen-image shape: a text_encoder (sharded or not), a vae, and the
    transformer whose weights the svdq file replaces."""
    te = _tensors(1, 18)
    if sharded_text_encoder:
        _write_sharded(root / "text_encoder", "model", te, per_shard=2)
    else:
        (root / "text_encoder").mkdir(parents=True)
        safetensors_torch.save_file(te, str(root / "text_encoder" / "model.safetensors"))
    (root / "vae").mkdir(parents=True)
    safetensors_torch.save_file(
        _tensors(2, 2), str(root / "vae" / "diffusion_pytorch_model.safetensors"))
    (root / "transformer").mkdir(parents=True)
    safetensors_torch.save_file(
        _tensors(3, 2),
        str(root / "transformer" / "diffusion_pytorch_model.safetensors"))
    (root / "model_index.json").write_text('{"_class_name": "QwenImagePipeline"}')
    (root / "scheduler").mkdir(parents=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}")
    return te


def _publish_svdq_flavor(fake_hub: Any, tmp_path: Path, *, sharded: bool):
    base = tmp_path / "base"
    te = _base_tree(base, sharded_text_encoder=sharded)
    tree, attrs = build_svdq_flavor_tree(
        base, _svdq_file(tmp_path / "svdq-fp4_r128-cozy.safetensors"),
        tmp_path / "flavor")
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    results = publish_flavors(
        ctx, [ProducedFlavor(path=str(tree), flavor=attrs["flavor"],
                             attributes=attrs)],
        destination_repo="cozy/qwen-image", tags=["svdq-fp4-r128"])
    paths = sorted(f["path"] for f in _FakeHub.state["publish_request"]["files"])
    return base, tree, te, paths, results


# --------------------------------------------------------------------------
# The multi-shard producer output — the te#137 case
# --------------------------------------------------------------------------

def test_a_sharded_passthrough_component_publishes_as_ONE_file(
    fake_hub: Any, tmp_path: Path,
) -> None:
    base, tree, te, paths, results = _publish_svdq_flavor(
        fake_hub, tmp_path, sharded=True)

    assert paths == [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.safetensors",
        "transformer/svdq-fp4_r128-cozy.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ], paths
    assert results and results[0].checkpoint_id

    # Every tensor survived the merge, byte-exact.
    got = {}
    with safe_open(str(tree / "text_encoder" / "model.safetensors"),
                   framework="pt", device="cpu") as f:
        for k in f.keys():
            got[k] = f.get_tensor(k)
    assert set(got) == set(te)
    for k, want in te.items():
        assert torch.equal(got[k], want), k


def test_the_source_snapshot_is_not_mutated(fake_hub: Any, tmp_path: Path) -> None:
    """The passthrough files are HARDLINKS into the source snapshot. Collapsing
    them must unlink OUR tree's names and nothing else — the ingested source
    stays exactly as the mirror left it, so a second flavor off the same source
    still has its input."""
    base, _tree, _te, _paths, _ = _publish_svdq_flavor(
        fake_hub, tmp_path, sharded=True)
    members = sorted(p.name for p in (base / "text_encoder").iterdir())
    assert members == [f"model-{i:05d}-of-00009.safetensors" for i in range(1, 10)] \
        + ["model.safetensors.index.json"], members


# --------------------------------------------------------------------------
# The single-file producer output — the other half, so the fix cannot be a
# rewrite-everything pass
# --------------------------------------------------------------------------

def test_an_unsharded_passthrough_component_publishes_unchanged(
    fake_hub: Any, tmp_path: Path,
) -> None:
    base, tree, te, paths, results = _publish_svdq_flavor(
        fake_hub, tmp_path, sharded=False)

    assert paths == [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.safetensors",
        "transformer/svdq-fp4_r128-cozy.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ], paths
    assert results and results[0].checkpoint_id

    # Untouched means UNTOUCHED: still the same inode the source holds, so a
    # component that needed nothing done to it costs no bytes and no rewrite.
    src = base / "text_encoder" / "model.safetensors"
    dst = tree / "text_encoder" / "model.safetensors"
    assert src.stat().st_ino == dst.stat().st_ino
    assert src.read_bytes() == dst.read_bytes()


# --------------------------------------------------------------------------
# The guard is a BACKSTOP, not a formality — it must still fail closed
# --------------------------------------------------------------------------

def test_publish_still_refuses_a_shard_set_the_copy_cannot_collapse(
    fake_hub: Any, tmp_path: Path,
) -> None:
    """A `-NNNNN-of-MMMMM` set with NO index is not collapsible (nothing names
    the members' order), so it must still be REFUSED rather than published.
    Normalizing the collapsible case must not soften the invariant."""
    tree = tmp_path / "flavor"
    (tree / "text_encoder").mkdir(parents=True)
    safetensors_torch.save_file(
        _tensors(7, 2),
        str(tree / "text_encoder" / "model-00001-of-00002.safetensors"))
    ctx = _Ctx(f"http://127.0.0.1:{fake_hub.server_port}")
    with pytest.raises(ConversionImplementationError,
                       match="sharded_producer_output"):
        publish_flavors(ctx, [ProducedFlavor(path=str(tree), flavor="x")],
                        destination_repo="cozy/qwen-image")


def test_an_index_naming_a_missing_shard_fails_the_PRODUCE(tmp_path: Path) -> None:
    """The klein-4b bug class: an index that disagrees with the bytes it names
    used to publish and die on a GPU pod at load. The merge verifies against
    the index, so it now dies here — in the producer, before any upload."""
    base = tmp_path / "base"
    _base_tree(base, sharded_text_encoder=True)
    next((base / "text_encoder").glob("model-00003-of-00009.safetensors")).unlink()
    with pytest.raises(ValueError, match="missing shard"):
        build_svdq_flavor_tree(
            base, _svdq_file(tmp_path / "svdq-fp4_r128-cozy.safetensors"),
            tmp_path / "flavor")
