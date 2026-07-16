"""gw#522: sharded-variant index names must be loadable by REAL diffusers.

The mirror producer composed the sharded index as
``diffusion_pytorch_model.fp16.safetensors.index.json`` (variant token before
the extension); diffusers >=0.28 ``_add_variant`` looks for
``diffusion_pytorch_model.safetensors.index.fp16.json``, falls through
safetensors, then dies on ``no file named diffusion_pytorch_model.fp16.bin``
(verified live: tensorhub/sdxl-base fp16, snapshot 02fea340). The canonical-
naming pass (gw#466, unified across ALL publish paths here) strips variant
tokens entirely, which sidesteps the convention split.

These tests build the exact broken tree shape with the real resharder, prove
real diffusers rejects it, and prove the canonical tree loads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import gen_worker.convert.clone as clone_mod
from gen_worker.convert.clone import OutputSpec, _publish_from_bank
from gen_worker.convert.writer import normalize_variant_filenames

diffusers = pytest.importorskip("diffusers")
torch = pytest.importorskip("torch")


def _tiny_unet_fp16_tree(tmp_path: Path) -> Path:
    """A snapshot tree whose unet is a single fp16-variant safetensors."""
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel(
        block_out_channels=(4, 8), layers_per_block=1, sample_size=8,
        in_channels=4, out_channels=4, norm_num_groups=2,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=2, cross_attention_dim=8,
    ).to(torch.float16)
    unet.save_pretrained(tmp_path / "unet", safe_serialization=True, variant="fp16")
    assert (tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").exists()
    return tmp_path


def _reshard_old_convention(tree: Path) -> None:
    """Run the REAL resharder with a tiny threshold — this is exactly how the
    live mirrors got their old-convention index names."""
    clone_mod._stage_oversize_safetensors(tree, max_shard_bytes=16 * 1024)


class TestShardedVariantIndex:
    def test_old_convention_tree_is_unloadable_by_real_diffusers(
        self, tmp_path: Path,
    ) -> None:
        """Negative control: the pre-fix tree shape dies exactly like the
        live sdxl-base mirror did."""
        tree = _tiny_unet_fp16_tree(tmp_path)
        _reshard_old_convention(tree)
        old_idx = tree / "unet" / "diffusion_pytorch_model.fp16.safetensors.index.json"
        assert old_idx.exists(), "resharder no longer composes the old name?"

        from diffusers import UNet2DConditionModel

        with pytest.raises(OSError, match="fp16"):
            UNet2DConditionModel.from_pretrained(tree / "unet", variant="fp16")

    def test_canonical_tree_loads_with_real_diffusers(
        self, tmp_path: Path,
    ) -> None:
        tree = _tiny_unet_fp16_tree(tmp_path)
        _reshard_old_convention(tree)
        normalize_variant_filenames(tree)

        unet_dir = tree / "unet"
        assert (unet_dir / "diffusion_pytorch_model.safetensors.index.json").exists()
        assert not list(unet_dir.glob("*.fp16.*"))

        from diffusers import UNet2DConditionModel

        # torch_dtype as the serve path passes it after on-disk dtype sniffing.
        loaded = UNet2DConditionModel.from_pretrained(
            unet_dir, torch_dtype=torch.float16)
        assert loaded.dtype == torch.float16
        assert sum(p.numel() for p in loaded.parameters()) > 0


# ---------------------------------------------------------------------------
# th#592 bank replay: manifests banked before canonical naming must MISS
# ---------------------------------------------------------------------------


class _FakePlan:
    source_ref = "acme/sdxl"
    revision = "deadbeef"

    def bank_files(self):
        return [("unet/diffusion_pytorch_model.fp16.safetensors.index.json", 10, "b3")]


class _FakeHub:
    def __init__(self, payload_files: list) -> None:
        self._files = payload_files
        self.commits: list = []

    def lookup_clone_manifests(self, destination: str, keys: list) -> dict:
        return {
            k: {
                "found": True,
                "ready": True,
                "payload": {
                    "files": self._files,
                    "flavor": "fp16",
                    "metadata": {},
                    "repo_spec": {},
                },
            }
            for k in keys
        }

    def commit(self, **kwargs: Any) -> Any:  # pragma: no cover — must not run
        self.commits.append(kwargs)
        raise AssertionError("bank publish must not replay old-convention names")


def test_bank_entries_with_old_convention_names_miss(caplog: Any) -> None:
    hub = _FakeHub([
        {"path": "unet/diffusion_pytorch_model.fp16-00001-of-00002.safetensors",
         "size_bytes": 5, "blake3": "x"},
        {"path": "unet/diffusion_pytorch_model.fp16.safetensors.index.json",
         "size_bytes": 5, "blake3": "y"},
    ])
    spec = OutputSpec(dtype="fp16", file_layout="diffusers", file_type="safetensors")
    result = _publish_from_bank(
        hub, plan=_FakePlan(), provider="huggingface", specs=[spec],
        bank_keys={spec.label: "k1"}, destination="acme/sdxl", tags=[],
        mode="merge", progress=None,
    )
    assert result is None
    assert hub.commits == []


def test_bank_entries_with_canonical_names_replay() -> None:
    """Canonical banked manifests still take the download-skip path."""
    files = [
        {"path": "unet/diffusion_pytorch_model-00001-of-00002.safetensors",
         "size_bytes": 5, "blake3": "x"},
        {"path": "unet/diffusion_pytorch_model.safetensors.index.json",
         "size_bytes": 5, "blake3": "y"},
    ]

    class _Hub(_FakeHub):
        def commit(self, **kwargs: Any) -> Any:
            self.commits.append(kwargs)

            class _C:
                revision_id = "r1"
                checkpoint_id = "c1"
                uploaded = 0
                deduped = 2
                total_bytes = 10

            return _C()

    hub = _Hub(files)
    spec = OutputSpec(dtype="fp16", file_layout="diffusers", file_type="safetensors")
    result = _publish_from_bank(
        hub, plan=_FakePlan(), provider="huggingface", specs=[spec],
        bank_keys={spec.label: "k1"}, destination="acme/sdxl", tags=[],
        mode="merge", progress=None,
    )
    assert result is not None
    assert len(hub.commits) == 1


# ---------------------------------------------------------------------------
# publish seam: the as-is lane gets canonical names too
# ---------------------------------------------------------------------------


def test_publish_seam_normalizes_any_cast_tree(tmp_path: Path) -> None:
    """The run_clone seam covers trees build_flavor_tree never touched
    (publish-as-is lane): variant-token names come out canonical."""
    d = tmp_path / "text_encoder"
    d.mkdir()
    (d / "model.fp16.safetensors").write_bytes(b"x")
    idx = {"metadata": {}, "weight_map": {"w": "model.fp16.safetensors"}}
    (d / "model.fp16.safetensors.index.json").write_text(json.dumps(idx))
    normalize_variant_filenames(tmp_path)
    assert (d / "model.safetensors").exists()
    assert (d / "model.safetensors.index.json").exists()
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    assert wm == {"w": "model.safetensors"}
