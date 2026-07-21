"""Lora-bucket compile cells (gw#561): Compile.lora_bucket declaration, lane
parsing/labels, branch-bearing lane apply/rollback through the real arming
path, and the lane-exact cell pick — all against real modules (CPU; the
GPU build/adopt/tax proof runs on the pod rig)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker import compile_cache
from gen_worker.api.decorators import Compile
from gen_worker.executor import _cell_lane_matches
from gen_worker.models import provision
from gen_worker.models.w8a8 import detect_w8a8_artifact, load_w8a8_denoiser, quantize_tree_w8a8
from gen_worker.models.w8a8_lora import RANK_BUCKETS, branch_bucket


@pytest.fixture(scope="module")
def w8a8_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("loracells") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return quantize_tree_w8a8(root, root.parent / "w8a8")


@pytest.fixture()
def w8a8_pipe(w8a8_tree: Path) -> Any:
    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe.unet = load_w8a8_denoiser(w8a8_tree, art, mode="rowwise")
    pipe._cozy_weight_lane = "w8a8"
    return pipe


@pytest.fixture()
def plain_pipe() -> Any:
    class _Pipe:
        pass

    class _Denoiser(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(16, 16)

    pipe = _Pipe()
    pipe.unet = _Denoiser()
    return pipe


# ---------------------------------------------------------------------------
# Declaration + lane grammar
# ---------------------------------------------------------------------------


def test_compile_lora_bucket_validation() -> None:
    cfg = Compile(shapes=((64, 64),), family="f", lora_bucket=32)
    assert cfg.lora_bucket == 32
    assert Compile(shapes=((64, 64),), family="f").lora_bucket == 0
    for bad in (-1, 8, 17, 256):
        with pytest.raises(ValueError):
            Compile(shapes=((64, 64),), family="f", lora_bucket=bad)


def test_lane_bucket_parses_stamp_and_token_forms() -> None:
    assert compile_cache.lane_bucket("") == ("", 0)
    assert compile_cache.lane_bucket("w8a8") == ("w8a8", 0)
    assert compile_cache.lane_bucket("w8a8-lora128") == ("w8a8", 128)
    assert compile_cache.lane_bucket("w8a16-lora32") == ("w8a16", 32)
    assert compile_cache.lane_bucket("fp8-hooks-lora64") == ("fp8-hooks", 64)
    assert compile_cache.lane_bucket("lora32") == ("", 32)
    # sparse stamps are eager-only and never parse as a cell bucket
    assert compile_cache.lane_bucket("w8a8-lora32-sparse") == ("w8a8-lora32-sparse", 0)


def test_lane_token_and_label_carry_bucket() -> None:
    assert compile_cache.lane_token("w8a8-lora128") == "w8a8-lora128"
    assert compile_cache.lane_token("fp8-hooks-lora32") == "w8a16-lora32"
    assert compile_cache.lane_token("lora32") == "lora32"
    assert compile_cache.flavor_label("h100-sxm", "2.13.0+cu130", "w8a8-lora128") == (
        "inductor-h100-sxm-torch2.13-w8a8-lora128")
    assert compile_cache.flavor_label("rtx-4090", "2.13.0", "lora32") == (
        "inductor-rtx-4090-torch2.13-lora32")


def test_cell_lane_roundtrip_through_ref() -> None:
    ref = "root/family-qwen-image#inductor-h100-sxm-torch2.13-w8a8-lora128"
    assert compile_cache.cell_lane(ref) == "w8a8-lora128"
    assert compile_cache.lane_bucket(compile_cache.cell_lane(ref)) == ("w8a8", 128)


# ---------------------------------------------------------------------------
# Lane apply/rollback (real w8a8 + plain denoisers)
# ---------------------------------------------------------------------------


def test_apply_lora_lane_stamps_and_allocates(w8a8_pipe: Any) -> None:
    assert compile_cache.apply_lora_lane(w8a8_pipe, 128)
    assert branch_bucket(w8a8_pipe.unet) == 128
    assert w8a8_pipe._cozy_weight_lane == "w8a8-lora128"
    meta = compile_cache.artifact_metadata(
        family="f", weight_lane="w8a8-lora128", lora_bucket=128)
    assert compile_cache.lane_drift(meta, w8a8_pipe) == ""
    assert meta["lora_bucket"] == 128
    # branchless cells refuse the branch-bearing pipeline (symmetric guard)
    assert "weight_lane" in compile_cache.lane_drift(
        compile_cache.artifact_metadata(family="f", weight_lane="w8a8"), w8a8_pipe)
    compile_cache.drop_lora_lane(w8a8_pipe)
    assert branch_bucket(w8a8_pipe.unet) == 0
    assert w8a8_pipe._cozy_weight_lane == "w8a8"


def test_apply_lora_lane_zero_bucket_is_noop(w8a8_pipe: Any) -> None:
    assert compile_cache.apply_lora_lane(w8a8_pipe, 0) is False
    assert branch_bucket(w8a8_pipe.unet) == 0


def test_apply_lora_lane_requires_denoiser() -> None:
    class _NoDenoiser:
        pass

    with pytest.raises(RuntimeError, match="branch-capable"):
        compile_cache.apply_lora_lane(_NoDenoiser(), 32)


def test_enable_compiled_rolls_back_branches_when_eager(plain_pipe: Any) -> None:
    """No cell + no CUDA => stays eager; the declared branch lane must not
    leak into eager serving (canonical zeroed slots cost +21-32% eager)."""
    cfg = Compile(shapes=((64, 64),), family="loracells-test", lora_bucket=32)
    armed = provision.enable_compiled(plain_pipe, cfg, cache_dir=None, artifact=None)
    assert armed is False
    assert branch_bucket(plain_pipe.unet) == 0
    from gen_worker.models.loading import pipeline_weight_lane

    assert pipeline_weight_lane(plain_pipe) == ""


def test_enable_compiled_w8a8_fail_closed_keeps_contract(w8a8_pipe: Any) -> None:
    cfg = Compile(shapes=((64, 64),), family="loracells-test", lora_bucket=128)
    with pytest.raises(compile_cache.CompiledLaneUnavailableError):
        provision.enable_compiled(w8a8_pipe, cfg, cache_dir=None, artifact=None)


# ---------------------------------------------------------------------------
# Lane-exact cell pick (the boot-attach filter)
# ---------------------------------------------------------------------------


def test_cell_pick_is_lane_and_bucket_exact() -> None:
    fam = "qwen-image"
    plain = f"root/family-{fam}#inductor-h100-sxm-torch2.13"
    w8a8 = f"root/family-{fam}#inductor-h100-sxm-torch2.13-w8a8"
    w8a8_l128 = f"root/family-{fam}#inductor-h100-sxm-torch2.13-w8a8-lora128"
    w8a8_l32 = f"root/family-{fam}#inductor-h100-sxm-torch2.13-w8a8-lora32"
    plain_l32 = f"root/family-{fam}#inductor-h100-sxm-torch2.13-lora32"

    def pick(ref: str, *, w8a8_lane: bool, bucket: int) -> bool:
        return _cell_lane_matches(ref, fam, want_lane="w8a8" if w8a8_lane else "", want_bucket=bucket)

    # branchless w8a8 endpoint: exactly the branchless w8a8 cell
    assert pick(w8a8, w8a8_lane=True, bucket=0)
    assert not pick(w8a8_l128, w8a8_lane=True, bucket=0)
    assert not pick(plain, w8a8_lane=True, bucket=0)
    # lora128 w8a8 endpoint: exactly the lora128 w8a8 cell
    assert pick(w8a8_l128, w8a8_lane=True, bucket=128)
    assert not pick(w8a8, w8a8_lane=True, bucket=128)
    assert not pick(w8a8_l32, w8a8_lane=True, bucket=128)
    assert not pick(plain_l32, w8a8_lane=True, bucket=128)
    # plain lora32 endpoint: never a w8a8 cell, never branchless
    assert pick(plain_l32, w8a8_lane=False, bucket=32)
    assert not pick(w8a8_l32, w8a8_lane=False, bucket=32)
    assert not pick(plain, w8a8_lane=False, bucket=32)
    # wrong family never matches
    assert not _cell_lane_matches(
        w8a8_l128, "sdxl", want_lane="w8a8", want_bucket=128)
    # w4a4 (gw#540): mandated-lane exactness, and plain endpoints never
    # fetch a quantized-lane cell
    w4a4 = f"root/family-{fam}#inductor-rtx-5090-torch2.13-w4a4"
    assert _cell_lane_matches(w4a4, fam, want_lane="w4a4", want_bucket=0)
    assert not _cell_lane_matches(w4a4, fam, want_lane="w8a8", want_bucket=0)
    assert not _cell_lane_matches(w4a4, fam, want_lane="", want_bucket=0)
    assert not _cell_lane_matches(w8a8, fam, want_lane="w4a4", want_bucket=0)


def test_rank_buckets_cover_declared_cells() -> None:
    # The produced buckets (32 civitai-common, 128 Lightning) are declared
    # RANK_BUCKETS members — the survey-tuned contract.
    assert 32 in RANK_BUCKETS and 128 in RANK_BUCKETS


def test_discovery_carries_lora_bucket() -> None:
    cfg = Compile(shapes=((64, 64),), family="f", lora_bucket=64)
    assert cfg.lora_bucket == 64
    # metadata parity: producer meta records the bucket beside the lane
    meta = compile_cache.artifact_metadata(
        family="f", weight_lane="w8a8-lora64", lora_bucket=64)
    assert meta["weight_lane"] == "w8a8-lora64"
    assert meta["lora_bucket"] == 64
