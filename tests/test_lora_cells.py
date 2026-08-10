"""Lora-bucket compile cells (gw#561, SDK v2 pgw#647): the decorator-level
``@endpoint(lora_bucket=...)`` declaration, lane parsing/labels,
branch-bearing lane apply/rollback through the real arming path, and the
lane-exact cell pick — all against real modules (CPU; the GPU
build/adopt/tax proof runs on the pod rig)."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker import Compile, endpoint, compile_cache
from gen_worker.models import provision
from gen_worker.models.w8a8 import detect_w8a8_artifact, load_w8a8_denoiser, quantize_tree_w8a8
from gen_worker.models.w8a8_lora import RANK_BUCKETS, branch_bucket
from gen_worker.registry import CompileCell, collect_from_namespace


class _In(msgspec.Struct):
    prompt: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


def _cfg(
    family: str, *, shapes=((64, 64),), targets=("unet",), lora_bucket=0,
) -> CompileCell:
    """The enriched compile-cell configuration the machinery consumes in v2
    (``EndpointSpec.compile_cell()``): lora_bucket lives here, never on
    ``Compile``."""
    return CompileCell(
        shapes=tuple(tuple(s) for s in shapes), targets=tuple(targets),
        family=family, regional=False, text_len=0, dynamic=(),
        lora_bucket=int(lora_bucket), guidance_scales=(),
    )


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


def test_endpoint_lora_bucket_validation() -> None:
    """SDK v2: lora_bucket is a DECORATOR declaration (it shapes the graph
    family whether or not compile= is present); Compile no longer carries it."""

    @endpoint(lora_bucket=32)
    def gen32(ctx, p: _In) -> _Out:
        return _Out()

    specs = collect_from_namespace(types.SimpleNamespace(gen32=gen32))
    assert specs[0].lora_bucket == 32

    @endpoint
    def gen0(ctx, p: _In) -> _Out:
        return _Out()

    assert collect_from_namespace(
        types.SimpleNamespace(gen0=gen0))[0].lora_bucket == 0

    for bad in (-1, 8, 17, 256):
        with pytest.raises(ValueError):
            endpoint(lora_bucket=bad)

    with pytest.raises(TypeError):
        Compile(shapes=((64, 64),), family="f", lora_bucket=32)  # type: ignore[call-arg]


def test_execution_lane_bucket_parses_stamp_and_token_forms() -> None:
    assert compile_cache.execution_lane_bucket("") == ("", 0)
    assert compile_cache.execution_lane_bucket("w8a8") == ("w8a8", 0)
    assert compile_cache.execution_lane_bucket("w8a8-lora128") == ("w8a8", 128)
    assert compile_cache.execution_lane_bucket("w8a16-lora32") == ("w8a16", 32)
    assert compile_cache.execution_lane_bucket("fp8-hooks-lora64") == ("fp8-hooks", 64)
    assert compile_cache.execution_lane_bucket("lora32") == ("", 32)
    # sparse stamps are eager-only and never parse as a cell bucket
    assert compile_cache.execution_lane_bucket("w8a8-lora32-sparse") == ("w8a8-lora32-sparse", 0)


def test_execution_lane_token_and_label_carry_bucket() -> None:
    assert compile_cache.execution_lane_token("w8a8-lora128") == "w8a8-lora128"
    assert compile_cache.execution_lane_token("fp8-hooks-lora32") == "w8a16-lora32"
    assert compile_cache.execution_lane_token("lora32") == "lora32"
    assert compile_cache.flavor_label("h100-sxm", "2.13.0+cu130", "w8a8-lora128") == (
        "inductor-h100-sxm-torch2.13-w8a8-lora128")
    assert compile_cache.flavor_label("rtx-4090", "2.13.0", "lora32") == (
        "inductor-rtx-4090-torch2.13-lora32")


# ---------------------------------------------------------------------------
# Lane apply/rollback (real w8a8 + plain denoisers)
# ---------------------------------------------------------------------------


def test_apply_lora_execution_lane_stamps_and_allocates(w8a8_pipe: Any) -> None:
    assert compile_cache.apply_lora_execution_lane(w8a8_pipe, 128)
    assert branch_bucket(w8a8_pipe.unet) == 128
    assert w8a8_pipe._cozy_weight_lane == "w8a8-lora128"
    meta = compile_cache.artifact_metadata(
        family="f", weight_lane="w8a8-lora128", lora_bucket=128)
    assert compile_cache.execution_lane_drift(meta, w8a8_pipe) == ""
    assert meta["lora_bucket"] == 128
    # branchless cells refuse the branch-bearing pipeline (symmetric guard)
    assert "weight_lane" in compile_cache.execution_lane_drift(
        compile_cache.artifact_metadata(family="f", weight_lane="w8a8"), w8a8_pipe)
    compile_cache.drop_lora_execution_lane(w8a8_pipe)
    assert branch_bucket(w8a8_pipe.unet) == 0
    assert w8a8_pipe._cozy_weight_lane == "w8a8"


def test_apply_lora_execution_lane_zero_bucket_is_noop(w8a8_pipe: Any) -> None:
    assert compile_cache.apply_lora_execution_lane(w8a8_pipe, 0) is False
    assert branch_bucket(w8a8_pipe.unet) == 0


def test_apply_lora_execution_lane_requires_denoiser() -> None:
    class _NoDenoiser:
        pass

    with pytest.raises(RuntimeError, match="branch-capable"):
        compile_cache.apply_lora_execution_lane(_NoDenoiser(), 32)


def test_enable_compiled_rolls_back_branches_when_eager(plain_pipe: Any) -> None:
    """No cell + no CUDA => stays eager; the declared branch lane must not
    leak into eager serving (canonical zeroed slots cost +21-32% eager)."""
    cfg = _cfg("loracells-test", lora_bucket=32)
    armed = provision.enable_compiled(plain_pipe, cfg, cache_dir=None, artifact=None).armed
    assert armed is False
    assert branch_bucket(plain_pipe.unet) == 0
    from gen_worker.models.loading import pipeline_weight_lane

    assert pipeline_weight_lane(plain_pipe) == ""


def test_enable_compiled_w8a8_fail_closed_keeps_contract(w8a8_pipe: Any) -> None:
    cfg = _cfg("loracells-test", lora_bucket=128)
    with pytest.raises(compile_cache.CompiledExecutionLaneUnavailableError):
        provision.enable_compiled(w8a8_pipe, cfg, cache_dir=None, artifact=None)


def test_rank_buckets_cover_declared_cells() -> None:
    # The produced buckets (32 civitai-common, 128 Lightning) are declared
    # RANK_BUCKETS members — the survey-tuned contract.
    assert 32 in RANK_BUCKETS and 128 in RANK_BUCKETS


def test_discovery_carries_lora_bucket() -> None:
    @endpoint(lora_bucket=64, compile=Compile(
        shapes=((64, 64),), family="f", text_len=0))
    def gen64(ctx, p: _In) -> _Out:
        return _Out()

    spec = collect_from_namespace(types.SimpleNamespace(gen64=gen64))[0]
    assert spec.lora_bucket == 64
    # The compile machinery consumes the enriched CompileCell, which folds
    # the decorator-level bucket into the declared graph family.
    cell = spec.compile_cell()
    assert cell is not None
    assert cell.lora_bucket == 64
    assert cell.family == "f"
    assert cell.contract_facts()["lora_bucket"] == 64
    # metadata parity: producer meta records the bucket beside the lane
    meta = compile_cache.artifact_metadata(
        family="f", weight_lane="w8a8-lora64", lora_bucket=64)
    assert meta["weight_lane"] == "w8a8-lora64"
    assert meta["lora_bucket"] == 64


def test_enable_compiled_skips_execution_lane_on_component_slot_without_target() -> None:
    """gw#627 live find: enable_compiled runs for EVERY worker-loaded setup
    slot — a bare component slot (sdxl's standalone AutoencoderKL vae)
    resolves none of cfg.targets and must stay branchless-eager instead of
    raising apply_lora_lane's no-denoiser error (which broke the whole
    model load, release-broken streak)."""

    class _Vae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Linear(8, 8)

    vae = _Vae()
    cfg = _cfg("loracells-test", lora_bucket=64)
    armed = provision.enable_compiled(vae, cfg, cache_dir=None, artifact=None).armed
    assert armed is False


def test_delivered_lora_cell_on_component_slot_is_ordinary_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gw#632 (r3 live find, the all_declared_functions_disabled chain): a
    DELIVERED family lora<bucket> cell arriving at a bare component slot
    (sdxl's standalone vae — no cfg.target resolves) must be an ordinary
    lane miss that stays eager. Before the effective-bucket fix the self-key
    check claimed the cell as this runtime's own (cfg bucket, '' lane) and
    raised CellSelectionBugError (`weight_lane 'lora64' != pipeline ''`),
    which cascaded into the gw#608 seeded-cell refusal and retired the pod."""

    # Pin the runtime axes so the self-verdict computes on a CPU host (the
    # live failure needs the seeded self-cell check to succeed — sku/sm come
    # from the GPU).
    rt = {
        "sku": "rtx-4090", "sm": "sm_89", "torch": "2.13.0+cu130",
        "triton": "3.7.1", "cuda": "13.0", "cuda_driver": "13020",
        "image_digest": "",
    }
    monkeypatch.setattr(compile_cache, "runtime_key", lambda: dict(rt))

    class _Vae(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = torch.nn.Linear(8, 8)

    cfg = _cfg("loracells-test", lora_bucket=64)
    src = tmp_path / "cellsrc"
    (src / "inductor" / "ab").mkdir(parents=True)
    (src / "inductor" / "ab" / "graph.py").write_text("code")
    meta = compile_cache.artifact_metadata(
        family="loracells-test", shapes=[(64, 64)], targets=["unet"],
        weight_lane="lora64", lora_bucket=64,
        declared_compile_contract=compile_cache.declared_compile_facts(cfg),
    )
    artifact = compile_cache.pack(src, tmp_path / "cell.tar.gz", meta)

    vae = _Vae()
    armed = provision.enable_compiled(
        vae, cfg, cache_dir=tmp_path / "cache", artifact=artifact).armed
    assert armed is False
