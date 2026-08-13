"""pgw#772: the serving lane is deterministic per (release x declared config).

The gw#534 voluntary bf16-resident upgrade probed LIVE free VRAM
(`bf16_resident_fits` / BF16_RESIDENT_MARGIN_GB) and made `lane` — the only
GPU-dependent axis of the ten in the cell key — a function of the
individual card's headroom: an RTX 4090's ~1.5 GiB surplus over an L4 (same
release, same image, both sm_89) flipped its base lane to "", a lane nothing
mints for, so the better card missed all 144 published checkpoints INCLUDING
its own same-SKU cell and served eager for life (th#1198 CP-D wire evidence;
the −21% request-level AOT win forfeited). The tax the upgrade dodged is
+1.9% for the structural fp8 storage lane (pgw#727 re-measure).

Red-verified on the pre-fix tree (probe restored): the headline test fails —
the high-headroom load lands base lane "" while the low-headroom load lands
"fp8-hooks", and the two requested cell keys diverge on the `lane` axis.

Standing rule (the pgw#765 `sku` guard, applied to `lane`): no adoption or
identity path may take a live device measurement as a hash input. A card
that CANNOT run the declared lane is different — the fit ladder's rungs are
declared, typed, reported transitions, and the second test pins that they
still engage.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from gen_worker import compile_cache as cc
from gen_worker.models import loading, memory
from gen_worker.models.provision import load_slot

_GiB = 1024 ** 3

_RT = {
    "sku": "rtx-4090", "sm": "sm_89", "torch": "2.13.0+cu130",
    "triton": "3.7.1", "cuda": "13.0", "cuda_driver": "13020",
    "image_digest": "",
}


class _ContractCfg:
    def __init__(self) -> None:
        self.shapes = ((1024, 1024),)
        self.targets = ("unet",)
        self.text_len = 0
        self.dynamic = ()
        self.regional = False
        self.lora_bucket = 64
        self.guidance_scales = ()


def _snapshot(tmp_path: Path, *, dtype: torch.dtype | None = None) -> Path:
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    )
    if dtype is not None:
        unet = unet.to(dtype)  # type: ignore[attr-defined]
    root = tmp_path / "src"
    pipe = DDPMPipeline(unet=unet, scheduler=DDPMScheduler())
    pipe.save_pretrained(str(root))  # type: ignore[attr-defined]
    return root


def _pin_free_vram(monkeypatch: pytest.MonkeyPatch, free_gb: float) -> None:
    """Every seam a load-time VRAM probe can reach, pre- and post-fix:
    loading's module-level import (fit ladder) and the memory module itself
    (the pre-fix `bf16_resident_fits` imported it lazily)."""
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda: free_gb)
    monkeypatch.setattr(loading, "get_available_vram_gb", lambda: free_gb)


def _load_and_key(
        root: Path, monkeypatch: pytest.MonkeyPatch, *, free_gb: float,
) -> "tuple[object, str, object]":
    _pin_free_vram(monkeypatch, free_gb)
    sl = load_slot(
        DDPMPipeline, str(root), slot="pipeline", device="cpu",
        binding=type("B", (), {"dtype": "", "storage_dtype": "fp8"})(),
    )
    execution_lane = cc.cell_base_execution_lane(sl.obj)
    cfg = _ContractCfg()
    # pgw#1059: the pre-trace surface is the obligation identity
    # (fleet_cells.arm_identity), not a cell key.
    from gen_worker import fleet_cells
    identity = fleet_cells.arm_identity(
        "sdxl", execution_lane, cfg.lora_bucket, cfg)
    return sl.obj, execution_lane, identity


def test_same_declared_config_same_key_across_free_vram(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """THE HEADLINE (pgw#772 acceptance): two runtimes differing ONLY in
    free VRAM, same (release-declared config, sm), resolve the same base
    lane and compute IDENTICAL requested cell keys — so both request the
    same published cell. Pre-fix: 999 GiB of headroom voluntarily upcast
    the declared fp8 lane to bf16-resident (base ""), key lane `lora64`,
    which no mint ever publishes."""
    monkeypatch.setattr(cc, "runtime_key", lambda: dict(_RT))
    with tempfile.TemporaryDirectory() as td:
        root = _snapshot(Path(td))
        # The L4-shaped host: headroom below the old 4 GiB margin.
        _, execution_lane_tight, key_tight = _load_and_key(root, monkeypatch, free_gb=4.0)
        # The 4090-shaped host: headroom to spare.
        _, execution_lane_rich, key_rich = _load_and_key(root, monkeypatch, free_gb=999.0)

    assert execution_lane_tight == execution_lane_rich == "fp8-hooks"
    # The issue's acceptance surface: the two computed fact dicts.
    assert key_tight.facts_dict() == key_rich.facts_dict()
    assert key_tight.token == key_rich.token
    # And it is the MINTED lane token (w8a16 = fp8-hooks on the wire), not
    # the bare `lora64` orphan lane the probe used to fork onto.
    assert key_tight.facts_dict()["lane"] == "w8a16-lora64"


def test_involuntary_cant_fit_rung_still_engages(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The distinction pgw#772 preserves: a card that CANNOT fit the
    declared lane still degrades through the fit ladder (declared rung,
    structurally reported) — determinism removed only the VOLUNTARY move."""
    with tempfile.TemporaryDirectory() as td:
        root = _snapshot(Path(td), dtype=torch.bfloat16)
        comp = loading.snapshot_component_weight_bytes(root)
        total = sum(comp.values())
        denoiser = comp.get("unet", 0)
        assert total > 0 and denoiser > 0
        # Free VRAM such that the snapshot does NOT fit as declared (bf16)
        # but DOES fit with the fp8-storage rung's halved denoiser:
        # budget = total - 0.25 * denoiser.
        free_gb = 2.0 + (total - 0.25 * denoiser) / float(_GiB)
        monkeypatch.setattr(
            loading, "runtime_fp8_storage_supported", lambda: True)
        _pin_free_vram(monkeypatch, free_gb)
        sl = load_slot(
            DDPMPipeline, str(root), slot="pipeline", device="cpu",
            binding=type("B", (), {"dtype": "", "storage_dtype": ""})(),
        )

    # The rung engaged, moved the lane involuntarily, and REPORTED it.
    assert getattr(sl.obj, "_cozy_adaptive_rung", "") == "fp8"
    assert getattr(sl.obj, "_cozy_weight_lane", "") == "fp8-hooks"
    assert sl.rung == "fp8"
    assert "adaptive fit rung" in sl.rung_detail


def test_voluntary_upcast_probe_stays_removed() -> None:
    """Standing guard: the free-VRAM upcast probe must not come back. A
    reintroduction under the old names trips here by name; one under a new
    name trips the headline test by behavior."""
    assert not hasattr(loading, "bf16_resident_fits")
    assert not hasattr(loading, "BF16_RESIDENT_MARGIN_GB")
    import inspect

    src = inspect.getsource(loading.load_from_pretrained)
    assert "bf16_resident" not in src
