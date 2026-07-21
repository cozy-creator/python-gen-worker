"""Quantization lanes (th#960/pgw#609 Phase 2b): a consolidated kept file
for the W8A8/W4A4/FP8 storage+load contract — the one bucket the coordinator
authorized a dedicated file for (no P-test home; this surface is orthogonal
to the worker<->hub lifecycle contract P1-P10 cover, but is flagship-
critical and carries real incident history).

Absorbed from (all deleted after this file lands): test_w8a8.py (gw#534),
test_w4a4.py (gw#540), test_fp8_and_emergency_loading.py (gw#389/th#546),
test_promote_device_integrity.py (gw#409, J17 9%-request-loss incident).
Their other ~40 tests (numerics-heavy GPU lanes, ladder/compile-key
bookkeeping, emergency-rung sizing arithmetic) have no incident pin and are
git-history-archived, not reproduced here (bucket-level triage, not
per-file absorb-or-accept-loss).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")


# ---------------------------------------------------------------------------
# gw#534: W8A8 fp8-GEMM contract — real tiny diffusers pipeline (CPU, no
# network), producer writes the exact artifact contract, loader dequants to
# bf16-resident and reproduces source weights to fp8 rounding.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ddpm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("quant") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return root


def test_w8a8_contract_artifact_detects_and_dequants_to_source_weights(
    tiny_ddpm: Path,
) -> None:
    import json

    from diffusers import DDPMPipeline, UNet2DModel

    from gen_worker.models import w8a8
    from gen_worker.models.loading import load_from_pretrained, pipeline_weight_lane
    from gen_worker.models.w8a8 import detect_w8a8_artifact, quantize_tree_w8a8

    w8a8_tree = quantize_tree_w8a8(tiny_ddpm, tiny_ddpm.parent / "w8a8")

    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None and art.component == "unet" and len(art.quantized) > 0
    cfg = json.loads((w8a8_tree / "unet" / "config.json").read_text())
    assert cfg["quantization_config"]["quant_algo"] == "FP8"

    w8a8.w8a8_gemm_mode.cache_clear() if hasattr(w8a8.w8a8_gemm_mode, "cache_clear") else None
    pipe = load_from_pretrained(DDPMPipeline, w8a8_tree)
    assert pipeline_weight_lane(pipe) in ("", "bf16-resident")
    ref = UNet2DModel.from_pretrained(str(tiny_ddpm / "unet"))
    name = art.quantized[0] + ".weight"
    a = ref.state_dict()[name].float()
    b = pipe.unet.state_dict()[name].float()
    rel = ((a - b).abs() / a.abs().clamp(min=1e-3)).max().item()
    assert rel < 0.13, "fp8 e4m3 dequant must reproduce source weights to fp8 rounding"


# ---------------------------------------------------------------------------
# gw#540: W4A4 nvfp4 contract — same real-pipeline shape, distinct format.
# ---------------------------------------------------------------------------


def test_w4a4_contract_artifact_detects_and_round_trips(tiny_ddpm: Path) -> None:
    from gen_worker.models.w4a4 import detect_w4a4_artifact, quantize_tree_w4a4

    w4a4_tree = quantize_tree_w4a4(tiny_ddpm, tiny_ddpm.parent / "w4a4")
    art = detect_w4a4_artifact(w4a4_tree)
    assert art is not None and art.component == "unet" and len(art.quantized) > 0

    # A w8a8-shaped tree must never cross-detect as w4a4 (the two contract
    # shapes share a directory layout; disambiguation is the bug class).
    from gen_worker.models.w8a8 import quantize_tree_w8a8

    w8a8_tree = quantize_tree_w8a8(tiny_ddpm, tiny_ddpm.parent / "w8a8-cross")
    assert detect_w4a4_artifact(w8a8_tree) is None


# ---------------------------------------------------------------------------
# gw#389/th#546: fp8 storage layerwise casting targets the denoiser
# specifically (not the whole pipeline) and defaults bf16 compute.
# ---------------------------------------------------------------------------


class _FakeDenoiser:
    def __init__(self) -> None:
        self.casting_calls: list = []

    def parameters(self):
        return iter(())

    def enable_layerwise_casting(self, *, storage_dtype: Any, compute_dtype: Any) -> None:
        self.casting_calls.append((storage_dtype, compute_dtype))


class _FakeDiffusionPipeline:
    transformer: _FakeDenoiser

    def __init__(self) -> None:
        self.transformer = _FakeDenoiser()


def test_fp8_storage_targets_denoiser_defaults_bf16_compute() -> None:
    from gen_worker.models.loading import apply_fp8_storage

    pipe = _FakeDiffusionPipeline()
    assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16) is True
    ((storage, compute),) = pipe.transformer.casting_calls
    assert storage is torch.float8_e4m3fn
    assert compute is torch.bfloat16


# ---------------------------------------------------------------------------
# gw#409 (J17: ~9% of requests lost to "tensors on different devices"): a
# promote whose .to() raises mid-move is refused and rolled back — never
# booked IN_VRAM as a mixed-device pipeline.
# ---------------------------------------------------------------------------


class _MovablePipe:
    def __init__(self) -> None:
        self.moves: list = []

    def to(self, device: str) -> "_MovablePipe":
        self.moves.append(device)
        return self


class _FailsToDevice(_MovablePipe):
    def __init__(self, fail_device: str) -> None:
        super().__init__()
        self._fail = fail_device

    def to(self, device: str) -> "_MovablePipe":
        if device == self._fail:
            raise RuntimeError("CUDA out of memory")
        return super().to(device)


def test_promote_move_failure_is_refused_never_booked_mixed_device(monkeypatch) -> None:
    from gen_worker.models import residency as residency_mod
    from gen_worker.models.residency import Residency, Tier

    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 64.0)
    events: list = []
    res = Residency(
        on_event=lambda ref, state, vb, dur=0: events.append((ref, state, vb)),
        vram_budget_bytes=24 * 1024**3,
    )
    pipe = _FailsToDevice("cuda")
    res.track_ram("m/a", pipe)

    assert res.promote("m/a") is False
    assert res.tier("m/a") is Tier.RAM
    assert res.vram_bytes("m/a") == 0
    assert pipe.moves[-1] == "cpu", "rollback must restore to cpu, never leave mixed-device"
    assert not any(s == residency_mod.IN_VRAM for _, s, _ in events)
