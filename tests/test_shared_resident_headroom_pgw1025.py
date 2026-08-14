"""``select_auto_mode`` must not count already-resident shared components twice.

The gw#479 shape: a second lane of a shared-component endpoint boots against
a card that ALREADY holds the shared text encoder / VAE. Free VRAM has
already been reduced by those bytes, and the requirement estimate
(``estimate_pipeline_size_gb``) counts them again — measured overstatement
~7.85 GB for z-image, ~15.5 GB for qwen-image. Enough to fall off the
resident rung entirely; under th#1107's since-deleted ``strict_vram`` that was not a slower
placement but a hard refusal (th#1043, the live failure).

Sizes are declared through tensor SHAPE, not allocation: a 1-element storage
expanded to the weight count, a real tensor with a real, distinct ``data_ptr``
so the storage dedupe behaves. The already-resident component additionally
declares its DEVICE, which is the one fact a host with no CUDA device cannot
produce — and the arithmetic, which is what the issue is about, runs for real.

The resident stand-in is deliberately NOT a fake tensor: ``_sum_tensor_bytes``
exempts those (a structure-only pipeline's virtual weights are not on any card).
``tests/_declared_residency`` puts 15.5 GB in front of the arithmetic without
depending on the production path mis-measuring anything.
"""

from __future__ import annotations

import pytest
import torch

from gen_worker.models import memory

from _declared_residency import host_component as _exclusive_component
from _declared_residency import resident_component as _resident_component


class _SharedLanePipeline:
    """qwen-image's second lane: 15.5 GB of shared components already on the
    card, 40 GB of exclusive denoiser weights still to place."""

    def __init__(self, *, shared_gb: float = 15.5, exclusive_gb: float = 40.0):
        self.components = {
            "text_encoder": _resident_component(shared_gb),
            "transformer": _exclusive_component(exclusive_gb),
        }


def test_the_estimates_disagree_by_exactly_the_shared_bytes() -> None:
    pipe = _SharedLanePipeline()
    assert memory.estimate_pipeline_size_gb(pipe) == pytest.approx(55.5)
    assert memory.estimate_cuda_resident_gb(pipe) == pytest.approx(15.5)


def test_resident_shared_bytes_are_not_charged_twice() -> None:
    """80 GB card, 45 GB free, 15.5 GB of it already this pipeline's.

    Pre-fix: requirement 55.5 > usable 43.0 -> ``model_offload``, a 5-10x
    tax on a lane that fits. Net requirement is 40.0, which fits.
    """
    mode = memory.select_auto_mode(
        pipeline=_SharedLanePipeline(),
        available_vram_gb=45.0,
        total_vram_gb=80.0,
    )
    assert mode == "off", (
        f"selected {mode!r}: the 15.5 GB already on the card was charged "
        "against free VRAM that had already paid for it"
    )


def test_shared_resident_lane_places_resident_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#1025 end to end through ``place_pipeline``: the overstatement used
    to turn a working RESIDENT placement into an offload one. th#1867 deleted
    the `strict_vram` refusal that made that misread fatal, but the misread
    itself is still a real defect — a lane charged twice for bytes already on
    the card serves slower than it needs to, which is exactly the efficiency
    question §1.35 says IS the question."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 45.0)
    monkeypatch.setattr(memory, "get_total_vram_gb", lambda *a, **k: 80.0)

    applied: list[str] = []

    def fake_apply(pipeline, *, mode, logger=None):
        applied.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(memory, "apply_low_vram_config", fake_apply)

    result = memory.place_pipeline(
        _SharedLanePipeline(), mode="auto", ref="qwen-image/edit",
    )
    assert applied == ["off"]
    assert result["mode"] == "off"


def test_refinement_still_keys_on_the_gross_requirement_per_sku() -> None:
    """pgw#750 is not disturbed. 70 GB SKU, 60.5 GB tree of which 15.5 GB is
    already resident: the FIT test passes on the 45 GB net, but the off vs
    vae_only refinement stays on the gross 60.5 against the SKU constant
    (67 - 60.5 = 6.5 < 8), so the traced decode graph class stays a function
    of the SKU and never of live residency."""
    mode = memory.select_auto_mode(
        pipeline=_SharedLanePipeline(shared_gb=15.5, exclusive_gb=45.0),
        available_vram_gb=50.0,
        total_vram_gb=70.0,
    )
    assert mode == "vae_only"


def test_no_total_probe_falls_back_to_the_net_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a total-capacity probe the refinement has only live free VRAM
    to work with, so it takes the net requirement like the fit test — never
    the double-counted one (pre-fix: 55.5 > 43.0 -> ``model_offload``)."""
    monkeypatch.setattr(memory, "get_total_vram_gb", lambda *a, **k: 0.0)
    mode = memory.select_auto_mode(
        pipeline=_SharedLanePipeline(), available_vram_gb=45.0)
    assert mode == "vae_only"


def test_a_pipeline_with_nothing_resident_is_unchanged() -> None:
    """The correction is exactly zero when nothing is on the card yet."""
    class _ColdPipeline:
        def __init__(self) -> None:
            self.components = {"transformer": _exclusive_component(40.0)}

    assert memory.estimate_cuda_resident_gb(_ColdPipeline()) == 0.0
    assert memory.select_auto_mode(
        pipeline=_ColdPipeline(), available_vram_gb=45.0,
        total_vram_gb=80.0) == "off"
    assert memory.select_auto_mode(
        pipeline=_ColdPipeline(), available_vram_gb=30.0,
        total_vram_gb=80.0) == "model_offload"
