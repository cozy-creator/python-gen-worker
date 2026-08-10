"""pgw#1025: ``select_auto_mode`` must not count already-resident shared
components twice.

The gw#479 shape: a second lane of a shared-component endpoint boots against
a card that ALREADY holds the shared text encoder / VAE. Free VRAM has
already been reduced by those bytes, and the requirement estimate
(``estimate_pipeline_size_gb``) counts them again — measured overstatement
~7.85 GB for z-image, ~15.5 GB for qwen-image. Enough to fall off the
resident rung entirely; under th#1107's ``strict_vram`` that is not a slower
placement but a hard refusal (th#1043, the live failure).

Sizes are declared through tensor SHAPE, not allocation: the exclusive
component is a 1-element storage expanded to the weight count (a real CPU
tensor with a real, distinct ``data_ptr`` so the storage dedupe behaves), and
the already-resident component is a fake CUDA tensor. That is the only way to
put a 15.5 GB CUDA-resident component in front of this arithmetic on a host
with no CUDA device — and the arithmetic, which is what the issue is about,
runs for real.
"""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from gen_worker.models import memory

_GIB = 1024 ** 3


def _bf16_count(gb: float) -> int:
    return int(gb * _GIB / 2)


def _resident_component(gb: float) -> torch.nn.Module:
    """A component whose weights are already on CUDA."""
    m = torch.nn.Module()
    with FakeTensorMode():
        m._parameters["weight"] = torch.empty(
            _bf16_count(gb), dtype=torch.bfloat16, device="cuda")
    return m


def _exclusive_component(gb: float) -> torch.nn.Module:
    """A component still on the host, waiting to be placed."""
    m = torch.nn.Module()
    m._parameters["weight"] = torch.empty(
        1, dtype=torch.bfloat16).expand(_bf16_count(gb))
    return m


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


def test_strict_vram_lane_places_instead_of_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """th#1043 end to end through ``place_pipeline``: the overstatement is
    what turned a working placement into a typed refusal, because th#1107
    refuses an auto SELECTION into a CPU-touching rung."""
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
        strict_vram=True,
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
