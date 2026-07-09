"""RAM->VRAM promote device integrity (gw#409).

The J17 juggle trace lost ~9% of requests to "Expected all tensors to be on
the same device, index is on cpu": a pipeline ``.to()`` that raised (mid-move
CUDA OOM) or skipped a component was still booked IN_VRAM, so the handler ran
a mixed-device pipeline and fataled mid-denoise. Residency now verifies every
module parameter/buffer after a move (paranoid post-promote walk), repairs
targeted misses, and rolls back + refuses instead of booking a mixed object.
"""

from __future__ import annotations

import pytest

from gen_worker.models import residency as residency_mod
from gen_worker.models.residency import Residency, Tier

_GiB = 1024 ** 3


@pytest.fixture(autouse=True)
def _plenty_of_ram(monkeypatch):
    monkeypatch.setattr(residency_mod, "get_available_ram_gb", lambda: 64.0)


def _res(events: list, budget_gb: int = 24) -> Residency:
    return Residency(
        on_event=lambda ref, state, vb: events.append((ref, state, vb)),
        vram_budget_bytes=budget_gb * _GiB,
    )


class _Pipe:
    """Torch-less movable object (moves always succeed, nothing to verify)."""

    def __init__(self) -> None:
        self.moves: list[str] = []

    def to(self, device: str) -> "_Pipe":
        self.moves.append(device)
        return self


class _FailsTo(_Pipe):
    """`.to()` raises for the given device (simulated mid-move CUDA OOM)."""

    def __init__(self, fail_device: str) -> None:
        super().__init__()
        self._fail = fail_device

    def to(self, device: str) -> "_FailsTo":
        if device == self._fail:
            raise RuntimeError("CUDA out of memory (simulated)")
        return super().to(device)


# --------------------------------------------------------------------------- #
# Move-failure rollback (no torch required)
# --------------------------------------------------------------------------- #


def test_promote_move_failure_is_refused_and_rolled_back() -> None:
    events: list = []
    res = _res(events)
    pipe = _FailsTo("cuda")
    res.track_ram("m/a", pipe)

    assert res.promote("m/a") is False
    assert res.tier("m/a") is Tier.RAM
    assert res.vram_bytes("m/a") == 0
    # Rollback attempted a restore to cpu; nothing was booked IN_VRAM.
    assert pipe.moves[-1] == "cpu"
    assert not any(s == residency_mod.IN_VRAM for _, s, _ in events)


def test_demote_move_failure_keeps_vram_tier() -> None:
    events: list = []
    res = _res(events)
    pipe = _FailsTo("cpu")
    res.track_vram("m/a", pipe, vram_bytes=3 * _GiB)
    events.clear()

    assert res.demote("m/a") is False
    assert res.tier("m/a") is Tier.VRAM
    assert res.vram_bytes("m/a") == 3 * _GiB
    assert not any(s == residency_mod.IN_RAM for _, s, _ in events)


def test_failed_promote_then_retry_succeeds() -> None:
    """A refused promote leaves a consistent RAM entry; a later attempt (after
    make_room freed VRAM) promotes normally."""
    events: list = []
    res = _res(events)

    class _FlakyOnce(_Pipe):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1

        def to(self, device: str) -> "_FlakyOnce":
            if device == "cuda" and self.failures > 0:
                self.failures -= 1
                raise RuntimeError("CUDA out of memory (simulated)")
            return super().to(device)

    pipe = _FlakyOnce()
    res.track_vram("m/a", pipe, vram_bytes=2 * _GiB)
    assert res.demote("m/a") is True
    assert res.promote("m/a") is False
    assert res.tier("m/a") is Tier.RAM
    assert res.promote("m/a") is True
    assert res.tier("m/a") is Tier.VRAM
    assert res.vram_bytes("m/a") == 2 * _GiB  # hint restored


# --------------------------------------------------------------------------- #
# Post-move completeness walk over real torch modules (buffers included)
# --------------------------------------------------------------------------- #


def _synthetic_pipeline():
    """Diffusers-shaped pipeline: module components with parameters, a
    persistent buffer, a NON-persistent buffer, and a non-module component."""
    torch = pytest.importorskip("torch")

    class _Pipeline:
        def __init__(self) -> None:
            self.unet = torch.nn.Linear(2, 2)
            self.unet.register_buffer("rope", torch.zeros(2))
            self.unet.register_buffer(
                "step_index", torch.zeros(2, dtype=torch.long), persistent=False
            )
            self.text_encoder = torch.nn.Linear(2, 2)
            self.scheduler = object()  # non-module component (no tensors)
            self.components = {
                "unet": self.unet,
                "text_encoder": self.text_encoder,
                "scheduler": self.scheduler,
            }
            self.moves: list[str] = []

        # Mimics the gw#409 miss: the whole-pipeline move skips a component.
        def to(self, device: str) -> "_Pipeline":
            self.moves.append(device)
            self.unet.to(device)
            return self

    return torch, _Pipeline()


def test_device_mismatches_walks_params_and_all_buffers() -> None:
    torch, pipe = _synthetic_pipeline()
    from gen_worker.models.memory import device_mismatches

    pipe.to("meta")  # moves unet only
    missed = device_mismatches(pipe, "meta")
    names = {(c, t) for c, t, _ in missed}
    assert ("text_encoder", "weight") in names
    assert ("text_encoder", "bias") in names
    assert not any(c == "unet" for c, _ in names)  # unet fully moved
    # And the walk sees buffers: un-move unet's buffers only.
    assert device_mismatches(pipe, "cpu")  # unet params+buffers now off-cpu
    unet_named = {t for c, t, _ in device_mismatches(pipe, "cpu") if c == "unet"}
    assert {"weight", "bias", "rope", "step_index"} <= unet_named


def test_promote_repairs_component_skipped_by_pipeline_to() -> None:
    torch, pipe = _synthetic_pipeline()
    from gen_worker.models.memory import device_mismatches

    events: list = []
    res = _res(events)
    res.track_ram("sdxl/variant", pipe)

    assert res.promote("sdxl/variant", device="meta") is True
    assert res.tier("sdxl/variant") is Tier.VRAM
    # The paranoid walk found and repaired the skipped text_encoder: EVERY
    # param+buffer (incl. the non-persistent one) is on the target device.
    assert device_mismatches(pipe, "meta") == []


def test_promote_refuses_unrepairable_component() -> None:
    torch, pipe = _synthetic_pipeline()

    class _Stubborn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def _apply(self, fn, recurse=True):  # .to() lands here
            raise RuntimeError("CUDA out of memory (simulated)")

    pipe.text_encoder = _Stubborn()
    pipe.components["text_encoder"] = pipe.text_encoder

    res = _res([])
    res.track_ram("sdxl/variant", pipe)
    assert res.promote("sdxl/variant", device="meta") is False
    assert res.tier("sdxl/variant") is Tier.RAM


def test_vram_fast_path_repairs_mixed_device_entry() -> None:
    torch, pipe = _synthetic_pipeline()
    from gen_worker.models.memory import device_mismatches

    res = _res([])
    pipe.unet.to("meta")  # text_encoder left on cpu -> mixed
    res.track_vram("sdxl/variant", pipe, vram_bytes=1 * _GiB)

    # tier is already VRAM: the fast path must still verify + repair.
    assert res.promote("sdxl/variant", device="meta") is True
    assert device_mismatches(pipe, "meta") == []
