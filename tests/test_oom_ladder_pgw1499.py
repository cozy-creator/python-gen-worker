"""pgw#1499: reactive OOM ladders — the retry is REAL, and it confesses.

Three mechanisms, all of them the CATCHABLE eager cases:

* a VAE decode that exhausts the card is retried TILED, shrinking a rung per
  further OOM;
* a denoise step that exhausts the card is retried with a finer attention
  slice, to a cap;
* an ASYNCHRONOUS out-of-memory (``AcceleratorError`` code 2) is recognised as
  an OOM at all, and the poisoned context is flushed before the retry.

Seam. The VAE half runs a REAL ``diffusers.AutoencoderKL`` — real
``decode``/``_decode``/``tiled_decode``, real blending — armed through the REAL
``memory.apply_low_vram_config`` install seam, with the OOM injected where a
real one lands: inside the decoder, on a tensor that is too big. Nothing about
the ladder is stubbed; only the card is. The events are drained through the
real activity sink, so the assertions read the ActivityUpdates a hub banks.

Every fixed arm below has a RED arm that flips a CONDITION (the ladder is not
armed, the cap is exhausted) rather than cutting lines.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest
import torch
from diffusers import AutoencoderKL

from gen_worker import activity as activity_mod
from gen_worker.models import memory, oom_ladder
from gen_worker.pb import worker_scheduler_pb2 as pb


class _Events:
    """The REAL activity sink the worker transport installs."""

    def __init__(self) -> None:
        self.sent: List[pb.WorkerMessage] = []
        self.loop = asyncio.new_event_loop()

    def __enter__(self) -> "_Events":
        async def _send(msg: pb.WorkerMessage) -> None:
            self.sent.append(msg)

        activity_mod.bind_sink(_send, self.loop)
        return self

    def __exit__(self, *exc: object) -> None:
        self.loop.run_until_complete(asyncio.sleep(0.02))
        activity_mod.reset_for_tests()
        self.loop.close()

    def degrades(self, phase: str) -> List[pb.ActivityUpdate]:
        return [
            m.activity_update for m in self.sent
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == activity_mod.KIND_SERVE_DEGRADE
            and m.activity_update.phase == phase
        ]


# ---------------------------------------------------------------------------
# 1. The tile solver — pure arithmetic, no card
# ---------------------------------------------------------------------------


def test_temporal_halves_first_then_spatial_doubles() -> None:
    """The ComfyUI solve: shrink time until one tile fits, then grow space."""
    # 16 latent frames, 128x128 latent, 1 byte per latent element.
    # A 32-edge tile costs 1024 per frame; the budget takes 4 frames of it.
    ladder = oom_ladder.solve_tile_ladder(
        latent_h=128, latent_w=128, latent_frames=16,
        bytes_per_latent=1.0, budget_bytes=32 * 32 * 4,
    )
    assert ladder[0] == oom_ladder.TilePlan(edge=32, frames=4)

    # Quadruple the budget and the SPATIAL tile doubles instead — the temporal
    # halving stops as soon as one tile fits, and the leftover goes to space.
    wide = oom_ladder.solve_tile_ladder(
        latent_h=128, latent_w=128, latent_frames=16,
        bytes_per_latent=1.0, budget_bytes=64 * 64 * 16,
    )
    assert wide[0] == oom_ladder.TilePlan(edge=64, frames=16)


def test_a_tile_the_size_of_the_frame_is_not_a_retry() -> None:
    """A latent no bigger than the base tile still gets a SMALLER rung 0 —
    retrying the exact shape that just OOMed is not a retry."""
    ladder = oom_ladder.solve_tile_ladder(
        latent_h=24, latent_w=24, latent_frames=0,
        bytes_per_latent=1.0, budget_bytes=0.0,
    )
    assert ladder[0].edge < 24
    assert ladder[0].edge == 8  # min_edge

    # And when the BUDGET says the whole frame fits, the allocator has just
    # said otherwise — the estimate loses.
    generous = oom_ladder.solve_tile_ladder(
        latent_h=24, latent_w=24, latent_frames=0,
        bytes_per_latent=1.0, budget_bytes=1e12,
    )
    assert generous[0].edge < 24


def test_the_ladder_descends_and_terminates() -> None:
    ladder = oom_ladder.solve_tile_ladder(
        latent_h=256, latent_w=256, latent_frames=8,
        bytes_per_latent=1.0, budget_bytes=32 * 32 * 8, max_rungs=8,
    )
    assert len(ladder) <= 8
    # Time first, then space, and never sideways.
    for prev, nxt in zip(ladder, ladder[1:]):
        assert (nxt.frames, nxt.edge) <= (prev.frames, prev.edge)
        assert (nxt.frames, nxt.edge) != (prev.frames, prev.edge)
    assert ladder[-1].frames == 1
    assert ladder[-1].edge >= 8


def test_bytes_per_latent_matches_the_measured_sd_family_coefficient() -> None:
    """ComfyUI's hand-measured sd15/SDXL decode coefficient is
    ``2178 * 64 * dtype_size`` bytes per latent element. The formula must land
    on it, or the first rung is guesswork dressed as arithmetic."""

    class _Cfg:
        block_out_channels = (128, 256, 512, 512)

    class _Vae:
        config = _Cfg()

    got = oom_ladder.decode_bytes_per_latent(_Vae(), dtype_bytes=2)
    assert 0.9 <= got / (2178 * 64 * 2) <= 1.1


# ---------------------------------------------------------------------------
# 2. The VAE ladder against a REAL diffusers AutoencoderKL
# ---------------------------------------------------------------------------


def _tiny_vae() -> Any:
    """A real AutoencoderKL with an 8x spatial compression and 8 channels."""
    vae: Any = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=64,
    )
    vae.eval()
    return vae


class _CardTooSmall(torch.nn.Module):
    """The card. Any decoder call over ``limit`` latent elements raises the
    allocator's real exception — which is exactly what a full-frame decode on a
    card that cannot hold it does. Everything below that limit really decodes."""

    def __init__(self, decoder: torch.nn.Module, limit: int) -> None:
        super().__init__()
        self.inner = decoder
        self.limit = limit
        self.calls: List[int] = []

    def forward(self, z: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        elems = int(z.shape[-1]) * int(z.shape[-2])
        self.calls.append(elems)
        if elems > self.limit:
            raise torch.cuda.OutOfMemoryError(
                f"CUDA out of memory. Tried to allocate {elems} MiB")
        return self.inner(z, *args, **kwargs)


class _Pipeline:
    """A diffusers pipeline reduced to the surface the ladder installs on."""

    def __init__(self, vae: Any) -> None:
        self.vae = vae
        self.components: Dict[str, Any] = {"vae": vae}

    def to(self, device: str) -> "_Pipeline":
        return self


def test_a_vae_oom_retries_tiled_and_confesses() -> None:
    vae = _tiny_vae()
    vae.decoder = _CardTooSmall(vae.decoder, limit=12 * 12)
    pipe = _Pipeline(vae)
    latent = torch.randn(1, 4, 24, 24)

    with _Events() as events:
        # THE REAL SEAM: placement arms the ladder. `off` is the rung that
        # pre-applies nothing, so tiling here can only be reactive.
        memory.apply_low_vram_config(pipe, mode="off")
        out = pipe.vae.decode(latent).sample

    assert out.shape == (1, 3, 192, 192)
    # The full frame was tried first and failed; the tiles that followed fit.
    assert vae.decoder.calls[0] == 24 * 24
    assert max(vae.decoder.calls[1:]) <= 12 * 12
    assert vae.use_tiling is True

    banked = events.degrades(oom_ladder.VAE_TILED_RETRY_PHASE)
    assert len(banked) == 1
    assert "retrying tiled" in banked[0].detail
    assert "OutOfMemoryError" in banked[0].detail


def test_red_arm_without_the_ladder_the_same_decode_raises() -> None:
    """The condition that is flipped is the INSTALL, nothing else."""
    vae = _tiny_vae()
    vae.decoder = _CardTooSmall(vae.decoder, limit=12 * 12)
    latent = torch.randn(1, 4, 24, 24)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        vae.decode(latent)


def test_the_ladder_is_armed_once_per_pipeline() -> None:
    """`apply_low_vram_config` runs again on every rung of a placement descent;
    a second wrapper around the first would double every retry."""
    pipe = _Pipeline(_tiny_vae())
    memory.apply_low_vram_config(pipe, mode="off")
    first = pipe.vae.decode
    memory.apply_low_vram_config(pipe, mode="off")
    assert pipe.vae.decode is first


def test_a_vae_that_can_never_fit_says_so_instead_of_looping() -> None:
    vae = _tiny_vae()
    vae.decoder = _CardTooSmall(vae.decoder, limit=0)
    pipe = _Pipeline(vae)
    with _Events():
        memory.apply_low_vram_config(pipe, mode="off")
        with pytest.raises(RuntimeError, match="every tile the ladder has"):
            pipe.vae.decode(torch.randn(1, 4, 24, 24))


# ---------------------------------------------------------------------------
# 3. The attention ladder on the denoise step
# ---------------------------------------------------------------------------


class _Denoiser(torch.nn.Module):
    """A real nn.Module — the point of the seam is that `module(...)` resolves
    `self.forward` as an INSTANCE attribute, so the wrapper is reached."""

    def __init__(self, ooms: int) -> None:
        super().__init__()
        self.left = ooms
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        self.calls += 1
        if self.left > 0:
            self.left -= 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return x * 2


class _SlicingPipeline(_Pipeline):
    def __init__(self, vae: Any, unet: Any) -> None:
        super().__init__(vae)
        self.unet = unet
        self.slices: List[Any] = []

    def enable_attention_slicing(self, slice_size: Any = "auto") -> None:
        self.slices.append(slice_size)


def test_a_denoise_oom_slices_attention_and_retries() -> None:
    unet = _Denoiser(ooms=1)
    pipe = _SlicingPipeline(_tiny_vae(), unet)
    x = torch.ones(2, 2)

    with _Events() as events:
        memory.apply_low_vram_config(pipe, mode="off")
        out = pipe.unet(x)

    assert torch.equal(out, x * 2)
    assert unet.calls == 2
    assert pipe.slices == ["auto"]
    banked = events.degrades(oom_ladder.ATTENTION_SLICED_RETRY_PHASE)
    assert len(banked) == 1
    assert "attention_slicing=auto" in banked[0].detail


def test_the_attention_ladder_stops_at_its_cap() -> None:
    unet = _Denoiser(ooms=99)
    pipe = _SlicingPipeline(_tiny_vae(), unet)
    with _Events():
        memory.apply_low_vram_config(pipe, mode="off")
        with pytest.raises(RuntimeError, match="every attention slice"):
            pipe.unet(torch.ones(2, 2))
    assert pipe.slices == list(oom_ladder.ATTENTION_LADDER)


def test_red_arm_without_the_ladder_the_step_raises() -> None:
    unet = _Denoiser(ooms=1)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        unet(torch.ones(2, 2))


# ---------------------------------------------------------------------------
# 4. The asynchronous OOM shape
# ---------------------------------------------------------------------------


def test_an_async_accelerator_oom_is_an_oom() -> None:
    """Before this issue an AcceleratorError read as an unclassified crash, so
    no ladder ran on the ASYNCHRONOUS shape of the same exhaustion."""
    by_code = torch.AcceleratorError("CUDA error: unspecified launch failure")
    by_code.error_code = 2  # type: ignore[attr-defined]
    assert memory.is_cuda_oom(by_code) is True

    by_text = torch.AcceleratorError("CUDA error: out of memory")
    assert memory.is_cuda_oom(by_text) is True

    other = torch.AcceleratorError("CUDA error: an illegal memory access")
    other.error_code = 700  # type: ignore[attr-defined]
    assert memory.is_cuda_oom(other) is False


def test_discarding_the_async_error_is_safe_without_a_card() -> None:
    """It runs on every OOM classification, so it may never raise — including
    on the cardless box where most of this code is exercised."""
    memory.discard_cuda_async_error()


def test_the_ladder_ignores_a_non_oom_failure() -> None:
    """A ValueError inside the decode is a defect, not a card size: it must
    reach the caller untouched, with its traceback."""
    class _Broken(torch.nn.Module):
        def forward(self, *_a: Any, **_k: Any) -> Any:
            raise ValueError("bad latent")

    vae = _tiny_vae()
    vae.decoder = _Broken()
    pipe = _Pipeline(vae)
    memory.apply_low_vram_config(pipe, mode="off")
    with pytest.raises(ValueError, match="bad latent"):
        pipe.vae.decode(torch.randn(1, 4, 24, 24))
    assert vae.use_tiling is False
