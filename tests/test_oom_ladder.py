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


def test_temporal_halves_first_then_spatial_doubles() -> None:
    """The ComfyUI solve: shrink time until one tile fits, then grow space."""
    ladder = oom_ladder.solve_tile_ladder(
        latent_h=128, latent_w=128, latent_frames=16,
        bytes_per_latent=1.0, budget_bytes=32 * 32 * 4,
    )
    assert ladder[0] == oom_ladder.TilePlan(edge=32, frames=4)

    wide = oom_ladder.solve_tile_ladder(
        latent_h=128, latent_w=128, latent_frames=16,
        bytes_per_latent=1.0, budget_bytes=64 * 64 * 16,
    )
    assert wide[0] == oom_ladder.TilePlan(edge=64, frames=16)


def test_a_tile_the_size_of_the_frame_is_not_a_retry() -> None:
    """A latent no bigger than the base tile still gets a SMALLER rung 0 — retrying the exact shape that just OOMed is not a retry."""
    ladder = oom_ladder.solve_tile_ladder(
        latent_h=24, latent_w=24, latent_frames=0,
        bytes_per_latent=1.0, budget_bytes=0.0,
    )
    assert ladder[0].edge < 24
    assert ladder[0].edge == 8

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
    for prev, nxt in zip(ladder, ladder[1:]):
        assert (nxt.frames, nxt.edge) <= (prev.frames, prev.edge)
        assert (nxt.frames, nxt.edge) != (prev.frames, prev.edge)
    assert ladder[-1].frames == 1
    assert ladder[-1].edge >= 8


def test_bytes_per_latent_matches_the_measured_sd_family_coefficient() -> None:
    """ComfyUI's hand-measured sd15/SDXL decode coefficient is ``2178 * 64 * dtype_size`` bytes per latent element."""

    class _Cfg:
        block_out_channels = (128, 256, 512, 512)

    class _Vae:
        config = _Cfg()

    got = oom_ladder.decode_bytes_per_latent(_Vae(), dtype_bytes=2)
    assert 0.9 <= got / (2178 * 64 * 2) <= 1.1


def _tiny_vae() -> Any:
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
        memory.apply_low_vram_config(pipe, mode="off")
        out = pipe.vae.decode(latent).sample

    assert out.shape == (1, 3, 192, 192)
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
    """`apply_low_vram_config` runs again on every rung of a placement descent; a second wrapper around the first would double every retry."""
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
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="every tile the ladder has") as caught:
                pipe.vae.decode(torch.randn(1, 4, 24, 24))
    assert "AUTOGRAD" not in str(caught.value)


def test_the_one_condition_under_which_tiling_cannot_help_is_named() -> None:
    """MEASURED on a real card (sd1.5 VAE, 320² latent, RTX 4070): with grad enabled every tile's activations are retained for backward, the retry accumulates instead of bounding, and all four rungs fail."""
    vae = _tiny_vae()
    vae.decoder = _CardTooSmall(vae.decoder, limit=0)
    pipe = _Pipeline(vae)
    with _Events():
        memory.apply_low_vram_config(pipe, mode="off")
        with torch.enable_grad():
            with pytest.raises(RuntimeError) as caught:
                pipe.vae.decode(torch.randn(1, 4, 24, 24))
    assert "AUTOGRAD IS ENABLED" in str(caught.value)
    assert "torch.no_grad()" in str(caught.value)


class _Denoiser(torch.nn.Module):

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


def test_an_async_accelerator_oom_is_an_oom() -> None:
    """Before this issue an AcceleratorError read as an unclassified crash, so no ladder ran on the ASYNCHRONOUS shape of the same exhaustion."""
    by_code = torch.AcceleratorError("CUDA error: unspecified launch failure")
    by_code.error_code = 2  # type: ignore[attr-defined]
    assert memory.is_cuda_oom(by_code) is True

    by_text = torch.AcceleratorError("CUDA error: out of memory")
    assert memory.is_cuda_oom(by_text) is True

    other = torch.AcceleratorError("CUDA error: an illegal memory access")
    other.error_code = 700  # type: ignore[attr-defined]
    assert memory.is_cuda_oom(other) is False


def test_discarding_the_async_error_is_safe_without_a_card() -> None:
    """It runs on every OOM classification, so it may never raise — including on the cardless box where most of this code is exercised."""
    memory.discard_cuda_async_error()


def test_the_ladder_ignores_a_non_oom_failure() -> None:
    """A ValueError inside the decode is a defect, not a card size: it must reach the caller untouched, with its traceback."""
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
