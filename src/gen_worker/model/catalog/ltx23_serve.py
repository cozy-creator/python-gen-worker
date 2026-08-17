"""LTX-Video 2.3's serve-role half: the token arithmetic and the ladder.

Serve-role safe (pgw#1331): no diffusers, no torch import at module scope, no
checkpoint, no network.

The one thing worth reading here is :func:`schedule_from_sigmas`, which is B4's
answer to the batch plan's *"ltx serves LITERAL ladders -> hard dep on B3's
explicit-sigma work"*: **there is no dependency.**
"""

from __future__ import annotations

from typing import Any, Final

from ..scheduler import Schedule
from ..spec import TunedValues

#: LTX conditions on Gemma3 through a 1024-token window, PINNED by the endpoint
#: on every call rather than inherited: *"diffusers' own default is the same
#: 1024; passing it is what turns a default into a contract."* The tokenizer
#: runs ``padding="max_length"``, so the sequence axis entering the DiT is
#: constant by construction.
TEXT_TOKENS: Final = 1024

#: The two conditioning widths the joint DiT takes. They are DIFFERENT — the
#: audio stream is projected to half the video stream's width — which is why
#: the call carries two encoder tensors rather than one.
CROSS_ATTENTION_DIM: Final = 4096
AUDIO_CROSS_ATTENTION_DIM: Final = 2048

#: ``vae_scale_factors = (8, 32, 32)`` on the transformer's own config:
#: 8x temporal, 32x on each spatial axis. With ``patch_size`` and
#: ``patch_size_t`` both 1, these are the ONLY divisors between pixels and
#: tokens — LTX patches inside the VAE, not in the DiT.
VAE_TEMPORAL: Final = 8
VAE_SPATIAL: Final = 32

#: Latent channels on both streams: 128 in, 128 out, video and audio alike.
LATENT_CHANNELS: Final = 128
AUDIO_LATENT_CHANNELS: Final = 128

#: ``audio_sampling_rate / audio_hop_length / audio_scale_factor``
#: = 16000 / 160 / 4 = 25.0 audio tokens per second of REALIZED duration.
AUDIO_TOKENS_PER_SECOND: Final = 16000 / 160 / 4

#: Flow-matching sigmas are read as ``sigma * NUM_TRAIN_TIMESTEPS`` by the
#: model, which the transformer config states as ``timestep_scale_multiplier``.
NUM_TRAIN_TIMESTEPS: Final = 1000


class Ltx23Tuned(TunedValues, frozen=True):
    """LTX-2.3's stamped recipe, migrated BY VALUE from ``@family("ltx-2")``.

    **The sigma ladders are tuned values, and that is the point.** LTX is
    step-distilled: it does not synthesize a schedule from a step count, it
    walks a fixed list. So ``steps`` is not a field here — the step count IS
    ``len(sigmas)``, which is how the endpoint reads it
    (``steps = len(recipe.sigmas)``).

    The defaults are byte-identical to the distilled constants the pipeline
    ships (``diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES`` and
    ``STAGE_2_...``), so a checkpoint the hub has not stamped serves exactly
    what it served before the cut. ``stage2_sigmas`` is literally the TAIL of
    ``sigmas`` — its last three entries — because the two-stage refine resumes
    the same ladder after a 2x latent upsample rather than starting a new one.
    """

    guidance: float = 1.0
    audio_guidance: float = 1.0
    sigmas: tuple[float, ...] = (
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
    )
    stage2_sigmas: tuple[float, ...] = (0.909375, 0.725, 0.421875)


def schedule_from_sigmas(sigmas: tuple[float, ...]) -> Schedule:
    """One stamped ladder, as a resolved :class:`~gen_worker.model.scheduler.Schedule`.

    **The B3 dependency the plan recorded for B4 does not exist.** The plan's
    reasoning was that ``FlowMatchEulerDiscrete.schedule()`` synthesizes sigmas
    from ``steps`` alone, so a family serving literal sigmas needs new
    scheduler machinery. It does not: ``Schedule`` is a public frozen dataclass
    over an explicit sigma tuple, and the SYNTHESIS is what
    ``FlowMatchEulerDiscrete`` adds on top. Handing it a stamped list is
    three lines and no new vocabulary.

    The terminal ``0.0`` is appended here rather than stamped, and that is a
    correctness point rather than a formality. LTX's stamped ladder ends at
    0.421875, and diffusers' ``FlowMatchEulerDiscreteScheduler`` appends the
    terminal sigma itself (``final_sigmas_type`` defaulting to zero) — so a
    catalog document that carried the zero would be double-counting a step
    that does not exist, and one that omitted it would leave the last step
    landing NEAR the clean sample instead of on it. ``Schedule`` refuses a
    ladder that does not terminate at zero, which turns that into an import-
    time error rather than a slightly-noisy render.
    """

    rows = tuple(float(value) for value in sigmas)
    if not rows:
        raise ValueError("a stamped ladder needs at least one sigma")
    if rows[-1] == 0.0:
        raise ValueError(
            "a stamped LTX ladder carries only the EVALUATED sigmas; the "
            "terminal 0.0 is appended by the schedule, not by the catalog"
        )
    return Schedule(sigmas=(*rows, 0.0), num_train_timesteps=NUM_TRAIN_TIMESTEPS)


def compute_dtype(layout: str) -> Any:
    """The dtype a runner's modules are BUILT at for one layout contract."""

    import torch

    return torch.bfloat16 if layout in ("bf16", "plain.bf16@1") else torch.float32


def latent_grid(width: int, height: int, frames: int) -> tuple[int, int, int]:
    """``(F_lat, H_lat, W_lat)``, per ``ltx_video_23/main.py:1026-1027``."""

    if frames < 1:
        raise ValueError(f"a clip needs at least one frame, got {frames}")
    return ((frames - 1) // VAE_TEMPORAL + 1, height // VAE_SPATIAL, width // VAE_SPATIAL)


def video_tokens(width: int, height: int, frames: int, *, last_frame: bool = False) -> int:
    """Video tokens one request presents to the joint DiT.

    ``F_lat * H_lat * W_lat``, plus ONE extra latent frame's worth when a
    ``last_frame`` keyframe is bound: it sits at a non-zero latent index, so
    ``prepare_latents`` APPENDS a frame rather than overwriting one. A
    ``first_frame`` keyframe overwrites and costs nothing, which is why the
    growth term is binary rather than open-ended.
    """

    f_lat, h_lat, w_lat = latent_grid(width, height, frames)
    return f_lat * h_lat * w_lat + (h_lat * w_lat if last_frame else 0)


def audio_tokens(frames: int, fps: int) -> int:
    """Audio tokens for one clip: ``round(frames / fps * 25)``.

    The REALIZED duration, not the nominal bucket, and the difference is
    load-bearing: 241 frames at 24 fps is 10.0417 s, so this is **251** and not
    ``round(10 * 25) = 250``. The endpoint found that by measurement.
    """

    if fps < 1:
        raise ValueError(f"fps must be positive, got {fps}")
    return round(frames / fps * AUDIO_TOKENS_PER_SECOND)


__all__ = [
    "AUDIO_CROSS_ATTENTION_DIM",
    "AUDIO_LATENT_CHANNELS",
    "AUDIO_TOKENS_PER_SECOND",
    "CROSS_ATTENTION_DIM",
    "LATENT_CHANNELS",
    "NUM_TRAIN_TIMESTEPS",
    "TEXT_TOKENS",
    "VAE_SPATIAL",
    "VAE_TEMPORAL",
    "Ltx23Tuned",
    "audio_tokens",
    "compute_dtype",
    "latent_grid",
    "schedule_from_sigmas",
    "video_tokens",
]
