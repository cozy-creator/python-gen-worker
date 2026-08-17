"""FLUX.1-schnell's SERVING half: tuned schemas, packing math, and the loop.

The pgw#1331 catalog convention: this module is everything the REQUEST PATH
needs and nothing else. It imports ``torch`` and nothing above it.

**Schnell is FLUX.1-dev's architecture with ONE bit flipped**, and that is
measured rather than assumed — ``tests/fixtures/flux1_schnell/`` caches the
serving release's own published configs, and the suite asserts every shared
number against them. Of the nine architecture fields both checkpoints publish,
eight are identical and ``guidance_embeds`` is false here and true there. So
this module imports FLUX.1's invariants from :mod:`~gen_worker.model.catalog.
flux1_dev_serve` instead of restating them: one definition of the VAE's affine,
one of the packing factor, one CLIP window.

That one bit is nonetheless why schnell is a separate ``ModelSpec`` and not
another instance of dev. It removes an INPUT from the denoiser's traced call —
``FluxPipeline`` branches on ``transformer.config.guidance_embeds`` and passes
``guidance=None`` — so the two have different graph classes, which is exactly
the line the class/instance split is drawn on.

Two further differences, both consequences of the distillation:

* **The text pin is 256, not 512** (th#1126). BFL's own reference passes
  ``max_sequence_length=256``; diffusers defaults to 512 and pads
  unconditionally, so the larger pin bought a 512-token T5 encode and 512 text
  tokens in every DiT block's joint attention, for nothing.
* **The schedule is STATIC** — ``shift`` 1.0 with ``use_dynamic_shifting``
  false, against dev's 3.0 and true. Sourced from the release, not inferred
  from the family's reputation: the block rides the export digest, so a wrong
  value here would silently re-key the family rather than fail.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule
from ..spec import TunedValues
from .flux1_dev_serve import (
    CLIP_TOKENS,
    LATENT_CHANNELS,
    PATCH,
    SCALING_FACTOR,
    SHIFT_FACTOR,
    VAE_STRIDE,
    clip_token_ids,
    compute_dtype,
    pack_latents,
    to_image_range,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.flux1_schnell import Flux1Schnell, Flux1SchnellTokens

#: T5 prompt tokens the text branch is padded to. Schnell's own pin (th#1126),
#: and the ONE constant here that is not FLUX.1-wide.
TEXT_TOKENS: Final = 256

#: The pixel-to-token stride, restated as a name because schnell's grid is
#: RECTANGULAR and dev's helpers are square-only.
TOKEN_STRIDE: Final = VAE_STRIDE * PATCH


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """The packed token grid ``(rows, cols)`` for one pixel size.

    Schnell serves a rectangular preset grid where dev serves squares, so the
    arithmetic is carried here per-axis rather than borrowed from
    ``flux1_dev_serve``'s single-edge helpers. The stride is the same 16.
    """

    return 2 * (height // TOKEN_STRIDE) // PATCH, 2 * (width // TOKEN_STRIDE) // PATCH


def packed_tokens(width: int, height: int) -> int:
    """Packed denoiser tokens for one pixel size — the family's bucket axis.

    Transposed presets collapse onto one coordinate (ie#685): FLUX.1's rope
    coordinates arrive as TENSORS, so the graph keys on token COUNT and
    1152x864 and 864x1152 are one graph class.
    """

    rows, cols = latent_grid(width, height)
    return rows * cols


class Flux1SchnellTuned(TunedValues, frozen=True):
    """Schnell's tuned-value SCHEMA. The values are catalog, per release slot.

    ``guidance`` is absent by construction, not by oversight: schnell carries no
    guidance embedding and is a timestep distillation, so there is no knob for a
    caller to turn. A field here would be a promise the graph cannot keep.
    """

    steps: int = 4
    #: The distillation's ceiling. 1-4 steps is the published contract, and a
    #: request asking for more is clamped with an adjustment recorded.
    max_steps: int = 4
    shift: float = 1.0


class Flux1SchnellLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay: every field is "no opinion" (``None``)."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    shift: float | None = None


# --------------------------------------------------------------- packing math


def unpack_latents(tokens: Tensor, *, rows: int, cols: int) -> Tensor:
    """The inverse of :func:`pack_latents`, to ``(B, C, rows*2, cols*2)``."""

    batch, _, channels = tokens.shape
    blocked = tokens.view(batch, rows, cols, channels // (PATCH * PATCH), PATCH, PATCH)
    return blocked.permute(0, 3, 1, 4, 2, 5).reshape(
        batch, channels // (PATCH * PATCH), rows * PATCH, cols * PATCH
    )


def latent_image_ids(rows: int, cols: int, *, device: Any, dtype: Any) -> Tensor:
    """The rope coordinate of every packed image token: ``(tokens, 3)``.

    Rank-2 and BATCHLESS, which is FLUX.1's shape and not flux.2-klein's
    ``(B, T, 4)``. Channel 0 is the unused temporal axis; 1 and 2 are the
    token's row and column.
    """

    import torch

    ids = torch.zeros(rows, cols, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(rows, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(cols, device=device, dtype=dtype)[None, :]
    return ids.reshape(rows * cols, 3)


def text_ids(tokens: int, *, device: Any, dtype: Any) -> Tensor:
    """All-zero rope coordinates for the text branch — FLUX.1 gives it none."""

    import torch

    return torch.zeros(tokens, 3, device=device, dtype=dtype)


def initial_latents(
    *, width: int, height: int, batch: int, seed: int, device: Any, dtype: Any
) -> Tensor:
    """Pure noise, packed, for one request. Seeded on the CPU, deliberately."""

    import torch

    rows, cols = latent_grid(width, height)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch, LATENT_CHANNELS, rows * PATCH, cols * PATCH,
        generator=generator, dtype=torch.float32,
    )
    return pack_latents(noise.to(device=device, dtype=dtype))


def denormalize_latents(tokens: Tensor, *, width: int, height: int) -> Tensor:
    """Undo the VAE's latent affine and unpack, ready for the decode call.

    The affine is FLUX.1's, imported rather than restated — dev and schnell
    share one autoencoder and therefore one pair of constants.
    """

    rows, cols = latent_grid(width, height)
    latents = unpack_latents(tokens, rows=rows, cols=cols)
    return (latents / SCALING_FACTOR) + SHIFT_FACTOR


# ---------------------------------------------------------------- the pipeline


def schedule_for(
    instance: Flux1Schnell, *, steps: int, width: int, height: int
) -> Schedule:
    """The sigma ladder for one request, from the family's OWN declared block.

    Schnell's block sets ``use_dynamic_shifting`` false, so ``image_seq_len`` is
    passed and ignored — the static ``shift`` applies. It is threaded anyway so
    that this call reads identically to dev's and a future re-declaration cannot
    silently change which arm runs.
    """

    scheduler: FlowMatchEulerDiscrete = instance.scheduler()
    return scheduler.schedule(steps, image_seq_len=packed_tokens(width, height))


def encode_prompt(
    instance: Flux1Schnell, *, clip_ids: Tensor, t5_ids: Tensor
) -> tuple[Tensor, Tensor]:
    """Both text branches, through the family's typed callables.

    Returns ``(prompt_embeds, pooled_projections)``: T5's per-token states drive
    the joint attention, CLIP-L's pooled vector conditions the modulation.
    """

    pooled = instance.clip(input_ids=clip_ids)
    embeds = instance.t5(input_ids=t5_ids)
    return embeds, pooled


def denoise(
    instance: Flux1Schnell,
    *,
    tokens: Flux1SchnellTokens,
    width: int,
    height: int,
    latents: Tensor,
    prompt_embeds: Tensor,
    pooled_projections: Tensor,
    schedule: Schedule,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    There is NO ``guidance`` parameter, and its absence is the whole difference
    between this loop and dev's. Schnell's transformer declares
    ``guidance_embeds: false``, so the traced call has no guidance input at all
    — not a zero, not a None threaded through, simply not a parameter. There is
    also no negative branch: the checkpoint is a timestep distillation and runs
    one forward per step.
    """

    import torch

    device = latents.device
    dtype = latents.dtype
    rows, cols = latent_grid(width, height)
    image_ids = latent_image_ids(rows, cols, device=device, dtype=dtype)
    txt_ids = text_ids(int(prompt_embeds.shape[1]), device=device, dtype=dtype)

    for index, sigma in enumerate(schedule.sigmas[:-1]):
        timestep = torch.full((latents.shape[0],), float(sigma), device=device, dtype=dtype)
        velocity = instance.denoiser(
            tokens=tokens,
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=image_ids,
            txt_ids=txt_ids,
        )
        latents = schedule.step(index, velocity, latents)
        yield index, latents


def generate(
    instance: Flux1Schnell,
    *,
    tokens: Flux1SchnellTokens,
    width: int,
    height: int,
    clip_ids: Tensor,
    t5_ids: Tensor,
    steps: int,
    seed: int,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    """One schnell generation, from token ids to VAE-ready latents.

    Returns the denormalized latents rather than pixels: the VAE decode is not
    a declared runner for this family, for the reason recorded on the
    declaration — schnell's preset grid is rectangular, so a token-count bucket
    cannot tell a decoder its output shape. The endpoint decodes, which is what
    it does today (`targets=("transformer",)`).

    Placement is READ off the inputs, never chosen.
    """

    device = clip_ids.device
    dtype = compute_dtype(str(instance.variant("denoiser", {"tokens": tokens}).layout))
    prompt_embeds, pooled = encode_prompt(instance, clip_ids=clip_ids, t5_ids=t5_ids)
    latents = initial_latents(
        width=width, height=height, batch=int(prompt_embeds.shape[0]),
        seed=seed, device=device, dtype=dtype,
    )
    schedule = schedule_for(instance, steps=steps, width=width, height=height)
    for index, latents in denoise(
        instance,
        tokens=tokens,
        width=width,
        height=height,
        latents=latents,
        prompt_embeds=prompt_embeds,
        pooled_projections=pooled,
        schedule=schedule,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return denormalize_latents(latents, width=width, height=height)


def t5_token_ids(ids: Sequence[int], *, device: Any) -> Tensor:
    """T5 token IDs, padded/truncated to SCHNELL's 256-token pin."""

    import torch

    row = list(ids)[:TEXT_TOKENS]
    row.extend([0] * (TEXT_TOKENS - len(row)))
    return torch.tensor([row], device=device, dtype=torch.long)


__all__ = [
    "CLIP_TOKENS",
    "LATENT_CHANNELS",
    "PATCH",
    "SCALING_FACTOR",
    "SHIFT_FACTOR",
    "TEXT_TOKENS",
    "TOKEN_STRIDE",
    "VAE_STRIDE",
    "Flux1SchnellLoraTuned",
    "Flux1SchnellTuned",
    "clip_token_ids",
    "compute_dtype",
    "denoise",
    "denormalize_latents",
    "encode_prompt",
    "generate",
    "initial_latents",
    "latent_grid",
    "latent_image_ids",
    "pack_latents",
    "packed_tokens",
    "schedule_for",
    "t5_token_ids",
    "text_ids",
    "to_image_range",
    "unpack_latents",
]
