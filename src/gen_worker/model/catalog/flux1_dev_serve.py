"""FLUX.1-dev's SERVING half: tuned schemas, packing math, and the loop.

The catalog convention pgw#1331 introduces, and the reason the family's entry is
two modules instead of one:

* ``flux1_dev.py`` is the **declaration**. Its ``build`` callables construct
  diffusers and transformers modules, so it is MINT-SIDE, and a serve process
  that imported it would be one static edge away from acquiring a model library.
* ``flux1_dev_serve.py`` — this module — is everything the REQUEST PATH needs
  and nothing else. It imports ``torch`` (the runtime) and nothing above it: no
  ``diffusers``, no ``transformers``, no pipeline object, no scheduler object.
  ``scripts/lint_serve_role_closure.py`` asserts that, walking function-local
  imports, and ``gen_worker.serve.role.FORBIDDEN_LIBRARIES`` is the list it
  reads.

The direction is one-way: the declaration imports from here, never the reverse.
That is what keeps ONE definition of the family's shape arithmetic — the mint
traces the graph classes at the token counts this module computes, so an
artifact cannot be minted at a shape the loop will not ask for.

**What "bare math" means here, concretely.** ``FluxPipeline.__call__`` is ~250
lines that reach the same arithmetic through a pipeline registry, a scheduler
object with mutable step state, a VAE image processor, and a component-loading
layer. Every one of those is a serve-path dependency bought to run: two
reshapes, a permute, an affine rescale, and one Euler step (which lives in
:mod:`gen_worker.model.scheduler`). The functions below are that arithmetic,
typed, with the family's own declared constants as their parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule
from ..spec import TunedValues

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.flux1_dev import Flux1Dev, Flux1DevResolution


#: T5 prompt tokens the text branch is padded to. A PINNED length, not an axis:
#: a variable text dimension reaching a statically compiled denoiser mints a new
#: graph per prompt length — unbounded and un-warmable.
TEXT_TOKENS: Final = 512

#: CLIP-L's context window. Fixed by the encoder's learned position embeddings,
#: so it is architecture and not a choice this catalog gets to make.
CLIP_TOKENS: Final = 77

#: The VAE's spatial stride, and the transformer's packing factor. Together they
#: turn a pixel edge into a token count, which is the ONLY arithmetic the
#: declaration does.
VAE_STRIDE: Final = 8
PATCH: Final = 2

#: The latent channel count the transformer packs 2x2 blocks of into 64.
LATENT_CHANNELS: Final = 16

#: The affine the VAE's latent space is normalized by. Architecture, not tuning:
#: these two are properties of the trained autoencoder.
SCALING_FACTOR: Final = 0.3611
SHIFT_FACTOR: Final = 0.1159


def latent_edge(resolution: int) -> int:
    """Latent rows/cols for one pixel edge.

    Rounded to an even number, because the transformer packs 2x2 latent blocks
    and an odd edge would leave a half-patch the packing cannot express.
    """

    return 2 * (resolution // (VAE_STRIDE * PATCH))


def packed_tokens(resolution: int) -> int:
    """Packed transformer tokens for one square pixel edge."""

    edge = latent_edge(resolution) // PATCH
    return edge * edge


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family.

    A layout token is opaque to the SDK — it records the token and never
    interprets it (torchcg G15) — so the mapping to a torch dtype is the
    family's, stated once here rather than inferred at each site that needs it.
    """

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


class Flux1DevTuned(TunedValues, frozen=True):
    """FLUX.1-dev's tuned-value SCHEMA. The values are catalog, per release slot.

    Every field is a knob a checkpoint may legitimately have a different opinion
    about. Nothing here is a graph fact: a value that changed the graph would
    belong on a bucket axis, where the type system can keep the class set
    exhaustive.

    It lives in the SERVING half because the request path resolves against it on
    every call, and the declaration half is the one the serve role may not
    import.
    """

    steps: int = 28
    guidance: float = 3.5
    shift: float = 3.0
    #: A CLAMP, never a wire reshape: a request asking for more is served at
    #: this value with an adjustment recorded, not refused.
    max_guidance: float | None = None


class Flux1DevLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay for the same family: every field is "no opinion".

    ``None`` means the overlay declines to override the checkpoint's own tuned
    value. That is why every field is optional and none carries a real default:
    an overlay with an opinion it did not mean to have would silently retune
    every checkpoint it is applied to.
    """

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    guidance: float | None = None
    shift: float | None = None


# --------------------------------------------------------------- packing math


def pack_latents(latents: Tensor) -> Tensor:
    """``(B, C, H, W)`` latents -> ``(B, H/2 * W/2, C*4)`` transformer tokens.

    One view, one permute, one reshape. This is the entirety of what the
    pipeline's ``_pack_latents`` does, and it is here rather than imported
    because importing it costs the package it lives in.
    """

    batch, channels, height, width = latents.shape
    blocked = latents.view(batch, channels, height // PATCH, PATCH, width // PATCH, PATCH)
    return blocked.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, (height // PATCH) * (width // PATCH), channels * PATCH * PATCH
    )


def unpack_latents(tokens: Tensor, *, edge: int) -> Tensor:
    """The inverse of :func:`pack_latents`, back to ``(B, C, edge, edge)``."""

    batch, _, channels = tokens.shape
    grid = edge // PATCH
    blocked = tokens.view(batch, grid, grid, channels // (PATCH * PATCH), PATCH, PATCH)
    return blocked.permute(0, 3, 1, 4, 2, 5).reshape(
        batch, channels // (PATCH * PATCH), edge, edge
    )


def latent_image_ids(edge: int, *, device: Any, dtype: Any) -> Tensor:
    """The rope coordinate of every packed image token: ``(tokens, 3)``.

    Channel 0 is the (unused, always-zero) temporal axis; channels 1 and 2 are
    the token's row and column in the packed grid.
    """

    import torch

    grid = edge // PATCH
    ids = torch.zeros(grid, grid, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(grid, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(grid, device=device, dtype=dtype)[None, :]
    return ids.reshape(grid * grid, 3)


def text_ids(tokens: int, *, device: Any, dtype: Any) -> Tensor:
    """The rope coordinate of every text token — all zero, by construction.

    Flux gives the text branch no positional identity in the joint rope; the
    tensor exists because the graph class's ingress declares it, not because it
    carries information.
    """

    import torch

    return torch.zeros(tokens, 3, device=device, dtype=dtype)


def initial_latents(
    *,
    resolution: int,
    batch: int,
    seed: int,
    device: Any,
    dtype: Any,
) -> Tensor:
    """Pure noise, packed, for one request. Seeded on the CPU, deliberately.

    A CPU generator makes a request's noise reproducible across GPU models and
    driver versions, which is what lets a receipt's seed mean the same thing on
    two different pods. The cost is one host-to-device copy of a few megabytes.
    """

    import torch

    edge = latent_edge(resolution)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch, LATENT_CHANNELS, edge, edge, generator=generator, dtype=torch.float32
    )
    return pack_latents(noise.to(device=device, dtype=dtype))


def denormalize_latents(tokens: Tensor, *, resolution: int) -> Tensor:
    """Undo the VAE's latent affine and unpack, ready for the decoder call."""

    latents = unpack_latents(tokens, edge=latent_edge(resolution))
    return (latents / SCALING_FACTOR) + SHIFT_FACTOR


def to_image_range(decoded: Tensor) -> Tensor:
    """The decoder's ``[-1, 1]`` output as ``[0, 1]``, clamped."""

    return (decoded / 2 + 0.5).clamp(0, 1)


# ---------------------------------------------------------------- the pipeline


def schedule_for(instance: Flux1Dev, *, steps: int, resolution: int) -> Schedule:
    """The sigma ladder for one request, from the family's OWN declared block.

    The scheduler's constants ride the export digest, so a re-declared schedule
    changes the family's identity rather than silently changing every request.
    """

    scheduler: FlowMatchEulerDiscrete = instance.scheduler()
    return scheduler.schedule(steps, image_seq_len=packed_tokens(resolution))


def encode_prompt(
    instance: Flux1Dev,
    *,
    clip_ids: Tensor,
    t5_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Both text branches, through the family's typed callables.

    Returns ``(prompt_embeds, pooled_projections)``: T5's per-token states drive
    the joint attention, CLIP-L's pooled vector conditions the modulation. The
    token IDs arrive already tokenized — tokenization is a pure-data step with
    no model in it, and keeping it outside means this function needs no
    tokenizer library either.
    """

    pooled = instance.clip(input_ids=clip_ids)
    embeds = instance.t5(input_ids=t5_ids)
    return embeds, pooled


def denoise(
    instance: Flux1Dev,
    *,
    resolution: Flux1DevResolution,
    latents: Tensor,
    prompt_embeds: Tensor,
    pooled_projections: Tensor,
    schedule: Schedule,
    guidance: float,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    A generator rather than a function returning the final tensor, so a caller
    can report progress, honour cancellation and stream previews without this
    module knowing what a request is. That is the same reason the scheduler is
    passed in resolved: nothing in this loop reads request state.

    ``guidance`` is FLUX.1-dev's EMBEDDED guidance — a conditioning input to the
    transformer, not classifier-free guidance. There is no negative branch and
    no second forward per step; a CFG family would declare a different loop.
    """

    import torch

    device = latents.device
    dtype = latents.dtype
    image_ids = latent_image_ids(latent_edge(resolution), device=device, dtype=dtype)
    txt_ids = text_ids(prompt_embeds.shape[1], device=device, dtype=dtype)
    guidance_input = torch.full((latents.shape[0],), guidance, device=device, dtype=dtype)

    for index, sigma in enumerate(schedule.sigmas[:-1]):
        timestep = torch.full((latents.shape[0],), sigma, device=device, dtype=dtype)
        velocity = instance.denoiser(
            resolution=resolution,
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=image_ids,
            txt_ids=txt_ids,
            guidance=guidance_input,
        )
        latents = schedule.step(index, velocity, latents)
        yield index, latents


def decode(
    instance: Flux1Dev,
    *,
    resolution: Flux1DevResolution,
    latents: Tensor,
) -> Tensor:
    """Packed latents to an image tensor in ``[0, 1]``, ``(B, 3, H, W)``."""

    return to_image_range(
        instance.decoder(
            resolution=resolution,
            latents=denormalize_latents(latents, resolution=resolution),
        )
    )


def generate(
    instance: Flux1Dev,
    *,
    resolution: Flux1DevResolution,
    clip_ids: Tensor,
    t5_ids: Tensor,
    steps: int,
    guidance: float,
    seed: int,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    """One FLUX.1-dev generation, end to end, with no model library imported.

    Every heavy call goes through a typed family callable, so the whole of this
    function runs unchanged against a compiled backing, an eager one, or a fake
    one — which is what lets it be tested in CI without a card and served from
    an artifact on a pod (pgw#1326's dual backing).

    Placement is READ off the inputs, never chosen: the token tensors arrive on
    the device the caller put them on, and the dtype comes from the layout the
    family's own denoiser variant was traced against. A loop that picked its own
    device is how a request lands on the wrong card.
    """

    device = clip_ids.device
    dtype = compute_dtype(str(instance.variant("denoiser", {"resolution": resolution}).layout))
    prompt_embeds, pooled = encode_prompt(instance, clip_ids=clip_ids, t5_ids=t5_ids)
    latents = initial_latents(
        resolution=resolution,
        batch=int(prompt_embeds.shape[0]),
        seed=seed,
        device=device,
        dtype=dtype,
    )
    schedule = schedule_for(instance, steps=steps, resolution=resolution)
    for index, latents in denoise(
        instance,
        resolution=resolution,
        latents=latents,
        prompt_embeds=prompt_embeds,
        pooled_projections=pooled,
        schedule=schedule,
        guidance=guidance,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return decode(instance, resolution=resolution, latents=latents)


def clip_token_ids(ids: Sequence[int], *, device: Any) -> Tensor:
    """CLIP-L token IDs, padded/truncated to the encoder's fixed window."""

    return _ids(ids, length=CLIP_TOKENS, device=device)


def t5_token_ids(ids: Sequence[int], *, device: Any) -> Tensor:
    """T5 token IDs, padded/truncated to the family's pinned text length."""

    return _ids(ids, length=TEXT_TOKENS, device=device)


def _ids(ids: Sequence[int], *, length: int, device: Any) -> Tensor:
    import torch

    row = list(ids)[:length]
    row.extend([0] * (length - len(row)))
    return torch.tensor([row], device=device, dtype=torch.long)


__all__ = [
    "CLIP_TOKENS",
    "LATENT_CHANNELS",
    "PATCH",
    "SCALING_FACTOR",
    "SHIFT_FACTOR",
    "TEXT_TOKENS",
    "VAE_STRIDE",
    "Flux1DevLoraTuned",
    "Flux1DevTuned",
    "clip_token_ids",
    "compute_dtype",
    "decode",
    "denoise",
    "denormalize_latents",
    "encode_prompt",
    "generate",
    "initial_latents",
    "latent_edge",
    "latent_image_ids",
    "pack_latents",
    "packed_tokens",
    "schedule_for",
    "t5_token_ids",
    "text_ids",
    "to_image_range",
    "unpack_latents",
]
