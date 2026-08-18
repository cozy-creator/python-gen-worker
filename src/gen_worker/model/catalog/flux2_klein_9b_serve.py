"""FLUX.2-klein-9B's SERVING half: its tuned schemas and its denoising loop.

The pgw#1331 catalog convention, same as ``flux1_dev_serve`` and
``flux2_klein_4b_serve``: this module is everything the REQUEST PATH needs and
nothing else. It imports ``torch`` and nothing above it — no ``diffusers``, no
``transformers``, no pipeline object. ``scripts/lint_serve_role_closure.py``
asserts that.

**Almost nothing here is 9B-specific, and that is the point.** klein-9B and
klein-4B are two architectures (different width, different depth, separately
registered hub families) sharing one FAMILY: identical VAE, identical packing,
identical 4-D rope grids, identical three-layer Qwen3 prompt stack, identical
scheduler block, identical token-bucket axis. Those invariants are imported
from :mod:`gen_worker.model.catalog.flux2_klein_4b_serve` rather than restated,
because pgw#1346's two authoring lanes agreed that whichever landed first would
hold the shared surface and the other would import it — B1 (4B) landed first.

**The import direction is a landing order, not a dependency claim** — those
constants are FLUX.2-klein truth and not 4B's. Owed follow-up, filed rather
than done here because it would edit a file another open lane owns: once both
declarations are on master, move the shared half into a neutral
``flux2_klein_serve`` module and collapse the two loops below onto one, typed
``AnyKlein: TypeAlias = Flux2Klein4b | Flux2Klein9b`` — exactly the shape B2
already uses for ``Sd15 | Sd2`` in ``sd15_serve``. That is a mechanical move
with no behaviour in it.

**What IS 9B's own**, and why each one is a class fact rather than a tuned one:

* the Qwen3 encoder is 4096 wide, not 2560, so the stacked prompt embedding —
  and therefore the denoiser's ``joint_attention_dim`` — is 12288, not 7680;
* the denoiser is 8 double blocks + 24 single blocks over 32 heads, against
  4B's 5 + 20 over 24. A differing architecture config is a different
  ``ModelSpec`` by construction (``model/runtime.py::_materialize`` — an
  instance carries only ref, tuned, backing and label), which is why this is a
  second catalog entry and not a third klein instance;
* the bf16 serving floor is 44 GB rather than 30.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule
from ..spec import TunedValues
from .flux2_klein_4b_serve import (
    PACKED_CHANNELS,
    TEXT_LAYERS,
    TEXT_TOKENS,
    compute_dtype,
    initial_latents,
    latent_grid,
    latent_image_ids,
    packed_tokens,
    text_ids,
    unpack_latents,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.flux2_klein_9b import Flux2Klein9b, Flux2Klein9bTokens


#: Qwen3-8B's hidden width, as klein-9B's ``text_encoder/config.json`` states
#: it (``hidden_size``), and the joint dimension the three-layer stack implies.
#: Declared as the product so a layer-count edit cannot leave the two
#: disagreeing — and so the 12288 in the architecture block below is checkable
#: rather than transcribed.
TEXT_WIDTH: Final = 4096
JOINT_DIM: Final = TEXT_WIDTH * len(TEXT_LAYERS)


class Flux2Klein9bTuned(TunedValues, frozen=True):
    """Klein-9B's tuned-value SCHEMA. The values are catalog, per release slot.

    The same shape as klein-4B's and deliberately its own struct: a tuned
    schema is published to the hub under the model's OWN name
    (``ModelSpec._register`` -> ``register_family``), and tensorhub validates
    ``flux2-klein-9b`` repo metadata against this one. Sharing 4B's object
    would register one schema under two names, which is the collision the
    per-model registration exists to prevent.

    The neutral defaults are the Base checkpoint's published card numbers,
    which is what the hub stamps for an unconfigured checkpoint — the same
    values the endpoint's retired ``Flux2Klein9bDefaults`` carried, migrated by
    value.
    """

    steps: int = 28
    guidance: float = 4.0
    shift: float = 3.0
    #: Whether this checkpoint is the step-distilled one. It gates the CFG
    #: branch rather than describing it: a distilled checkpoint runs one
    #: forward per step at any guidance, and asking it for a negative branch
    #: produces a second forward it was never trained to oppose.
    distilled: bool = False
    #: A CLAMP, never a wire reshape.
    max_guidance: float | None = None


class Flux2Klein9bLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay: every field is "no opinion" (``None``)."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    guidance: float | None = None
    shift: float | None = None


# ---------------------------------------------------------------- the pipeline


def schedule_for(
    instance: Flux2Klein9b, *, steps: int, width: int, height: int
) -> Schedule:
    """The sigma ladder for one request, from the model's OWN declared block."""

    scheduler: FlowMatchEulerDiscrete = instance.scheduler()
    return scheduler.schedule(steps, image_seq_len=packed_tokens(width, height))


def denoise(
    instance: Flux2Klein9b,
    *,
    tokens: Flux2Klein9bTokens,
    width: int,
    height: int,
    latents: Tensor,
    prompt_embeds: Tensor,
    negative_embeds: Tensor | None,
    schedule: Schedule,
    guidance: float,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    ``guidance`` is CLASSIFIER-FREE guidance. klein-9B's transformer takes NO
    guidance embedding (``guidance_embeds`` is false in its config and the call
    passes the literal ``None``), so a negative branch is a second SEQUENTIAL
    batch-1 forward per step — which is why every declared graph class is
    ``B=1`` and CFG is not a bucket axis.
    """

    import torch

    device = latents.device
    dtype = latents.dtype
    batch = int(latents.shape[0])
    rows, cols = latent_grid(width, height)
    image_ids = latent_image_ids(rows, cols, batch=batch, device=device)
    txt_ids = text_ids(int(prompt_embeds.shape[1]), batch=batch, device=device)

    for index, sigma in enumerate(schedule.sigmas[:-1]):
        timestep = torch.full((batch,), float(sigma), device=device, dtype=dtype)
        velocity = instance.denoiser(
            tokens=tokens,
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            img_ids=image_ids,
            txt_ids=txt_ids,
        )
        if negative_embeds is not None:
            uncond = instance.denoiser(
                tokens=tokens,
                hidden_states=latents,
                encoder_hidden_states=negative_embeds,
                timestep=timestep,
                img_ids=image_ids,
                txt_ids=txt_ids,
            )
            velocity = uncond + guidance * (velocity - uncond)
        latents = schedule.step(index, velocity, latents)
        yield index, latents


def unpack_for_vae(latents: Tensor, *, width: int, height: int) -> Tensor:
    """Packed tokens -> ``(B, PACKED_CHANNELS, rows, cols)``, ready for the VAE.

    Where this model's catalog-owned serve path STOPS, for the same measured
    reason as klein-4B's: the endpoint's fifteen preset sizes collapse onto
    only nine token counts (1184x880 and 880x1184 are both 4070 tokens), so a
    token bucket cannot tell the decoder its output shape, and keying the
    decoder on (rows, cols) would oblige a variant at every one of the 15x15
    cross-product compiled graphs. The VAE therefore stays EAGER — which is also what the
    endpoint already does (``Compile(targets=("transformer",))``) — and the
    learned BatchNorm affine and unpatchify are applied by the holder of those
    weights.
    """

    rows, cols = latent_grid(width, height)
    return unpack_latents(latents, rows=rows, cols=cols)


def generate(
    instance: Flux2Klein9b,
    *,
    tokens: Flux2Klein9bTokens,
    width: int,
    height: int,
    prompt_embeds: Tensor,
    negative_embeds: Tensor | None,
    steps: int,
    guidance: float,
    seed: int,
    distilled: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    """One klein-9B generation, from prompt embeddings to VAE-ready latents.

    Returns ``(B, PACKED_CHANNELS, rows, cols)`` — everything up to but not
    including the VAE decode. Every heavy call goes through a typed model
    callable, so this runs unchanged against a compiled backing, an eager one,
    or a fake one.

    Placement is READ off the inputs, never chosen. The CFG branch is taken
    only for a non-distilled checkpoint above guidance 1.0 — the pipeline's own
    condition, restated where the loop can see it.
    """

    device = prompt_embeds.device
    dtype = compute_dtype(str(instance.variant("denoiser", {"tokens": tokens}).layout))
    use_cfg = (not distilled) and guidance > 1.0 and negative_embeds is not None
    latents = initial_latents(
        width=width,
        height=height,
        batch=int(prompt_embeds.shape[0]),
        seed=seed,
        device=device,
        dtype=dtype,
    )
    schedule = schedule_for(instance, steps=steps, width=width, height=height)
    for index, latents in denoise(
        instance,
        tokens=tokens,
        width=width,
        height=height,
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds if use_cfg else None,
        schedule=schedule,
        guidance=guidance,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return unpack_for_vae(latents, width=width, height=height)


__all__ = [
    "JOINT_DIM",
    "PACKED_CHANNELS",
    "TEXT_LAYERS",
    "TEXT_TOKENS",
    "TEXT_WIDTH",
    "Flux2Klein9bLoraTuned",
    "Flux2Klein9bTuned",
    "compute_dtype",
    "denoise",
    "generate",
    "initial_latents",
    "latent_grid",
    "packed_tokens",
    "schedule_for",
    "unpack_for_vae",
]
