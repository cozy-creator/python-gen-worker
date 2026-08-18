"""FLUX.2-klein-4B's SERVING half: tuned schemas, packing math, and the loop.

The pgw#1331 catalog convention, same as ``flux1_dev_serve``: this module is
everything the REQUEST PATH needs and nothing else. It imports ``torch`` and
nothing above it — no ``diffusers``, no ``transformers``, no pipeline object.
``scripts/lint_serve_role_closure.py`` asserts that.

**Where klein-4B departs from FLUX.1-dev, and why each one matters here.**

* **One text branch, not two.** FLUX.1 pairs CLIP-L (a pooled vector) with
  T5-XXL (per-token states). Klein has a single Qwen3 encoder and NO pooled
  projection at all, so the denoiser's call carries no ``pooled_projections``
  and the loop has no second text stage.
* **The prompt embedding is a STACK of three intermediate layers**, not a last
  hidden state: layers 9/18/27 concatenated on the feature axis, which is the
  whole of why ``joint_attention_dim`` is 7680 = 3 x Qwen3's 2560. Reading only
  the final layer would produce a tensor of the right rank and the wrong
  meaning — the failure that type-checks.
* **Rope coordinates are 4-D** ``(T, H, W, L)`` and shaped ``(B, S, 4)``, where
  FLUX.1's are 3-D and unbatched. Both grids are dense ``cartesian_prod``
  coordinates rather than FLUX.1's mostly-zero text ids.
* **Packing is one reshape, not a 2x2 block shuffle.** Klein's VAE emits 32
  latent channels which the *checkpoint's own* patchify has already folded to
  128, so this module's ``pack`` is the trivial ``(B,C,H,W) -> (B,H*W,C)``.
* **The latent affine is LEARNED, not declared.** FLUX.1 normalizes by two
  config scalars this module can hold as constants; klein denormalizes by the
  VAE's BatchNorm ``running_mean``/``running_var`` — checkpoint WEIGHTS. They
  therefore cannot appear here, and the denormalization lives inside the
  decoder module's own traced forward instead. That is not a workaround: a
  serving constant that is really a weight is exactly the thing the
  checkpoint-free rule exists to keep out of the catalog.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule
from ..spec import TunedValues

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.flux2_klein_4b import Flux2Klein4b, Flux2Klein4bTokens


#: Qwen3 prompt tokens the text branch is padded to. A PINNED length, not an
#: axis, for the same reason as FLUX.1's 512: a variable text dimension
#: reaching a statically compiled denoiser mints a graph per prompt length.
TEXT_TOKENS: Final = 512

#: The Qwen3 hidden layers whose states are concatenated into the prompt
#: embedding. THREE intermediate layers, not the last one — this is a property
#: of how klein was trained, and it is what makes the joint dim 3x the width.
TEXT_LAYERS: Final = (9, 18, 27)

#: Qwen3-4B's hidden width, and the joint dimension it implies once the three
#: layers above are stacked. Declared as the product so a layer-count edit
#: cannot leave the two disagreeing.
TEXT_WIDTH: Final = 2560
JOINT_DIM: Final = TEXT_WIDTH * len(TEXT_LAYERS)

#: The VAE's spatial compression, and the checkpoint's own patchify factor.
#: Their product is the pixel-to-token stride: a 1024px edge is 64 tokens.
VAE_STRIDE: Final = 8
PATCH: Final = 2
TOKEN_STRIDE: Final = VAE_STRIDE * PATCH

#: The VAE's latent channels, and the packed channel count the denoiser's
#: ``in_channels`` must equal (32 x 2 x 2 = 128).
LATENT_CHANNELS: Final = 32
PACKED_CHANNELS: Final = LATENT_CHANNELS * PATCH * PATCH


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """The packed token grid ``(rows, cols)`` for one pixel size."""

    return height // TOKEN_STRIDE, width // TOKEN_STRIDE


def packed_tokens(width: int, height: int) -> int:
    """Packed denoiser tokens for one pixel size.

    This is the family's bucket coordinate: the nine values the declaration
    enumerates are exactly this function over the endpoint's preset grid.
    """

    rows, cols = latent_grid(width, height)
    return rows * cols


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family."""

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


class Flux2Klein4bTuned(TunedValues, frozen=True):
    """Klein-4B's tuned-value SCHEMA. The values are catalog, per release slot.

    The Base and Turbo checkpoints differ ONLY in these values — same graph
    classes, same runners, same loop — which is precisely why they are two
    instances of one model rather than two models (pgw#1346 B1).
    """

    steps: int = 28
    guidance: float = 4.0
    shift: float = 3.0
    #: Whether this checkpoint is the step-distilled one. It gates the CFG
    #: branch below rather than describing it: a distilled checkpoint runs one
    #: forward per step at any guidance, and asking it for a negative branch
    #: produces a second forward whose output it was never trained to oppose.
    distilled: bool = False
    #: A CLAMP, never a wire reshape.
    max_guidance: float | None = None


class Flux2Klein4bLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay: every field is "no opinion" (``None``)."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    guidance: float | None = None
    shift: float | None = None


# --------------------------------------------------------------- packing math


def pack_latents(latents: Tensor) -> Tensor:
    """``(B, C, H, W)`` -> ``(B, H*W, C)``. One reshape and one permute."""

    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def unpack_latents(tokens: Tensor, *, rows: int, cols: int) -> Tensor:
    """The inverse of :func:`pack_latents`, back to ``(B, C, rows, cols)``."""

    batch, _, channels = tokens.shape
    return tokens.permute(0, 2, 1).reshape(batch, channels, rows, cols)


def latent_image_ids(rows: int, cols: int, *, batch: int, device: Any) -> Tensor:
    """Rope coordinates for every packed image token: ``(B, rows*cols, 4)``.

    The four axes are ``(T, H, W, L)``: time and layer are singleton on the
    generate lane, so the grid is the dense row/column cartesian product. Int64
    by declaration — these index a rope table and are never cast to the
    compute dtype.
    """

    import torch

    coords = torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(rows, device=device),
        torch.arange(cols, device=device),
        torch.arange(1, device=device),
    )
    return coords.unsqueeze(0).expand(batch, -1, -1)


def text_ids(tokens: int, *, batch: int, device: Any) -> Tensor:
    """Rope coordinates for every text token: ``(B, tokens, 4)``.

    Text carries position on the LAST axis (``L``), where FLUX.1 gave its text
    branch no positional identity at all.
    """

    import torch

    coords = torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(1, device=device),
        torch.arange(1, device=device),
        torch.arange(tokens, device=device),
    )
    return coords.unsqueeze(0).expand(batch, -1, -1)


def initial_latents(
    *,
    width: int,
    height: int,
    batch: int,
    seed: int,
    device: Any,
    dtype: Any,
) -> Tensor:
    """Pure noise, packed, for one request. Seeded on the CPU, deliberately.

    A CPU generator makes a request's noise reproducible across GPU models and
    driver versions, which is what lets a receipt's seed mean the same thing on
    two different pods.
    """

    import torch

    rows, cols = latent_grid(width, height)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch, PACKED_CHANNELS, rows, cols, generator=generator, dtype=torch.float32
    )
    return pack_latents(noise.to(device=device, dtype=dtype))


# ---------------------------------------------------------------- the pipeline


def schedule_for(
    instance: Flux2Klein4b, *, steps: int, width: int, height: int
) -> Schedule:
    """The sigma ladder for one request, from the family's OWN declared block."""

    scheduler: FlowMatchEulerDiscrete = instance.scheduler()
    return scheduler.schedule(steps, image_seq_len=packed_tokens(width, height))


def stack_prompt_layers(hidden_states: Sequence[Tensor]) -> Tensor:
    """The three intermediate Qwen3 layers -> one ``(B, TEXT_TOKENS, JOINT_DIM)``.

    Klein conditions on layers 9/18/27 CONCATENATED on the feature axis, which
    is the whole of why the joint dim is 7680 rather than 2560. Reading only
    the final layer yields a tensor of the right rank and the wrong meaning.

    The encoder itself is not a declared runner (see the declaration module's
    note), so this takes the layer sequence a caller's eager Qwen3 already
    produced — ``output.hidden_states`` — and does only the stacking, which is
    the part that is family truth rather than library plumbing.
    """

    import torch

    stacked = torch.stack([hidden_states[index] for index in TEXT_LAYERS], dim=1)
    batch, layers, sequence, width = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(batch, sequence, layers * width)


def denoise(
    instance: Flux2Klein4b,
    *,
    tokens: Flux2Klein4bTokens,
    width: int,
    height: int,
    latents: Tensor,
    prompt_embeds: Tensor,
    negative_embeds: Tensor | None,
    schedule: Schedule,
    guidance: float,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    ``guidance`` here is CLASSIFIER-FREE guidance, and klein-4B's transformer
    takes NO guidance embedding (``guidance_embeds`` is false in its config and
    the call passes the literal ``None``). When a negative branch is supplied
    the loop runs two SEQUENTIAL batch-1 forwards per step and combines them —
    sequential, not batched, which is why the declared graph classes are all
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

    This is where the catalog's ownership of klein's serve path STOPS, and the
    reason is a real limit rather than a scoping choice. The VAE decode is not
    a declared runner because it cannot be keyed by this family's bucket axis:
    the endpoint's fifteen preset sizes collapse onto only nine token counts
    (1184x880 and 880x1184 are both 4070 tokens), so a token bucket cannot tell
    the decoder its output shape, while keying it on (rows, cols) as two axes
    would oblige a variant at every one of the 15x15 cross-product compiled graphs.

    So the VAE stays EAGER — which is also exactly what the endpoint does today
    (its ``Compile`` declares ``targets=("transformer",)``) — and the learned
    BatchNorm affine and unpatchify are applied by the holder of those weights.
    Recorded in pgw#1346 as the per-component keying limit K4 also lives on.
    """

    rows, cols = latent_grid(width, height)
    return unpack_latents(latents, rows=rows, cols=cols)


def generate(
    instance: Flux2Klein4b,
    *,
    tokens: Flux2Klein4bTokens,
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
    """One klein-4B generation, from prompt embeddings to VAE-ready latents.

    Returns ``(B, PACKED_CHANNELS, rows, cols)`` — everything up to but not
    including the VAE decode, which :func:`unpack_for_vae` explains is not a
    declared runner for this family. The caller applies the checkpoint's own
    learned affine and decodes.

    Every heavy call goes through a typed family callable, so this runs
    unchanged against a compiled backing, an eager one, or a fake one.

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


def prompt_token_ids(ids: Sequence[int], *, device: Any) -> Tensor:
    """Qwen3 token IDs, padded/truncated to the family's pinned text length."""

    row = list(ids)[:TEXT_TOKENS]
    row.extend([0] * (TEXT_TOKENS - len(row)))

    import torch

    return torch.tensor([row], device=device, dtype=torch.long)


__all__ = [
    "JOINT_DIM",
    "LATENT_CHANNELS",
    "PACKED_CHANNELS",
    "PATCH",
    "TEXT_LAYERS",
    "TEXT_TOKENS",
    "TEXT_WIDTH",
    "TOKEN_STRIDE",
    "VAE_STRIDE",
    "Flux2Klein4bLoraTuned",
    "Flux2Klein4bTuned",
    "compute_dtype",
    "denoise",
    "generate",
    "initial_latents",
    "latent_grid",
    "latent_image_ids",
    "pack_latents",
    "packed_tokens",
    "prompt_token_ids",
    "schedule_for",
    "stack_prompt_layers",
    "text_ids",
    "unpack_for_vae",
    "unpack_latents",
]
