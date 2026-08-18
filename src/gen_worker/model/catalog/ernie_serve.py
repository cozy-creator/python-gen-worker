"""ERNIE-Image's SERVING half: tuned schema, latent arithmetic, and the loop.

The pgw#1331 catalog convention: ``ernie.py`` is the DECLARATION and imports
diffusers inside its ``build`` callables; this module is what the request path
reads, and it imports ``torch`` and nothing above it.
``scripts/lint_serve_role_closure.py`` asserts that.

**One model, two instances — measured (pgw#1346 B3a).** ``baidu/ERNIE-Image``
and ``baidu/ERNIE-Image-Turbo`` publish transformer configs that differ in
exactly two keys, ``lora_rank`` and ``use_lora``, and NEITHER is a constructor
parameter of ``ErnieImageTransformer2DModel`` in the pinned diffusers — so the
module the mint traces is identical for both. Their scheduler blocks are
identical too (static shift 4.0). What differs is weights and the published
recipe (28 steps at CFG 4.0 versus 8 steps at CFG 1.0), which is the definition
of ``tuned``. The ernie ENDPOINT already said so structurally: its two
``@endpoint`` classes share ONE ``Compile`` object, and its own comment reads
*"The two classes are the two CFG arms of this one family compiled graph."*

**CFG is a BATCH AXIS here, not a call count.** ``ErnieImagePipeline``
concatenates the latents to batch 2 and chunks the prediction, so the guided
lane is a batch-2 GRAPH — unlike FLUX.2-klein and Qwen-Image, where a negative
branch is a second sequential batch-1 forward. That is why ``batch`` is a
declared bucket axis of this family and of neither of those, and it is the same
fact the endpoint declares as ``Fork("cfg", served=(True, False))``.

**The latent affine is LEARNED, so the decode is not this module's.**
ERNIE denormalizes by the VAE's BatchNorm ``running_mean``/``running_var`` —
checkpoint WEIGHTS — and then unpatchifies. A serving constant that is really a
weight cannot appear in a checkpoint-free catalog, so this module's boundary is
VAE-ready latents, exactly as ``flux2_klein_4b_serve``'s is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule
from ..spec import TunedValues
from ._packed_shape import latent_shape as _latent_shape
from ._packed_shape import shape_buckets, unpack_shape

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.ernie import Ernie, ErnieBatch, ErnieShape

#: The text-sequence pin the endpoint installs (`_pin_text_sequence`, ie#544).
#: A PIN, not an axis: ``ErnieImagePipeline`` tokenizes with ``padding=False``
#: and pads only to the longest prompt in the batch, so without it the tensor
#: entering the compiled DiT carried the real token count and every distinct
#: prompt length was its own graph.
TEXT_TOKENS: Final = 512

#: ``text_in_dim`` — the width of the Mistral3 text states the DiT cross-reads.
TEXT_WIDTH: Final = 3072

#: The DiT's ``in_channels``, which is the VAE's 32 channels ALREADY patchified
#: 2x2 by the pipeline (`_patchify_latents`: ``[B,32,H,W] -> [B,128,H/2,W/2]``).
LATENT_CHANNELS: Final = 128

#: The COMBINED VAE-plus-patchify stride, and the endpoint derives it the same
#: way from three agreeing facts. The presets prove it on their own:
#: ``pipeline_ernie_image`` refuses ``height % vae_scale_factor``, and 1200 /
#: 848 / 1264 / 1376 are divisible by 16 and NOT by 32 — a 32 would refuse this
#: endpoint's own 4:3 preset on every request.
LATENT_STRIDE: Final = 16

#: ERNIE-Image's reference resolution presets: the baidu/ERNIE-Image repo bakes
#: exactly these seven into its pipeline, and the endpoint's ``AspectRatio``
#: enum IS this table (ie#345 — no free width/height).
SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (1024, 1024),
    (1200, 896),
    (896, 1200),
    (1264, 848),
    (848, 1264),
    (1376, 768),
    (768, 1376),
)

#: The shape axis: every preset as its packed bucket value, sorted.
SHAPE_BUCKETS: Final[tuple[int, ...]] = shape_buckets(SHAPES)

#: The batch axis: 1 is the distilled (CFG-off) lane, 2 the guided one. The
#: cross product with the shapes is 14 graph classes — which is exactly the
#: entry count the endpoint's own declaration states it cannot collapse
#: ("ERNIE's latents reach the DiT as SPATIAL (B, C, H_lat, W_lat) through an
#: nn.Conv2d patch embed, so 1200x896 and 896x1200 are genuinely different conv
#: graphs. There is no honest dedupe here; the 14 is the real cost.").
BATCH_BUCKETS: Final[tuple[int, ...]] = (1, 2)


def latent_shape(code: int) -> tuple[int, int]:
    """Latent (rows, cols) for one packed shape bucket."""

    return _latent_shape(code, LATENT_STRIDE)


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family."""

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


class ErnieTuned(TunedValues, frozen=True):
    """ERNIE-Image's tuned-value SCHEMA. Catalog stamps the values per slot.

    Field-for-field the ernie endpoint's ``ErnieDefaults``, which is what makes
    ``inst.tuned`` a by-value replacement for ``ctx.defaults`` rather than a
    lossy one. The field carries the WIRE name ``num_inference_steps`` because
    the endpoint's ``RuntimeFormula`` resolves terms by same-named lookup over
    payload-then-recipe, and renaming it here would silently unresolve the
    steps term of both lanes' formulas.

    The 28-step neutral value is ie#533's corrected base (the model card's 50 is
    "more of a max than a recommendation"); the distilled lane's 8 is a
    CHECKPOINT value that the hub stamps onto the Turbo instance, not a second
    schema.
    """

    num_inference_steps: int = 28
    guidance: float = 4.0
    negative: str = ""


def schedule_for(instance: Ernie, *, steps: int) -> Schedule:
    """The sigma ladder for one request, from the family's OWN declared block.

    No ``image_seq_len``: ERNIE's published scheduler config sets
    ``use_dynamic_shifting`` false, so this ladder is a STATIC shift of 4.0 and
    never consults the resolution — which is why the shape axis above is a
    graph fact only, and the same 28 sigmas serve 1024x1024 and 768x1376.

    The raw ladder is the pipeline's own: ``linspace(1.0, 0.0, steps + 1)[:-1]``
    is exactly ``linspace(1.0, 1/steps, steps)``, the convention
    :meth:`FlowMatchEulerDiscrete.schedule` already synthesizes. **This refutes
    the B3 plan's premise that ERNIE's turbo lane needs an explicit-sigma
    ladder**: it needs a step COUNT, and the ladder follows.
    """

    scheduler: FlowMatchEulerDiscrete = instance.scheduler()
    return scheduler.schedule(steps)


def initial_latents(
    *, shape: int, batch: int, seed: int, device: Any, dtype: Any
) -> Tensor:
    """Pure noise for one request, at the DiT's own patchified channel count.

    Seeded on the CPU deliberately, for the reason the Flux and SDXL entries
    give: a request's noise then means the same thing on two different pods.
    Unscaled — a rectified flow starts at sigma 1.0 by construction, where a
    variance-exploding schedule would need ``init_noise_sigma``.
    """

    import torch

    rows, cols = latent_shape(shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch, LATENT_CHANNELS, rows, cols, generator=generator, dtype=torch.float32
    )
    return noise.to(device=device, dtype=dtype)


def text_batch(
    positive: Tensor, negative: Tensor | None, *, lengths: Sequence[int]
) -> tuple[Tensor, Tensor]:
    """The ``(text_bth, text_lens)`` pair one step feeds the DiT.

    UNCOND FIRST when a negative branch is present — the order
    ``ErnieImagePipeline`` builds (``uncond_text_hiddens + text_hiddens``) and
    the one :func:`denoise` chunks against. Swapping it inverts every prompt
    while producing a perfectly plausible image, which is why the pair is built
    here rather than left to the caller.

    ``text_lens`` travels beside the tensor and is what makes the pin
    output-neutral: the DiT masks padded keys with ``arange(Tmax) < text_lens``
    and derives the image position ids from the LENGTHS, never from the padded
    width.
    """

    import torch

    rows = [positive] if negative is None else [negative, positive]
    lens = list(lengths)
    if len(lens) != len(rows):
        raise ValueError(f"{len(lens)} lengths for {len(rows)} conditioning rows")
    return torch.cat(rows, dim=0), torch.tensor(
        [min(int(length), TEXT_TOKENS) for length in lens],
        device=positive.device,
        dtype=torch.long,
    )


def denoise(
    instance: Ernie,
    *,
    shape: ErnieShape,
    batch: ErnieBatch,
    latents: Tensor,
    text_bth: Tensor,
    text_lens: Tensor,
    schedule: Schedule,
    guidance: float,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    Classifier-free guidance rides ONE batched forward: the latents are
    repeated to the traced ``batch`` arity, so a guided step is one call and not
    two. ``batch`` is a declared bucket, so asking for guidance on the batch-1
    class is not a silent half-step — it selects a class that cannot serve it,
    and the mismatch is refused rather than rendered.
    """

    import torch

    guided = int(batch) == 2
    # ``Schedule.timesteps`` is ``sigma * num_train_timesteps``, which is what
    # the pipeline feeds (``t_batch = torch.full(..., t.item(), dtype=dtype)``
    # over ``scheduler.timesteps``) — and at the COMPUTE dtype, so this is not
    # SDXL's uncast-float32-scalar hazard.
    for index, moment in enumerate(schedule.timesteps):
        model_input = torch.cat([latents, latents], dim=0) if guided else latents
        timestep = torch.full(
            (int(batch),), moment, device=latents.device, dtype=latents.dtype
        )
        prediction = instance.denoiser(
            batch=batch,
            shape=shape,
            hidden_states=model_input,
            timestep=timestep,
            text_bth=text_bth,
            text_lens=text_lens,
        )
        if guided:
            uncond, cond = prediction.chunk(2, dim=0)
            prediction = uncond + guidance * (cond - uncond)
        latents = schedule.step(index, prediction, latents)
        yield index, latents


def generate(
    instance: Ernie,
    *,
    shape: ErnieShape,
    batch: ErnieBatch,
    text_bth: Tensor,
    text_lens: Tensor,
    steps: int,
    guidance: float,
    seed: int,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    """One ERNIE-Image generation, from text states to VAE-ready latents.

    Returns ``(1, LATENT_CHANNELS, rows, cols)`` — everything up to but not
    including the learned BatchNorm affine, the unpatchify and the VAE decode,
    which are the weight holder's because they ARE weights.

    Every heavy call goes through a typed family callable, so this runs
    unchanged against a compiled backing, an eager one, or a fake one.
    Placement is READ off the inputs, never chosen.
    """

    schedule = schedule_for(instance, steps=steps)
    latents = initial_latents(
        shape=int(shape),
        batch=1,
        seed=seed,
        device=text_bth.device,
        dtype=text_bth.dtype,
    )
    for index, latents in denoise(
        instance,
        shape=shape,
        batch=batch,
        latents=latents,
        text_bth=text_bth,
        text_lens=text_lens,
        schedule=schedule,
        guidance=guidance,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return latents


__all__ = [
    "BATCH_BUCKETS",
    "LATENT_CHANNELS",
    "LATENT_STRIDE",
    "SHAPES",
    "SHAPE_BUCKETS",
    "TEXT_TOKENS",
    "TEXT_WIDTH",
    "ErnieTuned",
    "compute_dtype",
    "denoise",
    "generate",
    "initial_latents",
    "latent_shape",
    "schedule_for",
    "text_batch",
    "unpack_shape",
]
