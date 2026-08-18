"""Stable Diffusion 1.5 and 2.x's SERVING half: tuned schemas and shape math.

Two families share this module because they share every piece of arithmetic in
it — the /8 VAE stride, the 77-token text length, the packed shape axis, the
CFG arity. What they do NOT share is their graphs, which is why they are two
declarations and not one with two instances; :mod:`gen_worker.model.catalog.
sd15` states that finding at length.

The same split ``flux1_dev_serve`` and ``sdxl_serve`` document: the declaration
modules import diffusers and transformers inside their ``build`` callables,
this module imports nothing above ``torch``, and the import direction is
one-way.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, cast

from ..scheduler import (
    AncestralSchedule,
    DdimSchedule,
    DiscreteSchedule,
    DpmSolverSchedule,
    MultistepHistory,
    UniPcHistory,
    UniPcSchedule,
)
from ..spec import TunedValues

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.sd2 import Sd2
    from ._generated.sd15 import Sd15, Sd15Shape

    #: Either U-Net family. The loop below is written ONCE against both,
    #: because the difference between them is the graph it calls and not the
    #: sequence of calls — which is the honest reading of "same runner set,
    #: different graphs" and the reason they share this module.
    AnySd: TypeAlias = Sd15 | Sd2

    #: The packed shape axis, spelled ONCE. ``Sd2Shape`` is the same closed set
    #: — ``SD2_SHAPES is SD15_SHAPES`` — and a test asserts the two bucket sets
    #: are equal so this annotation cannot quietly become a lie if one grid
    #: moves without the other.
    AnyShape: TypeAlias = Sd15Shape

#: CLIP's pinned prompt length. Both towers pad to ``model_max_length=77``.
TEXT_TOKENS: Final = 77

#: The VAE's spatial stride, and its latent channel count. Identical to SDXL's:
#: the SD VAE topology did not change across 1.x, 2.x and XL — only its
#: ``scaling_factor`` did, which is a weight-side fact and not a graph one.
VAE_STRIDE: Final = 8
LATENT_CHANNELS: Final = 4

#: Cond and uncond ride ONE call, as they do on SDXL. A CFG arity change is a
#: different graph, so it is a fact of the declaration and not a runtime flag.
CFG_BATCH: Final = 2

#: SD1.5's cross-attention width (CLIP-L) and SD2's (OpenCLIP-H). These two
#: numbers are the whole reason SD2 is not an INSTANCE of SD1.5: they size the
#: U-Net's every cross-attention projection, so the two are different graphs
#: carrying different weight shapes, and one compiled cell cannot serve both.
SD15_TEXT_WIDTH: Final = 768
SD2_TEXT_WIDTH: Final = 1024

#: SD1.5's community-standard 512-class shape grid — the set the sd15 endpoint
#: serves, carried across by value (pgw#1346 B2). Non-square by construction,
#: and 768x512 and 512x768 are two genuinely different conv graphs rather than
#: one graph asked politely.
SD15_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (432, 768),
    (480, 640),
    (512, 512),
    (512, 768),
    (640, 480),
    (768, 432),
    (768, 512),
)

#: SD2 and SD-Turbo serve the same grid through the same endpoint functions.
SD2_SHAPES: Final[tuple[tuple[int, int], ...]] = SD15_SHAPES


def pack_shape(width: int, height: int) -> int:
    """One (width, height) pair as ONE bucket-axis integer.

    The same decimal packing ``sdxl_serve.pack_shape`` documents, and for the
    same reason: a bucket axis is a closed set of positive integers and a
    runner's variants are the CROSS PRODUCT of its axes, so two axes would
    demand 49 traced classes to serve these seven shapes.
    """

    if not 0 < height < 10000:
        raise ValueError(f"height {height} is outside the packing's range")
    return width * 10000 + height


def unpack_shape(code: int) -> tuple[int, int]:
    """The (width, height) pair one packed bucket value names."""

    return divmod(code, 10000)


def shape_buckets(shapes: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """One shape grid as a sorted, unique bucket-axis value set."""

    return tuple(sorted({pack_shape(width, height) for width, height in shapes}))


SD15_SHAPE_BUCKETS: Final[tuple[int, ...]] = shape_buckets(SD15_SHAPES)
SD2_SHAPE_BUCKETS: Final[tuple[int, ...]] = shape_buckets(SD2_SHAPES)


def latent_shape(code: int) -> tuple[int, int]:
    """Latent (rows, cols) for one packed shape bucket. Rows before columns."""

    width, height = unpack_shape(code)
    return height // VAE_STRIDE, width // VAE_STRIDE


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THESE families."""

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


#: Every sampler name an SD1.5/SD2 recipe may be stamped with — the sd15
#: ENDPOINT's ``SD15Scheduler`` set carried across BY VALUE, plus the one name
#: the endpoint invokes without offering (``ddim_trailing``, the Hyper-SD15
#: recipe its ``generate_hyper`` function pins). A recipe stamped with a name
#: absent from this Literal is undecodable, so the set is the union of what the
#: payload may ASK for and what the handler may CHOOSE.
Sd15Sampler = Literal[
    "dpmpp_2m",
    "dpmpp_2m_karras",
    "dpmpp_2m_sde_karras",
    "euler",
    "euler_a",
    "ddim",
    "ddim_trailing",
    "unipc",
    "lcm",
]


class Sd15Tuned(TunedValues, frozen=True):
    """SD1.5's tuned-value SCHEMA.

    ``steps`` carries the endpoint's WIRE name ``num_inference_steps`` — a
    by-value migration of ``SD15Defaults``, whose own comment records that the
    wire name is load-bearing for payload-over-default resolution.
    """

    scheduler: Sd15Sampler = "dpmpp_2m_karras"
    num_inference_steps: int = 30
    guidance: float = 7.0
    negative: str = ""


class Sd15LoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay: every field is "no opinion" unless stated."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    num_inference_steps: int | None = None
    guidance: float | None = None
    scheduler: Sd15Sampler | None = None


class Sd2Tuned(TunedValues, frozen=True):
    """SD2 / SD-Turbo's tuned-value SCHEMA.

    Its OWN schema and not a reuse of :class:`Sd15Tuned`, for the reason
    th#1139 gives: a Turbo recipe validated against the SD1.5 vocabulary would
    accept a 30-step CFG-7 stamping that destroys a one-step distilled
    checkpoint. Same fields, different defaults, different validation identity.
    """

    scheduler: Sd15Sampler = "euler_a"
    num_inference_steps: int = 1
    guidance: float = 0.0
    negative: str = ""


class Sd2LoraTuned(TunedValues, frozen=True):
    """SD2's LoRA-kind overlay."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    num_inference_steps: int | None = None
    guidance: float | None = None
    scheduler: Sd15Sampler | None = None


# ---------------------------------------------------------------- the pipeline

#: The SD VAE's latent affine — 0.18215 for 1.x and 2.x, where SDXL uses
#: 0.13025. Weight-side truth: the decoder graph never sees it.
SCALING_FACTOR: Final = 0.18215


def token_ids(ids: Sequence[int], *, device: Any) -> Tensor:
    """CLIP token IDs, padded/truncated to the encoder's fixed 77-slot window."""

    import torch

    row = list(ids)[:TEXT_TOKENS]
    row.extend([0] * (TEXT_TOKENS - len(row)))
    return torch.tensor([row], device=device, dtype=torch.long)


#: Every schedule an SD1.5 / SD2 request can resolve to. See the sdxl entry
#: for why this is a union rather than a base class.
AnySchedule = (
    DiscreteSchedule | AncestralSchedule | DdimSchedule | DpmSolverSchedule | UniPcSchedule
)


def schedule_for(
    instance: AnySd, *, steps: int, objective: str = "epsilon"
) -> AnySchedule:
    """The schedule for one request, from the SAMPLER this checkpoint names.

    ``objective`` arrives per request: the SD2 768-v rows are v-prediction and
    the rest are epsilon, under ONE declaration — a checkpoint fact, so it
    cannot live in the family's block. The sampler is the same kind of fact and
    arrives the same way, off ``inst.tuned.scheduler`` (pgw#1346 K10).
    """

    return instance.scheduler().objective(objective).schedule(steps)


def encode_prompt(instance: AnySd, *, positive: Tensor, negative: Tensor) -> Tensor:
    """One text tower, both CFG branches, uncond FIRST."""

    import torch

    return torch.cat(
        [instance.clip(input_ids=negative), instance.clip(input_ids=positive)], dim=0
    )


def initial_latents(
    *, shape: AnyShape, seed: int, device: Any, dtype: Any, sigma: float
) -> Tensor:
    """Pure noise at the ladder's own starting scale. CPU-seeded, as Flux is."""

    import torch

    rows, cols = latent_shape(shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        1, LATENT_CHANNELS, rows, cols, generator=generator, dtype=torch.float32
    )
    return (noise * sigma).to(device=device, dtype=dtype)


#: The keying that makes an ANCESTRAL sampler reproducible (pgw#1346 K10).
#:
#: ``euler_a`` consumes a fresh noise tensor at every step, so "same seed, same
#: image" is a claim about that noise stream and not only about the initial
#: latents. Two properties are wanted and they pull against each other:
#: reproducibility across pods, and independence from the loop's shape.
#:
#: A running generator gives the first and not the second — step ``k``'s noise
#: then depends on every draw before it, so a resumed loop, a preview pass or a
#: reordered call site silently re-rolls the tail. So the stream is KEYED
#: instead: step ``k`` draws from its own CPU generator seeded by mixing the
#: request seed with ``k``. The mix is the 64-bit splitmix64 finalizer, chosen
#: because adjacent seeds must not produce correlated streams and
#: ``seed + k`` does exactly that — it makes step ``k`` of seed ``s`` identical
#: to step ``k-1`` of seed ``s+1``.
#:
#: CPU-seeded for the reason ``initial_latents`` gives: a CUDA generator's
#: stream is a property of the device, so a receipt replayed on another card
#: would resolve different noise. On the CPU it cannot.
_MIX: Final = 0x9E3779B97F4A7C15
_MASK: Final = 0xFFFFFFFFFFFFFFFF


def step_seed(seed: int, index: int) -> int:
    """The seed step ``index`` of request ``seed`` draws its noise from."""

    value = (seed + _MIX * (index + 1)) & _MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
    return (value ^ (value >> 31)) & _MASK


def step_noise(*, shape: int, seed: int, index: int, device: Any, dtype: Any) -> Tensor:
    """The ancestral noise for ONE step. Keyed by ``(seed, index)`` — see above."""

    import torch

    rows, cols = latent_shape(shape)
    generator = torch.Generator(device="cpu").manual_seed(step_seed(seed, index))
    noise = torch.randn(
        1, LATENT_CHANNELS, rows, cols, generator=generator, dtype=torch.float32
    )
    return noise.to(device=device, dtype=dtype)


def denoise(
    instance: AnySd,
    *,
    shape: AnyShape,
    latents: Tensor,
    prompt_embeds: Tensor,
    schedule: AnySchedule,
    guidance: float,
    seed: int = 0,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    No ``added_cond_kwargs``: SD1.5 and SD2 have no micro-conditioning block.
    That absence is the whole shape difference between this loop and SDXL's.
    """

    import torch

    def _noise(index: int, like: Tensor) -> Tensor:
        return step_noise(
            shape=int(shape), seed=seed, index=index, device=like.device, dtype=like.dtype
        )

    history: object = (
        schedule.begin() if isinstance(schedule, DpmSolverSchedule | UniPcSchedule) else None
    )

    for index, timestep in enumerate(schedule.timesteps):
        stacked = torch.cat([latents] * CFG_BATCH)
        # The pre-scale is the euler/ddim family's, NOT the multistep solvers'.
        # DPM-Solver++ and UniPC integrate in the unscaled parameterisation —
        # their `init_noise_sigma` is 1.0 — so dividing their latents by
        # `sqrt(sigma**2+1)` would be a wrong image and never an error. The
        # union type is what makes the omission a TYPE error rather than a
        # discovery on a pod.
        batched = (
            schedule.scale_model_input(index, stacked)
            if isinstance(schedule, DiscreteSchedule | AncestralSchedule | DdimSchedule)
            else stacked
        )
        prediction = instance.denoiser(
            shape=shape,
            sample=batched,
            timestep=torch.tensor(timestep, device=latents.device, dtype=torch.float32),
            encoder_hidden_states=prompt_embeds,
        )
        uncond, cond = prediction.chunk(2)
        guided = uncond + guidance * (cond - uncond)
        if isinstance(schedule, AncestralSchedule):
            # An ancestral step CONSUMES noise; the parameter is required, and
            # the noise is CPU-seeded and keyed by (seed, index) — see above.
            latents = schedule.step(index, guided, latents, _noise(index, latents))
        elif isinstance(schedule, DpmSolverSchedule):
            # A MULTISTEP solver carries history between steps, so the loop
            # threads it rather than the schedule holding it: a schedule that
            # kept `self._model_outputs` is one two concurrent requests share.
            latents, history = schedule.step(
                index,
                guided,
                latents,
                cast("MultistepHistory", history),
                noise=(
                    _noise(index, latents)
                    if schedule.algorithm_type == "sde-dpmsolver++"
                    else None
                ),
            )
        elif isinstance(schedule, UniPcSchedule):
            latents, history = schedule.step(
                index, guided, latents, cast("UniPcHistory", history)
            )
        else:
            latents = schedule.step(index, guided, latents)
        yield index, latents


def decode(instance: AnySd, *, shape: AnyShape, latents: Tensor) -> Tensor:
    """Latents to an image tensor in ``[0, 1]``, ``(B, 3, H, W)``."""

    decoded = instance.decoder(shape=shape, latents=latents / SCALING_FACTOR)
    return (decoded / 2 + 0.5).clamp(0, 1)


def generate(
    instance: AnySd,
    *,
    shape: AnyShape,
    positive: Tensor,
    negative: Tensor,
    steps: int,
    guidance: float,
    seed: int,
    objective: str = "epsilon",
    on_step: Any = None,
) -> Tensor:
    """One SD1.5 or SD2 generation, end to end, no model library on the path."""

    schedule = schedule_for(instance, steps=steps, objective=objective)
    prompt_embeds = encode_prompt(instance, positive=positive, negative=negative)
    latents = initial_latents(
        shape=shape,
        seed=seed,
        device=positive.device,
        dtype=prompt_embeds.dtype,
        sigma=schedule.init_noise_sigma,
    )
    for index, latents in denoise(
        instance,
        shape=shape,
        latents=latents,
        prompt_embeds=prompt_embeds,
        schedule=schedule,
        guidance=guidance,
        seed=seed,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return decode(instance, shape=shape, latents=latents)


__all__ = [
    "CFG_BATCH",
    "SCALING_FACTOR",
    "decode",
    "denoise",
    "encode_prompt",
    "generate",
    "initial_latents",
    "schedule_for",
    "step_noise",
    "step_seed",
    "AnySchedule",
    "token_ids",
    "LATENT_CHANNELS",
    "SD15_SHAPES",
    "SD15_SHAPE_BUCKETS",
    "SD15_TEXT_WIDTH",
    "SD2_SHAPES",
    "SD2_SHAPE_BUCKETS",
    "SD2_TEXT_WIDTH",
    "TEXT_TOKENS",
    "VAE_STRIDE",
    "Sd15LoraTuned",
    "Sd15Sampler",
    "Sd15Tuned",
    "Sd2LoraTuned",
    "Sd2Tuned",
    "compute_dtype",
    "latent_shape",
    "pack_shape",
    "shape_buckets",
    "unpack_shape",
]
