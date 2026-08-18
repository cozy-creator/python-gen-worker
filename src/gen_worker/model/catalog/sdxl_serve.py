"""Stable Diffusion XL's SERVING half: tuned schemas and shape arithmetic.

The same two-module split ``flux1_dev_serve`` documents (pgw#1331), applied to
the second catalog entry so the convention is the catalog's and not one
family's exception: ``sdxl.py`` is the DECLARATION and imports diffusers inside
its ``build`` callables; this module is what the request path reads, and it
imports nothing above ``torch``.

pgw#1346 B2 closed the gap this module's first draft recorded. SDXL used to
stop at "tuned schemas and latent arithmetic" because ``euler_discrete`` was
declared and unimplemented, so ``Sdxl`` had no ``scheduler()`` method at all.
It has one now: :class:`~gen_worker.model.scheduler.EulerDiscrete` reproduces
``diffusers``' ladder BIT-EXACTLY, and the declaration beside this module
carries the four-runner tree (both text towers, the U-Net, the VAE decoder)
rather than two thirds of it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast

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

    from ._generated.sdxl import Sdxl, SdxlShape

#: CLIP's pinned prompt length. Pinned, not an axis — a variable text dimension
#: reaching a statically compiled denoiser mints a new graph per prompt length.
TEXT_TOKENS: Final = 77

#: The VAE's spatial stride.
VAE_STRIDE: Final = 8

#: The two-batch classifier-free-guidance arity SDXL is traced at: cond and
#: uncond ride ONE call. A CFG arity change is a different graph, so it is a
#: fact of the declaration rather than a runtime flag.
CFG_BATCH: Final = 2

#: The micro-conditioning vector's width (original/crop/target sizes).
TIME_IDS: Final = 6

#: SDXL's latent channel count.
LATENT_CHANNELS: Final = 4

#: The width of the two text towers' per-token states, concatenated: CLIP-L's
#: 768 beside OpenCLIP-bigG's 1280 is the 2048 the U-Net cross-attends to.
CLIP_L_WIDTH: Final = 768
CLIP_G_WIDTH: Final = 1280
CROSS_ATTENTION_WIDTH: Final = CLIP_L_WIDTH + CLIP_G_WIDTH

#: SDXL's trained multi-aspect bucket grid (SDXL paper, arXiv:2307.01952
#: Appendix I): nine ~1MP shapes, every axis a multiple of 64. This is the
#: FAMILY's shape set, not a product decision — every SDXL-native fine-tune
#: shares the base model's grid, and a conv-heavy U-Net compiles one graph per
#: (H, W), so the pair is the graph class and not a scalar.
SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (640, 1536),
    (768, 1344),
    (832, 1216),
    (896, 1152),
    (1024, 1024),
    (1152, 896),
    (1216, 832),
    (1344, 768),
    (1536, 640),
)


def pack_shape(width: int, height: int) -> int:
    """One (width, height) pair as ONE bucket-axis integer.

    A bucket axis takes a closed set of positive INTEGERS and a runner's
    variants are the CROSS PRODUCT of its axes, so declaring ``width`` and
    ``height`` as two axes would demand 81 traced classes to serve the nine
    shapes SDXL actually has — nine times the mint, for classes no request can
    ever address. One packed axis is total and exact instead.

    The packing is decimal on purpose: ``13440768`` reads as "1344 by 0768" in
    a generated ``Literal``, which is the only place a human meets it. Endpoint
    handlers never spell it — a :class:`~gen_worker.model.spec.BucketMap` binds
    the payload's aspect enum onto this axis.

    ⚠️ pgw#1346 K9: this exists because ``Bucket`` cannot express a shape SET,
    only a shape PRODUCT. A tuple-valued bucket axis is the real fix and it is
    a ``spec.py``/codegen change; this encoding is the honest workaround, not
    the design.
    """

    if not 0 < height < 10000:
        raise ValueError(f"height {height} is outside the packing's range")
    return width * 10000 + height


def unpack_shape(code: int) -> tuple[int, int]:
    """The (width, height) pair one packed bucket value names."""

    return divmod(code, 10000)


#: Every shape as its packed bucket value, sorted — the axis's declared set.
SHAPE_BUCKETS: Final[tuple[int, ...]] = tuple(
    sorted(pack_shape(width, height) for width, height in SHAPES)
)


def latent_edge(resolution: int) -> int:
    """Latent rows/cols for one pixel edge."""

    return resolution // VAE_STRIDE


def latent_shape(code: int) -> tuple[int, int]:
    """Latent (rows, cols) for one packed shape bucket.

    Rows before columns, because that is the order the tensor carries them and
    the transposition is the defect this function exists to prevent: 1344x768
    and 768x1344 are two genuinely different conv graphs, so getting the order
    wrong picks the wrong compiled class rather than producing a wrong image.
    """

    width, height = unpack_shape(code)
    return latent_edge(height), latent_edge(width)


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family.

    A layout token is opaque to the SDK — it records the token and never
    interprets it (torchcg G15) — so the mapping to a torch dtype is the
    family's, stated once here rather than inferred at each site that needs it.
    """

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


#: Every sampler name an SDXL recipe may be stamped with. It is the sdxl
#: ENDPOINT's ``SdxlScheduler`` set, carried across BY VALUE (pgw#1346 B2) —
#: the three distilled entries were absent here and their absence would have
#: made every Hyper-SD / Lightning / DMD2 recipe in the catalog undecodable the
#: moment the endpoint resolved its tuned values through this schema instead of
#: ``ctx.defaults``. tensorhub generates ``sdxl.schema.json`` from this set, so
#: it is a hub-visible contract and not a local convenience.
SdxlSampler = Literal[
    "euler_a",
    "dpmpp_2m_karras",
    "dpmpp_2m_sde_karras",
    # Distilled-checkpoint samplers (few-step, near-zero CFG).
    "lcm",
    "euler_trailing",
    "ddim_trailing",
]


class SdxlTuned(TunedValues, frozen=True):
    """SDXL's tuned-value SCHEMA. Catalog stamps the values, per release slot.

    Field-for-field the sdxl endpoint's ``SdxlDefaults``, which is what makes
    ``inst.tuned`` a by-value replacement for ``ctx.defaults`` rather than a
    lossy one: a field this schema lacks is a stamped recipe value that
    silently stops reaching the handler.
    """

    scheduler: SdxlSampler = "euler_a"
    steps: int = 28
    guidance: float = 6.0
    #: The lineage's quality scaffolding, prepended unless the caller already
    #: brought their own. Present because the endpoint reads it; absent from
    #: the first draft of this schema, which would have dropped it silently.
    quality_preamble: str = ""
    negative: str = ""
    #: A CLAMP, never a wire reshape.
    max_guidance: float | None = None


class SdxlLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay: every field is "no opinion" unless stated."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    guidance: float | None = None
    max_guidance: float | None = None
    scheduler: SdxlSampler | None = None


# ---------------------------------------------------------------- the pipeline

#: The VAE's latent affine. Weight-side truth, not graph truth — the decoder
#: graph never sees it, so it is applied here on the way in.
SCALING_FACTOR: Final = 0.13025


def token_ids(ids: Sequence[int], *, device: Any) -> Tensor:
    """CLIP token IDs, padded/truncated to the encoders' fixed 77-slot window."""

    import torch

    row = list(ids)[:TEXT_TOKENS]
    row.extend([0] * (TEXT_TOKENS - len(row)))
    return torch.tensor([row], device=device, dtype=torch.long)


#: Every schedule an SDXL request can resolve to. A UNION and not a base
#: class: a DDIM trajectory has no sigmas and an ancestral step consumes noise,
#: so the three genuinely differ where it matters and a common supertype would
#: have to hide exactly those differences (pgw#1346 K10).
AnySchedule = (
    DiscreteSchedule | AncestralSchedule | DdimSchedule | DpmSolverSchedule | UniPcSchedule
)


def schedule_for(instance: Sdxl, *, steps: int, objective: str = "epsilon") -> AnySchedule:
    """The schedule for one request, from the SAMPLER this checkpoint names.

    Two checkpoint facts arrive here, both per request rather than from the
    declaration. ``objective`` — SDXL fine-tunes ship both epsilon and
    v-prediction under one architecture, and ``gen_worker.view`` already
    resolves it from the stamped objective for the diffusers path. And the
    SAMPLER, which ``instance.scheduler()`` reads off ``inst.tuned`` — the
    default is ``euler_a``, so this is the common path and not the exotic one.

    No ``image_seq_len``: none of these ladders consults the resolution at all,
    unlike Flux's.
    """

    return instance.scheduler().objective(objective).schedule(steps)


def encode_prompt(
    instance: Sdxl, *, positive: Tensor, negative: Tensor
) -> tuple[Tensor, Tensor]:
    """Both towers, both CFG branches, through the family's typed callables.

    Returns ``(prompt_embeds, pooled)`` already stacked to the traced
    ``CFG_BATCH`` arity, uncond FIRST — the order diffusers uses, and the one
    the guidance combination below is written against. Swapping it inverts
    every prompt while producing a perfectly plausible image, which is why the
    two are built here rather than left to the caller.
    """

    import torch

    rows = []
    pooled_rows = []
    for ids in (negative, positive):
        states_l = instance.clip_l(input_ids=ids)
        states_g, pooled = instance.clip_g(input_ids=ids)
        rows.append(torch.cat([states_l, states_g], dim=-1))
        pooled_rows.append(pooled)
    return torch.cat(rows, dim=0), torch.cat(pooled_rows, dim=0)


def time_ids(shape: int, *, device: Any, dtype: Any) -> Tensor:
    """SDXL's micro-conditioning row: original size, crop offset, target size.

    Uncropped and untouched — ``(h, w, 0, 0, h, w)`` — because a serving
    endpoint renders at the requested size rather than simulating a crop of a
    larger one. Repeated to the CFG arity so it rides one batched call.
    """

    import torch

    width, height = unpack_shape(shape)
    row = [height, width, 0, 0, height, width]
    return torch.tensor([row] * CFG_BATCH, device=device, dtype=dtype)


def initial_latents(
    *, shape: int, seed: int, device: Any, dtype: Any, sigma: float
) -> Tensor:
    """Pure noise at the ladder's own starting scale.

    Seeded on the CPU deliberately, for the reason the Flux entry gives: a
    request's noise then means the same thing on two different pods. Scaled by
    ``init_noise_sigma`` and not by 1.0 — a variance-EXPLODING schedule starts
    far from the data manifold, and starting at unit variance is the classic
    silent way to render a washed-out image.
    """

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
    instance: Sdxl,
    *,
    shape: SdxlShape,
    latents: Tensor,
    prompt_embeds: Tensor,
    pooled: Tensor,
    schedule: AnySchedule,
    guidance: float,
    seed: int = 0,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    Classifier-free guidance rides ONE batched forward: cond and uncond are the
    traced ``CFG_BATCH`` arity, so a step is one call and not two. ``guidance``
    at or below 1.0 still pays for the uncond half here — collapsing to a
    batch-1 graph is a DIFFERENT traced class (the endpoints' turbo lane), not
    a branch inside this loop.
    """

    import torch

    def _noise(index: int, like: Tensor) -> Tensor:
        return step_noise(
            shape=int(shape), seed=seed, index=index, device=like.device, dtype=like.dtype
        )

    history: object = (
        schedule.begin() if isinstance(schedule, DpmSolverSchedule | UniPcSchedule) else None
    )

    added = {
        "text_embeds": pooled,
        "time_ids": time_ids(int(shape), device=latents.device, dtype=latents.dtype),
    }
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
            added_cond_kwargs=added,
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


def decode(instance: Sdxl, *, shape: SdxlShape, latents: Tensor) -> Tensor:
    """Latents to an image tensor in ``[0, 1]``, ``(B, 3, H, W)``."""

    decoded = instance.decoder(shape=shape, latents=latents / SCALING_FACTOR)
    return (decoded / 2 + 0.5).clamp(0, 1)


def generate(
    instance: Sdxl,
    *,
    shape: SdxlShape,
    positive: Tensor,
    negative: Tensor,
    steps: int,
    guidance: float,
    seed: int,
    objective: str = "epsilon",
    on_step: Any = None,
) -> Tensor:
    """One SDXL generation, end to end, with no model library on the path.

    Every heavy call goes through a typed family callable, so this function
    runs unchanged against a compiled backing, an eager one, or a fake one.
    Placement is READ off the inputs and never chosen.
    """

    schedule = schedule_for(instance, steps=steps, objective=objective)
    prompt_embeds, pooled = encode_prompt(
        instance, positive=positive, negative=negative
    )
    latents = initial_latents(
        shape=int(shape),
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
        pooled=pooled,
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
    "time_ids",
    "token_ids",
    "CLIP_G_WIDTH",
    "CLIP_L_WIDTH",
    "CROSS_ATTENTION_WIDTH",
    "LATENT_CHANNELS",
    "SHAPES",
    "SHAPE_BUCKETS",
    "TEXT_TOKENS",
    "TIME_IDS",
    "VAE_STRIDE",
    "SdxlLoraTuned",
    "SdxlSampler",
    "SdxlTuned",
    "compute_dtype",
    "latent_edge",
    "latent_shape",
    "pack_shape",
    "unpack_shape",
]
