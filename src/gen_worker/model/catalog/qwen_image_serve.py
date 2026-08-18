"""Qwen-Image's SERVING half: tuned schema, packing math, and the loop.

The pgw#1331 catalog convention: ``qwen_image.py`` is the DECLARATION and
imports diffusers inside its ``build`` callables; this module is what the
request path reads, and it imports ``torch`` and nothing above it.
``scripts/lint_serve_role_closure.py`` asserts that.

**Where Qwen-Image departs from the other declared DiTs, and why each matters.**

* **CFG is a CALL COUNT, not a batch axis.** ``QwenImagePipeline`` runs TRUE
  CFG: two SEQUENTIAL batch-1 forwards per step, each through the same
  pinned-text graph — never SDXL's or ERNIE's single batch-2 pass. So the
  declaration's classes are all batch 1 and guidance is not a bucket. The
  endpoint's own declaration says the same thing in the other direction, by
  declaring the CFG regime a payload ``CompileAxis`` rather than a shape.
* **The true-CFG combination is NORM-PRESERVING.** After combining, the
  pipeline rescales the result to the conditional prediction's own per-token
  norm. It is one line and it is not optional: dropping it changes contrast and
  saturation on every guided render, which is the class of difference that
  looks like a checkpoint change rather than a bug.
* **The patch grid is BAKED, so transposed presets do NOT collapse.** ``H_pat``
  and ``W_pat`` reach the graph as python ints inside ``img_shapes`` and the
  rope tables are built from them, so 1472x1104 and 1104x1472 bake different
  constants at the same token count — two classes, not one. This is the exact
  opposite of FLUX.2-klein, whose rope coordinates arrive as TENSORS and whose
  presets therefore collapse onto token counts. It is why this family's bucket
  axis is a packed (width, height) pair rather than a token count.
* **The ladder is STRETCHED TO A TERMINAL SIGMA.** Qwen-Image is the only
  family in the catalog whose published scheduler config sets
  ``shift_terminal`` (0.02), which no other declared family carries and which
  :class:`~gen_worker.model.scheduler.FlowMatchEulerDiscrete` does not read.
  :mod:`gen_worker.model.flow_ladders` adds exactly that, over the same
  declared block.

**The boundary is VAE-ready latents.** The decode is not a declared runner —
the endpoint's ``Compile`` declares ``targets=("transformer",)`` — and the
latent affine is the VAE's own ``latents_mean``/``latents_std``, which belong
to the component that holds them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Final

from ..flow_ladders import FlowMatchLadder
from ..scheduler import Schedule
from ..spec import TunedValues
from ._packed_shape import latent_shape as _latent_shape
from ._packed_shape import shape_buckets, unpack_shape

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.qwen_image import QwenImage, QwenImageShape

#: The text-sequence pin the t2i endpoint installs (ie#544 -> the declared
#: ``Compile(text_len=)`` contract). A PIN, not an axis: diffusers' t2i pipeline
#: only TRIMS at ``max_sequence_length``, so the endpoint's ``encode_prompt``
#: wrapper pads every encode — conditional AND unconditional — back up to
#: exactly this, and always hands the denoiser a mask.
TEXT_TOKENS: Final = 512

#: ``joint_attention_dim`` — the width of the Qwen2.5-VL states the DiT reads.
JOINT_DIM: Final = 3584

#: The VAE's latent channels and spatial compression, the checkpoint's own 2x2
#: patchify, and the pixel-to-token stride their product implies. A 1328px edge
#: is 83 tokens across.
VAE_CHANNELS: Final = 16
VAE_STRIDE: Final = 8
PATCH: Final = 2
TOKEN_STRIDE: Final = VAE_STRIDE * PATCH

#: The denoiser's ``in_channels``: 16 latent channels folded through a 2x2
#: patch is 64.
PACKED_CHANNELS: Final = VAE_CHANNELS * PATCH * PATCH

#: Qwen-Image's preset grid, ENDPOINT-ordered: the official ~1.7 MP table the
#: final training phase optimized (the ``megapixels=2`` tier and the default),
#: then the ~1 MP budget tier. Fourteen rows, and the endpoint's own
#: ``aot/transformer-<w>x<h>.mint.json`` set is fourteen files — one per row.
SHAPES: Final[tuple[tuple[int, int], ...]] = (
    # megapixels=2 — Qwen/Qwen-Image's published aspect table.
    (1328, 1328),
    (1472, 1104),
    (1104, 1472),
    (1584, 1056),
    (1056, 1584),
    (1664, 928),
    (928, 1664),
    # megapixels=1 — the budget grid inside the model's supported envelope.
    (1024, 1024),
    (1152, 864),
    (864, 1152),
    (1248, 832),
    (832, 1248),
    (1280, 720),
    (720, 1280),
)

#: The shape axis: every preset as its packed bucket value, sorted.
SHAPE_BUCKETS: Final[tuple[int, ...]] = shape_buckets(SHAPES)


def patch_grid(code: int) -> tuple[int, int]:
    """The ``(rows, cols)`` patch grid one packed shape bucket names."""

    return _latent_shape(code, TOKEN_STRIDE)


def packed_tokens(code: int) -> int:
    """Denoiser tokens for one packed shape bucket.

    This is the flow-match schedule's ``image_seq_len``, so the ladder a
    request walks is a function of its bucket and of nothing else.
    """

    rows, cols = patch_grid(code)
    return rows * cols


def img_shapes(code: int, *, batch: int = 1) -> list[list[tuple[int, int, int]]]:
    """The ``img_shapes`` argument one call carries: ``[[(1, rows, cols)]] * B``.

    PYTHON INTS, and that is the whole reason this family is bucketed on a
    shape rather than a token count: the DiT's ``pos_embed`` builds its rope
    tables from these values, so they SPECIALIZE the traced graph and can never
    be torch symbols. The endpoint records the same fact from the other side —
    an attempt to mark the token axis dynamic died on
    ``you marked img_tok as dynamic but your code specialized it to a constant``.
    """

    rows, cols = patch_grid(code)
    return [[(1, rows, cols)]] * batch


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family."""

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


class QwenImageTuned(TunedValues, frozen=True):
    """Qwen-Image's tuned-value SCHEMA. Catalog stamps the values per slot.

    Field-for-field the qwen-image endpoint's ``QwenImageDefaults``, which is
    what makes ``inst.tuned`` a by-value replacement for ``ctx.defaults``
    rather than a lossy one. ``negative`` is Qwen's single-space convention and
    NOT the empty string — the pipeline's true-CFG branch treats an empty
    negative as "no unconditional prompt", which is a different render.

    The 30-step default is ie#488's gate-4 H100 same-seed sweep (s30-cfg4 at
    parity with the 50-step card copy at 60% of the walk); ``guidance`` is the
    true-CFG scale, where ``do_true_cfg`` is ``> 1`` with a negative present.
    ``max_guidance`` is a CLAMP, never a wire reshape.
    """

    steps: int = 30
    guidance: float = 4.0
    negative: str = " "
    max_guidance: float | None = None


def schedule_for(instance: QwenImage, *, steps: int, shape: int) -> Schedule:
    """The sigma ladder for one request, from the family's OWN declared block.

    Read through :class:`~gen_worker.model.flow_ladders.FlowMatchLadder` rather
    than through ``instance.scheduler()``, and the difference is REAL rather
    than stylistic: Qwen-Image's published block sets ``shift_terminal`` to
    0.02, which the SDK's flow-match class does not read, so the generated
    ``scheduler()`` would resolve an UNSTRETCHED ladder from the same numbers.
    Both readings are built from ``SCHEDULER_PARAMETERS``, which rides the
    export digest, so neither can drift from the declaration — one of them just
    honours two more of its keys. ``tests/test_qwen_image_pgw1346.py`` asserts
    they differ, so the reason this indirection exists is measured and not
    asserted in prose.

    Dynamic shifting is ON here, so the ladder DOES consult the resolution:
    a 1664x928 render and a 1024x1024 one walk different sigmas at the same
    step count.
    """

    # Indexed by SAMPLER (pgw#1346 K10): `SCHEDULER_PARAMETERS` is keyed by the
    # name a checkpoint is stamped with. This family declares exactly one, so
    # the key is a constant here rather than a checkpoint fact.
    return FlowMatchLadder.from_block(
        instance.SCHEDULER_PARAMETERS["flow_match_euler"]
    ).schedule(
        steps, image_seq_len=packed_tokens(shape)
    )


def pack_latents(latents: Tensor) -> Tensor:
    """``(B, 16, H, W)`` -> ``(B, (H/2)*(W/2), 64)``, the pipeline's own 2x2 fold."""

    batch, channels, height, width = latents.shape
    folded = latents.view(batch, channels, height // PATCH, PATCH, width // PATCH, PATCH)
    folded = folded.permute(0, 2, 4, 1, 3, 5)
    return folded.reshape(
        batch, (height // PATCH) * (width // PATCH), channels * PATCH * PATCH
    )


def unpack_latents(tokens: Tensor, *, shape: int) -> Tensor:
    """The inverse of :func:`pack_latents`, to the VAE's own 5-D input.

    ``(B, C, 1, rows*2, cols*2)`` — the singleton frame axis is
    ``AutoencoderKLQwenImage``'s, which decodes video-shaped latents and takes
    a still image as one frame.
    """

    batch, _, channels = tokens.shape
    rows, cols = patch_grid(shape)
    view = tokens.view(batch, rows, cols, channels // (PATCH * PATCH), PATCH, PATCH)
    view = view.permute(0, 3, 1, 4, 2, 5)
    return view.reshape(
        batch, channels // (PATCH * PATCH), 1, rows * PATCH, cols * PATCH
    )


def initial_latents(
    *, shape: int, batch: int, seed: int, device: Any, dtype: Any
) -> Tensor:
    """Pure noise for one request, already packed to the denoiser's token axis.

    Seeded on the CPU deliberately, for the reason the other entries give: a
    request's noise then means the same thing on two different pods. Unscaled —
    a rectified flow starts at sigma 1.0 by construction.
    """

    import torch

    rows, cols = patch_grid(shape)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch,
        VAE_CHANNELS,
        rows * PATCH,
        cols * PATCH,
        generator=generator,
        dtype=torch.float32,
    )
    return pack_latents(noise.to(device=device, dtype=dtype))


def prompt_mask(lengths: Sequence[int], *, device: Any) -> Tensor:
    """The ``encoder_hidden_states_mask`` for one padded batch of prompts.

    int64 and always present: the endpoint's pin returns a mask on every
    encode so the denoiser's input signature never depends on whether this
    particular prompt filled the 512-slot window.
    """

    import torch

    rows = [
        [1] * min(int(length), TEXT_TOKENS)
        + [0] * (TEXT_TOKENS - min(int(length), TEXT_TOKENS))
        for length in lengths
    ]
    return torch.tensor(rows, device=device, dtype=torch.long)


def denoise(
    instance: QwenImage,
    *,
    shape: QwenImageShape,
    latents: Tensor,
    prompt_embeds: Tensor,
    prompt_mask_ids: Tensor,
    negative_embeds: Tensor | None,
    negative_mask_ids: Tensor | None,
    schedule: Schedule,
    guidance: float,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    TRUE classifier-free guidance: when a negative branch is present the loop
    runs two SEQUENTIAL batch-1 forwards per step and combines them, then
    rescales the combination back to the conditional prediction's per-token
    norm. Sequential, not batched — which is why every declared graph class is
    batch 1 and CFG is not a bucket axis.

    The timestep the DiT receives is the SIGMA, not the 0..1000 moment: the
    pipeline passes ``timestep / 1000``. Getting that wrong conditions the
    model on a step a thousand times too large and renders noise.
    """

    import torch

    shapes = img_shapes(int(shape), batch=int(latents.shape[0]))
    use_cfg = negative_embeds is not None and guidance > 1.0
    for index, sigma in enumerate(schedule.sigmas[:-1]):
        timestep = torch.full(
            (int(latents.shape[0]),), sigma, device=latents.device, dtype=latents.dtype
        )
        prediction = instance.denoiser(
            shape=shape,
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask_ids,
            timestep=timestep,
            img_shapes=shapes,
        )
        if use_cfg:
            assert negative_embeds is not None  # narrowed by use_cfg
            uncond = instance.denoiser(
                shape=shape,
                hidden_states=latents,
                encoder_hidden_states=negative_embeds,
                encoder_hidden_states_mask=(
                    negative_mask_ids if negative_mask_ids is not None else prompt_mask_ids
                ),
                timestep=timestep,
                img_shapes=shapes,
            )
            combined = uncond + guidance * (prediction - uncond)
            # The norm-preserving rescale: `comb * (||cond|| / ||comb||)`, per
            # token. Upstream's own line, and dropping it shifts contrast and
            # saturation on every guided render.
            combined = combined * (
                torch.norm(prediction, dim=-1, keepdim=True)
                / torch.norm(combined, dim=-1, keepdim=True)
            )
            prediction = combined
        latents = schedule.step(index, prediction, latents)
        yield index, latents


def generate(
    instance: QwenImage,
    *,
    shape: QwenImageShape,
    prompt_embeds: Tensor,
    prompt_mask_ids: Tensor,
    negative_embeds: Tensor | None,
    negative_mask_ids: Tensor | None,
    steps: int,
    guidance: float,
    seed: int,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    """One Qwen-Image generation, from prompt states to VAE-ready latents.

    Returns ``(B, 16, 1, rows*2, cols*2)`` — everything up to but not including
    the VAE's ``latents_mean``/``latents_std`` affine and its decode, which
    belong to the component that holds them.

    Every heavy call goes through a typed family callable, so this runs
    unchanged against a compiled backing, an eager one, or a fake one.
    Placement is READ off the inputs, never chosen.
    """

    schedule = schedule_for(instance, steps=steps, shape=int(shape))
    latents = initial_latents(
        shape=int(shape),
        batch=int(prompt_embeds.shape[0]),
        seed=seed,
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    for index, latents in denoise(
        instance,
        shape=shape,
        latents=latents,
        prompt_embeds=prompt_embeds,
        prompt_mask_ids=prompt_mask_ids,
        negative_embeds=negative_embeds,
        negative_mask_ids=negative_mask_ids,
        schedule=schedule,
        guidance=guidance,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return unpack_latents(latents, shape=int(shape))


__all__ = [
    "JOINT_DIM",
    "PACKED_CHANNELS",
    "PATCH",
    "SHAPES",
    "SHAPE_BUCKETS",
    "TEXT_TOKENS",
    "TOKEN_STRIDE",
    "VAE_CHANNELS",
    "VAE_STRIDE",
    "QwenImageTuned",
    "compute_dtype",
    "denoise",
    "generate",
    "img_shapes",
    "initial_latents",
    "pack_latents",
    "packed_tokens",
    "patch_grid",
    "prompt_mask",
    "schedule_for",
    "unpack_latents",
    "unpack_shape",
]
