"""Z-Image's SERVING half: tuned schema, latent arithmetic, and the loop.

The pgw#1331 catalog convention: ``z_image.py`` is the DECLARATION and imports
diffusers inside its ``build`` callable; this module is what the request path
reads, and it imports ``torch`` and nothing above it.
``scripts/lint_serve_role_closure.py`` asserts that.

**One model, THREE lanes, and the axis they differ on is not the graph.** The
base checkpoint (``Tongyi-MAI/Z-Image``) and the official DMD Turbo one
(``Tongyi-MAI/Z-Image-Turbo``) publish transformer configs that are identical
key for key, so they are two INSTANCES; the third lane is the base checkpoint
under the PAI 8-step distillation LoRA, which is an overlay on the same
instance. What actually differs between the two checkpoints is the SCHEDULER:
base publishes ``shift: 6.0``, Turbo ``shift: 3.0``.

**That is why ``shift`` is a TUNED field here and is not in the endpoint's
schema.** Today each checkpoint's own ``scheduler/scheduler_config.json``
arrives with its weights, so the endpoint never had to name the difference. A
declared family has ONE scheduler block, which rides the export digest and is
class truth — so a per-checkpoint shift has exactly one honest home, and it is
``tuned``. Left undeclared, the DMD lane would walk the base lane's 6.0 ladder:
a two-fold change in where the schedule spends its steps, on a nine-step walk.
This is a schema ADDITION on migration (tensorhub generates ``z-image``'s schema
from it) and it is recorded as such rather than smuggled.

**Three things about this loop are unusual and all three are load-bearing.**

* the DiT is conditioned on ``1 - sigma``, not on ``sigma`` and not on the
  0..1000 moment: the pipeline feeds ``(1000 - t) / 1000``;
* its output is NEGATED before the step — it predicts the reverse velocity;
* CFG combines as ``pos + scale * (pos - neg)``, which is NOT the usual
  ``neg + scale * (pos - neg)``. At the same scale the two differ by a whole
  unit of guidance, so borrowing another family's line changes every image.

**CFG is a PYTREE ARITY, which is why it is a bucket.** ``ZImagePipeline``
repeats the latents and CONCATENATES the prompt-embed LISTS, so the guided arm
hands the transformer two entries where the unguided one hands it one. Two
graph classes (the endpoint measured 4327 vs 4373 nodes), not one graph at two
sizes — and the resolution really is dynamic within each, which is the
``shape_strategy="dynamic-collapse"`` the endpoint declares.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule
from ..spec import TunedValues

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor

    from ._generated.z_image import ZImage, ZImageBranches

#: The caption pin the endpoint installs (ie#544). A PIN, not an axis.
TEXT_TOKENS: Final = 512

#: ``cap_feat_dim`` — the width of the Qwen3 caption states the DiT reads.
CAPTION_WIDTH: Final = 2560

#: The VAE's latent channels and spatial compression.
#: ``2 ** (len(block_out_channels) - 1)`` is the pipeline's own derivation.
LATENT_CHANNELS: Final = 16
VAE_STRIDE: Final = 8

#: The transformer's patch sizes, from its config's ``all_patch_size`` /
#: ``all_f_patch_size``. They are PYTHON INTS in the traced call and specialize
#: the graph, so they are stated rather than left to a default that could move.
PATCH: Final = 2
FRAME_PATCH: Final = 1

#: Z-Image's preset grids: the 1 MP tier (seven aspects) and the 2 MP tier,
#: which is deliberately REDUCED to three aspects to bound compile-shape growth.
#: Upstream's guidance is that optimal resolutions sit within ~256px of the 1024
#: grid; the model card claims support to 2048x2048 of total area.
SHAPES_1MP: Final[tuple[tuple[int, int], ...]] = (
    (1024, 1024),
    (1152, 864),
    (864, 1152),
    (1248, 832),
    (832, 1248),
    (1280, 720),
    (720, 1280),
)
SHAPES_2MP: Final[tuple[tuple[int, int], ...]] = (
    (1408, 1408),
    (1920, 1088),
    (1088, 1920),
)
SHAPES: Final[tuple[tuple[int, int], ...]] = SHAPES_1MP + SHAPES_2MP

#: The CFG arity axis, and the family's ONLY bucket. 1 is both turbo lanes and
#: any distilled pick served at guidance 1.0; 2 is the guided base walk. The
#: endpoint's ``aot/`` set is two files — ``transformer-cfg-off.mint.json`` and
#: ``transformer-cfg-on.mint.json`` — which is this axis and nothing else.
BRANCH_BUCKETS: Final[tuple[int, ...]] = (1, 2)


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """The latent ``(rows, cols)`` for one pixel size."""

    return height // VAE_STRIDE, width // VAE_STRIDE


def latent_bounds() -> tuple[int, int]:
    """The inclusive range the declared latent extents may take, DERIVED.

    Read off the preset grid rather than chosen, so a symbol's range is the
    product decision's range: the smallest edge any preset asks for is 720px
    (90 latent) and the largest 1920px (240). A dynamic axis whose bounds are
    invented is how a class silently stops covering a row the wire admits.
    """

    edges = [edge for shape in SHAPES for edge in shape]
    return min(edges) // VAE_STRIDE, max(edges) // VAE_STRIDE


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family."""

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


class ZImageTuned(TunedValues, frozen=True):
    """Z-Image's tuned-value SCHEMA. Catalog stamps the values per slot.

    ``num_inference_steps`` carries the WIRE name deliberately (pgw#654 gap #4):
    the endpoint's ``RuntimeFormula`` resolves terms by same-named lookup over
    payload-then-recipe, so renaming it here would silently unresolve the steps
    term of the base lane's formula.

    ``shift`` is the field the endpoint's ``ZImageDefaults`` does not have — see
    this module's header. It is a per-CHECKPOINT scheduler fact (6.0 base, 3.0
    official Turbo) that arrives with the weights today and has to be stamped
    once a declared family owns the scheduler block.
    """

    num_inference_steps: int = 28
    guidance: float = 4.0
    shift: float = 6.0


def schedule_for(
    instance: ZImage, *, steps: int, shift: float | None = None
) -> Schedule:
    """The sigma ladder for one request, from the family's declared block.

    ``shift`` overrides the DECLARED value with this checkpoint's stamped one —
    pass ``inst.tuned.shift``. ``None`` means the declaration's own, which is
    the base checkpoint's 6.0. The override is a ``replace`` on the parsed
    scheduler, not a second parser: every other parameter, and all the
    validation, stays the SDK's.

    No ``image_seq_len``: Z-Image publishes ``use_dynamic_shifting: false``, so
    the ladder does not consult the resolution — even though the pipeline
    computes a ``mu`` and passes it, which ``set_timesteps`` then ignores. The
    raw ladder is ``get_default_z_image_sigmas`` =
    ``linspace(1.0, 1/steps, steps)``, exactly what
    :meth:`FlowMatchEulerDiscrete.schedule` synthesizes.
    """

    scheduler: FlowMatchEulerDiscrete = instance.scheduler()
    if shift is not None:
        scheduler = replace(scheduler, shift=float(shift))
    return scheduler.schedule(steps)


def initial_latents(
    *, width: int, height: int, batch: int, seed: int, device: Any
) -> Tensor:
    """Pure noise for one request, in FLOAT32 and staying there.

    float32 is not a hedge: ``ZImagePipeline`` allocates its latents in float32,
    casts a COPY to the transformer's dtype for each forward, and asserts the
    stepped result is still float32. Seeded on the CPU deliberately, so a
    request's noise means the same thing on two different pods.
    """

    import torch

    rows, cols = latent_grid(width, height)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        batch, LATENT_CHANNELS, rows, cols, generator=generator, dtype=torch.float32
    )
    return noise.to(device=device)


def caption_states(positive: Tensor, negative: Tensor | None) -> Tensor:
    """The caption batch one call carries: POSITIVE first, then negative.

    The pipeline's own order (``prompt_embeds + negative_prompt_embeds``), and
    it is the order :func:`denoise` splits against. Z-Image is the family where
    getting it backwards is least visible — its CFG line is
    ``pos + scale * (pos - neg)``, so a swap does not merely invert the prompt,
    it extrapolates AWAY from it.

    Stacked rather than a list because the traced call takes ONE tensor whose
    leading axis is the CFG arity — see the declaration's wrapper for the
    measured reason (a list is flattened per element, and the two arms would
    then be two different call signatures).
    """

    import torch

    if negative is None:
        return positive
    return torch.cat([positive, negative], dim=0)


def denoise(
    instance: ZImage,
    *,
    branches: ZImageBranches,
    latents: Tensor,
    captions: Tensor,
    schedule: Schedule,
    guidance: float,
) -> Iterator[tuple[int, Tensor]]:
    """The denoising loop, yielding ``(step index, latents)`` after each step.

    One forward per step in both arms: the guided arm is a batch-2 PYTREE, not
    two calls. The three family-specific facts this loop encodes — the
    ``1 - sigma`` conditioning, the negated output, and the
    ``pos + scale * (pos - neg)`` combination — are in this module's header, and
    each of them silently ruins the render if borrowed from another family.
    """

    import torch

    guided = int(branches) == 2
    samples = int(latents.shape[0])
    for index, sigma in enumerate(schedule.sigmas[:-1]):
        model_input = latents.repeat(2, 1, 1, 1) if guided else latents
        timestep = torch.full(
            (int(branches),), 1.0 - sigma, device=latents.device, dtype=torch.float32
        )
        stacked = instance.denoiser(
            branches=branches,
            # The frame axis is the pipeline's own `unsqueeze(2)`; the leading
            # axis is the CFG arity and is this family's bucket.
            x=model_input.unsqueeze(2),
            t=timestep,
            cap_feats=captions,
        )
        # (branches, C, 1, rows, cols) -> (branches, C, rows, cols).
        prediction = stacked.squeeze(2).to(torch.float32)
        if guided:
            positive = prediction[:samples]
            negative = prediction[samples:]
            prediction = positive + guidance * (positive - negative)
        # The DiT predicts the REVERSE velocity, so the flow-match step needs
        # its negation. Upstream's own line, and the one whose omission
        # produces a confidently wrong image rather than an obvious failure.
        latents = schedule.step(index, -prediction, latents.to(torch.float32))
        yield index, latents


def generate(
    instance: ZImage,
    *,
    branches: ZImageBranches,
    width: int,
    height: int,
    captions: Tensor,
    steps: int,
    guidance: float,
    seed: int,
    shift: float | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> Tensor:
    """One Z-Image generation, from caption states to VAE-ready latents.

    Returns ``(B, 16, rows, cols)`` in float32 — everything up to but not
    including the VAE's shift/scale affine and its decode, which belong to the
    component that holds them.

    Every heavy call goes through a typed family callable, so this runs
    unchanged against a compiled backing, an eager one, or a fake one.
    Placement is READ off the inputs, never chosen.
    """

    schedule = schedule_for(instance, steps=steps, shift=shift)
    latents = initial_latents(
        width=width,
        height=height,
        batch=1,
        seed=seed,
        device=captions.device,
    )
    for index, latents in denoise(
        instance,
        branches=branches,
        latents=latents,
        captions=captions,
        schedule=schedule,
        guidance=guidance,
    ):
        if on_step is not None:
            on_step(index, len(schedule))
    return latents


__all__ = [
    "BRANCH_BUCKETS",
    "CAPTION_WIDTH",
    "FRAME_PATCH",
    "LATENT_CHANNELS",
    "PATCH",
    "SHAPES",
    "SHAPES_1MP",
    "SHAPES_2MP",
    "TEXT_TOKENS",
    "VAE_STRIDE",
    "ZImageTuned",
    "caption_states",
    "compute_dtype",
    "denoise",
    "generate",
    "initial_latents",
    "latent_bounds",
    "latent_grid",
    "schedule_for",
]
