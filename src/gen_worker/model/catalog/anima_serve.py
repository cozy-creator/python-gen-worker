"""Anima's SERVING half: tuned schemas, the shape grid, and the sigma ladder.

The pgw#1331 catalog convention: this module is everything the REQUEST PATH
needs and nothing else. It imports no model library at all — not even ``torch``
— because everything Anima's catalog entry owns today is arithmetic.

**Anima's serve path is arithmetic and NOT a loop, and that is the honest
shape.** Every other catalog entry here carries its denoising loop, because it
also carries declared runners the loop calls through. Anima has none — see
:mod:`gen_worker.model.catalog.anima`'s ``ANIMA`` for the two measured reasons
— so what the catalog can own for it is exactly what does not need a module:
the tuned vocabulary, the pixel/latent/token arithmetic, and the schedule.

**The schedule is the reusable finding.** DiffSynth builds Anima's ladder with
``FlowMatchScheduler("Z-Image")``, which reads as a bespoke scheduler and is
not one: its ``set_timesteps_z_image`` is

    sigmas = linspace(1.0, 0.0, N + 1)[:-1]
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)      # shift = 3.0
    timesteps = sigmas * 1000

with the terminal zero supplied by ``step()``'s ``sigma_ = 0`` guard rather
than by the table. That is ``flow_match_euler_discrete`` with a STATIC shift of
3.0 and no dynamic shifting — the same closed form
:class:`~gen_worker.model.scheduler.FlowMatchEulerDiscrete` already implements,
including the appended terminal zero. So Anima owes NO new scheduler math; the
B3 batch plan listed it under "explicit-sigma ladders" on the strength of the
name, and the measurement says otherwise. ``tests/test_anima_pgw1346.py``
differences the two against a transcription of DiffSynth's own formula.

Two consequences worth stating because they are easy to get wrong:

* **the ladder does not consult resolution.** Unlike FLUX.1-dev's dynamic
  shift, Anima's is static, so 512x512 and 1536x1536 walk identical sigmas.
  Anything that keys a schedule cache on resolution is caching a constant;
* **the endpoint never passes ``sigma_shift``**, so the 3.0 below is the value
  every production request runs, not a library default that might be overridden.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ..scheduler import FlowMatchEulerDiscrete, Schedule, SchedulerValue
from ..spec import TunedValues

#: The VAE's spatial compression and the DiT's patch size. Their product is the
#: pixel-to-token stride, and it is also the endpoint's own division factor
#: (DiffSynth pins ``height_division_factor = width_division_factor = 16``), so
#: a preset that is not a multiple of 16 gets silently rounded up before it ever
#: reaches a graph.
VAE_STRIDE: Final = 8
PATCH: Final = 2
TOKEN_STRIDE: Final = VAE_STRIDE * PATCH

#: The VAE's latent channels, which are also the DiT's in/out channels — Anima
#: patchifies INSIDE the transformer, so unlike FLUX there is no packed channel
#: count distinct from the latent one.
LATENT_CHANNELS: Final = 16

#: The Qwen3 prompt length the text branch is padded to. A PINNED length: the
#: pipeline tokenizes ``padding="max_length", max_length=512``, so the tensor
#: entering the DiT is 512 wide on every request.
TEXT_TOKENS: Final = 512

#: The width of those Qwen3-0.6B hidden states, and therefore of the DiT's
#: cross-attention context. NOT the DiT's own residual width (2048) — the two
#: are different numbers and conflating them is the mistake this constant
#: exists to prevent.
TEXT_WIDTH: Final = 1024

#: Anima's scheduler block, as DiffSynth's ``Z-Image`` template states it.
#: Declared as a block rather than as a constructed object so it reads exactly
#: like every other catalog entry's, and so it can ride an export digest
#: unchanged the day Anima earns declared runners.
SCHEDULER: Final[Mapping[str, SchedulerValue]] = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}


class AnimaTuned(TunedValues, frozen=True):
    """Anima's tuned-value SCHEMA. The values are catalog, per release slot.

    Field-for-field the endpoint's retired ``AnimaDefaults``, migrated BY VALUE
    — including the field NAMES. ``num_inference_steps`` is spelled with the
    WIRE name deliberately (pgw#654 gap #4): ``RuntimeFormula`` resolves its
    terms by SAME-NAMED lookup across the payload and the resolved recipe, and
    the anima endpoint's live formula is ``a + b*num_inference_steps``. Spelling
    it ``steps`` here would leave that formula with an unresolvable term and no
    error until a pod evaluated it.

    The neutral defaults are the model card's own (CFG ~4-5, 30-50 steps).
    """

    num_inference_steps: int = 35
    guidance: float = 4.5
    negative: str = ""


class AnimaLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay — and for Anima this is not a formality.

    Anima's Turbo lane is a step-distilled LoRA riding the BASE checkpoint's
    binding (``ModelBinding.loras``), so its recipe describes the ADAPTER and
    not the checkpoint: ten steps at CFG 1.0, applied to weights whose own
    resolved recipe says 35 at 4.5. The endpoint holds those two numbers as
    module constants with a comment reading *"the eventual home for these two
    numbers is the ADAPTER repo's own kind='lora' recipe metadata"*. This is
    that home. Every field is "no opinion" (``None``) so a plain style LoRA
    overrides nothing.
    """

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    num_inference_steps: int | None = None
    guidance: float | None = None


def latent_grid(width: int, height: int) -> tuple[int, int]:
    """The VAE latent grid ``(rows, cols)`` for one pixel size."""

    return height // VAE_STRIDE, width // VAE_STRIDE


def latent_shape(width: int, height: int, *, batch: int = 1) -> tuple[int, int, int, int]:
    """The initial latent tensor's shape, ``(B, C, rows, cols)``.

    Batch is 1 on every served request: ``AnimaImagePipeline.__call__`` exposes
    no images-per-prompt knob at all, and its classifier-free guidance runs two
    SEQUENTIAL forwards rather than a doubled batch.
    """

    rows, cols = latent_grid(width, height)
    return batch, LATENT_CHANNELS, rows, cols


def denoiser_tokens(width: int, height: int) -> int:
    """Patchified DiT tokens for one pixel size.

    This is what a bucket axis would be keyed on the day Anima earns declared
    runners, and it is stated now so the promotion is not also a derivation.
    """

    rows, cols = latent_grid(width, height)
    return (rows // PATCH) * (cols // PATCH)


def scheduler() -> FlowMatchEulerDiscrete:
    """Anima's schedule, built from its own declared block."""

    return FlowMatchEulerDiscrete.from_block(SCHEDULER)


def schedule_for(steps: int) -> Schedule:
    """The sigma ladder for one request.

    No ``image_seq_len``, and its absence is the measurement: Anima shifts
    STATICALLY, so the ladder is a function of the step count alone.
    """

    return scheduler().schedule(steps)


__all__ = [
    "LATENT_CHANNELS",
    "PATCH",
    "SCHEDULER",
    "TEXT_TOKENS",
    "TEXT_WIDTH",
    "TOKEN_STRIDE",
    "VAE_STRIDE",
    "AnimaLoraTuned",
    "AnimaTuned",
    "denoiser_tokens",
    "latent_grid",
    "latent_shape",
    "schedule_for",
    "scheduler",
]
