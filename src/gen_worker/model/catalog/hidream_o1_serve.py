"""HiDream-O1's SERVING half: the tuned schema, the shape grid, the samplers.

The pgw#1331 catalog convention: everything the REQUEST PATH needs and nothing
else. It imports no model library — not even ``torch`` — because everything the
catalog owns for this model today is arithmetic and a choice.

**Read the declaration module's header before this one.** HiDream-**O1** is not
HiDream-**I1**, and almost every assumption a reader brings from the rest of
this catalog is false here: there is no VAE, no CLIP/T5/Llama text tower, no
latent space. The transformer eats RGB pixels through a 32x32 patch embed and
text through its OWN embedding table, in one unified sequence.

Two things follow that shape this module:

* **the pixel-to-token stride is the patch size and nothing else** (32), because
  there is no VAE to compress first. ``latent_channels`` is 3 and the scale
  factor is 1 — stated explicitly in :mod:`gen_worker.model.catalog.hidream_o1`
  rather than left as an absence, because a reader looking for the VAE will
  otherwise assume it was forgotten;
* **the sampler is chosen PER REQUEST**, from the resolved recipe's
  ``model_type`` and the number of reference images. That is pgw#1346 K10 in its
  purest form: the choice is a TUNED value, and a declaration can name one
  scheduler. :func:`sampler_for` is that choice, stated as data so the migration
  has one place to read it from instead of re-deriving the endpoint's branch.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from ..spec import TunedValues

#: The transformer's patch size, and therefore the full pixel-to-token stride.
#: There is no VAE in this model, so this is the ONLY spatial reduction on the
#: path. Every served size must be a multiple of it — the pipeline pins its
#: height and width division factors to exactly this number.
PATCH_SIZE: Final = 32

#: The image channels the denoiser eats. THREE, because they are RGB pixels.
IMAGE_CHANNELS: Final = 3

#: One timestep token rides the unified sequence beside the text and the image.
TIMESTEP_TOKENS: Final = 1

#: The endpoint's own prompt caps. Not the model's — the model applies no
#: padding, no ``max_length`` and no truncation at all, which is precisely why
#: its text axis cannot be pinned and its compile block was deleted. These are
#: the endpoint's declared bounds, migrated by value so the family surface
#: carries the only limits that actually exist.
MAX_PROMPT_CHARS: Final = 8000
MAX_PROMPT_TOKENS: Final = 512


class HiDreamO1Sampler(StrEnum):
    """The three samplers this model's serving path reaches.

    A closed set here and NOT a ``SchedulerKind``: the kinds enum names what a
    catalog declaration may declare, and none of these is declarable while
    ``GraphModelSpec.scheduler`` is single-valued (pgw#1346 K10). This enum
    names what the model REACHES, which is a different and currently larger set
    — and writing the difference down is the point.
    """

    #: The 28-entry re-noising ladder, implemented in
    #: :mod:`gen_worker.model.scheduler_hidream`.
    FLASH = "flash"
    #: The same 28-entry ladder walked by a plain flow-match Euler step.
    FLOW_MATCH = "flow_match"
    #: UniPC multistep predictor-corrector, in flow-prediction mode. NOT
    #: implemented in this repo — see :func:`sampler_for`.
    UNIPC = "unipc"


class HiDreamO1Tuned(TunedValues, frozen=True):
    """HiDream-O1's tuned-value SCHEMA, field-for-field the endpoint's.

    Migrated BY VALUE including the field NAMES, and here that is not a
    preference: tensorhub already publishes ``hidream-o1.schema.json`` with
    exactly these names, and its ``num_inference_steps`` property carries a
    comment recording that it was renamed FROM ``steps`` by migration 0046 and
    that the endpoint-side rename had to ship in the same window. Re-spelling
    any field here would make every stamped hidream-o1 catalog row undecodable
    — the same defect class pgw#1346 B2 found in the pre-existing ``SdxlTuned``.

    ``model_type`` is the field the sampler branch reads, which makes it a
    class-level fact about SERVING and not merely a label: see
    :func:`sampler_for`.
    """

    model_type: str = "dev"
    num_inference_steps: int = 28
    cfg_scale: float = 1.0
    shift: float = 1.0
    noise_scale: float = 7.5
    noise_clip_std: float = 2.5
    #: A CLAMP on the caller's guidance, never a wire reshape. The dev recipes
    #: set it to 1.0 — a distilled checkpoint asked for real CFG produces a
    #: second forward it was never trained to oppose.
    max_cfg: float | None = None


def sampler_for(tuned: HiDreamO1Tuned, *, reference_images: int) -> HiDreamO1Sampler:
    """Which sampler one request runs, from the resolved recipe and the payload.

    The endpoint's own branch, moved to the family that owns it and stated as
    one function so the migration reads it rather than re-deriving it:

    * ``model_type == "full"`` -> UniPC, on every lane;
    * ``model_type == "dev"`` with exactly ONE reference image -> flow-match;
    * ``model_type == "dev"`` otherwise (text-to-image, or two or more
      references) -> flash.

    The dev pair walk the SAME 28-entry ladder and differ only in the step: the
    flash one re-noises. So the branch that looks like a scheduler choice is
    really a stochastic/deterministic choice on one schedule.

    **This is the shape K10 has to grow to hold.** A declaration can name one
    scheduler; this model needs a set keyed by a tuned value AND a payload
    count. Returning the enum member rather than a constructed object is
    deliberate — the caller builds what it can, and UniPC is not implemented in
    this repo at all (it is the ``full`` lane, and B3b's owed math is the flash
    ladder, which IS implemented). Constructing an unimplemented sampler is a
    refusal that belongs at the call site, where the request can be answered.
    """

    if tuned.model_type == "full":
        return HiDreamO1Sampler.UNIPC
    if reference_images == 1:
        return HiDreamO1Sampler.FLOW_MATCH
    return HiDreamO1Sampler.FLASH


def image_tokens(width: int, height: int) -> int:
    """Vision tokens one image contributes to the unified sequence."""

    return (height // PATCH_SIZE) * (width // PATCH_SIZE)


def sequence_length(
    width: int, height: int, *, prompt_tokens: int, reference_tokens: int = 0
) -> int:
    """The unified sequence one request builds: text, target, references, TMS.

    Written out because it is the arithmetic that decides this family's shape
    story, and the story is that there IS no fixed shape: ``prompt_tokens`` is
    unbounded by the model (no padding, no truncation) and ``reference_tokens``
    is a multiset over up to eleven images at count-dependent sizes. Two free
    terms in one sequence length is why the endpoint deleted its compile block,
    and this function is that sentence as code.
    """

    return (
        prompt_tokens
        + image_tokens(width, height)
        + reference_tokens
        + TIMESTEP_TOKENS
    )


def reference_edge(width: int, height: int, *, count: int) -> int:
    """The longest edge each reference image is resized to, for ``count`` refs.

    The upstream ladder, verbatim: references share the token budget, so the
    more you send the smaller each one is rendered. Carried here because it is
    the other half of :func:`sequence_length`'s reference term and because a
    reader estimating cost from "eleven images" without it will be wrong by a
    large factor.
    """

    if count < 1:
        raise ValueError("a reference edge is only defined for at least one image")
    longest = max(width, height)
    if count == 1:
        return longest
    if count == 2:
        return longest * 48 // 64
    if count <= 4:
        return longest // 2
    if count <= 8:
        return longest * 24 // 64
    return longest // 4


__all__ = [
    "IMAGE_CHANNELS",
    "MAX_PROMPT_CHARS",
    "MAX_PROMPT_TOKENS",
    "PATCH_SIZE",
    "TIMESTEP_TOKENS",
    "HiDreamO1Sampler",
    "HiDreamO1Tuned",
    "image_tokens",
    "reference_edge",
    "sampler_for",
    "sequence_length",
]
