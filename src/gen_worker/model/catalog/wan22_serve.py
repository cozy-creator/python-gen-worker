"""Wan 2.2's serve-role half: the arithmetic and the tuned vocabulary.

Serve-role safe (pgw#1331): no diffusers, no checkpoint, no network. The
architecture constants and the ``build`` callables that consume them live in
:mod:`gen_worker.model.catalog.wan22`, which the serve role may not import.

**One tuned schema for three models, and that is a measured fact rather than a
convenience.** The endpoint registers exactly one vocabulary — ``@family(
"wan22")`` in ``wan-2.2/src/wan_2_2/defaults.py:25`` — and all three of its
``@endpoint`` classes read it. The three DiTs are three architectures (see
``wan22``'s module docstring) but they share one inference vocabulary, because
what a request tunes (steps, the two experts' guidance, the shift) is the same
question for all three.
"""

from __future__ import annotations

from typing import Any, Final

from ..spec import TunedValues

#: Wan trains on a 1000-step noise schedule; every timestep the fleet quotes
#: (t=875 boundary, t=937.5 ladder rungs) is in these units.
#: ``wan-2.2/src/wan_2_2/scheduling.py:57``.
NUM_TRAIN_TIMESTEPS: Final = 1000

#: The UMT5 prompt window, PINNED by the endpoint on every call rather than
#: inherited: ``_MAX_SEQUENCE_LENGTH = 512`` (``wan_2_2/main.py:365``), and its
#: own comment says why — "diffusers' own default is the same 512, but a
#: default is not a contract". ``text_dim`` is the encoder's width.
TEXT_TOKENS: Final = 512
TEXT_DIM: Final = 4096

#: The DiT patch, identical on all three checkpoints' ``transformer/config.json``
#: (``patch_size: [1, 2, 2]``): no temporal patching, 2x2 spatial.
PATCH_T: Final = 1
PATCH_H: Final = 2
PATCH_W: Final = 2

#: VAE compression, and it is where the two lineages part. A14B carries the
#: Wan 2.1 VAE (``AutoencoderKLWan``, ``dim_mult=[1,2,4,4]`` -> 8x spatial);
#: TI2V-5B carries the Wan 2.2 VAE, whose own config states
#: ``scale_factor_spatial: 16`` outright. Temporal is 4x on both.
#: ``wan_2_2/main.py:396-397``.
A14B_SPATIAL: Final = 8
A14B_TEMPORAL: Final = 4
TI2V_SPATIAL: Final = 16
TI2V_TEMPORAL: Final = 4


class Wan22Tuned(TunedValues, frozen=True):
    """The ``wan22`` inference vocabulary, migrated BY VALUE from the endpoint.

    Field-for-field ``wan-2.2/src/wan_2_2/defaults.py:53-57``, defaults
    included. Two of them carry semantics that a reader cannot recover from
    the names, so they are restated here because losing them is how a stamped
    catalog value stops meaning what it meant:

    * ``guidance`` is the HIGH-noise expert's CFG scale and ``guidance_2`` the
      LOW-noise expert's. On TI2V-5B — one dense DiT, no expert pair — only
      ``guidance`` is consulted.
    * ``max_guidance`` is a CLAMP applied to BOTH, at the endpoint's own
      resolution site. It is never a wire reshape.
    * ``shift`` is the flow-matching shift. ``None`` means "leave the
      checkpoint's own scheduler config alone", which is not the same as 1.0.

    ``num_inference_steps`` carries the WIRE name (pgw#692 / th#1174, migration
    0046 renamed it from ``steps``); ``forbid_unknown_fields`` on the base means
    the two spellings can never coexist in one stamped document.

    **What is NOT here, and is owed to the hub rather than to this file:**
    ``steps_high`` / ``steps_low``. The endpoint's own comment
    (``wan_2_2/main.py:624-627``) says they "belong in the `wan22` family schema
    so a repo can publish its own budgets like it publishes `steps`/`shift`",
    and the declared loop below reads exactly those two parameters. They are
    left out because adding a field to a stamped vocabulary is a tensorhub
    change (K8's ordering law: the hub lands first), not an authoring one.
    """

    num_inference_steps: int = 40
    guidance: float = 4.0
    guidance_2: float = 3.0
    max_guidance: float | None = None
    shift: float | None = None


def compute_dtype(layout: str) -> Any:
    """The dtype a runner's modules are BUILT at for one layout contract.

    Every Wan lane traces in bf16 — the fp8 that ``cozy.fp8-rowwise@1`` names
    is a LOAD-TIME rung applied to bf16-traced classes (``models/w8a8.py``),
    not a second traced graph — so this is one branch today and a seam
    tomorrow, spelled the same way ``sdxl_serve`` and ``flux2_klein_4b_serve``
    spell it.
    """

    import torch

    return torch.bfloat16 if layout in ("bf16", "plain.bf16@1") else torch.float32


def packed_shape(width: int, height: int) -> int:
    """``(width, height)`` as ONE positive integer bucket coordinate.

    The pgw#1346 K9 workaround, reused verbatim from B2's ``sdxl``: a runner's
    variants are the CROSS PRODUCT of its axes, so ``width`` x ``height`` would
    demand a traced class at every compiled graph of a grid whose diagonal is the only
    part anyone serves. ``1280x720`` reads as ``12800720`` in the generated
    ``Literal`` and ``720x1280`` as ``7201280`` — two coordinates, not four.

    **Video does not make this worse, and that is the measured B4 answer to
    K9's video question.** ``frames`` is a SEPARATE declared axis and it
    composes honestly: Wan's served set is one frame count per family (81 on
    A14B, 121 on TI2V-5B), so ``shape`` x ``frames`` is 2x1 and 1x1
    respectively — TOTAL, with no phantom compiled graph. The packed trick is still owed
    a real fix (a tuple-valued bucket axis), but it is owed for the same two
    numbers it was owed for in B2, not for three.
    """

    if not (0 < width < 10000 and 0 < height < 10000):
        raise ValueError(
            f"packed_shape encodes each side in four decimal digits; "
            f"got {width}x{height}"
        )
    return width * 10000 + height


def unpack_shape(packed: int) -> tuple[int, int]:
    """The inverse of :func:`packed_shape`, so the encoding is never guessed."""

    return divmod(packed, 10000)


def latent_grid(
    width: int, height: int, frames: int, *, spatial: int, temporal: int
) -> tuple[int, int, int]:
    """``(F_lat, H_lat, W_lat)`` the VAE hands the DiT.

    ``wan_2_2/main.py:400-401``, reproduced rather than approximated. The
    temporal term is the one that surprises: ``(frames - 1) // temporal + 1``,
    not ``frames // temporal`` — Wan's VAE keeps the first frame whole and
    compresses the rest, so 81 frames at 4x is 21 latent frames and not 20.
    """

    if frames < 1:
        raise ValueError(f"a clip needs at least one frame, got {frames}")
    return ((frames - 1) // temporal + 1, height // spatial, width // spatial)


def denoiser_tokens(
    width: int, height: int, frames: int, *, spatial: int, temporal: int
) -> int:
    """How many patch tokens one clip presents to the DiT.

    ``F_lat * ceil(H_lat/2) * ceil(W_lat/2)`` — ``wan_2_2/main.py:517``, whose
    own comment records that this relation is not expressible in the ``Dim``
    API and so has to be resolved by the caller. TI2V-5B's declared
    ``N_tok`` of 27280 is exactly this at 1280x704x121.

    It is NOT the bucket axis: A14B's timestep is one scalar per batch and its
    graph is a function of the ``(F_lat, H_lat, W_lat)`` volume rather than of
    the token count, so two transposed shapes with equal token counts are
    still two graphs. That is the fact ``packed_shape`` exists for.
    """

    f_lat, h_lat, w_lat = latent_grid(
        width, height, frames, spatial=spatial, temporal=temporal
    )
    return f_lat * -(-h_lat // PATCH_H) * -(-w_lat // PATCH_W)


__all__ = [
    "A14B_SPATIAL",
    "A14B_TEMPORAL",
    "NUM_TRAIN_TIMESTEPS",
    "PATCH_H",
    "PATCH_T",
    "PATCH_W",
    "TEXT_DIM",
    "TEXT_TOKENS",
    "TI2V_SPATIAL",
    "TI2V_TEMPORAL",
    "Wan22Tuned",
    "compute_dtype",
    "denoiser_tokens",
    "latent_grid",
    "packed_shape",
    "unpack_shape",
]
