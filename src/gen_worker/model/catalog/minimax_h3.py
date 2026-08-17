"""MiniMax-H3, declared — and this file IS pgw#1346's fork **F2**, resolved.

F2, as the W2 batch plan wrote it: *"If ``MiniMaxH3Scheduler`` +
``MiniMaxH3LoopSchedulerStep`` do not project onto a ``Scheduler`` block, H3
either gets an eager ``Model`` (losing the compiled backing it has today) or the
``Scheduler`` vocabulary grows a 'host-implemented' escape."*

**Both horns of that fork rest on premises this lane measured to be false, and
the third option — the one the fork did not offer — is the right one.**

**1. There is no compiled backing to lose.** The endpoint says so itself:
``minimax-h3/src/minimax_h3/main.py:2960`` — *"THIS FAMILY IS NOT AOT-DECLARABLE
(ie#652: 737k static classes)"* — and it is the only video endpoint in the fleet
with **no ``aot/`` directory and no ``*.mint.json``** (wan-2.2, ltx-video-2.3,
qwen-image, sdxl, z-image and both klein dirs all have one). What H3 runs today
is ``torch.compile`` over a single dynamic sequence axis, and the eager tier
keeps that: ``ModelSpec`` withholds a MINTED cell, not dynamo.

**2. The ``Scheduler`` vocabulary does not need a host-implemented escape,
because it already IS one.** ``recipe_v1``: *"The scheduler block is a name plus
finite JSON scalars. torchcg validates its shape and never interprets it: the
host implements the named scheduler."* A bespoke ``MiniMaxH3Scheduler`` is the
case the block was designed for, and its whole configuration surface is ONE
positive float (``shift``, ``scheduling_minimax_h3.py:73-84``) — a finite JSON
scalar. Nothing about it is unrepresentable.

**3. The loop is STAGED, not host — and declaring ``host`` would be the lie.**
``loop.kind: host`` exists for iteration *"until the model says stop"*, and
``recipe_v1`` refuses a repeat count under it precisely so a fabricated bound
cannot be read as a real one. H3's iteration is not data-dependent in either of
its two loops:

* the denoise loop runs ``len(scheduler.timesteps)`` steps, resolved from a
  payload ``Literal[20, 30, 50]``;
* the ``long_video`` loop runs one chunk per requested slot, ``1..MAX_SLOTS``
  with ``MAX_SLOTS = 24`` declared on the wire.

Its per-step body is two blocks in a fixed order — ``["denoiser", "update"]``
(``denoise.py:277-278``) — of which only the first is a graph: ``update`` is the
scheduler step, which ``recipe_v1`` puts in the host by definition. Video and
audio are ONE fused transformer call, not two alternating runners
(``denoise.py:124-132`` returns ``noise_pred, audio_noise_pred`` from one
forward). So the composition is ``Stage("denoiser", repeat=...)`` — the plainest
staged loop there is.

**THE VERDICT: F2 dissolves. H3 is an eager ``ModelSpec`` for reasons that have
nothing to do with its loop, and the loop question has a clean answer that
costs nothing.** Three facts, none of them the scheduler and none of them the
loop, are what keep it off the graph tier — and each is checkable rather than
argued:

(a) **There is no architecture source in this repo.** H3's transformer, both
    VAEs and its scheduler are VENDORED in the endpoint at a pinned diffusers
    SHA; ``diffusers`` 0.39 (this repo's own pin) exports no MiniMax class at
    all. ``Runner.build`` has nothing to construct, so there is no fake-tensor
    export, so there are no bindings. This is the same shape as B1's
    ``Flux2Klein9b`` finding ("its class-level arch config has no source here")
    and it is the honest reason, stated rather than dressed up as a loop
    problem.
(b) **The scheduler PARAMETERS are checkpoint-level.** ``video_shift`` and
    ``audio_shift`` are fields of the family's stamped recipe
    (``@family("minimax-h3")``, ``main.py:376-378``) that the catalog may
    restamp, and ``recipe_v1`` G11 makes the recipe *structurally*
    checkpoint-free. Writing 12.0 and 3.0 into a class-level ``Scheduler``
    block would put a stamped value inside the family digest. That is
    pgw#1346 **K10** — the sampler/schedule being a tuned value against a
    single-valued class-level block — showing up a third time (B2 found it on
    sdxl/sd15's payload enum, B4 finds it on wan's per-checkpoint solver, and
    here on the scheduler's own parameters).
(c) **A compiled H3 denoiser would not be checkpoint-free either.** te#171's
    AdaLN cache replaces 26.02 GB of ``adaln_proj`` weights with a table
    addressed by the RESOLVED SCHEDULE — ``CacheKey(steps, video_shift,
    audio_shift)`` — and a forward pre-hook rewrites ``timestep`` and
    ``timestep_indices`` on every step so the block's
    ``adaln_indices = timestep_indices * 3 + token_tags`` gathers the right
    rows. Bake that and the graph's identity depends on a request's step count
    and a checkpoint's shifts, which §4.27's checkpoint-free class layer
    forbids. The endpoint already treats it that way: ``bind_schedule`` runs
    per request.

**What this declaration therefore carries** is everything ``ModelSpec`` can
honestly hold: the name, the tuned vocabulary, and the ie#740 layout axes
preserved BY VALUE — including the th#1754 ``vram78g`` floor, which is a
production-incident number and is the whole reason K1 exists.
"""

from __future__ import annotations

from typing import Final

from ..spec import ModelSpec, TunedValues


#: MiniMax-H3 conditions on ``hidden_states[50]`` of a 62-layer Qwen3-VL stack,
#: not on its last layer. A CLASS fact: it is a module constant mirrored on the
#: pipeline as a no-argument property (``modular_pipeline.py:258-266``), applied
#: once at ``setup()``, idempotent, and request-independent.
#:
#: Declared here because the trim it licenses — drop layers 50.., replace the
#: final norm with an identity, drop the LM head — CHANGES THE CONDITIONER'S KEY
#: SET, and a key set that changes without being declared is exactly the te#185
#: failure (a 71 GB fetch onto four rented H100s that died as an md5 miss inside
#: a detection helper). The trimmed encoder is 51 kept layers with no head and
#: does not resolve against the untrimmed hash.
CONDITIONING_LAYER: Final = 50

#: Rows per distinct timestep in one ``adaln_proj`` output. The DiT computes
#: ``adaln_indices = timestep_indices * MODALITY_NUM + token_tags``, so the
#: modulation table is addressed by (step-local timestep index, modality tag)
#: and its height is a multiple of three. This is the "timestep-indexed adaLN"
#: the batch plan names, stated as the arithmetic rather than as prose.
MODALITY_NUM: Final = 3

#: The DiT's geometry, measured off the published safetensors headers (638
#: keys, 66,280,430,080 B) rather than copied from a config. Kept because it is
#: what makes the residency arithmetic auditable: ``adaln_proj`` alone is
#: 26,020,915,200 B — **39.3%** of the DiT — and its input dim is
#: ``TIME_EMBED_DIM`` exactly, which is the entire premise of te#171's cache.
HIDDEN_SIZE: Final = 5376
NUM_LAYERS: Final = 50
NUM_REFINER_LAYERS: Final = 2
TIME_EMBED_DIM: Final = 2688
#: 56 heads x 128 = 7168, deliberately WIDER than ``HIDDEN_SIZE``.
NUM_ATTENTION_HEADS: Final = 56
ATTENTION_HEAD_DIM: Final = 128

#: The video VAE's compression, from its own config's factor products:
#: ``spatial_downsample_factors`` -> 16, ``temporal_downsample_factors`` -> 4.
#: The temporal geometry is stated by the class as ``17n + 5`` pixel frames ->
#: ``5n + 2`` latent frames.
VAE_SPATIAL: Final = 16
VAE_TEMPORAL: Final = 4

#: The audio VAE is waveform-in / waveform-out with no mel front end and no
#: separate vocoder — its decoder IS BigVGAN. 32 kHz over a hop of
#: ``prod(encoder_rates) = 800`` gives 40 audio latents per second.
AUDIO_SAMPLE_RATE: Final = 32000
AUDIO_LATENTS_PER_SECOND: Final = 40.0

#: The two spellings of H3's attention projections, and the reason
#: ``models/key_topology.py`` exists. The upstream/native tree fuses q, k and v
#: HEAD-INTERLEAVED into ``blocks.N.attn.qkv_proj`` (535 keys); every artifact
#: the fleet holds is the diffusers repackaging with split
#: ``transformer_blocks.N.attn.to_q/to_k/to_v`` (638 keys). They share EXACTLY
#: ONE key, ``token_refiner.final_norm.weight``.
#:
#: Recorded on the declaration because the naive ``torch.cat([Wq, Wk, Wv])``
#: fusion does not crash — it hands head 0 the q-slices of heads 0, 1 and 2 as
#: its q, k and v, and measures ~90% error against the serve path while an
#: adapter trained on it converges beautifully and serves garbage.
NATIVE_KEY_COUNT: Final = 535
DIFFUSERS_KEY_COUNT: Final = 638


class MiniMaxH3Tuned(TunedValues, frozen=True):
    """H3's stamped recipe, migrated BY VALUE from ``@family("minimax-h3")``.

    Two fields, and the absences are as load-bearing as the presences: H3-Base
    is GUIDANCE-DISTILLED, so there is no guider, no ``negative_prompt``, no
    ``guidance_scale``, and every step runs exactly one forward. Declaring any
    of them would name a knob the model does not have.

    The two shifts are checkpoint facts because H3 steps **two** schedulers
    inside one transformer call — ``shift=12.0`` on the video ladder,
    ``shift=3.0`` on the audio one. They are also, jointly with the step count,
    the identity of te#171's AdaLN cache: ``CacheKey(steps, video_shift,
    audio_shift)``, because *"the two shifts are what make a schedule of N steps
    a DIFFERENT schedule"*.

    That is why they are HERE and not in a ``Scheduler`` block: a stamped value
    inside a class-level, checkpoint-free document would be a G11 violation
    (see this module's docstring, point (b)).
    """

    video_shift: float = 12.0
    audio_shift: float = 3.0


#: MiniMax-H3 — the eager tier, PERMANENTLY as far as this lane can see, and
#: not as a waiting room.
#:
#: ``layouts`` and ``layout_requirements`` are the retired ``Slot``'s, by value
#: and with their reasons intact (``minimax_h3/main.py:3004-3060``). The pairing
#: is not two rungs of one component: ``hf.fp8-blockwise@1`` is the te#172 fp8
#: TEXT ENCODER (transformers' FineGrainedFP8, 128x128 block scales) and
#: ``plain.bf16@1`` is the DiT, whose own fp8 is a RUNTIME torchao quantize off
#: bf16 bytes and therefore not a layout on disk. The ``vram78g`` floor is
#: th#1754's and guards the bf16 lane alone.
MINIMAX_H3: Final = ModelSpec(
    name="minimax_h3",
    tuned=MiniMaxH3Tuned,
    layouts={"*": ("hf.fp8-blockwise@1", "plain.bf16@1")},
    layout_requirements={"plain.bf16@1": "vram78g"},
)

__all__ = [
    "ATTENTION_HEAD_DIM",
    "AUDIO_LATENTS_PER_SECOND",
    "AUDIO_SAMPLE_RATE",
    "CONDITIONING_LAYER",
    "DIFFUSERS_KEY_COUNT",
    "HIDDEN_SIZE",
    "MINIMAX_H3",
    "MODALITY_NUM",
    "MiniMaxH3Tuned",
    "NATIVE_KEY_COUNT",
    "NUM_ATTENTION_HEADS",
    "NUM_LAYERS",
    "NUM_REFINER_LAYERS",
    "TIME_EMBED_DIM",
    "VAE_SPATIAL",
    "VAE_TEMPORAL",
]
