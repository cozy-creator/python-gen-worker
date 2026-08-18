"""LTX-Video 2.3, declared. ONE joint audio-video DiT, TWO counted stages.

Two things about this family are unlike anything already in the catalog, and
both are expressible without extending the vocabulary:

**1. One runner, two stages, two step counts.** The production 1080p+ recipe is
not a single denoise: 8 distilled steps at half resolution, ONE 2x latent
upsample, then 3 refinement steps at full resolution on the tail of the same
sigma ladder. That is ``Stage("denoiser", repeat="stage1_steps")`` followed by
``Stage("denoiser", repeat="stage2_steps")`` — one runner named twice, two
declared parameters. ``recipe_v1``'s loop is an ORDERED LIST of stages, not a
set, so a runner appearing twice is a composition it already describes; the
declaration says nothing about the bucket each stage runs at because a stage's
inputs are host code.

The upsample between them is deliberately NOT a stage here: it is a separate
pipeline the endpoint binds in its own slot (``LTX2LatentUpsamplePipeline``),
so it is a K5 sibling model and not a runner of this family. The endpoint draws
the same line — ``Compile(targets=("transformer",))``, with its own reason
recorded: *"the upsampler/VAE/vocoder are small and stay eager."*

**2. Audio is not a second model; it is a second token stream through the same
weights.** The 22B DiT denoises video AND audio latents jointly and returns two
tensors from one forward. There is no audio DiT, and the vocoder is a component
of the same pipeline. So the family has one runner, and ``audio_tokens`` is a
BUCKET AXIS beside ``video_tokens`` rather than a second declaration.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**

----

**THE K9-VIDEO DISPOSITION, which this family is the sharp case for.**

pgw#1346 K9 (B2) recorded that ``Bucket`` expresses a shape PRODUCT and not a
shape SET, and worked around it with one packed decimal axis. B4 was asked
whether video's third coordinate breaks that. The answer has two halves and
they point opposite ways:

* **Wan does not need more than K9 already gives.** ``frames`` is a genuine
  product axis there (one trained frame grid per family), so ``shape`` x
  ``frames`` is total with no phantom compiled graph. See ``wan22_serve.packed_shape``.
* **LTX is where the product genuinely fails, and it fails on axes that are
  not spatial at all.** The endpoint's own mint declaration says it outright:
  the graph is a function of the TOKEN COUNTS alone (``num_frames``,
  ``height``, ``width`` and ``fps`` are dead inside the forward because
  ``video_coords`` is precomputed, so 704x1280 and 1280x704 are ONE graph),
  and the real axes are ``T_v`` x ``T_a`` x ``T_at`` x ``T_kf``. Those are
  independent in principle and SPARSE in practice: the endpoint enumerates
  **82 graph classes on h100 and 115 on b200** over **20 / 26** distinct token
  counts and **28 / 39** distinct ``(T_v, T_a)`` pairs — so the cross product
  of the axes' value sets is roughly 3x the set anyone serves.

**So K9-video is REAL and it is filed rather than contorted.** No packed
encoding helps here: packing ``(T_v, T_a)`` into one integer would make a
9-digit literal that hides a sparse set behind a dense-looking axis, which is
worse than B2's ``13440768`` (that one packed two numbers a reader can still
see, over a set that was genuinely a product once split). The fix K9 already
names — a tuple-valued bucket axis, i.e. an axis whose values are COORDINATES
rather than scalars — is the same fix this needs, and it is the one that would
let a family declare a SET.

What is declared below is therefore the **committed mint coordinates and their
two-stage partners**, which is a set that IS a product: the 4K/241-frame lane's
stage-1 and stage-2 token counts, crossed with the two audio lengths 241 frames
can carry (10.04 s at 24 fps and 5.02 s at 48 fps). Four variants, total by
construction, and two of them are byte-for-byte the rows
``aot/transformer-b200-tv261120-ta{126,251}-tat1.mint.json`` already declare.
Every other preset row is owed to the tuple-valued axis, and is enumerated
above rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..spec import (
    Bucket,
    CallExample,
    GraphModelSpec,
    Loop,
    Parameter,
    Runner,
    Scheduler,
    Stage,
)
from .ltx23_serve import (
    AUDIO_CROSS_ATTENTION_DIM,
    AUDIO_LATENT_CHANNELS,
    CROSS_ATTENTION_DIM,
    LATENT_CHANNELS,
    NUM_TRAIN_TIMESTEPS,
    TEXT_TOKENS,
    Ltx23Tuned,
    compute_dtype,
)

#: LTX-2.3's transformer architecture, from the published
#: ``Lightricks/LTX-2.3``-lineage ``transformer/config.json``
#: (``diffusers/LTX-2.3-Distilled-Diffusers@432e0d3c``). Class-level truth: no
#: weight, no checkpoint ref, no tuned value appears here or can.
#:
#: The audio half of the block is what makes this one model rather than two:
#: ``audio_in_channels`` / ``audio_num_attention_heads`` /
#: ``audio_cross_attention_dim`` are the SAME module's parameters, not a
#: sibling network's.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "in_channels": LATENT_CHANNELS,
    "out_channels": LATENT_CHANNELS,
    "patch_size": 1,
    "patch_size_t": 1,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "cross_attention_dim": CROSS_ATTENTION_DIM,
    "vae_scale_factors": (8, 32, 32),
    "pos_embed_max_pos": 20,
    "base_height": 2048,
    "base_width": 2048,
    "gated_attn": True,
    "cross_attn_mod": True,
    "audio_in_channels": AUDIO_LATENT_CHANNELS,
    "audio_out_channels": AUDIO_LATENT_CHANNELS,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": AUDIO_CROSS_ATTENTION_DIM,
    "audio_scale_factor": 4,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_hop_length": 160,
    "audio_gated_attn": True,
    "audio_cross_attn_mod": True,
    "num_layers": 48,
    "activation_fn": "gelu-approximate",
    "qk_norm": "rms_norm_across_heads",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "caption_channels": 3840,
    "attention_bias": True,
    "attention_out_bias": True,
    "rope_theta": 10000.0,
    "rope_double_precision": True,
    "causal_offset": 1,
    "timestep_scale_multiplier": NUM_TRAIN_TIMESTEPS,
    "cross_attn_timestep_scale_multiplier": 1000,
    "rope_type": "split",
    "use_prompt_embeddings": False,
    "perturbed_attn": True,
}

#: The declared video-token coordinates: the two-stage 4K/241-frame lane.
#:
#: 261120 is the full-resolution row the fleet has already committed a mint
#: request for (3840x2176 at 241 frames with a ``last_frame`` keyframe:
#: ``31 * 68 * 120 + 68 * 120``). 65280 is its stage-1 partner at half
#: resolution (1920x1088: ``31 * 34 * 60 + 34 * 60``) — declared because the
#: production recipe runs BOTH, and a family whose loop states two stages while
#: its buckets cover one of them describes a pipeline nobody runs.
VIDEO_TOKENS: Final = (65280, 261120)

#: The two audio lengths 241 frames can carry. ``T_a`` is an axis a
#: ``(w, h, frames)`` row CANNOT express — the endpoint's own note records the
#: correction: its comment "frame_rate does not change the DiT graph; only
#: num_frames does" is FALSE for the audio stream, because
#: ``T_a = round(frames / fps * 25)`` reads the frame RATE.
AUDIO_TOKENS: Final = (126, 251)

#: Audio-timestep tokens. 1 on generate / extend / a2v — the whole audio stream
#: shares one timestep — and ``T_a`` on the edit lane, which is per-token.
#: Only the served value is declared; the edit lane is a fork the endpoint
#: itself records as a separate class set.
AUDIO_TIMESTEP_TOKENS: Final = 1


def _denoiser(layout: str) -> Any:
    """The joint AV DiT, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import LTX2VideoTransformer3DModel
    from torch import nn

    torch.set_default_dtype(compute_dtype(layout))
    transformer: Any = LTX2VideoTransformer3DModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer(**dict(TRANSFORMER))

        def forward(
            self,
            hidden_states: Any,
            audio_hidden_states: Any,
            encoder_hidden_states: Any,
            audio_encoder_hidden_states: Any,
            timestep: Any,
            audio_timestep: Any,
            sigma: Any,
            audio_sigma: Any,
            encoder_attention_mask: Any,
            audio_encoder_attention_mask: Any,
            video_coords: Any,
            audio_coords: Any,
        ) -> Any:
            # Every non-tensor argument is pinned to the value all four served
            # call sites pass, and each pin is a fact the endpoint declares as
            # a Fork rather than a default it happens to inherit:
            #
            # * `isolate_modalities=False` — the text-to-audio path that would
            #   set it is not reachable from any worker function;
            # * `spatio_temporal_guidance_blocks=None` and
            #   `perturbation_mask=None` — the STG path is the ONLY one that
            #   arms LTX's two data-dependent ops
            #   (`torch.all(perturbation_mask == 0)`, a branch on a 0-d tensor
            #   value) and is a genuine export blocker. Declaring it unserved is
            #   what keeps this graph exportable at all;
            # * `use_cross_timestep=False`.
            #
            # `num_frames` / `height` / `width` / `fps` are absent rather than
            # pinned: the pipeline always precomputes `video_coords`, which
            # makes them DEAD inside the forward. That is also why this family's
            # bucket axis is a token count and not a (w, h, frames) row.
            return self.transformer(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                audio_timestep=audio_timestep,
                sigma=sigma,
                audio_sigma=audio_sigma,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=False,
                attention_kwargs=None,
                return_dict=False,
            )

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    t_v = int(bucket["video_tokens"])
    t_a = int(bucket["audio_tokens"])
    return CallExample(
        params=(
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "audio_encoder_hidden_states",
            "timestep",
            "audio_timestep",
            "sigma",
            "audio_sigma",
            "encoder_attention_mask",
            "audio_encoder_attention_mask",
            "video_coords",
            "audio_coords",
        ),
        kwargs={
            "hidden_states": torch.zeros(1, t_v, LATENT_CHANNELS, dtype=dtype),
            "audio_hidden_states": torch.zeros(
                1, t_a, AUDIO_LATENT_CHANNELS, dtype=dtype
            ),
            "encoder_hidden_states": torch.zeros(
                1, TEXT_TOKENS, CROSS_ATTENTION_DIM, dtype=dtype
            ),
            "audio_encoder_hidden_states": torch.zeros(
                1, TEXT_TOKENS, AUDIO_CROSS_ATTENTION_DIM, dtype=dtype
            ),
            # PER-TOKEN on video, ONE on audio: the video stream carries a
            # per-row timestep so conditioning rows can be pinned clean while
            # the rest denoise, and the audio stream shares one.
            "timestep": torch.zeros(1, t_v, dtype=torch.float32),
            "audio_timestep": torch.zeros(
                1, AUDIO_TIMESTEP_TOKENS, dtype=torch.float32
            ),
            "sigma": torch.zeros(1, dtype=torch.float32),
            "audio_sigma": torch.zeros(1, dtype=torch.float32),
            "encoder_attention_mask": torch.ones(1, TEXT_TOKENS, dtype=torch.int64),
            "audio_encoder_attention_mask": torch.ones(
                1, TEXT_TOKENS, dtype=torch.int64
            ),
            # RoPE coordinates, float32 and never the compute dtype. Video is
            # 3-D (t, h, w) and audio 1-D (t), each as a (start, end) pair on
            # the last axis.
            "video_coords": torch.zeros(1, 3, t_v, 2, dtype=torch.float32),
            "audio_coords": torch.zeros(1, 1, t_a, 2, dtype=torch.float32),
        },
    )


#: LTX-Video 2.3.
#:
#: **The scheduler block is one parameter, and the omissions are deliberate.**
#: LTX is step-distilled and supplies its ladder LITERALLY — every served call
#: site passes ``sigmas=``, so diffusers' synthesizing branch
#: (``np.linspace(...) if sigmas is None``) is never taken on this path. The
#: shift constants the pipeline reads therefore never touch a served schedule,
#: and declaring them would put unverified numbers inside the family digest.
#: What DOES belong is ``num_train_timesteps``, because it is the unit the
#: stamped sigmas are read in, and it is the transformer config's own
#: ``timestep_scale_multiplier``. The ladder itself lives on ``tuned``, where a
#: per-checkpoint value belongs — see ``ltx23_serve.schedule_from_sigmas``.
#:
#: The ie#740 floors are preserved BY VALUE from the retired ``Slot``
#: (``ltx_video_23/main.py:3320-3331``), with their reasons: sm89 is the
#: DECODABLE floor for the rowwise fp8 lane, and the 78 GB is tagged in the
#: endpoint as a PRODUCTION INCIDENT — *"two B200s at $6.83/hr rented and
#: refused."*
LTX23: Final = GraphModelSpec(
    name="ltx23",
    tuned=Ltx23Tuned,
    buckets=(
        Bucket("audio_tokens", AUDIO_TOKENS),
        Bucket("video_tokens", VIDEO_TOKENS),
    ),
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram78g",
    },
    runners=(
        Runner(
            "denoiser",
            build=_denoiser,
            example=_denoiser_example,
            axes=("audio_tokens", "video_tokens"),
            # The one component the endpoint compiles: `Compile(targets=
            # ("transformer",))`, with the upsampler / VAE / vocoder staying
            # eager by its own choice. `component` is a SERVING fact (pgw#1346
            # W1b-2) and is not exported, so it moves no digest.
            component="transformer",
        ),
    ),
    loop=Loop(
        stages=(
            Stage("denoiser", repeat="stage1_steps"),
            Stage("denoiser", repeat="stage2_steps"),
        )
    ),
    parameters=(
        # The distilled ladders are 8 and 3 long. The bounds are the LADDER
        # LENGTHS a stamped recipe may carry, not the shipped values: the
        # values are `tuned`, and `len(sigmas)` is the step count.
        Parameter("stage1_steps", minimum=1, maximum=50),
        Parameter("stage2_steps", minimum=1, maximum=50),
    ),
    # A set of ONE (pgw#1346 K10): this family's tuned schema names no sampler
    # because it serves exactly this schedule, so `inst.scheduler()` still
    # takes no argument and still returns the concrete class.
    schedulers={
        "flow_match_euler": Scheduler(
            "flow_match_euler_discrete", {"num_train_timesteps": NUM_TRAIN_TIMESTEPS}
        )
    },
)

__all__ = [
    "AUDIO_TIMESTEP_TOKENS",
    "AUDIO_TOKENS",
    "LTX23",
    "TRANSFORMER",
    "VIDEO_TOKENS",
]
