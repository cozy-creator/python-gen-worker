"""Stable Diffusion XL, declared. The second catalog entry, and the contrast.

SDXL is here beside FLUX.1-dev on purpose: the two exercise opposite ends of
the declaration vocabulary from ONE authoring contract, which is the claim
pgw#1326's "One SDK shape" makes and this is the evidence for it.

* Flux is a DiT: packed tokens, a flow-match scheduler, a guidance embedding.
* SDXL is a U-Net: 4-channel latents, a conv-heavy graph, a micro-conditioning
  block (``text_embeds`` + ``time_ids``) that arrives as a MAPPING parameter —
  so the generated binding's ``added_cond_kwargs`` is typed
  ``Mapping[str, Tensor]`` from the leaves' own paths (torchcg G4), with
  nothing per-family in the generator to make that happen.

The conv-heavy shape is also why ``shape`` is a bucket rather than a symbolic
axis: DESIGN-RULINGS §4.30 forbids trading a conv family's served efficiency
for compile speed, and a symbolic latent H/W turns off inductor's channels-last
layout optimisation (measured at +7.2% on sdxl). Buckets keep each class
statically specialised, and the closed ``Literal`` keeps the set exhaustive.

Same rules as the Flux entry: diffusers is imported only inside ``build`` and
the config below is architecture (checkpoint-free). pgw#1346 B2 added the two
text towers and the ``euler_discrete`` block, so this entry is now the same
FOUR-runner shape the Flux entry is: nothing in SDXL's composition is left
riding an eager model at serve time.

**This module is MINT-SIDE** (pgw#1331). The tuned schemas and the latent
arithmetic the request path reads live in
:mod:`gen_worker.model.catalog.sdxl_serve`, which imports nothing above
``torch``; this half reads from there, never the reverse.
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
    Stage,
)
from .sd_samplers import sd_schedulers
from .sdxl_serve import (
    CFG_BATCH,
    CLIP_G_WIDTH,
    CLIP_L_WIDTH,
    CROSS_ATTENTION_WIDTH,
    LATENT_CHANNELS,
    SHAPE_BUCKETS,
    TEXT_TOKENS,
    TIME_IDS,
    SdxlLoraTuned,
    SdxlTuned,
    compute_dtype,
    latent_shape,
)

#: SDXL's U-Net architecture. Every SDXL fine-tune shares it, which is exactly
#: why one compiled graph serves all of them.
UNET: Final[Mapping[str, Any]] = {
    "sample_size": 128,
    "in_channels": 4,
    "out_channels": 4,
    "center_input_sample": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    "block_out_channels": (320, 640, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 2048,
    "transformer_layers_per_block": (1, 2, 10),
    "attention_head_dim": (5, 10, 20),
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "norm_num_groups": 32,
}

#: SDXL's VAE architecture (4 latent channels, /8 spatial).
VAE: Final[Mapping[str, Any]] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "down_block_types": (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ),
    "up_block_types": (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ),
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.13025,
}


def _denoiser(layout: str) -> Any:
    import torch
    from diffusers import UNet2DConditionModel
    from torch import nn

    torch.set_default_dtype(compute_dtype(layout))
    # Bound as a value, not called through the imported name: diffusers ships
    # no complete stubs, and this keeps the untyped boundary at ONE line per
    # build instead of a `type: ignore` on every attribute of the result.
    unet: Any = UNet2DConditionModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = unet(**dict(UNET))

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            added_cond_kwargs: Any,
        ) -> Any:
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    rows, cols = latent_shape(int(bucket["shape"]))
    return CallExample(
        params=("sample", "timestep", "encoder_hidden_states", "added_cond_kwargs"),
        kwargs={
            "sample": torch.zeros(CFG_BATCH, LATENT_CHANNELS, rows, cols, dtype=dtype),
            # float32 deliberately: euler-family samplers present a float32
            # scalar timestep and ddim/dpmpp-family samplers present int64. The
            # integer -> float32 recast is `ingress_selection_v1`'s ONE
            # permitted dtype normalization, and it is value-preserving on a
            # rank-0 feed — so declaring float32 admits both without the
            # sampler becoming a compile axis.
            "timestep": torch.zeros((), dtype=torch.float32),
            "encoder_hidden_states": torch.zeros(
                CFG_BATCH, TEXT_TOKENS, CROSS_ATTENTION_WIDTH, dtype=dtype
            ),
            "added_cond_kwargs": {
                "text_embeds": torch.zeros(CFG_BATCH, 1280, dtype=dtype),
                "time_ids": torch.zeros(CFG_BATCH, TIME_IDS, dtype=dtype),
            },
        },
    )


def _decoder(layout: str) -> Any:
    import torch
    from diffusers import AutoencoderKL
    from torch import nn

    torch.set_default_dtype(compute_dtype(layout))
    autoencoder: Any = AutoencoderKL

    class _Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae = autoencoder(**dict(VAE))

        def forward(self, latents: Any) -> Any:
            return self.vae.decode(latents, return_dict=False)[0]

    return _Decoder().eval()


def _decoder_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    rows, cols = latent_shape(int(bucket["shape"]))
    return CallExample(
        params=("latents",),
        kwargs={"latents": torch.zeros(1, LATENT_CHANNELS, rows, cols, dtype=dtype)},
    )


#: CLIP-L's text tower — SDXL's ``text_encoder``. SDXL reads its PENULTIMATE
#: hidden state, not its last: the final layer's states are tuned for CLIP's
#: contrastive objective and the layer below carries more of the prompt.
CLIP_L_TEXT: Final[Mapping[str, Any]] = {
    "vocab_size": 49408,
    "hidden_size": CLIP_L_WIDTH,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": TEXT_TOKENS,
    "hidden_act": "quick_gelu",
}

#: OpenCLIP bigG's text tower — SDXL's ``text_encoder_2``. It contributes BOTH
#: halves of SDXL's conditioning: its penultimate states are concatenated with
#: CLIP-L's to make the 2048-wide cross-attention input, and its projected
#: pooled vector is the ``text_embeds`` half of the micro-conditioning block.
#: ``gelu``, not ``quick_gelu``: OpenCLIP and OpenAI CLIP differ here, and the
#: substitution is silent and wrong rather than loud and wrong.
CLIP_G_TEXT: Final[Mapping[str, Any]] = {
    "vocab_size": 49408,
    "hidden_size": CLIP_G_WIDTH,
    "intermediate_size": 5120,
    "num_hidden_layers": 32,
    "num_attention_heads": 20,
    "max_position_embeddings": TEXT_TOKENS,
    "hidden_act": "gelu",
    "projection_dim": CLIP_G_WIDTH,
}

#: SDXL's trained noise schedule, DECLARED rather than defaulted. Every value
#: here is stated because :class:`~gen_worker.model.scheduler.EulerDiscrete`
#: defaults to diffusers' CLASS defaults (linear betas over [1e-4, 2e-2],
#: linspace spacing) and Stable Diffusion uses none of them — an omitted
#: parameter would resolve to a schedule this family was never trained on.
#: ``steps_offset=1`` and ``timestep_spacing="leading"`` are what every SDXL
#: checkpoint's own ``scheduler/scheduler_config.json`` carries.
TRAINED: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1,
}

#: The trained schedule under the ``euler`` sampler — i.e. with the one
#: parameter that is ``EulerDiscrete``'s and not the checkpoint's. Kept as its
#: own name because it is the block every measurement in
#: ``tests/test_sd_stage1_pgw1346.py`` differences against.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    **TRAINED,
    "final_sigmas_type": "zero",
}

#: The scheduler SET, keyed by the sampler a checkpoint is stamped with
#: (pgw#1346 K10). These are the three names in ``SdxlSampler`` that map onto
#: a scheduler this SDK implements; the other three (``dpmpp_2m_karras``,
#: ``dpmpp_2m_sde_karras``, ``lcm``) are MULTISTEP solvers owed to B3/B4 and
#: are refused BY NAME rather than served under a neighbouring schedule.
#:
#: ``euler_a`` is the one that matters most: it is ``SdxlTuned.scheduler``'s
#: DEFAULT, so before K10 the family's single declared ``euler_discrete``
#: block was its trained schedule and not the one most requests asked for.
SCHEDULERS: Final = sd_schedulers(
    TRAINED,
    ("ddim_trailing", "dpmpp_2m_karras", "dpmpp_2m_sde_karras", "euler_a", "euler_trailing"),
)


def _arm_hidden_state_capture(model: Any) -> Any:
    """Install transformers' hidden-state capture hooks BEFORE the trace.

    ``output_hidden_states=True`` makes a transformers forward call
    ``maybe_install_capturing_hooks``, which takes a ``threading.Lock`` — an
    unsupported context manager under ``torch.export(strict=True)``, and the
    export dies with "Unsupported context manager" naming nothing useful.
    Installing them here flips the ``_output_capturing_hooks_installed`` flag
    the traced forward early-returns on, so the lock is never reached.

    Both SDXL text towers need the PENULTIMATE hidden state and bigG needs its
    pooled projection from the SAME pass, so reading only ``last_hidden_state``
    is not an option and neither is running the 694M tower twice.
    """

    from transformers.utils.output_capturing import maybe_install_capturing_hooks

    maybe_install_capturing_hooks(model)
    return model


def _clip_l(layout: str) -> Any:
    import torch
    from torch import nn
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.set_default_dtype(compute_dtype(layout))
    config: Any = CLIPTextConfig
    text_model: Any = CLIPTextModel

    class _ClipL(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = _arm_hidden_state_capture(
                text_model(config(**dict(CLIP_L_TEXT)))
            )

        def forward(self, input_ids: Any) -> Any:
            out = self.text_encoder(input_ids=input_ids, output_hidden_states=True)
            return out.hidden_states[-2]

    return _ClipL().eval()


def _clip_g(layout: str) -> Any:
    """The bigG tower, wrapped down to the TWO outputs SDXL consumes.

    Two, not one, and they leave through one call rather than two: the pooled
    projection and the penultimate states come from the same forward pass, so
    splitting them into two runners would run a 694M-parameter tower twice per
    request to save nothing.
    """

    import torch
    from torch import nn
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection

    torch.set_default_dtype(compute_dtype(layout))
    config: Any = CLIPTextConfig
    text_model: Any = CLIPTextModelWithProjection

    class _ClipG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = _arm_hidden_state_capture(
                text_model(config(**dict(CLIP_G_TEXT)))
            )

        def forward(self, input_ids: Any) -> Any:
            out = self.text_encoder(input_ids=input_ids, output_hidden_states=True)
            return out.hidden_states[-2], out.text_embeds

    return _ClipG().eval()


def _clip_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    del bucket, layout  # token IDs are integers; the layout decides no dtype here
    return CallExample(
        params=("input_ids",),
        kwargs={"input_ids": torch.zeros(1, TEXT_TOKENS, dtype=torch.long)},
    )


#: Stable Diffusion XL. Four runners — the whole pipeline, as FLUX.1-dev is.
#:
#: The shape axis is PACKED (``sdxl_serve.pack_shape``) because SDXL's nine
#: trained buckets are a SET of (width, height) pairs and not a product of two
#: independent axes; see that function for why the obvious two-axis spelling
#: would mint 81 classes to serve 9. The text towers declare no axis: both pad
#: to CLIP's 77 positions by construction, so bucketing them would generate
#: classes nothing selects.
SDXL: Final = GraphModelSpec(
    name="sdxl",
    tuned=SdxlTuned,
    lora_tuned=SdxlLoraTuned,
    # ie#740's floors, migrated BY VALUE from the sdxl endpoint's `Slot` onto
    # the declaration W1b-1 gave them a home on. Not re-derived and not
    # rounded — both numbers are production incidents:
    #
    #   `vram12g` is the value ie#704's stopgap block itself calls "the honest
    #   SERVING floor" and names as its revert target, after `vram_gb = 20`
    #   over-constrained SERVING to fix MINT PLACEMENT through one overloaded
    #   scalar. Measured on both sides: 9.3 GiB resident across six cards
    #   (ie#706), and a 16 GiB A4000 has served this endpoint. The MINT half
    #   is owed to pgw#1208 and is deliberately NOT restated here — putting it
    #   back would keep the overload alive under a new name.
    #
    #   `sm89+` is the HONEST floor for the rowwise handle, not the fastest
    #   one: the rowwise GEMM wants sm90, but the contract is still DECODABLE
    #   at sm89 through the per-tensor path, and what a requirement guards is
    #   decodability.
    #
    # Keyed by COMPONENT PATH (`"*"`), which is why K4 kept it off
    # `Runner.layouts`: SDXL is two shape-bearing runners over a four-component
    # tree, so a per-text-encoder demand has nowhere else to go.
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={"cozy.fp8-rowwise@1": "sm89+", "plain.bf16@1": "vram12g"},
    buckets=(Bucket("shape", SHAPE_BUCKETS),),
    runners=(
        Runner("clip_g", build=_clip_g, example=_clip_example),
        Runner("clip_l", build=_clip_l, example=_clip_example),
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("shape",)),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("shape",)),
    ),
    loop=Loop(
        stages=(
            Stage("clip_l"),
            Stage("clip_g"),
            Stage("denoiser", repeat="steps"),
            Stage("decoder"),
        )
    ),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    schedulers=SCHEDULERS,
)

__all__ = [
    "CLIP_G_TEXT",
    "CLIP_L_TEXT",
    "SCHEDULER",
    "SCHEDULERS",
    "TRAINED",
    "SDXL",
    "UNET",
    "VAE",
]
