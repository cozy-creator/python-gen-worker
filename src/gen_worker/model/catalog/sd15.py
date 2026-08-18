"""Stable Diffusion 1.5 and 2.x, declared. TWO families, and why.

pgw#1346's B2 scoping proposed ``Sd15`` with an ``sd2`` INSTANCE, on B1's
family-collapse rule: one model, N instances, split only when the RUNNER SET
differs. The runner set does NOT differ here — both are clip / denoiser /
decoder — and the recommendation is still wrong, because the rule's real
predicate is the GRAPH, and B1's own worked example says so: FLUX.1-schnell got
its own entry over a single boolean (``guidance_embeds``) that changes the
graph. Measured for SD:

===============================  ==============  ==============
fact                             SD1.5           SD2
===============================  ==============  ==============
text tower                       CLIP-L          OpenCLIP-H
``cross_attention_dim``          768             1024
``attention_head_dim``           8 (uniform)     (5, 10, 20, 20)
``use_linear_projection``        False           True
===============================  ==============  ==============

Every cross-attention projection in the U-Net is a different SHAPE, and the
attention head partition differs on top of it. An instance is "weights + tuned
values + a ref label" (``model/runtime.py`` draws that line); these two do not
share a traced class, a compiled graph, or even a weight shape, so calling them
one model would put two incompatible graphs behind one exhaustive ``Literal``
and the fake backing would be the only thing that ever worked.

The sd15 ENDPOINT already knew: its own comment reads *"SD1.5 is 4/768 and SD2
is 4/1024, and both families read this same tuple"*, and it declares ``@family
("sd15")`` and ``@family("sd2")`` as two vocabularies. Two declarations here is
the migration of that fact, not a deviation from it.

**What they DO share is arithmetic**, and that lives once in
:mod:`gen_worker.model.catalog.sd15_serve` — the shape packing, the latent
stride, the CFG arity, the sampler vocabulary. The build helpers below are
parameterised over the two architecture blocks for the same reason: the
difference between these families is four numbers, and writing it as four
numbers is what keeps it reviewable.

This module is MINT-SIDE. diffusers and transformers appear only inside
``build`` callables.
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
from .sd15_serve import (
    CFG_BATCH,
    LATENT_CHANNELS,
    SD2_SHAPE_BUCKETS,
    SD2_TEXT_WIDTH,
    SD15_SHAPE_BUCKETS,
    SD15_TEXT_WIDTH,
    TEXT_TOKENS,
    Sd2LoraTuned,
    Sd2Tuned,
    Sd15LoraTuned,
    Sd15Tuned,
    compute_dtype,
    latent_shape,
)

#: SD1.5's U-Net. Four resolution levels where SDXL has three, and a uniform
#: 8-dim attention head where SDXL partitions (5, 10, 20).
SD15_UNET: Final[Mapping[str, Any]] = {
    "sample_size": 64,
    "in_channels": LATENT_CHANNELS,
    "out_channels": LATENT_CHANNELS,
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": SD15_TEXT_WIDTH,
    "attention_head_dim": 8,
    "use_linear_projection": False,
    "norm_num_groups": 32,
}

#: SD2's U-Net. Same topology, four different numbers — see the module note.
SD2_UNET: Final[Mapping[str, Any]] = {
    **SD15_UNET,
    "sample_size": 96,
    "cross_attention_dim": SD2_TEXT_WIDTH,
    "attention_head_dim": (5, 10, 20, 20),
    "use_linear_projection": True,
}

#: The SD VAE. Structurally SDXL's; only ``scaling_factor`` differs, and that
#: is a weight-side constant the decoder graph never sees.
VAE: Final[Mapping[str, Any]] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": LATENT_CHANNELS,
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
    "sample_size": 512,
    "scaling_factor": 0.18215,
}

#: CLIP-L's text tower — SD1.5's ``text_encoder``. SD1.5 reads its LAST hidden
#: state, where SDXL and SD2 read the penultimate one. Not a detail: the two
#: differ by a final layer norm, and reading the wrong one shifts every prompt.
SD15_TEXT: Final[Mapping[str, Any]] = {
    "vocab_size": 49408,
    "hidden_size": SD15_TEXT_WIDTH,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": TEXT_TOKENS,
    "hidden_act": "quick_gelu",
}

#: OpenCLIP-H's text tower — SD2's ``text_encoder``. SD2 reads the PENULTIMATE
#: state (``clip_skip=1`` is baked into the released pipeline), which is why
#: the wrapper below differs from SD1.5's by one index.
SD2_TEXT: Final[Mapping[str, Any]] = {
    "vocab_size": 49408,
    "hidden_size": SD2_TEXT_WIDTH,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "max_position_embeddings": TEXT_TOKENS,
    "hidden_act": "gelu",
}

#: SD1.5's and SD2's trained noise schedule. Identical between them, and
#: identical to SDXL's — the three share a noise schedule and share nothing
#: else. Stated in full for the reason ``sdxl.SCHEDULER`` is: EulerDiscrete
#: defaults to diffusers' class defaults, which no Stable Diffusion uses.
TRAINED: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 1,
}

#: The trained schedule under the ``euler`` sampler. Kept as its own name
#: because it is the block the B2 measurements difference against.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    **TRAINED,
    "final_sigmas_type": "zero",
}

#: The scheduler SET, keyed by the sampler a checkpoint is stamped with
#: (pgw#1346 K10). Four of the nine names ``Sd15Sampler`` admits map onto a
#: scheduler this SDK implements. The five absent ones (``dpmpp_2m``,
#: ``dpmpp_2m_karras`` — which is `Sd15Tuned`'s DEFAULT — ``dpmpp_2m_sde_karras``,
#: ``unipc``, ``lcm``) are MULTISTEP solvers owed to B3/B4, and a checkpoint
#: stamped with one is refused BY NAME rather than served under a neighbouring
#: schedule. Both SD1.5 and SD2 offer the same four: `Sd2Tuned` shares
#: `Sd15Sampler` and defaults to ``euler_a``, which IS covered.
SCHEDULERS: Final = sd_schedulers(
    TRAINED,
    (
        "ddim",
        "ddim_trailing",
        "dpmpp_2m",
        "dpmpp_2m_karras",
        "dpmpp_2m_sde_karras",
        "euler",
        "euler_a",
        "unipc",
    ),
)


def _make_denoiser(config: Mapping[str, Any]) -> Any:
    def build(layout: str) -> Any:
        import torch
        from diffusers import UNet2DConditionModel
        from torch import nn

        torch.set_default_dtype(compute_dtype(layout))
        # Bound as a value: diffusers ships no complete stubs, and this keeps
        # the untyped boundary at ONE line per build.
        unet: Any = UNet2DConditionModel

        class _Denoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.unet = unet(**dict(config))

            def forward(
                self, sample: Any, timestep: Any, encoder_hidden_states: Any
            ) -> Any:
                return self.unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

        return _Denoiser().eval()

    return build


def _make_denoiser_example(width: int) -> Any:
    def example(bucket: Mapping[str, int], layout: str) -> CallExample:
        import torch

        dtype = compute_dtype(layout)
        rows, cols = latent_shape(int(bucket["shape"]))
        return CallExample(
            params=("sample", "timestep", "encoder_hidden_states"),
            kwargs={
                "sample": torch.zeros(CFG_BATCH, LATENT_CHANNELS, rows, cols, dtype=dtype),
                # float32 deliberately: the euler-family samplers present a
                # float32 scalar timestep and the ddim/dpmpp family present
                # int64. The integer -> float32 recast is
                # `ingress_selection_v1`'s ONE permitted dtype normalization
                # and it is value-preserving on a rank-0 feed, so declaring
                # float32 admits both without the sampler becoming a compile
                # axis. Identical reasoning to the sdxl entry's.
                "timestep": torch.zeros((), dtype=torch.float32),
                "encoder_hidden_states": torch.zeros(
                    CFG_BATCH, TEXT_TOKENS, width, dtype=dtype
                ),
            },
        )

    return example


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


def _make_clip(config: Mapping[str, Any], *, penultimate: bool) -> Any:
    def build(layout: str) -> Any:
        import torch
        from torch import nn
        from transformers import CLIPTextConfig, CLIPTextModel
        # Installed HERE and not lazily inside the traced forward: a
        # transformers call with `output_hidden_states=True` takes a
        # `threading.Lock` on first use, which `torch.export(strict=True)`
        # refuses as an unsupported context manager. Arming the flag before
        # the trace makes that path early-return. Same note as sdxl's.
        from transformers.utils.output_capturing import maybe_install_capturing_hooks

        torch.set_default_dtype(compute_dtype(layout))
        text_config: Any = CLIPTextConfig
        text_model: Any = CLIPTextModel
        index = -2 if penultimate else -1

        class _Clip(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                encoder = text_model(text_config(**dict(config)))
                maybe_install_capturing_hooks(encoder)
                self.text_encoder = encoder

            def forward(self, input_ids: Any) -> Any:
                out = self.text_encoder(input_ids=input_ids, output_hidden_states=True)
                return out.hidden_states[index]

        return _Clip().eval()

    return build


def _clip_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    del bucket, layout  # token IDs are integers; the layout decides no dtype here
    return CallExample(
        params=("input_ids",),
        kwargs={"input_ids": torch.zeros(1, TEXT_TOKENS, dtype=torch.long)},
    )


#: Stable Diffusion 1.5.
SD15: Final = GraphModelSpec(
    name="sd15",
    tuned=Sd15Tuned,
    lora_tuned=Sd15LoraTuned,
    # ie#740's floor, migrated BY VALUE from the sd15 endpoint's `Slot`. Both
    # of its `@endpoint`s declare the same one, and bf16 is the only lane
    # either serves — an SD1.5-class U-Net is ~2 GB in bf16, so 6 covers
    # weights plus activations on the smallest tier.
    layouts={"*": ("plain.bf16@1",)},
    layout_requirements={"plain.bf16@1": "vram6g"},
    buckets=(Bucket("shape", SD15_SHAPE_BUCKETS),),
    runners=(
        Runner(
            "clip",
            build=_make_clip(SD15_TEXT, penultimate=False),
            example=_clip_example,
        ),
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("shape",)),
        Runner(
            "denoiser",
            build=_make_denoiser(SD15_UNET),
            example=_make_denoiser_example(SD15_TEXT_WIDTH),
            axes=("shape",),
        ),
    ),
    loop=Loop(
        stages=(Stage("clip"), Stage("denoiser", repeat="steps"), Stage("decoder"))
    ),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    schedulers=SCHEDULERS,
)

#: Stable Diffusion 2.x, which is also SD-Turbo's architecture.
SD2: Final = GraphModelSpec(
    name="sd2",
    tuned=Sd2Tuned,
    lora_tuned=Sd2LoraTuned,
    # ie#740's floor, migrated BY VALUE from the sd15 endpoint's `Slot`. Both
    # of its `@endpoint`s declare the same one, and bf16 is the only lane
    # either serves — an SD1.5-class U-Net is ~2 GB in bf16, so 6 covers
    # weights plus activations on the smallest tier.
    layouts={"*": ("plain.bf16@1",)},
    layout_requirements={"plain.bf16@1": "vram6g"},
    buckets=(Bucket("shape", SD2_SHAPE_BUCKETS),),
    runners=(
        Runner(
            "clip",
            build=_make_clip(SD2_TEXT, penultimate=True),
            example=_clip_example,
        ),
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("shape",)),
        Runner(
            "denoiser",
            build=_make_denoiser(SD2_UNET),
            example=_make_denoiser_example(SD2_TEXT_WIDTH),
            axes=("shape",),
        ),
    ),
    loop=Loop(
        stages=(Stage("clip"), Stage("denoiser", repeat="steps"), Stage("decoder"))
    ),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    schedulers=SCHEDULERS,
)

__all__ = [
    "SCHEDULER",
    "SCHEDULERS",
    "SD2",
    "SD2_TEXT",
    "SD2_UNET",
    "SD15",
    "SD15_TEXT",
    "SD15_UNET",
    "VAE",
]
