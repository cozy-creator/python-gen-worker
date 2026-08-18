"""Qwen-Image, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule: an endpoint imports
:class:`~gen_worker.model.catalog.QwenImage` — the generated binding beside
this file — and never touches diffusers. This module is where diffusers is
allowed, and only inside the ``build`` callables, which run at MINT time and on
an eager-capable serving pod.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs lives in
:mod:`gen_worker.model.catalog.qwen_image_serve`.

**Checkpoint-free.** The block below is Qwen-Image's architecture, read from
``Qwen/Qwen-Image``'s published ``transformer/config.json``.

**This declares the TEXT-TO-IMAGE arm, and the edit arm is NOT an instance of
it — measured, pgw#1346 B3a.** The W2 batch plan scoped
``QwenImage (+edit instance)``. ``Qwen/Qwen-Image-Edit-2511``'s transformer
config sets ``zero_cond_t: true``, which IS a constructor parameter of
``QwenImageTransformer2DModel`` and therefore changes the module the mint
traces; by B1's class/instance rule — an instance carries only weights,
``tuned`` and a ref label — a differing architecture config is a different
``ModelSpec``. (The t2i config's own ``pooled_projection_dim`` is the mirror
case and is why the rule needs measuring rather than eyeballing: it is NOT a
constructor parameter in the pinned diffusers, so it changes nothing.)

**The edit arm is also not AUTHORABLE yet, and that is the endpoint's own
posture rather than a scoping shortcut.** ``QwenImageEdit`` deliberately
declares no ``compile=``: its traced class set is 56 token counts derived by a
FUNCTION (``presets.edit_token_counts()``), and pgw#1112 item 3 requires the
boundary-shrink's parity to be proven ON A POD, on the served w8a8 lane, before
those counts are declared — because declaring them ARMS a mint. Writing them
into a ``Bucket`` here would do exactly what that item forbids, one layer down.
Recorded as owed, not skipped.

**One runner.** The endpoint compiles the transformer and nothing else
(``targets=("transformer",)``), and the reason is recorded in its own
declaration: the pgw#728 survey measured the DiT only, and re-arming the
decoder needs a measurement (wan measured its own at 0.46x — a LOSS; ltx's at
1.32-1.41x — a win; both signs live in this fleet's data). The ~15.5 GiB
Qwen2.5-VL text encoder stays eager and is shared, content-addressed, with the
edit arm.
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
from .qwen_image_serve import (
    JOINT_DIM,
    PACKED_CHANNELS,
    SHAPE_BUCKETS,
    TEXT_TOKENS,
    QwenImageTuned,
    compute_dtype,
    img_shapes,
    packed_tokens,
)

#: Qwen-Image's 20B MMDiT, from ``Qwen/Qwen-Image``'s published
#: ``transformer/config.json``. Class-level truth: no weight, no checkpoint
#: ref, no tuned value appears here or can.
#:
#: ``guidance_embeds`` is FALSE, which is why guidance on this family means
#: TRUE CFG — a second sequential forward, a call count, not a tensor input.
#: The published config's ``pooled_projection_dim`` is deliberately absent: it
#: is not a constructor parameter of ``QwenImageTransformer2DModel`` in the
#: pinned diffusers, and a declaration that passed it would refuse to build.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "patch_size": 2,
    "in_channels": PACKED_CHANNELS,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": JOINT_DIM,
    "guidance_embeds": False,
    "axes_dims_rope": (16, 56, 56),
}

#: Qwen-Image's scheduler block, as the checkpoint's own
#: ``scheduler/scheduler_config.json`` states it. DECLARED here so it rides the
#: export digest.
#:
#: TWO of these keys are read by :mod:`gen_worker.model.flow_ladders` and not
#: by :class:`~gen_worker.model.scheduler.FlowMatchEulerDiscrete`, and they are
#: not decoration: ``shift_terminal`` stretches the whole ladder so its last
#: evaluated sigma is 0.02 rather than the shifted ``1/steps``, and this is the
#: only family in the catalog that publishes one. ``max_image_seq_len`` is 8192
#: here, twice FLUX's, and ``max_shift`` 0.9 rather than 1.15 — so the dynamic
#: shift this family resolves at 1 MP is genuinely its own.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential",
}


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import QwenImageTransformer2DModel
    from torch import nn

    # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot be
    # swapped in place, so the dtype has to be in force while the module is
    # BUILT. `fake_structure()` restores the process default afterwards.
    torch.set_default_dtype(compute_dtype(layout))
    transformer: Any = QwenImageTransformer2DModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer(**dict(TRANSFORMER))

        def forward(
            self,
            hidden_states: Any,
            encoder_hidden_states: Any,
            encoder_hidden_states_mask: Any,
            timestep: Any,
            img_shapes: Any,
        ) -> Any:
            # Every non-tensor argument the served call does not use is pinned
            # to the value it passes. `guidance` is None because
            # `guidance_embeds` is false; the three controlnet/extra-condition
            # arguments are None because nothing on this path supplies them,
            # and a `None` left to a default that could move is a graph that
            # could change without the declaration changing.
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                guidance=None,
                attention_kwargs=None,
                controlnet_block_samples=None,
                additional_t_cond=None,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    shape = int(bucket["shape"])
    tokens = packed_tokens(shape)
    return CallExample(
        params=(
            "hidden_states",
            "encoder_hidden_states",
            "encoder_hidden_states_mask",
            "timestep",
            "img_shapes",
        ),
        kwargs={
            "hidden_states": torch.zeros(1, tokens, PACKED_CHANNELS, dtype=dtype),
            "encoder_hidden_states": torch.zeros(1, TEXT_TOKENS, JOINT_DIM, dtype=dtype),
            # int64 EXACTLY: the mask is a padding mask, never the compute
            # dtype. All ones — the pinned window is fully attended in the
            # example, so the trace cannot specialize on a shorter prompt.
            "encoder_hidden_states_mask": torch.ones(1, TEXT_TOKENS, dtype=torch.long),
            # The compute dtype, and the SIGMA rather than the 0..1000 moment:
            # the pipeline passes `timestep / 1000`.
            "timestep": torch.zeros(1, dtype=dtype),
            # PYTHON INTS. They specialize the graph — which is precisely why
            # this family's bucket is a (width, height) pair and not a token
            # count, and why transposed presets are two classes.
            "img_shapes": img_shapes(shape),
        },
    )


#: Qwen-Image (text-to-image). ONE runner over ONE axis: fourteen presets,
#: fourteen graph classes — the same fourteen the endpoint's own
#: ``aot/transformer-<w>x<h>.mint.json`` set carries, one file per row.
#:
#: The ie#740 floors are preserved BY VALUE from the endpoint's retired
#: ``Slot`` (pgw#1346 K1): ``sm89+`` is the DECODABLE floor for the rowwise fp8
#: lane (``W8A8_MIN_SM``; the rowwise GEMM's sm90 is the fast path, not the
#: floor) and ``vram72g`` is the bf16 lane's, agreeing with all fourteen
#: ``aot/transformer-*.mint.json`` ``declared_vram_gb``. Keyed by COMPONENT
#: PATH per K4's ruling, and on the MODEL's layout axis rather than the
#: runner's traced-variant axis: a weight lane is a load-time rung (th#546's
#: fit ladder), not a graph class.
QWEN_IMAGE: Final = GraphModelSpec(
    name="qwen_image",
    tuned=QwenImageTuned,
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram72g",
    },
    buckets=(Bucket("shape", SHAPE_BUCKETS),),
    runners=(
        Runner(
            "denoiser",
            build=_denoiser,
            example=_denoiser_example,
            axes=("shape",),
            # W1b-2's serving fact: `build` makes a WEIGHTLESS module from
            # config, so serving eagerly means reaching the weight-bearing one
            # the loader produced. It is NOT exported — the digest beside this
            # file is unchanged by it.
            component="transformer",
        ),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"),)),
    # The base function's declared payload range (ge=10, le=80) widened at the
    # bottom to 1, because the turbo lane's fixed 8-step distill regime and the
    # boot warm-up's single step both run through the same family parameter.
    parameters=(Parameter("steps", minimum=1, maximum=80),),
    # A set of ONE (pgw#1346 K10): this family's tuned schema names no sampler
    # because it serves exactly this schedule, so `inst.scheduler()` still
    # takes no argument and still returns the concrete class.
    schedulers={"flow_match_euler": Scheduler("flow_match_euler_discrete", SCHEDULER)},
)

__all__ = [
    "QWEN_IMAGE",
    "SCHEDULER",
    "TRANSFORMER",
]
