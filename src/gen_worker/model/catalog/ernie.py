"""ERNIE-Image, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule: an endpoint imports
:class:`~gen_worker.model.catalog.Ernie` — the generated binding beside this
file — and never touches diffusers. This module is where diffusers is allowed,
and only inside the ``build`` callables, which run at MINT time and on an
eager-capable serving pod.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs lives in
:mod:`gen_worker.model.catalog.ernie_serve`.

**Checkpoint-free.** The block below is ERNIE-Image's architecture, read from
the published ``transformer/config.json`` of ``baidu/ERNIE-Image``. It is not
any checkpoint's weights, which is what lets ONE declaration serve both
published checkpoints — see ``ernie_serve``'s header for the measurement that
makes Base and Turbo two INSTANCES rather than two models.

**One runner, and it is the endpoint's own choice.** The ernie endpoint
narrows ``targets=("transformer",)`` from the SDK default, with its reason
recorded: the default built ``vae.decode`` plans out of the transformer's own
input rows, and the decode is one call per request against 20+ denoiser steps.
The Mistral3 text encoder and the Ministral3 prompt enhancer are eager for the
same reason plus a harder one — the enhancer is a SAMPLED autoregressive LLM
pass, which is not a graph class at all.

**14 declared classes, and that number is the endpoint's own.** Seven presets
times two CFG arms. The endpoint states why it cannot be collapsed: ERNIE's
latents reach the DiT as SPATIAL ``(B, C, H_lat, W_lat)`` through an
``nn.Conv2d`` patch embed, so 1200x896 and 896x1200 are genuinely different
conv graphs — the same reason DESIGN-RULINGS §4.30 keeps SDXL on buckets.
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
from .ernie_serve import (
    BATCH_BUCKETS,
    LATENT_CHANNELS,
    SHAPE_BUCKETS,
    TEXT_TOKENS,
    TEXT_WIDTH,
    ErnieTuned,
    compute_dtype,
    latent_shape,
)

#: ERNIE-Image's 8B single-stream DiT, from ``baidu/ERNIE-Image``'s published
#: ``transformer/config.json``. Class-level truth: no weight, no checkpoint
#: ref, no tuned value appears here or can.
#:
#: The Turbo checkpoint's config carries two further keys — ``lora_rank`` and
#: ``use_lora`` — and NEITHER is a constructor parameter of
#: ``ErnieImageTransformer2DModel`` in the pinned diffusers, so they cannot
#: change the module this builds. That is the measurement behind one
#: declaration for two checkpoints, and ``tests/test_ernie_pgw1346.py`` asserts
#: it against the real signature rather than restating it here.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_layers": 36,
    "ffn_hidden_size": 12288,
    "in_channels": LATENT_CHANNELS,
    "out_channels": LATENT_CHANNELS,
    "patch_size": 1,
    "text_in_dim": TEXT_WIDTH,
    "rope_theta": 256,
    "rope_axes_dim": (32, 48, 48),
    "eps": 1e-06,
    "qk_layernorm": True,
}

#: ERNIE-Image's scheduler block, as the checkpoint's own
#: ``scheduler/scheduler_config.json`` states it — both checkpoints publish the
#: identical block. DECLARED here so it rides the export digest: a re-declared
#: schedule changes the family's identity instead of silently changing every
#: request.
#:
#: ``use_dynamic_shifting`` is FALSE, so the four resolution-interpolation
#: constants beside it are inert; they are carried because the checkpoint
#: carries them and a block that quietly drops half a published config is how a
#: later reader concludes the resolution never mattered. ``shift_terminal`` is
#: published as ``null`` and is therefore absent rather than zero — a scheduler
#: block holds finite JSON scalars, and 0.0 would be a DIFFERENT ladder.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "shift": 4.0,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "time_shift_type": "exponential",
}


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import ErnieImageTransformer2DModel
    from torch import nn

    # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot be
    # swapped in place, so the dtype has to be in force while the module is
    # BUILT. `fake_structure()` restores the process default afterwards.
    torch.set_default_dtype(compute_dtype(layout))
    # Bound as a value, not called through the imported name: diffusers ships
    # no complete stubs, and this keeps the untyped boundary at ONE line per
    # build instead of a `type: ignore` on every attribute of the result.
    transformer: Any = ErnieImageTransformer2DModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer(**dict(TRANSFORMER))

        def forward(
            self,
            hidden_states: Any,
            timestep: Any,
            text_bth: Any,
            text_lens: Any,
        ) -> Any:
            return self.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                text_bth=text_bth,
                text_lens=text_lens,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    batch = int(bucket["batch"])
    rows, cols = latent_shape(int(bucket["shape"]))
    return CallExample(
        params=("hidden_states", "timestep", "text_bth", "text_lens"),
        kwargs={
            "hidden_states": torch.zeros(batch, LATENT_CHANNELS, rows, cols, dtype=dtype),
            # The compute dtype, not float32: the pipeline materializes the
            # timestep with `torch.full(..., dtype=self.transformer.dtype)`.
            "timestep": torch.ones(batch, dtype=dtype),
            "text_bth": torch.zeros(batch, TEXT_TOKENS, TEXT_WIDTH, dtype=dtype),
            # int64 EXACTLY: `_pad_text` builds it `dtype=torch.long`, and the
            # module `.float()`s it internally — which is the tell that it does
            # not arrive as one. Zero-length would mask every key, so the
            # example states a real length.
            "text_lens": torch.full((batch,), TEXT_TOKENS, dtype=torch.long),
        },
    )


#: ERNIE-Image. ONE runner over TWO axes, which is the endpoint's own 14.
#:
#: The ie#740 floor is preserved BY VALUE from the endpoint's retired ``Slot``
#: (pgw#1346 K1): ``vram32g`` on ``plain.bf16@1``, and bf16 is the ONLY lane
#: this family serves — both ernie ``@endpoint`` classes declare exactly one
#: layout, unlike qwen-image and z-image, so there is no fp8 rung to guard and
#: none is invented here.
ERNIE: Final = GraphModelSpec(
    name="ernie",
    tuned=ErnieTuned,
    layouts={"*": ("plain.bf16@1",)},
    layout_requirements={"plain.bf16@1": "vram32g"},
    buckets=(
        Bucket("batch", BATCH_BUCKETS),
        Bucket("shape", SHAPE_BUCKETS),
    ),
    runners=(
        Runner(
            "denoiser",
            build=_denoiser,
            example=_denoiser_example,
            axes=("batch", "shape"),
            # W1b-2's serving fact: `build` makes a WEIGHTLESS module from
            # config, so serving eagerly means reaching the weight-bearing one
            # the loader produced. It is NOT exported — the digests beside this
            # file are unchanged by it.
            component="transformer",
        ),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"),)),
    # The union of the two lanes' declared payload ranges: the base function
    # takes 1..100 and the turbo function 1..16, and a family parameter bounds
    # the family rather than either lane.
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    # A set of ONE (pgw#1346 K10): this family's tuned schema names no sampler
    # because it serves exactly this schedule, so `inst.scheduler()` still
    # takes no argument and still returns the concrete class.
    schedulers={"flow_match_euler": Scheduler("flow_match_euler_discrete", SCHEDULER)},
)

__all__ = [
    "ERNIE",
    "SCHEDULER",
    "TRANSFORMER",
]
