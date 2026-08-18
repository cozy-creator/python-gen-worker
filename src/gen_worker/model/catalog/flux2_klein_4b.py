"""FLUX.2-klein-4B, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule: an endpoint imports
:class:`~gen_worker.model.catalog.Flux2Klein4b` — the generated binding beside
this file — and never touches diffusers. This module is where diffusers and
transformers are allowed, and only inside the ``build`` callables, which run at
MINT time and on an eager-capable serving pod.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs lives in
:mod:`gen_worker.model.catalog.flux2_klein_4b_serve`.

**Checkpoint-free, and the architecture config is why.** The blocks below are
klein-4B's architecture, read from the published ``transformer/config.json``
and ``text_encoder/config.json``. They are not any checkpoint's weights, which
is what lets ONE declaration serve both published checkpoints.

**ONE model, TWO instances — measured, not assumed (pgw#1346 B1).**
``FLUX.2-klein-4B`` (step-distilled, "Turbo") and ``FLUX.2-klein-base-4B``
(undistilled, "Base") ship BYTE-IDENTICAL transformer configs — same
``num_layers``, ``num_single_layers``, ``num_attention_heads``,
``joint_attention_dim``, ``in_channels``, ``guidance_embeds``. They differ only
in weights and in the published recipe (28 steps at guidance 4.0 versus 4 steps
at guidance 1.0), which is the definition of ``tuned``. So they are two
instances here rather than two declarations.

The same measurement REFUTES folding klein-9B in beside them: 9B is a different
width, is registered as its own hub family (``flux2-klein-9b``), and declares
its own VRAM envelope. A differing architecture config is a different
``ModelSpec`` by construction — instances carry only weights, ``tuned`` and a
ref label — so 9B gets its own catalog entry when its config is available.

**Two runners, not four, and no pooled branch.** Klein conditions on a single
Qwen3 encoder whose three intermediate layers are stacked into one 7680-wide
embedding; there is no CLIP tower and no ``pooled_projections`` input. The VAE
decode is deliberately NOT a runner — see ``flux2_klein_4b_serve.unpack_for_vae``
for the bucket-keying reason, which is the endpoint's own choice too
(``targets=("transformer",)``).
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
from .flux2_klein_4b_serve import (
    JOINT_DIM,
    PACKED_CHANNELS,
    TEXT_LAYERS,
    TEXT_TOKENS,
    TEXT_WIDTH,
    Flux2Klein4bLoraTuned,
    Flux2Klein4bTuned,
    compute_dtype,
)

#: klein-4B's transformer architecture, from the checkpoint's published
#: ``transformer/config.json``. Class-level truth: no weight, no checkpoint
#: ref, no tuned value appears here or can.
#:
#: ``guidance_embeds`` is FALSE, and that is a graph fact rather than a
#: preference: klein takes no guidance embedding, so the traced call has no
#: ``guidance`` input and the pipeline passes the literal ``None``. Guidance on
#: this family is CLASSIFIER-FREE — a second sequential forward, a call count,
#: not a tensor.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "patch_size": 1,
    "in_channels": PACKED_CHANNELS,
    "num_layers": 5,
    "num_single_layers": 20,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": JOINT_DIM,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "guidance_embeds": False,
}

#: Qwen3-4B's architecture, as klein's ``text_encoder`` uses it. Not built here
#: — see the runner note below — but declared because the denoiser's
#: ``joint_attention_dim`` is DERIVED from it (3 stacked layers x 2560), and a
#: reader who cannot see that derivation cannot check the 7680.
TEXT_ENCODER: Final[Mapping[str, Any]] = {
    "hidden_size": TEXT_WIDTH,
    "num_hidden_layers": 36,
    "hidden_states_layers": TEXT_LAYERS,
}

#: klein-4B's scheduler block, as the checkpoint's own
#: ``scheduler/scheduler_config.json`` states it. DECLARED here so it rides the
#: export digest: a re-declared schedule changes the family's identity instead
#: of silently changing every request.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}

#: The nine packed-token coordinates the endpoint's preset grid produces. They
#: are the family's bucket axis; the aspect/megapixel grid that generates them
#: is ENDPOINT policy and maps on through ``BucketMap`` (greenfield B5).
#: Fifteen (w, h) presets collapse onto these nine because a token count is
#: orientation-free.
TOKEN_BUCKETS: Final = (4056, 4070, 4089, 4096, 4116, 7744, 8160, 14400, 16384)


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import Flux2Transformer2DModel
    from torch import nn

    # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot be
    # swapped in place, so the dtype has to be in force while the module is
    # BUILT.
    torch.set_default_dtype(compute_dtype(layout))
    transformer: Any = Flux2Transformer2DModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer(**dict(TRANSFORMER))

        def forward(
            self,
            hidden_states: Any,
            encoder_hidden_states: Any,
            timestep: Any,
            img_ids: Any,
            txt_ids: Any,
        ) -> Any:
            # Every non-tensor argument is pinned to the value the served path
            # actually passes. `num_ref_tokens` and `ref_fixed_timestep` are
            # SPECIALIZING python scalars — a nonzero value bakes a different
            # graph — so they are stated outright rather than inherited from a
            # default that could move. `kv_cache` is None because
            # Flux2KleinKVPipeline is not on this path at all.
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                joint_attention_kwargs=None,
                kv_cache=None,
                kv_cache_mode=None,
                num_ref_tokens=0,
                ref_fixed_timestep=0.0,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    tokens = int(bucket["tokens"])
    return CallExample(
        params=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_ids",
            "txt_ids",
        ),
        kwargs={
            "hidden_states": torch.zeros(1, tokens, PACKED_CHANNELS, dtype=dtype),
            "encoder_hidden_states": torch.zeros(1, TEXT_TOKENS, JOINT_DIM, dtype=dtype),
            "timestep": torch.zeros(1, dtype=dtype),
            # Rope grids are int64 coordinates, never the compute dtype.
            "img_ids": torch.zeros(1, tokens, 4, dtype=torch.long),
            "txt_ids": torch.zeros(1, TEXT_TOKENS, 4, dtype=torch.long),
        },
    )


#: FLUX.2-klein-4B — ONE declared runner, the denoiser.
#:
#: **Why the text encoder and the VAE are not runners here**, stated because
#: FLUX.1-dev's entry declares all four of its components and a reader is owed
#: the difference rather than left to infer it:
#:
#: * it is the endpoint's OWN choice, not a scoping shortcut — klein's
#:   ``Compile`` declares ``targets=("transformer",)``, and its mint
#:   declaration gives the reason: compiling the shared Qwen3 encoder and VAE
#:   "would mint larger cells that do not exercise W8A8 scaled-mm and cannot be
#:   shared safely across the Base/edit and Turbo serving regimes";
#: * the VAE additionally CANNOT be keyed by this family's bucket axis — see
#:   ``flux2_klein_4b_serve.unpack_for_vae``;
#: * the text encoder additionally does not trace on transformers 5.x, whose
#:   ``output_hidden_states`` path installs capture hooks under a threading
#:   lock that dynamo refuses. Klein needs three INTERMEDIATE layers (9/18/27),
#:   so reaching them means re-implementing ``Qwen3Model.forward`` over the
#:   layer list — either truncating the stack (and breaking the 36-layer
#:   checkpoint's weight mapping) or carrying nine dead layers into the
#:   artifact. Both are real decisions that belong to whoever closes this eager
#:   tail; neither should be smuggled in under a 4B model's first landing.
#:
#: Recorded as owed rather than silently skipped (pgw#1346 B1).
#:
#: The layout axes are preserved BY VALUE from the endpoint's retired ``Slot``
#: (pgw#1346 K1, ie#740): sm89 is the DECODABLE floor for the rowwise fp8 lane
#: — the rowwise GEMM's sm90 is the fast path, not the floor — and the bf16
#: lane's 30 GB agrees with ``aot/transformer-4b.mint.json``'s
#: ``declared_vram_gb``. These are production floors; losing one silently is
#: the failure K1 exists to prevent.
FLUX2_KLEIN_4B: Final = GraphModelSpec(
    name="flux2_klein_4b",
    tuned=Flux2Klein4bTuned,
    lora_tuned=Flux2Klein4bLoraTuned,
    buckets=(Bucket("tokens", TOKEN_BUCKETS),),
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram30g",
    },
    runners=(
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("tokens",)),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"),)),
    parameters=(Parameter("steps", minimum=1, maximum=50),),
    # A set of ONE (pgw#1346 K10): this family's tuned schema names no sampler
    # because it serves exactly this schedule, so `inst.scheduler()` still
    # takes no argument and still returns the concrete class.
    schedulers={"flow_match_euler": Scheduler("flow_match_euler_discrete", SCHEDULER)},
)

__all__ = [
    "FLUX2_KLEIN_4B",
    "SCHEDULER",
    "TEXT_ENCODER",
    "TOKEN_BUCKETS",
    "TRANSFORMER",
]
