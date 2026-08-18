"""FLUX.2-klein-9B, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule: an endpoint imports
:class:`~gen_worker.model.catalog.Flux2Klein9b` — the generated binding beside
this file — and never touches diffusers. This module is where diffusers is
allowed, and only inside the ``build`` callables, which run at MINT time and on
an eager-capable serving pod.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs lives in
:mod:`gen_worker.model.catalog.flux2_klein_9b_serve`.

**The architecture config has a SOURCE, and that source is the reason this
entry exists at all.** pgw#1346 B1 authored klein-4B and recorded that 9B was
NOT authorable: no 9B ``transformer/config.json`` is cached on any authoring
box, and both klein-9b endpoints deliberately carry no checkpoint ref
(ie#524/th#980 makes the ref a deploy-time binding), so the class-level width
had nothing to be derived from. It is derived here from the SERVING RELEASE'S
OWN TREE — the published configs, seven JSON documents totalling 4.8 KB, read
through the hub catalog's resolve route and cached under
``tests/fixtures/flux2_klein_9b/`` with the release id and every file digest.
No weights were fetched and none may be added there.
``tests/test_flux2_klein_9b_pgw1346.py`` re-hashes the fixture and asserts the
blocks below against it, so the declaration cannot silently drift from the
checkpoint it describes.

**Two models, not one — the measurement B1 predicted, now made.** klein-4B and
klein-9B are separately registered hub families (``flux2-klein-4b`` /
``flux2-klein-9b``) with separate VRAM envelopes, and the fetched configs
confirm the architectures differ: 8 double + 24 single blocks over 32 heads at
``joint_attention_dim`` 12288, against 4B's 5 + 20 over 24 at 7680. An instance
carries only ref, tuned, backing and label (``model/runtime.py::_materialize``),
so a differing architecture config is a different ``ModelSpec`` by
construction.

**ONE model, TWO instances — the same measurement, the other way round.**
``flux2-klein-9b`` (step-distilled, "Turbo") and ``flux2-klein-base-9b``
(undistilled, "Base") ship transformer configs that differ ONLY in
``_name_or_path``, a provenance string with no architecture in it. Every
architectural field is equal and both trees carry byte-identical
``scheduler``/``text_encoder``/``vae`` configs. They differ in weights and in
the published recipe (28 steps at guidance 4.0 versus 4 steps at guidance 1.0),
which is the definition of ``tuned``.

**One runner, and no pooled branch.** Klein conditions on a single Qwen3
encoder whose three intermediate layers are stacked into one 12288-wide
embedding; there is no CLIP tower and no ``pooled_projections`` input. Neither
the text encoder nor the VAE decode is a declared runner — see the runner note
below, which is klein-4B's reasoning at a second width.
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
from .flux2_klein_4b import SCHEDULER, TOKEN_BUCKETS
from .flux2_klein_4b_serve import PACKED_CHANNELS, TEXT_LAYERS, TEXT_TOKENS
from .flux2_klein_9b_serve import (
    JOINT_DIM,
    TEXT_WIDTH,
    Flux2Klein9bLoraTuned,
    Flux2Klein9bTuned,
    compute_dtype,
)

#: klein-9B's transformer architecture, from the published
#: ``transformer/config.json`` of BOTH 9B releases (they agree on every field
#: below). Class-level truth: no weight, no checkpoint ref, no tuned value
#: appears here or can.
#:
#: ``guidance_embeds`` is FALSE, and that is a graph fact rather than a
#: preference: klein takes no guidance embedding, so the traced call has no
#: ``guidance`` input and the pipeline passes the literal ``None``. Guidance on
#: this family is CLASSIFIER-FREE — a second sequential forward, a call count,
#: not a tensor.
#:
#: ``joint_attention_dim`` is spelled as the DERIVED ``JOINT_DIM`` rather than
#: the literal 12288, so it cannot drift from "three stacked Qwen3-8B layers".
TRANSFORMER: Final[Mapping[str, Any]] = {
    "patch_size": 1,
    "in_channels": PACKED_CHANNELS,
    "num_layers": 8,
    "num_single_layers": 24,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "joint_attention_dim": JOINT_DIM,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "guidance_embeds": False,
}

#: Qwen3-8B's architecture, as klein-9B's ``text_encoder`` uses it. Not built
#: here — see the runner note — but declared because the denoiser's
#: ``joint_attention_dim`` is DERIVED from it, and a reader who cannot see that
#: derivation cannot check the 12288.
TEXT_ENCODER: Final[Mapping[str, Any]] = {
    "hidden_size": TEXT_WIDTH,
    "num_hidden_layers": 36,
    "hidden_states_layers": TEXT_LAYERS,
}


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
            # actually passes, restating the endpoint's own `Compile(args=...)`
            # block. `num_ref_tokens` and `ref_fixed_timestep` are SPECIALIZING
            # python scalars — a nonzero value bakes a different graph — so they
            # are stated outright rather than inherited from a default that
            # could move. `kv_cache` is None because Flux2KleinKVPipeline is not
            # on this path at all (the endpoint declares that fork unserved).
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


#: FLUX.2-klein-9B — ONE declared runner, the denoiser.
#:
#: **Why the text encoder and the VAE are not runners here.** Both reasons are
#: klein-4B's, and both are width-independent, so they reproduce exactly:
#:
#: * it is the endpoint's OWN choice — klein-9b's ``Compile`` declares
#:   ``targets=("transformer",)``, and its mint declaration gives the reason:
#:   compiling the shared Qwen3 encoder and VAE "would mint larger compiled graphs that
#:   do not exercise W8A8 scaled-mm and cannot be shared safely across the
#:   Base/edit and Turbo serving regimes";
#: * the VAE additionally CANNOT be keyed by this model's bucket axis — see
#:   ``flux2_klein_9b_serve.unpack_for_vae``;
#: * the text encoder additionally does not trace on transformers 5.x, whose
#:   ``output_hidden_states`` path installs capture hooks under a threading
#:   lock that dynamo refuses. Klein needs three INTERMEDIATE layers (9/18/27),
#:   so reaching them means re-implementing ``Qwen3Model.forward`` over the
#:   layer list — either truncating the 36-layer stack (breaking the
#:   checkpoint's weight mapping) or carrying nine dead layers into the
#:   artifact.
#:
#: Recorded as owed rather than silently skipped, and the tail is the SAME tail
#: klein-4B left open — one fix closes both (pgw#1346 B1/B3b).
#:
#: The bucket axis is klein-4B's, and that is a measurement rather than a
#: shortcut: the two endpoints ship BYTE-IDENTICAL ``presets.py`` files, and a
#: token count is a function of pixel size and the VAE stride, neither of which
#: the denoiser's width touches. The test recomputes it from the 9b endpoint's
#: own preset grid.
#:
#: The layout axes are preserved BY VALUE from the endpoint's retired ``Slot``
#: (pgw#1346 K1, ie#740): sm89 is the DECODABLE floor for the rowwise fp8 lane
#: — the rowwise GEMM's sm90 is the fast path, not the floor — and the bf16
#: lane's 44 GB is the endpoint's own scalar, matching
#: ``aot/transformer-9b.mint.json``'s declared serve envelope. These are
#: production floors; losing one silently is the failure K1 exists to prevent.
FLUX2_KLEIN_9B: Final = GraphModelSpec(
    name="flux2_klein_9b",
    tuned=Flux2Klein9bTuned,
    lora_tuned=Flux2Klein9bLoraTuned,
    buckets=(Bucket("tokens", TOKEN_BUCKETS),),
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram44g",
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
    "FLUX2_KLEIN_9B",
    "TEXT_ENCODER",
    "TRANSFORMER",
]
