"""Z-Image, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule: an endpoint imports
:class:`~gen_worker.model.catalog.ZImage` — the generated binding beside this
file — and never touches diffusers. This module is where diffusers is allowed,
and only inside the ``build`` callable, which runs at MINT time and on an
eager-capable serving pod.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs lives in
:mod:`gen_worker.model.catalog.z_image_serve`.

**Checkpoint-free, and the two published checkpoints agree on it.**
``Tongyi-MAI/Z-Image`` and ``Tongyi-MAI/Z-Image-Turbo`` publish transformer
configs that are identical key for key — 30 layers, 3840 wide, 30 heads, 16
latent channels, the same rope axes — so they are two INSTANCES of this one
model. They differ in weights and in the published scheduler ``shift`` (6.0
versus 3.0), which ``ZImageTuned.shift`` carries.

**TWO declared classes, and the resolution is SYMBOLIC inside each.** This is
the first catalog family whose shape axis is not a bucket, and it is the
endpoint's own measured shape: ``shape_strategy="dynamic-collapse"`` collapses
ten preset rows onto one program per CFG arm because
``transformer_z_image.py`` contains ZERO ``nn.Conv*`` layers — so #730's
channels-last argument for keeping conv families on static buckets simply does
not apply here. The declaration says the same thing by declaring ONE bucket
axis (the CFG pytree arity) and handing ``torch.export`` two symbolic latent
extents through ``CallExample.dynamic``.

**The two symbolic extents are only reachable because of two rewrites**, and
they live beside this module in :mod:`gen_worker.model.catalog.z_image_graph`:
without them the export graph-breaks on a lazily-built rope table and is
refused for an equality guard that pins the very symbols being declared. Both
are the endpoint's own (ie#630, ie#637), moved to where the mint traces.
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
from .z_image_serve import (
    BRANCH_BUCKETS,
    CAPTION_WIDTH,
    FRAME_PATCH,
    LATENT_CHANNELS,
    PATCH,
    TEXT_TOKENS,
    ZImageTuned,
    compute_dtype,
    latent_bounds,
)

#: Z-Image's 6B DiT, from ``Tongyi-MAI/Z-Image``'s published
#: ``transformer/config.json``. Class-level truth: no weight, no checkpoint
#: ref, no tuned value appears here or can.
#:
#: ``siglip_feat_dim`` is published as ``null`` on the base checkpoint and
#: absent on Turbo, which is the same constructed module either way — it is
#: omitted here rather than declared ``None``, because a declaration states
#: what the architecture IS.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "all_patch_size": (PATCH,),
    "all_f_patch_size": (FRAME_PATCH,),
    "in_channels": LATENT_CHANNELS,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "cap_feat_dim": CAPTION_WIDTH,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
}

#: Z-Image's scheduler block, as the BASE checkpoint's own
#: ``scheduler/scheduler_config.json`` states it. DECLARED here so it rides the
#: export digest.
#:
#: The block is three keys because the published config is three keys — no
#: dynamic shifting, so no resolution-interpolation constants exist to carry.
#: The official Turbo checkpoint publishes the same block with ``shift: 3.0``,
#: and that difference is a per-CHECKPOINT value: it is stamped through
#: ``ZImageTuned.shift`` rather than forked into a second declaration, because
#: a shift changes no graph.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "shift": 6.0,
    "use_dynamic_shifting": False,
}


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import ZImageTransformer2DModel
    from torch import nn

    from .z_image_graph import install

    # Both rewrites BEFORE construction: the rope one replaces the class
    # `ZImageTransformer2DModel.__init__` resolves as a module global, so it
    # cannot be applied afterwards. See `z_image_graph` for the two measured
    # export failures they remove.
    install()

    # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot be
    # swapped in place, so the dtype has to be in force while the module is
    # BUILT. `fake_structure()` restores the process default afterwards.
    torch.set_default_dtype(compute_dtype(layout))
    transformer: Any = ZImageTransformer2DModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer(**dict(TRANSFORMER))

        def forward(self, x: Any, t: Any, cap_feats: Any) -> Any:
            # STACKED TENSORS IN, ONE STACKED TENSOR OUT, and the unbind/stack
            # the module wants happens INSIDE the traced region. This is not
            # cosmetic — it is what makes the CFG arity a bucket at all.
            #
            # ⚠️ MEASURED (pgw#1346 B3a): the module's own signature takes
            # LISTS, and a list is flattened by `torch.export` into one input
            # PER ELEMENT — so the two arms exported two different call
            # signatures (three flat inputs versus five) and the export was
            # refused with `signature_disagreement`. torchcg G2 is right to
            # refuse it: ONE runner is ONE typed binding, and variants may
            # differ only in concrete dimensions. Stacking makes the arity a
            # concrete DIMENSION of one input, which is exactly what a bucket
            # is allowed to vary.
            #
            # The two lines moved in are the pipeline's own
            # (`list(latent_model_input.unbind(dim=0))` before the call, a
            # `torch.stack` after it). The NEGATION and the guidance
            # combination stay OUT: they are loop arithmetic, and the traced
            # class is the module.
            return torch.stack(
                self.transformer(
                    x=list(x.unbind(dim=0)),
                    t=t,
                    cap_feats=list(cap_feats.unbind(dim=0)),
                    return_dict=False,
                    patch_size=PATCH,
                    f_patch_size=FRAME_PATCH,
                )[0],
                dim=0,
            )

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    branches = int(bucket["branches"])
    minimum, maximum = latent_bounds()
    # The patchify folds 2x2, so the latent extents are multiples of 2 — stated
    # as a DERIVED dim rather than as a bare range, which is what keeps the
    # patchify's own view() from re-deriving a divisibility guard.
    rows = 2 * torch.export.Dim("rows_half", min=minimum // 2, max=maximum // 2)
    cols = 2 * torch.export.Dim("cols_half", min=minimum // 2, max=maximum // 2)
    # The example is traced at 1024x1024, the anchor of the 1 MP grid. Any
    # declared row would do — the extents are symbolic — and this one is the
    # cheapest to reason about.
    return CallExample(
        params=("x", "t", "cap_feats"),
        kwargs={
            # (branches, C, F, rows, cols): the frame axis is the pipeline's
            # own `unsqueeze(2)`, and the leading axis is the CFG arity — a
            # CONCRETE dimension, which is what lets one runner carry both arms.
            "x": torch.zeros(branches, LATENT_CHANNELS, 1, 128, 128, dtype=dtype),
            # float32 and UNCAST, deliberately: `timestep = (1000 - t) / 1000`
            # reaches the module with no `.to(...)` anywhere on its path. This
            # is the fleet's least-cast ingress row (ie#629).
            "t": torch.zeros(branches, dtype=torch.float32),
            "cap_feats": torch.zeros(
                branches, TEXT_TOKENS, CAPTION_WIDTH, dtype=dtype
            ),
        },
        dynamic={"x": {3: rows, 4: cols}, "t": None, "cap_feats": None},
    )


#: Z-Image. ONE runner over ONE axis — the CFG pytree arity — and TWO classes,
#: which is exactly the endpoint's own ``aot/transformer-cfg-{on,off}.mint.json``
#: pair.
#:
#: The ie#740 floors are preserved BY VALUE from the endpoint's retired
#: ``Slot`` (pgw#1346 K1): ``sm89+`` is the DECODABLE floor for the rowwise fp8
#: lane (``W8A8_MIN_SM``; the rowwise GEMM's sm90 is the fast path, not the
#: floor) and ``vram40g`` is the bf16 lane's, agreeing with both
#: ``aot/transformer-cfg-*.mint.json`` ``declared_vram_gb``. Both of the
#: endpoint's slots declare the identical pair, which is the case K1's ruling
#: is built for: two bindings of one model state ONE demand.
Z_IMAGE: Final = GraphModelSpec(
    name="z_image",
    tuned=ZImageTuned,
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram40g",
    },
    buckets=(Bucket("branches", BRANCH_BUCKETS),),
    runners=(
        Runner(
            "denoiser",
            build=_denoiser,
            example=_denoiser_example,
            axes=("branches",),
            # W1b-2's serving fact: `build` makes a WEIGHTLESS module from
            # config, so serving eagerly means reaching the weight-bearing one
            # the loader produced. It is NOT exported — the digest beside this
            # file is unchanged by it.
            component="transformer",
        ),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"),)),
    # The base function's declared payload range. The two turbo functions cap
    # at 16, and a family parameter bounds the family rather than either lane.
    parameters=(Parameter("steps", minimum=1, maximum=80),),
    # A set of ONE (pgw#1346 K10): this family's tuned schema names no sampler
    # because it serves exactly this schedule, so `inst.scheduler()` still
    # takes no argument and still returns the concrete class.
    schedulers={"flow_match_euler": Scheduler("flow_match_euler_discrete", SCHEDULER)},
)

__all__ = [
    "SCHEDULER",
    "TRANSFORMER",
    "Z_IMAGE",
]
