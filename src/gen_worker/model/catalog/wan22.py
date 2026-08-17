"""Wan 2.2, declared. THREE models, and the count is the B4 finding.

The pgw#1346 W2 batch plan scoped B4 as ``Wan22Ti2v`` + ``Wan22A14b`` — one
model for the dense 5B checkpoint and one for "the A14B pair, because it is two
DiTs". Measured against the published ``transformer/config.json`` of all three
checkpoints, the shipped mint declarations and the hub's own family list, that
pairing is REFUTED in one direction and confirmed in the other:

``T2V-A14B`` and ``I2V-A14B`` are **two models, not two instances**, on exactly
B1's rule (class = architecture config, instance = weights + tuned + ref):

===================  ==========  ==========  ============
config key           T2V-A14B    I2V-A14B    TI2V-5B
===================  ==========  ==========  ============
``in_channels``      16          **36**      **48**
``out_channels``     16          16          **48**
``num_layers``       40          40          **30**
``num_attention_heads`` 40       40          **24**
``ffn_dim``          13824       13824       **14336**
===================  ==========  ==========  ============

I2V's 36 is not a width tweak: the endpoint's own mint file spells it out —
``"conditioning": "channel_concat_36 = 16 noisy + 4 mask + 16 cond"``
(``wan-2.2/aot/i2v-a14b.mint.json``) versus T2V's ``"text_only"``. A different
input channel count is a different first convolution and a different traced
call, which is the same evidence class B1 used to split FLUX.1-dev from
FLUX.1-schnell. The hub agrees: ``wan-22-t2v-a14b``, ``wan-22-i2v-a14b`` and
``wan-22-ti2v-5b`` are three separately registered architecture strings
(``serverless-endpoints/KNOWN_FAMILIES``), and the repo ships three separate
``aot/*.mint.json`` declarations.

**Confirmed in the other direction: A14B really is two DiTs, and the loop says
so.** ``transformer`` (high-noise expert, trained on t in [875, 1000]) and
``transformer_2`` (low-noise, t in [0, 875]) publish BYTE-IDENTICAL configs —
so they are one graph class twice, not two — but they are two weight sets run
in a stated order, which is exactly what two runners over two counted stages
expresses. TI2V-5B is dense and declares one.

**And the declaration is an improvement on the pipeline it describes.** diffusers
switches experts on a ``boundary_ratio`` THRESHOLD, so "how many steps does each
expert get" is an implicit consequence of ``(steps, shift, boundary_ratio)`` —
which is why, in the endpoint's own words, "the split moved silently whenever
the shift moved" (``wan_2_2/scheduling.py:5-16``). The loop below states the
budget directly, as ``steps_high`` and ``steps_low``. That is how the rest of
the ecosystem already thinks about it, and the vocabulary carries it natively.

**The runner -> component map W1b-2 said did not exist anywhere in the repo now
does, and A14B is the family it was missing FOR.** ``Runner(component=)`` names
where a runner lives in the loaded tree, and here the two runners resolve to
``transformer`` and ``transformer_2`` — which is what turns "the expert pair is
two weight sets over one graph class" from a comment into the thing the eager
backing actually reaches. ``component`` is a SERVING fact and is not exported,
so setting it moves no export digest.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs is in
:mod:`gen_worker.model.catalog.wan22_serve`.
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
from .wan22_serve import (
    A14B_SPATIAL,
    A14B_TEMPORAL,
    TEXT_DIM,
    TEXT_TOKENS,
    TI2V_SPATIAL,
    TI2V_TEMPORAL,
    Wan22Tuned,
    compute_dtype,
    latent_grid,
    packed_shape,
)

#: T2V-A14B's transformer architecture, read from
#: ``Wan-AI/Wan2.2-T2V-A14B-Diffusers@5be7df96:transformer/config.json``. Both
#: experts publish this identically, which is why ONE constant serves two
#: runners: the pair is two weight sets over one graph class.
TRANSFORMER_T2V_A14B: Final[Mapping[str, Any]] = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 40,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": TEXT_DIM,
    "freq_dim": 256,
    "ffn_dim": 13824,
    "num_layers": 40,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-06,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}

#: I2V-A14B, from ``Wan2.2-I2V-A14B-Diffusers@596658fd``. ONE key differs from
#: T2V and it is the one that makes this a different model: ``in_channels`` 36
#: = 16 noisy + 4 mask + 16 conditioning, channel-concatenated before the patch
#: embedding.
#:
#: ``image_dim`` stays ``None`` deliberately. Wan **2.1** I2V took a CLIP image
#: embedding through a cross-attention branch and diffusers still gates
#: ``encode_image`` on that config key; Wan 2.2 I2V conditions through the
#: channel concat instead, so the branch is absent from the graph.
TRANSFORMER_I2V_A14B: Final[Mapping[str, Any]] = {
    **TRANSFORMER_T2V_A14B,
    "in_channels": 36,
}

#: TI2V-5B, from ``Wan2.2-TI2V-5B-Diffusers@b8fff731``. A narrower, shallower,
#: dense DiT over the Wan 2.2 VAE's 48-channel latents — hence ``in_channels``
#: and ``out_channels`` both 48 where the A14B pair is 16.
TRANSFORMER_TI2V_5B: Final[Mapping[str, Any]] = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "in_channels": 48,
    "out_channels": 48,
    "text_dim": TEXT_DIM,
    "freq_dim": 256,
    "ffn_dim": 14336,
    "num_layers": 30,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-06,
    "image_dim": None,
    "added_kv_proj_dim": None,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}

#: The A14B lanes' two served shapes and their one trained frame grid. Both
#: come from the endpoint's own preset table (``wan_2_2/main.py:237-311``) and
#: are the exact rows its ``aot/{t2v,i2v}-a14b.mint.json`` declare:
#: ``[[1280, 720, 81], [720, 1280, 81]]``.
#:
#: 81 frames is the TRAINED temporal grid (4k+1 at 16 fps, ~5 s), not a product
#: choice: 24/48/60 fps are DELIVERY rungs reached by RIFE after the VAE
#: decode, and the endpoint's own note records that the grids divide exactly
#: (81 -> 121 / 241 / 301).
A14B_SHAPES: Final = (packed_shape(720, 1280), packed_shape(1280, 720))
A14B_FRAMES: Final = (81,)

#: TI2V-5B serves one canvas at one frame grid: 1280x704 at 121 frames
#: (24 fps native, ~5 s), per ``aot/ti2v-5b.mint.json``. 704x1280 portrait is
#: deliberately absent — the endpoint drops it as sub-720 width.
TI2V_SHAPES: Final = (packed_shape(1280, 704),)
TI2V_FRAMES: Final = (121,)


def _wan_denoiser(config: Mapping[str, Any]) -> Any:
    """Build one Wan DiT, wrapped so its traced call is the binding's call."""

    def build(layout: str) -> Any:
        import torch
        from diffusers import WanTransformer3DModel
        from torch import nn

        # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot
        # be swapped in place, so the dtype has to be in force while the module
        # is BUILT.
        torch.set_default_dtype(compute_dtype(layout))
        transformer: Any = WanTransformer3DModel

        class _Denoiser(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer = transformer(**dict(config))

            def forward(
                self, hidden_states: Any, timestep: Any, encoder_hidden_states: Any
            ) -> Any:
                # Every non-tensor argument is pinned to the value the served
                # path passes, and each is a SPECIALIZING scalar rather than an
                # inherited default. `encoder_hidden_states_image` is None on
                # both Wan 2.2 lanes: 2.2 I2V conditions by channel concat, so
                # the CLIP branch 2.1 used is not on this path at all.
                return self.transformer(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=None,
                    return_dict=False,
                    attention_kwargs=None,
                )[0]

        return _Denoiser().eval()

    return build


def _a14b_example(channels: int) -> Any:
    """The A14B call: ONE timestep scalar per batch.

    ``aot/{t2v,i2v}-a14b.mint.json`` records it as
    ``"timestep_kind": "scalar_per_batch"``, and the endpoint declares the
    corresponding fork (``expand_timesteps`` served ``False``) with the measured
    reason: per-token timesteps flip rank branches, 5066 nodes versus 6437.
    """

    def example(bucket: Mapping[str, int], layout: str) -> CallExample:
        import torch

        dtype = compute_dtype(layout)
        width, height = divmod(int(bucket["shape"]), 10000)
        f_lat, h_lat, w_lat = latent_grid(
            width, height, int(bucket["frames"]),
            spatial=A14B_SPATIAL, temporal=A14B_TEMPORAL,
        )
        return CallExample(
            params=("hidden_states", "timestep", "encoder_hidden_states"),
            kwargs={
                "hidden_states": torch.zeros(
                    1, channels, f_lat, h_lat, w_lat, dtype=dtype
                ),
                # int64, not the compute dtype: Wan's condition embedder takes
                # the raw trained timestep and builds its own sinusoid.
                "timestep": torch.zeros(1, dtype=torch.long),
                "encoder_hidden_states": torch.zeros(
                    1, TEXT_TOKENS, TEXT_DIM, dtype=dtype
                ),
            },
        )

    return example


def _ti2v_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    """The TI2V-5B call: a PER-TOKEN float32 timestep.

    ``aot/ti2v-5b.mint.json``'s ``"timestep_kind": "per_token_float32"``. This
    is the other side of the A14B fork above and it is the second reason
    TI2V-5B cannot be an instance of either A14B model: the calls differ in
    rank, not only in shape, and torchcg G2 makes one runner one signature.
    """

    import torch

    dtype = compute_dtype(layout)
    width, height = divmod(int(bucket["shape"]), 10000)
    f_lat, h_lat, w_lat = latent_grid(
        width, height, int(bucket["frames"]),
        spatial=TI2V_SPATIAL, temporal=TI2V_TEMPORAL,
    )
    tokens = f_lat * -(-h_lat // 2) * -(-w_lat // 2)
    return CallExample(
        params=("hidden_states", "timestep", "encoder_hidden_states"),
        kwargs={
            "hidden_states": torch.zeros(
                1, int(TRANSFORMER_TI2V_5B["in_channels"]), f_lat, h_lat, w_lat,
                dtype=dtype,
            ),
            "timestep": torch.zeros(1, tokens, dtype=torch.float32),
            "encoder_hidden_states": torch.zeros(
                1, TEXT_TOKENS, TEXT_DIM, dtype=dtype
            ),
        },
    )


# ---------------------------------------------------------------------------
# THE SCHEDULER BLOCK IS DELIBERATELY ABSENT ON ALL THREE, AND THAT IS pgw#1346
# K10 RECURRING — measured here for the second time, on a different family
# shape than B2 found it on.
#
# B2 found K10 as "the SAMPLER is a TUNED value and `Scheduler` is
# single-valued", from two endpoints whose PAYLOAD offers a sampler enum. Wan
# is the stronger case, because no payload is involved at all: the sampler is
# the CHECKPOINT's own, and one model serves two of them.
#
#   * BASE lineage -> `UniPCMultistepScheduler` with `use_flow_sigmas=True`,
#     `prediction_type="flow_prediction"`, `solver_order=2`, `solver_type="bh2"`,
#     `predict_x0=True`, `final_sigmas_type="zero"`, `lower_order_final=True`,
#     `flow_shift` 5.0 as shipped (TI2V-5B) / 3.0 (A14B mirrors, overridden to
#     12.0 at serve). This is the official Wan solver and swapping it is not
#     free: the endpoint measured **+81% for UniPC at 40 steps** when an earlier
#     revision silently rebuilt every shifted pick as FlowMatchEuler.
#   * DISTILLED lineage (Lightning / Seko LoRA markers) -> flow-match Euler on
#     the trained uniform ladder, at shift 5.0 (T2V) / 3.0 (I2V).
#
# Both are reachable on ONE declared model, selected by which adapter the
# request attaches. A single `Scheduler(...)` block would name one of them and
# quietly serve the other lane the wrong schedule, so this declaration names
# NEITHER and the models carry `scheduler=None`. That is not a gap left by
# scoping: `GraphModelSpec.scheduler` is optional precisely so a family with no
# single honest answer can decline to invent one, and codegen emits no
# `scheduler()` method, so a handler that wants one gets an AttributeError on
# the author's machine rather than a plausible wrong ladder on a pod.
#
# **AND THIS IS NOW K10 IN ITS STRONGEST FORM, because the math is no longer
# the obstacle.** B3-math (pgw#923) landed `unipc_multistep` as bare typed math
# and verified the flow lane at all three served `flow_shift` values
# (12.0 / 5.0 / 3.0) and every reachable step count — it is in `SchedulerKind`
# and in `IMPLEMENTED` today, and B4 proves below that the distilled lane's
# ladder is `flow_match_euler_discrete` exactly. So BOTH of this family's
# solvers are implemented, and the declaration STILL cannot carry them: the
# blocker was never a missing kind, it is that `GraphModelSpec.scheduler` is
# ONE block and codegen emits ONE `scheduler()`. K10's fix — a scheduler SET
# keyed by tuned name, `inst.scheduler()` reading `inst.tuned` — is what these
# three models are waiting on, and nothing else.
#
# WHAT B4 CLOSES, by measurement rather than by work: the plan's "wan needs a
# distilled flow-match" row is ALREADY IMPLEMENTED. `FlowMatchEulerDiscrete`
# under a STATIC shift resolves sigma_i = shift*x/(1+(shift-1)*x) over
# x_i = (N-i)/N — which is, term for term, the endpoint's own
# `distilled_sigmas()` and `shifted_sigma()` pair. The endpoint had to subclass
# diffusers only because diffusers double-shifts (its `__init__` shifts
# sigma_max/sigma_min and `set_timesteps` shifts again, landing a 4-step
# shift-5 run on t=24 instead of 625). This module's math has never had that
# bug. `tests/test_wan22_pgw1346.py` asserts the two live-verified ladders
# exactly. WHAT REMAINS OWED IS UniPC ALONE, and the 27-key served config above
# is its reference.
# ---------------------------------------------------------------------------


#: Wan 2.2 T2V-A14B — text to video, the expert pair, and the only Wan lane
#: with an fp8 layout.
#:
#: The ie#740 floors are preserved BY VALUE from the retired ``Slot``
#: (``wan_2_2/main.py:2152-2160``), including the reason each one is what it is:
#: sm89 is the DECODABLE floor for the rowwise lane (the GEMM's sm90 is the
#: fast path, not the floor), and the 80 GB is a PRODUCTION INCIDENT class
#: number that ``aot/t2v-a14b.mint.json`` declares for exactly this lane.
WAN22_T2V_A14B: Final = GraphModelSpec(
    name="wan22_t2v_a14b",
    tuned=Wan22Tuned,
    buckets=(Bucket("frames", A14B_FRAMES), Bucket("shape", A14B_SHAPES)),
    layouts={"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={
        "cozy.fp8-rowwise@1": "sm89+",
        "plain.bf16@1": "vram80g",
    },
    runners=(
        Runner(
            "denoiser_high",
            build=_wan_denoiser(TRANSFORMER_T2V_A14B),
            example=_a14b_example(int(TRANSFORMER_T2V_A14B["in_channels"])),
            axes=("frames", "shape"),
            component="transformer",
        ),
        Runner(
            "denoiser_low",
            build=_wan_denoiser(TRANSFORMER_T2V_A14B),
            example=_a14b_example(int(TRANSFORMER_T2V_A14B["in_channels"])),
            axes=("frames", "shape"),
            component="transformer_2",
        ),
    ),
    loop=Loop(
        stages=(
            Stage("denoiser_high", repeat="steps_high"),
            Stage("denoiser_low", repeat="steps_low"),
        )
    ),
    parameters=(
        # 1..80 on each half, because the endpoint's own payload bound is
        # `ge=1, le=80` on the TOTAL and either expert may legally take all of
        # it at the edges of the supported shift interval.
        Parameter("steps_high", minimum=1, maximum=80),
        Parameter("steps_low", minimum=1, maximum=80),
    ),
)

#: Wan 2.2 I2V-A14B — image to video. Same expert pair, same shapes, a
#: 36-channel first convolution, and bf16 ONLY.
#:
#: The missing fp8 lane is deliberate and is preserved as an absence: the
#: endpoint's comment reads "fp8 was never measured on the I2V lane, so its
#: binding stays bf16" (``wan_2_2/main.py:800``). Uniformising the two A14B
#: declarations would silently claim a rung nobody measured.
WAN22_I2V_A14B: Final = GraphModelSpec(
    name="wan22_i2v_a14b",
    tuned=Wan22Tuned,
    buckets=(Bucket("frames", A14B_FRAMES), Bucket("shape", A14B_SHAPES)),
    layouts={"*": ("plain.bf16@1",)},
    layout_requirements={"plain.bf16@1": "vram80g"},
    runners=(
        Runner(
            "denoiser_high",
            build=_wan_denoiser(TRANSFORMER_I2V_A14B),
            example=_a14b_example(int(TRANSFORMER_I2V_A14B["in_channels"])),
            axes=("frames", "shape"),
            component="transformer",
        ),
        Runner(
            "denoiser_low",
            build=_wan_denoiser(TRANSFORMER_I2V_A14B),
            example=_a14b_example(int(TRANSFORMER_I2V_A14B["in_channels"])),
            axes=("frames", "shape"),
            component="transformer_2",
        ),
    ),
    loop=Loop(
        stages=(
            Stage("denoiser_high", repeat="steps_high"),
            Stage("denoiser_low", repeat="steps_low"),
        )
    ),
    parameters=(
        Parameter("steps_high", minimum=1, maximum=80),
        Parameter("steps_low", minimum=1, maximum=80),
    ),
)

#: Wan 2.2 TI2V-5B — one dense DiT over the 16x/4x Wan 2.2 VAE, and the whole
#: point of ie#740's per-lane floors: a 5B model that sat behind the same 80 GB
#: endpoint scalar as the A14B pair while its own ``aot/ti2v-5b.mint.json`` had
#: said ``declared_vram_gb = 24.0`` all along.
WAN22_TI2V_5B: Final = GraphModelSpec(
    name="wan22_ti2v_5b",
    tuned=Wan22Tuned,
    buckets=(Bucket("frames", TI2V_FRAMES), Bucket("shape", TI2V_SHAPES)),
    layouts={"*": ("plain.bf16@1",)},
    layout_requirements={"plain.bf16@1": "vram24g"},
    runners=(
        Runner(
            "denoiser",
            build=_wan_denoiser(TRANSFORMER_TI2V_5B),
            example=_ti2v_example,
            axes=("frames", "shape"),
            component="transformer",
        ),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"),)),
    parameters=(Parameter("steps", minimum=1, maximum=80),),
)

__all__ = [
    "A14B_FRAMES",
    "A14B_SHAPES",
    "TI2V_FRAMES",
    "TI2V_SHAPES",
    "TRANSFORMER_I2V_A14B",
    "TRANSFORMER_T2V_A14B",
    "TRANSFORMER_TI2V_5B",
    "WAN22_I2V_A14B",
    "WAN22_T2V_A14B",
    "WAN22_TI2V_5B",
]
