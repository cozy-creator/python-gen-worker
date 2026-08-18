"""pgw#1346 B4 — the video DiT families, and the claims B4's verdicts rest on.

Every test here is a claim about something OUTSIDE this repo (a published
config, an endpoint's shipped mint declaration, a live-verified ladder,
diffusers' own arithmetic) checked against the declaration that claims to
describe it. The ones that matter most:

1. **Wan is THREE models, not the two the batch plan scoped**, and the
   discriminator is architecture config — recomputed here from the numbers the
   checkpoints publish, not asserted from the declaration.
2. **A14B's two experts are ONE graph class run twice**, which is why they are
   two runners over two counted stages rather than two models.
3. **The "distilled flow-match" scheduler math B4 was told it owed is ALREADY
   IMPLEMENTED**, proved against the two ladders the endpoint live-verified.
4. **The ie#740 serving floors survive the Slot retirement BY VALUE**, asserted
   as parsed NUMBERS across five declarations.
5. **F2 is resolved**: MiniMax-H3's loop is staged-shaped, ``loop.kind: host``
   would forbid the bounded step count it really has, and the reason H3 is
   eager is architecture-source availability rather than either.
6. **K9-video**: ``shape`` x ``frames`` is TOTAL on Wan (no phantom compiled graph) and
   genuinely sparse on LTX (filed, not contorted).
"""

from __future__ import annotations

from typing import Any

import pytest

from gen_worker.model.errors import ModelError
from gen_worker.model.scheduler import (
    IMPLEMENTED,
    FlowMatchEulerDiscrete,
    Schedule,
    SchedulerKind,
)
from gen_worker.model.spec import Loop, LoopKind, Stage
from gen_worker.model.catalog import ltx23_serve as lx
from gen_worker.model.catalog import wan22_serve as wn
from gen_worker.model.catalog.ltx23 import (
    AUDIO_TOKENS,
    LTX23,
    TRANSFORMER as LTX_TRANSFORMER,
    VIDEO_TOKENS,
)
from gen_worker.model.catalog.minimax_h3 import (
    CONDITIONING_LAYER,
    DIFFUSERS_KEY_COUNT,
    MINIMAX_H3,
    MODALITY_NUM,
    NATIVE_KEY_COUNT,
    MiniMaxH3Tuned,
)
from gen_worker.model.catalog.wan22 import (
    A14B_FRAMES,
    A14B_SHAPES,
    TI2V_FRAMES,
    TI2V_SHAPES,
    TRANSFORMER_I2V_A14B,
    TRANSFORMER_T2V_A14B,
    TRANSFORMER_TI2V_5B,
    WAN22_I2V_A14B,
    WAN22_T2V_A14B,
    WAN22_TI2V_5B,
)
from gen_worker.model.spec import GraphModelSpec, ModelSpec

torch = pytest.importorskip("torch")

from gen_worker.model.catalog import (  # noqa: E402
    Ltx23,
    MinimaxH3,
    Wan22I2vA14b,
    Wan22T2vA14b,
    Wan22Ti2v5b,
)
from gen_worker.model.catalog._generated.ltx23 import Ltx23VideoTokens  # noqa: E402


#: The endpoint's own shipped mint declarations, transcribed HERE so every
#: bucket test derives the family's axis from the ENDPOINT's decision rather
#: than from the declaration it is checking.
#: ``wan-2.2/aot/{t2v,i2v}-a14b.mint.json`` and ``aot/ti2v-5b.mint.json``.
A14B_MINT_SHAPES: tuple[tuple[int, int, int], ...] = ((1280, 720, 81), (720, 1280, 81))
TI2V_MINT_SHAPES: tuple[tuple[int, int, int], ...] = ((1280, 704, 121),)


# ------------------------------------------------------- Wan: three, not two


def test_wan_is_three_models_because_the_architecture_configs_differ() -> None:
    """The B4 verdict, on B1's rule: a differing arch config is a different model.

    Recomputed from the published ``transformer/config.json`` of the three
    checkpoints. Two of the three differences would be invisible in a diff of
    the two A14B endpoint files, which declare their shapes symbolically.
    """

    assert TRANSFORMER_T2V_A14B["in_channels"] == 16
    # 16 noisy + 4 mask + 16 conditioning, channel-concatenated. A different
    # first convolution, so a different graph and a different traced call.
    assert TRANSFORMER_I2V_A14B["in_channels"] == 36
    assert TRANSFORMER_TI2V_5B["in_channels"] == 48

    # ...and I2V differs from T2V in EXACTLY that one key, which is why the
    # anti-triplication move is to share the constant and split the model —
    # not to pretend the two are one.
    differing = {
        key
        for key in set(TRANSFORMER_T2V_A14B) | set(TRANSFORMER_I2V_A14B)
        if TRANSFORMER_T2V_A14B.get(key) != TRANSFORMER_I2V_A14B.get(key)
    }
    assert differing == {"in_channels"}

    # TI2V-5B is a different network on four more axes besides.
    assert TRANSFORMER_TI2V_5B["num_layers"] == 30
    assert TRANSFORMER_T2V_A14B["num_layers"] == 40
    assert TRANSFORMER_TI2V_5B["num_attention_heads"] == 24
    assert TRANSFORMER_T2V_A14B["num_attention_heads"] == 40
    assert TRANSFORMER_TI2V_5B["ffn_dim"] == 14336
    assert TRANSFORMER_T2V_A14B["ffn_dim"] == 13824
    assert TRANSFORMER_TI2V_5B["out_channels"] == 48


def test_the_three_wan_declarations_resolve_three_different_graphs() -> None:
    """The strongest form of the verdict: the EXPORTS disagree.

    A shared declaration would resolve one ingress for one bucket. These
    resolve three, from the same bucket coordinate where two of them share
    one, which is a fact about the traced call rather than about the source.
    """

    bucket = {"frames": 81, "shape": wn.packed_shape(1280, 720)}
    t2v = Wan22T2vA14b.fake().variant("denoiser_high", bucket).ingress
    i2v = Wan22I2vA14b.fake().variant("denoiser_high", bucket).ingress
    assert t2v != i2v
    # TI2V does not even share the bucket, and its call differs in RANK.
    ti2v = Wan22Ti2v5b.fake().variant(
        "denoiser", {"frames": 121, "shape": wn.packed_shape(1280, 704)}
    ).ingress
    assert ti2v not in (t2v, i2v)


def test_the_a14b_experts_are_one_graph_class_run_twice() -> None:
    """Two runners, ONE class identity — the honest reading of the MoE pair.

    ``transformer`` and ``transformer_2`` publish byte-identical configs, so
    declaring two runners is a statement about WEIGHT SETS and ORDER, not about
    two graphs. If these ever diverge, this test says so before a mint does.
    """

    bucket = {"frames": 81, "shape": wn.packed_shape(1280, 720)}
    model = Wan22T2vA14b.fake()
    high = model.variant("denoiser_high", bucket)
    low = model.variant("denoiser_low", bucket)
    assert high.ingress == low.ingress
    # Equal ingress digests imply equal signatures, because the signature is a
    # projection of the ingress (torchcg G16).
    assert high.ingress_digest == low.ingress_digest


def test_the_loop_states_the_expert_budget_instead_of_deriving_it() -> None:
    """What the declaration adds over the pipeline it describes.

    diffusers switches experts on a ``boundary_ratio`` threshold, so the split
    is an implicit consequence of (steps, shift, boundary_ratio) — which is why
    the endpoint's own module says "the split moved silently whenever the shift
    moved". Two counted stages say it outright.
    """

    for model in (WAN22_T2V_A14B, WAN22_I2V_A14B):
        loop = model.staged_loop
        assert loop.kind is LoopKind.STAGED
        assert tuple((s.runner, s.repeat) for s in loop.stages) == (
            ("denoiser_high", "steps_high"),
            ("denoiser_low", "steps_low"),
        )
        assert tuple(p.name for p in model.parameters) == ("steps_high", "steps_low")

    # TI2V-5B is dense: one runner, one counted stage, one parameter.
    assert tuple(r.name for r in WAN22_TI2V_5B.runners) == ("denoiser",)
    assert tuple(
        (s.runner, s.repeat) for s in WAN22_TI2V_5B.staged_loop.stages
    ) == (("denoiser", "steps"),)


def test_the_two_experts_resolve_to_two_components_of_one_loaded_tree() -> None:
    """W1b-2's runner -> component map, and A14B is what it was missing for.

    The map "does not exist anywhere in the repo" was W1b-2's own finding; it
    landed with pgw#916, and this is the family that needs it most — two
    runners with the SAME graph class are distinguishable only by where they
    live in the loaded pipeline. Asserted alongside the fact that makes it
    safe: `component` is a SERVING fact, so it does not ride the export digest
    and restating one cannot re-key an artifact.
    """

    for model in (WAN22_T2V_A14B, WAN22_I2V_A14B):
        components = {r.name: r.component for r in model.runners}
        assert components == {
            "denoiser_high": "transformer",
            "denoiser_low": "transformer_2",
        }
    # TI2V-5B is dense — one expert, one component, no `transformer_2`.
    assert [r.component for r in WAN22_TI2V_5B.runners] == ["transformer"]
    # LTX compiles the joint DiT and nothing else, which is the endpoint's own
    # `Compile(targets=("transformer",))`.
    assert [r.component for r in LTX23.runners] == ["transformer"]


def test_the_component_path_does_not_ride_the_export_digest() -> None:
    """A serving fact must not re-key an artifact — the same claim B2 made for
    the ie#740 floors, on the other new field.

    The committed export documents are byte-compared by
    `scripts/check_model_bindings.py`; this asserts the reason that stays true:
    nothing in the exported runner row mentions a component path.
    """

    import json
    from importlib import resources

    document = json.loads(
        resources.files("gen_worker.model.catalog._generated")
        .joinpath("wan22_t2v_a14b.export.json")
        .read_text()
    )
    assert "transformer_2" not in json.dumps(document)
    assert {row["name"] for row in document["runners"]} == {
        "denoiser_high",
        "denoiser_low",
    }


# --------------------------------------------- the scheduler math B4 was owed


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        # Both ladders are quoted from `wan_2_2/scheduling.py:128-130`, where
        # they are recorded as LIVE-VERIFIED against the shipped checkpoints.
        (4, (1000.0, 937.5, 833.3, 625.0)),
        (8, (1000.0, 972.2, 937.5, 892.9, 833.3, 750.0, 625.0, 416.7)),
    ],
)
def test_the_distilled_flow_match_ladder_is_already_implemented(
    steps: int, expected: tuple[float, ...]
) -> None:
    """B4's "distilled flow-match owed" row, CLOSED by measurement.

    The endpoint had to subclass diffusers because diffusers double-shifts: its
    ``__init__`` shifts ``sigma_max``/``sigma_min`` and ``set_timesteps`` shifts
    again, landing a 4-step shift-5 run on t=24 instead of 625. This module's
    static-shift branch resolves ``sigma_i = s*x/(1+(s-1)*x)`` over
    ``x_i = (N-i)/N``, which is exactly the endpoint's ``distilled_sigmas`` fed
    through its ``shifted_sigma`` — so no new scheduler kind is owed for the
    distilled Wan lanes at all. Only UniPC remains.
    """

    ladder = FlowMatchEulerDiscrete(
        num_train_timesteps=wn.NUM_TRAIN_TIMESTEPS, shift=5.0, use_dynamic_shifting=False
    ).schedule(steps)
    assert tuple(round(t, 1) for t in ladder.timesteps) == expected
    # The unshifted ladder IS the endpoint's `distilled_sigmas(N)`.
    assert len(ladder) == steps
    assert ladder.sigmas[-1] == 0.0


def test_the_unshifted_ladder_is_the_endpoints_own_distilled_sigmas() -> None:
    """The identity behind the test above, asserted directly rather than implied.

    ``distilled_sigmas(N) == [(N-i)/N for i in range(N)]``
    (``wan_2_2/scheduling.py:151``), and shift 1.0 is the identity map.
    """

    for steps in (4, 8, 12, 40):
        ladder = FlowMatchEulerDiscrete(shift=1.0, use_dynamic_shifting=False).schedule(
            steps
        )
        assert tuple(round(s, 12) for s in ladder.sigmas[:-1]) == tuple(
            round((steps - i) / steps, 12) for i in range(steps)
        )


def test_the_wan_models_decline_to_name_a_scheduler() -> None:
    """K10, recurring — and the declaration refuses to guess.

    One Wan model serves TWO solvers: the checkpoint's own UniPC on flow sigmas
    for the base lineage, flow-match Euler on the trained uniform ladder for
    the distilled one, selected by which adapter a request attaches. A
    single-valued ``Scheduler`` block would name one and silently serve the
    other lane a schedule it was not trained on, so all three name none — and
    codegen therefore emits no ``scheduler()`` method, which turns a handler
    that wants one into an AttributeError on the author's machine.
    """

    for model in (WAN22_T2V_A14B, WAN22_I2V_A14B, WAN22_TI2V_5B):
        assert model.schedulers == {}
    for binding in (Wan22T2vA14b, Wan22I2vA14b, Wan22Ti2v5b):
        assert not hasattr(binding, "scheduler")

    # LTX, by contrast, has ONE honest answer and declares it — as a set of
    # one, keyed by sampler (pgw#1346 K10).
    assert list(LTX23.schedulers) == ["flow_match_euler"]
    assert LTX23.schedulers["flow_match_euler"].name == "flow_match_euler_discrete"
    assert hasattr(Ltx23, "scheduler")


def test_k10_is_now_a_declaration_limit_and_not_a_missing_implementation() -> None:
    """K10's sharpest form — and the LIMIT IS GONE. What remains is authoring.

    B3-math (pgw#923) landed `unipc_multistep` as bare typed math and verified
    the flow lane at all three served `flow_shift` values; B4 proves above that
    the distilled lane's ladder is `flow_match_euler_discrete` exactly. So the
    obstacle was never a missing scheduler kind — it was that
    `GraphModelSpec.scheduler` was ONE block and codegen emitted ONE
    `scheduler()`, leaving a family whose CHECKPOINT chooses the solver nowhere
    to put the second one.

    pgw#1346 K10 replaced that field with a SET keyed by the sampler a
    checkpoint is stamped with, so this test inverts: both solvers are
    implemented AND the declaration can now hold both. **What Wan still owes is
    the KEY**, not the mechanism — `Wan22Tuned` declares no sampler field, so
    there is no stamped value for `inst.scheduler()` to resolve. Adding it by
    value from the endpoint (the same by-value migration `SdxlTuned` made) is
    the whole of the remaining work, and it is B4's authoring call rather than
    a declaration-surface one.
    """

    assert SchedulerKind.UNIPC_MULTISTEP in IMPLEMENTED
    assert SchedulerKind.FLOW_MATCH_EULER_DISCRETE in IMPLEMENTED
    # The field that has to hold both is now a SET, and it is empty rather than
    # wrong: an empty set declares nothing, where a single-valued field would
    # have had to name one of the two solvers and be wrong for every checkpoint
    # served by the other.
    assert WAN22_T2V_A14B.schedulers == {}
    field = type(WAN22_T2V_A14B).__dataclass_fields__["schedulers"]
    assert "Mapping[str, Scheduler]" in str(field.type)
    # The key Wan owes: no sampler field to resolve against yet.
    assert "scheduler" not in WAN22_T2V_A14B.tuned.__struct_fields__  # type: ignore[union-attr]


def test_the_h3_eager_binding_is_a_type_with_no_runners() -> None:
    """B5's eager-tier codegen, applied to B4's one eager model.

    The eager tier produces a real `Model` subclass carrying the tuned schema
    and nothing else — no bucket literals, no runner callables, no
    `scheduler()`. That is what makes `MINIMAX_H3` annotatable on a handler
    parameter without claiming a composition it does not have.
    """

    assert MinimaxH3.Tuned is MiniMaxH3Tuned
    instance = MinimaxH3.fake()
    assert isinstance(instance.tuned, MiniMaxH3Tuned)
    assert (instance.tuned.video_shift, instance.tuned.audio_shift) == (12.0, 3.0)
    # No runner callable and no scheduler accessor: the two things codegen
    # emits only when a composition is declared.
    for absent in ("denoiser", "scheduler"):
        assert not hasattr(MinimaxH3, absent)
    # `variant()` is inherited from `Model`, and on the eager tier it REFUSES
    # rather than answering — there is no export to resolve a class from.
    with pytest.raises(ModelError):
        instance.variant("denoiser", {})


# ------------------------------------------------- ie#740 floors, BY VALUE


def test_the_ie740_serving_floors_are_preserved_by_value() -> None:
    """K1: the retired Slot's requirements axis, moved and NOT re-derived.

    Parsed by Slot's OWN normalizer, so each floor survives as a NUMBER a fit
    check can compare rather than as a string somebody must re-read. Every one
    of these is a production incident: the 80 is the lane the endpoint's single
    scalar was the maximum of, the 24 is the 5B model that sat behind it, and
    the 78 is tagged in the LTX endpoint as "two B200s at $6.83/hr rented and
    refused".
    """

    t2v = WAN22_T2V_A14B.layout_requirements
    assert t2v["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert t2v["plain.bf16@1"].minimum.min_vram_gb == 80.0
    # Each lane states only its own floor; a leak across lanes declines cards
    # that can serve.
    assert t2v["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert t2v["plain.bf16@1"].minimum.min_sm == 0

    assert WAN22_I2V_A14B.layout_requirements["plain.bf16@1"].minimum.min_vram_gb == 80.0
    assert WAN22_TI2V_5B.layout_requirements["plain.bf16@1"].minimum.min_vram_gb == 24.0

    ltx = LTX23.layout_requirements
    assert ltx["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert ltx["plain.bf16@1"].minimum.min_vram_gb == 78.0

    # th#1754's floor, on the eager tier — the layout axes are the MODEL's, so
    # they survive a model having no graph classes at all.
    h3 = MINIMAX_H3.layout_requirements
    assert h3["plain.bf16@1"].minimum.min_vram_gb == 78.0


def test_the_i2v_lane_has_no_fp8_and_the_absence_is_the_point() -> None:
    """"fp8 was never measured on the I2V lane, so its binding stays bf16."

    Uniformising the two A14B declarations would claim a rung nobody measured.
    Asserted so a later tidy-up cannot quietly add it.
    """

    t2v_layouts = WAN22_T2V_A14B.layouts
    i2v_layouts = WAN22_I2V_A14B.layouts
    assert t2v_layouts is not None and i2v_layouts is not None
    assert tuple(t2v_layouts["*"]) == ("cozy.fp8-rowwise@1", "plain.bf16@1")
    assert tuple(i2v_layouts["*"]) == ("plain.bf16@1",)


def test_the_weight_lane_is_not_a_traced_graph_variant() -> None:
    """fp8 is a LOAD-TIME rung, so it is a serving capability and not a class."""

    for model in (WAN22_T2V_A14B, LTX23):
        for runner in model.runners:
            assert runner.layouts == ("bf16",)


# ------------------------------------------------------------- K9, for video


def test_the_wan_bucket_product_is_total_which_is_k9s_video_answer() -> None:
    """``shape`` x ``frames`` generates EXACTLY the endpoint's mint rows.

    This is the half of K9-video that turns out fine. ``frames`` is a genuine
    product axis on Wan — one trained temporal grid per family — so adding it
    beside B2's packed ``shape`` costs no phantom compiled graph: 2 x 1 on A14B and
    1 x 1 on TI2V-5B, which is the exact set the shipped mint declarations
    carry.
    """

    derived = {
        (wn.packed_shape(w, h), f) for w, h, f in A14B_MINT_SHAPES
    }
    declared = {
        (bucket["shape"], bucket["frames"])
        for bucket in WAN22_T2V_A14B.runner("denoiser_high").buckets(
            WAN22_T2V_A14B.axis_values
        )
    }
    assert derived == declared
    assert len(declared) == len(A14B_MINT_SHAPES) == 2

    ti2v_derived = {(wn.packed_shape(w, h), f) for w, h, f in TI2V_MINT_SHAPES}
    ti2v_declared = {
        (bucket["shape"], bucket["frames"])
        for bucket in WAN22_TI2V_5B.runner("denoiser").buckets(
            WAN22_TI2V_5B.axis_values
        )
    }
    assert ti2v_derived == ti2v_declared == {(12800704, 121)}


def test_the_packed_shape_axis_round_trips_and_reads() -> None:
    """B2's K9 workaround, reused: the literal is readable as "W by H"."""

    assert wn.packed_shape(1280, 720) == 12800720
    assert wn.packed_shape(720, 1280) == 7201280
    assert wn.unpack_shape(12800704) == (1280, 704)
    assert A14B_SHAPES == (7201280, 12800720)
    assert TI2V_SHAPES == (12800704,)
    assert A14B_FRAMES == (81,) and TI2V_FRAMES == (121,)
    # Transposed shapes are TWO coordinates, because Wan's DiT reads the
    # (F, H, W) volume and not a token count: 1280x720 and 720x1280 are two
    # graphs even though their token counts agree.
    assert wn.denoiser_tokens(
        1280, 720, 81, spatial=wn.A14B_SPATIAL, temporal=wn.A14B_TEMPORAL
    ) == wn.denoiser_tokens(
        720, 1280, 81, spatial=wn.A14B_SPATIAL, temporal=wn.A14B_TEMPORAL
    )
    assert wn.packed_shape(1280, 720) != wn.packed_shape(720, 1280)


def test_the_ltx_axes_are_a_sparse_set_which_is_k9_video_filed() -> None:
    """The half of K9-video that is REAL, stated as arithmetic.

    LTX's graph is a function of the token counts alone, and its two token axes
    are independent in principle but sparse in practice: the endpoint
    enumerates 82 graph classes (h100) over 20 distinct token counts and 28
    distinct ``(T_v, T_a)`` pairs — so the full axis product is roughly 3x the
    set anyone serves. What is DECLARED is the sub-product that is exact.
    """

    declared = {
        (bucket["video_tokens"], bucket["audio_tokens"])
        for bucket in LTX23.runner("denoiser").buckets(LTX23.axis_values)
    }
    assert len(declared) == len(VIDEO_TOKENS) * len(AUDIO_TOKENS) == 4
    # ...and the two committed mint requests are among them, by construction.
    assert (261120, 126) in declared and (261120, 251) in declared


# ------------------------------------------------------- the token arithmetic


def test_the_wan_latent_grid_keeps_the_first_frame_whole() -> None:
    """``(frames - 1) // temporal + 1`` — 81 frames at 4x is 21, not 20."""

    assert wn.latent_grid(1280, 720, 81, spatial=8, temporal=4) == (21, 90, 160)
    assert wn.latent_grid(720, 1280, 81, spatial=8, temporal=4) == (21, 160, 90)
    assert wn.latent_grid(1280, 704, 121, spatial=16, temporal=4) == (31, 44, 80)


def test_the_ti2v_token_count_is_the_endpoints_own_resolved_number() -> None:
    """27280 = 31 * ceil(44/2) * ceil(80/2), the value ``main.py:517`` resolves."""

    assert (
        wn.denoiser_tokens(
            1280, 704, 121, spatial=wn.TI2V_SPATIAL, temporal=wn.TI2V_TEMPORAL
        )
        == 27280
    )


def test_the_ltx_video_token_coordinates_recompute_from_the_presets() -> None:
    """261120 and 65280 DERIVED, so the declared axis cannot drift.

    The committed mint row is the 4K/241-frame preset with a ``last_frame``
    keyframe bound; its stage-1 partner is the same preset at half resolution,
    which is what the two-stage recipe actually runs first.
    """

    assert lx.video_tokens(3840, 2176, 241, last_frame=True) == 261120
    assert lx.video_tokens(1920, 1088, 241, last_frame=True) == 65280
    assert VIDEO_TOKENS == (65280, 261120)
    # The keyframe term is BINARY: a `first_frame` overwrites tokens and costs
    # nothing, a `last_frame` appends exactly one latent frame's worth.
    assert (
        lx.video_tokens(3840, 2176, 241, last_frame=True)
        - lx.video_tokens(3840, 2176, 241)
        == (2176 // 32) * (3840 // 32)
    )


def test_the_ltx_audio_axis_reads_the_frame_rate_which_a_shape_row_cannot() -> None:
    """T_a = round(frames / fps * 25), and 241f is 251 rather than 250.

    The endpoint's own comment — "frame_rate does not change the DiT graph;
    only num_frames does" — is FALSE for the audio stream, and this is the
    arithmetic that refutes it: one frame count, two legal frame rates, two
    different graphs.
    """

    assert lx.audio_tokens(241, 24) == 251
    assert lx.audio_tokens(241, 48) == 126
    assert AUDIO_TOKENS == (126, 251)
    # The trap: the NOMINAL 10 s bucket would give 250. 241 frames at 24 fps is
    # 10.0417 s.
    assert round(10 * lx.AUDIO_TOKENS_PER_SECOND) == 250


# ------------------------------------------------ LTX's literal sigma ladders


def test_the_literal_ladder_needs_no_new_scheduler_machinery() -> None:
    """B4's declared dependency on B3's explicit-sigma work does NOT exist.

    ``Schedule`` is a public frozen dataclass over an explicit sigma tuple, and
    the SYNTHESIS from a step count is what ``FlowMatchEulerDiscrete`` adds on
    top of it. A step-distilled family hands it the stamped list.
    """

    tuned = lx.Ltx23Tuned()
    stage1 = lx.schedule_from_sigmas(tuned.sigmas)
    stage2 = lx.schedule_from_sigmas(tuned.stage2_sigmas)
    assert isinstance(stage1, Schedule) and isinstance(stage2, Schedule)
    assert len(stage1) == 8 and len(stage2) == 3
    assert stage1.sigmas[:-1] == tuned.sigmas
    assert stage1.sigmas[-1] == 0.0 and stage2.sigmas[-1] == 0.0
    # The timesteps the model is conditioned on are sigma * 1000, which is the
    # transformer config's own `timestep_scale_multiplier`.
    assert LTX_TRANSFORMER["timestep_scale_multiplier"] == 1000
    assert stage1.timesteps[0] == 1000.0


def test_stage_two_is_the_tail_of_stage_one() -> None:
    """The refine RESUMES the ladder after a 2x upsample; it does not restart it."""

    tuned = lx.Ltx23Tuned()
    assert tuned.sigmas[-len(tuned.stage2_sigmas):] == tuned.stage2_sigmas
    assert tuned.stage2_sigmas == (0.909375, 0.725, 0.421875)


def test_a_stamped_ladder_carrying_the_terminal_zero_is_refused() -> None:
    """A catalog document must not double-count a step that does not exist."""

    with pytest.raises(ValueError, match="terminal"):
        lx.schedule_from_sigmas((1.0, 0.5, 0.0))
    with pytest.raises(ValueError):
        lx.schedule_from_sigmas(())


def test_the_ltx_loop_declares_two_counted_stages_over_one_runner() -> None:
    """One runner named twice — the composition, not a workaround.

    ``recipe_v1``'s loop is an ORDERED LIST of stages, so a runner appearing
    twice is something the vocabulary already describes.
    """

    loop = LTX23.staged_loop
    assert tuple((s.runner, s.repeat) for s in loop.stages) == (
        ("denoiser", "stage1_steps"),
        ("denoiser", "stage2_steps"),
    )
    assert tuple(r.name for r in LTX23.runners) == ("denoiser",)


# ---------------------------------------------------------- the loops, run


def _wan_a14b_call(tokens_shape: tuple[int, int, int], channels: int) -> dict[str, Any]:
    f_lat, h_lat, w_lat = tokens_shape
    return {
        "hidden_states": torch.zeros(1, channels, f_lat, h_lat, w_lat),
        "timestep": torch.zeros(1, dtype=torch.long),
        "encoder_hidden_states": torch.zeros(1, wn.TEXT_TOKENS, wn.TEXT_DIM),
    }


def test_a_fake_backed_a14b_runs_both_experts_in_the_declared_order() -> None:
    """The MoE loop, hubless and cardless, through the real typed callables.

    Not a mock of the SDK — it IS the SDK, with the only part that needs a card
    replaced. The assertion is that the declared budget is what runs: two
    high-noise forwards then two low-noise ones, which is the shipped
    Lightning 2H+2L row.
    """

    instance = Wan22T2vA14b.fake(tuned=wn.Wan22Tuned(num_inference_steps=4, shift=5.0))
    grid = wn.latent_grid(1280, 720, 81, spatial=8, temporal=4)
    call = _wan_a14b_call(grid, int(TRANSFORMER_T2V_A14B["in_channels"]))
    ladder = FlowMatchEulerDiscrete(shift=5.0, use_dynamic_shifting=False).schedule(4)

    latents = call["hidden_states"]
    order: list[str] = []
    for index in range(len(ladder)):
        expert = "denoiser_high" if index < 2 else "denoiser_low"
        order.append(expert)
        velocity = getattr(instance, expert)(
            frames=81,
            shape=wn.packed_shape(1280, 720),
            **{**call, "hidden_states": latents},
        )
        latents = ladder.step(index, velocity, latents)

    assert order == ["denoiser_high", "denoiser_high", "denoiser_low", "denoiser_low"]
    assert tuple(latents.shape) == (1, 16, *grid)


def test_a_fake_backed_ltx_runs_both_stages_on_the_stamped_ladders() -> None:
    """The two-stage loop, on the literal ladders, through the typed callable.

    The stage-2 bucket is a DIFFERENT video-token coordinate than stage 1's,
    which is what the 2x latent upsample between them buys — and the reason the
    loop declares stages while saying nothing about which bucket each runs at.
    """

    tuned = lx.Ltx23Tuned()
    instance = Ltx23.fake(tuned=tuned)
    seen: list[tuple[int, int]] = []

    lanes: tuple[tuple[tuple[float, ...], Ltx23VideoTokens], ...] = (
        (tuned.sigmas, 65280),
        (tuned.stage2_sigmas, 261120),
    )
    for stage, (sigmas, t_v) in enumerate(lanes):
        ladder = lx.schedule_from_sigmas(sigmas)
        latents = torch.zeros(1, t_v, lx.LATENT_CHANNELS)
        audio = torch.zeros(1, 251, lx.AUDIO_LATENT_CHANNELS)
        for index in range(len(ladder)):
            video_v, audio_v = instance.denoiser(
                audio_tokens=251,
                video_tokens=t_v,
                hidden_states=latents,
                audio_hidden_states=audio,
                encoder_hidden_states=torch.zeros(
                    1, lx.TEXT_TOKENS, lx.CROSS_ATTENTION_DIM
                ),
                audio_encoder_hidden_states=torch.zeros(
                    1, lx.TEXT_TOKENS, lx.AUDIO_CROSS_ATTENTION_DIM
                ),
                timestep=torch.zeros(1, t_v),
                audio_timestep=torch.zeros(1, 1),
                sigma=torch.zeros(1),
                audio_sigma=torch.zeros(1),
                encoder_attention_mask=torch.ones(1, lx.TEXT_TOKENS, dtype=torch.int64),
                audio_encoder_attention_mask=torch.ones(
                    1, lx.TEXT_TOKENS, dtype=torch.int64
                ),
                video_coords=torch.zeros(1, 3, t_v, 2),
                audio_coords=torch.zeros(1, 1, 251, 2),
            )
            latents = ladder.step(index, video_v, latents)
            audio = ladder.step(index, audio_v, audio)
            seen.append((stage, index))

    # 8 distilled steps at half resolution, then 3 refinement steps at full.
    assert [s for s, _ in seen].count(0) == 8
    assert [s for s, _ in seen].count(1) == 3


# ------------------------------------------------------- F2: MiniMax-H3


def test_h3_is_an_eager_model_and_declares_no_graph() -> None:
    """The F2 outcome, in the type system.

    ``ModelSpec`` and not ``GraphModelSpec``: no runners, no loop, no scheduler
    block — because H3's architecture has no source in this repo (it is
    vendored in the endpoint at a pinned diffusers SHA, and diffusers ships no
    MiniMax class), not because its loop is inexpressible.
    """

    assert isinstance(MINIMAX_H3, ModelSpec)
    assert not isinstance(MINIMAX_H3, GraphModelSpec)
    assert MINIMAX_H3.runners == ()
    assert MINIMAX_H3.tuned is MiniMaxH3Tuned


def test_h3_has_no_minted_backing_to_lose() -> None:
    """F2's cost premise, refuted at the source that would carry the cost.

    F2 weighed the eager tier as "losing the compiled backing it has today".
    minimax-h3 is the only video endpoint with no ``aot/`` directory and no
    mint declaration, and its own source says why: the family is not
    AOT-declarable (737k static classes). What it runs is ``torch.compile``
    over one dynamic sequence axis, which the eager tier does not touch.

    Asserted here as the declaration's own shape: an eager ``ModelSpec``
    withholds a MINTED compiled graph and nothing else, so there is no ``variants()`` to
    lose.
    """

    assert not hasattr(MINIMAX_H3, "variants")
    assert MINIMAX_H3.layouts is not None


def test_declaring_a_host_loop_would_forbid_the_step_count_h3_really_has() -> None:
    """Why ``loop.kind: host`` is the WRONG shape here, proved on the vocabulary.

    ``host`` is for iteration whose count no document can state, and
    ``recipe_v1`` refuses a repeat count under it so a fabricated bound cannot
    be read as a real one. H3's denoise runs a payload ``Literal[20, 30, 50]``
    and its ``long_video`` chunk loop runs 1..24 declared on the wire — both
    bounded. So a host loop would have to DROP a bound the family genuinely
    has, which is a worse lie than the one it exists to prevent.
    """

    # The staged form H3 would take is legal and ordinary.
    staged = Loop(stages=(Stage("denoiser", repeat="steps"),))
    assert staged.kind is LoopKind.STAGED

    # The host form cannot carry that count at all.
    with pytest.raises(ModelError, match="never a repeat count"):
        Loop(
            stages=(Stage("denoiser", repeat="steps"),),
            kind=LoopKind.HOST,
        )
    # A host loop with no count is legal — and it is exactly the thing H3 is
    # not: it says the iteration is data-dependent.
    assert Loop(stages=(Stage("denoiser"),), kind=LoopKind.HOST).kind is LoopKind.HOST


def test_h3s_scheduler_parameters_are_tuned_values_not_class_facts() -> None:
    """K10 again — the real reason a ``Scheduler`` block cannot carry H3.

    The two shifts are stamped per checkpoint AND are jointly the identity of
    te#171's AdaLN cache: ``CacheKey(steps, video_shift, audio_shift)``,
    because two shifts are what make a schedule of N steps a DIFFERENT
    schedule. ``recipe_v1`` G11 makes the class-level document structurally
    checkpoint-free, so they belong on ``tuned`` and they are there.
    """

    tuned = MiniMaxH3Tuned()
    assert (tuned.video_shift, tuned.audio_shift) == (12.0, 3.0)
    fields = {row.name for row in __import__("msgspec").structs.fields(MiniMaxH3Tuned)}
    assert fields == {"schema_version", "video_shift", "audio_shift"}
    # H3-Base is guidance-distilled: no guider, no negative prompt, no
    # guidance scale. Declaring one would name a knob the model does not have.
    assert "guidance" not in fields


def test_h3s_class_level_anatomy_is_declared_not_inferred() -> None:
    """The class-level facts the anatomy work established, carried by value.

    The conditioner trim is a CLASS fact (a module constant, applied once at
    setup, request-independent) and the adaLN index arithmetic is the literal
    meaning of "timestep-indexed adaLN". The two key-set counts are the te#185
    incident: one shared key out of 638 and 535.
    """

    assert CONDITIONING_LAYER == 50
    assert MODALITY_NUM == 3
    assert (NATIVE_KEY_COUNT, DIFFUSERS_KEY_COUNT) == (535, 638)


# ---------------------------------------- K11: the name the hub cannot join


def test_the_declared_names_are_not_the_hubs_architecture_strings() -> None:
    """NEW BLOCKER K11, filed as a red proof rather than as prose.

    ``ModelSpec.name`` does double duty: it is the generated symbol root (so
    torchcg G1 constrains it to ``[a-z][a-z0-9_]*`` — no hyphens, no dots) AND
    the key ``register_family`` publishes the tuned schema under, which is what
    tensorhub validates repo metadata against. Every hub architecture string
    these four families actually carry is hyphenated, and tensorhub's
    ``Normalize`` drops only ``.`` and spaces — so the two spellings cannot
    meet.

    Wan makes it sharper still: its THREE graph identities share ONE stamped
    vocabulary root (``wan22``), so even a hyphen-tolerant name would not fix
    it. The two identities are genuinely different axes and the field set has
    one slot for both.
    """

    declared = {
        WAN22_T2V_A14B.name,
        WAN22_I2V_A14B.name,
        WAN22_TI2V_5B.name,
        LTX23.name,
        MINIMAX_H3.name,
    }
    assert declared == {
        "wan22_t2v_a14b",
        "wan22_i2v_a14b",
        "wan22_ti2v_5b",
        "ltx23",
        "minimax_h3",
    }

    # tensorhub's Normalize, mirrored: lowercase, trim, drop "." and " ".
    def normalize(value: str) -> str:
        return value.strip().lower().replace(".", "").replace(" ", "")

    hub_strings = {
        "wan-22-t2v-a14b",
        "wan-22-i2v-a14b",
        "wan-22-ti2v-5b",
        "ltx-23",
        "minimax-h3",
    }
    assert not (declared & {normalize(row) for row in hub_strings})

    # The three Wan graph identities share ONE tuned vocabulary root, which is
    # the half no naming rule can reconcile.
    assert (
        WAN22_T2V_A14B.tuned
        is WAN22_I2V_A14B.tuned
        is WAN22_TI2V_5B.tuned
        is wn.Wan22Tuned
    )
