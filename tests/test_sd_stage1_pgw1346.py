"""pgw#1346 B2 — the U-Net families serve end to end, on bare typed math.

The Flux lane's four claims (pgw#1331), applied to the batch that owed the
harder half of them:

1. **The bare math is the same math** — and here "same" means BIT-IDENTICAL,
   not within a tolerance. ``euler_discrete``'s ladder descends from a trained
   noise schedule rather than from a closed form, so reproducing it means
   reproducing the PRECISION it was produced at; the float64 reading of the
   same algebra is 201 float32 ULP away, and one of these tests exists purely
   to keep that number from becoming folklore.
2. **The whole composition is graph classes** — SDXL grew its two text towers,
   so no SD family leaves a tower riding an eager model at serve time.
3. **SD2 is not SD1.5 with different numbers.** B2's scoping proposed an
   instance; the shapes say a declaration. Tested, not asserted in a comment.
4. **The gaps are still shaped like refusals** — including the one this batch
   CLOSED, so the mechanism is fenced at the generator now that no catalog
   family exercises it.

The ddim decision (pgw#1346 owed it) is recorded at the bottom, as a test over
what the endpoints actually invoke rather than as prose.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast, get_args

import pytest

from gen_worker.model.catalog import Sd2, Sd15, Sdxl
from gen_worker.model.catalog import sd15_serve as sd
from gen_worker.model.catalog import sdxl_serve as sx
from gen_worker.model.catalog.sd15 import SCHEDULER as SD_SCHEDULER
from gen_worker.model.catalog.sd15 import SD2, SD15
from gen_worker.model.catalog.sdxl import SCHEDULER as SDXL_SCHEDULER
from gen_worker.model.catalog.sdxl import SDXL
from gen_worker.model.errors import ModelError
from gen_worker.model.scheduler import (
    IMPLEMENTED,
    DiscreteSchedule,
    EulerDiscrete,
    SchedulerKind,
    parse_kind,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gen_worker.model.catalog._generated.sdxl import SdxlShape

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

#: Every step count the sdxl and sd15 endpoints can actually reach: their
#: stamped defaults (28 / 30 / 1), the distilled recipes their handlers pin
#: (Lightning 4 and 8, Hyper-SD 4, DMD2 4, SD-Turbo 1), and the ends of the
#: declared `steps` parameter range.
ENDPOINT_STEPS = (1, 4, 8, 20, 25, 28, 30, 40, 50, 100)


def _ulp(ours: Any, theirs: Any) -> int:
    """Distance in float32 units in the last place. 0 is bit equality."""

    a = np.asarray(ours, dtype=np.float32)
    b = np.asarray(theirs, dtype=np.float32)
    return int(
        np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max()
    )


def _reference(block: dict[str, Any], **overrides: Any) -> Any:
    diffusers = pytest.importorskip("diffusers")
    merged = {**block, **overrides}
    return diffusers.EulerDiscreteScheduler(
        num_train_timesteps=int(merged["num_train_timesteps"]),
        beta_start=float(merged["beta_start"]),
        beta_end=float(merged["beta_end"]),
        beta_schedule=str(merged["beta_schedule"]),
        timestep_spacing=str(merged["timestep_spacing"]),
        steps_offset=int(merged["steps_offset"]),
        prediction_type=str(merged["prediction_type"]),
        final_sigmas_type=str(merged["final_sigmas_type"]),
        rescale_betas_zero_snr=bool(merged.get("rescale_betas_zero_snr", False)),
        interpolation_type="linear",
    )


# ---------------------------------------------------------------- the bare math


@pytest.mark.parametrize("steps", ENDPOINT_STEPS)
@pytest.mark.parametrize(
    "spacing",
    [
        # `leading` is what every SD/SDXL checkpoint's own scheduler config
        # carries, so it is the `euler` sampler's spacing. `trailing` is the
        # `euler_trailing` sampler — SDXL-Lightning's published recipe, which
        # the sdxl endpoint's `_TURBO_RECIPES` pins for both its 4- and 8-step
        # arms. `linspace` is diffusers' class default and reachable only by a
        # checkpoint that declares nothing; included so the default is not the
        # one path nobody measured.
        "leading",
        "trailing",
        "linspace",
    ],
)
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_the_euler_discrete_ladder_is_bit_identical_to_diffusers(
    steps: int, spacing: str, objective: str, zero_snr: bool
) -> None:
    """Zero ULP, not one — and the bar was one.

    ``v_prediction`` is paired with ``rescale_betas_zero_snr`` because
    ``gen_worker.view`` pairs them for the diffusers path: a v-pred checkpoint
    on this fleet is ALWAYS served with the zero-terminal-SNR beta rescale, so
    measuring the objective without it would measure a configuration no request
    reaches.
    """

    block = {
        **dict(SDXL_SCHEDULER),
        "timestep_spacing": spacing,
        "prediction_type": objective,
        "rescale_betas_zero_snr": zero_snr,
    }
    theirs = _reference(block)
    theirs.set_timesteps(steps)
    ours = EulerDiscrete.from_block(block).schedule(steps)

    assert len(ours.sigmas) == len(theirs.sigmas)
    assert _ulp(ours.sigmas, theirs.sigmas.numpy()) == 0
    assert _ulp(ours.timesteps, theirs.timesteps.numpy()) == 0
    assert _ulp([ours.init_noise_sigma], [float(theirs.init_noise_sigma)]) == 0


@pytest.mark.parametrize("spacing", ["leading", "trailing"])
@pytest.mark.parametrize(
    ("objective", "zero_snr"), [("epsilon", False), ("v_prediction", True)]
)
def test_a_whole_denoising_loop_is_bitwise_the_diffusers_loop(
    spacing: str, objective: str, zero_snr: bool
) -> None:
    """28 steps of ``scale_model_input`` + ``step``, element for element.

    The ladder being right is half of it; the STEP is the other half, and it is
    the half where the algebraically obvious rewrite is wrong. Run at float32
    because that is where the two are comparable: upstream upcasts its sample
    to float32 internally and this module cannot name a dtype, so float32 is
    the precision at which "the same math" is a decidable claim.
    """

    block = {
        **dict(SDXL_SCHEDULER),
        "timestep_spacing": spacing,
        "prediction_type": objective,
        "rescale_betas_zero_snr": zero_snr,
    }
    theirs = _reference(block)
    theirs.set_timesteps(28)
    schedule = EulerDiscrete.from_block(block).schedule(28)

    torch.manual_seed(0)
    ours_sample = torch.randn(2, 4, 32, 32) * float(schedule.init_noise_sigma)
    their_sample = ours_sample.clone()
    for index, timestep in enumerate(theirs.timesteps):
        prediction = torch.randn(2, 4, 32, 32)
        assert torch.equal(
            schedule.scale_model_input(index, ours_sample),
            theirs.scale_model_input(their_sample, timestep),
        )
        ours_sample = schedule.step(index, prediction, ours_sample)
        their_sample = theirs.step(prediction, timestep, their_sample).prev_sample
        assert torch.equal(ours_sample, their_sample)


def test_the_float64_reading_of_the_same_algebra_is_measurably_wrong() -> None:
    """Why the implementation narrows to float32 stage by stage.

    This is the finding pgw#1331 asked every scheduler lane to state, made
    executable so it cannot rot into a comment nobody re-derives: computing the
    identical algebra honestly in float64 is HUNDREDS of float32 ULP away from
    what the fleet serves today, because ``alphas_cumprod`` is a thousand-term
    cumulative product and float32 is the precision it is DEFINED at.
    """

    theirs = _reference(dict(SDXL_SCHEDULER))
    theirs.set_timesteps(28)

    total = int(SDXL_SCHEDULER["num_train_timesteps"])
    start = math.sqrt(float(SDXL_SCHEDULER["beta_start"]))
    end = math.sqrt(float(SDXL_SCHEDULER["beta_end"]))
    running = 1.0
    table = []
    for index in range(total):
        root = start + (end - start) * index / (total - 1)
        running *= 1.0 - root * root
        table.append(math.sqrt((1.0 - running) / running))

    reference_table = np.asarray(
        ((1 - theirs.alphas_cumprod) / theirs.alphas_cumprod) ** 0.5, dtype=np.float32
    )
    # 201 ULP on the trained table itself…
    assert _ulp(table, reference_table) > 200

    naive = [table[int(t)] + (table[int(t) + 1] - table[int(t)]) * (t - int(t))
             for t in theirs.timesteps.tolist()]
    # …and 25 on the 28 sigmas a request actually walks. Interpolation averages
    # some of the drift away, which is exactly why "it looks fine" is not a
    # measurement: still 25x the bar, and monotonically worse with step count.
    assert _ulp(naive + [0.0], theirs.sigmas.numpy()) > 15

    # …and the shipped implementation, on the same inputs, is exact.
    assert _ulp(EulerDiscrete.from_block(dict(SDXL_SCHEDULER)).schedule(28).sigmas,
                theirs.sigmas.numpy()) == 0


def test_the_declared_block_is_the_scheduler_s_only_source_of_constants() -> None:
    """A constant hardcoded in the math would be a second declaration of it.

    Sharper here than it is for Flux: :class:`EulerDiscrete` defaults to
    DIFFUSERS' class defaults — linear betas over [1e-4, 2e-2] with linspace
    spacing — which no Stable Diffusion was ever trained on. A family that
    forgot to declare its schedule would resolve to a plausible, wrong one, so
    the defaults are deliberately NOT Stable Diffusion's.
    """

    bare = EulerDiscrete()
    assert (bare.beta_schedule, bare.timestep_spacing, bare.steps_offset) == (
        "linear",
        "linspace",
        0,
    )
    for spec, block in ((SDXL, SDXL_SCHEDULER), (SD15, SD_SCHEDULER), (SD2, SD_SCHEDULER)):
        assert spec.scheduler is not None
        declared = EulerDiscrete.from_block(dict(block))
        assert declared.beta_schedule == "scaled_linear"
        assert declared.beta_start == 0.00085 and declared.beta_end == 0.012
        assert (declared.timestep_spacing, declared.steps_offset) == ("leading", 1)
        assert dict(spec.scheduler.parameters) == dict(block)


@pytest.mark.parametrize(
    "block",
    [
        {"timestep_spacing": "backwards"},
        {"beta_schedule": "cosine"},
        {"prediction_type": "flow_prediction"},
        {"final_sigmas_type": "sigma_max"},
        {"num_train_timesteps": 1.5},
        {"steps_offset": True},
    ],
)
def test_a_scheduler_block_is_parsed_not_coerced(block: dict[str, Any]) -> None:
    """Every one of these changes the LADDER, so none of them falls through."""

    with pytest.raises(ModelError):
        EulerDiscrete.from_block({**dict(SDXL_SCHEDULER), **block})


def test_a_v_prediction_checkpoint_carries_the_zero_snr_rescale_with_it() -> None:
    """``objective()`` reproduces the pairing ``gen_worker.view`` already makes.

    The two paths must not be able to disagree about what "v_prediction" means:
    the diffusers path sets ``rescale_betas_zero_snr=True`` whenever the
    stamped objective is v-pred, so this one does too, and it is the SCHEDULER
    that carries it rather than every caller remembering.
    """

    base = EulerDiscrete.from_block(dict(SDXL_SCHEDULER))
    assert base.prediction_type == "epsilon" and not base.rescale_betas_zero_snr
    v_pred = base.objective("v_prediction")
    assert v_pred.prediction_type == "v_prediction" and v_pred.rescale_betas_zero_snr
    assert base.objective("epsilon") is base
    assert v_pred.schedule(28).sigmas != base.schedule(28).sigmas
    with pytest.raises(ModelError):
        base.objective("sample")


def test_a_schedule_terminating_anywhere_but_zero_is_refused() -> None:
    with pytest.raises(ModelError):
        DiscreteSchedule(
            sigmas=(14.6, 0.1),
            timesteps=(999.0,),
            num_train_timesteps=1000,
            prediction_type="epsilon",
            init_noise_sigma=14.6,
        )
    with pytest.raises(ModelError):
        DiscreteSchedule(
            sigmas=(14.6, 1.0, 0.0),
            timesteps=(999.0,),
            num_train_timesteps=1000,
            prediction_type="epsilon",
            init_noise_sigma=14.6,
        )


# ------------------------------------------------------- the whole composition


def test_every_declarable_scheduler_has_math_and_the_gap_is_still_fenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IMPLEMENTED`` is TOTAL, and the absent-method mechanism still works.

    pgw#1331 proved the mechanism through SDXL, whose scheduler had no math.
    Both names carry math now, so nothing in the catalog exercises the gap and
    the mechanism has to be fenced at the GENERATOR instead — otherwise the
    next declared-but-unimplemented scheduler discovers it by shipping.
    """

    from gen_worker.model import codegen

    assert set(IMPLEMENTED) == set(SchedulerKind)
    assert Sdxl.SCHEDULER is SchedulerKind.EULER_DISCRETE
    assert Sd15.SCHEDULER is SchedulerKind.EULER_DISCRETE
    assert isinstance(Sdxl.fake().scheduler(), EulerDiscrete)

    rendered = codegen.render_module(
        Sdxl.EXPORT,
        spec_module="gen_worker.model.catalog.sdxl",
        spec_attr="SDXL",
    )
    assert "def scheduler(self)" in rendered
    monkeypatch.delitem(IMPLEMENTED, SchedulerKind.EULER_DISCRETE)
    without = codegen.render_module(
        Sdxl.EXPORT,
        spec_module="gen_worker.model.catalog.sdxl",
        spec_attr="SDXL",
    )
    assert "def scheduler(self)" not in without


def test_sdxl_declares_every_stage_of_the_pipeline() -> None:
    """Two text towers, the U-Net and the VAE decoder — nothing left eager."""

    assert tuple(runner.name for runner in SDXL.runners) == (
        "clip_g",
        "clip_l",
        "decoder",
        "denoiser",
    )
    assert SDXL.loop is not None
    assert tuple(stage.runner for stage in SDXL.loop.stages) == (
        "clip_l",
        "clip_g",
        "denoiser",
        "decoder",
    )
    # bigG returns BOTH of the things SDXL reads from it, from ONE pass.
    assert len(Sdxl.EXPORT.runner("clip_g").variants[0].outputs) == 2
    assert len(Sdxl.EXPORT.runner("clip_l").variants[0].outputs) == 1


def test_sd2_is_a_declaration_and_not_an_instance_of_sd15() -> None:
    """B2's scoping proposed an ``sd2`` INSTANCE. The shapes refuse it.

    An instance is weights + tuned values + a ref label; these two do not share
    a traced class or even a weight SHAPE. Every number below is a different
    cross-attention projection in the U-Net, so one compiled cell cannot serve
    both and one exhaustive ``Literal`` must not name both.
    """

    from gen_worker.model.catalog.sd15 import SD2_UNET, SD15_UNET

    assert SD15.name == "sd15" and SD2.name == "sd2"
    assert SD15_UNET["cross_attention_dim"] != SD2_UNET["cross_attention_dim"]
    assert SD15_UNET["attention_head_dim"] != SD2_UNET["attention_head_dim"]
    assert SD15_UNET["use_linear_projection"] != SD2_UNET["use_linear_projection"]
    # The runner SET is identical, which is exactly why the set is not the test.
    assert tuple(r.name for r in SD15.runners) == tuple(r.name for r in SD2.runners)
    assert Sd15.EXPORT_DIGEST != Sd2.EXPORT_DIGEST


def test_the_shape_axis_is_the_endpoints_own_bucket_grid() -> None:
    """Nine SDXL shapes and seven SD1.5 ones, non-square, packed onto one axis.

    Two axes would be a CROSS PRODUCT — 81 and 49 traced classes for 9 and 7
    real ones — so the pair is packed. The packing has to round-trip and the
    latent order has to be (rows, cols): 1344x768 and 768x1344 are different
    conv graphs, and transposing them picks a wrong class rather than making a
    wrong image.
    """

    assert len(sx.SHAPE_BUCKETS) == 9 and len(sd.SD15_SHAPE_BUCKETS) == 7
    for width, height in sx.SHAPES:
        code = sx.pack_shape(width, height)
        assert sx.unpack_shape(code) == (width, height)
        assert sx.latent_shape(code) == (height // 8, width // 8)
    assert sx.latent_shape(sx.pack_shape(1344, 768)) == (96, 168)
    assert sx.latent_shape(sx.pack_shape(768, 1344)) == (168, 96)
    # The bucket set the family declares IS the set the generated Literal has.
    assert set(SDXL.axis_values["shape"]) == set(sx.SHAPE_BUCKETS)
    # `sd15_serve`'s loop is annotated with `Sd15Shape` for BOTH families, so
    # the two grids being identical is a load-bearing fact and not a
    # coincidence: if one moves, this fails before the annotation lies.
    assert sd.SD15_SHAPE_BUCKETS == sd.SD2_SHAPE_BUCKETS
    assert set(SD15.axis_values["shape"]) == set(SD2.axis_values["shape"])


def test_the_tuned_schemas_carry_every_field_the_endpoints_stamp() -> None:
    """``inst.tuned`` replaces ``ctx.defaults`` BY VALUE or it loses recipes.

    A field this schema lacks is a stamped catalog value that stops reaching
    the handler silently, and a sampler name absent from the Literal makes an
    already-stamped recipe undecodable. Both were true of the first draft of
    ``SdxlTuned``: ``quality_preamble`` was missing and three of the endpoint's
    six sampler names were.
    """

    sdxl_fields = set(Sdxl.Tuned.__struct_fields__)
    assert {"scheduler", "steps", "guidance", "quality_preamble", "negative",
            "max_guidance"} <= sdxl_fields
    assert set(get_args(sx.SdxlSampler)) == {
        "euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras",
        "lcm", "euler_trailing", "ddim_trailing",
    }
    assert {"scheduler", "num_inference_steps", "guidance", "negative"} <= set(
        Sd15.Tuned.__struct_fields__
    )
    # SD2's schema is its OWN (th#1139), and the DEFAULTS are the difference
    # that matters: a Turbo recipe validated against SD1.5's vocabulary would
    # accept a 30-step CFG-7 stamping that destroys a one-step checkpoint.
    assert Sd15.Tuned().num_inference_steps == 30 and Sd15.Tuned().guidance == 7.0
    assert Sd2.Tuned().num_inference_steps == 1 and Sd2.Tuned().guidance == 0.0
    assert Sdxl.Tuned().steps == 28 and Sdxl.Tuned().scheduler == "euler_a"


# -------------------------------------------------------- real inference shape


def test_a_fake_backed_sdxl_generates_an_image_through_the_typed_callables() -> None:
    """The whole loop, hubless and cardless, through the real code path.

    Not a mock of the SDK — it IS the SDK with the one part that needs a card
    replaced. So this exercises both text towers, the CFG batching, the
    micro-conditioning block, the bit-exact schedule and four typed callables.
    """

    instance = Sdxl.fake()
    # `pack_shape` returns a plain int; the binding wants the closed Literal.
    # An endpoint reaches it through a `BucketMap` instead of a cast — this is
    # the one place a test has to spell the value itself.
    shape = cast("SdxlShape", sx.pack_shape(1024, 1024))
    seen: list[tuple[int, int]] = []
    image = sx.generate(
        instance,
        shape=shape,
        positive=sx.token_ids([1, 2, 3], device="cpu"),
        negative=sx.token_ids([], device="cpu"),
        steps=3,
        guidance=6.0,
        seed=7,
        on_step=lambda index, total: seen.append((index, total)),
    )
    assert seen == [(0, 3), (1, 3), (2, 3)]
    assert tuple(image.shape) == (1, 3, 1024, 1024)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    again = sx.generate(
        instance,
        shape=shape,
        positive=sx.token_ids([1, 2, 3], device="cpu"),
        negative=sx.token_ids([], device="cpu"),
        steps=3,
        guidance=6.0,
        seed=7,
    )
    assert torch.equal(image, again)


@pytest.mark.parametrize(
    ("model", "width", "height"),
    [(Sd15, 512, 512), (Sd15, 768, 512), (Sd2, 512, 512)],
)
def test_a_fake_backed_sd_generates_an_image_through_the_typed_callables(
    model: Any, width: int, height: int
) -> None:
    """Both U-Net declarations, including the non-square bucket.

    768x512 is here on purpose: it is the shape the packed axis exists for, and
    a transposed latent would produce a (1, 3, 512, 768) image that reads as
    "close enough" in a shape assertion written loosely.
    """

    instance = model.fake()
    shape = cast("sd.AnyShape", sd.pack_shape(width, height))
    image = sd.generate(
        instance,
        shape=shape,
        positive=sd.token_ids([1, 2, 3], device="cpu"),
        negative=sd.token_ids([], device="cpu"),
        steps=2,
        guidance=7.0,
        seed=3,
    )
    assert tuple(image.shape) == (1, 3, height, width)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0


def test_the_initial_latents_start_at_the_ladders_own_scale() -> None:
    """A variance-EXPLODING schedule does not start at unit variance.

    ``leading`` spacing makes ``init_noise_sigma`` ``sqrt(sigma_max**2 + 1)``.
    Starting at 1.0 instead is the classic silent way to render a washed-out
    image — no error, just a wrong picture — so the scale is asserted.

    ~10.8 and not the ~14.6 of the full trained table, because ``leading``
    starts a whole step BELOW the top of the ladder (its highest timestep at 28
    steps is 946, not 999). That gap is exactly the difference the three
    spacings exist to express, and it is why a distilled recipe naming one is
    destroyed by another.
    """

    schedule = EulerDiscrete.from_block(dict(SDXL_SCHEDULER)).schedule(28)
    assert schedule.timesteps[0] == 946.0
    assert 10.0 < schedule.init_noise_sigma < 11.0
    latents = sx.initial_latents(
        shape=sx.pack_shape(1024, 1024),
        seed=1,
        device="cpu",
        dtype=torch.float32,
        sigma=schedule.init_noise_sigma,
    )
    assert tuple(latents.shape) == (1, 4, 128, 128)
    assert float(latents.std()) > 10.0
    other = sx.initial_latents(
        shape=sx.pack_shape(1024, 1024),
        seed=2,
        device="cpu",
        dtype=torch.float32,
        sigma=schedule.init_noise_sigma,
    )
    assert not torch.equal(latents, other)


def test_the_ie740_serving_floors_migrated_by_value() -> None:
    """The PARSED NUMBERS, because "by value" is a claim about numbers.

    Both are production incidents rather than margins. sdxl's ``vram12g`` is
    the value ie#704's stopgap block itself calls "the honest SERVING floor"
    and names as its revert target, after one overloaded scalar
    over-constrained serving to fix mint placement; its ``sm89+`` is the
    DECODABILITY floor for the rowwise handle, not the fastest one. sd15's
    ``vram6g`` is what both of its ``@endpoint``s declare.

    Also asserted: the floors do NOT ride the export digest. A serving floor is
    a fact about the MACHINE and a cell's identity is graph x sm x toolchain,
    so restating a floor must not re-key every artifact — which is why
    ``check_model_bindings.py`` stays byte-clean across this change.
    """

    sdxl = SDXL.layout_requirements
    assert sdxl["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert sdxl["plain.bf16@1"].minimum.min_vram_gb == 12.0
    assert dict(SDXL.layouts or {}) == {"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")}

    for spec in (SD15, SD2):
        assert spec.layout_requirements["plain.bf16@1"].minimum.min_vram_gb == 6.0
        assert dict(spec.layouts or {}) == {"*": ("plain.bf16@1",)}

    # The floor is declaration-side only; the committed export does not carry it.
    assert "layout_requirements" not in Sdxl.EXPORT.dumps()


# ----------------------------------------------- what the compiled gauntlet needs


def test_a_mint_can_be_narrowed_to_one_shape_so_a_gauntlet_row_is_affordable() -> None:
    """``--bucket shape=…``, and why it had to exist before the sm_86 row runs.

    B2 grew SDXL from 2 buckets x 2 runners (4 classes) to 9 buckets x 2
    shape-bearing runners + 2 towers (**20 classes**). #1348's row 14 mints
    sdxl on an rtx-a4500 and adopts on an rtx-a4000; at 20 classes it pays 5x
    what it used to for 19 classes it never adopts. Narrowing to the one shape
    the row's workload renders makes it 3 — denoiser, decoder, and the towers
    that declare no shape axis at all and must NOT be dropped by a shape
    filter, which is the half of this that is easy to get wrong.
    """

    from gen_worker.model.mint import variants_of

    everything = variants_of(SDXL)
    assert len(everything) == 20

    one_shape = variants_of(SDXL, buckets={"shape": [sx.pack_shape(1024, 1024)]})
    assert {runner.name for runner, _, _ in one_shape} == {
        "clip_g", "clip_l", "decoder", "denoiser"
    }
    assert len(one_shape) == 4

    just_the_unet = variants_of(
        SDXL, only=("denoiser",), buckets={"shape": [sx.pack_shape(1024, 1024)]}
    )
    assert len(just_the_unet) == 1
    assert just_the_unet[0][1] == {"shape": sx.pack_shape(1024, 1024)}

    # An axis or a value the family does not declare is REFUSED, not silently
    # empty: a typo'd `--bucket` that minted zero classes would report a green
    # row that compiled nothing, which is the rig's own most expensive lesson.
    with pytest.raises(ModelError):
        variants_of(SDXL, buckets={"resolution": [1024]})
    with pytest.raises(ModelError):
        variants_of(SDXL, buckets={"shape": [5120512]})


# ------------------------------------------------------------ the ddim decision


def test_ddim_is_reached_by_the_endpoints_and_is_deliberately_not_declared() -> None:
    """pgw#1346 owed a ``ddim`` decision. It is: REACHED, and NOT DECLARABLE.

    The enumeration, from the endpoints themselves rather than from intent:

    * ``sd15``'s ``SD15Scheduler`` payload enum offers ``"ddim"``;
    * ``sd15``'s ``generate_hyper`` handler pins ``"ddim_trailing"``
      UNCONDITIONALLY — a function reaches DDIM with no payload involved;
    * ``sdxl``'s ``SdxlScheduler`` offers ``"ddim_trailing"`` too.

    So the honest answer to "does an endpoint function reach ddim" is YES. It
    is still not implemented in this batch, and the reason is structural rather
    than budgetary: ``GraphModelSpec.scheduler`` is ONE ``Scheduler`` block and
    codegen emits ONE ``scheduler()`` method, so a second implemented kind
    would be a class no declaration in the catalog could attach to. Declaring
    classes nothing can select is the exact thing the catalog refused to do
    with SDXL's text encoders one release ago.

    **The blocker this exposes (pgw#1346 K10): the sampler is a TUNED value,
    not a family constant.** ``SdxlTuned.scheduler`` and ``Sd15Tuned.scheduler``
    already carry it per checkpoint, and the six/eight names they admit map
    onto four different diffusers scheduler CLASSES. The declaration therefore
    needs a scheduler SET keyed by tuned name — ``inst.scheduler()`` taking
    ``inst.tuned.scheduler`` — before ddim, dpmpp or lcm can be implemented at
    all. Until then the family's declared block is its TRAINED schedule, which
    is the right single value if only one is allowed.
    """

    assert "ddim" not in {kind.value for kind in SchedulerKind}
    with pytest.raises(ModelError):
        parse_kind("ddim")
    with pytest.raises(ModelError):
        parse_kind("ddim_trailing")

    # The names ARE in the tuned vocabularies, which is what makes this a
    # deferral with a live consequence rather than a name nobody uses.
    assert "ddim_trailing" in get_args(sx.SdxlSampler)
    assert {"ddim", "ddim_trailing"} <= set(get_args(sd.Sd15Sampler))
