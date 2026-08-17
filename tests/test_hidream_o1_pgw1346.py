"""pgw#1346 B3b — HiDream-O1 is declared, and its flash ladder is real math.

Four claims, each against the thing it is a claim ABOUT:

1. **The tuned schema is the endpoint's AND the hub's, by value and by name.**
   tensorhub already publishes ``hidream-o1.schema.json``; every property in it
   is asserted against this schema, because a re-spelled field would make every
   stamped catalog row undecodable — pgw#1346 B2's ``SdxlTuned`` defect class.
2. **The sampler is a TUNED value and a PAYLOAD count**, which is pgw#1346 K10
   at its sharpest: three schedulers behind one declaration slot. The branch is
   asserted as a function rather than described.
3. **The flash ladder is implemented and reproducible.** Twenty-eight authored
   timesteps, an exact sigma mapping, a linear noise ramp, and a re-noising step
   — measured at B2's instrument, which here means asserting exactness rather
   than a tolerance, because there is no torch-derived reference to bound.
4. **The eager tier's two open gaps are asserted, not narrated**: no binding
   exists for an eager model (K11), and the hub handle this model is known by
   cannot be spelled as a ``ModelSpec`` name (K11b).
"""

from __future__ import annotations

import msgspec
import pytest

from gen_worker.families import family_for
from gen_worker.model.catalog import hidream_o1_serve as hd
from gen_worker.model.catalog.hidream_o1 import (
    HIDREAM_O1,
    HUB_FAMILY,
    MAX_REFERENCE_IMAGES,
    PRESETS,
    TRANSFORMER,
    VISION_TOWER,
)
from gen_worker.model.scheduler_hidream import (
    FLASH_TIMESTEPS,
    NUM_TRAIN_TIMESTEPS,
    HiDreamO1Flash,
)
from gen_worker.model.spec import GraphModelSpec

torch = pytest.importorskip("torch")

#: tensorhub's `hidream-o1.schema.json` properties and defaults, transcribed
#: HERE so this file compares the declaration against the CONTRACT rather than
#: against itself. Migration 0046 renamed `steps` -> `num_inference_steps` in
#: the schema AND in every stored row.
HUB_SCHEMA_DEFAULTS: dict[str, object] = {
    "model_type": "dev",
    "num_inference_steps": 28,
    "cfg_scale": 1.0,
    "shift": 1.0,
    "noise_scale": 7.5,
    "noise_clip_std": 2.5,
    "max_cfg": None,
}

#: The endpoint's three published recipes, transcribed from its own test
#: fixtures. They are CATALOG values, not code — asserted here only to show the
#: schema decodes each of them, which is what "migrated by value" has to mean.
ENDPOINT_RECIPES: dict[str, dict[str, object]] = {
    "dev": {"model_type": "dev", "num_inference_steps": 28, "cfg_scale": 1.0,
            "shift": 1.0, "noise_scale": 7.5, "noise_clip_std": 2.5, "max_cfg": 1.0},
    "dev_2604": {"model_type": "dev", "num_inference_steps": 28, "cfg_scale": 1.0,
                 "shift": 1.0, "noise_scale": 8.0, "noise_clip_std": 8.0, "max_cfg": 1.0},
    "full": {"model_type": "full", "num_inference_steps": 50, "cfg_scale": 5.0,
             "shift": 3.0, "noise_scale": 8.0, "noise_clip_std": 2.5, "max_cfg": None},
}

#: DiffSynth's ``set_timesteps_hidream_o1_image_dev`` table, transcribed HERE
#: from upstream. The REFERENCE: without it the ladder tests below would derive
#: the sigmas from the same tuple they are checking, and a one-sided edit to
#: either copy would pass. ``diffsynth`` is not a gen-worker dependency — one of
#: the reasons this model is eager-tiered — so transcription is the only
#: available reference, exactly as it is for Anima's schedule.
UPSTREAM_FLASH_TIMESTEPS: tuple[int, ...] = (
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
)

#: The endpoint's `_O1_ASPECTS` grid, transcribed as (width, height).
ENDPOINT_PRESETS: tuple[tuple[int, int], ...] = (
    (2048, 2048), (2304, 1728), (1728, 2304), (2560, 1440), (1440, 2560),
    (2496, 1664), (1664, 2496), (3104, 1312), (1312, 3104),
    (2304, 1792), (1792, 2304),
)


# ------------------------------------------------------------- the tuned schema


def test_the_tuned_schema_is_the_hubs_published_contract() -> None:
    """Field for field against ``hidream-o1.schema.json``, names included."""

    fields = {row.name: row for row in msgspec.structs.fields(hd.HiDreamO1Tuned)}
    assert set(fields) - {"schema_version"} == set(HUB_SCHEMA_DEFAULTS)
    neutral = hd.HiDreamO1Tuned()
    for name, default in HUB_SCHEMA_DEFAULTS.items():
        assert getattr(neutral, name) == default, name
    # The rename migration 0046 performed, asserted from this side too.
    assert "steps" not in fields


@pytest.mark.parametrize("name", sorted(ENDPOINT_RECIPES))
def test_every_published_recipe_decodes_against_the_schema(name: str) -> None:
    """A schema that cannot decode a stamped row is a migration that lost data."""

    recipe = ENDPOINT_RECIPES[name]
    decoded = msgspec.convert(recipe, hd.HiDreamO1Tuned)
    for field, value in recipe.items():
        assert getattr(decoded, field) == value, field


def test_the_tuned_schema_is_published_under_the_declarations_name() -> None:
    assert family_for("hidream_o1") is hd.HiDreamO1Tuned
    assert HIDREAM_O1.tuned is hd.HiDreamO1Tuned
    #: No LoRA vocabulary: this model has no adapter lane at all.
    assert HIDREAM_O1.lora_tuned is None


def test_the_hub_handle_cannot_be_spelled_as_a_model_name() -> None:
    """pgw#1346 K11b — a real divergence, asserted so it stays visible.

    ``ModelSpec.name`` is a GENERATED-SYMBOL identifier and a symbol cannot
    carry a hyphen. tensorhub knows this family as ``hidream-o1`` and publishes
    a schema under that name; ``ModelSpec._register`` publishes under
    ``hidream_o1``. Until they are reconciled, the hub's schema is fed by the
    endpoint's ``@family`` and the declaration's is a second, unrelated entry.
    """

    assert HUB_FAMILY == "hidream-o1"
    assert HIDREAM_O1.name == "hidream_o1"
    assert HIDREAM_O1.name != HUB_FAMILY
    assert "-" not in HIDREAM_O1.name
    assert family_for(HUB_FAMILY) is None


# ------------------------------------------------------------ the ie#740 floors


def test_the_ie740_serving_floors_are_preserved_by_value() -> None:
    assert HIDREAM_O1.layouts == {"*": ("cozy.fp8-rowwise@1", "plain.bf16@1")}
    requirements = HIDREAM_O1.layout_requirements
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_sm == 89
    assert requirements["plain.bf16@1"].minimum.min_vram_gb == 22.0
    assert requirements["cozy.fp8-rowwise@1"].minimum.min_vram_gb == 0.0
    assert requirements["plain.bf16@1"].minimum.min_sm == 0


# ------------------------------------------------------- the pixel-space shapes


def test_there_is_no_vae_so_the_stride_is_the_patch_size() -> None:
    """The fact most likely to be assumed away: O1 eats PIXELS.

    Three channels, no latent space, no scale factor. A reader carrying the
    rest of this catalog's assumptions will look for a VAE and, not finding
    one, conclude it was forgotten.
    """

    assert hd.IMAGE_CHANNELS == 3 == TRANSFORMER["in_channels"]
    assert hd.PATCH_SIZE == 32 == TRANSFORMER["patch_size"]
    assert hd.image_tokens(2048, 2048) == (2048 // 32) ** 2 == 4096


@pytest.mark.parametrize(("width", "height"), ENDPOINT_PRESETS)
def test_every_preset_is_a_whole_number_of_patches(width: int, height: int) -> None:
    assert width % hd.PATCH_SIZE == 0 and height % hd.PATCH_SIZE == 0
    assert hd.image_tokens(width, height) > 0


def test_the_declared_presets_are_the_endpoints_trained_resolutions() -> None:
    assert PRESETS == ENDPOINT_PRESETS
    assert len(PRESETS) == 11


def test_the_sequence_length_has_two_free_terms_and_that_is_the_shape_story() -> None:
    """Why no compile block exists, as arithmetic rather than as prose.

    The prompt term is unbounded by the model (no padding, no truncation) and
    the reference term is a multiset over up to eleven count-sized images. Two
    free terms in one sequence is not a bucket axis.
    """

    base = hd.sequence_length(2048, 2048, prompt_tokens=0)
    assert base == 4096 + hd.TIMESTEP_TOKENS
    # The same target size, two different sequences, from the text alone.
    assert hd.sequence_length(2048, 2048, prompt_tokens=17) == base + 17
    assert hd.sequence_length(2048, 2048, prompt_tokens=512) == base + 512
    # And again from the references alone.
    assert hd.sequence_length(2048, 2048, prompt_tokens=8, reference_tokens=900) == (
        base + 8 + 900
    )


def test_references_shrink_as_they_multiply() -> None:
    """The upstream budget ladder, carried by value.

    A reader estimating cost from "eleven images" without it is wrong by a large
    factor: the eleventh reference is rendered at a quarter of the longest edge.
    """

    longest = 2048
    assert hd.reference_edge(longest, longest, count=1) == 2048
    assert hd.reference_edge(longest, longest, count=2) == 1536
    assert hd.reference_edge(longest, longest, count=4) == 1024
    assert hd.reference_edge(longest, longest, count=8) == 768
    assert hd.reference_edge(longest, longest, count=MAX_REFERENCE_IMAGES) == 512
    with pytest.raises(ValueError):
        hd.reference_edge(longest, longest, count=0)


def test_the_vision_towers_patch_is_not_the_diffusion_patch() -> None:
    """16 and 32, two numbers doing two jobs in one model."""

    assert VISION_TOWER["patch_size"] == 16
    assert TRANSFORMER["patch_size"] == 32
    assert VISION_TOWER["out_hidden_size"] == TRANSFORMER["hidden_size"] == 4096


def test_the_attention_is_grouped_query_which_is_a_serving_hazard() -> None:
    """32 query heads over 8 key/value heads — recorded because the endpoint
    fences two attention backends out of CI for exactly this ratio."""

    assert TRANSFORMER["num_attention_heads"] == 32
    assert TRANSFORMER["num_key_value_heads"] == 8
    assert TRANSFORMER["head_dim"] * TRANSFORMER["num_attention_heads"] == (
        TRANSFORMER["hidden_size"]
    )
    assert sum(TRANSFORMER["mrope_section"]) == TRANSFORMER["head_dim"] // 2


# ------------------------------------------------------------- K10's live shape


@pytest.mark.parametrize(
    ("model_type", "references", "expected"),
    [
        ("dev", 0, hd.HiDreamO1Sampler.FLASH),
        ("dev", 1, hd.HiDreamO1Sampler.FLOW_MATCH),
        ("dev", 2, hd.HiDreamO1Sampler.FLASH),
        ("dev", 11, hd.HiDreamO1Sampler.FLASH),
        ("full", 0, hd.HiDreamO1Sampler.UNIPC),
        ("full", 1, hd.HiDreamO1Sampler.UNIPC),
        ("full", 11, hd.HiDreamO1Sampler.UNIPC),
    ],
)
def test_the_sampler_is_chosen_from_a_tuned_value_and_a_payload_count(
    model_type: str, references: int, expected: hd.HiDreamO1Sampler
) -> None:
    """pgw#1346 K10, at its sharpest: THREE samplers behind one slot.

    ``GraphModelSpec.scheduler`` is a single ``Scheduler`` and codegen emits one
    ``scheduler()`` method, so declaring any one of these would be declaring the
    wrong one on most requests. K10's set surface is deliberately untouched
    here; this function is the shape it has to grow to hold.
    """

    tuned = hd.HiDreamO1Tuned(model_type=model_type)
    assert hd.sampler_for(tuned, reference_images=references) is expected


def test_the_reachable_sampler_set_is_larger_than_the_declarable_one() -> None:
    """The difference between what a model REACHES and what it may DECLARE."""

    from gen_worker.model.scheduler import IMPLEMENTED, SchedulerKind

    reachable = {member.value for member in hd.HiDreamO1Sampler}
    assert reachable == {"flash", "flow_match", "unipc"}
    declarable = {kind.value for kind in SchedulerKind}
    assert reachable & declarable == set()
    assert set(IMPLEMENTED) == set(SchedulerKind)


# ------------------------------------------------------------- the flash ladder


def test_the_flash_ladder_is_the_upstream_table_entry_for_entry() -> None:
    """Against the transcribed reference, not against itself.

    The ladder is AUTHORED — no formula derives it — so the only thing that can
    check it is a second copy of the authored values. Every other assertion in
    this file about the ladder derives from ``FLASH_TIMESTEPS``, which means
    this is the one test that would catch a mistyped entry.
    """

    assert FLASH_TIMESTEPS == UPSTREAM_FLASH_TIMESTEPS


def test_the_flash_ladder_is_an_authored_table_of_twenty_eight_timesteps() -> None:
    assert len(FLASH_TIMESTEPS) == 28
    assert FLASH_TIMESTEPS[0] == 999 and FLASH_TIMESTEPS[-1] == 8
    assert all(
        later < earlier
        for earlier, later in zip(FLASH_TIMESTEPS, FLASH_TIMESTEPS[1:], strict=False)
    )
    assert all(isinstance(value, int) for value in FLASH_TIMESTEPS)


def test_the_sigmas_are_the_timesteps_over_a_thousand_exactly() -> None:
    """B2's instrument, and here it is exactness rather than a tolerance.

    There is no torch-derived reference to bound: the table is integers and the
    mapping is one IEEE double division, so this ladder is identical on every
    machine, ISA and torch build. That is the same inversion B2 landed for
    ``EulerDiscrete`` — our ladder is reproducible where the reference was not.
    """

    schedule = HiDreamO1Flash().schedule()
    assert len(schedule) == 28
    assert schedule.sigmas[-1] == 0.0
    for index, timestep in enumerate(FLASH_TIMESTEPS):
        assert schedule.sigmas[index] == timestep / NUM_TRAIN_TIMESTEPS
    # And the model-unit timesteps come back out of the schedule unchanged.
    assert schedule.timesteps == tuple(float(t) for t in FLASH_TIMESTEPS)


def test_the_ladder_refuses_a_step_count_it_cannot_honour() -> None:
    """Answering a 4-step request with 28 steps silently is how a request's
    declared cost stops describing its work. The upstream sampler accepts the
    count and ignores it; this one says so."""

    flash = HiDreamO1Flash()
    assert len(flash.schedule(28)) == 28
    for steps in (1, 4, 27, 29, 50):
        with pytest.raises(Exception, match="fixed 28-entry table"):
            flash.schedule(steps)


def test_the_noise_scale_is_a_linear_ramp_and_the_endpoint_flattens_it() -> None:
    """Two fields kept distinct because upstream's are, even though the
    endpoint passes the same resolved ``noise_scale`` for both."""

    flat = HiDreamO1Flash(noise_scale_start=7.5, noise_scale_end=7.5)
    assert flat.noise_scales == (7.5,) * 28
    ramp = HiDreamO1Flash(noise_scale_start=0.0, noise_scale_end=27.0)
    assert ramp.noise_scales[0] == 0.0
    assert ramp.noise_scales[-1] == 27.0
    assert ramp.noise_scales[1] == pytest.approx(1.0)
    # The dev-2604 recipe's own amplitude, reached through the tuned schema.
    tuned = msgspec.convert(ENDPOINT_RECIPES["dev_2604"], hd.HiDreamO1Tuned)
    lane = HiDreamO1Flash(
        noise_scale_start=tuned.noise_scale,
        noise_scale_end=tuned.noise_scale,
        noise_clip_std=tuned.noise_clip_std,
    )
    assert lane.noise_scales[0] == 8.0 and lane.noise_clip_std == 8.0


def test_the_noise_clip_is_measured_from_the_tensor_not_assumed() -> None:
    """A bf16 draw's realised deviation is not exactly one, so the bound has to
    come from the tensor. Zero disables the clip, upstream's own convention."""

    generator = torch.Generator(device="cpu").manual_seed(11)
    noise = torch.randn(4096, generator=generator) * 3.0
    flash = HiDreamO1Flash(noise_clip_std=2.5)
    clipped = flash.clip_noise(noise)
    bound = 2.5 * float(noise.std())
    assert float(clipped.abs().max()) <= bound + 1e-6
    assert float(noise.abs().max()) > bound
    assert HiDreamO1Flash(noise_clip_std=0.0).clip_noise(noise) is noise


def test_the_flash_step_is_euler_to_x0_and_then_a_re_noise() -> None:
    """The departure from flow-match Euler, asserted against its own algebra.

    ``denoised = x - v*sigma`` then ``x = sigma_next*noise*scale + (1-sigma_next)*denoised``.
    Stochastic by construction, which is why the noise is a caller-owned input:
    a scheduler reaching for a global RNG makes a receipt's seed meaningless.
    """

    flash = HiDreamO1Flash(noise_scale_start=2.0, noise_scale_end=2.0, noise_clip_std=0.0)
    schedule = flash.schedule()
    sample = torch.full((8,), 0.5)
    velocity = torch.full((8,), 0.25)
    noise = torch.full((8,), 0.125)
    index = 3
    sigma = schedule.sigmas[index]
    sigma_next = schedule.sigmas[index + 1]
    expected = sigma_next * (2.0 * noise) + (1.0 - sigma_next) * (sample - sigma * velocity)
    assert torch.allclose(flash.step(schedule, index, velocity, sample, noise), expected)


def test_the_flash_step_refuses_an_out_of_range_index() -> None:
    flash = HiDreamO1Flash()
    schedule = flash.schedule()
    tensor = torch.zeros(4)
    for index in (-1, 28, 99):
        with pytest.raises(Exception, match="outside this schedule"):
            flash.step(schedule, index, tensor, tensor, tensor)


def test_the_flash_step_is_deterministic_for_a_given_noise() -> None:
    """Same inputs, same bytes — the property that makes a seed portable."""

    flash = HiDreamO1Flash()
    schedule = flash.schedule()
    generator = torch.Generator(device="cpu").manual_seed(3)
    noise = torch.randn(64, generator=generator)
    sample = torch.linspace(-1.0, 1.0, 64)
    velocity = torch.linspace(1.0, -1.0, 64)
    first = flash.step(schedule, 0, velocity, sample, noise)
    second = flash.step(schedule, 0, velocity, sample, noise)
    assert torch.equal(first, second)


def test_the_flash_sampler_refuses_a_nonsense_block() -> None:
    for kwargs in (
        {"noise_scale_start": float("nan")},
        {"noise_scale_end": float("inf")},
        {"noise_clip_std": -1.0},
    ):
        with pytest.raises(Exception, match="finite non-negative"):
            HiDreamO1Flash(**kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ K11, stated


def test_the_eager_tier_is_a_declaration_with_no_binding() -> None:
    """pgw#1346 K11, the same gap Anima's suite asserts, from the other family.

    ``ModelExport`` refuses a runner-less document, so no ``Model`` subclass
    exists for an eager model and the endpoint's fourteen ``ctx.defaults`` reads
    have nowhere to migrate to yet. This gates all of B5 as well.
    """

    from gen_worker.model.catalog import _FAMILIES

    assert HIDREAM_O1.runners == ()
    assert not isinstance(HIDREAM_O1, GraphModelSpec)
    assert not any(name.startswith("HiDream") for name in _FAMILIES)
