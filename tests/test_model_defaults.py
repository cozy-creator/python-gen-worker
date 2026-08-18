"""The pgw#1377 read-side decode + Knob resolution, exercised end to end.

The decode matrix (absent / partial / full / ill-typed / unclassified /
mistyped), the narrowest-range knob merge, the evolution rule in both
directions, and clamp caller-visibility through the REAL
``RequestContext.clamp`` ledger — no mock context.
"""

from __future__ import annotations

import warnings

import msgspec
import pytest

from gen_worker.models import (
    LORA_OVERLAYS,
    MODEL_TYPES,
    CheckpointDefaultsUnclassified,
    DefaultsDecodeError,
    Knob,
    ModelTypeMismatch,
    SD15,
    SDXL,
    decode_defaults,
    decode_model_defaults,
    defaults_vocabularies,
    model_type_by_name,
    model_type_for_contract,
)
from gen_worker.families import GenerationDefaults
from gen_worker.request_context import RequestContext


# ── platform values (zero-arg = the platform opinion, servable) ──────────────


def test_sdxl_zero_arg_is_the_platform_opinion() -> None:
    d = SDXL.Defaults()
    assert d.steps == Knob(28, lo=1, hi=80, name="steps")
    assert d.guidance == Knob(6.0, lo=1.5, hi=15.0, name="guidance")
    assert d.cfg is True
    # Paul's ruling: the quality vocabulary lives HERE, not in endpoint code.
    assert d.positive_preamble == "masterpiece, best quality"
    assert d.negative_preamble == "worst quality, low quality"
    # scheduler None = trust the tree's shipped scheduler config (layer 3).
    assert d.scheduler is None
    assert d.timesteps == ()
    # Checkpoint-level fact, decoupled from cfg (the guidance axis).
    assert d.step_distilled is False


def test_sdxl_lora_zero_arg_is_lightning_shaped() -> None:
    d = SDXL.Lora.Defaults()
    assert d.cfg is False
    assert d.scheduler == "euler_trailing"
    assert d.steps.default == 4
    assert d.timesteps == ()
    assert d.strength == Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    # Inert while cfg=False; the base platform knob, so a row that flips cfg
    # on without narrowing still serves sanely.
    assert d.guidance == SDXL.Defaults().guidance
    # Both defaults types ARE the one nominal recipe type.
    assert isinstance(d, SDXL.Recipe)
    assert isinstance(SDXL.Defaults(), SDXL.Recipe)


def test_every_zero_arg_defaults_is_servable() -> None:
    """Zero-arg constructions double as trace fixtures (pgw#1377 point 7):
    every knob default must sit inside its own [lo, hi]."""
    for name, cls in defaults_vocabularies().items():
        d = cls()
        for f in msgspec.structs.fields(cls):
            knob = getattr(d, f.name)
            if not isinstance(knob, Knob):
                continue
            if knob.lo is not None:
                assert knob.default >= knob.lo, f"{name}.{f.name}"
            if knob.hi is not None:
                assert knob.default <= knob.hi, f"{name}.{f.name}"


def test_every_knob_names_its_own_field() -> None:
    """The clamp ledger names fields through ``Knob.name`` — a drifted name
    would mislabel every adjustment row."""
    for wire_name, cls in defaults_vocabularies().items():
        for f in msgspec.structs.fields(cls):
            if isinstance(f.default, Knob):
                assert f.default.name == f.encode_name, f"{wire_name}.{f.encode_name}"


# ── the decode matrix ────────────────────────────────────────────────────────


def test_absent_row_decodes_to_platform_values() -> None:
    assert decode_model_defaults(SDXL, model="sdxl", defaults=None) == SDXL.Defaults()
    assert decode_model_defaults(SDXL, model="sdxl", defaults={}) == SDXL.Defaults()


def test_partial_row_overlays_field_by_field() -> None:
    d = decode_model_defaults(
        SDXL,
        model="sdxl",
        defaults={"cfg": False, "steps": {"default": 8}},
    )
    assert d.cfg is False
    assert d.steps.default == 8
    # Untouched halves of a knob keep the platform range.
    assert (d.steps.lo, d.steps.hi) == (1, 80)
    # Untouched fields keep the platform values.
    assert d.guidance == SDXL.Defaults().guidance
    assert d.positive_preamble == "masterpiece, best quality"


def test_full_row_overrides_every_field() -> None:
    row = {
        "steps": {"default": 6, "lo": 4, "hi": 10},
        "guidance": {"default": 2.0, "lo": 1.5, "hi": 3.0},
        "cfg": False,
        "positive_preamble": "",
        "negative_preamble": "",
        "scheduler": "lcm",
        "timesteps": [999, 749, 499, 249],
    }
    d = decode_model_defaults(SDXL, model="sdxl", defaults=row)
    assert d.steps == Knob(6, lo=4, hi=10, name="steps")
    assert d.guidance == Knob(2.0, lo=1.5, hi=3.0, name="guidance")
    assert d.cfg is False
    assert d.positive_preamble == ""
    assert d.scheduler == "lcm"
    assert d.timesteps == (999, 749, 499, 249)


def test_knob_ranges_merge_to_the_narrowest_layer() -> None:
    d = decode_model_defaults(
        SDXL,
        model="sdxl",
        defaults={"guidance": {"default": 5.0, "lo": 1.0, "hi": 9.0}},
    )
    # The row narrows hi (9.0 < 15.0) but cannot widen lo (1.0 < 1.5 loses).
    assert (d.guidance.lo, d.guidance.hi) == (1.5, 9.0)
    assert d.guidance.default == 5.0


def test_a_row_default_outside_the_merged_range_is_pulled_inside() -> None:
    # resolve(None) returns the default UNCLAMPED, so decode keeps the
    # struct servable by construction.
    d = decode_model_defaults(
        SDXL, model="sdxl", defaults={"guidance": {"default": 20.0}}
    )
    assert d.guidance.default == 15.0


@pytest.mark.parametrize(
    ("row", "field"),
    [
        ({"steps": "fast"}, "steps"),
        ({"steps": {"default": 8.5}}, "steps"),
        ({"cfg": "yes"}, "cfg"),
        ({"scheduler": "ddim_trailing"}, "scheduler"),
        ({"timesteps": ["a"]}, "timesteps"),
    ],
)
def test_ill_typed_rows_are_typed_refusals_naming_the_field(
    row: dict[str, object], field: str
) -> None:
    with pytest.raises(DefaultsDecodeError) as caught:
        decode_model_defaults(SDXL, model="sdxl", defaults=row)
    assert caught.value.field == field
    assert field in str(caught.value)
    assert caught.value.model == "sdxl"


def test_unclassified_serves_platform_fallbacks_with_the_named_warning() -> None:
    with pytest.warns(CheckpointDefaultsUnclassified) as caught:
        d = decode_model_defaults(SDXL, model=None, defaults={"steps": {"default": 5}})
    # Fallbacks, never a silent guess: the untyped row is not decoded.
    assert d == SDXL.Defaults()
    assert CheckpointDefaultsUnclassified.code == "checkpoint_defaults_unclassified"
    assert "unclassified" in str(caught[0].message)


def test_a_mistyped_checkpoint_is_a_typed_refusal_not_a_warning() -> None:
    with pytest.raises(ModelTypeMismatch) as caught:
        decode_model_defaults(SDXL, model="sd15", defaults=None)
    assert (caught.value.expected, caught.value.actual) == ("sdxl", "sd15")


def test_a_base_mismatched_adapter_is_refused_at_bind() -> None:
    """pgw#1377 acceptance (f): an sd15.lora row where sdxl.lora is expected."""
    with pytest.raises(ModelTypeMismatch):
        decode_model_defaults(SDXL.Lora, model="sd15.lora", defaults=None)


def test_a_lora_row_decodes_through_the_adapter_surface() -> None:
    d = decode_model_defaults(
        SDXL.Lora,
        model="sdxl.lora",
        defaults={
            "trigger_words": ["dmd2"],
            "strength": {"default": 0.8},
            "steps": {"default": 4},
            "scheduler": "lcm",
            "timesteps": [999, 749, 499, 249],
        },
    )
    assert d.trigger_words == ("dmd2",)
    assert d.strength.default == 0.8
    assert (d.strength.lo, d.strength.hi) == (-4.0, 4.0)
    assert d.scheduler == "lcm"
    assert d.timesteps == (999, 749, 499, 249)


# ── the evolution rule, both directions (pgw#1377 acceptance c) ──────────────


class _V1Defaults(msgspec.Struct, frozen=True):
    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")


class _V2Defaults(msgspec.Struct, frozen=True):
    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    shine: float = 0.5  # the additive v2 field


def test_a_v1_row_decodes_under_a_v2_struct_on_fallbacks() -> None:
    v1_row = {"steps": {"default": 12}}
    d = decode_defaults(_V2Defaults, v1_row)
    assert d.steps.default == 12
    assert d.shine == 0.5


def test_a_v2_row_decodes_under_a_v1_struct_ignoring_the_unknown_field() -> None:
    v2_row = {"steps": {"default": 12}, "shine": 0.9}
    d = decode_defaults(_V1Defaults, v2_row)
    assert d.steps.default == 12
    assert not hasattr(d, "shine")


def test_unknown_knob_keys_are_ignored_too() -> None:
    d = decode_defaults(_V1Defaults, {"steps": {"default": 12, "name": "bogus", "extra": 1}})
    # The decode restamps the field name; wire input never names knobs.
    assert d.steps.name == "steps"
    assert d.steps.default == 12


# ── Knob.resolve through the real RequestContext ledger ──────────────────────


def test_resolve_none_returns_the_checkpoint_default_with_no_adjustment() -> None:
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-1")
    d = decode_model_defaults(SDXL, model="sdxl", defaults={"steps": {"default": 20}})
    assert d.steps.resolve(None, ctx) == 20
    assert ctx.adjustments == ()


def test_resolve_clamps_caller_visibly_into_the_narrowed_range() -> None:
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-2")
    d = decode_model_defaults(
        SDXL, model="sdxl", defaults={"guidance": {"hi": 9.0}}
    )
    assert d.guidance.resolve(14.0, ctx) == 9.0
    (row,) = ctx.adjustments
    assert row["field"] == "guidance"
    assert row["requested"] == "14.0"
    assert float(row["applied"]) == 9.0
    assert "range" in row["reason"]


def test_resolve_in_range_changes_nothing_and_records_nothing() -> None:
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-3")
    d = SDXL.Defaults()
    assert d.guidance.resolve(7.5, ctx) == 7.5
    assert ctx.adjustments == ()


def test_resolve_never_rejects_inside_the_envelope() -> None:
    # The API Meta bounds rejected upstream; whatever reaches resolve serves.
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-4")
    d = decode_model_defaults(SD15, model="sd15", defaults=None)
    assert d.steps.resolve(79, ctx) == 79
    assert d.steps.resolve(-5, ctx) == 1  # clamped, not raised
    assert ctx.adjustments[0]["field"] == "steps"


# ── the contract-file (main_v2.py) usage, against fixture rows ───────────────


def test_the_contract_files_exact_usage_holds() -> None:
    """Every ``main_v2.py`` defaults expression over fixture hub rows: the
    recipe-driven single entrypoint — ``recipe: SDXL.Recipe`` from the
    distillation adapter's defaults when one rides, else the checkpoint's own;
    ``recipe.cfg`` gates guidance/negatives, ``recipe.scheduler`` gates the
    scheduler swap (None = keep the checkpoint's own), pinned timesteps ride
    the cfg-off arm."""
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-main-v2")
    d = decode_model_defaults(
        SDXL,
        model="sdxl",
        defaults={"guidance": {"default": 5.0, "hi": 9.0}},
    )

    # The stacking gate is step_distilled, NOT cfg (`if turbo is not None and
    # d.step_distilled: ctx.warn(...); turbo = None` — warn-and-serve, never
    # an error): a guidance-distilled full-step checkpoint (cfg=False,
    # step_distilled=False) MAY take a turbo LoRA.
    assert not d.step_distilled
    guidance_distilled = decode_model_defaults(
        SDXL, model="sdxl", defaults={"cfg": False}
    )
    assert not guidance_distilled.cfg and not guidance_distilled.step_distilled
    fused_merge = decode_model_defaults(
        SDXL, model="sdxl",
        defaults={"cfg": False, "step_distilled": True, "scheduler": "lcm"},
    )
    assert fused_merge.step_distilled  # -> the adapter is ignored with a warn

    # No adapter: the recipe is the checkpoint's own Defaults — one nominal
    # type, both Defaults inherit SDXL.Recipe.
    recipe: SDXL.Recipe = d
    assert isinstance(recipe, SDXL.Recipe)
    steps = recipe.steps.resolve(None, ctx)  # payload sent None
    assert steps == 28
    assert recipe.cfg
    guidance = recipe.guidance.resolve(14.0, ctx)  # inside the API envelope
    assert guidance == 9.0  # clamped to the row's narrowed hi

    prompt = "a cat"
    negative = ""
    if d.positive_preamble and d.positive_preamble not in prompt:
        prompt = f"{d.positive_preamble}, {prompt}"
    if d.negative_preamble and d.negative_preamble not in negative:
        negative = f"{d.negative_preamble}, {negative}" if negative else d.negative_preamble
    assert prompt == "masterpiece, best quality, a cat"
    assert negative == "worst quality, low quality"
    # Skipped when already present — no double preamble.
    again = prompt
    if d.positive_preamble and d.positive_preamble not in again:
        again = f"{d.positive_preamble}, {again}"
    assert again == prompt

    # schedule=None means: keep the checkpoint's own scheduler (nullcontext).
    assert recipe.scheduler is None

    # A distillation adapter rides: its defaults ARE the recipe.
    turbo = decode_model_defaults(
        SDXL.Lora,
        model="sdxl.lora",
        defaults={"scheduler": "lcm", "timesteps": [999, 749, 499, 249]},
    )
    recipe = turbo
    assert not recipe.cfg  # the cfg-off arm: no guidance, no negatives
    assert recipe.scheduler == "lcm"  # -> LCMScheduler swap
    assert list(recipe.timesteps) == [999, 749, 499, 249]  # pinned ladder
    assert recipe.steps.resolve(None, ctx) == 4

    # The independent-axes counterexample: a Hyper-SD-style CFG-preserving
    # few-step adapter row — few-step AND cfg on at its recommended 5-8.
    hyper = decode_model_defaults(
        SDXL.Lora,
        model="sdxl.lora",
        defaults={"cfg": True, "guidance": {"default": 6.5, "lo": 5.0, "hi": 8.0},
                  "steps": {"default": 8}},
    )
    assert hyper.cfg
    assert hyper.guidance.resolve(None, ctx) == 6.5
    assert hyper.guidance.resolve(14.0, ctx) == 8.0
    assert hyper.steps.resolve(None, ctx) == 8


# ── the vocabulary registry + ingest fingerprint seam ────────────────────────


def test_the_launch_vocabulary_is_the_ruled_set() -> None:
    assert [mt.name for mt in MODEL_TYPES] == [
        "sdxl", "sd15", "sd2", "hidream-o1", "wan22",
    ]
    assert [ov.name for ov in LORA_OVERLAYS] == ["sdxl.lora", "sd15.lora"]
    assert model_type_by_name("sdxl") is SDXL
    assert model_type_by_name("flux") is None


def test_contract_stamps_classify_through_the_fingerprint() -> None:
    # A real registered stamp from tensorfs's built-in contracts.
    assert model_type_for_contract("sdxl.clip-g-fused-qkv@1") is SDXL
    assert model_type_for_contract("sd15.diffusers-bf16@1") is SD15
    # Unrecognized = unclassified, legal and visible — never a guess.
    assert model_type_for_contract("minimax.h3-dit-native@1") is None


def test_canonical_scheduler_configs_are_the_training_schedules() -> None:
    """The ingest-synthesis data (Paul's ruling: a bare scheduler class
    carries library-default betas, not the family's training schedule)."""
    import json

    sdxl = SDXL.canonical_scheduler_config
    assert (sdxl["beta_start"], sdxl["beta_end"]) == (0.00085, 0.012)
    assert sdxl["beta_schedule"] == "scaled_linear"
    assert sdxl["prediction_type"] == "epsilon"
    assert sdxl["_class_name"] == "EulerDiscreteScheduler"
    sd15 = SD15.canonical_scheduler_config
    assert (sd15["beta_start"], sd15["beta_end"]) == (0.00085, 0.012)
    # JSON-serializable as-is: ingest writes it verbatim as
    # scheduler_config.json into a classified tree that ships none.
    for mt in MODEL_TYPES:
        json.dumps(dict(mt.canonical_scheduler_config))
    # No canonical recorded for these yet — ingest synthesizes nothing
    # (flagged in the tracker; do not invent a family's noise schedule).
    from gen_worker.models import HiDreamO1, SD2, Wan22

    assert SD2.canonical_scheduler_config == {}
    assert HiDreamO1.canonical_scheduler_config == {}
    assert Wan22.canonical_scheduler_config == {}


def test_model_types_are_vocabularies_not_values() -> None:
    with pytest.raises(TypeError):
        SDXL()
    with pytest.raises(TypeError):
        SDXL.Lora()


def test_the_package_root_serves_the_names_lazily() -> None:
    import gen_worker.models as pkg

    assert pkg.SDXL is SDXL
    assert "SDXL" in dir(pkg)


def test_no_warning_leaks_from_a_classified_decode() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decode_model_defaults(SDXL, model="sdxl", defaults={"cfg": False})
