from __future__ import annotations

import pathlib
import warnings

import msgspec
import pytest

from gen_worker.models import model_types
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
from gen_worker.models.model_types import SdxlLoraDefaults
from gen_worker.request_context import RequestContext


def test_sdxl_zero_arg_is_the_platform_opinion() -> None:
    d = SDXL.Defaults()
    assert d.steps == Knob(28, lo=1, hi=80, name="steps")
    assert d.guidance == Knob(6.0, lo=1.5, hi=15.0, name="guidance")
    assert d.cfg is True
    assert d.positive_preamble == "masterpiece, best quality"
    assert d.negative_preamble == "worst quality, low quality"
    assert not hasattr(d, "scheduler")
    assert d.timesteps == ()
    assert d.step_distilled is False


def test_sdxl_lora_zero_arg_is_lightning_shaped() -> None:
    d = SDXL.Lora.Defaults()
    assert d.cfg is False
    assert d.scheduler == "euler_trailing"
    assert d.distillation is False
    assert d.steps.default == 4
    assert d.timesteps == ()
    assert d.strength == Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    assert d.guidance == SDXL.Defaults().guidance
    assert isinstance(d, SDXL.Config)
    assert isinstance(SDXL.Defaults(), SDXL.Config)


def test_every_zero_arg_defaults_is_servable() -> None:
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
    """The clamp ledger names fields through ``Knob.name`` — a drifted name would mislabel every adjustment row."""
    for wire_name, cls in defaults_vocabularies().items():
        for f in msgspec.structs.fields(cls):
            if isinstance(f.default, Knob):
                assert f.default.name == f.encode_name, f"{wire_name}.{f.encode_name}"


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
    assert (d.steps.lo, d.steps.hi) == (1, 80)
    assert d.guidance == SDXL.Defaults().guidance
    assert d.positive_preamble == "masterpiece, best quality"


def test_full_row_overrides_every_field() -> None:
    row = {
        "steps": {"default": 6, "lo": 4, "hi": 10},
        "guidance": {"default": 2.0, "lo": 1.5, "hi": 3.0},
        "cfg": False,
        "positive_preamble": "",
        "negative_preamble": "",
        "step_distilled": True,
        "timesteps": [999, 749, 499, 249],
        "scheduler": "lcm",
    }
    d = decode_model_defaults(SDXL, model="sdxl", defaults=row)
    assert d.steps == Knob(6, lo=4, hi=10, name="steps")
    assert d.guidance == Knob(2.0, lo=1.5, hi=3.0, name="guidance")
    assert d.cfg is False
    assert d.positive_preamble == ""
    assert d.step_distilled is True
    assert not hasattr(d, "scheduler")
    assert d.timesteps == (999, 749, 499, 249)


def test_knob_ranges_merge_to_the_narrowest_layer() -> None:
    d = decode_model_defaults(
        SDXL,
        model="sdxl",
        defaults={"guidance": {"default": 5.0, "lo": 1.0, "hi": 9.0}},
    )
    assert (d.guidance.lo, d.guidance.hi) == (1.5, 9.0)
    assert d.guidance.default == 5.0


def test_a_row_default_outside_the_merged_range_is_pulled_inside() -> None:
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
    assert d == SDXL.Defaults()
    assert CheckpointDefaultsUnclassified.code == "checkpoint_defaults_unclassified"
    assert "unclassified" in str(caught[0].message)


def test_a_mistyped_checkpoint_is_a_typed_refusal_not_a_warning() -> None:
    with pytest.raises(ModelTypeMismatch) as caught:
        decode_model_defaults(SDXL, model="sd15", defaults=None)
    assert (caught.value.expected, caught.value.actual) == ("sdxl", "sd15")


def test_a_base_mismatched_adapter_is_refused_at_bind() -> None:
    with pytest.raises(ModelTypeMismatch):
        decode_model_defaults(SDXL.Lora, model="sd15.lora", defaults=None)


def test_an_out_of_vocabulary_adapter_scheduler_is_refused() -> None:
    with pytest.raises(DefaultsDecodeError) as caught:
        decode_model_defaults(
            SDXL.Lora, model="sdxl.lora", defaults={"scheduler": "ddim_trailing"}
        )
    assert caught.value.field == "scheduler"


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
            "distillation": True,
        },
    )
    assert d.distillation is True
    assert d.trigger_words == ("dmd2",)
    assert d.strength.default == 0.8
    assert (d.strength.lo, d.strength.hi) == (-4.0, 4.0)
    assert d.scheduler == "lcm"
    assert d.timesteps == (999, 749, 499, 249)


class _V1Defaults(msgspec.Struct, frozen=True):
    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")


class _V2Defaults(msgspec.Struct, frozen=True):
    steps: Knob[int] = Knob(28, lo=1, hi=80, name="steps")
    shine: float = 0.5


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
    assert d.steps.name == "steps"
    assert d.steps.default == 12


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
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-4")
    d = decode_model_defaults(SD15, model="sd15", defaults=None)
    assert d.steps.resolve(79, ctx) == 79
    assert d.steps.resolve(-5, ctx) == 1
    assert ctx.adjustments[0]["field"] == "steps"


def test_the_contract_files_exact_usage_holds() -> None:
    """Every ``main_v2.py`` defaults expression over fixture hub rows: the config-driven single entrypoint — ``config: SDXL.Config`` from the distillation adapter's defaults when one rides, else the check..."""
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-main-v2")
    d = decode_model_defaults(
        SDXL,
        model="sdxl",
        defaults={"guidance": {"default": 5.0, "hi": 9.0}},
    )

    assert not d.step_distilled
    guidance_distilled = decode_model_defaults(
        SDXL, model="sdxl", defaults={"cfg": False}
    )
    assert not guidance_distilled.cfg and not guidance_distilled.step_distilled
    fused_merge = decode_model_defaults(
        SDXL, model="sdxl",
        defaults={"cfg": False, "step_distilled": True},
    )
    assert fused_merge.step_distilled

    config: SDXL.Config = d
    assert isinstance(config, SDXL.Config)
    steps = config.steps.resolve(None, ctx)
    assert steps == 28
    assert config.cfg
    guidance = config.guidance.resolve(14.0, ctx)
    assert guidance == 9.0

    prompt = "a cat"
    if d.positive_preamble and d.positive_preamble not in prompt:
        prompt = f"{d.positive_preamble}, {prompt}"
    assert prompt == "masterpiece, best quality, a cat"
    again = prompt
    if d.positive_preamble and d.positive_preamble not in again:
        again = f"{d.positive_preamble}, {again}"
    assert again == prompt
    negative = ""
    if config.cfg and d.negative_preamble and d.negative_preamble not in negative:
        negative = f"{d.negative_preamble}, {negative}" if negative else d.negative_preamble
    assert negative == "worst quality, low quality"

    assert not hasattr(config, "scheduler")

    turbo = decode_model_defaults(
        SDXL.Lora,
        model="sdxl.lora",
        defaults={"scheduler": "lcm", "timesteps": [999, 749, 499, 249],
                  "distillation": True},
    )
    config = turbo
    assert not config.cfg
    assert turbo.scheduler == "lcm"
    assert turbo.distillation
    assert list(config.timesteps) == [999, 749, 499, 249]
    assert config.steps.resolve(None, ctx) == 4

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


def _pick_scheduler(
    request: str | None, turbo: "SdxlLoraDefaults | None"
) -> str | None:
    served = {"dpmpp_2m_karras", "dpmpp_2m", "euler", "euler_trailing",
              "euler_a", "unipc", "ddim", "lcm"}
    if request is not None:
        return request
    if turbo is not None and turbo.scheduler is not None:
        if turbo.scheduler in served:
            return turbo.scheduler
        return None
    return None


@pytest.mark.parametrize("adapter_rides", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("request_scheduler", [None, "euler"])
def test_the_serving_interaction_matrix(
    adapter_rides: bool, cfg: bool, pinned: bool, request_scheduler: str | None
) -> None:
    """The ruled interaction matrix (scheduler-override × pinned-timesteps × cfg × adapter-state): the decoded config fields drive main_v2.py's arms for every combination — no combination raises, every co..."""
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-matrix")
    ladder = [999, 749, 499, 249]
    row: dict[str, object] = {"cfg": cfg}
    if pinned:
        row["timesteps"] = ladder

    turbo: SdxlLoraDefaults | None = None
    if adapter_rides:
        turbo = decode_model_defaults(
            SDXL.Lora, model="sdxl.lora", defaults={**row, "distillation": True}
        )
        config: SDXL.Config = turbo
    else:
        config = decode_model_defaults(SDXL, model="sdxl", defaults=row)

    assert config.cfg is cfg

    picked = _pick_scheduler(request_scheduler, turbo)
    if request_scheduler is not None:
        assert picked == request_scheduler
    elif adapter_rides:
        assert picked == "euler_trailing"
    else:
        assert picked is None

    steps = config.steps.resolve(7, ctx)
    timesteps: list[int] | None = None
    if config.timesteps:
        if request_scheduler is None:
            steps, timesteps = len(config.timesteps), list(config.timesteps)
    if pinned and request_scheduler is None:
        assert (steps, timesteps) == (4, ladder)
    else:
        assert (steps, timesteps) == (7, None)

    guidance = config.guidance.resolve(None, ctx) if config.cfg else 0.0
    assert guidance == (6.0 if cfg else 0.0)


def test_no_defaults_field_comment_dangles_without_its_field() -> None:
    import re

    source = (
        pathlib.Path(model_types.__file__).read_text(encoding="utf-8")
    )
    dangling: list[str] = []
    for match in re.finditer(
        r"class (\w*Defaults)\(msgspec\.Struct.*?(?=\nclass |\Z)", source, re.S
    ):
        name, body = match.group(1), match.group(0)
        lines = body.split("\n")
        for i, line in enumerate(lines):
            if not line.strip().startswith("#:"):
                continue
            nxt = None
            for cand in lines[i + 1:]:
                if cand.strip().startswith("#:") or not cand.strip():
                    continue
                nxt = cand
                break
            if nxt is None or not re.match(r"^    \w+\s*:", nxt):
                dangling.append(
                    f"{name}: comment {line.strip()[:60]!r} is followed by "
                    f"{(nxt or '<END OF STRUCT>').strip()[:40]!r}, not a field"
                )
                break
    assert not dangling, (
        "a Defaults field comment survives with no field under it — the field "
        "was deleted and its documentation was not:\n  " + "\n  ".join(dangling)
    )


def test_every_model_type_exports_a_defaults_vocabulary() -> None:
    missing = {mt.name for mt in MODEL_TYPES} - set(defaults_vocabularies())
    assert not missing, (
        f"{sorted(missing)} are in MODEL_TYPES but export no defaults "
        f"vocabulary — the export emitter reads defaults_vocabularies(), so "
        f"these families would ship no schema, silently. Add them there too."
    )


def test_the_launch_vocabulary_is_the_ruled_set() -> None:
    assert [mt.name for mt in MODEL_TYPES] == [
        "sdxl", "sd15", "sd2", "hidream-o1", "wan22", "minimax-h3", "rife",
        "qwen3.6-27b-mtp", "qwen3.6-35b-a3b", "flux1", "flux2-klein",
        "krea-2", "anima", "ernie", "qwen-image", "z-image",
        "stable-audio", "musicgen",
        "ltx-2", "ltx-2-upsampler",
        "internvl-u",
        "trellis2", "hunyuan3d",
        "joycaption",
    ]
    assert [ov.name for ov in LORA_OVERLAYS] == ["sdxl.lora", "sd15.lora"]
    assert model_type_by_name("sdxl") is SDXL
    from gen_worker.models import Flux1, Flux2Klein, MusicGen, StableAudio

    assert model_type_by_name("flux1") is Flux1
    assert model_type_by_name("flux2-klein") is Flux2Klein
    assert model_type_by_name("flux") is None
    from gen_worker.models import InternVLU

    assert model_type_by_name("internvl-u") is InternVLU
    assert model_type_by_name("internvl-U") is None
    assert model_type_by_name("internvl") is None
    from gen_worker.models import Qwen36A3b, Qwen36Mtp

    assert model_type_by_name("qwen3.6-27b-mtp") is Qwen36Mtp
    assert model_type_by_name("qwen3.6-35b-a3b") is Qwen36A3b
    from gen_worker.models import Hunyuan3d, Trellis2

    assert model_type_by_name("trellis2") is Trellis2
    assert model_type_by_name("hunyuan3d") is Hunyuan3d
    assert model_type_by_name("qwen") is None
    assert model_type_by_name("qwen3.6") is None
    from gen_worker.models import QwenImage, ZImage

    assert model_type_by_name("qwen-image") is QwenImage
    assert model_type_by_name("z-image") is ZImage
    assert model_type_by_name("qwen-image-edit") is None


TENSORFS_130_OWED: frozenset[str] = frozenset()
# pgw#1621 REOPENED this list with one row and then CLOSED it again in the same
# change, which is the behaviour the row exists to have rather than a tidy-up.
#
# The row was `trellis2.dit`: v1's `trellis2.dit-bf16@1` (tensorfs#132) had no
# v2 counterpart, so `trellis-3d` was refused at import again — the exact
# blocking state tensorfs#130 had closed, one family over. tensorfs#152
# (`ac9c9d4`) banked the headers, `trellis2.dit@1` exists, and the assertion
# below went red on the vendor bump naming the stale row. It also closed
# `flux1`, `stable-audio`, `qwen-image`, `internvl-u`, `krea-2` and `rife`,
# none of which ever reached this list.
#
# `trellis2` is worth remembering for a different reason: its upstream
# `ckpts/` is one flat directory holding three DIFFERENT 640-key DiTs that
# separate only on `input_layer.weight` ([1536,8]/[1536,32]/[1536,64]).
# Directory-based grouping could not express that at all, so tensorfs#152 had
# to group by shard-family name and key disjointness — and all 21 pre-existing
# topology digests came out byte-identical, which is what says the regrouping
# changed the expressible set and not the existing answers.

PICKLE_ONLY: frozenset[str] = frozenset({"hunyuan3d.dit"})


def _library_lacks(name: str) -> bool:

    assert name in TENSORFS_130_OWED or name in PICKLE_ONLY, (
        f"{name!r} is absent from the vendored v2 topology corpus and is on "
        f"NEITHER the tensorfs#130 work list nor the pickle-only list. Either "
        f"a document was removed, or this list is stale."
    )
    return not _library_has(name)


def _library_has(name: str) -> bool:
    """Whether the VENDORED tensorfs corpus carries a TOPOLOGY ``name``.

    pgw#1599 deleted ``ModelType.canonical_contract`` — a model TYPE cannot
    own a layout, because a layout is a property of a CHECKPOINT and the
    serving class's ``lanes=`` is what commits to one. The underlying fact
    these tests were really asserting survives, and this is where it lives
    now: does tensorfs publish a document for this family at all? An endpoint
    that cannot name one cannot declare ``lanes=`` and is refused at import.

    pgw#1621 moved WHICH document answers that. v1 published one document per
    LANE (`ltx-2.diffusers-bf16@1`) and this helper asked
    `contracts.get(<name>)`. v2 splits the lane in two: a TOPOLOGY (which
    tensors, at what shapes — extracted mechanically from banked headers) and
    a QUANT RULE (eight of them, for the whole fleet). The family question is
    the TOPOLOGY question, so that is what is asked here — and the `-bf16`
    suffix these callers used to pass is not part of a topology handle,
    because a topology carries no dtype at all.
    """
    from gen_worker.models.tensor_layout_contract import known_topologies

    return f"{name}@1" in known_topologies()


def test_the_llm_roots_declare_no_lane_and_no_card_budget() -> None:
    from gen_worker.models import Qwen36A3b, Qwen36Mtp

    for model_type in (Qwen36Mtp, Qwen36A3b):
        assert not hasattr(model_type, "canonical_contract")
        assert model_type.canonical_scheduler_config == {}
        defaults = model_type.Defaults()
        assert defaults.max_tokens.default == 256
        assert defaults.max_tokens.lo == 1
        assert defaults.max_tokens.hi is None
        assert defaults.top_p.default == 0.95
        assert (defaults.temperature.lo, defaults.temperature.hi) == (None, None)

    assert Qwen36Mtp.Defaults().temperature.default == 0.6
    assert Qwen36A3b.Defaults().temperature.default == 0.7


def test_the_audio_roots_are_one_stable_audio_and_a_lane_less_musicgen() -> None:
    from gen_worker.models import MusicGen, StableAudio

    assert model_type_by_name("stable-audio") is StableAudio
    assert model_type_by_name("foundation-1") is None
    assert model_type_by_name("musicgen") is MusicGen

    # tensorfs#130: musicgen HAS a document, and pgw#1621's split shows why
    # the v1 name was two facts glued together. The TOPOLOGY is
    # `musicgen.transformers@1` — the checkpoint is a single-file transformers
    # tree, not a diffusers one, so the format segment says so — and the
    # precision half is the separate ratified rule `plain.f16@1`.
    #
    # ⚠️ Its DISPLAY name is `musicgen.transformers-fp16@1`: **`fp16` in the
    # display name, `f16` in the rule handle.** Never derive one from the
    # other by string surgery (pinned in test_lane_contracts.py).
    assert _library_has("musicgen.transformers")
    from gen_worker.models.tensor_layout_contract import display_names

    assert display_names()["musicgen.transformers@1+plain.f16@1"] == (
        "musicgen.transformers-fp16@1")
    assert MusicGen.canonical_scheduler_config == {}
    assert StableAudio.canonical_scheduler_config["prediction_type"] == "v_prediction"
    assert StableAudio.canonical_scheduler_config["_class_name"] == (
        "CosineDPMSolverMultistepScheduler"
    )


def test_the_two_3d_roots_declare_their_lanes_by_evidence() -> None:
    from gen_worker.models import Hunyuan3d, Trellis2
    from gen_worker.models.model_types import Rife

    # TRELLIS.2 briefly lost its document in the v2 cut and got it back in the
    # same change. v1 shipped `trellis2.dit-bf16@1` (tensorfs#132); the first
    # v2 corpus banked no trellis2 HEADERS, so `trellis-3d` could not declare
    # `lanes=` — the blocking state tensorfs#130 had closed, one family over.
    # It was recorded on `TENSORFS_130_OWED` rather than tolerated, the
    # assertion here was written to go RED the day the headers landed, and
    # tensorfs#152 (`ac9c9d4`) landed them. This is the other side of that
    # assertion: `trellis2.dit@1` EXISTS, so the family is declarable.
    assert _library_has("trellis2.dit")

    # Hunyuan3D has NO document and never had one. Its
    # checkpoint is a PICKLE, so no safetensors-shaped document can describe
    # it; a sentinel would assert a document is OWED, which would be a
    # standing lie with a to-do attached. Absent is honest — and under
    # pgw#1599 it is also LOUD: its Model class cannot declare `lanes=` and
    # is refused at import until the repack-to-safetensors job runs pod-side.
    assert _library_lacks("hunyuan3d.dit")

    del Rife

    assert Trellis2.canonical_scheduler_config == {}
    assert Hunyuan3d.canonical_scheduler_config == {}

    assert Trellis2.Defaults().steps.default == 12
    assert Hunyuan3d.Defaults().num_shape_steps.default == 50
    assert Hunyuan3d.Defaults().guidance_scale.default == 5.0


def test_the_3d_fingerprints_do_not_claim_the_shared_dit_fragment() -> None:
    from gen_worker.models import Hunyuan3d, Trellis2

    assert model_type_for_contract("trellis2.dit-bf16@1") is Trellis2
    assert model_type_for_contract("dit.blocks-fused-qkv@1") is not Trellis2
    assert model_type_for_contract("dit.blocks-fused-qkv@1") is None
    assert model_type_for_contract("hunyuan3d.anything@1") is Hunyuan3d
    assert model_type_for_contract("hunyuan3d.anything@1") is not Trellis2
    assert model_type_for_contract("trellis2.dit-bf16@1") is not Hunyuan3d


def test_contract_stamps_classify_through_the_fingerprint() -> None:
    """pgw#1621: the fingerprint matches the TOPOLOGY HALF of a v2 stamp pair.

    Which architecture a checkpoint IS, is a fact about which tensors it has —
    never about how they are quantized — so the same topology must classify
    identically under every quant rule, and it does (asserted below and in
    test_lane_contracts.py). Under v1 the patterns saw a whole lane handle,
    which meant a rule named after a family would have started matching.
    """
    from gen_worker.models import MiniMaxH3, Rife

    # Real vendored v2 topologies.
    assert model_type_for_contract("sdxl.clip-g-fused@1") is SDXL
    assert model_type_for_contract("sd15.diffusers@1") is SD15
    # v1 could not tell inpainting from base SDXL at all; v2 banks it as its
    # own topology, and it classifies to the one SDXL vocabulary.
    assert model_type_for_contract("sdxl-inpainting.diffusers@1") is SDXL
    # Both packaged H3 layouts classify to the one H3 vocabulary — a split
    # to_q/to_k/to_v tree and a fused qkv_proj one are two topologies related
    # by a ratified morphism, not one handle with a side note.
    assert model_type_for_contract("minimax-h3.native@1") is MiniMaxH3
    assert model_type_for_contract("minimax-h3.diffusers@1") is MiniMaxH3

    # The QUANT half is never read: the render classifies as the bare
    # topology does, for every rule.
    for quant in ("plain.bf16@1", "cozy.fp8-rowwise@1", "cozy.nvfp4-flat@1"):
        assert model_type_for_contract(f"sdxl.diffusers@1+{quant}") is SDXL

    # The NAME seam still classifies a family with no vendored topology at
    # all — `rife.*` has no v2 document (its headers are not banked), and a
    # recorded stamp still maps to the vocabulary. Unclassified is legal;
    # so is classified-without-a-document.
    assert model_type_for_contract("rife.flownet@1") is Rife


def test_the_audio_fingerprints_do_not_cross_claim_with_flux() -> None:
    from gen_worker.models import Flux1, Flux2Klein, MusicGen, StableAudio

    assert model_type_for_contract("stable-audio.diffusers-fp16@1") is StableAudio
    assert model_type_for_contract("musicgen.native@1") is MusicGen

    for stamp in ("stable-audio.diffusers-fp16@1", "musicgen.native@1"):
        assert model_type_for_contract(stamp) not in (Flux1, Flux2Klein)
    for stamp in ("flux1.diffusers-bf16@1", "flux2-klein.diffusers-bf16@1"):
        assert model_type_for_contract(stamp) not in (StableAudio, MusicGen)

    from fnmatch import fnmatchcase

    assert not fnmatchcase("stable-audio.diffusers-fp16@1", "flux*")
    assert fnmatchcase("stable-audio.diffusers-fp16@1", "stable-audio.*")
    assert StableAudio.contracts == ("stable-audio.*",)
    assert MusicGen.contracts == ("musicgen.*",)
    assert model_type_for_contract("flux.diffusers-bf16@1") is None
    from gen_worker.models import Flux1, Flux2Klein

    assert model_type_for_contract("flux1.diffusers-bf16@1") is Flux1
    assert model_type_for_contract("flux2-klein.diffusers-bf16@1") is Flux2Klein
    assert model_type_for_contract("dit.blocks-fused-qkv@1") is None


def test_canonical_scheduler_configs_are_the_training_schedules() -> None:
    """The ingest-synthesis data (Paul's ruling: a bare scheduler class carries library-default betas, not the family's training schedule)."""
    import json

    sdxl = SDXL.canonical_scheduler_config
    assert (sdxl["beta_start"], sdxl["beta_end"]) == (0.00085, 0.012)
    assert sdxl["beta_schedule"] == "scaled_linear"
    assert sdxl["prediction_type"] == "epsilon"
    assert sdxl["_class_name"] == "EulerDiscreteScheduler"
    sd15 = SD15.canonical_scheduler_config
    assert (sd15["beta_start"], sd15["beta_end"]) == (0.00085, 0.012)
    for mt in MODEL_TYPES:
        json.dumps(dict(mt.canonical_scheduler_config))
    from gen_worker.models import HiDreamO1, SD2, Wan22

    from gen_worker.models import MiniMaxH3, Rife

    assert SD2.canonical_scheduler_config == {}
    assert HiDreamO1.canonical_scheduler_config == {}
    assert Wan22.canonical_scheduler_config == {}
    assert MiniMaxH3.canonical_scheduler_config == {}
    assert Rife.canonical_scheduler_config == {}
    from gen_worker.models import Flux1, Flux2Klein

    assert Flux1.canonical_scheduler_config == {}
    klein = Flux2Klein.canonical_scheduler_config
    assert klein["_class_name"] == "FlowMatchEulerDiscreteScheduler"
    for beta_field in ("beta_start", "beta_end", "beta_schedule", "trained_betas"):
        assert beta_field not in klein
    assert (klein["base_shift"], klein["max_shift"], klein["shift"]) == (0.5, 1.15, 3.0)
    assert klein["use_dynamic_shifting"] is True
    assert (klein["base_image_seq_len"], klein["max_image_seq_len"]) == (256, 4096)
    assert klein["num_train_timesteps"] == 1000
    # h3's DiT names a real document. pgw#1621 re-key: the TOPOLOGY handle is
    # `minimax-h3.diffusers@1` (v1 spelled it `minimax.h3-dit-diffusers@1`,
    # with the family split across the `.` and the precision glued on).
    assert _library_has("minimax-h3.diffusers")
    assert _library_has("minimax-h3.native")
    # Rife HAS a v2 topology, and it arrived by an unusual route worth naming.
    # v1's shared `rife.flownet-fp32@1` had no counterpart in the first v2
    # corpus, and the recorded reason was "no upstream header to extract from,
    # re-derive from the producing module's state_dict" — which was FALSE:
    # tensorhub SERVES the produced checkpoint, so tensorfs#152 banked the
    # headers off the hub tree itself. `rife.flownet@1` therefore describes
    # exactly what the fleet binds rather than an upstream packaging nobody
    # ships. Its pair is `rife.flownet@1+plain.f32@1`.
    assert _library_has("rife.flownet")


def test_flux_platform_values_are_the_shipped_endpoint_numbers() -> None:
    from gen_worker.models import Flux1, Flux2Klein

    f1 = Flux1.Defaults()
    assert f1.steps == Knob(28, lo=1, hi=100, name="steps")
    assert f1.guidance == Knob(3.5, lo=0.0, hi=10.0, name="guidance")
    assert f1.cfg is False
    assert f1.step_distilled is False
    assert f1.max_sequence_length == 512

    k = Flux2Klein.Defaults()
    assert k.steps == Knob(28, lo=1, hi=50, name="steps")
    assert k.guidance == Knob(4.0, lo=1.0, hi=10.0, name="guidance")
    assert k.cfg is True
    assert k.max_sequence_length == 512

    for d in (f1, k):
        assert not hasattr(d, "scheduler")
        assert not hasattr(d, "timesteps")
    assert not hasattr(Flux1, "Lora")
    assert not hasattr(Flux2Klein, "Lora")


def test_flux_platform_floors_admit_the_distilled_checkpoints() -> None:
    """The floors are the ENDPOINTS' checkpoint facts, not their wire bounds."""
    from gen_worker.models import Flux1, Flux2Klein

    schnell = decode_model_defaults(Flux1, model="flux1", defaults={
        "steps": {"default": 4, "hi": 4},
        "guidance": {"default": 0.0, "lo": 0.0, "hi": 0.0},
        "step_distilled": True,
        "max_sequence_length": 256,
    })
    assert schnell.guidance == Knob(0.0, lo=0.0, hi=0.0, name="guidance")
    assert schnell.steps.default == 4 and schnell.steps.hi == 4
    assert schnell.step_distilled is True
    assert schnell.max_sequence_length == 256

    turbo = decode_model_defaults(Flux2Klein, model="flux2-klein", defaults={
        "steps": {"default": 4},
        "guidance": {"default": 1.0},
        "cfg": False,
        "step_distilled": True,
    })
    assert turbo.steps.default == 4
    assert turbo.guidance.default == 1.0
    assert turbo.cfg is False and turbo.step_distilled is True

    flex2 = decode_model_defaults(Flux1, model="flux1", defaults={
        "steps": {"default": 28, "hi": 100}, "cfg": True,
    })
    assert flex2.steps.hi == 100 and flex2.cfg is True


def test_qwen_and_z_platform_values_are_the_shipped_endpoint_numbers() -> None:
    from gen_worker.models import QwenImage, ZImage

    q = QwenImage.Defaults()
    assert q.steps.default == 30 and q.guidance.default == 4.0
    assert q.negative == " ", "Qwen's uncond convention is a SPACE, never ''"
    assert q.cfg is True and q.step_distilled is False
    assert q.max_sequence_length == 512
    assert not hasattr(q, "max_guidance")

    z = ZImage.Defaults()
    assert z.steps.default == 28 and z.guidance.default == 4.0
    assert z.cfg is True and z.step_distilled is False
    assert z.max_sequence_length == 512

    assert not hasattr(QwenImage, "Lora")
    assert not hasattr(ZImage, "Lora")


def test_qwen_and_z_platform_floors_admit_their_distilled_checkpoints() -> None:
    from gen_worker.models import QwenImage, ZImage

    dmd = decode_model_defaults(ZImage, model="z-image", defaults={
        "steps": {"default": 9}, "guidance": {"default": 0.0},
        "cfg": False, "step_distilled": True,
    })
    assert dmd.steps.default == 9
    assert dmd.guidance.default == 0.0 and dmd.guidance.lo == 0.0
    assert dmd.cfg is False and dmd.step_distilled is True

    pai = decode_model_defaults(ZImage, model="z-image", defaults={
        "steps": {"default": 8, "hi": 16}, "guidance": {"default": 0.0},
        "cfg": False, "step_distilled": True,
    })
    assert pai.steps.default == 8 and pai.steps.hi == 16

    lightning = decode_model_defaults(QwenImage, model="qwen-image", defaults={
        "steps": {"default": 8}, "guidance": {"default": 1.0},
        "cfg": False, "step_distilled": True,
    })
    assert lightning.steps.default == 8
    assert lightning.guidance.default == 1.0
    assert lightning.cfg is False and lightning.step_distilled is True

    edit = decode_model_defaults(QwenImage, model="qwen-image", defaults={
        "max_sequence_length": 1024,
    })
    assert edit.max_sequence_length == 1024
    assert edit.steps.default == 30, "the edit row shares the family recipe"


def test_a_base_handlers_wire_floor_really_would_corrupt_a_distilled_row() -> None:
    """The RED control for the test above: prove the hazard is real HERE, so the floors are known-good rather than merely never-exercised."""
    from gen_worker.models.defaults_decode import decode_defaults

    class PortedZImage(msgspec.Struct, frozen=True):
        """z-image's BASE wire floor ge=1.0 (main.py:278) ported as-is."""

        guidance: Knob[float] = Knob(4.0, lo=1.0, hi=15.0, name="guidance")

    class PortedQwen(msgspec.Struct, frozen=True):
        """qwen's BASE wire floor ge=10 (main.py:317) ported as-is."""

        steps: Knob[int] = Knob(30, lo=10, hi=80, name="steps")

    corrupted = decode_defaults(PortedZImage, {"guidance": {"default": 0.0}})
    assert corrupted.guidance.default == 1.0, "the port silently serves CFG on"

    corrupted2 = decode_defaults(PortedQwen, {"steps": {"default": 8}})
    assert corrupted2.steps.default == 10, "the port silently serves 10 steps"

    from gen_worker.models import QwenImage, ZImage

    assert decode_model_defaults(
        ZImage, model="z-image", defaults={"guidance": {"default": 0.0}},
    ).guidance.default == 0.0
    assert decode_model_defaults(
        QwenImage, model="qwen-image", defaults={"steps": {"default": 8}},
    ).steps.default == 8


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


from gen_worker.models import Ltx2, Ltx2Defaults, Ltx2Upsampler
from gen_worker.models.defaults_decode import decode_defaults
from gen_worker.models.model_types import LTX2_SCHEDULER_CONFIG

_DEV_ROW = {
    "guidance": 3.0,
    "audio_guidance": 3.0,
    "sigmas": [1.0, 0.75, 0.5, 0.25],
    "cfg": True,
}


def test_the_root_is_the_registered_family_name_not_the_endpoint_slug() -> None:
    """`register_family("ltx-2", ...)` is the family owner's own shipped call; `ltx-video-2.3` is an endpoint slug and was never the vocabulary name."""

    assert Ltx2.name == "ltx-2"
    assert model_type_by_name("ltx-2") is Ltx2
    assert model_type_by_name("ltx-video-2.3") is None


def test_the_lane_resolves_and_the_SENTINEL_shape_is_gone() -> None:
    """pgw#1621 replaced the sentinel with an impossibility.

    v1 exported a module constant per lane (`LTX2_DIFFUSERS_BF16`) which was
    either a real `Contract` or a `MissingContract` sentinel that refused every
    read — the shape existed so a ModelType could precede its per-lane
    document. v2 has no per-lane document to precede: a lane is composed at
    the declaration site from a topology and a rule, both of which must
    already be in the corpus or `parse_lane_stamp` refuses. So the sentinel is
    not merely unused, it is unwritable, and this asserts BOTH halves — the
    constants are gone AND the lane they stood for resolves.
    """
    from gen_worker.models import model_types as mt
    from gen_worker.models.tensor_layout_contract import (
        capability_floor_for_rule,
        parse_lane_stamp,
        rule_dtype,
    )

    for dead in ("MissingContract", "MissingContractError",
                 "LTX2_DIFFUSERS_BF16", "TRELLIS2_DIT_BF16",
                 "SDXL_DIFFUSERS_BF16", "SD15_DIFFUSERS_BF16"):
        assert not hasattr(mt, dead), f"{dead} is back in model_types"

    stamp = parse_lane_stamp(("ltx2.diffusers@1", "plain.bf16@1"), where="test")
    assert stamp.render() == "ltx2.diffusers@1+plain.bf16@1"
    # The facts the old sentinel refused to answer come off the RULE now.
    assert rule_dtype(stamp.quant) == "bfloat16"
    assert capability_floor_for_rule(stamp.quant) == 80


def test_the_upsampler_now_HAS_its_lane_document() -> None:
    """The full arc, in one assertion.

    Absent was honest, and a `MissingContract` sentinel would have been a
    standing lie with a to-do attached — so this type carried NOTHING. pgw#1599
    then made the absence LOUD rather than quiet: with `lanes=()` deleted,
    `Model[Ltx2Upsampler]` could not be declared AT ALL, which is what turned a
    quiet gap into a blocking one. tensorfs#130 closed it by making a contract
    CHEAP (generate-from-header) rather than optional, and the class can now
    name a real document.

    That sequence is the whole design in miniature: refuse the implicit, and
    the refusal becomes the work list.

    pgw#1621 re-key: the document is a TOPOLOGY now — `ltx2-upsampler.diffusers@1`,
    with no dtype in the handle, because the precision half is a separate
    ratified rule."""

    assert _library_has("ltx2-upsampler.diffusers")


def test_the_scheduler_config_is_ltx_s_own_and_not_klein_s() -> None:
    """Fetched verbatim from the pinned revision."""

    assert LTX2_SCHEDULER_CONFIG["use_dynamic_shifting"] is False
    assert LTX2_SCHEDULER_CONFIG["shift"] == 1.0
    assert LTX2_SCHEDULER_CONFIG["base_shift"] == 0.95
    assert LTX2_SCHEDULER_CONFIG["max_shift"] == 2.05


def test_both_types_are_registered() -> None:
    names = [mt.name for mt in MODEL_TYPES]
    assert "ltx-2" in names and "ltx-2-upsampler" in names


def test_the_ltx_2_fingerprint_does_not_capture_the_upsampler() -> None:
    """`ltx-2.*` is an fnmatch pattern and `ltx-2-upsampler...` shares its first five characters."""

    assert model_type_for_contract("ltx2.diffusers@1") is Ltx2
    assert model_type_for_contract("ltx2-upsampler.diffusers@1") is Ltx2Upsampler
    assert model_type_for_contract("ltx2-upsampler.diffusers@1") is not Ltx2
    # ...and it holds through the full stamp pair too, where the topology is
    # the PREFIX of the render — which is why `model_type_for_contract` splits
    # on `+` FIRST rather than globbing the whole string.
    assert model_type_for_contract(
        "ltx2-upsampler.diffusers@1+plain.bf16@1") is Ltx2Upsampler


def test_ltx2_declares_no_knob_at_all() -> None:
    """The structural reason this family is immune."""

    for field in msgspec.structs.fields(Ltx2Defaults):
        assert not isinstance(field.default, Knob), field.name


def test_a_distilled_sibling_row_survives_the_merge_unmodified() -> None:
    """GREEN."""

    merged = decode_defaults(Ltx2Defaults, _DEV_ROW, model_name="ltx-2")

    assert merged.guidance == 3.0
    assert merged.audio_guidance == 3.0
    assert merged.sigmas == (1.0, 0.75, 0.5, 0.25)
    assert merged.cfg is True
    assert merged.stage2_sigmas == Ltx2Defaults().stage2_sigmas
    assert merged.max_sequence_length == 1024


def test_the_zero_arg_default_is_the_SERVED_distilled_recipe() -> None:
    """`Defaults()` zero-arg is the platform opinion and must be SERVABLE."""

    d = Ltx2Defaults()
    assert len(d.sigmas) == 8
    assert d.sigmas[0] == 1.0 and d.sigmas[-1] == 0.421875
    assert len(d.stage2_sigmas) == 3
    assert d.guidance == 1.0 and d.audio_guidance == 1.0
    assert d.cfg is False


def test_RED_the_same_row_shape_IS_clamped_when_the_field_is_a_knob() -> None:
    """The control."""

    class _KnobbyDefaults(msgspec.Struct, frozen=True):
        steps: Knob[int] = Knob(30, lo=30, hi=80, name="steps")
        guidance: Knob[float] = Knob(4.0, lo=1.0, name="guidance")

    clamped = decode_defaults(
        _KnobbyDefaults, {"steps": {"default": 10}, "guidance": {"default": 0.0}},
        model_name="knobby",
    )

    assert clamped.steps.default == 30, "the clamp hazard must reproduce"
    assert clamped.guidance.default == 1.0, "the clamp hazard must reproduce"


@pytest.mark.parametrize("field", ["sigmas", "stage2_sigmas"])
def test_a_malformed_sigma_ladder_refuses_rather_than_being_coerced(
    field: str,
) -> None:
    from gen_worker.models.defaults_decode import DefaultsDecodeError

    with pytest.raises(DefaultsDecodeError):
        decode_defaults(Ltx2Defaults, {field: "not-a-ladder"}, model_name="ltx-2")
