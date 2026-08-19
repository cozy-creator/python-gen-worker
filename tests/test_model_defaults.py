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
from gen_worker.models.model_types import SdxlLoraDefaults
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
    # Checkpoints carry NO scheduler metadata — the tree IS the choice
    # (Paul's tree-only ruling; ingest synthesis covers scheduler-less trees).
    assert not hasattr(d, "scheduler")
    assert d.timesteps == ()
    # Checkpoint-level fact, decoupled from cfg (the guidance axis).
    assert d.step_distilled is False


def test_sdxl_lora_zero_arg_is_lightning_shaped() -> None:
    d = SDXL.Lora.Defaults()
    assert d.cfg is False
    assert d.scheduler == "euler_trailing"  # the adapter's scheduler DEMAND
    assert d.distillation is False  # rows for distill adapters set it True
    assert d.steps.default == 4
    assert d.timesteps == ()
    assert d.strength == Knob(1.0, lo=-4.0, hi=4.0, name="strength")
    # Inert while cfg=False; the base platform knob, so a row that flips cfg
    # on without narrowing still serves sanely.
    assert d.guidance == SDXL.Defaults().guidance
    # Both defaults types ARE the one nominal config type.
    assert isinstance(d, SDXL.Config)
    assert isinstance(SDXL.Defaults(), SDXL.Config)


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
        "step_distilled": True,
        "timesteps": [999, 749, 499, 249],
        # Checkpoints carry no scheduler metadata: a stray key is an UNKNOWN
        # field the evolution rule ignores, never a refusal.
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


def test_an_out_of_vocabulary_adapter_scheduler_is_refused() -> None:
    # ddim_trailing is outside the launch SchedulerName vocabulary (additive
    # evolution can admit it); an out-of-vocabulary DEMAND is typed garbage.
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
    config-driven single entrypoint — ``config: SDXL.Config`` from the
    distillation adapter's defaults when one rides, else the checkpoint's
    own; positive preamble applies in EVERY mode while negatives exist only
    under CFG; the scheduler chain is request > adapter demand > the tree
    stands; a pinned ladder belongs to the config's own scheduler."""
    ctx: RequestContext[GenerationDefaults] = RequestContext("req-main-v2")
    d = decode_model_defaults(
        SDXL,
        model="sdxl",
        defaults={"guidance": {"default": 5.0, "hi": 9.0}},
    )

    # The stacking gate is step_distilled, NOT cfg — and it WARNS-AND-IGNORES
    # the adapter (`turbo = None`), never an error: a guidance-distilled
    # full-step checkpoint (cfg=False, step_distilled=False) MAY take one.
    assert not d.step_distilled
    guidance_distilled = decode_model_defaults(
        SDXL, model="sdxl", defaults={"cfg": False}
    )
    assert not guidance_distilled.cfg and not guidance_distilled.step_distilled
    fused_merge = decode_model_defaults(
        SDXL, model="sdxl",
        defaults={"cfg": False, "step_distilled": True},
    )
    assert fused_merge.step_distilled  # -> adapter ignored with a ctx.warn

    # No adapter: the config is the checkpoint's own Defaults — one nominal
    # type, both Defaults inherit SDXL.Config.
    config: SDXL.Config = d
    assert isinstance(config, SDXL.Config)
    steps = config.steps.resolve(None, ctx)  # payload sent None
    assert steps == 28
    assert config.cfg
    guidance = config.guidance.resolve(14.0, ctx)  # inside the API envelope
    assert guidance == 9.0  # clamped to the row's narrowed hi

    # Positive preamble: EVERY mode (the positive prompt always exists);
    # skipped when already present. Negative preamble: CFG arm only.
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

    # _pick_scheduler chain, no adapter: request None + no demand -> None,
    # the tree's shipped scheduler stands (nullcontext arm); checkpoints
    # carry no scheduler field at all.
    assert not hasattr(config, "scheduler")

    # A distillation adapter rides: its defaults ARE the config, and its
    # scheduler DEMAND drives the swap (the base tree cannot know it).
    turbo = decode_model_defaults(
        SDXL.Lora,
        model="sdxl.lora",
        defaults={"scheduler": "lcm", "timesteps": [999, 749, 499, 249],
                  "distillation": True},
    )
    config = turbo
    assert not config.cfg  # the cfg-off arm: no guidance, no negatives
    assert turbo.scheduler == "lcm"  # adapter demand -> LCMScheduler swap
    assert turbo.distillation  # the distillation-slot marker (hub-validated)
    assert list(config.timesteps) == [999, 749, 499, 249]  # pinned ladder
    assert config.steps.resolve(None, ctx) == 4

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


def _pick_scheduler(
    request: str | None, turbo: "SdxlLoraDefaults | None"
) -> str | None:
    """main_v2.py's chain verbatim: request > adapter demand > None (tree)."""
    served = {"dpmpp_2m_karras", "dpmpp_2m", "euler", "euler_trailing",
              "euler_a", "unipc", "ddim", "lcm"}
    if request is not None:
        return request
    if turbo is not None and turbo.scheduler is not None:
        if turbo.scheduler in served:
            return turbo.scheduler
        return None  # warn + the tree stands
    return None


@pytest.mark.parametrize("adapter_rides", [False, True])
@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("request_scheduler", [None, "euler"])
def test_the_serving_interaction_matrix(
    adapter_rides: bool, cfg: bool, pinned: bool, request_scheduler: str | None
) -> None:
    """The ruled interaction matrix (scheduler-override × pinned-timesteps ×
    cfg × adapter-state): the decoded config fields drive main_v2.py's arms
    for every combination — no combination raises, every conflict resolves
    by the documented precedence."""
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
        assert picked == request_scheduler  # the request always wins
    elif adapter_rides:
        # zero-arg demand is euler_trailing unless the row said otherwise
        assert picked == "euler_trailing"
    else:
        assert picked is None  # the tree stands

    # The step ladder: a pinned ladder owns the step count, unless the
    # request overrode the scheduler it belongs to (then it is dropped).
    steps = config.steps.resolve(7, ctx)
    timesteps: list[int] | None = None
    if config.timesteps:
        if request_scheduler is None:
            steps, timesteps = len(config.timesteps), list(config.timesteps)
    if pinned and request_scheduler is None:
        assert (steps, timesteps) == (4, ladder)
    else:
        assert (steps, timesteps) == (7, None)

    # cfg gates guidance resolution; the cfg-off arm serves guidance 0.0.
    guidance = config.guidance.resolve(None, ctx) if config.cfg else 0.0
    assert guidance == (6.0 if cfg else 0.0)


# ── the vocabulary registry + ingest fingerprint seam ────────────────────────


def test_the_launch_vocabulary_is_the_ruled_set() -> None:
    # se#769 wave 3 (pgw#1427) appends krea-2, anima and ernie. The list is
    # pinned deliberately: a type appearing here without a ruling is the thing
    # this assertion exists to catch, so growing it is an EDIT, never a fixup.
    assert [mt.name for mt in MODEL_TYPES] == [
        "sdxl", "sd15", "sd2", "hidream-o1", "wan22", "minimax-h3", "rife",
        "qwen3.6-27b-mtp", "qwen3.6-35b-a3b", "flux1", "flux2-klein",
        "krea-2", "anima", "ernie", "qwen-image", "z-image",
    ]
    assert [ov.name for ov in LORA_OVERLAYS] == ["sdxl.lora", "sd15.lora"]
    assert model_type_by_name("sdxl") is SDXL
    # pgw#1393: FLUX.1 (dev/schnell/Flex.2) and FLUX.2 Klein (4b/9b) are TWO
    # roots, and neither is spelled bare "flux".
    from gen_worker.models import Flux1, Flux2Klein

    assert model_type_by_name("flux1") is Flux1
    assert model_type_by_name("flux2-klein") is Flux2Klein
    assert model_type_by_name("flux") is None
    # pgw#1422: the two qwen3.6 LLM roots are likewise SEPARATE vocabularies
    # (temperature 0.6 vs 0.7 — the family owner registered two schemas), and
    # neither is spelled bare "qwen" or shares a root with `qwen-image`.
    from gen_worker.models import Qwen36A3b, Qwen36Mtp

    assert model_type_by_name("qwen3.6-27b-mtp") is Qwen36Mtp
    assert model_type_by_name("qwen3.6-35b-a3b") is Qwen36A3b
    assert model_type_by_name("qwen") is None
    assert model_type_by_name("qwen3.6") is None
    # pgw#1426: qwen-image covers t2i AND Qwen-Image-Edit-2511 (one root), and
    # z-image covers the base AND the Decoupled-DMD Turbo.
    from gen_worker.models import QwenImage, ZImage

    assert model_type_by_name("qwen-image") is QwenImage
    assert model_type_by_name("z-image") is ZImage
    assert model_type_by_name("qwen-image-edit") is None


def test_the_llm_roots_declare_no_lane_and_no_card_budget() -> None:
    """pgw#1422. Both qwen3.6 roots are EXTERNAL-BINARY runtimes (llama.cpp,
    vLLM) that never call `ctx.load`, so they carry no canonical contract —
    the `Rife` shape, and an ABSENT contract rather than a `MissingContract`
    sentinel. And `max_tokens` must never inherit an endpoint's card budget:
    a platform `hi` only ever NARROWS a checkpoint row (the pgw#1393
    `Flux1.guidance` defect), so a 16k/32k KV cap here would make the
    family's real 262k context unreachable forever."""
    from gen_worker.models import Qwen36A3b, Qwen36Mtp

    for model_type in (Qwen36Mtp, Qwen36A3b):
        assert model_type.canonical_contract is None
        assert model_type.canonical_scheduler_config == {}
        defaults = model_type.Defaults()
        assert defaults.max_tokens.default == 256
        assert defaults.max_tokens.lo == 1
        assert defaults.max_tokens.hi is None
        assert defaults.top_p.default == 0.95
        # no bounds are sourced for the sampler knobs, so none are declared
        assert (defaults.temperature.lo, defaults.temperature.hi) == (None, None)

    assert Qwen36Mtp.Defaults().temperature.default == 0.6
    assert Qwen36A3b.Defaults().temperature.default == 0.7


def test_contract_stamps_classify_through_the_fingerprint() -> None:
    # A real registered stamp from tensorfs's built-in contracts.
    assert model_type_for_contract("sdxl.clip-g-fused-qkv@1") is SDXL
    assert model_type_for_contract("sd15.diffusers-bf16@1") is SD15
    # Both packaged H3 layouts classify to the one H3 vocabulary.
    from gen_worker.models import MiniMaxH3, Rife

    assert model_type_for_contract("minimax.h3-dit-native@1") is MiniMaxH3
    assert model_type_for_contract("minimax.h3-dit-diffusers@1") is MiniMaxH3
    assert model_type_for_contract("rife.flownet@1") is Rife
    # Unrecognized = unclassified, legal and visible — never a guess.
    assert model_type_for_contract("flux.diffusers-bf16@1") is None
    # pgw#1393: the two flux roots fingerprint separately, and the SHARED
    # block-spelling fragment classifies NOTHING — its own description says it
    # is "shared by Flux-family and timm-derived transformers", so matching on
    # it would claim every timm ViT for Flux.
    from gen_worker.models import Flux1, Flux2Klein

    assert model_type_for_contract("flux1.diffusers-bf16@1") is Flux1
    assert model_type_for_contract("flux2-klein.diffusers-bf16@1") is Flux2Klein
    assert model_type_for_contract("dit.blocks-fused-qkv@1") is None


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

    from gen_worker.models import MiniMaxH3, Rife

    assert SD2.canonical_scheduler_config == {}
    assert HiDreamO1.canonical_scheduler_config == {}
    assert Wan22.canonical_scheduler_config == {}
    assert MiniMaxH3.canonical_scheduler_config == {}
    assert Rife.canonical_scheduler_config == {}
    # pgw#1393: Flux is FLOW-MATCHING — there is no beta schedule to record at
    # all, and FlowMatchEulerDiscreteScheduler's shift parameters are
    # resolution-dependent, so this never borrows SDXL's scaled_linear betas.
    #
    # tensorfs#136 CORRECTED THE REASON. pgw#1393 recorded "HF-gated and could
    # not be fetched", which has since dissolved: both files are whole-file
    # entries in the hub's resolve manifest for tensorhub/flux1-dev and
    # tensorhub/flux1-schnell, and both were read. It stays empty on the
    # measured fact instead — the two checkpoints under this one root DISAGREE
    # (dev shift 3.0 with use_dynamic_shifting True; schnell shift 1.0 with it
    # False), so no single value is right for the root, and Klein's 3.0/dynamic
    # would be right for dev and wrong for schnell. A `{}` whose reason has
    # gone stale is the one the next reader clears wrongly.
    from gen_worker.models import Flux1, Flux2Klein

    assert Flux1.canonical_scheduler_config == {}
    # ...but FLUX.2 Klein's IS recorded now: black-forest-labs/FLUX.2-klein-4B
    # is the ONE BFL flux repo that is not HF-gated, so its shipped
    # scheduler_config.json was fetched verbatim (pgw#1393 follow-up).
    klein = Flux2Klein.canonical_scheduler_config
    assert klein["_class_name"] == "FlowMatchEulerDiscreteScheduler"
    # FLOW-MATCHING: a shift ladder, NOT a beta schedule. Asserting the
    # absence is the point — this is what "don't copy SDXL across" means.
    for beta_field in ("beta_start", "beta_end", "beta_schedule", "trained_betas"):
        assert beta_field not in klein
    assert (klein["base_shift"], klein["max_shift"], klein["shift"]) == (0.5, 1.15, 3.0)
    # The reason no frozen triple could have been invented: the effective
    # shift is a function of image sequence length.
    assert klein["use_dynamic_shifting"] is True
    assert (klein["base_image_seq_len"], klein["max_image_seq_len"]) == (256, 4096)
    assert klein["num_train_timesteps"] == 1000
    # Rife is the one AUXILIARY type: no canonical lane, and inventing a
    # tensorfs contract name for its diffusers-layout artifact would be a guess.
    assert Rife.canonical_contract is None
    assert MiniMaxH3.canonical_contract is not None


# ── the flux family (pgw#1393) ───────────────────────────────────────────────


def test_flux_platform_values_are_the_shipped_endpoint_numbers() -> None:
    """Every value cited to the flux endpoints' own source (pgw#1393)."""
    from gen_worker.models import Flux1, Flux2Klein

    f1 = Flux1.Defaults()
    # flux.1-dev/main.py:69-70 == flux.1-schnell/main.py:61-62, both under
    # register_family("flux1", ...) — the family owner saying dev and schnell
    # are ONE vocabulary.
    assert f1.steps == Knob(28, lo=1, hi=100, name="steps")
    assert f1.guidance == Knob(3.5, lo=0.0, hi=10.0, name="guidance")
    # flux.1-dev/main.py:277-280: guidance is the DISTILLATION EMBEDDING, a
    # DiT input tensor — not CFG. Both BFL checkpoints serve cfg-off.
    assert f1.cfg is False
    assert f1.step_distilled is False
    assert f1.max_sequence_length == 512  # :117

    k = Flux2Klein.Defaults()
    assert k.steps == Knob(28, lo=1, hi=50, name="steps")  # :84, :306
    assert k.guidance == Knob(4.0, lo=1.0, hi=10.0, name="guidance")  # :85, :310
    # flux.2-klein-4b/main.py:123-129, :307 — Klein Base runs a real second
    # uncond forward. The opposite of Flux1, which is why these are two types.
    assert k.cfg is True
    assert k.max_sequence_length == 512  # :121

    # Sourcing rule: no knob was invented where none could be sourced.
    for d in (f1, k):
        assert not hasattr(d, "scheduler")
        assert not hasattr(d, "timesteps")
    # No flux endpoint registers a lora vocabulary, so there is no overlay.
    assert not hasattr(Flux1, "Lora")
    assert not hasattr(Flux2Klein, "Lora")


def test_flux_platform_floors_admit_the_distilled_checkpoints() -> None:
    """The floors are the ENDPOINTS' checkpoint facts, not their wire bounds.

    ``_merge_*_knob`` only ever NARROWS and clamps a row's default into the
    platform range, so a platform floor copied from a Base handler's payload
    envelope silently rewrites the distilled checkpoint's own recipe. Both
    cases below were MEASURED failing before the floors were corrected.
    """
    from gen_worker.models import Flux1, Flux2Klein

    # flux.1-schnell/main.py:388-389 pins guidance_scale=0.0. Under dev's wire
    # ge=1.0 (flux.1-dev/main.py:281) this decoded to lo=1.0, hi=0.0 — empty.
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

    # flux.2-klein-4b/main.py:94-95: Turbo's published recipe is 4 steps at
    # guidance 1.0. The Base HANDLER declares ge=12 / ge=1.5 (:306, :310).
    turbo = decode_model_defaults(Flux2Klein, model="flux2-klein", defaults={
        "steps": {"default": 4},
        "guidance": {"default": 1.0},
        "cfg": False,
        "step_distilled": True,
    })
    assert turbo.steps.default == 4
    assert turbo.guidance.default == 1.0
    assert turbo.cfg is False and turbo.step_distilled is True

    # flux.1-schnell/main.py:267 — the Flex.2 lane's le=100 is why the platform
    # ceiling is 100 and not dev's 50: the merge cannot widen.
    flex2 = decode_model_defaults(Flux1, model="flux1", defaults={
        "steps": {"default": 28, "hi": 100}, "cfg": True,
    })
    assert flex2.steps.hi == 100 and flex2.cfg is True


def test_qwen_and_z_platform_values_are_the_shipped_endpoint_numbers() -> None:
    """pgw#1426. Every number below is the family's OWN v1 endpoint code."""
    from gen_worker.models import QwenImage, ZImage

    q = QwenImage.Defaults()
    # qwen-image/src/qwen_image/main.py:138-140.
    assert q.steps.default == 30 and q.guidance.default == 4.0
    assert q.negative == " ", "Qwen's uncond convention is a SPACE, never ''"
    assert q.cfg is True and q.step_distilled is False
    # :173 — the t2i pin; the edit checkpoint's ROW carries 1024 (:303).
    assert q.max_sequence_length == 512
    # v1's `max_guidance` clamp field does NOT come across: Knob.hi IS the clamp.
    assert not hasattr(q, "max_guidance")

    z = ZImage.Defaults()
    # z-image/src/z_image/main.py:90-91.
    assert z.steps.default == 28 and z.guidance.default == 4.0
    assert z.cfg is True and z.step_distilled is False
    assert z.max_sequence_length == 512  # :131

    # Neither family registers a lora vocabulary with a sourceable strength
    # range, so neither declares an overlay (the Flux1/Flux2Klein posture).
    assert not hasattr(QwenImage, "Lora")
    assert not hasattr(ZImage, "Lora")


def test_qwen_and_z_platform_floors_admit_their_distilled_checkpoints() -> None:
    """pgw#1426, and it is RED-CONTROLLED below rather than merely asserted.

    Both families ship a distilled sibling, so a platform floor copied from the
    BASE handler's wire envelope would silently rewrite the distilled row --
    ``_merge_*_knob`` only narrows, and clamps a row's default INTO the range.
    """
    from gen_worker.models import QwenImage, ZImage

    # z-image/src/z_image/main.py:245 — official Turbo's card recipe is 9
    # scheduler steps, and :702 PINS guidance 0.0. The base handler's wire
    # floor is ge=1.0 (:278); copying it would have made this row empty.
    dmd = decode_model_defaults(ZImage, model="z-image", defaults={
        "steps": {"default": 9}, "guidance": {"default": 0.0},
        "cfg": False, "step_distilled": True,
    })
    assert dmd.steps.default == 9
    assert dmd.guidance.default == 0.0 and dmd.guidance.lo == 0.0
    assert dmd.cfg is False and dmd.step_distilled is True

    # :244 — the PAI 2603 8-step distill reaches the same state as an overlay.
    pai = decode_model_defaults(ZImage, model="z-image", defaults={
        "steps": {"default": 8, "hi": 16}, "guidance": {"default": 0.0},
        "cfg": False, "step_distilled": True,
    })
    assert pai.steps.default == 8 and pai.steps.hi == 16

    # qwen-image/src/qwen_image/main.py:430 — the Lightning regime is 8 steps
    # with CFG off. The base handler declares ge=10 (:317) and ge=1.5 (:324);
    # either copied up would corrupt this row.
    lightning = decode_model_defaults(QwenImage, model="qwen-image", defaults={
        "steps": {"default": 8}, "guidance": {"default": 1.0},
        "cfg": False, "step_distilled": True,
    })
    assert lightning.steps.default == 8
    assert lightning.guidance.default == 1.0
    assert lightning.cfg is False and lightning.step_distilled is True

    # The edit checkpoint narrows ONLY the text pin (:303).
    edit = decode_model_defaults(QwenImage, model="qwen-image", defaults={
        "max_sequence_length": 1024,
    })
    assert edit.max_sequence_length == 1024
    assert edit.steps.default == 30, "the edit row shares the family recipe"


def test_a_base_handlers_wire_floor_really_would_corrupt_a_distilled_row() -> None:
    """The RED control for the test above: prove the hazard is real HERE, so
    the floors are known-good rather than merely never-exercised.

    An assertion that has never failed is not known to be able to fail, and the
    corruption is silent by construction -- no exception, just a different
    recipe -- so nothing else would have caught a mechanical port.
    """
    from gen_worker.models.defaults_decode import decode_defaults

    class PortedZImage(msgspec.Struct, frozen=True):
        """z-image's BASE wire floor ge=1.0 (main.py:278) ported as-is."""

        guidance: Knob[float] = Knob(4.0, lo=1.0, hi=15.0, name="guidance")

    class PortedQwen(msgspec.Struct, frozen=True):
        """qwen's BASE wire floor ge=10 (main.py:317) ported as-is."""

        steps: Knob[int] = Knob(30, lo=10, hi=80, name="steps")

    # Against Turbo's pinned guidance 0.0.
    corrupted = decode_defaults(PortedZImage, {"guidance": {"default": 0.0}})
    assert corrupted.guidance.default == 1.0, "the port silently serves CFG on"

    # Against Lightning's 8 steps.
    corrupted2 = decode_defaults(PortedQwen, {"steps": {"default": 8}})
    assert corrupted2.steps.default == 10, "the port silently serves 10 steps"

    # And the SHIPPED floors do not do that -- the same two rows, decoded
    # through the real vocabularies, keep their own values.
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
