"""SDK v2 surface (pgw#647): derived component tree, sibling-as-part lint,
CompileAxis payload classes, the text-sequence lint, Resources v2, derived
config schema, per-request views, and the text pin helper.

Real code paths per th#1105 — the registry walk, the decorator, the view
machinery and the pad helper run for real; only CUDA is absent.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional

import msgspec
import pytest

from gen_worker import (
    AxisClass,
    Compile,
    CompileAxis,
    DynamicDim,
    Hub,
    RequestContext,
    Resources,
    Slot,
    TextLengthExceededError,
    endpoint,
    pad_text_sequence,
)
from gen_worker.api.compile_axis import extract_payload_axes, warm_guidance_values
from gen_worker.api.slot import resolve_slot
from gen_worker.api.tree import (
    KIND_CONFIG,
    KIND_WEIGHTS,
    derive_components,
    is_introspectable,
    validate_no_sibling_parts,
)
from gen_worker.families.base import GenerationDefaults, family
from gen_worker.registry import extract_specs
from gen_worker.view import UnknownSamplerError, clone_scheduler, for_request


@family("v2-testfam")
class _V2Defaults(GenerationDefaults, frozen=True):
    steps: int = 28
    guidance: float = 6.0


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _Out(msgspec.Struct):
    ok: bool = True


# ---------------------------------------------------------------------------
# Component tree derivation.
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Diffusers-shaped: self-describes via _get_signature_keys."""

    @classmethod
    def _get_signature_keys(cls, obj):
        return {"unet", "vae", "text_encoder", "tokenizer", "scheduler"}, set()

    @classmethod
    def from_pretrained(cls, path, **kw):
        return cls()


def test_tree_derives_and_classifies_parts():
    tree = derive_components(_FakePipeline)
    assert tree == {
        "unet": KIND_WEIGHTS,
        "vae": KIND_WEIGHTS,
        "text_encoder": KIND_WEIGHTS,
        "tokenizer": KIND_CONFIG,
        "scheduler": KIND_CONFIG,
    }


def test_str_and_plain_classes_are_not_introspectable():
    assert not is_introspectable(str)
    assert derive_components(str) is None

    class Plain:
        pass

    assert derive_components(Plain) is None


def test_real_diffusers_pipeline_tree_derives():
    diffusers = pytest.importorskip("diffusers")
    tree = derive_components(diffusers.StableDiffusionXLPipeline)
    assert tree is not None
    assert tree.get("unet") == KIND_WEIGHTS
    assert tree.get("vae") == KIND_WEIGHTS
    assert tree.get("scheduler") == KIND_CONFIG
    assert tree.get("tokenizer") == KIND_CONFIG


def test_sibling_component_slot_is_rejected():
    slots = {
        "pipeline": Slot(_FakePipeline),
        "vae": Slot(_FakePipeline),  # 'vae' IS a part of pipeline's tree
    }
    with pytest.raises(ValueError, match="COMPONENT.*catalog data"):
        validate_no_sibling_parts("Owner.gen", slots, {})


def test_sibling_str_slot_next_to_tree_is_rejected():
    slots = {
        "pipeline": Slot(_FakePipeline),
        "turbo_lora": Slot(str),
    }
    with pytest.raises(ValueError, match="sibling-as-part|adapters riding"):
        validate_no_sibling_parts("Owner.gen", slots, {})


def test_all_str_multi_slot_escape_hatch_survives():
    slots = {"weights": Slot(str), "mmproj": Slot(str)}
    validate_no_sibling_parts("Owner.gen", slots, {})  # no raise


def test_registry_records_derived_tree():
    @endpoint(models={"pipeline": Slot(_FakePipeline, selected_by="model")})
    class Gen:
        def setup(self, pipeline: _FakePipeline) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext[_V2Defaults], p: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(Gen)
    assert spec.slot_components["pipeline"]["unet"] == KIND_WEIGHTS
    assert spec.defaults_type is _V2Defaults
    assert spec.slot_family["pipeline"] == "v2-testfam"


# ---------------------------------------------------------------------------
# CompileAxis payload classes.
# ---------------------------------------------------------------------------


class _AxisIn(msgspec.Struct):
    prompt: str = ""
    guidance_scale: Annotated[float, CompileAxis(classes=(
        AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
        AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
    ))] = 5.0
    aspect: Annotated[str, CompileAxis(classes="enum")] = "1:1"


def test_axis_extraction_and_classification():
    with pytest.raises(ValueError):
        extract_payload_axes("owner", _AxisIn)  # plain str is not enumerable

    class _EnumIn(msgspec.Struct):
        guidance_scale: Annotated[float, CompileAxis(classes=(
            AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
            AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
        ))] = 5.0

    axes = extract_payload_axes("owner", _EnumIn)
    assert [a.field for a in axes] == ["guidance_scale"]
    axis = axes[0]
    assert axis.class_names == ("cfg_off", "cfg_on")
    assert axis.classify(0.0) == "cfg_off"
    assert axis.classify(7.5) == "cfg_on"
    assert warm_guidance_values(axes) == (0.0, 5.0)


def test_enum_axis_from_literal():
    class _LitIn(msgspec.Struct):
        aspect: Annotated[Literal["1:1", "3:4", "16:9"], CompileAxis(classes="enum")] = "1:1"

    (axis,) = extract_payload_axes("owner", _LitIn)
    assert axis.class_names == ("1:1", "3:4", "16:9")
    assert axis.classify("3:4") == "3:4"
    assert axis.classify("9:16") is None  # out of contract


def test_axis_warm_must_satisfy_its_own_match():
    with pytest.raises(ValueError, match="does not satisfy"):
        AxisClass("cfg_off", match=lambda v: v == 0, warm=5.0)


# ---------------------------------------------------------------------------
# The text-sequence lint + shape-contract declaration.
# ---------------------------------------------------------------------------


def test_compile_without_text_axis_is_a_decoration_error():
    with pytest.raises(ValueError, match="text-sequence axis"):
        @endpoint(compile=Compile(shapes=((1024, 1024),), family="v2fam"))
        class Gen:
            def setup(self) -> None:
                pass

            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()


def test_compile_with_pinned_text_len_passes():
    @endpoint(compile=Compile(shapes=((1024, 1024),), family="v2fam", text_len=77))
    class Gen:
        def setup(self) -> None:
            pass

        def generate(self, ctx: RequestContext, p: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(Gen)
    assert spec.compile is not None and spec.compile.text_len == 77


def test_compile_with_dynamic_sequence_passes_and_batch_axis_supported():
    @endpoint(compile=Compile(
        shapes=((1024, 1024),), family="v2fam",
        dynamic=(DynamicDim("sequence", min=64, max=512),
                 DynamicDim("batch", min=2, max=16)),
    ))
    class Gen:
        def setup(self) -> None:
            pass

        def generate(self, ctx: RequestContext, p: _In) -> _Out:
            return _Out()

    (spec,) = extract_specs(Gen)
    cell = spec.compile_cell()
    assert cell is not None
    facts = cell.contract_facts()
    assert facts["dynamic"] == [
        {"dim": "sequence", "min": 64, "max": 512},
        {"dim": "batch", "min": 2, "max": 16},
    ]


def test_dynamic_dim_min_must_respect_01_specialization():
    # torch's 0/1 specialization is not overridable (ie#543): min >= 2.
    with pytest.raises(ValueError, match="min must be >= 2"):
        DynamicDim("batch", min=1, max=8)


def test_contract_digest_changes_with_the_contract():
    def cell(**kw):
        @endpoint(compile=Compile(shapes=((1024, 1024),), family="v2fam", **kw))
        class Gen:
            def setup(self) -> None:
                pass

            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()

        (spec,) = extract_specs(Gen)
        return spec.compile_cell().contract_digest()

    a = cell(text_len=77)
    b = cell(text_len=512)
    c = cell(text_len=0)
    assert len({a, b, c}) == 3  # pre-fix and post-fix cells key differently


# ---------------------------------------------------------------------------
# Resources v2.
# ---------------------------------------------------------------------------


def test_resources_v2_deleted_fields_raise():
    # `vram_gb` and `ram_gb` stay deleted; their successors are the explicitly
    # non-binding `vram_gb_hint` / `ram_gb_hint` (pgw#670).
    for kw in ({"vram_gb": 12}, {"ram_gb": 48}, {"min_compute_capability": 8.0}):
        with pytest.raises(TypeError):
            Resources(**kw)


def test_resources_v2_compute_capability_is_restored_as_a_hard_floor():
    # pgw#660 REVERSED this part of the v2 cut (Paul, 2026-07-26). The cut was
    # right that precision-per-card is the ladder's call and wrong that an
    # architecture floor is the same thing: scaled_mm below sm_89 is
    # incapability, not a slower rung, and the hub placed the fp8 producer on
    # sm_80 A100s for want of the declaration. See test_compute_capability_pgw660.
    assert Resources(compute_capability=8.9).compute_capability == 8.9
    assert Resources(compute_capability=8.9).gpu is True


def test_resources_v2_hint_and_gpu_count_imply_gpu():
    assert Resources(vram_gb_hint=8).gpu is True
    assert Resources(gpu_count=2).gpu is True
    assert Resources().gpu is False
    with pytest.raises(ValueError):
        Resources(gpu_count=0)


# ---------------------------------------------------------------------------
# Derived config schema + neutral resolution.
# ---------------------------------------------------------------------------


def test_resolve_slot_neutral_defaults_without_catalog_metadata():
    slot = Slot(str, default_checkpoint=Hub("acme/ckpt"))
    resolved = resolve_slot(
        "pipeline", slot, ref=slot.default_checkpoint,
        defaults_cls=_V2Defaults, family="v2-testfam",
    )
    assert resolved.defaults == _V2Defaults()


def test_resolve_slot_catalog_metadata_wins():
    slot = Slot(str, default_checkpoint=Hub("acme/ckpt"))
    resolved = resolve_slot(
        "pipeline", slot, ref=slot.default_checkpoint,
        defaults_cls=_V2Defaults, family="v2-testfam",
        raw_metadata_json='{"steps": 8, "guidance": 1.0}',
    )
    assert resolved.defaults.steps == 8
    assert resolved.defaults.guidance == 1.0


def test_ctx_defaults_reads_root_slot():
    from gen_worker.api.slot import ResolvedSlot

    ctx = RequestContext(
        "req-1",
        resolved_slots={"pipeline": ResolvedSlot(
            ref=Hub("acme/ckpt"), defaults=_V2Defaults(steps=12))},
    )
    assert ctx.defaults.steps == 12


def test_bad_ctx_annotation_is_a_walk_error():
    with pytest.raises(ValueError, match="GenerationDefaults"):
        @endpoint
        class Gen:
            def generate(self, ctx: "RequestContext[int]", p: _In) -> _Out:  # noqa: F821
                return _Out()

        extract_specs(Gen)


# ---------------------------------------------------------------------------
# v2 handler contract: (self, ctx, payload) only; setup required for models.
# ---------------------------------------------------------------------------


def test_extra_handler_params_are_rejected():
    with pytest.raises(TypeError, match="exactly \\(self, ctx, payload\\)"):
        @endpoint(models={"pipeline": Hub("acme/ckpt")})
        class Gen:
            def setup(self, pipeline: str) -> None:
                self.pipeline = pipeline

            def generate(self, ctx: RequestContext, p: _In, pipeline: str) -> _Out:
                return _Out()


def test_class_models_require_setup():
    with pytest.raises(ValueError, match="require a setup"):
        @endpoint(models={"pipeline": Hub("acme/ckpt")})
        class Gen:
            def generate(self, ctx: RequestContext, p: _In) -> _Out:
                return _Out()


# ---------------------------------------------------------------------------
# Per-request views.
# ---------------------------------------------------------------------------


class _FakeSchedulerA:
    def __init__(self, config=None, **overrides):
        self.config = dict(config or {"num_train_timesteps": 1000})
        self.config.update(overrides)
        self.step_index = 0

    @classmethod
    def from_config(cls, config, **overrides):
        return cls(dict(config), **overrides)


class _FakeUnet:
    pass


class _ViewPipeline:
    def __init__(self):
        self.unet = _FakeUnet()
        self.vae = _FakeUnet()
        self.scheduler = _FakeSchedulerA()
        self._internal_dict = {"scheduler": ("x", "FakeSchedulerA")}


def test_for_request_shares_modules_and_owns_scheduler():
    pipe = _ViewPipeline()
    pipe.scheduler.step_index = 17  # dirty state from a previous request
    view = for_request(pipe)
    assert view.unet is pipe.unet          # weights shared by reference
    assert view.vae is pipe.vae
    assert view.scheduler is not pipe.scheduler  # scheduler is view-private
    assert view.scheduler.step_index == 0        # fresh trajectory
    assert pipe.scheduler.step_index == 17       # instance untouched
    assert type(view) is type(pipe)


def test_for_request_applies_v_prediction_objective():
    pipe = _ViewPipeline()
    view = for_request(pipe, objective="v_prediction")
    assert view.scheduler.config["prediction_type"] == "v_prediction"
    assert view.scheduler.config["rescale_betas_zero_snr"] is True
    assert "prediction_type" not in pipe.scheduler.config


def test_clone_scheduler_unknown_sampler_is_typed():
    pipe = _ViewPipeline()
    with pytest.raises(UnknownSamplerError, match="known:"):
        clone_scheduler(pipe, sampler="not-a-sampler")


def test_for_request_sampler_swap_with_real_diffusers():
    diffusers = pytest.importorskip("diffusers")
    pipe = _ViewPipeline()
    pipe.scheduler = diffusers.EulerDiscreteScheduler()
    view = for_request(pipe, sampler="euler_a")
    assert type(view.scheduler).__name__ == "EulerAncestralDiscreteScheduler"
    assert type(pipe.scheduler).__name__ == "EulerDiscreteScheduler"


# ---------------------------------------------------------------------------
# Text pin helper (ie#544 absorbed).
# ---------------------------------------------------------------------------


def test_pad_text_sequence_is_canonically_strided():
    torch = pytest.importorskip("torch")
    # dim 0 of size 1 — exactly the case where .contiguous() is a no-op and
    # a size-only pin is not a pin (ie#544).
    embeds = torch.randn(1, 33, 8)[:, :, :]
    mask = torch.ones(1, 33, dtype=torch.long)
    out, out_mask = pad_text_sequence(embeds, 77, mask=mask)
    assert tuple(out.shape) == (1, 77, 8)
    assert out.stride() == (77 * 8, 8, 1)  # canonical row-major strides
    assert torch.equal(out[:, :33, :], embeds)
    assert torch.equal(out[:, 33:, :], torch.zeros(1, 44, 8))
    assert tuple(out_mask.shape) == (1, 77)
    assert int(out_mask.sum()) == 33


def test_pad_text_sequence_over_length_is_typed_refusal():
    torch = pytest.importorskip("torch")
    embeds = torch.randn(1, 100, 8)
    with pytest.raises(TextLengthExceededError):
        pad_text_sequence(embeds, 77)
