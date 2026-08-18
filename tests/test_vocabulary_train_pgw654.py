"""pgw#654 train mechanics: the class-union compile contract (pgw#647 gap
#1 + gap #6 dual text pins), the multi-slot root rule (gap #7), the
lora_state_dict lint (gap #9), the DERIVED warm plan (axis cross-product,
sibling dedup, warm overrides), ``ctx.adjusted``/``ctx.clamp``, and
RuntimeFormula's resolved-effective evaluation (gap #4). Real registry/
warmup/context code under test; no torch needed.
"""

from __future__ import annotations

from typing import Annotated, Optional

import msgspec
import pytest

from gen_worker import (
    AxisClass,
    Compile,
    CompileAxis,
    RequestContext,
    Resources,
    Slot,
    endpoint,
    worker_function,
)
from gen_worker.api.formula import RuntimeFormula
from gen_worker.families.base import GenerationDefaults, register_family
from gen_worker.registry import extract_specs

_GUIDANCE_AXIS = CompileAxis(classes=(
    AxisClass("cfg_off", match=lambda v: v == 0, warm=0.0),
    AxisClass("cfg_on", match=lambda v: v != 0, warm=5.0),
))


class GenIn(msgspec.Struct):
    prompt: str
    aspect_ratio: Annotated[str, CompileAxis(classes="enum")] = "1:1"
    guidance_scale: Annotated[Optional[float], _GUIDANCE_AXIS] = None
    steps: int = 2


class TurboIn(msgspec.Struct):
    # The distilled contract: NO guidance field — its graph class
    # is the one containing 0 (no wire CFG == batch-1).
    prompt: str
    aspect_ratio: Annotated[str, CompileAxis(classes="enum")] = "1:1"


class Out(msgspec.Struct):
    y: str


# Constrain the enum axis to two members so cross-products stay readable.
GenIn.__annotations__["aspect_ratio"] = Annotated[
    str, CompileAxis(classes=(
        AxisClass("sq", match=lambda v: v == "1:1", warm="1:1"),
        AxisClass("wide", match=lambda v: v == "16:9", warm="16:9"),
    ))
]
TurboIn.__annotations__["aspect_ratio"] = GenIn.__annotations__["aspect_ratio"]


@endpoint(
    models={"pipeline": Slot(object, selected_by="")},
    resources=Resources(gpu=True),
    compile=Compile(family="fam-x", shapes=((1024, 1024),), targets=("unet",), text_len=77),
)
class ToyEndpoint:
    def setup(self, pipeline: object) -> None:
        self.pipeline = pipeline

    @worker_function(objectives=("epsilon", "v_prediction"), distilled=False)
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        return Out(y="g")

    @worker_function(objectives=("epsilon",))
    def generate_turbo(self, ctx: RequestContext, p: TurboIn) -> Out:
        return Out(y="t")


# ---------------------------------------------------------------------------
# gap #1: sibling functions share ONE compile contract via the class union
# ---------------------------------------------------------------------------


def test_sibling_functions_share_one_compile_contract() -> None:
    specs = {s.attr_name: s for s in extract_specs(ToyEndpoint)}
    a = specs["generate"].compile_contract()
    b = specs["generate_turbo"].compile_contract()
    assert a.guidance_scales == b.guidance_scales == (0.0, 5.0)
    assert a.contract_digest() == b.contract_digest()


def test_union_is_class_scoped_not_cross_class() -> None:
    # Sibling @endpoint CLASSES keep divergent contracts by design (ernie's
    # base/turbo two-class shape): a turbo-only class has NO guidance union.
    @endpoint(
        models={"pipeline": Slot(object)},
        resources=Resources(gpu=True),
        compile=Compile(family="fam-solo", shapes=((512, 512),), targets=("unet",), text_len=77),
    )
    class TurboOnly:
        def setup(self, pipeline: object) -> None:
            self.pipeline = pipeline

        def generate_turbo(self, ctx: RequestContext, p: TurboIn) -> Out:
            return Out(y="t")

    specs = {s.attr_name: s for s in extract_specs(TurboOnly)}
    assert specs["generate_turbo"].compile_contract().guidance_scales == ()


# ---------------------------------------------------------------------------
# gap #6: per-function text pins union into the compiled graph contract
# ---------------------------------------------------------------------------


class EditIn(msgspec.Struct):
    prompt: str


@endpoint(
    models={"t2i": Slot(object, root=True), "edit": Slot(object)},
    resources=Resources(gpu=True),
    compile=Compile(family="fam-dual", shapes=((1024, 1024),), targets=("transformer",), text_len=512),
)
class DualPin:
    def setup(self, t2i: object, edit: object) -> None:
        self.t2i, self.edit = t2i, edit

    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        return Out(y="g")

    @worker_function(text_len=1024)
    def edit_image(self, ctx: RequestContext, p: EditIn) -> Out:
        return Out(y="e")


def test_per_function_text_pins_digest_as_one_dual_pin_contract() -> None:
    specs = {s.attr_name: s for s in extract_specs(DualPin)}
    a = specs["generate"].compile_contract()
    b = specs["edit_image"].compile_contract()
    # Each function keeps ITS effective pin (warm calls trace through it)...
    assert a.text_len == 512
    assert b.text_len == 1024
    # ...but the contract digests the class UNION, so the compiled graph is shared.
    assert a.contract_text_lens() == b.contract_text_lens() == (512, 1024)
    assert a.contract_digest() == b.contract_digest()


def test_text_len_override_requires_compile() -> None:
    with pytest.raises(ValueError, match="text_len"):
        @endpoint(models={"pipeline": Slot(object)})
        class NoCompile:
            def setup(self, pipeline: object) -> None:
                self.pipeline = pipeline

            @worker_function(text_len=512)
            def generate(self, ctx: RequestContext, p: EditIn) -> Out:
                return Out(y="g")

        extract_specs(NoCompile)


# ---------------------------------------------------------------------------
# gap #7: multi-slot root is a loud decoration-time requirement
# ---------------------------------------------------------------------------


def test_multi_slot_without_root_is_a_decoration_error() -> None:
    with pytest.raises(ValueError, match="root"):
        @endpoint(models={"t2i": Slot(object), "edit": Slot(object)})
        class Ambiguous:
            def setup(self, t2i: object, edit: object) -> None:
                self.t2i, self.edit = t2i, edit

            def generate(self, ctx: RequestContext, p: EditIn) -> Out:
                return Out(y="g")


def test_two_roots_is_a_decoration_error() -> None:
    with pytest.raises(ValueError, match="more than one"):
        @endpoint(models={"a": Slot(object, root=True), "b": Slot(object, root=True)})
        class TwoRoots:
            def setup(self, a: object, b: object) -> None:
                self.a, self.b = a, b

            def generate(self, ctx: RequestContext, p: EditIn) -> Out:
                return Out(y="g")


def test_pipeline_named_slot_is_the_implicit_root() -> None:
    @endpoint(models={"pipeline": Slot(object), "refiner": Slot(object)})
    class Implicit:
        def setup(self, pipeline: object, refiner: object) -> None:
            self.pipeline, self.refiner = pipeline, refiner

        def generate(self, ctx: RequestContext, p: EditIn) -> Out:
            return Out(y="g")

    assert extract_specs(Implicit)


def test_ctx_root_resolution_honors_declared_root_and_explicit_slot() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import resolve_slot
    from _example_family import ExampleDefaults

    t2i = resolve_slot("t2i", Slot(object), ref=HF("acme/t2i"),
                       defaults_cls=ExampleDefaults, objective="flow")
    edit = resolve_slot("edit", Slot(object), ref=HF("acme/edit"),
                        defaults_cls=ExampleDefaults, objective="flow")
    ctx: RequestContext = RequestContext(
        request_id="r1", resolved_slots={"t2i": t2i, "edit": edit},
        root_slot="t2i",
    )
    assert ctx._root_slot() is t2i
    assert ctx._root_slot("edit") is edit


def test_ctx_ambiguous_root_raises_never_falls_back() -> None:
    from gen_worker.api.binding import HF
    from gen_worker.api.slot import resolve_slot
    from _example_family import ExampleDefaults

    a = resolve_slot("a", Slot(object), ref=HF("acme/a"), defaults_cls=ExampleDefaults)
    b = resolve_slot("b", Slot(object), ref=HF("acme/b"), defaults_cls=ExampleDefaults)
    ctx: RequestContext = RequestContext(
        request_id="r1", resolved_slots={"a": a, "b": b},
    )
    with pytest.raises(ValueError, match="root"):
        ctx._root_slot()
    with pytest.raises(ValueError, match="root"):
        _ = ctx.defaults


# ---------------------------------------------------------------------------
# gap #9: non-introspectable lora_bucket root must declare lora_state_dict
# ---------------------------------------------------------------------------


class _CustomPipe:
    pass


class _ConvertingPipe:
    @classmethod
    def lora_state_dict(cls, src: object) -> dict:  # pragma: no cover - shape only
        return {}


def test_lora_bucket_without_converter_fails_decoration() -> None:
    with pytest.raises(ValueError, match="lora_state_dict"):
        @endpoint(models={"pipeline": Slot(_CustomPipe)}, lora_bucket=64)
        class NoConverter:
            def setup(self, pipeline: _CustomPipe) -> None:
                self.pipeline = pipeline

            def generate(self, ctx: RequestContext, p: EditIn) -> Out:
                return Out(y="g")


def test_lora_bucket_with_converter_decorates() -> None:
    @endpoint(models={"pipeline": Slot(_ConvertingPipe)}, lora_bucket=64)
    class WithConverter:
        def setup(self, pipeline: _ConvertingPipe) -> None:
            self.pipeline = pipeline

        def generate(self, ctx: RequestContext, p: EditIn) -> Out:
            return Out(y="g")

    assert extract_specs(WithConverter)


# ---------------------------------------------------------------------------
# derived warm plan: cross-product + sibling dedup + overrides
# ---------------------------------------------------------------------------


def test_derived_warm_plan_cross_products_axes_per_function() -> None:
    from gen_worker import warmup as warmup_mod

    specs = extract_specs(ToyEndpoint)
    jobs, skips = warmup_mod.plan(specs, decl_warmup=None)
    assert not skips
    by_fn: dict = {}
    for job in jobs:
        by_fn.setdefault(job.spec.attr_name, []).append(job)
    # generate: 2 aspect classes x 2 guidance classes = 4 graph runs.
    assert len(by_fn["generate"]) == 4
    # turbo runs its OWN aspect classes (causal per-alias proof) — its
    # missing guidance field classifies as the class containing 0, so the
    # graph itself is generate's cfg_off (cache hit, no second trace).
    assert len(by_fn["generate_turbo"]) == 2
    for j in by_fn["generate_turbo"]:
        assert dict(j.graph_key[1:])["guidance_scale"] == "cfg_off"
    # Payloads carry the axis warm representatives.
    built = [j.build("/tmp") for j in by_fn["generate"]]
    combos = {(p.aspect_ratio, p.guidance_scale) for p in built}
    assert combos == {("1:1", 0.0), ("1:1", 5.0), ("16:9", 0.0), ("16:9", 5.0)}
    # Required strs synthesize; step fields clamp to their declared floor
    # (pgw#654 warm-tax fix — a warm run never pays a full recipe's steps).
    assert all(p.steps == 1 and p.prompt == "warmup" for p in built)


def test_warm_override_applies_to_non_axis_fields_only() -> None:
    from gen_worker import warmup as warmup_mod

    class FramesIn(msgspec.Struct):
        prompt: str
        frames: int = 9

    @endpoint(models={"pipeline": Slot(object)}, resources=Resources(gpu=True))
    class Video:
        def setup(self, pipeline: object) -> None:
            self.pipeline = pipeline

        @worker_function(warm={"frames": 17},
                         warm_reason="frame count changes the traced graph")
        def generate(self, ctx: RequestContext, p: "FramesIn") -> Out:
            return Out(y="v")

    # Local struct: patch hints resolution by exporting to module globals.
    globals()["FramesIn"] = FramesIn
    specs = extract_specs(Video)
    jobs, _ = warmup_mod.plan(specs, decl_warmup=None)
    assert len(jobs) == 1 and jobs[0].declared
    assert jobs[0].build("/tmp").frames == 17


def test_warm_override_on_axis_field_is_a_walk_time_error() -> None:
    with pytest.raises(ValueError, match="AXIS"):
        @endpoint(models={"pipeline": Slot(object)})
        class BadWarm:
            def setup(self, pipeline: object) -> None:
                self.pipeline = pipeline

            @worker_function(warm={"guidance_scale": 3.0}, warm_reason="nope")
            def generate(self, ctx: RequestContext, p: GenIn) -> Out:
                return Out(y="g")

        extract_specs(BadWarm)


def test_dual_pin_execution_lanes_never_dedupe_across_text_pins() -> None:
    from gen_worker import warmup as warmup_mod

    specs = extract_specs(DualPin)
    jobs, _ = warmup_mod.plan(specs, decl_warmup=None)
    names = {j.spec.attr_name for j in jobs}
    # Different text pins are different graphs: the edit lane keeps its run.
    assert "edit_image" in names and "generate" in names


# ---------------------------------------------------------------------------
# ctx.adjusted / ctx.clamp — the caller-visible adjustment primitive
# ---------------------------------------------------------------------------


def test_adjusted_records_structured_rows() -> None:
    ctx: RequestContext = RequestContext(request_id="r1")
    ctx.adjusted("guidance_scale", 15, 10, "model maximum")
    assert ctx._adjustments == [{
        "field": "guidance_scale", "requested": "15", "applied": "10",
        "reason": "model maximum",
    }]


def test_clamp_emits_only_when_the_value_changes() -> None:
    ctx: RequestContext = RequestContext(request_id="r1")
    assert ctx.clamp("guidance_scale", 7.0, hi=10.0) == 7.0
    assert ctx._adjustments == []
    assert ctx.clamp("guidance_scale", 15.0, hi=10.0, reason="model maximum") == 10.0
    assert len(ctx._adjustments) == 1
    assert ctx._adjustments[0]["applied"] == "10.0"


# ---------------------------------------------------------------------------
# gap #4: RuntimeFormula validates + evaluates on resolved effective values
# ---------------------------------------------------------------------------


class FormulaDefaults(GenerationDefaults, frozen=True):
    num_inference_steps: Optional[int] = None


register_family("fam-formula", FormulaDefaults)


class FormulaIn(msgspec.Struct):
    prompt: str
    num_inference_steps: Optional[int] = None
    megapixels: float = 1.0


def test_formula_accepts_recipe_resolved_fields() -> None:
    rf = RuntimeFormula("a + b*num_inference_steps + c*num_inference_steps*megapixels")
    # Wire default is None -> old validation refused; the derived defaults
    # schema resolves it (the v2 Optional -> ctx.defaults pattern).
    rf.validate_for_payload(FormulaIn, "owner", defaults_type=FormulaDefaults)


def test_formula_still_refuses_unresolvable_fields() -> None:
    rf = RuntimeFormula("a + b*num_inference_steps")

    class Bare(msgspec.Struct):
        prompt: str
        num_inference_steps: Optional[int] = None

    with pytest.raises(ValueError, match="defaults schema"):
        rf.validate_for_payload(Bare, "owner", defaults_type=None)


def test_formula_evaluates_payload_over_defaults() -> None:
    rf = RuntimeFormula("a + b*num_inference_steps")
    defaults = FormulaDefaults(num_inference_steps=28)
    # Omitted wire value -> the resolved recipe's value.
    vals = rf.term_values_from_struct(FormulaIn(prompt="x"), defaults)
    assert vals == {"1": 1.0, "num_inference_steps": 28.0}
    # Explicit payload value wins.
    vals = rf.term_values_from_struct(
        FormulaIn(prompt="x", num_inference_steps=4), defaults)
    assert vals == {"1": 1.0, "num_inference_steps": 4.0}


# ---------------------------------------------------------------------------
# ModelRef.label — the one model_used/logging label (endpoints delete theirs)
# ---------------------------------------------------------------------------


def test_model_ref_label_mirrors_registry_wire_forms() -> None:
    from gen_worker import HF, Civitai, Hub

    assert HF("acme/gonzalomo-xl").label == "acme/gonzalomo-xl"
    # th#1987 deleted the default AND the elision: every release is stamped.
    assert Hub("acme/wai-illustrious", release="prod").label \
        == "acme/wai-illustrious@prod"
    assert Hub("acme/wai-illustrious", release="latest").label \
        == "acme/wai-illustrious@latest"
    assert Civitai("civitai/123", version="456").label == "civitai/123@456"


def test_model_ref_label_is_injective_ie727() -> None:
    """ie#727's residue: the old fold appended the release only when the label
    carried no `@` at all, so two distinct bindings could mint one label."""
    from gen_worker import HF, Civitai, Hub, ModelScope
    from gen_worker.api.binding import ModelRef, wire_ref

    # The side channel WINS over a release riding the ref (Hub's own contract),
    # and `label` now says the same artifact `wire_ref` fetches. The old fold
    # saw the `@` already in `path` and reported the ref that was NOT fetched.
    stale = Hub("acme/wai-illustrious@old", release="new")
    assert stale.label == "acme/wai-illustrious@new" == str(wire_ref(stale))

    # A digest pin keeps the digest, not the release it was picked from.
    hexd = "ab" * 32
    pinned = Hub(f"acme/repo@sha256:{hexd}", release="prod")
    assert pinned.label == f"acme/repo@sha256:{hexd}" == str(wire_ref(pinned))

    # Distinct revisions are distinct labels — HF's axis reached `label` only
    # through `release`, which HF never sets, so every revision read as the
    # bare repo.
    assert HF("acme/xl", revision="r1").label == "acme/xl@r1"
    assert HF("acme/xl", revision="r2").label == "acme/xl@r2"
    assert ModelScope("acme/anima", revision="v3").label == "acme/anima@v3"

    labels = {
        b.label
        for b in (
            Hub("acme/repo", release="prod"),
            Hub("acme/repo", release="staging"),
            HF("acme/repo", revision="r1"),
            Civitai("123", version="456"),
            Civitai("123", version="789"),
            Civitai("123"),
        )
    }
    assert len(labels) == 6

    # `release` is the tensorhub axis and nothing else fetches by it: refused
    # where it cannot mean anything, rather than dropped inside `label`.
    with pytest.raises(ValueError, match="no release axis"):
        ModelRef(source="civitai", path="123", version="456", release="prod")
