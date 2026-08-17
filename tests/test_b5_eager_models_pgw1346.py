"""pgw#1346 B5 — the boundary endpoints' eager declarations, and the tier that
makes them expressible.

Three claims, and each one had to be measured rather than assumed:

1. **The eager tier could not reach a handler at all.** ``@endpoint(models=)``
   binds a generated ``Model`` subclass (``Bind`` refuses anything else), and
   the only producer of one was ``codegen.render``, which requires a
   ``ModelExport`` — a document that REFUSES zero runners and demands a loop
   and a tuned reference. So eleven honest declarations would have had nowhere
   to land. ``eager_model_v1`` + ``render_eager`` is that gap closed with the
   W1b lane's own conventions: a committed document, a pure renderer, a byte
   fence.
2. **The floors migrate BY VALUE.** ie#740's scalars are production incidents;
   this suite asserts the PARSED NUMBERS, not the strings, for every B5
   declaration that carries one.
3. **The four A19 ``layouts_undeclarable`` reasons survive.** ``ModelSpec``'s
   only alternative is ``DEFAULT_LAYOUT = "bf16"``, which would label GGUF and
   pickle bytes as bf16 — worse than losing the field, which is K2's finding.
"""

from __future__ import annotations

import json

import msgspec
import pytest

from gen_worker._vendor.torchcg.recipe import (
    FamilyName,
    LoopKind,
    RecipeError,
    parse_layout_contract,
)
from gen_worker.families.base import GenerationDefaults
from gen_worker.model.bind import Bind
from gen_worker.model.codegen import class_name, render_eager_module
from gen_worker.model.errors import ModelError, ModelRefusal
from gen_worker.model.export import export_eager_model, export_model
from gen_worker.model.snapshot import EagerExport, ExportedLoop, ModelExport, TunedRef
from gen_worker.model.spec import (
    Bucket,
    CallExample,
    GraphModelSpec,
    Loop,
    ModelSpec,
    Runner,
    Stage,
    TunedValues,
)
from gen_worker.model.runtime import Model
from gen_worker.model.tuned import tuned_from_catalog
from gen_worker.models.tensor_layout_contract import (
    LayoutDeclarationError,
    validate_layout_handle,
)

from gen_worker.model.catalog.boundary_3d import HUNYUAN3D, TRELLIS_3D, Hunyuan3dTuned
from gen_worker.model.catalog.boundary_audio import (
    CHATTERBOX,
    FOUNDATION_1,
    MUSICGEN,
    STABLE_AUDIO_OPEN,
)
from gen_worker.model.catalog.boundary_llm import (
    INTERNVL_U,
    JOYCAPTION,
    QWEN36_27B_MTP,
    QWEN36_35B_A3B,
)
from gen_worker.model.catalog.flex2_preview import FLEX2_PREVIEW

#: Every B5 weight-bearing declaration, by its family handle.
B5 = {
    spec.name: spec
    for spec in (
        QWEN36_35B_A3B,
        QWEN36_27B_MTP,
        INTERNVL_U,
        JOYCAPTION,
        MUSICGEN,
        CHATTERBOX,
        STABLE_AUDIO_OPEN,
        FOUNDATION_1,
        TRELLIS_3D,
        HUNYUAN3D,
        FLEX2_PREVIEW,
    )
}

#: The generated class each declaration is bound and annotated as.
GENERATED = {
    "qwen36_35b_a3b": "Qwen3635bA3b",
    "qwen36_27b_mtp": "Qwen3627bMtp",
    "internvl_u": "InternvlU",
    "joycaption": "Joycaption",
    "musicgen": "Musicgen",
    "chatterbox": "Chatterbox",
    "stable_audio_open": "StableAudioOpen",
    "foundation_1": "Foundation1",
    "trellis_3d": "Trellis3d",
    "hunyuan3d": "Hunyuan3d",
    "flex2_preview": "Flex2Preview",
}


def _catalog(family: str) -> type[Model]:
    from gen_worker.model import catalog

    generated: type[Model] = getattr(catalog, GENERATED[family])
    return generated


def test_b5_declares_exactly_eleven_weight_bearing_models() -> None:
    """The batch's own inventory, asserted so a silently-dropped endpoint is red.

    4 LLM/VLM + 4 audio + 2 3D + Flex2Preview. The three NO-MODEL endpoints
    (dj-utils, music-analysis, quality-benchmark) contribute nothing here on
    purpose: they declare no model surface at all and migrate by doing nothing,
    so a declaration for them would be an invention.
    """

    assert len(B5) == 11
    assert all(type(spec) is ModelSpec for spec in B5.values()), (
        "every B5 declaration is EAGER-only; a GraphModelSpec here would owe "
        "runners, a loop and a scheduler that none of these models has"
    )


# ── 1. the gap: an eager model had no way to become a handler parameter ──────


def test_a_runner_less_model_cannot_be_a_family_export() -> None:
    """The red proof that ``eager_model_v1`` is necessary, not decorative.

    ``ModelExport`` refuses zero runners by design — bucket coverage is what
    makes a generated ``Literal`` exhaustive — so widening it to carry B5's
    declarations would have deleted the invariant for documents that have no
    ``Literal`` at all.
    """

    with pytest.raises(ModelError) as caught:
        ModelExport(
            family=FamilyName("joycaption"),
            buckets=(),
            runners=(),
            loop=ExportedLoop(kind=LoopKind.STAGED, stages=()),
            tuned=TunedRef(module="m", qualname="T"),
        )
    assert caught.value.reason is ModelRefusal.SNAPSHOT_INVALID
    assert "at least one runner" in str(caught.value)

    # …while the eager document carries the same model without complaint.
    assert export_eager_model(JOYCAPTION).family == "joycaption"


def test_export_eager_model_refuses_a_graph_declaration() -> None:
    """A graph model exported as an eager one would generate bindings with no
    typed callables — a silent capability loss, so it is a refusal."""


    class _Tuned(TunedValues, frozen=True):
        steps: int = 1

    graph = GraphModelSpec(
        name="b5_graph_probe",
        tuned=_Tuned,
        buckets=(Bucket("shape", (64,)),),
        runners=(
            Runner(
                "denoiser",
                build=lambda layout: None,
                example=lambda bucket, layout: CallExample(params=("x",)),
                axes=("shape",),
            ),
        ),
        loop=Loop(stages=(Stage("denoiser"),)),
    )
    with pytest.raises(ModelError) as caught:
        export_eager_model(graph)
    assert "GraphModelSpec" in str(caught.value)
    assert "export_model()" in str(caught.value)


def test_every_b5_model_is_a_bindable_generated_class() -> None:
    """The whole point: ``@endpoint(models={...})`` takes the CLASS.

    ``Bind`` refuses a declaration outright, so an eager ``ModelSpec`` with no
    generated class could not be bound to a handler parameter at all — which is
    the state B5 found and this suite fences.
    """

    for family in B5:
        generated = _catalog(family)
        assert Bind(generated).model is generated
        assert generated.FAMILY == family
    with pytest.raises(ModelError) as caught:
        Bind(JOYCAPTION)  # type: ignore[arg-type]
    assert caught.value.reason is ModelRefusal.FAMILY_INVALID


def test_an_eager_class_refuses_graph_questions_by_naming_the_tier() -> None:
    """No runner variant, no decode session — and the refusal says WHY.

    A bare ``AttributeError`` would read as a broken binding. Under the F3
    ruling this is a permanent citizen, so the sentence has to say so.
    """

    instance = _catalog("qwen36_35b_a3b").fake()
    for call in (lambda: instance.variant("denoiser"), lambda: instance.session().__enter__()):
        with pytest.raises(ModelError) as caught:
            call()
        message = str(caught.value)
        assert "EAGER" in message and "pgw#1346 F3" in message

    # …and the graph tier is untouched: a real export still answers.
    from gen_worker.model.catalog import Sdxl

    assert Sdxl.EXPORT is not None
    assert Sdxl.EXPORT.runner("denoiser") is not None


# ── 2. the layout axes, migrated BY VALUE ────────────────────────────────────


#: ie#740's floors as the endpoints declare them TODAY, in GB. Asserted as
#: parsed numbers a fit check can compare, never as the strings.
FLOORS = {
    "joycaption": 24.0,
    "musicgen": 12.0,
    "stable_audio_open": 8.0,
    "foundation_1": 8.0,
    "flex2_preview": 36.0,
}

#: The B5 declarations that carry NO floor. Listed rather than defaulted: a
#: floor invented for one of these would be indistinguishable from a migrated
#: one, and inventing floors is exactly what ie#740 forbade.
NO_FLOOR = {
    "qwen36_35b_a3b",
    "qwen36_27b_mtp",
    "internvl_u",
    "chatterbox",
    "trellis_3d",
    "hunyuan3d",
}


def test_the_ie740_floors_migrate_as_parsed_numbers() -> None:
    for family, gigabytes in FLOORS.items():
        spec = B5[family]
        requirement = spec.layout_requirements["plain.bf16@1"]
        assert requirement.min_terms().min_vram_gb == gigabytes, family
        assert requirement.min_terms().min_sm == 0, (
            f"{family} declares an SM floor its endpoint never did"
        )
    for family in NO_FLOOR:
        assert not B5[family].layout_requirements, family
    assert set(FLOORS) | NO_FLOOR == set(B5)


def test_a_requirement_can_only_guard_a_contract_the_model_accepts() -> None:
    """A requirement over nothing is never checked, so it is refused — at the
    declaration AND at the document, because both are entry points."""

    with pytest.raises(LayoutDeclarationError):
        ModelSpec(
            name="b5_floor_probe",
            layouts={"*": ("plain.bf16@1",)},
            layout_requirements={"cozy.fp8-rowwise@1": "vram24g"},
        )
    with pytest.raises(ModelError) as caught:
        EagerExport(
            family="b5_floor_probe",  # type: ignore[arg-type]
            layouts=(("*", ("plain.bf16@1",)),),
            layout_requirements=(("cozy.fp8-rowwise@1", "vram24g"),),
        )
    assert "never checked" in str(caught.value)


#: The four A19 reasons, verbatim, keyed by family. K2's five dirs minus anima,
#: which is B3's — verified here so the split is a fact rather than a memory.
UNDECLARABLE = {
    "qwen36_35b_a3b": "vLLM compressed-tensors fp8 has no registered quant descriptor",
    "qwen36_27b_mtp": "`gguf.native@1` is a TOPOLOGY handle and this axis is QUANT",
    "trellis_3d": "overlay-indirected source-built tree",
    "hunyuan3d": "pickle .ckpt",
}


def test_the_four_undeclarable_reasons_survive_the_migration() -> None:
    """``DEFAULT_LAYOUT = "bf16"`` would label GGUF and pickle bytes bf16.

    That is K2's finding, and it is why the field had to move onto the
    declaration rather than be dropped with ``Slot``.
    """

    for family, fragment in UNDECLARABLE.items():
        spec = B5[family]
        assert fragment in spec.layouts_undeclarable, family
        assert spec.layouts is None, (
            f"{family} declares layouts AND a reason none is nameable"
        )
    for family, spec in B5.items():
        if family in UNDECLARABLE:
            continue
        assert spec.layouts == {"*": ("plain.bf16@1",)}, family
        assert not spec.layouts_undeclarable, family


def test_layouts_and_undeclarable_stay_mutually_exclusive_in_the_document() -> None:
    with pytest.raises(ModelError) as caught:
        EagerExport(
            family="b5_probe",  # type: ignore[arg-type]
            layouts=(("*", ("plain.bf16@1",)),),
            layouts_undeclarable="both, somehow",
        )
    assert "never both" in str(caught.value)


def test_a_model_layout_handle_is_not_a_runner_layout_contract() -> None:
    """K4, as a red proof rather than a comment.

    ``Runner.layouts`` speaks torchcg's identifier grammar; a MODEL's layout
    demand speaks the hub's registered handles, and ``plain.bf16@1`` is legal in
    exactly one of them. Decoding the eager document with the runner parser
    rejected every real B5 declaration — measured, then fixed.
    """

    assert validate_layout_handle("plain.bf16@1", where="probe") == "plain.bf16@1"
    with pytest.raises(RecipeError):
        parse_layout_contract("plain.bf16@1")


# ── 3. the documents and the bindings agree, byte for byte ───────────────────


def test_every_committed_eager_document_round_trips_and_pins_its_binding() -> None:
    for family, spec in B5.items():
        generated = _catalog(family)
        document = generated.EAGER
        assert document is not None
        assert generated.EXPORT is None, (
            f"{family} carries a graph export; it declares no graph classes"
        )
        assert str(document.family) == family
        assert EagerExport.loads(document.dumps()) == document
        assert generated.EXPORT_DIGEST == document.digest()
        # …and the document is what THIS declaration implies, so a declaration
        # edited without re-exporting is red here and not only in the fence.
        assert export_eager_model(spec) == document


def test_the_rendered_binding_is_a_pure_function_of_the_document() -> None:
    document = export_eager_model(MUSICGEN)
    once = render_eager_module(
        document,
        spec_module="gen_worker.model.catalog.boundary_audio",
        spec_attr="MUSICGEN",
    )
    twice = render_eager_module(
        EagerExport.loads(document.dumps()),
        spec_module="gen_worker.model.catalog.boundary_audio",
        spec_attr="MUSICGEN",
    )
    assert once == twice
    assert once.endswith("\n") and not once.endswith("\n\n")
    assert f"class {class_name('musicgen')}(Model):" in once
    assert "def denoiser(" not in once, "an eager binding emits no runner callable"


def test_the_catalog_index_carries_one_name_per_eager_model() -> None:
    """An eager model contributes its class and no ``Layout``/bucket alias,
    because it has no traced variants for one to be exhaustive over."""

    from gen_worker.model import catalog

    for family, name in GENERATED.items():
        assert name in catalog.__all__, family
        assert getattr(catalog, name).FAMILY == family
        assert f"{name}Layout" not in catalog.__all__


# ── 4. K8: which names reach the hub, and which owe a tensorhub PR ───────────


#: The B5 declarations that publish a tuned schema, and therefore put a name
#: into the hub's vocabulary (``ModelSpec._register`` -> ``register_family``).
REGISTERS = {
    "qwen36_35b_a3b",
    "qwen36_27b_mtp",
    "internvl_u",
    "chatterbox",
    "foundation_1",
    "hunyuan3d",
    "flex2_preview",
}


def test_only_a_model_with_tuned_values_reaches_the_hub() -> None:
    """K8, settled by W1b-1 and confirmed here for B5's own eleven.

    The four with no inference vocabulary — joycaption, musicgen,
    stable-audio-open, trellis-3d — register NOTHING, which is why they owe no
    tensorhub PR despite being absent from ``KNOWN_FAMILIES``. Their endpoints
    declare no ``@family`` and read no ``ctx.defaults`` today, so the honest
    declaration has no schema to publish.
    """

    for family in REGISTERS:
        assert B5[family].tuned is not None, family
    for family in set(B5) - REGISTERS:
        assert B5[family].tuned is None, family
        assert _catalog(family).Tuned is GenerationDefaults, family
    assert REGISTERS == {name for name, spec in B5.items() if spec.tuned is not None}


#: Each migrated tuned schema, field-for-field against the endpoint's own
#: ``@family(...)`` struct. A field this table lacks is a stamped recipe value
#: that would silently stop reaching the handler.
TUNED_FIELDS: dict[str, dict[str, object]] = {
    "qwen36_35b_a3b": {"max_tokens": 256, "temperature": 0.7, "top_p": 0.95},
    "qwen36_27b_mtp": {"max_tokens": 256, "temperature": 0.6, "top_p": 0.95},
    "internvl_u": {"num_inference_steps": 20},
    "chatterbox": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    "foundation_1": {"num_inference_steps": 200, "negative": "Low quality."},
    "hunyuan3d": {"num_shape_steps": 50, "guidance_scale": 5.0},
    "flex2_preview": {"num_inference_steps": 28, "guidance": 3.5},
}


def test_each_tuned_schema_is_its_endpoints_vocabulary_by_value() -> None:
    for family, expected in TUNED_FIELDS.items():
        schema = B5[family].tuned
        assert schema is not None
        neutral = schema()
        for field, value in expected.items():
            assert getattr(neutral, field) == value, f"{family}.{field}"
        declared = set(expected) | {"schema_version"}
        actual = {row.name for row in msgspec.structs.fields(schema)}
        assert actual == declared, family


def test_a_stamped_recipe_decodes_onto_an_eager_instance() -> None:
    """The real path ``ctx.defaults`` is replaced by, on an eager model.

    ``tuned_from_catalog`` is the same function the graph tier uses; nothing
    about the eager tier needed a second decoder.
    """

    generated = _catalog("hunyuan3d")
    stamped = json.dumps({"num_shape_steps": 30, "guidance_scale": 7.5})
    tuned = tuned_from_catalog(generated, stamped)
    instance = generated.fake().with_tuned(tuned)
    assert isinstance(instance.tuned, Hunyuan3dTuned)
    assert instance.tuned.num_shape_steps == 30
    assert instance.tuned.guidance_scale == 7.5

    # A tuned-less model still resolves: the bare base is a real struct, not a
    # sentinel, so a handler holding one is not a special case.
    bare = _catalog("musicgen").fake()
    assert isinstance(bare.tuned, GenerationDefaults)

    # …and a malformed stamp REFUSES rather than serving neutral values.
    with pytest.raises(ModelError):
        tuned_from_catalog(generated, '{"num_shape_steps": "thirty"}')


def test_an_external_binary_model_has_no_backing_for_adopt_to_fold() -> None:
    """MEASURED, and it is what B5's MIGRATION half still owes (not this one).

    ``Model.adopt`` folds an eager module map and an armed compiled cell into
    one ``DualBacking``, and refuses when both are absent. An external-binary
    model has NEITHER by definition: llama-server owns the weights in another
    process, and vLLM in another still. W1b-2 supplies the eager half from
    ``Runner(component=)``, which an eager model has no runners to carry.

    So the DECLARATION half is complete and this refusal is correct — what is
    still owed is the serving surface a migrated handler reaches through (the
    snapshot PATH for the eight self-loading runtimes, the constructed pipeline
    for the three that hand one over). Recorded as a passing test rather than
    prose so the day it stops refusing, this goes red and someone reads it.
    """

    with pytest.raises(ModelError) as caught:
        _catalog("qwen36_27b_mtp").adopt(ref="hub:cozy/qwen36-27b-mtp@prod")
    assert caught.value.reason is ModelRefusal.BACKING_MISSING
