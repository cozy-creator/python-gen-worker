"""pgw#1346 W1b — the declaration axes the `Slot` deletion needs (K1–K5).

Every axis here is exercised through the REAL objects the fleet uses: the real
`ModelSpec`/`GraphModelSpec` constructors with `Slot`'s own layout normalizers,
the real `@endpoint` decorator, and the real discovery walk that writes
`endpoint.lock`. Nothing is stubbed, because the property under test is that a
production floor MEANS THE SAME THING after it moves.

The floors asserted below are the live ie#740 values read out of
`serverless-endpoints` on 2026-08-17 — krea-2's `vram72g`, flux.1-schnell's
`vram36g`, the `sm89+` guard on the fp8-rowwise lane. They are production
incidents preserved by value, which is exactly why this file compares NUMBERS
and not spellings.
"""

from __future__ import annotations

import pytest

from gen_worker.model.bind import Bind, as_bind
from gen_worker.model.errors import ModelError
from gen_worker.model.spec import GraphModelSpec, ModelSpec
from gen_worker.models.tensor_layout_contract import LayoutDeclarationError

from harness.model_toys_pgw1332 import ToyTuned

BF16 = "plain.bf16@1"
FP8 = "cozy.fp8-rowwise@1"


# --------------------------------------------------------------------------
# K1 — layout_requirements, the ie#740 execution floors, BY VALUE
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("endpoint_name", "compact", "vram", "sm"),
    [
        ("krea-2", "vram72g", 72.0, 0),
        ("flux.1-schnell", "vram36g", 36.0, 0),
        ("minimax-h3", "vram78g", 78.0, 0),
        ("ernie", "vram32g", 32.0, 0),
        ("musicgen", "vram12g", 12.0, 0),
        ("qwen-image", "sm89+", 0.0, 89),
    ],
)
def test_a_production_floor_survives_the_move_onto_the_model_by_value(
    endpoint_name: str, compact: str, vram: float, sm: int
) -> None:
    """A floor that moves off `Slot` must still mean the same NUMBER.

    These six are live declarations in `serverless-endpoints`. The reason this
    asserts the parsed terms rather than the string is that the string is not
    the contract — `min_vram_gb` is what a placement decision reads, and a
    second parser that spelled it differently is how a floor silently stops
    guarding the incident it was written for.
    """

    spec = ModelSpec(
        name=f"floor_{endpoint_name.replace('-', '_').replace('.', '_')}",
        layouts={"*": (BF16, FP8)},
        layout_requirements={BF16 if vram else FP8: compact},
    )
    (requirement,) = spec.layout_requirements.values()
    assert requirement.minimum.min_vram_gb == vram
    assert requirement.minimum.min_sm == sm
    # `recommended` is additive and gates nothing; the compact form IS the
    # minimum, and inventing a recommendation from it would gate on a value
    # nobody declared.
    assert requirement.recommended is None


def test_a_requirement_guarding_a_contract_the_model_does_not_accept_is_refused() -> None:
    with pytest.raises(LayoutDeclarationError) as caught:
        ModelSpec(name="floor_unaccepted", layouts={"*": (BF16,)},
                  layout_requirements={FP8: "sm89+"})
    assert FP8 in str(caught.value)


def test_a_requirement_with_no_layouts_to_guard_is_refused() -> None:
    with pytest.raises(LayoutDeclarationError) as caught:
        ModelSpec(name="floor_no_layouts", layout_requirements={BF16: "vram72g"})
    assert "without layouts=" in str(caught.value)


# --------------------------------------------------------------------------
# K2 — layouts_undeclarable, the explicit third rung
# --------------------------------------------------------------------------


def test_undeclarable_takes_a_reason_and_refuses_a_blank_one() -> None:
    spec = ModelSpec(
        name="gguf_probe",
        layouts_undeclarable="GGUF: the quant axis has no registered handle",
    )
    assert spec.layouts_undeclarable.startswith("GGUF:")
    assert spec.layouts is None
    with pytest.raises(LayoutDeclarationError):
        ModelSpec(name="gguf_blank_probe", layouts_undeclarable="   ")


def test_declaring_both_layouts_and_undeclarable_is_refused() -> None:
    """The tri-state may not be collapsed from both ends at once."""
    with pytest.raises(LayoutDeclarationError) as caught:
        ModelSpec(name="both_probe", layouts={"*": (BF16,)},
                  layouts_undeclarable="also undeclarable")
    assert "mutually exclusive" in str(caught.value)


# --------------------------------------------------------------------------
# K4 — the demand stays keyed by COMPONENT PATH, not by runner
# --------------------------------------------------------------------------


def test_a_per_component_demand_is_expressible_and_the_star_is_the_default() -> None:
    """SDXL declares two runners over a four-component tree.

    Collapsing this onto `Runner.layouts` would make a per-text-encoder demand
    inexpressible, which is why the component-path keying survived the move.
    """

    spec = ModelSpec(
        name="component_probe",
        layouts={"*": (BF16,), "text_encoder": (FP8, BF16)},
    )
    assert spec.layouts is not None
    assert spec.layouts["*"] == (BF16,)
    # Canonical order, not written order: the set is a compatibility FILTER and
    # a position in it must never read as a preference.
    assert set(spec.layouts["text_encoder"]) == {FP8, BF16}
    assert spec.layouts["text_encoder"] == tuple(sorted(spec.layouts["text_encoder"]))


def test_runner_layouts_is_a_different_axis_and_keeps_its_meaning() -> None:
    """`Runner.layouts` says which layouts a GRAPH CLASS has traced variants
    for; `ModelSpec.layouts` says which layouts the model's code can execute.
    A model may accept fp8 bytes for a component it has no fp8 graph class for.
    """

    from harness.model_toys_pgw1332 import TOY_DIFFUSION

    assert TOY_DIFFUSION.runners[0].layouts == ("bf16",)
    assert TOY_DIFFUSION.layouts is None


# --------------------------------------------------------------------------
# K5 — the eager tier is permanent, and an auxiliary model has no vocabulary
# --------------------------------------------------------------------------


def test_an_auxiliary_model_declares_no_tuned_schema_and_registers_nothing() -> None:
    """A RIFE interpolator / latent upsampler / tokenizer tree has no
    inference vocabulary. Publishing an empty schema under its name would put a
    name into the hub's vocabulary that answers no question (K8).
    """

    from gen_worker.families.base import family_registry

    before = dict(family_registry())
    spec = ModelSpec(name="rife_interpolator_probe")
    assert spec.tuned is None
    assert dict(family_registry()) == before, (
        "an auxiliary model published a tuned schema; K8 says only a model with "
        "tuned values reaches the hub's vocabulary"
    )


def test_a_lora_vocabulary_with_no_base_vocabulary_to_refine_is_refused() -> None:
    with pytest.raises(ModelError) as caught:
        ModelSpec(name="lora_orphan_probe", lora_tuned=ToyTuned)
    assert "without tuned=" in str(caught.value)


def test_a_graph_model_still_owes_a_tuned_schema() -> None:
    """`tuned` is optional only on the EAGER tier. A declared graph serves
    generation requests and its parameters are what a tuned schema names.
    """

    with pytest.raises(ModelError) as caught:
        GraphModelSpec(name="graph_no_tuned_probe")
    assert "tuned" in str(caught.value)


# --------------------------------------------------------------------------
# K3 — Bind, the endpoint-coupled axes
# --------------------------------------------------------------------------


def test_a_bare_model_class_and_a_bare_bind_mean_the_same_thing() -> None:
    from gen_worker.model.catalog import Sdxl

    assert as_bind(Sdxl) == Bind(model=Sdxl)
    assert as_bind(Bind(Sdxl, selected_by="model")).selected_by == "model"


def test_bind_refuses_anything_that_is_not_a_generated_model_class() -> None:
    with pytest.raises(ModelError) as caught:
        Bind(object)  # type: ignore[type-var]
    assert "generated model class" in str(caught.value)


def test_bind_refuses_a_declaration_where_a_generated_class_belongs() -> None:
    """The declaration is what you EXPORT; what an endpoint binds is the class
    generated from it. A declaration has no typed callables on it, so binding
    one would hand a handler something that looks resolved and is not.
    """

    from harness.model_toys_pgw1332 import TOY_DIFFUSION

    with pytest.raises(ModelError):
        Bind(TOY_DIFFUSION)  # type: ignore[arg-type]


def test_binding_two_roots_is_a_decoration_time_error() -> None:
    import msgspec

    from gen_worker import RequestContext, endpoint
    from gen_worker.model.catalog import Sdxl

    class In(msgspec.Struct, frozen=True):
        pass

    class Out(msgspec.Struct, frozen=True):
        pass

    class TwoRoots:
        def run(
            self, ctx: RequestContext, p: In, left: Sdxl, right: Sdxl
        ) -> Out:
            return Out()

    with pytest.raises(ValueError) as caught:
        endpoint(families={
            "left": Bind(Sdxl, root=True),
            "right": Bind(Sdxl, root=True),
        })(TwoRoots)
    assert "root" in str(caught.value)


def test_binding_several_models_and_marking_no_root_is_fine() -> None:
    """The model surface is genuinely smaller than the `Slot` one it replaces:
    a handler names every model it binds, by parameter, so there is no
    ambiguity for a root to settle.
    """

    import msgspec

    from gen_worker import RequestContext, endpoint
    from gen_worker.model.catalog import Sdxl

    class In(msgspec.Struct, frozen=True):
        pass

    class Out(msgspec.Struct, frozen=True):
        pass

    class NoRoot:
        def run(
            self, ctx: RequestContext, p: In, left: Sdxl, right: Sdxl
        ) -> Out:
            return Out()

    decorated = endpoint(families={"left": Sdxl, "right": Sdxl})(NoRoot)
    declared = getattr(decorated, "__gen_worker_endpoint__").families
    assert sorted(declared) == ["left", "right"]
    assert not any(row.root for row in declared.values())
