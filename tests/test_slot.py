"""``Slot`` (pgw#520): decoration validation, back-compat with bare
``ModelRef`` bindings, and the ``models={}``/``model=`` normalization split
between the plain binding map (executor/CLI/prefetch) and the Slot metadata
map (discovery emission, resolution chain)."""

from __future__ import annotations

import msgspec
import pytest

from gen_worker import HF, Hub, RequestContext, Resources, Slot, endpoint
from gen_worker.families import SdxlDefaults
from gen_worker.registry import extract_specs


class _In(msgspec.Struct):
    prompt: str = ""
    model: str = ""


class _BadIn(msgspec.Struct):
    prompt: str = ""
    model: int = 0


class _Out(msgspec.Struct):
    y: str


def test_slot_requires_a_type_for_pipeline_cls() -> None:
    with pytest.raises(TypeError, match="class"):
        Slot("not-a-class")  # type: ignore[arg-type]


def test_slot_default_checkpoint_must_be_a_model_ref() -> None:
    with pytest.raises(TypeError, match="ModelRef"):
        Slot(object, default_checkpoint="o/r")  # type: ignore[arg-type]


def test_slot_default_config_must_be_family_defaults() -> None:
    with pytest.raises(TypeError, match="FamilyDefaults"):
        Slot(object, default_config=object())  # type: ignore[arg-type]


def test_slot_family_from_default_config_registration() -> None:
    s = Slot(object, default_checkpoint=HF("o/r"), default_config=SdxlDefaults(steps=30))
    assert s.family == "sdxl"


def test_slot_family_empty_with_no_default_config() -> None:
    s = Slot(object, default_checkpoint=HF("o/r"))
    assert s.family == ""


def test_bare_modelref_and_slot_coexist_in_models_dict() -> None:
    """A bare ModelRef is sugar for Slot(<inferred class>,
    default_checkpoint=ref) — both forms populate the plain binding map
    every existing model-injection call site understands; only the Slot key
    also lands in `.slots`."""
    @endpoint(models={
        "pipeline": Slot(object, selected_by="model", default_checkpoint=HF("o/pipeline"),
                          default_config=SdxlDefaults(steps=28)),
        "vae": Hub("o/vae"),
    })
    class Gen:
        def setup(self, pipeline: object, vae: object) -> None: ...

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(y="ok")

    (s,) = extract_specs(Gen)
    assert set(s.models) == {"pipeline", "vae"}
    assert list(s.slots) == ["pipeline"]
    assert s.models["pipeline"].path == "o/pipeline"
    assert s.models["vae"].path == "o/vae"
    assert s.slot_family == {"pipeline": "sdxl"}


def test_slot_with_no_selected_by_and_no_default_omits_the_binding_entry() -> None:
    """A hub-only Slot with no ``selected_by`` (never branches per request)
    and no code-side default contributes no static binding — there's
    nothing for the executor's build-time placement/download to pin
    locally; it only resolves via a live hub mapping."""
    @endpoint(model=Slot(object))
    class Gen:
        def setup(self, model: object) -> None: ...

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(y="ok")

    (s,) = extract_specs(Gen)
    assert list(s.models) == []
    assert list(s.slots) == ["model"]
    assert s.slots["model"].default_checkpoint is None


def test_slot_selected_by_without_default_checkpoint_errors_at_registration() -> None:
    """pgw#524 item 4: a request-branching slot (selected_by set) with no
    default_checkpoint has nothing to seed the hub mapping/bootstrap
    placement with — tensorhub rejects this at manifest registration, so
    the SDK fails at author time instead."""
    with pytest.raises(ValueError, match="requires default_checkpoint"):
        @endpoint(model=Slot(object, selected_by="model"))
        class Gen:
            def setup(self, model: object) -> None: ...

            def generate(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(y="ok")

        extract_specs(Gen)


def test_model_shorthand_accepts_a_slot() -> None:
    @endpoint(model=Slot(object, selected_by="model", default_checkpoint=HF("o/r")))
    class Gen:
        def setup(self, model: object) -> None: ...

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(y="ok")

    (s,) = extract_specs(Gen)
    assert list(s.models) == ["model"]
    assert list(s.slots) == ["model"]
    assert s.models["model"].path == "o/r"


def test_selected_by_must_name_an_existing_payload_field() -> None:
    with pytest.raises(ValueError, match="names no field"):
        @endpoint(model=Slot(object, selected_by="nonexistent", default_checkpoint=HF("o/r")))
        class Gen:
            def setup(self, model: object) -> None: ...

            def generate(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(y="ok")

        extract_specs(Gen)


def test_selected_by_must_be_plain_str_typed() -> None:
    with pytest.raises(ValueError, match="plain str"):
        @endpoint(model=Slot(object, selected_by="model", default_checkpoint=HF("o/r")))
        class Gen:
            def setup(self, model: object) -> None: ...

            def generate(self, ctx: RequestContext, data: _BadIn) -> _Out:
                return _Out(y="ok")

        extract_specs(Gen)


def test_slot_validated_per_handler_not_per_class() -> None:
    """One models= decl is shared by every method on the class; a Slot's
    selected_by is validated against EACH handler's own payload type
    (methods can have divergent payload structs)."""
    @endpoint(models={"pipeline": Slot(object, selected_by="model", default_checkpoint=HF("o/r"))})
    class Gen:
        def setup(self, pipeline: object) -> None: ...

        def generate(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(y="ok")

        def generate_bad(self, ctx: RequestContext, data: _BadIn) -> _Out:
            return _Out(y="ok")

    with pytest.raises(ValueError, match="plain str"):
        extract_specs(Gen)


def test_resources_and_slots_do_not_conflict_in_class_validation() -> None:
    """Slots participate in the setup()-param consumability check the same
    way bare bindings do."""
    with pytest.raises(ValueError, match="pipeline"):
        @endpoint(models={"pipeline": Slot(object, default_checkpoint=HF("o/r"))},
                  resources=Resources(vram_gb=8))
        class Bad:
            def setup(self, other_name: object) -> None: ...

            def generate(self, ctx: RequestContext, data: _In) -> _Out:
                return _Out(y="ok")
