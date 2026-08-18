"""The strict-typing acceptance for pgw#1377 (Paul's end-to-end-typed ruling).

``assert_type`` claims here are checked by the CI mypy-strict pass over
``tests`` — a loosened surface (an ``Any`` leak, a shared int|float return, a
broken generic projection) turns this module red STATICALLY. The runtime
asserts keep the same expressions honest as behavior.

It also pins the typevar plumbing agreed with the SDK-core lane (pgw#1382):
``LoadContext`` generic over the ModelType subclass, with
``def defaults(self: "LoadContext[ModelType[D]]") -> D`` projecting the
Defaults struct out of the generic — ``_LoadContext`` below is that spelling,
so ``ctx.defaults()`` on a ``_LoadContext[SDXL]`` is ``SDXL.Defaults``
statically, no cast caller-side.
"""

from __future__ import annotations

from typing import Generic, Mapping, TypeVar, assert_type, cast

import msgspec

from gen_worker.models import Knob, ModelType, SchedulerName, SDXL
from gen_worker.models.defaults_decode import CarriesDefaults, decode_model_defaults
from gen_worker.models.model_types import SdxlDefaults, SdxlLoraDefaults, SdxlRecipe
from gen_worker.families import GenerationDefaults
from gen_worker.request_context import RequestContext

D = TypeVar("D", bound=msgspec.Struct)
MT_co = TypeVar("MT_co", bound=ModelType[msgspec.Struct], covariant=True)


class _LoadContext(Generic[MT_co]):
    """The pgw#1382 ``LoadContext[SDXL]`` typevar plumbing, in miniature."""

    def __init__(
        self,
        model_type: type[MT_co],
        *,
        model: str | None,
        defaults: Mapping[str, object] | None,
    ) -> None:
        self._model_type = model_type
        self._model = model
        self._row = defaults

    def defaults(self: "_LoadContext[ModelType[D]]") -> D:
        # The ONE internal cast: the generic base cannot declare the
        # ``Defaults`` class attribute (ClassVar can't be generic), so the
        # structural protocol re-asserts it here. Callers stay cast-free.
        model_type = cast("CarriesDefaults[D]", self._model_type)
        return decode_model_defaults(model_type, model=self._model, defaults=self._row)


def test_the_generic_chain_projects_the_defaults_type() -> None:
    ctx = _LoadContext(SDXL, model="sdxl", defaults={"cfg": False})
    d = ctx.defaults()
    assert_type(d, SdxlDefaults)
    assert SDXL.Defaults is SdxlDefaults  # the contract-file spelling
    assert isinstance(d, SDXL.Defaults)
    assert d.cfg is False


def test_knob_resolution_is_typed_per_instantiation() -> None:
    rctx: RequestContext[GenerationDefaults] = RequestContext("typed-1")
    d = decode_model_defaults(SDXL, model="sdxl", defaults=None)
    assert_type(d, SdxlDefaults)

    assert_type(d.steps, Knob[int])
    assert_type(d.guidance, Knob[float])

    steps = d.steps.resolve(None, rctx)
    assert_type(steps, int)  # NOT int | float
    guidance = d.guidance.resolve(14.0, rctx)
    assert_type(guidance, float)

    assert_type(d.cfg, bool)
    assert_type(d.step_distilled, bool)
    assert_type(d.positive_preamble, str)
    assert_type(d.negative_preamble, str)
    assert_type(d.timesteps, tuple[int, ...])

    assert steps == 28 and guidance == 14.0


class _FakeAdapter:
    """The ``turbo: Adapter | None`` shape main_v2.py reads defaults from."""

    def __init__(self, defaults: SdxlLoraDefaults) -> None:
        self.defaults = defaults


def test_the_recipe_is_one_nominal_type() -> None:
    """main_v2.py annotates ``recipe: SDXL.Recipe`` — both Defaults types
    inherit it, so ``turbo.defaults if turbo else d`` needs no union."""
    rctx: RequestContext[GenerationDefaults] = RequestContext("typed-2")
    d = SDXL.Defaults()
    turbo: _FakeAdapter | None = _FakeAdapter(SDXL.Lora.Defaults())

    recipe: SdxlRecipe = turbo.defaults if turbo is not None else d
    assert_type(recipe, SdxlRecipe)
    assert SDXL.Recipe is SdxlRecipe  # the contract-file spelling

    assert_type(recipe.cfg, bool)
    assert_type(recipe.steps, Knob[int])
    assert_type(recipe.guidance, Knob[float])
    assert_type(recipe.timesteps, tuple[int, ...])
    steps = recipe.steps.resolve(None, rctx)
    assert_type(steps, int)
    guidance = recipe.guidance.resolve(None, rctx)
    assert_type(guidance, float)

    assert recipe.cfg is False
    assert (steps, guidance) == (4, 6.0)
    assert turbo is not None
    assert isinstance(d, SdxlRecipe) and isinstance(turbo.defaults, SdxlRecipe)

    lora = SDXL.Lora.Defaults()
    assert_type(lora, SdxlLoraDefaults)
    assert_type(lora.strength, Knob[float])
    assert_type(lora.trigger_words, tuple[str, ...])
    assert_type(lora.distillation, bool)
    # The scheduler demand lives on the ADAPTER overlay only.
    scheduler: SchedulerName | None = lora.scheduler
    assert scheduler == "euler_trailing"
