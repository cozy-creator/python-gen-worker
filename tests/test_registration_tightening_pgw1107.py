"""Exactly one declaration — the REGISTRATION tightening.

Every family declares on ``@endpoint(compile=)``. A "register only if
``compile.classes and compile.family``" gate silently DROPS a declaration in the
export vocabulary that forgets its classes or its family: nothing registers,
nothing mints, and on a pod that is indistinguishable from an endpoint which
never declared AOT at all.

The naive fix — "a ``compile=`` block must carry classes" — is the other wrong
answer, and it is the one this file exists to keep out. SIX inference endpoints
ship a thin, class-less ``compile=`` for the DYNAMO lane and declare no export
contract at all: ``ernie``, ``flux.1-dev``, ``flux.1-schnell``, ``krea-2``,
``minimax-h3`` and ``sd15`` (``flux.1-schnell`` and ``sd15`` declare two
families each). Every one would fail a classes-required invariant at import. A
class-less compile is a DECISION, not an omission.

So the gate asks about INTENT instead: a ``Compile`` that reaches for the
export vocabulary (``classes``/``dims``/``forks``/``inputs``/``args``/
``blockers``/``shape_strategy``/``warm_changes_key``) is declaring an export
contract and is held to carrying classes and a family; a ``Compile`` that
carries none of it is a dynamo-lane block and is registered nowhere.
"""

from __future__ import annotations

from typing import Any, Tuple

import msgspec
import pytest

from gen_worker import (
    Arg, Compile, Dim, DynamicDim, Fork, GraphClass, Input, MintBlocker,
    RequestContext, endpoint,
)
from gen_worker.api.export_contract import (
    DeclarationError, EXPORT_CONTRACT_FIELDS, declares_export_contract,
    export_declaration, has_export_declaration, registered_export_families,
    reset_export_declarations,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    reset_export_declarations()
    yield
    reset_export_declarations()


class _In(msgspec.Struct):
    text: str = ""


class _Out(msgspec.Struct):
    response: str


def _decorate(compile_decl: Compile) -> type:
    """Decorate a real endpoint class — the production registration path."""

    @endpoint(compile=compile_decl)
    class _Ep:
        def go(self, ctx: RequestContext, data: _In) -> _Out:
            return _Out(response=data.text)

    return _Ep


# ---------------------------------------------------------------------------
# 1. THE SIX CLASS-LESS DYNAMO ENDPOINTS — verified against
#    inference-endpoints master (6c5b1330), and the reason the gate is not a
#    one-line flip. A tightening that red-lines a working endpoint is a
#    regression wearing an invariant's clothes.
# ---------------------------------------------------------------------------


#: The declaration SHAPES the six ship, transcribed from their `main.py`. Not
#: the endpoints themselves (this repo cannot import them) — the shapes, which
#: is what the gate actually sees.
CLASS_LESS_DYNAMO: Tuple[Tuple[str, Compile], ...] = (
    ("ernie", Compile(
        family="ernie", shapes=((1024, 1024), (1152, 896)), text_len=512)),
    ("flux.1-dev", Compile(
        family="flux1-dev", shapes=((1024, 1024),), text_len=512)),
    ("flux.1-schnell", Compile(
        family="flux1-schnell", shapes=((1024, 1024),), text_len=256)),
    ("flux.1-schnell (2nd family)", Compile(
        family="flux1-schnell-redux", shapes=((1024, 1024),), text_len=256)),
    ("krea-2", Compile(
        family="krea-2", shapes=((1024, 1024),), targets=("transformer",),
        text_len=512)),
    # The prior lane's list said FIVE. minimax-h3 is the
    # sixth, and the one with the most export-adjacent-LOOKING declaration —
    # `regional=True` plus a declared dynamic sequence range, and still no
    # export contract. An intent test that mistook either for export intent
    # would take it down.
    ("minimax-h3", Compile(
        family="minimax-h3", shapes=((1024, 1024),), targets=("transformer",),
        regional=True,
        dynamic=(DynamicDim("sequence", min=2, max=8192),))),
    ("sd15", Compile(
        family="sd15", shapes=((512, 512),), targets=("unet",), text_len=77)),
    ("sd15 (2nd family)", Compile(
        family="sd15-inpaint", shapes=((512, 512),), targets=("unet",),
        text_len=77)),
)


@pytest.mark.parametrize(
    "name,decl", CLASS_LESS_DYNAMO, ids=[n for n, _ in CLASS_LESS_DYNAMO])
def test_a_class_less_dynamo_endpoint_decorates_and_registers_nothing(
    name: str, decl: Compile,
) -> None:
    """The load-bearing half of the tightening. These six are correct as
    written: they compile under dynamo and declare no AOT export contract, so
    the classes-required invariant must not reach them."""
    assert declares_export_contract(decl) is False, (
        f"{name} was read as declaring an export contract; the intent test is "
        f"too broad and would red-line a working endpoint")

    _decorate(decl)  # must not raise

    assert registered_export_families() == (), (
        f"{name} registered an export declaration it never made — the mint "
        f"would then ask it for graph classes it does not have")
    assert has_export_declaration(decl.family) is False


def test_the_six_are_the_whole_dynamo_only_shape_and_nothing_else_is() -> None:
    """The complement: every field in the export vocabulary flips intent on.

    Written as a sweep over :data:`EXPORT_CONTRACT_FIELDS` so a field ADDED to
    the vocabulary without being added to the gate fails here rather than
    silently reopening the drop-on-the-floor hole.
    """
    base = dict(family="sweep", shapes=((64, 64),), targets=("transformer",),
                text_len=0)
    values = {
        "classes": (GraphClass(dims={"B": 1}),),
        "dims": (Dim("B", carried_by=(("x", 0),)),),
        "forks": (Fork("cfg", served=(False,), unserved=(True,),
                       reason="default_value"),),
        "inputs": (Input("x", shape=("B", 4), dtype="model"),),
        "args": (Arg("n", value=1),),
        "blockers": (MintBlocker(
            id="B1", what="w", evidence="e", resolves_when="r"),),
        "shape_strategy": "static-rows",
        "warm_changes_key": False,
    }
    assert set(values) == set(EXPORT_CONTRACT_FIELDS), (
        "the export-contract vocabulary changed; add the new field to this "
        "sweep AND to EXPORT_CONTRACT_FIELDS, or the gate silently stops "
        "seeing declarations that use it")

    assert declares_export_contract(Compile(**base)) is False
    # `classes` and `dims` require each other at construction (rows are
    # coordinates over named dims), so they are the one pair tested together;
    # every other field stands alone.
    pairs = {"classes": "dims", "dims": "classes"}
    for field, value in values.items():
        partner = pairs.get(field)
        extra = {partner: values[partner]} if partner else {}
        decl = Compile(**base, **{field: value}, **extra)  # type: ignore[arg-type]
        assert declares_export_contract(decl) is True, field


# ---------------------------------------------------------------------------
# 2. THE HOLE THE TIGHTENING CLOSES — a declaration that MEANT to export and
#    is malformed. RED on master: the gate drops it, so `_decorate` returns
#    happily and `registered_export_families()` is empty either way.
# ---------------------------------------------------------------------------


def test_an_export_declaration_that_forgets_its_classes_is_REFUSED() -> None:
    """dims, inputs, a shape strategy — every word of an export contract
    except the coordinate rows. On master this registered NOTHING and said
    NOTHING; the family then read on a pod as "never declared AOT", which is
    the same sentence a deliberately dynamo-only endpoint produces.

    This is `_Thunk.build()`'s classes check, on the path that now reaches it.
    """
    with pytest.raises(DeclarationError) as excinfo:
        _decorate(Compile(
            family="malformed-noclasses", shapes=((1024, 1024),),
            targets=("transformer",), text_len=77,
            dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
            inputs=(Input("hidden_states", shape=("B", 4, 128, 128),
                          dtype="bfloat16"),),
            shape_strategy="static-rows", warm_changes_key=False,
            classes=()))

    assert "no graph classes" in str(excinfo.value)
    assert "malformed-noclasses" in str(excinfo.value)
    assert registered_export_families() == ()


def test_an_export_declaration_that_forgets_its_FAMILY_is_REFUSED() -> None:
    """The other half of the old gate's `and`. A family-less export
    declaration has no name to register under and no name to mint for — and
    on master it, too, was dropped without a word."""
    with pytest.raises(DeclarationError) as excinfo:
        _decorate(Compile(
            family="", shapes=((1024, 1024),), targets=("transformer",),
            text_len=77,
            dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
            classes=(GraphClass(dims={"B": 1}),),
            inputs=(Input("hidden_states", shape=("B", 4, 128, 128),
                          dtype="bfloat16"),),
            shape_strategy="static-rows", warm_changes_key=False))

    assert "no Compile.family" in str(excinfo.value)
    assert registered_export_families() == ()


def test_a_BLOCKED_declaration_still_has_to_carry_its_classes() -> None:
    """pgw#1115's blockers say "not yet", never "and therefore incomplete".
    ltx-video-2.3 folded 82 graph classes AND three open blockers; a blocked
    family that also dropped its class table would look identical to a
    correctly blocked one."""
    with pytest.raises(DeclarationError):
        _decorate(Compile(
            family="blocked-noclasses", shapes=((1024, 1024),),
            targets=("transformer",), text_len=77,
            blockers=(MintBlocker(id="B1", what="w", evidence="e",
                                  resolves_when="r"),)))


# ---------------------------------------------------------------------------
# 3. THE WELL-FORMED DECLARATION STILL REGISTERS — the gate re-aimed, not
#    removed. Includes the blocked-but-complete shape (ltx's).
# ---------------------------------------------------------------------------


def _well_formed(family: str, **over: Any) -> Compile:
    return Compile(
        family=family, shapes=((1024, 1024),), targets=("transformer",),
        text_len=77,
        dims=(Dim("B", carried_by=(("hidden_states", 0),)),),
        classes=(GraphClass(dims={"B": 1}),),
        inputs=(Input("hidden_states", shape=("B", 4, 128, 128),
                      dtype="bfloat16"),),
        shape_strategy="static-rows", warm_changes_key=False, **over)


def test_a_well_formed_export_declaration_registers_at_decoration() -> None:
    _decorate(_well_formed("well-formed"))

    assert registered_export_families() == ("well-formed",)
    decl = export_declaration("well-formed")
    assert decl is not None and decl.classes


def test_a_complete_but_BLOCKED_declaration_registers_and_carries_its_ids() -> None:
    _decorate(_well_formed("blocked-complete", blockers=(MintBlocker(
        id="OQ-2", what="w", evidence="e", resolves_when="r"),)))

    decl = export_declaration("blocked-complete")
    assert decl is not None
    assert [b.id for b in decl.open_blockers] == ["OQ-2"]


def test_collection_applies_the_same_intent_test_as_the_decorator() -> None:
    """`registry.register_declared_exports` is the SECOND gate — it re-reads
    every collected spec's `compile=`. Two gates asking different questions is
    how the dual-declaration hazard got in; they ask one question now."""
    from gen_worker.registry import register_declared_exports

    class _Spec:
        def __init__(self, compile_decl: Any) -> None:
            self.compile = compile_decl

    dynamo = Compile(family="dyn-only", shapes=((64, 64),), text_len=0)
    assert register_declared_exports([_Spec(dynamo)]) == ()  # type: ignore[arg-type]
    assert registered_export_families() == ()

    assert register_declared_exports(
        [_Spec(_well_formed("collected"))]) == ("collected",)  # type: ignore[arg-type]
