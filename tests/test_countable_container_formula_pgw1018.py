"""pgw#1018: a RuntimeFormula could not express a countable-container term, so
a reference-heavy request was cost-modeled as if its references were free.

Measured by the H3 endpoint lane (ie#612): ref2va runs **208 s vs 80 s** for
the same duration once references are present — ~2.6x — and the term for it
could not be declared at all. The lane left it out rather than faking a
constant, which is the honest failure and is why this is a filed gap instead
of a wrong number in production.

**The spelling the issue proposed (`len(references)`) is refused here, on
purpose.** The formula STRING is the wire contract: `api/formula.py` mirrors
tensorhub `internal/formula` byte-for-byte, calls are excluded from the
grammar on both sides, and the term KEY is canonicalized from the expression —
so a `len(...)` term would mint a key the hub can never reproduce. The
platform already HAS this vocabulary and it is not a call: the hub's
`internal/price` (th#833 / ie#600) binds an identifier naming an ARRAY or MAP
to that container's ITEM COUNT. This closes the gap by teaching the worker the
same reading, so pricing and runtime prediction share one string.
"""

from __future__ import annotations

import textwrap
from typing import Dict, List, Optional

import msgspec
import pytest

from gen_worker.api.formula import FormulaParseError, RuntimeFormula


class RefPayload(msgspec.Struct):
    prompt: str = ""
    num_inference_steps: int = 28
    references: List[str] = msgspec.field(default_factory=list)
    knobs: Dict[str, float] = msgspec.field(default_factory=dict)
    maybe_refs: Optional[List[str]] = None
    caption: str = ""


FORMULA = "a + b*num_inference_steps + c*references"


def test_a_container_field_is_a_legal_term() -> None:
    """The refusal that blocked H3: `references` has no numeric wire default
    and never can have one, so declaration died before the endpoint could
    price it."""
    rf = RuntimeFormula(FORMULA)
    rf.validate_for_payload(RefPayload, "ref2va")
    assert rf.fields == ("num_inference_steps", "references")
    # The key is the bare identifier — exactly what the hub canonicalizes for
    # the same source. A term key that only one side can mint is not a term.
    assert [t.key for t in rf.terms] == ["1", "num_inference_steps", "references"]


def test_the_term_is_the_item_count() -> None:
    rf = RuntimeFormula(FORMULA)
    for n in (0, 1, 5):
        got = rf.term_values_from_struct(
            RefPayload(num_inference_steps=20, references=["r"] * n))
        assert got == {"1": 1.0, "num_inference_steps": 20.0, "references": float(n)}


def test_a_map_counts_its_entries() -> None:
    """`maxProperties` is the hub's twin of `maxItems`; a name->value map is a
    countable container on both sides."""
    rf = RuntimeFormula("a + b*knobs")
    rf.validate_for_payload(RefPayload, "ref2va")
    assert rf.term_values_from_struct(
        RefPayload(knobs={"x": 1.0, "y": 2.0})) == {"1": 1.0, "knobs": 2.0}


def test_a_declared_container_that_arrives_null_is_zero_items() -> None:
    """`Optional[list] = None` is the common no-extra-references call. Zero is
    a READING of a declared container, not a guess: the field says what it is.
    A missing NUMBER is a different thing and still declines below."""
    rf = RuntimeFormula("a + b*maybe_refs")
    rf.validate_for_payload(RefPayload, "ref2va")
    assert rf.term_values_from_struct(RefPayload()) == {"1": 1.0, "maybe_refs": 0.0}
    assert rf.term_values_from_struct(
        RefPayload(maybe_refs=["a", "b", "c"])) == {"1": 1.0, "maybe_refs": 3.0}


def test_a_missing_number_still_declines() -> None:
    """The posture that must NOT widen: an unevaluable numeric term returns
    None so the hub falls back to its own evaluation, instead of folding a
    fabricated 0 into the learned constants."""
    class Loose(msgspec.Struct):
        num_inference_steps: Optional[int] = None

    rf = RuntimeFormula("a + b*num_inference_steps")
    assert rf.term_values_from_struct(Loose()) is None


def test_a_string_is_not_a_countable_container() -> None:
    """`str`/`bytes` are Sized and are deliberately NOT countable — the hub's
    `price.numericValue` counts an array or an object and nothing else.
    Admitting a prompt's length here would silently price one."""
    with pytest.raises(ValueError, match="numeric/bool wire default"):
        RuntimeFormula("a + b*caption").validate_for_payload(RefPayload, "ref2va")


def test_len_is_not_grammar() -> None:
    """The adjudication, pinned: no calls, on either side of the wire."""
    for src in ("a + b*len(references)", "a + count(references)",
                "a + b*count:references"):
        with pytest.raises(FormulaParseError):
            RuntimeFormula(src)


def test_endpoint_declares_a_reference_term_end_to_end(tmp_path, monkeypatch) -> None:
    """The whole path H3 hit: a real `@endpoint` with a real payload, through
    real discovery. This is the case that raised at IMPORT."""
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "ep_pgw1018"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(textwrap.dedent("""
        from typing import List

        import msgspec
        from gen_worker import RequestContext, RuntimeFormula, endpoint

        class In_(msgspec.Struct):
            prompt: str = ""
            num_inference_steps: int = 28
            references: List[str] = msgspec.field(default_factory=list)

        class Out_(msgspec.Struct):
            y: str

        @endpoint(runtime=RuntimeFormula(
            "a + b*num_inference_steps + c*references"
        ))
        def generate(ctx: RequestContext, data: In_) -> Out_:
            return Out_(y="ok")
    """))

    from gen_worker.discovery.discover import discover_functions

    (fn,) = discover_functions(tmp_path, main_module="ep_pgw1018.main")
    assert fn["runtime_formula"] == "a + b*num_inference_steps + c*references"
