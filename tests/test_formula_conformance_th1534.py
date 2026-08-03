"""Cross-language formula limits and canonicalization contract (th#1534).

The fixture is vendored byte-identically in Tensorhub under
``internal/formula/testdata`` because each repository must validate in
isolation. The JSON, rather than two hand-authored case lists, owns the cases.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gen_worker.api.formula import (
    FormulaLimits,
    FormulaParseError,
    FormulaRefusal,
    RuntimeFormula,
)

_DOCUMENT = json.loads(
    (Path(__file__).parent / "testdata" / "formula_vectors.json").read_text()
)
_BASE_LIMITS = FormulaLimits(**_DOCUMENT["limits"])


def _limits(overrides: dict[str, Any]) -> FormulaLimits:
    return replace(_BASE_LIMITS, **overrides)


def _values(values: dict[str, Any]) -> dict[str, float]:
    sentinels = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}
    return {
        name: sentinels.get(value, value) if isinstance(value, str) else value
        for name, value in values.items()
    }


@pytest.mark.parametrize(
    "vector", _DOCUMENT["vectors"], ids=[v["name"] for v in _DOCUMENT["vectors"]]
)
def test_formula_conformance_vector(vector: dict[str, Any]) -> None:
    limits = _limits(vector.get("limits", {}))
    source = vector.get("source", "")
    if repeat := vector.get("repeat_parentheses", 0):
        source = f"a+b*{'(' * repeat}x{')' * repeat}"
    if vector.get("parse_error"):
        with pytest.raises(FormulaParseError):
            RuntimeFormula(source, limits=limits)
        return

    formula = RuntimeFormula(source, limits=limits)
    if "term_keys" in vector:
        assert [term.key for term in formula.terms] == vector["term_keys"]
    if "values" not in vector:
        return

    if vector.get("evaluation_error") and vector.get("refusal"):
        with pytest.raises(FormulaRefusal) as raised:
            formula.term_values(_values(vector["values"]))
        if "error_contains" in vector:
            assert vector["error_contains"] in str(raised.value)
        return

    got = formula.term_values(_values(vector["values"]))
    if vector.get("evaluation_error"):
        assert got is None
    else:
        assert got == vector["term_values"]
