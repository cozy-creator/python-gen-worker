"""Cross-language formula limits and canonicalization contract.

The fixture is vendored byte-identically in Tensorhub under
``internal/formula/testdata`` because each repository must validate in
isolation. The JSON, rather than two hand-authored case lists, owns the cases.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
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

_DEFAULT_CORPUS = Path(__file__).parent / "testdata" / "formula_vectors.json"
_CORPUS = Path(os.environ.get("FORMULA_VECTOR_CORPUS", _DEFAULT_CORPUS))
_DOCUMENT = json.loads(_CORPUS.read_text())
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


def test_formula_corpus_digest_gate_can_go_red(tmp_path: Path) -> None:
    corpus = tmp_path / "formula_vectors.json"
    digest = tmp_path / "FORMULA_VECTORS_DIGEST"
    corpus.write_bytes((Path(__file__).parent / "testdata" / "formula_vectors.json").read_bytes())
    digest.write_bytes((Path(__file__).parent / "testdata" / "FORMULA_VECTORS_DIGEST").read_bytes())
    corpus.write_bytes(corpus.read_bytes() + b"\n")

    got = subprocess.run(
        [
            os.fspath(Path(__file__).parents[1] / "scripts" / "check_formula_corpus_digest.py"),
            "--corpus",
            os.fspath(corpus),
            "--digest",
            os.fspath(digest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "changed without its digest" in got.stdout


def test_formula_semantic_fence_can_go_red(tmp_path: Path) -> None:
    document = json.loads(_DEFAULT_CORPUS.read_text())
    document["vectors"][0]["term_values"]["x"] = 4
    corpus = tmp_path / "formula_vectors.json"
    corpus.write_text(json.dumps(document))

    got = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-q",
            os.fspath(Path(__file__)),
            "-k",
            "test_formula_conformance_vector",
        ],
        env={**os.environ, "FORMULA_VECTOR_CORPUS": os.fspath(corpus)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "ordinary evaluation" in got.stdout
    assert "4" in got.stdout


def test_formula_peer_drift_gate_can_go_red(tmp_path: Path) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    testdata = Path(__file__).parent / "testdata"
    for name in ("formula_vectors.json", "FORMULA_VECTORS_DIGEST"):
        (peer / name).write_bytes((testdata / name).read_bytes())
    (peer / "formula_vectors.json").write_bytes(
        (peer / "formula_vectors.json").read_bytes() + b"\n"
    )

    got = subprocess.run(
        ["bash", os.fspath(Path(__file__).parents[1] / "scripts" / "formula-vector-drift.sh")],
        env={"PATH": "/usr/bin:/bin", "FORMULA_VECTOR_PEER_DIR": os.fspath(peer)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert got.returncode == 1
    assert "formula_vectors.json differs" in got.stderr
