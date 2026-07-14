"""th#597 C5: parse_model_ref (provider=tensorhub) must pass the shared
grammar vectors (tests/testdata/ref_grammar_vectors.json), vendored
byte-identically in tensorhub (internal/orchestrator/release/testdata/) for
Go's ParseCanonicalRef. Any grammar change edits the fixture in BOTH repos."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker.models.refs import parse_model_ref

_VECTORS = json.loads(
    (Path(__file__).parent / "testdata" / "ref_grammar_vectors.json").read_text()
)["vectors"]


@pytest.mark.parametrize("vec", _VECTORS, ids=[v["ref"] or "<empty>" for v in _VECTORS])
def test_ref_grammar_vector(vec: dict) -> None:
    if vec.get("error"):
        with pytest.raises(ValueError):
            parse_model_ref(vec["ref"], provider="tensorhub")
        return
    parsed = parse_model_ref(vec["ref"], provider="tensorhub")
    th = parsed.tensorhub
    assert th is not None
    assert th.owner == vec["owner"]
    assert th.repo == vec["repo"]
    assert th.tag == vec["tag"]
    assert (th.digest or "") == vec["digest"]
    assert (th.flavor or "") == vec["flavor"]
