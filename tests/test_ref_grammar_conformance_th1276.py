"""th#597 C5 conformance: the shared ref-grammar vectors are THE contract.

``tests/testdata/ref_grammar_vectors.json`` is vendored byte-identically in
tensorhub (``internal/orchestrator/release/testdata/``). Until th#1276 the file
was decorative — nothing loaded it in either repo, so the two parsers could
drift silently. This test (and its Go twin) make the fixture load-bearing.

th#1276 ruling under test: the grammar's default tag is ``prod`` (the stable
serving pointer); ``latest`` (the moving publish pointer) is now an ordinary
tag that must be written explicitly and round-trips stamped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker.models.refs import (
    TensorhubRef,
    format_model_ref,
    normalize_model_ref,
    parse_model_ref,
)

VECTORS_PATH = Path(__file__).parent / "testdata" / "ref_grammar_vectors.json"
_DOC = json.loads(VECTORS_PATH.read_text())
_VECTORS = _DOC["vectors"]

_OK = [v for v in _VECTORS if not v.get("error")]
_ERR = [v for v in _VECTORS if v.get("error")]


def _id(v: dict) -> str:
    return v["ref"] or "<empty>"


@pytest.mark.parametrize("vec", _OK, ids=[_id(v) for v in _OK])
def test_vector_parses_to_the_declared_fields(vec: dict) -> None:
    th = parse_model_ref(vec["ref"], provider="tensorhub").tensorhub
    assert th is not None
    assert (th.owner, th.repo, th.tag) == (vec["owner"], vec["repo"], vec["tag"])
    assert (th.digest or "") == vec["digest"]
    assert (th.flavor or "") == vec["flavor"]


@pytest.mark.parametrize("vec", _OK, ids=[_id(v) for v in _OK])
def test_vector_mints_the_declared_canonical_form(vec: dict) -> None:
    assert normalize_model_ref(vec["ref"]) == vec["canonical"]


@pytest.mark.parametrize("vec", _OK, ids=[_id(v) for v in _OK])
def test_canonical_form_is_a_fixed_point(vec: dict) -> None:
    """parse(canonical) == parse(ref), and normalizing twice changes nothing."""
    canonical = vec["canonical"]
    assert parse_model_ref(canonical).tensorhub == parse_model_ref(vec["ref"]).tensorhub
    assert normalize_model_ref(canonical) == canonical


@pytest.mark.parametrize("vec", _ERR, ids=[_id(v) for v in _ERR])
def test_error_vector_is_refused(vec: dict) -> None:
    with pytest.raises(ValueError):
        parse_model_ref(vec["ref"], provider="tensorhub")


def test_bare_ref_means_prod_not_latest() -> None:
    """The th#1276 ruling itself, stated once in plain terms."""
    assert parse_model_ref("owner/repo").tensorhub.tag == "prod"
    # prod elides, latest stamps — the elision flipped sides.
    assert normalize_model_ref("owner/repo:prod") == "owner/repo"
    assert normalize_model_ref("owner/repo:latest") == "owner/repo:latest"


def test_explicit_latest_survives_a_format_parse_round_trip() -> None:
    """The new edge: `latest` must never be silently absorbed into a bare ref."""
    for raw in (
        "owner/repo:latest",
        "owner/repo:latest#fp8",
        "owner/repo:latest@blake3:" + "ab" * 32,
    ):
        assert parse_model_ref(normalize_model_ref(raw)).tensorhub.tag == "latest"


def test_default_constructed_ref_agrees_with_the_parser() -> None:
    """TensorhubRef's field default and the parser default are one value."""
    assert TensorhubRef(owner="owner", repo="repo").tag == "prod"
    assert format_model_ref(parse_model_ref("owner/repo")) == TensorhubRef(
        owner="owner", repo="repo"
    ).canonical()
