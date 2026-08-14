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

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gen_worker.models.refs import (
    TensorhubRef,
    format_model_ref,
    normalize_model_ref,
    parse_model_ref,
)

VECTORS_PATH = Path(__file__).parent / "testdata" / "ref_grammar_vectors.json"
REF_DIGEST_PATH = Path(__file__).parent / "testdata" / "REF_GRAMMAR_DIGEST"
_DOC = json.loads(VECTORS_PATH.read_text())
_VECTORS = _DOC["vectors"]

_OK = [v for v in _VECTORS if not v.get("error")]
_ERR = [v for v in _VECTORS if v.get("error")]


def _id(v: dict) -> str:
    return v["ref"] or "<empty>"


def _contract_paths() -> tuple[Path, Path]:
    return (
        Path(os.environ.get("REF_GRAMMAR_VECTOR_FILE", VECTORS_PATH)),
        Path(os.environ.get("REF_GRAMMAR_DIGEST_FILE", REF_DIGEST_PATH)),
    )


def _recorded_digest(path: Path) -> str:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split()[0]
    return ""


def test_ref_grammar_corpus_digest_th1914() -> None:
    corpus, digest_file = _contract_paths()
    actual = hashlib.sha256(corpus.read_bytes()).hexdigest()
    recorded = _recorded_digest(digest_file)
    assert recorded == actual, (
        "ref_grammar_vectors.json changed without its independent digest: "
        f"recorded={recorded!r}, actual={actual}"
    )


def test_ref_grammar_digest_gate_can_go_red_th1914(tmp_path: Path) -> None:
    candidate = tmp_path / "ref_grammar_vectors.json"
    candidate.write_bytes(VECTORS_PATH.read_bytes() + b"\n")
    env = os.environ.copy()
    env["REF_GRAMMAR_VECTOR_FILE"] = str(candidate)
    env["REF_GRAMMAR_DIGEST_FILE"] = str(REF_DIGEST_PATH)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{Path(__file__).resolve()}::test_ref_grammar_corpus_digest_th1914",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "changed without its independent digest" in proc.stdout + proc.stderr


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
