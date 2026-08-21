from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gen_worker.models.refs import (
    RefFragmentRemoved,
    RetiredTagRef,
    TensorhubRef,
    fold_ref,
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


def _address(v: dict) -> str:
    return v.get("address", v["canonical"])


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
    assert (th.owner, th.repo, th.release) == (
        vec["owner"], vec["repo"], vec["release"])
    assert (th.digest or "") == vec["digest"]
    assert (th.fragment or "") == vec["fragment"]
    assert th.lane_spec == vec.get("lane_spec", "")


@pytest.mark.parametrize("vec", _OK, ids=[_id(v) for v in _OK])
def test_vector_mints_the_declared_address(vec: dict) -> None:
    assert normalize_model_ref(vec["ref"]) == _address(vec)


@pytest.mark.parametrize("vec", _OK, ids=[_id(v) for v in _OK])
def test_canonical_form_is_a_fixed_point(vec: dict) -> None:
    """parse(canonical) == parse(ref), and re-normalizing changes nothing."""
    canonical = vec["canonical"]
    assert parse_model_ref(canonical).tensorhub == parse_model_ref(vec["ref"]).tensorhub
    assert normalize_model_ref(canonical) == _address(vec)
    assert normalize_model_ref(_address(vec)) == _address(vec)


@pytest.mark.parametrize("vec", _ERR, ids=[_id(v) for v in _ERR])
def test_error_vector_is_refused(vec: dict) -> None:
    with pytest.raises(ValueError):
        parse_model_ref(vec["ref"], provider="tensorhub")


_RETIRED_TAG_REFS = (
    "owner/repo:prod",
    "owner/repo:latest",
    "owner/repo:v2#fp8",
)


def test_bare_ref_names_no_release() -> None:
    assert parse_model_ref("owner/repo").tensorhub.release == ""
    for retired in _RETIRED_TAG_REFS:
        with pytest.raises(RetiredTagRef) as err:
            parse_model_ref(retired)
        assert "owner/repo@<release>" in str(err.value)


def test_release_tail_round_trips_verbatim() -> None:
    for raw in ("owner/repo@prod", "owner/repo@latest", "owner/repo@2026.08"):
        assert normalize_model_ref(raw) == raw
        assert parse_model_ref(normalize_model_ref(raw)).tensorhub.release != ""


def test_digest_wins_over_release_in_the_one_at_slot() -> None:
    """One `@` slot, and the exact answer takes it — the Go twin's TestDigestWinsOverRelease."""
    hexd = "ab" * 32
    th = parse_model_ref(f"owner/repo@r1@sha256:{hexd}").tensorhub
    assert (th.release, th.digest) == ("r1", f"sha256:{hexd}")
    assert th.canonical() == f"owner/repo@sha256:{hexd}"


def test_lane_spec_rides_beside_the_address_never_inside_it() -> None:
    th = parse_model_ref("owner/repo@prod?quant=plain.bf16@1").tensorhub
    assert (th.release, th.lane_spec) == ("prod", "quant=plain.bf16@1")
    assert th.canonical() == "owner/repo@prod"
    assert normalize_model_ref(th.canonical()) == "owner/repo@prod"


def test_a_fragment_side_query_is_still_discarded() -> None:
    """The lockfile-attribution `?` on the FRAGMENT half keeps its old meaning: discarded, not stored."""
    th = parse_model_ref("root/family-sdxl#inductor-rtx-4090-torch2.9?src=lockfile").tensorhub
    assert (th.fragment, th.lane_spec) == ("inductor-rtx-4090-torch2.9", "")


def test_a_weight_ref_fragment_is_a_typed_refusal_th2031() -> None:
    for ref in ("owner/repo#fp8", "owner/repo@prod#fp8",
                "owner/repo@prod?quant=plain.bf16@1#fp8",
                "notroot/family-sdxl#inductor-rtx-4090-torch2.9",
                "root/sdxl#inductor-rtx-4090-torch2.9"):
        with pytest.raises(RefFragmentRemoved) as err:
            parse_model_ref(ref)
        assert "?<contract pattern>" in str(err.value)
    for graph in ("root/family-sdxl#inductor-rtx-4090-torch2.9",
                 "root/family-sdxl@prod#inductor-rtx-4090-torch2.9"):
        th = parse_model_ref(graph).tensorhub
        assert th.fragment == "inductor-rtx-4090-torch2.9"
        assert th.canonical() == graph


def test_folding_a_release_onto_a_spec_ref_mints_no_double_at() -> None:
    folded = fold_ref("owner/repo@prod?quant=plain.bf16@1", release="staging")
    assert folded == "owner/repo@staging"
    hexd = "cd" * 32
    digest_folded = fold_ref(f"owner/repo@sha256:{hexd}?quant=plain.bf16@1")
    assert digest_folded == f"owner/repo@sha256:{hexd}"
    for minted in (folded, digest_folded):
        assert minted.count("@") == 1 and "?" not in minted
        assert normalize_model_ref(minted) == minted


def test_an_empty_lane_spec_is_refused_by_name() -> None:
    """A bare `?` is not "any variant" — it is a caller who meant to write one and did not."""
    for ref in ("owner/repo@prod?", "owner/repo?"):
        with pytest.raises(ValueError) as err:
            parse_model_ref(ref)
        assert "omit the '?' entirely" in str(err.value)


def test_default_constructed_ref_agrees_with_the_parser() -> None:
    """TensorhubRef's field default and the parser default are one value: NO release."""
    assert TensorhubRef(owner="owner", repo="repo").release == ""
    assert format_model_ref(parse_model_ref("owner/repo")) == TensorhubRef(
        owner="owner", repo="repo"
    ).canonical()
