import pytest

from gen_worker.model_refs import parse_model_ref


def test_parse_cozy_default_scheme() -> None:
    p = parse_model_ref("org/repo")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.org == "org"
    assert p.cozy.repo == "repo"
    assert p.cozy.tag == "latest"


def test_parse_cozy_tag() -> None:
    p = parse_model_ref("cozy:org/repo:v1")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.tag == "v1"
    assert p.cozy.digest is None


def test_parse_cozy_digest() -> None:
    p = parse_model_ref("org/repo@sha256:abcd")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.digest == "abcd"


def test_parse_hf_basic() -> None:
    p = parse_model_ref("hf:org/repo")
    assert p.scheme == "hf"
    assert p.hf is not None
    assert p.hf.repo_id == "org/repo"
    assert p.hf.revision is None


def test_parse_hf_revision() -> None:
    p = parse_model_ref("hf:org/repo@main")
    assert p.scheme == "hf"
    assert p.hf is not None
    assert p.hf.revision == "main"


def test_parse_invalid_hf() -> None:
    with pytest.raises(ValueError):
        parse_model_ref("hf:justonepart")

