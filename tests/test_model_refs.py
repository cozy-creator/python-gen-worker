import pytest

from gen_worker.model_refs import parse_model_ref


def test_parse_cozy_default_scheme() -> None:
    p = parse_model_ref("owner/repo")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.owner == "owner"
    assert p.cozy.repo == "repo"
    assert p.cozy.tag == "latest"


def test_parse_cozy_tag() -> None:
    p = parse_model_ref("cozy:owner/repo:v1")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.tag == "v1"
    assert p.cozy.digest is None


def test_parse_cozy_digest() -> None:
    p = parse_model_ref("owner/repo@sha256:abcd")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.digest == "sha256:abcd"


def test_parse_cozy_digest_blake3() -> None:
    p = parse_model_ref("owner/repo@blake3:abcd")
    assert p.scheme == "cozy"
    assert p.cozy is not None
    assert p.cozy.digest == "blake3:abcd"


def test_parse_hf_basic() -> None:
    p = parse_model_ref("hf:owner/repo")
    assert p.scheme == "hf"
    assert p.hf is not None
    assert p.hf.repo_id == "owner/repo"
    assert p.hf.revision is None


def test_parse_hf_revision() -> None:
    p = parse_model_ref("hf:owner/repo@main")
    assert p.scheme == "hf"
    assert p.hf is not None
    assert p.hf.revision == "main"


def test_parse_invalid_hf() -> None:
    with pytest.raises(ValueError):
        parse_model_ref("hf:justonepart")
