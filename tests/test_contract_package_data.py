"""th#1947 §4.2 — the cross-repo corpora must be importable package data.

The defect this fences: every corpus lived under ``tests/testdata/`` only, and
the built wheel and sdist carry ``src/gen_worker`` — so a consumer that pinned a
released version could not read them at all, and the ecosystem vendored byte
copies fenced by drift scripts that fetched a MOVING branch tip. A peer's merge
timing then became every other repo's CI event.

``tests/testdata`` stays as a byte-gated projection while the peers that still
read it from a source checkout move onto the pin. It is not a second authority:
the equality test below is what makes that safe, and it is what must go red if
someone edits one copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gen_worker.contracts import CONTRACT_FILES, read_contract

_TESTDATA = Path(__file__).parent / "testdata"


def test_every_contract_is_importable_package_data() -> None:
    for name in CONTRACT_FILES:
        data = read_contract(name)
        assert data, name
        if name.endswith(".json"):
            json.loads(data)


def test_unknown_contract_name_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown gen-worker contract"):
        read_contract("../pyproject.toml")


@pytest.mark.parametrize("name", CONTRACT_FILES)
def test_testdata_projection_is_exactly_canonical(name: str) -> None:
    """Remove this projection once every peer reads the pinned package."""
    assert (_TESTDATA / name).read_bytes() == read_contract(name)


@pytest.mark.parametrize(
    "corpus,digest_name",
    [
        ("cozy_runtime_env_vectors.json", "COZY_RUNTIME_ENV_DIGEST"),
        ("formula_vectors.json", "FORMULA_VECTORS_DIGEST"),
        ("hub_worker_boundary_contracts.json", "HUB_WORKER_BOUNDARY_CONTRACTS_DIGEST"),
        ("ref_grammar_vectors.json", "REF_GRAMMAR_DIGEST"),
        ("worker_value_contracts.json", "WORKER_VALUE_CONTRACTS_DIGEST"),
    ],
)
def test_packaged_corpus_matches_its_packaged_digest(corpus: str, digest_name: str) -> None:
    """The digest sidecar describes the corpus shipped BESIDE it, not a checkout."""
    import hashlib

    # The sidecars are not written in one format: some are a bare digest, some
    # are `sha256sum` output (digest, two spaces, filename). Both are legitimate
    # and neither is this change's to normalise, so read the first field.
    recorded = next(
        line.split()[0]
        for line in read_contract(digest_name).decode().splitlines()
        if line.strip() and not line.startswith("#")
    )
    assert hashlib.sha256(read_contract(corpus)).hexdigest() == recorded
