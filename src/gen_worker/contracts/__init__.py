"""Versioned cross-repo contract corpora shipped with this package.

th#1947 §4.2. These corpora are the AUTHORITY for values that two repositories
must agree on, and they ship IN THE PACKAGE: a consumer that pins
``{authority, version, sha256}`` must be able to consume it without vendoring a
byte copy fenced against a moving branch tip.

Shipping them here makes the pin possible: a consumer installs a pinned version
and reads the exact bytes it pinned. The accessor shape deliberately matches
``torchcg.contracts`` so the two authorities read identically.

Corpus CONTENT is owned by the issue that introduced each corpus; this package
only makes the existing bytes importable.
"""

from importlib.resources import files
from typing import Final

CONTRACT_FILES: Final = (
    "COZY_RUNTIME_ENV_DIGEST",
    "FORMULA_VECTORS_DIGEST",
    "HUB_WORKER_BOUNDARY_CONTRACTS_DIGEST",
    "REF_GRAMMAR_DIGEST",
    "WORKER_VALUE_CONTRACTS_DIGEST",
    "cozy_runtime_env_vectors.json",
    "formula_vectors.json",
    "hub_worker_boundary_contracts.json",
    "posture_wire_vectors.json",
    "ref_grammar_vectors.json",
    "topology_wire_vectors.json",
    "worker_value_contracts.json",
)


def read_contract(name: str) -> bytes:
    """Read one canonical contract without depending on a source checkout."""
    if name not in CONTRACT_FILES:
        raise ValueError(f"unknown gen-worker contract: {name!r}")
    return files(__package__).joinpath(name).read_bytes()


__all__ = ["CONTRACT_FILES", "read_contract"]
