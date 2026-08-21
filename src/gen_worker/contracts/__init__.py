"""Versioned cross-repo contract corpora shipped with this package."""

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
