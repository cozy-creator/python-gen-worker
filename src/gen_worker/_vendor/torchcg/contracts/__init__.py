"""Versioned compiled-graph identity corpora shipped with this package."""

from importlib.resources import files
from typing import Final

CONTRACT_FILES: Final = (
    "KEY_GRAMMAR_DIGEST",
    "call_ingress_v1.json",
    "compiled_graph_key_vectors.json",
    "graph_class_identity_v4.json",
    "ingress_selection_v1.json",
    "literal_identity_v1.json",
    "recipe_v1.json",
    "toolchain_identity_v1.json",
)


def read_contract(name: str) -> bytes:
    """Read one canonical contract without depending on a source checkout."""
    if name not in CONTRACT_FILES:
        raise ValueError(f"unknown compiled-graph contract: {name!r}")
    return files(__package__).joinpath(name).read_bytes()


__all__ = ["CONTRACT_FILES", "read_contract"]
