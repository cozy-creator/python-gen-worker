"""Mint-lane production: turn traced TCG declarations into a shipped key set.

This is the half that owns a tracer's OUTPUT — never a tracer. ``boot_key``
(mint-lane tooling since pgw#1327) exports each declared graph class on fake
tensors and hands the canonical declarations here; this module validates them
through TCG, extracts the graph axis plus the per-class metadata the issue names
(graph-class handle, ingress digest, target), and writes the document.

Two emission sites, one document:

* **A pod that traced** — ``boot_key.derive`` records the closure in this
  machine's cache, which is §4.28's compile-once-run-forever promise and also
  the file an operator copies out to bake into the next image.
* **``scripts/emit_cg_keyset.py``** — the same derivation run deliberately, off
  the serving path, printing the document for baking beside ``endpoint.lock``.

Nothing here folds a key. The document ships the machine-independent half only;
see :mod:`gen_worker.keyset.fold` for why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

from .document import ClosureRow, GraphClassRow
from .identifiers import (
    ClassHash, ClosureDigest, GraphClassName, KeySetError, parse_class_hash,
    parse_graph_class_name)
from .store import build_row, write_closure

__all__ = [
    "DECLARATION_FIELDS",
    "class_hashes_of",
    "record_closure",
    "rows_from_declarations",
]

# pgw#1299: these are ``GraphClassDeclaration``'s FIELD names, not TCG's
# artifact-metadata block. ``graph_class`` here is the field holding a class
# NAME; ``GRAPH_CLASS_BLOCK`` is the metadata block key. Equal strings, different
# vocabularies — substituting the block name would couple this wire format to a
# rename of something else entirely.
DECLARATION_FIELDS = frozenset({
    "class_hash",
    "class_dims",
    "fork",
    "graph",
    "graph_class",  # tcg-vocab: declaration field name, not the metadata block
    "graph_witness",
    "literal_values",
    "lora_bucket",
    "placement",
    "range_digest",
    "strict",
    "target",
})


def rows_from_declarations(
    declarations: Mapping[str, str],
) -> Dict[str, GraphClassRow]:
    """Reconstruct TCG declarations and emit one validated row per class.

    The trace child reports all public declaration facts, not a worker keying
    block. Reconstructing ``GraphClassDeclaration`` makes TCG validate graph
    ingress, range, witness, coordinates and literals; comparing its DERIVED hash
    to the child's stated hash catches a corrupt or drifted child wire. Only
    after that does anything become a shipped fact.
    """
    from gen_worker._vendor.torchcg import GraphClassDeclaration

    rows: Dict[str, GraphClassRow] = {}
    for name, canonical in declarations.items():
        handle = parse_graph_class_name(name)
        try:
            payload = json.loads(canonical)
        except (TypeError, ValueError) as exc:
            raise KeySetError(
                "invalid_declaration",
                f"graph class {name!r} has invalid declaration JSON: {exc}",
            ) from exc
        if not isinstance(payload, dict) or set(payload) != DECLARATION_FIELDS:
            raise KeySetError(
                "invalid_declaration",
                f"graph class {name!r} has an open TCG declaration schema")
        try:
            restated = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise KeySetError(
                "invalid_declaration",
                f"graph class {name!r} declaration is not finite JSON") from exc
        if canonical != restated:
            raise KeySetError(
                "invalid_declaration",
                f"graph class {name!r} declaration is not canonical JSON")
        if payload.get("graph_class") != name:  # tcg-vocab: declaration field
            raise KeySetError(
                "invalid_declaration",
                f"graph class map names {name!r}, declaration names "
                f"{payload.get('graph_class')!r}")  # tcg-vocab: declaration field
        try:
            declaration = GraphClassDeclaration(
                graph_class=payload["graph_class"],  # tcg-vocab: declaration field
                target=payload["target"],
                graph=payload["graph"],
                graph_witness=payload["graph_witness"],
                range_digest=payload["range_digest"],
                fork=tuple(tuple(row) for row in payload["fork"]),
                class_dims=tuple(tuple(row) for row in payload["class_dims"]),
                strict=payload["strict"],
                lora_bucket=payload["lora_bucket"],
                literal_values=payload["literal_values"],
                placement=tuple(payload["placement"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise KeySetError(
                "invalid_declaration",
                f"graph class {name!r} is not a valid TCG declaration: {exc}",
            ) from exc
        stated = payload.get("class_hash")
        if stated != declaration.class_hash:
            raise KeySetError(
                "invalid_declaration",
                f"graph class {name!r} states class_hash {stated!r}, TCG "
                f"derives {declaration.class_hash!r}")
        rows[str(handle)] = GraphClassRow(
            graph_class=str(handle),
            class_hash=str(parse_class_hash(declaration.class_hash)),
            ingress_digest=str(declaration.range_digest),
            target=str(declaration.target),
        )
    if not rows:
        raise KeySetError(
            "invalid_declaration", "the trace produced no graph class at all")
    return rows


def class_hashes_of(
    rows: Mapping[str, GraphClassRow],
) -> Dict[GraphClassName, ClassHash]:
    """The graph axis alone, for folding."""
    return {
        parse_graph_class_name(name): parse_class_hash(row.class_hash)
        for name, row in rows.items()
    }


def record_closure(
    cache_dir: Optional[Path],
    digest: ClosureDigest,
    rows: Mapping[str, GraphClassRow],
    *,
    family: str,
    function: str,
    tcg_version: str,
    emitted_by: str = "",
) -> ClosureRow:
    """Build the closure row and record it in this machine's cache.

    Returns the row whether or not the write landed: an unwritable cache is a
    slower next boot, never a wrong key, and the caller still needs the row to
    hand to whoever is baking an image.
    """
    row = build_row(
        family=family,
        function=function,
        tcg_version=tcg_version,
        classes=rows,
        emitted_by=emitted_by,
    )
    write_closure(cache_dir, digest, row)
    return row
