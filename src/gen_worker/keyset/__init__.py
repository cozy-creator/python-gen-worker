"""``cg-keyset-v1``: the derived compiled-graph key set, shipped as DATA.

pgw#1327. A serve pod's ``cg-key-v1`` set is a pure function of code the mint
lane already ran, so it is emitted once at mint time and consumed at serve boot
— instead of being re-derived by ``torch.export`` child processes on every fresh
pod. The package is deliberately tracer-free: it holds the identifier grammar,
the versioned document, the closure address, the fold, and the two lanes'
entry points, and it imports neither ``boot_key`` nor ``boot_trace_child``.

Read in this order: :mod:`identifiers` (the grammar), :mod:`document` (the
schema), :mod:`closure` (the address and why it is the whole safety story),
:mod:`fold` (graph axis x runtime axes), :mod:`store` (where documents live),
:mod:`hub` (the platform's own store, pgw#1353 option (b) / th#2123),
:mod:`boot` (serve-side consumption), :mod:`emit` (mint-side production).

----------------------------------------------------------------------------
FLAGGED FOR PAUL — a §4.29 MECHANISM AMENDMENT IS PENDING AND IS NOT MINE TO MAKE
----------------------------------------------------------------------------
DESIGN-RULINGS §4.29 states the mechanism verbatim: *"the worker itself is the
one that figures this out, by tracing the graphs."* This package keeps §4.29's
SUBSTANCE unchanged — adoption is still pull-by-key through the hub, still one
artifact or MISS and never a listing, and admission still verifies the answer
against the key — but it moves the DERIVATION to mint time and ships the result
as data, because a Rust serve host cannot trace and a serve pod should not spend
60 s of ``torch.export`` to learn what the mint lane already computed.

That is a change to a ruling's mechanism. **It requires Paul's amendment of
§4.29 and has not been made.** Built exactly as specified in pgw#1327 with the
flag left in place, here and in the tracker. §4.28 and §4.30 are untouched: no
forge, no mint requests, no compile fleet, and the minter stays an ordinary
Python serving pod that mints as a side effect of serving.
----------------------------------------------------------------------------
"""

from __future__ import annotations

from .boot import closure_of, key_set_from_data
from .closure import CLOSURE_VERSION, CONTRACT_MODULES, closure_digest, tcg_version
from .document import (
    KEYSET_FILENAME, KEYSET_SCHEMA, KEYSET_VERSION, ClosureRow, GraphClassRow,
    KeySetDocument, ShippedClass, ShippedClosure, decode, empty, encode,
    parse_closure)
from .fold import DerivedKeySet, KeySource, MemoVerdict, fold_entry_keys
from .hub import HubTier, fetch_closure, publish_closure, single_closure_document
from .identifiers import (
    ClassHash, ClosureDigest, CompiledGraphKey, FamilyName, GraphClassName,
    IngressDigest, KeySetError, parse_class_hash, parse_closure_digest,
    parse_compiled_graph_key, parse_family_name, parse_graph_class_name,
    parse_ingress_digest)

__all__ = [
    "CLOSURE_VERSION",
    "CONTRACT_MODULES",
    "ClassHash",
    "ClosureDigest",
    "ClosureRow",
    "CompiledGraphKey",
    "DerivedKeySet",
    "FamilyName",
    "GraphClassName",
    "GraphClassRow",
    "HubTier",
    "IngressDigest",
    "KEYSET_FILENAME",
    "KEYSET_SCHEMA",
    "KEYSET_VERSION",
    "KeySetDocument",
    "KeySetError",
    "KeySource",
    "MemoVerdict",
    "ShippedClass",
    "ShippedClosure",
    "closure_digest",
    "closure_of",
    "decode",
    "empty",
    "encode",
    "fetch_closure",
    "fold_entry_keys",
    "key_set_from_data",
    "parse_class_hash",
    "parse_closure",
    "parse_closure_digest",
    "parse_compiled_graph_key",
    "parse_family_name",
    "parse_graph_class_name",
    "parse_ingress_digest",
    "publish_closure",
    "single_closure_document",
    "tcg_version",
]
