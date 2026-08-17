"""Validated identifier types for the shipped compiled-graph key set.

Parse-don't-validate, per pgw#1326 "Strict typing": *"Identifiers are validated
types, never bare ``str``"*. Every value that crosses this package's boundary
arrives as raw JSON and leaves as a ``NewType`` that only the parser in this
module can mint, so a caller cannot accidentally pass a graph-class NAME where a
class HASH belongs — both are short opaque strings and mypy is the only thing
that can tell them apart at a call site.

The grammar model is ``torchcg.identity.is_compiled_graph_key``: refuse SHAPE at
the boundary, name the reason, and never coerce.
"""

from __future__ import annotations

import re
from typing import NewType

__all__ = [
    "MAX_HANDLE",
    "ClassHash",
    "ClosureDigest",
    "CompiledGraphKey",
    "FamilyName",
    "GraphClassName",
    "IngressDigest",
    "KeySetError",
    "parse_class_hash",
    "parse_closure_digest",
    "parse_compiled_graph_key",
    "parse_family_name",
    "parse_graph_class_name",
    "parse_ingress_digest",
]


class KeySetError(ValueError):
    """A key-set document, or one identifier inside it, is not admissible.

    ``reason`` is a stable token (the boot-adopt vocabulary carries the two that
    reach an event: ``keyset_invalid`` and ``keyset_absent``); ``detail`` is the
    sentence a human needs. Never a bare ``ValueError``: a refusal here means a
    pod declined to state a key, and "why" is the whole product of the refusal.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"{reason}: {detail}")


#: One graph class's human handle inside a family's declaration — TCG's
#: ``GraphClassDeclaration.graph_class`` field. A HANDLE, never an identity
#: (pgw#1326 "Catalog escape hatches": names resolve through the document,
#: they never key anything).
GraphClassName = NewType("GraphClassName", str)

#: TCG's ``class_hash`` — the machine-independent GRAPH half of a ``cg-key-v1``.
#: This is the only identity fact this package ships, and it is exactly why the
#: document travels: it is a pure function of code, so the mint lane can compute
#: it once and every pod that runs that code folds it with its own runtime axes.
ClassHash = NewType("ClassHash", str)

#: TCG's ``range_digest`` — ``CallIngress.digest()``, the class's call-ingress
#: identity. Per-class metadata the issue names; the typed-binding codegen of
#: pgw#1332 keys its signatures off this same value.
IngressDigest = NewType("IngressDigest", str)

#: The document's ADDRESS: what a per-class graph hash is a pure function of.
#: See :mod:`gen_worker.keyset.closure`.
ClosureDigest = NewType("ClosureDigest", str)

#: A folded ``cg-key-v1`` value — graph x sm x toolchain. Produced only by
#: :func:`gen_worker.keyset.fold.fold_entry_keys`; never shipped, never stored.
CompiledGraphKey = NewType("CompiledGraphKey", str)

#: A family handle (``CompileSpec.family``).
FamilyName = NewType("FamilyName", str)


_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")

#: A handle is bounded and printable, and nothing more. The SHAPE rules for a
#: graph-class name are TCG's (below) — inventing a stricter grammar here is how
#: a real declaration (``transformer/cfg=false``) becomes an unreadable key set.
MAX_HANDLE = 256
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _canonical(reason: str, what: str, value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise KeySetError(reason, f"{what} must be a canonical non-empty string")
    return value


def parse_graph_class_name(value: object) -> GraphClassName:
    """A declaration's graph-class handle.

    The rules are TCG's own (``GraphClassDeclaration.__post_init__``): non-empty,
    no backslash, and no ``/``-segment that is empty, ``.`` or ``..``. Nested
    names (``vae/decoder``) and fork-bearing names (``transformer/cfg=false``)
    are ordinary. Restated in that form rather than as a character class,
    because a stricter grammar here would refuse declarations TCG accepts — and
    a key set that cannot name a real class is a pod that traces forever.
    """
    text = _canonical("keyset_invalid", "graph class name", value)
    if len(text) > MAX_HANDLE or _CONTROL.search(text):
        raise KeySetError(
            "keyset_invalid",
            f"graph class name {text[:40]!r} is unbounded or unprintable")
    if "\\" in text or any(
        part in ("", ".", "..") for part in text.split("/")
    ):
        raise KeySetError(
            "keyset_invalid", f"unsafe graph class name {text!r}")
    return GraphClassName(text)


def parse_family_name(value: object) -> FamilyName:
    """A family handle. Bounded, printable, and not a path."""
    text = _canonical("keyset_invalid", "family name", value)
    if len(text) > MAX_HANDLE or _CONTROL.search(text):
        raise KeySetError(
            "keyset_invalid",
            f"family name {text[:40]!r} is unbounded or unprintable")
    if "/" in text or "\\" in text:
        raise KeySetError(
            "keyset_invalid", f"family name {text!r} is not a bare handle")
    return FamilyName(text)


def parse_class_hash(value: object) -> ClassHash:
    """TCG's 16-hex ``class_hash``. The same grammar ``boot_key`` enforced on
    its local memo, moved here so ONE parser admits the graph axis whether it
    arrives from a trace, from this pod's cache, or from a shipped document."""
    text = _canonical("keyset_invalid", "class hash", value)
    if _HEX16.fullmatch(text) is None:
        raise KeySetError(
            "keyset_invalid",
            f"class hash {text!r} is not TCG's 16-hex class_hash")
    return ClassHash(text)


def parse_ingress_digest(value: object) -> IngressDigest:
    """TCG's 32-hex ``range_digest`` (``CallIngress.digest()``)."""
    text = _canonical("keyset_invalid", "ingress digest", value)
    if _HEX32.fullmatch(text) is None:
        raise KeySetError(
            "keyset_invalid",
            f"ingress digest {text!r} is not TCG's 32-hex range_digest")
    return IngressDigest(text)


def parse_closure_digest(value: object) -> ClosureDigest:
    text = _canonical("keyset_invalid", "closure digest", value)
    if _HEX32.fullmatch(text) is None:
        raise KeySetError(
            "keyset_invalid", f"closure digest {text!r} is not 32 hex characters")
    return ClosureDigest(text)


def parse_compiled_graph_key(value: object) -> CompiledGraphKey:
    """Admit a folded key through TCG's OWN boundary grammar.

    Deliberately delegated rather than re-spelled: a second regex for the key
    shape is a second answer to "is this a key", and pgw#1299's scar is exactly
    that class of drift.
    """
    from gen_worker._vendor.torchcg.identity import is_compiled_graph_key

    text = _canonical("keyset_invalid", "compiled graph key", value)
    if not is_compiled_graph_key(text):
        raise KeySetError(
            "keyset_invalid",
            f"{text!r} does not have TCG's compiled-graph key shape")
    return CompiledGraphKey(text)
