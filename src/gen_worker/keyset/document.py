"""``cg-keyset-v1`` — the derived key set, as DATA.

pgw#1327. A serve pod used to learn its own ``cg-key-v1`` set by running
``torch.export`` in child processes at boot ("< 60 s, every time, ~99 % of it
the traces"), which made torch, diffusers and the endpoint's model code a hard
dependency of a pod that will never compile anything. Every input to those
traces is code the mint lane already ran, so the OUTPUT — one TCG ``class_hash``
per declared graph class — is shipped instead.

What the document holds, and what it deliberately does not
---------------------------------------------------------
It holds the **graph half** of the identity: per class, TCG's machine-independent
``class_hash`` plus its metadata (graph-class handle, ingress digest, target).
It does **not** hold folded ``cg-key-v1`` values. The other two axes — ``sm`` and
``toolchain`` — cost milliseconds to restate and MUST be restated every boot: an
sm or a toolchain that changed has to move every key on the very next boot, and a
shipped key would answer with the minting pod's. One document therefore serves
every SKU that runs this code, which is the same reason ``boot_key``'s memo only
ever held the graph half.

Versioned, and unknown versions are refused LOUDLY
--------------------------------------------------
``schema``/``version`` are checked before any field shape is trusted, and every
struct forbids unknown fields. A reader that finds a version it does not know
raises :class:`~gen_worker.keyset.identifiers.KeySetError` rather than reading
the fields it recognises — a partially-understood key set is not a narrower
answer, it is a wrong one, and its wrong answer selects a wrong graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import msgspec

from .identifiers import (
    ClassHash, ClosureDigest, FamilyName, GraphClassName, IngressDigest,
    KeySetError, parse_class_hash, parse_closure_digest, parse_family_name,
    parse_graph_class_name, parse_ingress_digest)

__all__ = [
    "GraphClassRow",
    "ClosureRow",
    "KeySetDocument",
    "ShippedClass",
    "ShippedClosure",
    "KEYSET_FILENAME",
    "KEYSET_SCHEMA",
    "KEYSET_VERSION",
    "decode",
    "encode",
    "empty",
]

#: The document's own name. Bumped only when the MEANING of a stored value
#: changes; readers that do not know a version refuse the whole file.
KEYSET_SCHEMA = "cg-keyset-v1"
KEYSET_VERSION = 1

#: The on-disk name, wherever a key set is shipped or cached.
KEYSET_FILENAME = "cg-keyset-v1.json"


class GraphClassRow(msgspec.Struct, frozen=True, kw_only=True,
                    forbid_unknown_fields=True):
    """One declared graph class, as the mint lane derived it."""

    #: TCG ``GraphClassDeclaration.graph_class`` — the handle, not an identity.
    graph_class: str
    #: TCG ``class_hash``: the machine-independent graph axis. THE payload.
    class_hash: str
    #: TCG ``range_digest`` = ``CallIngress.digest()``. Per-class metadata the
    #: typed-binding codegen (pgw#1332) keys its signatures off.
    ingress_digest: str
    #: The declaration's compile target this class belongs to.
    target: str


class ClosureRow(msgspec.Struct, frozen=True, kw_only=True,
                 forbid_unknown_fields=True):
    """Every class one closure declares, plus the forensics of its emission."""

    family: str
    function: str
    #: The vendored TCG rev whose declaration semantics produced these hashes.
    #: Also folded into the closure digest, so this is forensics, not a gate.
    tcg_version: str
    classes: Dict[str, GraphClassRow]
    emitted_unix: int = 0
    #: Free-form provenance ("gen-worker 0.120.0 mint", a rig name). Never read
    #: by a decision — a document is trusted because its closure digest matches
    #: this pod's code, never because of who signed the sentence.
    emitted_by: str = ""


class KeySetDocument(msgspec.Struct, frozen=True, kw_only=True,
                     forbid_unknown_fields=True):
    """closure digest -> that closure's class set."""

    schema: str
    version: int
    closures: Dict[str, ClosureRow]


@dataclass(frozen=True)
class ShippedClass:
    """One class of a shipped closure, with every identifier parsed."""

    graph_class: GraphClassName
    class_hash: ClassHash
    ingress_digest: IngressDigest
    target: str


@dataclass(frozen=True)
class ShippedClosure:
    """A closure's class set after every identifier has been parsed.

    This is what the boot path consumes. Nothing downstream ever sees the raw
    document: by the time a key is folded, each hash has passed TCG's own
    grammar and each handle is a ``GraphClassName``.
    """

    digest: ClosureDigest
    family: FamilyName
    function: str
    tcg_version: str
    classes: Tuple[ShippedClass, ...]
    emitted_unix: int = 0
    emitted_by: str = ""

    @property
    def class_hashes(self) -> Dict[GraphClassName, ClassHash]:
        return {row.graph_class: row.class_hash for row in self.classes}


def empty() -> KeySetDocument:
    return KeySetDocument(
        schema=KEYSET_SCHEMA, version=KEYSET_VERSION, closures={})


def encode(document: KeySetDocument) -> bytes:
    """Canonical bytes for a document. Deterministic: msgspec emits struct
    fields in declaration order and ``closures`` is written sorted, so two
    emissions of the same facts are byte-identical and a shipped document can
    be content-addressed by whoever bakes it."""
    if document.schema != KEYSET_SCHEMA or document.version != KEYSET_VERSION:
        raise KeySetError(
            "keyset_invalid",
            f"refusing to encode {document.schema!r} v{document.version} as "
            f"{KEYSET_SCHEMA} v{KEYSET_VERSION}")
    ordered = KeySetDocument(
        schema=document.schema,
        version=document.version,
        closures={k: document.closures[k] for k in sorted(document.closures)},
    )
    return msgspec.json.format(msgspec.json.encode(ordered), indent=1)


def decode(raw: bytes) -> KeySetDocument:
    """Parse a key-set document, refusing unknown versions and fields.

    The envelope is read as plain JSON FIRST so that a future ``cg-keyset-v2``
    is answered with "this reader does not know v2" instead of a field-shape
    complaint about whichever field v2 happened to add.
    """
    try:
        payload: Any = msgspec.json.decode(raw)
    except (msgspec.DecodeError, ValueError) as exc:
        raise KeySetError(
            "keyset_invalid", f"key set is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise KeySetError("keyset_invalid", "key set must be a JSON object")
    schema = payload.get("schema")
    version = payload.get("version")
    if schema != KEYSET_SCHEMA or version != KEYSET_VERSION:
        raise KeySetError(
            "keyset_invalid",
            f"key set declares {schema!r} v{version!r}; this worker reads "
            f"{KEYSET_SCHEMA} v{KEYSET_VERSION} and refuses to read a document "
            f"it does not fully understand")
    try:
        return msgspec.convert(payload, type=KeySetDocument, strict=True)
    except msgspec.ValidationError as exc:
        raise KeySetError(
            "keyset_invalid", f"key set does not match {KEYSET_SCHEMA}: {exc}"
        ) from exc


def parse_closure(
    document: KeySetDocument, digest: ClosureDigest,
) -> ShippedClosure:
    """The validated view of ONE closure. Raises when the row is unusable."""
    row = document.closures.get(str(digest))
    if row is None:
        raise KeySetError(
            "keyset_absent",
            f"the key set holds no closure {digest}")
    if not row.classes:
        raise KeySetError(
            "keyset_invalid",
            f"closure {digest} declares an empty class set; a cell with no "
            f"class set has no identity (pgw#716/#758)")
    classes: list[ShippedClass] = []
    for name, entry in sorted(row.classes.items()):
        handle = parse_graph_class_name(name)
        stated = parse_graph_class_name(entry.graph_class)
        if stated != handle:
            raise KeySetError(
                "keyset_invalid",
                f"closure {digest} maps {name!r} to a row naming "
                f"{entry.graph_class!r}")
        classes.append(ShippedClass(
            graph_class=handle,
            class_hash=parse_class_hash(entry.class_hash),
            ingress_digest=parse_ingress_digest(entry.ingress_digest),
            target=str(entry.target),
        ))
    return ShippedClosure(
        digest=parse_closure_digest(str(digest)),
        family=parse_family_name(row.family),
        function=str(row.function),
        tcg_version=str(row.tcg_version),
        classes=tuple(classes),
        emitted_unix=int(row.emitted_unix),
        emitted_by=str(row.emitted_by),
    )


def closure_row(
    *,
    family: str,
    function: str,
    tcg_version: str,
    classes: Mapping[str, GraphClassRow],
    emitted_unix: int = 0,
    emitted_by: str = "",
) -> ClosureRow:
    """Build one closure row, validating every identifier on the way IN.

    Emission validates with the same parsers consumption does, so a document
    this worker wrote is one this worker can read — the failure mode where a
    producer and a consumer disagree about a grammar cannot occur here.
    """
    if not classes:
        raise KeySetError(
            "keyset_invalid", "a closure row needs at least one graph class")
    clean: Dict[str, GraphClassRow] = {}
    for name, row in classes.items():
        handle = parse_graph_class_name(name)
        parse_class_hash(row.class_hash)
        parse_ingress_digest(row.ingress_digest)
        if parse_graph_class_name(row.graph_class) != handle:
            raise KeySetError(
                "keyset_invalid",
                f"class map names {name!r}, row names {row.graph_class!r}")
        clean[str(handle)] = row
    parse_family_name(family)
    return ClosureRow(
        family=str(family),
        function=str(function),
        tcg_version=str(tcg_version),
        classes={k: clean[k] for k in sorted(clean)},
        emitted_unix=int(emitted_unix),
        emitted_by=str(emitted_by),
    )
