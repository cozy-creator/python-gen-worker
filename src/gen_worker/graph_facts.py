"""Worker-owned FACTS about a declaration and about a mint obligation — never
identity.

pgw#1277 split the old ``compiled_graph_key`` module along the boundary it had
always described but never enforced. Compiled-graph IDENTITY — the three
``cg-key-v1`` axes, their canonical fold, the key grammar and the derivation of
a key from an artifact's recorded facts — belongs to
:mod:`torchcg.identity` and lives there only. What survives here
is everything the worker owns because TCG has no concept of it:

* the DECLARED ENVELOPE (:func:`envelope_facts`) — the serving region an author
  declared. A property of the COLLECTION, so pgw#1176 evicted it from the key;
  it rides the derived contract manifest instead.
* the CONTRACT MANIFEST label (:func:`manifest_digest`) — which class SET one
  declaration traces to. Telemetry and coverage; nothing resolves, downloads,
  verifies or arms it. The moment it becomes something a pod downloads, the
  wrong atom is back.
* the OBLIGATION SUBJECT (:class:`SlotSubject`) — WHICH checkpoint one setup
  slot resolved to.

THE ASYMMETRY that makes the split correct, and the reason the subject must
never migrate into TCG: the KEY is the computation and must not OVER-split;
the ARM TOKEN and the boot-key memo are an OBLIGATION and a CACHE LOOKUP and
must not UNDER-split. Over-splitting an obligation costs one re-mint;
under-splitting one binds a pipeline to a compiled graph nobody proved is its
computation. One compiled graph legally serves every checkpoint whose graph it
is — weight VALUES are never hashed — so keying on the checkpoint would put
every fine-tune in its own key space. What the subject does is stop two
DIFFERENT checkpoints sharing one pending mint, one store memo or one boot-key
row on the strength of an assumption nothing checked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from gen_worker._vendor.torchcg import is_compiled_graph_key

#: The MANIFEST block recording one declaration's DECLARED ENVELOPE.
#:
#: Naming note (pgw#1059 §F): this is the DECLARED envelope. The OTHER
#: "envelope" in this codebase is the artifact-metadata blob itself (the hub
#: error envelope). Where both appear, write "the artifact-metadata envelope"
#: vs "the declared envelope".
EXPORT_ENVELOPE_KEY = "declared_envelope"


class GraphFactsError(ValueError):
    """A recorded fact block cannot be canonicalized."""


def _refuse_key_shaped(where: str, name: str, value: str) -> None:
    """A KEY where a FACT DIGEST belongs is a category error, not a bad value.

    Keys and the 16-hex fact digests are all :class:`str`, so nothing in the
    type system distinguishes them — that is the honest reason a whole class of
    confusion type-checks cleanly. Refuse it by name here rather than letting it
    fold into a digest nobody can restate.
    """
    if is_compiled_graph_key(value):
        raise GraphFactsError(
            f"{where}: {name}={value!r} is a COMPILED-GRAPH KEY where a fact "
            f"digest belongs. A key is the OUTPUT of identity, never an input "
            f"to it — folding one in here would produce a digest no artifact "
            f"can restate.")


def facts_digest(facts: Mapping[str, Any]) -> str:
    """16-hex canonical digest of one recorded fact block.

    Computed identically from live probes and from recorded metadata, so a
    stamp can never disagree with the facts it summarizes.
    """
    encoded = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def envelope_facts(block: Mapping[str, Any]) -> Dict[str, Any]:
    """The canonical form of one DECLARED-ENVELOPE block — the single
    canonicalizer of the declared serving region, so no two consumers can
    canonicalize the same declaration differently.

    pgw#1176: a MANIFEST fact, never a key axis. It digests the UNION of the
    ladder across the whole declaration, which is a property of the collection;
    per graph class the tracing-relevant half of it is already inside that
    class's own hash via ``range_digest`` and ``class_dims``. Keying on it
    re-minted 35 unchanged classes every time an author added an aspect ratio.

    ``overlay`` is the behavior-posture slot (pgw#1059 amendment 5): a typed,
    allowlisted author-settings overlay digested into the envelope when one is
    declared. The menu is EMPTY today, so the slot is omitted whenever falsy,
    which keeps every current declaration's canonical form free of a field that
    says "unchanged".
    """
    facts: Dict[str, Any] = {
        "v": 1,
        "shapes": sorted(
            [int(v) for v in row] for row in (block.get("shapes") or ())),
        "text_lens": sorted({int(v) for v in (block.get("text_lens") or ())}),
        "guidance": sorted(float(v) for v in (block.get("guidance") or ())),
    }
    overlay = block.get("overlay")
    if overlay:
        facts["overlay"] = {
            str(k): str(v) for k, v in sorted(dict(overlay).items())}
    return facts


def envelope_digest(block: Mapping[str, Any]) -> str:
    """The digest of one declared-envelope block."""
    return facts_digest(envelope_facts(block))


@dataclass(frozen=True)
class SlotSubject:
    """WHICH checkpoint one setup slot resolved to.

    ``refs`` is the slot's wire ref (th#1941 left one ref per slot — a composed
    manifest has no second ref to name); ``snapshot_digest`` is the materialized
    tree's content digest when the resolver stated one (it is ``""`` for a slot
    resolved without one, which is a narrower statement, never a different
    subject).
    """

    slot: str
    refs: Tuple[str, ...] = ()
    snapshot_digest: str = ""


def subject_facts(subjects: Iterable[SlotSubject]) -> Dict[str, Any]:
    """The canonical SUBJECT block for one arm/trace — sorted by slot, so two
    callers that resolved the same slots in different orders state one fact."""
    return {
        "v": 1,
        "slots": [
            [sub.slot, list(sub.refs), sub.snapshot_digest]
            for sub in sorted(tuple(subjects), key=lambda s: s.slot)
        ],
    }


def subject_digest(subjects: Iterable[SlotSubject]) -> str:
    """16-hex digest of the resolved subject, or ``""`` when the caller could
    state none.

    ``""`` is the honest answer for a pipeline whose slot resolution this
    process never saw (an endpoint that builds its own pipeline out of a
    path-valued slot, reached through ``arm_compile()``); it UNDER-splits, so
    every caller that CAN state a subject states one.
    """
    subs = tuple(subjects)
    if not subs:
        return ""
    return facts_digest(subject_facts(subs))


def manifest_digest(class_hashes: Iterable[str]) -> str:
    """The coverage LABEL of one declaration's class set — 16 hex of sha256
    over the newline-joined SORTED per-class hashes.

    This is ``combined_graph_hash``'s arithmetic, VERBATIM (pgw#716: sorted by
    the hash string itself, single ``\\n`` joins, no trailing newline, UTF-8
    bytes) and NOT its job. pgw#1176 demoted it from identity to a
    telemetry/coverage label:

    * it is what the hub folds compile-health rows under — one row per
      ``(manifest_digest, sm, toolchain)``, coverage n/m, last failure per class
      (§1.7). It is deliberately sm- and toolchain-FREE so that tuple is not
      degenerate;
    * nothing resolves, downloads, verifies or arms it.

    A manifest is a VIEW. The moment it becomes something a pod downloads or a
    hub hands back as one row, the wrong atom is back.
    """
    rows = [str(h) for h in class_hashes]
    for row in rows:
        _refuse_key_shaped("manifest digest", "class_hash", row)
    joined = "\n".join(sorted(rows))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "EXPORT_ENVELOPE_KEY",
    "GraphFactsError",
    "SlotSubject",
    "envelope_digest",
    "envelope_facts",
    "facts_digest",
    "manifest_digest",
    "subject_digest",
    "subject_facts",
]
