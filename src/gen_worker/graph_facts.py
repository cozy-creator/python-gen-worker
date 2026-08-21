"""Worker-owned FACTS about a declaration and about a mint obligation — never identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

from gen_worker._vendor.torchcg import is_compiled_graph_key

EXPORT_ENVELOPE_KEY = "declared_envelope"


class GraphFactsError(ValueError):
    """A recorded fact block cannot be canonicalized."""


def _refuse_key_shaped(where: str, name: str, value: str) -> None:
    if is_compiled_graph_key(value):
        raise GraphFactsError(
            f"{where}: {name}={value!r} is a COMPILED-GRAPH KEY where a fact "
            f"digest belongs. A key is the OUTPUT of identity, never an input "
            f"to it — folding one in here would produce a digest no artifact "
            f"can restate.")


def facts_digest(facts: Mapping[str, Any]) -> str:
    """16-hex canonical digest of one recorded fact block."""
    encoded = json.dumps(
        dict(facts), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def envelope_facts(block: Mapping[str, Any]) -> Dict[str, Any]:
    """The canonical form of one DECLARED-ENVELOPE block — the single canonicalizer of the declared serving region, so no two consumers can canonicalize the same declaration differently."""
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
    """WHICH checkpoint one setup slot resolved to."""

    slot: str
    refs: Tuple[str, ...] = ()
    snapshot_digest: str = ""


def subject_facts(subjects: Iterable[SlotSubject]) -> Dict[str, Any]:
    """The canonical SUBJECT block for one arm/trace — sorted by slot, so two callers that resolved the same slots in different orders state one fact."""
    return {
        "v": 1,
        "slots": [
            [sub.slot, list(sub.refs), sub.snapshot_digest]
            for sub in sorted(tuple(subjects), key=lambda s: s.slot)
        ],
    }


def subject_digest(subjects: Iterable[SlotSubject]) -> str:
    """16-hex digest of the resolved subject, or ``""`` when the caller could state none."""
    subs = tuple(subjects)
    if not subs:
        return ""
    return facts_digest(subject_facts(subs))


def manifest_digest(specialization_hashes: Iterable[str]) -> str:
    """The coverage LABEL of one declaration's class set — 16 hex of sha256 over the newline-joined SORTED per-specialization hashes."""
    rows = [str(h) for h in specialization_hashes]
    for row in rows:
        _refuse_key_shaped("manifest digest", "specialization_hash", row)
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
