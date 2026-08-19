"""What the CATALOG says about a resolved checkpoint — or the honest
statement that nobody said anything.

pgw#1333. The three serving facts (``objective`` / ``distilled`` /
``distilled_status``) travel as ONE struct, never as loose defaulted scalars
read with ``.get(name, "")``. That shape has one failure mode and it cost a
paid pod: a code path that never received the facts is INDISTINGUISHABLE from
a checkpoint the catalog classified as nothing, so a wire gap is reported as a
catalog defect.

So the facts get a TYPE, and the gap gets a type of its own:

* :class:`ServingFacts` — the catalog was asked and answered. An empty
  ``objective`` here is a real answer ("nothing measured this axis").
* :class:`FactsUnavailable` — nobody asked. ``owed_by`` names WHO owes the
  stamp, in the vocabulary of the wire field or the code path that dropped
  it, so the complaint points at the gap instead of at the checkpoint.

**Neither one is a refusal (pgw#1339 / th#2099).** Both are ABSENCES, and
absence of evidence is the normal input to a degraded run — it is confessed
through the `serve_degrade` seam and the request serves. Only a checkpoint
that positively CONTRADICTS a declared contract still refuses. The
distinction the two members buy is therefore in the SENTENCE, not in the
verdict: an operator reading "the catalog classified nothing" goes and
classifies the checkpoint, and one reading "hello_ack.go did not stamp" goes
and fixes the hub. Both used to read as "your checkpoint is broken", and the
one that shipped fatal took two measured production endpoints down.

The union is not optional and carries no default anywhere it is stored: a
slot resolution that cannot say which of the two it is cannot be constructed.
That is the whole design — an empty string is never again allowed to stand in
for "we do not know".
"""

from __future__ import annotations

from typing import Union

import msgspec

#: Every training objective the platform classifies (mirrors tensorhub's
#: ``internal/modelfamily``). ``""`` is deliberately NOT a member — an
#: unclassified checkpoint is ``ServingFacts(objective="")``, a stamped
#: statement, not a missing one.
OBJECTIVES = ("epsilon", "v_prediction", "flow")

#: ``distilled``'s evidence axis. ``""`` IS a member: the hub omits the key
#: whenever the stored column is empty, so "nothing measured the axis" is a
#: live stamped value distinct from an evidenced ``classified``.
DISTILLED_STATUSES = ("", "classified", "unclassified", "inconclusive")


class ServingFacts(msgspec.Struct, frozen=True, kw_only=True, tag="stamped"):
    """The resolved checkpoint's stamped facts, as the catalog answered.

    Validated on construction rather than at the gate that reads it: a
    typo'd objective reaching ``_finish_resolved`` there produces "not in the
    declared objectives", which reads as a compatibility refusal and is
    really a decode bug.
    """

    objective: str = ""
    distilled: bool = False
    distilled_status: str = ""

    def __post_init__(self) -> None:
        if self.objective and self.objective not in OBJECTIVES:
            raise ValueError(
                f"unknown training objective {self.objective!r} "
                f"(valid: {OBJECTIVES}, or '' for unclassified)")
        if self.distilled_status not in DISTILLED_STATUSES:
            raise ValueError(
                f"unknown distilled_status {self.distilled_status!r} "
                f"(valid: {DISTILLED_STATUSES})")


class FactsUnavailable(
    msgspec.Struct, frozen=True, kw_only=True, tag="unstamped",
):
    """No serving facts were ever resolved for this slot.

    NOT "the checkpoint has none" — that is ``ServingFacts(objective="")``.
    This says the resolution never happened, and ``owed_by`` names the sender
    or code path that would have supplied it, so a refusal blames the gap by
    name and an operator knows which side to fix.
    """

    owed_by: str

    def __post_init__(self) -> None:
        if not str(self.owed_by).strip():
            raise ValueError(
                "FactsUnavailable must name who owes the stamp — an anonymous "
                "gap is the empty string this type exists to abolish")


#: One slot's serving-fact evidence: answered, or explicitly not asked.
SlotEvidence = Union[ServingFacts, FactsUnavailable]


# pgw#1425: `facts_or_degrade` is DELETED, not baselined. It collapsed the
# `SlotEvidence` union, whose only producer is `dispatch.order_from_binding` —
# which reads `objective`/`distilled`/`distilled_status` off a CATALOG-stamped
# `ModelBinding`. pgw#1373 deleted the catalog/declaration architecture, so
# there is no stamp to read and no declared serving contract to check one
# against: producer and consumer died in one ruling. `SlotEvidence` itself
# survives for `child_contract`; the rest of the chain is pgw#1425's
# remaining-69 triage.


__all__ = [
    "DISTILLED_STATUSES",
    "OBJECTIVES",
    "FactsUnavailable",
    "ServingFacts",
    "SlotEvidence",
]
