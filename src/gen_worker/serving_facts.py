"""What the CATALOG says about a resolved checkpoint — or the honest
statement that nobody said anything.

pgw#1333. The three serving facts (``objective`` / ``distilled`` /
``distilled_status``) used to travel as three loose scalars on
``dispatch.SlotOrder``, every one of them defaulted, and every consumer read
them with ``.get(name, "")``. That shape has exactly one failure mode and it
cost a paid pod: a code path that never received the facts is
INDISTINGUISHABLE from a checkpoint the catalog classified as nothing, so
``api.slot._finish_resolved`` refused an ``epsilon`` checkpoint by saying
"resolved checkpoint carries no training objective" — and the refusal was
read, correctly given its text, as a catalog defect. It was a wire gap.

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


def facts_or_degrade(
    evidence: SlotEvidence, *, slot: str, what: str,
) -> "tuple[ServingFacts, str]":
    """The stamped facts, plus the sentence to CONFESS if nobody stamped them.

    The ONE place the union is collapsed. pgw#1333 made this raise, and the
    attribution it built was right — the message blames the *sender* that
    skipped the stamp, never the checkpoint's catalog row. What was wrong was
    the refusal: a wire gap on our own side is not grounds to decline a paid
    customer request (pgw#1339 / th#2099). So the gap now degrades — an empty
    :class:`ServingFacts` and a non-empty reason — and the caller serves while
    saying exactly whose stamp is missing.

    An empty second element means there is nothing to confess.
    """
    if isinstance(evidence, ServingFacts):
        return evidence, ""
    return ServingFacts(), (
        f"slot {slot!r}: no serving facts were resolved for this checkpoint "
        f"({what}) — {evidence.owed_by} did not stamp objective/distilled/"
        f"distilled_status, so there is no evidence to check the invoked "
        f"function's declared serving contract against. This is a wire gap, "
        f"NOT a claim about the checkpoint's catalog row; the request still "
        f"serves and this slot's declared contract goes UNCHECKED.")


__all__ = [
    "DISTILLED_STATUSES",
    "OBJECTIVES",
    "FactsUnavailable",
    "ServingFacts",
    "SlotEvidence",
    "facts_or_degrade",
]
