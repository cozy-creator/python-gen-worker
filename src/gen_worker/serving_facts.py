"""What the CATALOG says about a resolved checkpoint — or the honest statement that nobody said anything."""

from __future__ import annotations

from typing import Union

import msgspec

OBJECTIVES = ("epsilon", "v_prediction", "flow")

DISTILLED_STATUSES = ("", "classified", "unclassified", "inconclusive")


class ServingFacts(msgspec.Struct, frozen=True, kw_only=True, tag="stamped"):
    """The resolved checkpoint's stamped facts, as the catalog answered."""

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
    """No serving facts were ever resolved for this slot."""

    owed_by: str

    def __post_init__(self) -> None:
        if not str(self.owed_by).strip():
            raise ValueError(
                "FactsUnavailable must name who owes the stamp — an anonymous "
                "gap is the empty string this type exists to abolish")


SlotEvidence = Union[ServingFacts, FactsUnavailable]


__all__ = [
    "DISTILLED_STATUSES",
    "OBJECTIVES",
    "FactsUnavailable",
    "ServingFacts",
    "SlotEvidence",
]
