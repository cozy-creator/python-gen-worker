"""Neutral dispatch orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .serving_facts import FactsUnavailable, ServingFacts, SlotEvidence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pb import worker_scheduler_pb2 as pb


BOOT_SENDER_OWES = (
    "the hub's boot sender (tensorhub hello_ack.go leaves "
    "DesiredInstance.models[].objective / .distilled / .distilled_status "
    "unset; the dispatch path stamps the same three fields)"
)


@dataclass(frozen=True, kw_only=True)
class SlotOrder:
    """One dispatched slot binding, as far as the head resolved it."""

    ref: str = ""
    inference_defaults: str = ""
    facts: SlotEvidence


def order_from_binding(
    binding: "pb.ModelBinding", *, ref: str = "", owed_by: str = "",
) -> SlotOrder:
    """THE projection of one wire ``ModelBinding`` into a neutral order."""
    objective = str(getattr(binding, "objective", "") or "")
    distilled = bool(getattr(binding, "distilled", False))
    status = str(getattr(binding, "distilled_status", "") or "")
    facts: SlotEvidence
    if owed_by and not objective and not distilled and not status:
        facts = FactsUnavailable(owed_by=owed_by)
    else:
        facts = ServingFacts(
            objective=objective, distilled=distilled, distilled_status=status)
    return SlotOrder(
        ref=str(ref or getattr(binding, "ref", "") or "").strip(),
        inference_defaults=str(getattr(binding, "inference_defaults", "") or ""),
        facts=facts,
    )


@dataclass(frozen=True)
class AdapterOrder:
    """One LoRA overlay riding a slot's base model."""

    ref: str
    weight: float = 1.0
    inference_defaults: str = ""


__all__ = [
    "BOOT_SENDER_OWES", "AdapterOrder", "SlotOrder", "order_from_binding",
]
