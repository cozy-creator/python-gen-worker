"""Neutral dispatch orders.

The wire-reading head ``executor.handle_run_job`` projects its message into
these values at the boundary. The driver and every shared helper read ONLY
these: ``pb.RunJob`` never crosses into shared machinery, which is what makes
the head replaceable without touching the driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .serving_facts import FactsUnavailable, ServingFacts, SlotEvidence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pb import worker_scheduler_pb2 as pb


#: pgw#1333: WHO owes the serving facts on the BOOT path.
#: ``DesiredInstance.models`` is the same ``ModelBinding`` message the
#: dispatch path stamps, but tensorhub's ``hello_ack.go`` leaves
#: ``objective`` / ``distilled`` / ``distilled_status`` zero on it. Named
#: here, once, so every boot-side reader refuses with the SAME sentence and
#: an operator reading it knows which repo owes the fix.
BOOT_SENDER_OWES = (
    "the hub's boot sender (tensorhub hello_ack.go leaves "
    "DesiredInstance.models[].objective / .distilled / .distilled_status "
    "unset; the dispatch path stamps the same three fields)"
)


@dataclass(frozen=True, kw_only=True)
class SlotOrder:
    """One dispatched slot binding, as far as the head resolved it.

    ``ref`` is a canonical tensorhub-CAS ref or ``""`` (unbound — the driver
    drops an optional slot, and the head may leave a required slot for the
    code-declared default).

    ``facts`` carries NO default (pgw#1333). Every construction site must say
    whether the catalog answered (:class:`~gen_worker.serving_facts.ServingFacts`)
    or was never asked (:class:`~gen_worker.serving_facts.FactsUnavailable`),
    because the three loose defaulted scalars this replaced made those two
    states one state — and the state they collapsed to reads, at the refusal,
    as "this checkpoint is unclassified".
    """

    ref: str = ""
    inference_defaults: str = ""
    facts: SlotEvidence


def order_from_binding(
    binding: "pb.ModelBinding", *, ref: str = "", owed_by: str = "",
) -> SlotOrder:
    """THE projection of one wire ``ModelBinding`` into a neutral order.

    One function for every sender (``RunJob.models``, ``DesiredInstance.models``
    on the boot/preload paths), so a path cannot quietly read fewer fields
    than another — which is exactly how the boot path came to build
    ``SlotOrder(ref=ref)`` and drop three fields the message already carried.

    ``owed_by`` names the sender for the case where the message carries no
    serving facts AT ALL. proto3 scalars have no presence, so a wholly
    zero-valued triple cannot be distinguished from a genuinely unclassified
    checkpoint on the wire; with ``owed_by`` given, the zero triple is read as
    the wire gap it is on senders known not to stamp yet. Pass it ONLY for
    such a sender — the dispatch path stamps, so its zero triple is an answer
    and ``owed_by`` must be empty there.
    """
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
