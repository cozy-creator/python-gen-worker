"""A release-declared slot that fails to resolve emits a TYPED declared-fault
label, not a bare ``ValueError``.

A bare ``ValueError`` carries no origin claim, so the hub can only classify it
by WORKER VERSION — and every protocol-stale worker sits below
``MinRefProvenanceWorkerVersion``, so its FATALs are DISCARDED and neither the
load-failure streak nor the spend brake ever sees the deterministic fault. The
hub side allowlists ``declaredslotresolutionerror`` and classifies it
``EvidenceDeclaredRefLoad``.

The FIXED/``selected_by`` split is the load-bearing half, not a detail: the
hub's own rule is "never add a label a payload field can participate in
producing". A ``Slot(selected_by="model")`` slot is picked by the payload, so
its resolution failure is the CALLER's, and typing it would let one bad
request feed the release's health streak.
"""

from __future__ import annotations

import pytest

from gen_worker import executor, warmup
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.api.errors import DeclaredSlotResolutionError
from gen_worker.api.slot import Slot
from gen_worker.request_context import RequestContext


class _FakeSpec:
    """The two fields ``resolved_slots_kwargs`` reads for this path.

    Neither slot has a ``default_checkpoint`` and the warm shape supplies no
    hub binding, so both fail resolution at ``resolve_slot``'s ``ref is None``
    — the same shape as an unresolvable declared ref in production.
    """

    def __init__(self) -> None:
        self.slots = {
            "pipeline": Slot(str),                        # FIXED declaration
            "model": Slot(str, selected_by="model"),      # payload picks
        }
        self.models: dict = {}
        self.defaults_type = None
        self.slot_family: dict = {}
        self.objectives = None
        self.distilled = None


def _kwargs() -> dict:
    kw = warmup.resolved_slots_kwargs(_FakeSpec(), None)
    assert set(kw["slot_errors"]) == {"pipeline", "model"}, kw
    return kw


def test_fixed_declared_slot_raises_the_typed_origin_error() -> None:
    ctx = RequestContext(request_id="r1", **_kwargs())
    with pytest.raises(DeclaredSlotResolutionError) as caught:
        ctx.slots["pipeline"]
    assert "pipeline" in str(caught.value)
    # Handlers that already catch ValueError keep working.
    assert isinstance(caught.value, ValueError)


def test_selected_by_slot_stays_an_untyped_value_error() -> None:
    ctx = RequestContext(request_id="r2", **_kwargs())
    with pytest.raises(ValueError) as caught:
        ctx.slots["model"]
    assert type(caught.value) is ValueError, (
        "a payload-picked slot must NOT carry the declared-fault label")


def test_the_hub_reads_the_class_name_as_the_fatal_label() -> None:
    """``_map_exception`` is what puts the label on the wire, and the hub
    matches on the lowercased text before the first colon."""
    ctx = RequestContext(request_id="r3", **_kwargs())
    try:
        ctx.slots["pipeline"]
    except Exception as exc:  # noqa: BLE001 — the mapper's own input
        status, message = executor._map_exception(exc)
    assert status == pb.JOB_STATUS_FATAL
    assert message.split(":", 1)[0] == "DeclaredSlotResolutionError"

    try:
        ctx.slots["model"]
    except Exception as exc:  # noqa: BLE001
        picked_status, picked_message = executor._map_exception(exc)
    assert picked_status == pb.JOB_STATUS_FATAL
    assert picked_message.split(":", 1)[0] == "ValueError"


def test_resolved_slots_carry_no_declared_errors() -> None:
    empty = warmup.resolved_slots_kwargs(type("S", (), {"slots": {}})(), None)
    assert empty["declared_slot_errors"] == ()
