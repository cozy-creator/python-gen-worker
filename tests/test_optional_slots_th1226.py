"""th#1226: an endpoint may declare a slot the deploy is free to leave unbound.

Which lanes a release serves is deploy config — but before
this, every declared Slot was force-resolved at dispatch (``RetryableError``)
and a partial Hot ``DesiredInstance`` was refused outright, so a two-lane
endpoint could only ever be deployed with BOTH lanes bound. That is what made
`qwen-image` unservable in bf16: two 20B transformers plus a shared encoder is
~123 GB resident, and it could not be split.

A slot is optional exactly when its ``setup()`` parameter carries a default.
These run the real Worker/Lifecycle/Executor over the hub-double gRPC
boundary — the same path a live single-lane deploy takes.
"""

from __future__ import annotations

import msgspec

from gen_worker.api.binding import wire_ref
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready, is_result_for
from harness.toy_endpoints import OPTIONAL_EDIT, OPTIONAL_T2I, EchoIn, EchoOut

_T2I_BYTES = b"t2i-lane-bytes"
_EDIT_BYTES = b"edit-lane-bytes"


def _payload(**kw: object) -> bytes:
    return msgspec.msgpack.encode(EchoIn(**kw))  # type: ignore[arg-type]


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


def test_optional_slot_is_derived_from_the_setup_default() -> None:
    """The signature is the single declaration — no second `optional=` flag
    to drift out of sync with it."""
    from harness.toy_endpoints import OptionalExecutionLaneEndpoint

    slots = getattr(OptionalExecutionLaneEndpoint, "__gen_worker_endpoint__").slots
    assert slots["t2i"].optional is False
    assert slots["edit"].optional is True


def test_defaulted_param_whose_annotation_rejects_none_is_a_decoration_error() -> None:
    """``edit: Pipe = None`` type-lies; injection reads the annotation."""
    import pytest

    from gen_worker import Hub, RequestContext, Slot, endpoint

    with pytest.raises(ValueError, match="does not admit None"):

        @endpoint(models={"pipeline": Slot(str, default_checkpoint=Hub("h/p", release="prod"))})
        class _Bad:
            def setup(self, pipeline: str = None) -> None:  # type: ignore[assignment]
                self.p = pipeline

            def go(self, ctx: RequestContext, data: EchoIn) -> EchoOut:
                return EchoOut(response="x")


def test_single_execution_lane_deploy_binds_only_the_required_slot(tmp_path) -> None:
    """A Hot DesiredInstance naming ONLY `t2i` is accepted, setup runs with
    the unbound slot's own default, and the bound lane serves."""
    blobs = BlobHost(tmp_path)
    try:
        t2i_ref = wire_ref(OPTIONAL_T2I)
        t2i_snap = blobs.one_file_snapshot("snap-t2i", "t2i", _T2I_BYTES)

        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[t2i_ref],
                    snapshots={t2i_ref: t2i_snap},
                    hot=[pb.DesiredInstance(
                        function_name="optional-t2i",
                        models=[pb.ModelBinding(slot="t2i", ref=t2i_ref)],
                    )],
                ),
            ))
            # Pre-fix this never arrived: the partial bind raised
            # ValidationError and every ref went MODEL_STATE_FAILED.
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "optional-t2i" in m.state_delta.available_functions
                and m.state_delta.observed_residency_generation == 1
            )
            conn.send(run_job=pb.RunJob(
                request_id="r-t2i", attempt=1, function_name="optional-t2i",
                input_payload=_payload(),
                models=[pb.ModelBinding(slot="t2i", ref=t2i_ref)],
                snapshots={t2i_ref: t2i_snap},
            ))
            res = conn.wait_for(is_result_for("r-t2i")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            # `edit_bound=False` is the proof setup() took its own default
            # rather than being handed a path for a slot nobody bound.
            assert _decode(res.inline).response == (
                f"t2i:{_T2I_BYTES.decode()}:edit_bound=False"
            )
    finally:
        blobs.shutdown()


def test_unbound_execution_lane_refuses_typed_and_never_crashes_setup(tmp_path) -> None:
    """Calling the lane this deploy did not bind is a client-visible INVALID
    refusal that names the slot — not a retry storm, not a worker crash."""
    blobs = BlobHost(tmp_path)
    try:
        t2i_ref = wire_ref(OPTIONAL_T2I)
        t2i_snap = blobs.one_file_snapshot("snap-t2i", "t2i", _T2I_BYTES)

        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[t2i_ref],
                    snapshots={t2i_ref: t2i_snap},
                ),
            ))
            conn.send(run_job=pb.RunJob(
                request_id="r-edit", attempt=1, function_name="optional-edit",
                input_payload=_payload(),
                models=[pb.ModelBinding(slot="t2i", ref=t2i_ref)],
                snapshots={t2i_ref: t2i_snap},
            ))
            res = conn.wait_for(is_result_for("r-edit")).job_result
            assert res.status == pb.JOB_STATUS_INVALID, res.safe_message
            assert "edit" in res.safe_message
    finally:
        blobs.shutdown()


def test_both_execution_lanes_still_bind_unchanged(tmp_path) -> None:
    """Optionality must not change the dual-lane deploy — the shape every
    existing release uses."""
    blobs = BlobHost(tmp_path)
    try:
        t2i_ref, edit_ref = wire_ref(OPTIONAL_T2I), wire_ref(OPTIONAL_EDIT)
        t2i_snap = blobs.one_file_snapshot("snap-t2i", "t2i", _T2I_BYTES)
        edit_snap = blobs.one_file_snapshot("snap-edit", "edit", _EDIT_BYTES)

        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[t2i_ref, edit_ref],
                    snapshots={t2i_ref: t2i_snap, edit_ref: edit_snap},
                    hot=[pb.DesiredInstance(
                        function_name="optional-edit",
                        models=[
                            pb.ModelBinding(slot="t2i", ref=t2i_ref),
                            pb.ModelBinding(slot="edit", ref=edit_ref),
                        ],
                    )],
                ),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "optional-edit" in m.state_delta.available_functions
                and m.state_delta.observed_residency_generation == 1
            )
            conn.send(run_job=pb.RunJob(
                request_id="r-both", attempt=1, function_name="optional-edit",
                input_payload=_payload(),
                models=[
                    pb.ModelBinding(slot="t2i", ref=t2i_ref),
                    pb.ModelBinding(slot="edit", ref=edit_ref),
                ],
                snapshots={t2i_ref: t2i_snap, edit_ref: edit_snap},
            ))
            res = conn.wait_for(is_result_for("r-both")).job_result
            assert res.status == pb.JOB_STATUS_OK, res.safe_message
            assert _decode(res.inline).response == f"edit:{_EDIT_BYTES.decode()}"
    finally:
        blobs.shutdown()


def test_manifest_marks_the_optional_slot() -> None:
    """The hub must be able to tell a partial bind is legal — absent field
    still means required, so existing manifests are unchanged."""
    from harness.toy_endpoints import OptionalExecutionLaneEndpoint

    from gen_worker.discovery.discover import _slot_to_manifest

    slots = getattr(OptionalExecutionLaneEndpoint, "__gen_worker_endpoint__").slots
    assert "optional" not in _slot_to_manifest("t2i", slots["t2i"], family="")
    assert _slot_to_manifest("edit", slots["edit"], family="")["optional"] is True
