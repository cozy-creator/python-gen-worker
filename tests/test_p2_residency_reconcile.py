"""P2 (th#960/pgw#609 design table): HelloAck full-replace DesiredResidency
-> download -> verify -> ModelEvent chain, over a real hub-double + real
blake3 blob host. Events are fenced by snapshot digest + generation (a late
event after a tag-move never misattributes to the new identity); a mutable
tag can move to new bytes under the SAME wire ref.

Not covered here (documented deviation, one line): "adopt-to-self no-op
re-arm" (gw#604/gw#607) is compile-cell/fleet_cells adoption machinery, not
model residency — its worker-side tests live in test_executor_adopt.py,
currently under a different lane's active dirty WIP per the th#960 tracker
note ("foreign WIP ... compile_cache/executor/fleet_cells ... left alone").
Duplicating that surface here risks colliding with in-flight work; the
th#960 checkpoint records this as an open follow-up for whoever owns that
lane next.
"""

from __future__ import annotations

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost, CorruptingBlobHost
from harness.hub_double import hub_double, is_model_event, is_ready, is_result_for
from harness.toy_endpoints import EchoIn, EchoOut

_MODEL_REF = "harness/residency-tiny"


def _decode(data: bytes) -> EchoOut:
    return msgspec.msgpack.decode(data, type=EchoOut)


def test_desired_residency_downloads_warms_and_serves(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        payload = b"tiny-weights"
        snapshot = blobs.one_file_snapshot("snap-1", "blob", payload)
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            ready = conn.wait_for(is_ready)
            # Gated until its model loads: present in loading, not available.
            assert "model-echo" not in ready.state_delta.available_functions
            assert "model-echo" in ready.state_delta.loading_functions

            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[_MODEL_REF],
                    hot=[pb.DesiredInstance(
                        function_name="model-echo",
                        models=[pb.ModelBinding(slot="model", ref=_MODEL_REF)],
                    )],
                    snapshots={_MODEL_REF: snapshot},
                ),
            ))
            downloading = conn.wait_for(
                is_model_event(_MODEL_REF, pb.MODEL_STATE_DOWNLOADING)
            ).model_event
            on_disk = conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK)).model_event
            in_ram = conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_IN_RAM)).model_event
            for event in (downloading, on_disk, in_ram):
                assert event.snapshot_digest == "snap-1"
                assert event.residency_generation == 1
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "model-echo" in m.state_delta.available_functions
                and m.state_delta.observed_residency_generation == 1
            )
            conn.send(run_job=pb.RunJob(
                request_id="r1", attempt=1, function_name="model-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
            res = conn.wait_for(is_result_for("r1")).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert _decode(res.inline).response == payload.decode()
    finally:
        blobs.shutdown()


def test_mutable_tag_move_fences_events_by_digest_and_generation(tmp_path) -> None:
    """A moved tag keeps the SAME wire ref but new bytes/digest; every event
    (and the resumed declarative baseline) must carry the NEW generation —
    a late event never silently downgrades to a stale identity."""
    blobs = BlobHost(tmp_path)
    try:
        payload_a = b"tiny-weights-a"
        payload_b = b"moved-tag-weights-b"
        snap_a = blobs.one_file_snapshot("snap-a", "blob-a", payload_a)
        snap_b = blobs.one_file_snapshot("snap-b", "blob-b", payload_b)

        with hub_double() as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1, disk_refs=[_MODEL_REF],
                    hot=[pb.DesiredInstance(
                        function_name="model-echo",
                        models=[pb.ModelBinding(slot="model", ref=_MODEL_REF)],
                    )],
                    snapshots={_MODEL_REF: snap_a},
                ),
            ))
            conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_IN_RAM))

            # Disk-only move to B (no hot instance): tensorhub prepositions
            # disk residency ahead of any dispatch.
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=2, disk_refs=[_MODEL_REF], snapshots={_MODEL_REF: snap_b},
                ),
            ))
            moved_disk = conn.wait_for(
                lambda m: is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK)(m)
                and m.model_event.snapshot_digest == "snap-b"
            ).model_event
            assert moved_disk.residency_generation == 2

            # A late RunJob may legitimately still ask for A while desired
            # residency has moved to B/gen2; afterward the resumed
            # declarative loop restores B WITH generation 2, not gen0.
            resumed_b = lambda m: (
                is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK)(m)
                and m.model_event.snapshot_digest == "snap-b"
                and m.model_event.residency_generation == 2
            )
            b_events_before = conn.count(resumed_b)
            conn.send(run_job=pb.RunJob(
                request_id="r-priority-a", attempt=1, function_name="model-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
                snapshots={_MODEL_REF: snap_a}))
            conn.wait_for(lambda m: (
                is_model_event(_MODEL_REF, pb.MODEL_STATE_IN_RAM)(m)
                and m.model_event.snapshot_digest == "snap-a"
                and m.model_event.residency_generation == 0
            ))
            priority_a = conn.wait_for(is_result_for("r-priority-a")).job_result
            assert priority_a.status == pb.JOB_STATUS_OK
            assert _decode(priority_a.inline).response == payload_a.decode()
            conn.wait_for_count(resumed_b, b_events_before + 1)
            assert harness.worker.executor.store.resident_identity(_MODEL_REF) == ("snap-b", 2)
    finally:
        blobs.shutdown()


def test_corrupt_blob_is_never_trusted_silently(tmp_path) -> None:
    """A blob that fails its blake3 verify on download must not be served —
    the model either quarantines/retries or the function stays unavailable;
    it never silently loads mismatched bytes. Real corruption via a real
    HTTP server serving wrong bytes for the declared digest (no mocking of
    the download/verify path)."""
    blobs = CorruptingBlobHost(tmp_path, corrupt="blob")
    try:
        real_payload = b"tiny-weights-that-will-be-corrupted"
        snapshot = blobs.one_file_snapshot("snap-corrupt", "blob", real_payload)
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1, disk_refs=[_MODEL_REF],
                    hot=[pb.DesiredInstance(
                        function_name="model-echo",
                        models=[pb.ModelBinding(slot="model", ref=_MODEL_REF)],
                    )],
                    snapshots={_MODEL_REF: snapshot},
                ),
            ))
            # The verify failure must surface as a typed FAILED ModelEvent —
            # never a silent IN_RAM with mismatched bytes.
            failed_or_ram = conn.wait_for(lambda m: (
                m.WhichOneof("msg") == "model_event" and m.model_event.ref == _MODEL_REF
                and m.model_event.state in (pb.MODEL_STATE_FAILED, pb.MODEL_STATE_IN_RAM)
            ))
            assert failed_or_ram.model_event.state == pb.MODEL_STATE_FAILED, (
                "corrupted bytes must never reach MODEL_STATE_IN_RAM"
            )
    finally:
        blobs.shutdown()
