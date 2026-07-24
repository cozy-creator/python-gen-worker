"""pgw#628 (th#1070 residency protocol v2, worker half): success observations
are (ref, digest, state) — content-addressed, idempotent, and safe to emit
twice. A re-received desired plan (hub redrive, overdue resend, reconnect) is
the hub asking for a resync, so the reconcile pass re-announces verified
cached identities once per applied-HelloAck epoch even when nothing changed.
Within one epoch the identity dedupe still holds (no event spam). The gw#614
no-cancel-on-same-set behavior is untouched — under v2 it is simply correct
instead of a trap.
"""

from __future__ import annotations

import time

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import Conn, hub_double, is_model_event, is_ready

_MODEL_REF = "harness/residency-tiny"


def _disk_only_ack(snapshot: pb.Snapshot, generation: int) -> pb.HelloAck:
    return pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        desired_residency=pb.DesiredResidency(
            generation=generation,
            disk_refs=[_MODEL_REF],
            snapshots={_MODEL_REF: snapshot},
        ),
    )


def _count_on_disk(conn: Conn) -> int:
    with conn._recv_cond:
        return sum(
            1
            for m in conn.received
            if m.WhichOneof("msg") == "model_event"
            and m.model_event.ref == _MODEL_REF
            and m.model_event.state == pb.MODEL_STATE_ON_DISK
        )


def _wait_on_disk_count(conn: Conn, want: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while _count_on_disk(conn) < want:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"expected {want} ON_DISK re-reports, saw {_count_on_disk(conn)}"
            )
        time.sleep(0.05)


def test_reissued_plan_republishes_held_identity(tmp_path) -> None:
    """Each applied plan re-send yields exactly ONE fresh ON_DISK re-report
    carrying the exact (ref, digest): the idempotent resync a v2 hub can
    always absorb and that heals a lost success observation. Within an
    epoch the dedupe holds — re-announce is per applied ack, never spam."""
    blobs = BlobHost(tmp_path)
    try:
        snapshot = blobs.one_file_snapshot("snap-1", "blob", b"tiny-weights")
        with hub_double() as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)

            conn.send(hello_ack=_disk_only_ack(snapshot, generation=1))
            first = conn.wait_for(
                is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK)
            ).model_event
            assert first.snapshot_digest == "snap-1"
            assert first.residency_generation == 1

            # The hub re-sends the SAME plan (redrive / overdue resend): the
            # worker must re-announce the held bytes, not stay silent behind
            # its identity dedupe.
            baseline = _count_on_disk(conn)
            conn.send(hello_ack=_disk_only_ack(snapshot, generation=1))
            _wait_on_disk_count(conn, baseline + 1)
            events = [
                m.model_event
                for m in conn.received
                if m.WhichOneof("msg") == "model_event"
                and m.model_event.ref == _MODEL_REF
                and m.model_event.state == pb.MODEL_STATE_ON_DISK
            ]
            assert events[-1].snapshot_digest == "snap-1"
            assert events[-1].residency_generation == 1

            # And exactly one per applied plan: no runaway re-announce loop.
            time.sleep(0.5)
            assert _count_on_disk(conn) == baseline + 1

            # A third re-send opens a third epoch: one more re-report.
            conn.send(hello_ack=_disk_only_ack(snapshot, generation=1))
            _wait_on_disk_count(conn, baseline + 2)
            time.sleep(0.5)
            assert _count_on_disk(conn) == baseline + 2
    finally:
        blobs.shutdown()
