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
from harness.hub_double import Conn, WorkerHarness, hub_double, is_model_event, is_ready
from harness.progress_wait import await_count

# pgw#795 (the v0.78.0 release blocker): this ref must NOT be one a toy
# endpoint declares. It used to be `harness/residency-tiny`, which
# harness/toy_endpoints.py binds — so ~2 s after boot the eager first-boot
# promoted the ref to RAM, and from that moment every re-sent plan
# re-announced the held identity as IN_RAM instead of ON_DISK. The test was
# therefore racing a worker-side timer: it needed its three ON_DISK
# re-reports inside a ~2 s window, which a loaded CI runner does not grant.
# That, not the 15 s deadline it died on, is why v0.78.0 failed twice. With
# an UNDECLARED ref nothing promotes it, the tier stays DISK, and the
# property holds at any spacing (verified at 0.5 s and 5 s gaps).
_MODEL_REF = "harness/republish-tiny"


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


def _wait_on_disk_count(conn: Conn, harness: WorkerHarness, want: int) -> float:
    """Wait for ``want`` ON_DISK re-reports — on PROGRESS, never on a clock.

    pgw#795: this wait used to be ``deadline = time.monotonic() + 15.0``, and
    on 2026-07-30 it failed the v0.78.0 publish job twice, after seeing 2 of 3
    re-reports. The deadline was the messenger (see ``_MODEL_REF`` for the
    disease), and a wall clock is a bad messenger: it reports "slow" for
    everything, including "this can never happen now". What ends the wait now
    is evidence, not duration:

    * the worker ended the stream, or its thread exited — no further re-report
      is possible, so fail at once;
    * the re-report count has not moved for a staleness window this run
      measured for itself. Only that count counts as progress: the worker being
      alive, or busy, or chattering about other things is not evidence that the
      answer is coming, and a window that resets on those never closes.
    """
    started = time.monotonic()
    await_count(
        lambda: _count_on_disk(conn),
        want,
        what=f"ON_DISK re-reports of {_MODEL_REF}",
        cadence=conn.cadence,
        gone=lambda: (
            "the worker ended the stream" if conn.client_done.is_set()
            else None if harness.alive
            else "the worker thread exited"
        ),
    )
    return time.monotonic() - started


def _settle(answered_in_s: float) -> None:
    """Let a runaway re-announce loop, if there is one, show itself.

    A fixed sleep here is sound where a fixed deadline was not: its expiry
    makes the test PASS, so a slow runner weakens the check instead of failing
    the release. It still scales with the machine — the quiet period is a
    multiple of how long the re-announce it is checking actually took on this
    machine.
    """
    time.sleep(max(0.5, 10 * answered_in_s))


def test_reissued_plan_republishes_held_identity(tmp_path) -> None:
    """Each applied plan re-send yields exactly ONE fresh ON_DISK re-report
    carrying the exact (ref, digest): the idempotent resync a v2 hub can
    always absorb and that heals a lost success observation. Within an
    epoch the dedupe holds — re-announce is per applied ack, never spam."""
    blobs = BlobHost(tmp_path)
    try:
        snapshot = blobs.one_file_snapshot("snap-1", "blob", b"tiny-weights")
        with hub_double() as (scheduler, harness):
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
            answered_in = _wait_on_disk_count(conn, harness, baseline + 1)
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
            _settle(answered_in)
            assert _count_on_disk(conn) == baseline + 1

            # A third re-send opens a third epoch: one more re-report.
            conn.send(hello_ack=_disk_only_ack(snapshot, generation=1))
            answered_in = _wait_on_disk_count(conn, harness, baseline + 2)
            _settle(answered_in)
            assert _count_on_disk(conn) == baseline + 2
    finally:
        blobs.shutdown()
