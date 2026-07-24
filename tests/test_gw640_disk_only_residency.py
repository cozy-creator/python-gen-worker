"""gw#640: a Hub()-bound function whose desired residency is disk-only
(disk=1 hot=0) must converge and keep the worker alive.

Live shape (th#1085 cold-boot gate, RunPod CPU pod): the hub delivers the
model on disk with NO hot instance, the gw#591 boot-setup watch completes
setup — and the worker process exited 1-2s later, silently, ~18x per run.
"""

from __future__ import annotations

import msgspec

from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_model_event, is_ready, is_result_for
from harness.toy_endpoints import EchoIn, EchoOut

_MODEL_REF = "harness/residency-tiny"


def test_disk_only_residency_sets_up_and_worker_survives(tmp_path) -> None:
    blobs = BlobHost(tmp_path)
    try:
        payload = b"tiny-weights"
        snapshot = blobs.one_file_snapshot("snap-640", "blob", payload)
        with hub_double() as (scheduler, harness):
            conn = scheduler.wait_connection(0)
            ready = conn.wait_for(is_ready)
            assert "model-echo" in ready.state_delta.loading_functions

            # disk=1, hot=0 — the gw#640 shape.
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[_MODEL_REF],
                    snapshots={_MODEL_REF: snapshot},
                ),
            ))
            conn.wait_for(is_model_event(_MODEL_REF, pb.MODEL_STATE_ON_DISK))

            # gw#591 boot-setup watch must advertise the function...
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "model-echo" in m.state_delta.available_functions
            )
            # ...and the worker must still be alive to serve it.
            conn.send(run_job=pb.RunJob(
                request_id="r640", attempt=1, function_name="model-echo",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x"))))
            res = conn.wait_for(is_result_for("r640")).job_result
            assert res.status == pb.JOB_STATUS_OK
            assert msgspec.msgpack.decode(res.inline, type=EchoOut).response == payload.decode()
            assert harness.exit_code is None, (
                f"worker process exited (code={harness.exit_code}) on a disk-only "
                "desired residency"
            )
    finally:
        blobs.shutdown()
