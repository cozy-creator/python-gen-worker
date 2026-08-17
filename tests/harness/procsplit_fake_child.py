"""A compute-child PEER (not a mock of our code) for pgw#763 stage-2 races.

The real child in ``procsplit_child_main.py`` is the primary subject of this
suite. This process exists only for timings the real child cannot produce
deterministically — chiefly "write a JobResult and die in the same breath",
where whether the parent has read the frame yet is a kernel-buffer race. It
speaks the same frame protocol over the same socket, exactly as the hub-double
speaks real gRPC.

Behaviour is chosen by PGW763_FAKE_MODE:
  result_then_die : accept + OK result for each RunJob, then SIGKILL itself
                    immediately (frame and death adjacent).
  result_then_exit: accept + OK result, then exit 0 (deliberate exit with a
                    result the parent still owes the hub).
  result_then_recycle
                  : accept + OK result, then exit EXIT_JOB_RECYCLE (pgw#1324's
                    run-once lifecycle). The parent must RESPAWN — this is
                    neither a crash nor a shutdown, and rc 0 or a plain
                    non-zero would give the wrong one.
  ignore_sigterm  : serve nothing, ignore SIGTERM forever (TimeoutStopSec).
  spontaneous_result_then_exit
                  : write one JobResult on connect and exit 0 immediately, so
                    the parent owns a durable result it can never ship (the
                    stream needs a Hello this child never answers).
  forge_hello     : pgw#763 delta 1+2. Answers T_HELLO_REQ with a Hello that
                    lies about everything a compute child must not be trusted
                    to assert: another worker's id and release, a fabricated
                    gpu_name/gpu_sm/VRAM, and a HostCanary of invented floats.
                    Tenant code reaches the real child's Hello builder, so this
                    is a faithful stand-in for what it could send — and each of
                    these values is a FLEET-WIDE verdict key (th#1310), not a
                    per-pod cosmetic.
  forge_metrics   : pgw#763 delta 3. Serves each RunJob instantly and reports
                    JobMetrics claiming hours of runtime, a fabricated
                    concurrency and a fabricated RSS — the code being billed
                    writing its own bill (th#1309).
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.procsplit import EXIT_JOB_RECYCLE, frames  # noqa: E402

MODE = os.environ.get("PGW763_FAKE_MODE", "result_then_die")
WORKER_ID = os.environ.get("PGW763_WORKER_ID", "split-fake-child")


FORGED_WORKER_ID = "victim-worker-0000"
FORGED_RELEASE_ID = "victim-release-0000"
FORGED_GPU_NAME = "NVIDIA H200-141GB"
FORGED_VRAM_BYTES = 141 * (1 << 30)
FORGED_MEMCPY_GBPS = 999.0
# Three hours of "runtime" for a job the parent watched take milliseconds.
FORGED_RUNTIME_MS = 3 * 60 * 60 * 1000
FORGED_CONCURRENCY = 97
FORGED_RSS_BYTES = 999 * (1 << 30)


def _hello() -> pb.Hello:
    if MODE == "forge_hello":
        return pb.Hello(
            worker_id=FORGED_WORKER_ID,
            release_id=FORGED_RELEASE_ID,
            protocol_version=pb.PROTOCOL_VERSION_CURRENT,
            resources=pb.WorkerResources(
                gpu_count=8,
                vram_total_bytes=FORGED_VRAM_BYTES,
                gpu_name=FORGED_GPU_NAME,
                gpu_sm="90",
                torch_version="9.9.9",
                gen_worker_version="0.0.0-forged",
                image_digest="sha256:forged",
                instance_id="pod-belonging-to-someone-else",
                host_canary=pb.HostCanary(
                    memcpy_gbps=FORGED_MEMCPY_GBPS,
                    d2h_gbps=FORGED_MEMCPY_GBPS,
                    cpu_single_mbps=FORGED_MEMCPY_GBPS,
                    cpu_multi_mbps=FORGED_MEMCPY_GBPS,
                    pinned_alloc_ok=True,
                    vcpus=256,
                    ram_total_gb=2048.0,
                    interconnect="nvlink",
                    peer_gbps=FORGED_MEMCPY_GBPS,
                    peer_access=True,
                    topo_link="NV18",
                ),
            ),
        )
    return pb.Hello(
        worker_id=WORKER_ID,
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
    )


async def main() -> int:
    if MODE == "ignore_sigterm":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    reader, writer = await asyncio.open_unix_connection(
        os.environ["GEN_WORKER_CHILD_SOCKET"]
    )
    fw = frames.FrameWriter(writer)
    if MODE == "spontaneous_result_then_exit":
        await fw.frame(frames.T_WORKER_MSG, pb.WorkerMessage(
            job_result=pb.JobResult(
                request_id="r-orphan", attempt=1,
                status=pb.JOB_STATUS_OK, inline=b"never-shipped",
            ),
        ).SerializeToString())
        os._exit(0)
    while True:
        try:
            ftype, payload = await frames.read_frame(reader)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            return 0
        if ftype == frames.T_HELLO_REQ:
            await fw.frame(frames.T_HELLO, _hello().SerializeToString())
            continue
        if MODE == "forge_hello":
            continue
        if ftype != frames.T_SCHED:
            continue
        msg = pb.SchedulerMessage.FromString(payload)
        if msg.WhichOneof("msg") != "run_job":
            continue
        run = msg.run_job
        if MODE == "ignore_sigterm":
            continue
        await fw.frame(frames.T_WORKER_MSG, pb.WorkerMessage(
            job_accepted=pb.JobAccepted(request_id=run.request_id, attempt=run.attempt),
        ).SerializeToString())
        result = pb.JobResult(
            request_id=run.request_id,
            attempt=run.attempt,
            status=pb.JOB_STATUS_OK,
            inline=b"fake-ok",
        )
        if MODE == "forge_metrics":
            # The code being billed, writing its own bill (th#1309).
            result.metrics.CopyFrom(pb.JobMetrics(
                runtime_ms=FORGED_RUNTIME_MS,
                queue_ms=FORGED_RUNTIME_MS,
                slot_held_ms=FORGED_RUNTIME_MS,
                finalize_wall_ms=FORGED_RUNTIME_MS,
                concurrency_at_start=FORGED_CONCURRENCY,
                rss_at_end_bytes=FORGED_RSS_BYTES,
                output_count=3,
                output_media_duration_s=0.0,
                lane="fake-lane",
            ))
        await fw.frame(frames.T_WORKER_MSG, pb.WorkerMessage(
            job_result=result,
        ).SerializeToString())
        if MODE == "result_then_die":
            os.kill(os.getpid(), signal.SIGKILL)
        if MODE == "result_then_exit":
            os._exit(0)
        if MODE == "result_then_recycle":
            os._exit(EXIT_JOB_RECYCLE)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
