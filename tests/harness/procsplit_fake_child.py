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
  ignore_sigterm  : serve nothing, ignore SIGTERM forever (TimeoutStopSec).
  spontaneous_result_then_exit
                  : write one JobResult on connect and exit 0 immediately, so
                    the parent owns a durable result it can never ship (the
                    stream needs a Hello this child never answers).
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from gen_worker.procsplit import frames  # noqa: E402

MODE = os.environ.get("PGW763_FAKE_MODE", "result_then_die")
WORKER_ID = os.environ.get("PGW763_WORKER_ID", "split-fake-child")


def _hello() -> pb.Hello:
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
        await fw.frame(frames.T_WORKER_MSG, pb.WorkerMessage(
            job_result=pb.JobResult(
                request_id=run.request_id,
                attempt=run.attempt,
                status=pb.JOB_STATUS_OK,
                inline=b"fake-ok",
            ),
        ).SerializeToString())
        if MODE == "result_then_die":
            os.kill(os.getpid(), signal.SIGKILL)
        if MODE == "result_then_exit":
            os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
