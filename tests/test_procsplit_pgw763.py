"""pgw#763 layer 1: control/compute process split — integration suite.

Real everything on the worker side of the socket: a REAL ParentControl
(real Transport, real SendQueue, real supervision) speaking real gRPC to the
hub-double, spawning a REAL compute-child subprocess (real Worker/executor/
lifecycle wired to ChildTransport). The only double is the hub, as everywhere
else in this suite (th#960).

Covers, per the lane's acceptance list:
  1. a handler death does NOT kill the stream/pod — the dead job is
     attributed typed (ComputeProcessDied) on the live stream;
  2. the respawned child serves the NEXT job;
  3. cancellation stays prompt across the process boundary (measured);
  4. crash-looping children are DETECTED and reported typed while respawn
     continues (StartLimitBurst semantics — never cap-and-brick);
  5. a wedged (SIGSTOPped) child is killed by the WatchdogSec analog and
     the pod recovers;
  6. the seam's cost is measured (frame RTT + 64 MiB throughput).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import List, Optional

import grpc
import msgspec
import pytest

from gen_worker import worker_fatal
from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.procsplit import frames
from gen_worker.procsplit.parent import DEATH_LABEL, ParentControl

from harness.hub_double import FakeScheduler, is_accept_for, is_ready, is_result_for

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CHILD_MAIN = TESTS_DIR / "harness" / "procsplit_child_main.py"

BOOT_TIMEOUT_S = 120.0  # child imports torch; generous but real


class _In(msgspec.Struct):
    text: str = ""


def _payload(text: str = "") -> bytes:
    return msgspec.msgpack.encode(_In(text=text))


class SplitHarness:
    """One hub-double + one in-process ParentControl + real child subprocesses."""

    def __init__(
        self,
        tmp: Path,
        *,
        child_cmd: Optional[List[str]] = None,
        watchdog_budget_s: float = 60.0,
        start_limit_burst: int = 3,
        start_limit_interval_s: float = 600.0,
    ) -> None:
        self.scheduler = FakeScheduler()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        pb_grpc.add_WorkerSchedulerServicer_to_server(self.scheduler, self.server)
        port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="split-parent",
            worker_jwt="",
        )
        child_env = {
            "PYTHONPATH": os.pathsep.join(
                [str(TESTS_DIR), str(SRC_DIR), os.environ.get("PYTHONPATH", "")]
            ),
            "TENSORHUB_CACHE_DIR": str(tmp / "cache"),
            "GEN_WORKER_CHILD_WATCHDOG_PING_S": "0.5",
        }
        self.pc = ParentControl(
            settings,
            child_cmd=child_cmd or [sys.executable, str(CHILD_MAIN)],
            child_env=child_env,
            socket_path=str(tmp / "ctl.sock"),
            respawn_backoff_base_s=0.1,
            respawn_backoff_cap_s=0.5,
            transport_backoff_base_s=0.05,
            transport_backoff_cap_s=0.2,
            watchdog_budget_s=watchdog_budget_s,
            start_limit_burst=start_limit_burst,
            start_limit_interval_s=start_limit_interval_s,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.exit_code = self.pc.run()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def close(self) -> None:
        self.pc.stop()
        self._thread.join(20.0)
        self.server.stop(grace=0)


@pytest.fixture()
def captured_dials(monkeypatch):
    """Keep post-mortem dials in-process (no real HTTP) and observable."""
    dials: List[str] = []

    def _capture(settings, detail):
        dials.append(detail)
        return True

    monkeypatch.setattr(worker_fatal, "report_worker_detail", _capture)
    return dials


@pytest.fixture()
def split(tmp_path, captured_dials):
    h = SplitHarness(tmp_path)
    try:
        yield h
    finally:
        h.close()


def test_child_death_keeps_stream_attributes_job_and_respawn_serves_next(
    split, captured_dials,
):
    conn0 = split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn0.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)

    # A normal job completes through the seam.
    conn0.send(run_job=pb.RunJob(
        request_id="r-echo-1", attempt=1, function_name="echo",
        input_payload=_payload("hi")))
    ok = conn0.wait_for(is_result_for("r-echo-1"), timeout=30.0)
    assert ok.job_result.status == pb.JOB_STATUS_OK
    assert b"echo:hi" in ok.job_result.inline

    # The handler SIGKILLs its own process: the cgroup-OOM death shape.
    conn0.send(run_job=pb.RunJob(
        request_id="r-die-1", attempt=1, function_name="die-hard",
        input_payload=_payload()))
    died = conn0.wait_for(is_result_for("r-die-1"), timeout=30.0)
    assert died.job_result.status == pb.JOB_STATUS_FATAL
    assert DEATH_LABEL in died.job_result.safe_message
    assert "function=die-hard" in died.job_result.safe_message

    # The parent (and the pod) survived; the stream identity was kept and the
    # respawned child re-syncs on a fresh connection and serves the NEXT job.
    assert split.alive and split.exit_code is None
    conn1 = split.scheduler.wait_connection(1, timeout=BOOT_TIMEOUT_S)
    conn1.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)
    conn1.send(run_job=pb.RunJob(
        request_id="r-echo-2", attempt=1, function_name="echo",
        input_payload=_payload("again")))
    ok2 = conn1.wait_for(is_result_for("r-echo-2"), timeout=30.0)
    assert ok2.job_result.status == pb.JOB_STATUS_OK
    assert b"echo:again" in ok2.job_result.inline

    # Typed exit capture dialed (gw#640 carried forward), but no crash-loop
    # claim for a single death.
    assert any("compute_process_exit" in d for d in captured_dials)
    assert not any("compute_crash_loop" in d for d in captured_dials)


def test_cancel_stays_prompt_across_the_boundary(split):
    conn = split.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)
    conn.send(run_job=pb.RunJob(
        request_id="r-sleepy", attempt=1, function_name="sleepy",
        input_payload=_payload()))
    conn.wait_for(is_accept_for("r-sleepy"), timeout=30.0)
    t0 = time.monotonic()
    conn.send(cancel_job=pb.CancelJob(request_id="r-sleepy", attempt=1))
    res = conn.wait_for(is_result_for("r-sleepy"), timeout=10.0)
    latency = time.monotonic() - t0
    assert res.job_result.status == pb.JOB_STATUS_CANCELED
    # The handler polls every 50ms; the seam adds one UDS hop. Anything past
    # 3s would be a real regression users feel.
    assert latency < 3.0, f"cancel took {latency:.2f}s across the process boundary"
    print(f"\ncancel latency across seam: {latency * 1000:.0f}ms")


def test_crash_loop_is_detected_reported_and_respawn_continues(
    tmp_path, captured_dials,
):
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"],
        start_limit_burst=3,
        start_limit_interval_s=60.0,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and h.pc._spawn_count < 5:
            time.sleep(0.2)
        # Restart=on-failure with StartLimitBurst DETECTION: the parent keeps
        # respawning past the burst (never cap-and-brick)...
        assert h.pc._spawn_count >= 5
        assert h.alive and h.exit_code is None
        # ...and says so, typed, distinctly from a one-off handler death.
        assert any("compute_crash_loop" in d for d in captured_dials)
        assert any("deaths_before_hello" in d for d in captured_dials)
        # No serving Hello was ever advertised by a child that cannot boot.
        assert not h.scheduler.connections
    finally:
        h.close()


def test_wedged_child_is_killed_by_watchdog_and_pod_recovers(tmp_path, captured_dials):
    h = SplitHarness(tmp_path, watchdog_budget_s=3.0)
    try:
        conn0 = h.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
        conn0.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)
        conn0.send(run_job=pb.RunJob(
            request_id="r-freeze", attempt=1, function_name="freeze",
            input_payload=_payload()))
        # SIGSTOP silences the child's sd_notify-style pings; the parent must
        # kill it and attribute the job as a watchdog hang.
        died = conn0.wait_for(is_result_for("r-freeze"), timeout=30.0)
        assert died.job_result.status == pb.JOB_STATUS_FATAL
        assert DEATH_LABEL in died.job_result.safe_message
        assert "watchdog_hang" in died.job_result.safe_message
        conn1 = h.scheduler.wait_connection(1, timeout=BOOT_TIMEOUT_S)
        conn1.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)
        conn1.send(run_job=pb.RunJob(
            request_id="r-after-freeze", attempt=1, function_name="echo",
            input_payload=_payload("thawed")))
        ok = conn1.wait_for(is_result_for("r-after-freeze"), timeout=30.0)
        assert ok.job_result.status == pb.JOB_STATUS_OK
    finally:
        h.close()


def test_seam_cost_frame_rtt_and_throughput(tmp_path, capsys):
    """Measure the boundary's cost: small-frame RTT (the cancel/dispatch hop)
    and bulk throughput at the gRPC message ceiling (64 MiB)."""

    sock = str(tmp_path / "bench.sock")
    results = {}

    async def _bench() -> None:
        async def _server(reader, writer):
            fw = frames.FrameWriter(writer)
            try:
                while True:
                    ftype, payload = await frames.read_frame(reader)
                    await fw.frame(ftype, payload)
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                # py3.12 Server.wait_closed() waits for handler writers.
                fw.close()

        server = await asyncio.start_unix_server(_server, path=sock)
        reader, writer = await asyncio.open_unix_connection(sock)
        fw = frames.FrameWriter(writer)

        # Small-frame round trips (the shape of a cancel or a progress event).
        small = b"x" * 64
        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            await fw.frame(frames.T_WATCHDOG, small)
            await frames.read_frame(reader)
        rtt_ms = (time.perf_counter() - t0) / n * 1000.0

        # Bulk: 64 MiB frames, the largest message the gRPC stream allows.
        big = b"x" * (64 * 1024 * 1024)
        reps = 4
        t0 = time.perf_counter()
        for _ in range(reps):
            await fw.frame(frames.T_WORKER_MSG, big)
            await frames.read_frame(reader)
        elapsed = time.perf_counter() - t0
        # Each rep moves the payload twice (echo), so credit 2x.
        throughput_mb_s = (len(big) * reps * 2 / (1024 * 1024)) / elapsed

        results["rtt_ms"] = rtt_ms
        results["throughput_mb_s"] = throughput_mb_s
        writer.close()
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), 5.0)
        except asyncio.TimeoutError:
            pass

    asyncio.run(_bench())
    print(
        f"\nseam cost: small-frame RTT {results['rtt_ms']:.3f}ms, "
        f"64MiB-frame throughput {results['throughput_mb_s']:.0f} MB/s"
    )
    # Loose floors: a regression to whole milliseconds per hop or to
    # double-digit MB/s would change the product answer.
    assert results["rtt_ms"] < 5.0
    assert results["throughput_mb_s"] > 200.0
