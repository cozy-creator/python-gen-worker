"""pgw#763 layer 1: control/compute process split — integration suite.

Real everything on the worker side of the socket: a REAL ParentControl
(real Transport, real SendQueue, real supervision) speaking real gRPC to the
hub-double, spawning a REAL compute-child subprocess (real Worker/executor/
lifecycle wired to ChildTransport). The only double is the hub, as everywhere
else in this suite.

Covers, per the lane's acceptance list:
  1. a handler death does NOT kill the stream/pod — the dead job is
     attributed typed (ComputeProcessDied) on the live stream;
  2. the respawned child serves the NEXT job;
  3. cancellation stays prompt across the process boundary (measured);
  4. a pre-Hello crash loop is DETECTED, reported typed, and BOUNDED — the
     parent exits 1 after the boot-death limit instead of billing a respawn
     loop forever; a terminal typed boot verdict (T_BOOT_FATAL)
     exits after ONE spawn with the report relayed on the parent's credential;
  5. a wedged (SIGSTOPped) child is killed by the WatchdogSec analog and
     the pod recovers;
  6. the seam's cost is measured (frame RTT + 64 MiB throughput).
"""

from __future__ import annotations

import asyncio
import os
import signal
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

from harness.hub_double import (
    FakeScheduler,
    is_accept_for,
    is_ready,
    is_result_for,
    measure_child_boot_cost_s,
)
from harness.progress_wait import Cadence, await_count, await_progress

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
CHILD_MAIN = TESTS_DIR / "harness" / "procsplit_child_main.py"
FAKE_CHILD = TESTS_DIR / "harness" / "procsplit_fake_child.py"

_BOOT_RECORD_NAME = "gen-worker-boot-record.json"
_INFLIGHT_NAME = "gen-worker-inflight.json"
_CRASH_REGISTRY_NAME = "gen-worker-crash-streaks.json"
_FAULT_DUMP_NAME = "gen-worker-fault-dump.txt"


def postmortem_dir(tmp: Path) -> Path:
    d = tmp / "postmortem"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        stop_timeout_s: float = 120.0,
        stop_flush_timeout_s: float = 30.0,
        beat_interval_s: float = 0.0,
        extra_child_env: Optional[dict] = None,
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
            # Test-scoped post-mortem markers: the parent consumes the child's
            # in-flight file, so both sides must agree on a per-test dir and
            # never touch the host's (or another test's) markers.
            "GEN_WORKER_BOOT_RECORD": str(postmortem_dir(tmp) / _BOOT_RECORD_NAME),
        }
        child_env.update(extra_child_env or {})
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
            stop_timeout_s=stop_timeout_s,
            stop_flush_timeout_s=stop_flush_timeout_s,
            beat_interval_s=beat_interval_s,
        )
        self.exit_code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        # a parent that exited can never dial in (definitive), and a
        # child boot's silence is worth whatever it costs on THIS runner.
        self.scheduler.worker_alive = lambda: self.alive
        self.scheduler.boot_cost = lambda: measure_child_boot_cost_s(child_env)
        self._thread.start()

    def _run(self) -> None:
        self.exit_code = self.pc.run()

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def signal(self, signum: int) -> None:
        """Deliver a signal to the PARENT (the container's PID 1 in split
        mode). The parent runs in this process, so the real handler is invoked
        the way asyncio's signal handler would."""
        loop = self.pc._loop
        assert loop is not None
        loop.call_soon_threadsafe(self.pc._forward_signal, signum)

    def wait_exit(self, timeout: float) -> Optional[int]:
        self._thread.join(timeout)
        return self.exit_code

    def close(self) -> None:
        self.pc.stop()
        self._thread.join(20.0)
        self.server.stop(grace=0)


@pytest.fixture(autouse=True)
def isolated_postmortem(tmp_path, monkeypatch):
    """The parent CONSUMES the dying child's markers (pgw#676 parity), so the
    in-process parent must read the same per-test dir the child writes — never
    the host's /tmp markers, and never another test's."""
    from gen_worker import postmortem

    d = postmortem_dir(tmp_path)
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", d / _INFLIGHT_NAME)
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", d / _CRASH_REGISTRY_NAME)
    monkeypatch.setattr(postmortem, "FAULT_DUMP_PATH", d / _FAULT_DUMP_NAME)
    return d


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
    conn0 = split.scheduler.wait_connection(0)
    conn0.wait_for(is_ready)

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
    conn1 = split.scheduler.wait_connection(1)
    conn1.wait_for(is_ready)
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
    conn = split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)
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


def test_boot_crash_loop_is_bounded_reported_and_exits_1(
    tmp_path, captured_dials,
):
    """pgw#826: a child that repeatedly dies BEFORE Hello has served nothing
    and never will — after the boot-death limit the parent reports typed and
    exits 1 instead of billing a respawn loop forever. (Post-Hello deaths keep
    Restart=on-failure semantics: see the respawn test above.)"""
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"],
        start_limit_burst=3,
        start_limit_interval_s=60.0,
    )
    try:
        code = h.wait_exit(120.0)
        assert code == 1, f"parent should exit 1 on a boot crash loop, got {code}"
        assert h.pc._spawn_count == 3  # the boot-death limit, then give-up
        assert h.pc.terminal_exit_reason.startswith("boot_crash_loop:")
        # Detection still fires distinctly, then the bound gives up typed.
        assert any("compute_crash_loop" in d for d in captured_dials)
        assert any("compute_boot_crash_loop" in d for d in captured_dials)
        # No serving Hello was ever advertised by a child that cannot boot.
        assert not h.scheduler.connections
    finally:
        h.close()


@pytest.fixture()
def captured_reports(monkeypatch):
    """Keep the parent's HardwareUnsuitable relay in-process and observable."""
    from gen_worker import hardware_report

    reports: List[object] = []

    def _capture(settings, report):
        reports.append(report)
        return True

    monkeypatch.setattr(hardware_report, "deliver_hardware_report", _capture)
    return reports


def test_boot_hardware_fatal_is_terminal_reported_and_exits_1(
    tmp_path, captured_dials, captured_reports,
):
    """pgw#826: ONE typed terminal boot verdict (the compute child's CUDA probe
    refusing) ends the pod — the parent relays the HardwareUnsuitable report on
    its own credential and exits 1, with no respawn."""
    code = (
        "import sys;"
        "from gen_worker.procsplit.child import send_boot_fatal;"
        "send_boot_fatal({'reason_class': 'cuda_unavailable',"
        " 'detail': 'probe says no'});"
        "sys.exit(1)"
    )
    h = SplitHarness(tmp_path, child_cmd=[sys.executable, "-c", code])
    try:
        exit_code = h.wait_exit(120.0)
        assert exit_code == 1, f"parent should exit 1 on a terminal boot fatal, got {exit_code}"
        assert h.pc._spawn_count == 1  # terminal: never respawned
        assert h.pc.terminal_exit_reason == "boot_fatal:cuda_unavailable"
        assert any("compute_boot_fatal" in d for d in captured_dials)
        assert len(captured_reports) == 1
        assert captured_reports[0].reason_class == "cuda_unavailable"
        assert captured_reports[0].detail == "probe says no"
        assert not h.scheduler.connections
    finally:
        h.close()


def test_wedged_child_is_killed_by_watchdog_and_pod_recovers(tmp_path, captured_dials):
    h = SplitHarness(tmp_path, watchdog_budget_s=3.0)
    try:
        conn0 = h.scheduler.wait_connection(0)
        conn0.wait_for(is_ready)
        conn0.send(run_job=pb.RunJob(
            request_id="r-freeze", attempt=1, function_name="freeze",
            input_payload=_payload()))
        # SIGSTOP silences the child's sd_notify-style pings; the parent must
        # kill it and attribute the job as a watchdog hang.
        died = conn0.wait_for(is_result_for("r-freeze"), timeout=30.0)
        assert died.job_result.status == pb.JOB_STATUS_FATAL
        assert DEATH_LABEL in died.job_result.safe_message
        assert "watchdog_hang" in died.job_result.safe_message
        conn1 = h.scheduler.wait_connection(1)
        conn1.wait_for(is_ready)
        conn1.send(run_job=pb.RunJob(
            request_id="r-after-freeze", attempt=1, function_name="echo",
            input_payload=_payload("thawed")))
        ok = conn1.wait_for(is_result_for("r-after-freeze"), timeout=30.0)
        assert ok.job_result.status == pb.JOB_STATUS_OK
    finally:
        h.close()


def test_seam_cost_frame_rtt_and_throughput(tmp_path, capsys):
    """Measure the boundary's COST — never the runner's speed.

    Both arms of both measurements are taken in this run, interleaved, against a
    raw-socket echo on the same event loop and the same transport. The framed
    arm differs from the raw one by exactly the seam: a header pack, one extra
    read, one payload copy. That difference is the product question and it is
    the only thing asserted.

    The bulk arm used to assert an absolute `> 200 MB/s` floor with no baseline
    at all. That is a statement about the machine, not about the frame layer,
    and under `-n 4` on a shared box it is the flake this file was known for.
    Interleaving matters too: measuring the arms back-to-back lets a load ramp
    between them masquerade as seam overhead.
    """

    sock = str(tmp_path / "bench.sock")
    raw_sock = sock + ".raw"
    results = {}

    # Best-of-rounds, applied identically to both arms: contention only ever
    # makes a sample worse, so the best sample of each is the closest either
    # arm gets to its own floor cost — and comparing floors is what isolates
    # the seam. Never a retry-until-green: the arms cannot diverge in treatment.
    ROUNDS = 3
    TRIPS = 200
    # Bulk moves 512 MiB total across both arms — the same memory traffic the
    # single-arm version cost, now with a baseline to compare against.
    BULK_ROUNDS = 2
    BULK_REPS = 1
    BULK = 64 * 1024 * 1024  # the largest message the gRPC stream allows

    async def _bench() -> None:
        async def _framed_server(reader, writer):
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

        async def _raw_server(reader, writer):
            try:
                while True:
                    size = int.from_bytes(await reader.readexactly(8), "big")
                    writer.write(await reader.readexactly(size))
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass

        raw_server = await asyncio.start_unix_server(_raw_server, path=raw_sock)
        raw_reader, raw_writer = await asyncio.open_unix_connection(raw_sock)
        server = await asyncio.start_unix_server(_framed_server, path=sock)
        reader, writer = await asyncio.open_unix_connection(sock)
        fw = frames.FrameWriter(writer)

        async def _raw_trip(payload: bytes) -> None:
            raw_writer.write(len(payload).to_bytes(8, "big"))
            raw_writer.write(payload)
            await raw_writer.drain()
            await raw_reader.readexactly(len(payload))

        async def _framed_trip(ftype, payload: bytes) -> None:
            await fw.frame(ftype, payload)
            await frames.read_frame(reader)

        small = b"x" * 64
        big = b"x" * BULK
        raw_rtt_ms = framed_rtt_ms = float("inf")
        raw_mb_s = framed_mb_s = 0.0

        for _ in range(ROUNDS):
            # Small frames: the shape of a cancel or a progress event.
            t0 = time.perf_counter()
            for _ in range(TRIPS):
                await _raw_trip(small)
            raw_rtt_ms = min(raw_rtt_ms, (time.perf_counter() - t0) / TRIPS * 1000.0)

            t0 = time.perf_counter()
            for _ in range(TRIPS):
                await _framed_trip(frames.T_WATCHDOG, small)
            framed_rtt_ms = min(
                framed_rtt_ms, (time.perf_counter() - t0) / TRIPS * 1000.0)

        for _ in range(BULK_ROUNDS):
            # Bulk at the message ceiling. Each rep moves the payload twice
            # (echo), so credit 2x — identically on both arms.
            moved_mb = BULK * BULK_REPS * 2 / (1024 * 1024)

            t0 = time.perf_counter()
            for _ in range(BULK_REPS):
                await _raw_trip(big)
            raw_mb_s = max(raw_mb_s, moved_mb / (time.perf_counter() - t0))

            t0 = time.perf_counter()
            for _ in range(BULK_REPS):
                await _framed_trip(frames.T_WORKER_MSG, big)
            framed_mb_s = max(framed_mb_s, moved_mb / (time.perf_counter() - t0))

        results.update(
            rtt_ms=framed_rtt_ms, baseline_rtt_ms=raw_rtt_ms,
            throughput_mb_s=framed_mb_s, baseline_throughput_mb_s=raw_mb_s,
        )
        writer.close()
        raw_writer.close()
        server.close()
        raw_server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), 5.0)
        except asyncio.TimeoutError:
            pass

    asyncio.run(_bench())
    print(
        f"\nseam cost: small-frame RTT {results['rtt_ms']:.3f}ms "
        f"(raw-socket baseline {results['baseline_rtt_ms']:.3f}ms), "
        f"64MiB-frame throughput {results['throughput_mb_s']:.0f} MB/s "
        f"(raw-socket baseline {results['baseline_throughput_mb_s']:.0f} MB/s)"
    )
    # A framed hop is a header pack, one extra read and a payload copy over the
    # raw echo it rides. A regression to whole milliseconds of per-hop framing
    # cost changes the product answer; a slow runner slows both arms.
    assert results["rtt_ms"] <= results["baseline_rtt_ms"] * 20.0, (
        f"framed RTT {results['rtt_ms']:.3f}ms vs raw baseline "
        f"{results['baseline_rtt_ms']:.3f}ms — the frame layer itself got slow"
    )
    assert results["throughput_mb_s"] >= results["baseline_throughput_mb_s"] / 4.0, (
        f"framed bulk {results['throughput_mb_s']:.0f} MB/s vs raw baseline "
        f"{results['baseline_throughput_mb_s']:.0f} MB/s — the frame layer is "
        f"copying or chunking the 64MiB path, not just heading it"
    )


# ---------------------------------------------------------------------------
# Stage 2: drain / stop hardening. Every row here is a race between the
# DURABLE queue's retirement and the teardown of something that carries it —
# the stream, the child link, or the parent process itself.
# ---------------------------------------------------------------------------


def test_hub_drain_exits_zero_and_lets_the_child_finish(split, captured_dials):
    """A hub Drain is a DELIBERATE shutdown end to end.

    The child's own drain flushes through the parent's queue and the child
    exits 0. The parent must exit 0 too — and must not SIGKILL the child that
    is still unloading instances just because the stream ended first.
    """
    conn = split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)
    conn.send(run_job=pb.RunJob(
        request_id="r-pre-drain", attempt=1, function_name="echo",
        input_payload=_payload("bye")))
    ok = conn.wait_for(is_result_for("r-pre-drain"), timeout=30.0)
    assert ok.job_result.status == pb.JOB_STATUS_OK

    conn.send(drain=pb.Drain(deadline_ms=30_000))
    assert split.wait_exit(60.0) == 0, "a drain must exit the parent 0"
    assert split.pc._proc is None
    # Nothing about a drain is a death: no typed exit capture, no stop-timeout
    # escalation, and no ComputeProcessDied attribution.
    assert not any("compute_process_exit" in d for d in captured_dials), captured_dials
    assert not any("compute_stop_timeout" in d for d in captured_dials)
    assert not any(
        m.WhichOneof("msg") == "job_result" and DEATH_LABEL in m.job_result.safe_message
        for m in conn.received
    )
    assert split.pc.unretired_results_at_exit == 0


def test_child_death_during_drain_reports_the_job_and_does_not_respawn(
    split, captured_dials,
):
    """The variant stage 1 left open: a child that dies MID-DRAIN.

    Restart=on-failure must not apply to a pod the hub has already retired —
    respawning would re-advertise capacity — but the in-flight job still has
    to be attributed, and that FATAL has to survive the shutdown flush that
    is running concurrently.
    """
    conn = split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)
    conn.send(run_job=pb.RunJob(
        request_id="r-drain-victim", attempt=1, function_name="sleepy",
        input_payload=_payload()))
    conn.wait_for(is_accept_for("r-drain-victim"), timeout=30.0)
    # Drain waits for tenant work, so the child stays alive draining...
    conn.send(drain=pb.Drain(deadline_ms=120_000))
    time.sleep(1.0)
    assert split.pc._proc is not None
    # ...and then the kernel takes it (the cgroup-OOM shape, mid-drain).
    split.pc._proc.kill()

    died = conn.wait_for(is_result_for("r-drain-victim"), timeout=30.0)
    assert died.job_result.status == pb.JOB_STATUS_FATAL
    assert DEATH_LABEL in died.job_result.safe_message
    assert "function=sleepy" in died.job_result.safe_message

    assert split.wait_exit(60.0) == 0, "the drain is still honored"
    assert split.pc._spawn_count == 1, "no respawn into a drain"
    assert len(split.scheduler.connections) == 1, "no fresh Hello after the drain"
    assert split.pc.unretired_results_at_exit == 0, "the death FATAL shipped"
    assert any("compute_process_exit" in d for d in captured_dials)


def test_sigterm_forward_escalates_at_the_stop_deadline(tmp_path, captured_dials):
    """TimeoutStopSec: a child that ignores SIGTERM is SIGKILLed on a budget.

    Uses the frame-speaking child peer (see procsplit_fake_child) because the
    real child handles SIGTERM correctly by design — the point here is the
    parent's own deadline, not the child's cooperation.
    """
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        stop_timeout_s=3.0,
        extra_child_env={"PGW763_FAKE_MODE": "ignore_sigterm"},
    )
    try:
        h.scheduler.wait_connection(0, timeout=30.0)
        h.signal(signal.SIGTERM)
        assert h.wait_exit(40.0) == 0, "a signal shutdown exits 0"
        assert any("compute_stop_timeout" in d for d in captured_dials), captured_dials
        assert h.pc._spawn_count == 1, "a forwarded-signal death is not a crash"
    finally:
        h.close()


def test_result_written_just_before_death_is_neither_lost_nor_blamed(
    tmp_path, captured_dials,
):
    """The child's LAST frames must land before attribution runs.

    A JobResult written microseconds before the process dies is still in the
    socket buffer when waitpid returns. Closing the link there discards it and
    then reports the finished job as ComputeProcessDied — a FATAL for work
    that actually succeeded. Uses the child peer so the write and the SIGKILL
    are adjacent; the real child cannot make that timing deterministic.
    """
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        extra_child_env={"PGW763_FAKE_MODE": "result_then_die"},
    )
    try:
        conn = h.scheduler.wait_connection(0, timeout=30.0)
        conn.send(run_job=pb.RunJob(
            request_id="r-last-frame", attempt=1, function_name="echo",
            input_payload=_payload("x")))
        res = conn.wait_for(is_result_for("r-last-frame"), timeout=30.0)
        assert res.job_result.status == pb.JOB_STATUS_OK, res.job_result.safe_message
        assert res.job_result.inline == b"fake-ok"
        # The child did die, and the pod recovered from it... (pgw#795: the
        # respawn is the progress signal, not a clock)
        await_count(
            lambda: h.pc._spawn_count, 2,
            what="respawn after the child's post-result death",
            cadence=Cadence(),
            gone=lambda: None if h.alive else f"parent exited code={h.exit_code}",
        )
        assert h.pc._spawn_count >= 2
        # ...but the completed job was never re-reported as a death.
        assert not any(
            m.WhichOneof("msg") == "job_result"
            and m.job_result.request_id == "r-last-frame"
            and m.job_result.status == pb.JOB_STATUS_FATAL
            for m in conn.received
        )
    finally:
        h.close()


def test_pgw1324_a_job_recycle_respawns_without_booking_a_death(
    tmp_path, captured_dials,
):
    """pgw#1324: the run-once lifecycle's PARENT half.

    A `@job` finishes and its compute child leaves with `EXIT_JOB_RECYCLE` so
    the next job starts on a process that has never imported this one's world.
    The parent must read that as neither a crash nor a shutdown, and the two
    neighbouring codes give the wrong answer in opposite directions: rc 0 makes
    the parent STOP SUPERVISING (`_finish_deliberate_exit` returns), and any
    other non-zero books a death — a dial, a fault against the pod, and growing
    backoff. A job pod would then read as crash-looping once per submission to
    th#2014's ledger, which is the kind of self-inflicted fault signal that
    gets healthy pods condemned.

    RED by deleting the `rc == EXIT_JOB_RECYCLE` branch in
    `parent.py::_supervise` — the respawn still happens, but `_handle_child_death`
    dials a `compute_process_exit` post-mortem carrying `cause=exit:75` for a
    process that left on purpose.
    """
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        extra_child_env={"PGW763_FAKE_MODE": "result_then_recycle"},
    )
    try:
        conn = h.scheduler.wait_connection(0, timeout=30.0)
        conn.send(run_job=pb.RunJob(
            request_id="r-recycle", attempt=1, function_name="echo",
            input_payload=_payload("x")))
        res = conn.wait_for(is_result_for("r-recycle"), timeout=30.0)
        assert res.job_result.status == pb.JOB_STATUS_OK, res.job_result.safe_message
        # A FRESH CHILD: the whole point of the exit.
        await_count(
            lambda: h.pc._spawn_count, 2,
            what="respawn after the run-once job recycle",
            cadence=Cadence(),
            gone=lambda: None if h.alive else f"parent exited code={h.exit_code}",
        )
        # ...and the parent is still alive supervising it, which rc 0 would not
        # have left true.
        assert h.alive, f"parent exited code={h.exit_code}"
        # No death was booked against the pod for a deliberate departure:
        # `_handle_child_death` would have dialled `cause=exit:75`.
        assert not any("exit:75" in d for d in captured_dials), captured_dials
        # And the completed job was never re-reported as a death.
        assert not any(
            m.WhichOneof("msg") == "job_result"
            and m.job_result.request_id == "r-recycle"
            and m.job_result.status == pb.JOB_STATUS_FATAL
            for m in conn.received
        )
    finally:
        h.close()


def test_parent_exit_with_unretired_results_is_reported_typed(
    tmp_path, captured_dials,
):
    """Unshippable durable results die with the parent — but not silently.

    The child peer writes a JobResult and exits 0 before the parent has any
    stream at all (the handshake needs a Hello the dead child never answers),
    so the result can only be lost. The parent must account for it typed
    rather than exiting as if the queue had been empty.
    """
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD)],
        stop_flush_timeout_s=2.0,
        extra_child_env={"PGW763_FAKE_MODE": "spontaneous_result_then_exit"},
    )
    try:
        assert h.wait_exit(60.0) == 0
        assert h.pc.unretired_results_at_exit == 1, captured_dials
        assert any(
            "compute_parent_exit" in d and "unretired_results=1" in d
            for d in captured_dials
        ), captured_dials
    finally:
        h.close()


def test_signal_death_consumes_the_inflight_marker_and_records_the_streak(
    tmp_path, captured_dials, isolated_postmortem,
):
    """pgw#676/pgw#714 parity in split mode.

    gw#640's supervisor consumed the dying child's in-flight marker, attached
    the faulthandler tail, and recorded the per-function native-crash streak
    that the NEXT boot's gate refuses on. The control parent must do the same
    or the split silently disarms that gate and leaves stale markers to
    misattribute the next death.
    """
    from gen_worker import postmortem

    inflight = isolated_postmortem / _INFLIGHT_NAME
    registry = isolated_postmortem / _CRASH_REGISTRY_NAME
    h = SplitHarness(tmp_path)
    try:
        conn = h.scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-marker", attempt=1, function_name="die-hard",
            input_payload=_payload()))
        died = conn.wait_for(is_result_for("r-marker"))
        assert died.job_result.status == pb.JOB_STATUS_FATAL

        # The job_result is NOT the event these assertions want. The
        # durable attribution is deliberately emitted first (parent.py
        # `_handle_child_death` step 1) and the post-mortem — signal
        # attribution, streak write, dial — is step 2, several awaits and one
        # network hop later. Asserting step 2 the instant step 1 lands is a race
        # the parent wins on an idle box and loses under `-n 4`; that is what
        # "parallel-load flake" meant. Wait for the forensics themselves, giving
        # up only when the parent is gone.
        def _parent_gone():
            return None if h.alive else f"the parent exited code={h.exit_code}"

        exits = await_progress(
            lambda: [d for d in captured_dials if "compute_process_exit" in d],
            bool,
            what="the compute_process_exit dial for the dead child",
            cadence=Cadence(),
            gone=_parent_gone,
            render=lambda ds: f"{len(ds)} exit dial(s)",
        )
        assert "native_crash_streaks" in exits[0], exits[0]
        assert "die-hard" in exits[0]
        streaks = await_progress(
            lambda: postmortem.native_crash_streaks(registry),
            lambda seen: "die-hard" in seen,
            what="the die-hard native-crash streak recorded on disk",
            cadence=Cadence(),
            gone=_parent_gone,
        )
        assert streaks["die-hard"]["count"] == 1, streaks
        # Consumed: a stale marker would misattribute the next death. Checked
        # after the streak, which proves the attribution actually ran — a bare
        # `not exists()` would also pass if it had never run at all.
        assert not inflight.exists()
    finally:
        h.close()


def test_starved_compile_is_held_while_a_dead_child_is_still_killed(
    tmp_path, captured_dials,
):
    """pgw#771: loop silence ARMS the hang verdict; the open activity DECIDES.

    A self-mint compile starves the child's event loop, so the frame ping stops
    — and stage 1's watchdog SIGKILLed exactly that, labelled watchdog_hang.
    That is th#1299 one layer down and strictly worse, because no hub-side hold
    can rescue a child the parent already killed. The thread-sourced liveness
    pipe carries the same kernel-accounted evidence activity.watchdog trusts,
    so the parent holds the verdict (typed) and the job completes.

    The converse — a child where NO thread runs — is still killed: see
    test_wedged_child_is_killed_by_watchdog_and_pod_recovers (SIGSTOP), which
    shares this watchdog and this budget.
    """
    h = SplitHarness(tmp_path, watchdog_budget_s=3.0)
    try:
        conn = h.scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-starve", attempt=1, function_name="starve-loop",
            input_payload=_payload("9")))   # 3x the watchdog budget
        res = conn.wait_for(is_result_for("r-starve"), timeout=90.0)
        assert res.job_result.status == pb.JOB_STATUS_OK, res.job_result.safe_message
        assert b"compiled:9" in res.job_result.inline
        # The child was NOT killed and the pod never blinked.
        assert h.pc._spawn_count == 1
        assert not any("watchdog_hang" in d for d in captured_dials), captured_dials
        assert not any("compute_process_exit" in d for d in captured_dials)
        # ...and the hold is legible, not silent tolerance.
        assert any("compute_hang_verdict_held" in d for d in captured_dials), captured_dials
        assert any("activity=self_mint_compile" in d for d in captured_dials)
    finally:
        h.close()


def test_parent_originates_the_beat_while_the_child_is_starved(
    tmp_path, captured_dials,
):
    """pgw#771: the beat the hub reaps on must be unstarvable.

    The child declares the cadence and used to be its only sender, so a starved
    child stopped beating and the hub killed a live pod at ~6 misses — the
    split relayed the silence rather than curing it. The parent is the control
    plane (no torch, nothing to starve), so it originates the beat: the last
    state the child published, re-sent on the child's own promised cadence.

    Hub-side patience is deliberately NOT assumed: th#1299's activity hold was
    reverted, and an open, freshly-advancing mint activity buys ZERO tolerance
    (heartbeat_contract_th1299_test.go). The beat has to actually arrive.
    """
    h = SplitHarness(tmp_path, watchdog_budget_s=60.0, beat_interval_s=1.0)
    try:
        conn = h.scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        before = sum(1 for m in conn.received if m.WhichOneof("msg") == "state_delta")
        conn.send(run_job=pb.RunJob(
            request_id="r-beat", attempt=1, function_name="starve-loop",
            input_payload=_payload("8")))   # 8x the beat interval, loop pegged
        res = conn.wait_for(is_result_for("r-beat"), timeout=90.0)
        assert res.job_result.status == pb.JOB_STATUS_OK, res.job_result.safe_message
        during = sum(
            1 for m in conn.received if m.WhichOneof("msg") == "state_delta"
        ) - before
        # The child's loop was pegged for ~8 intervals, so anything that
        # arrived came from the parent. Six misses is the hub's reap.
        assert h.pc.parent_beats_sent >= 4, (
            f"parent sent only {h.pc.parent_beats_sent} beats while the child "
            "was starved — the hub would reap this live pod"
        )
        assert during >= 4, f"hub saw only {during} state_deltas during the starve"
    finally:
        h.close()


def test_wedged_child_keeps_the_stream_alive_and_is_reported_not_hidden(
    tmp_path, captured_dials,
):
    """A fully wedged (SIGSTOPped) child must not take the POD down.

    Paul's contract, and the security driver's version of it: the parent is the
    only honest claimant of "the worker is alive and reachable", so it keeps
    beating; and the parent's /proc measurement is the only claim about the
    child's progress that tenant code cannot forge, so the stall is REPORTED
    rather than reaching the hub as silence. The hub kills nothing while the
    parent reports honestly — the child is what gets replaced.
    """
    h = SplitHarness(
        tmp_path,
        # The beat must accumulate BEFORE the verdict arms: the stall report
        # rides the same arm-then-decide ladder as the kill (a child still
        # sending frames is waiting, not stalled), so the budget is the window
        # in which the parent's beats are the only thing keeping the pod
        # reachable.
        watchdog_budget_s=8.0,
        beat_interval_s=1.0,
    )
    try:
        conn = h.scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        before = sum(1 for m in conn.received if m.WhichOneof("msg") == "state_delta")
        conn.send(run_job=pb.RunJob(
            request_id="r-wedge-beat", attempt=1, function_name="freeze",
            input_payload=_payload()))
        # SIGSTOP freezes every thread: the loop, the frame ping, the liveness
        # thread. Nothing the child owns can speak for it.
        stall = await_progress(
            lambda: [d for d in captured_dials if "compute_child_stalled" in d],
            bool,
            what="the compute_child_stalled dial for the frozen child",
            cadence=Cadence(),
            gone=lambda: None if h.alive else f"the parent exited code={h.exit_code}",
            render=lambda ds: f"{len(ds)} stall dial(s)",
        )[0]
        assert "r-wedge-beat#1" in stall, stall
        assert "measured by the parent from /proc" in stall
        # The pod stayed reachable across the wedge: the parent's beats landed
        # while the child could not send a thing.
        during = sum(
            1 for m in conn.received if m.WhichOneof("msg") == "state_delta"
        ) - before
        assert h.pc.parent_beats_sent >= 4, h.pc.parent_beats_sent
        assert during >= 4, f"hub saw {during} state_deltas while the child was frozen"
        assert h.alive and h.exit_code is None, "the parent (the worker) stayed up"

        # And the beat is not an immortality bug: it is the PARENT's claim, so
        # when the parent dies — the true worker-death case — the stream goes
        # silent and the hub's own reap applies.
        h.pc.stop()
        h._thread.join(20.0)
        settled = sum(1 for m in conn.received if m.WhichOneof("msg") == "state_delta")
        time.sleep(3.0)   # 3 beat intervals
        assert sum(
            1 for m in conn.received if m.WhichOneof("msg") == "state_delta"
        ) == settled, "beats continued after the parent died — liveness would be a lie"
    finally:
        h.close()


def test_a_waiting_job_is_never_called_stalled(tmp_path, captured_dials):
    """The regression the first real-stack soak caught.

    `marco-polo-slow` is 15s of `await asyncio.sleep` — zero CPU, zero disk, by
    design — and the parent's /proc witness reported it STALLED twice while it
    was perfectly healthy. Evidence alone cannot tell waiting from wedged; only
    a child whose LOOP has gone silent can carry that claim. Same trap
    activity.note_progress exists for (an I/O-bound fill is CPU-light and still
    progressing).
    """
    h = SplitHarness(tmp_path, watchdog_budget_s=60.0, beat_interval_s=1.0)
    try:
        conn = h.scheduler.wait_connection(0)
        conn.wait_for(is_ready)
        conn.send(run_job=pb.RunJob(
            request_id="r-wait", attempt=1, function_name="async-wait",
            input_payload=_payload("8")))   # 8s of pure awaiting
        res = conn.wait_for(is_result_for("r-wait"), timeout=60.0)
        assert res.job_result.status == pb.JOB_STATUS_OK, res.job_result.safe_message
        assert not any("compute_child_stalled" in d for d in captured_dials), (
            "a legitimately waiting job was reported as stalled: " +
            str([d for d in captured_dials if "compute_child_stalled" in d])
        )
        assert h.pc._spawn_count == 1 and h.alive
    finally:
        h.close()


# ---------------------------------------------------------------------------
# The seam made the Hello AWAITABLE, and that silently un-classified every
# handshake refusal. Second regression from the first real-stack soak.
# ---------------------------------------------------------------------------


class _RefusingScheduler(pb_grpc.WorkerSchedulerServicer):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code, self._details = code, details

    def Connect(self, request_iterator, context):  # noqa: N802
        context.abort(self._code, self._details)


class _AwaitableHelloHandlers:
    """The split parent's shape: `build_hello` is a coroutine, because the
    Hello is fetched from the compute child over the seam."""

    def __init__(self, delay_s: float) -> None:
        self._delay = delay_s

    async def build_hello(self) -> pb.Hello:
        await asyncio.sleep(self._delay)
        return pb.Hello(worker_id="split-parent")

    async def on_hello_ack(self, ack) -> None:  # pragma: no cover - never acked
        pass

    async def on_message(self, msg) -> None:  # pragma: no cover - never acked
        pass

    async def on_disconnect(self) -> None:
        pass


@pytest.mark.parametrize("hello_delay_s", [0.0, 0.05])
def test_awaitable_hello_keeps_a_permanent_refusal_fatal(hello_delay_s: float) -> None:
    """A refusal that cannot heal must still exit, not spin.

    grpc.aio reports the first `write()` on an already-terminated call as
    `InvalidStateError: RPC already finished`, which run()'s catch-all logs as a
    nameless "connection failed" and retries forever. Awaiting the child for the
    Hello put that await between `Connect()` and the write on EVERY dial, so in
    split mode the whole handshake taxonomy — UNAUTHENTICATED's fatal-exit
    ladder (the th#1311 revocation path), not_leader redirects,
    protocol_version_mismatch, worker_id_mismatch — degraded to an infinite
    retry loop. Observed live: a hub restart produced six backoff rounds of
    `InvalidStateError` instead of the honest UNAVAILABLE.
    """
    from gen_worker.transport import FatalTransportError, Transport

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(
        _RefusingScheduler(
            grpc.StatusCode.FAILED_PRECONDITION,
            "worker_id_mismatch: hello=w1 jwt_sub=w2",
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="split-parent",
            worker_jwt="",
        )
        transport = Transport(
            settings,
            _AwaitableHelloHandlers(hello_delay_s),
            backoff_base_s=0.01,
            backoff_cap_s=0.05,
        )

        async def _drive() -> None:
            with pytest.raises(FatalTransportError, match="permanent registration"):
                await asyncio.wait_for(transport.run(), 20.0)

        asyncio.run(_drive())
    finally:
        server.stop(grace=0)


def test_a_finished_rpc_still_yields_its_real_status() -> None:
    """The belt for the same class: whenever grpc.aio does swallow the status,
    the finished call is still asked for it, so the classifier keeps working."""
    from gen_worker.transport import _terminal_rpc_error

    async def _drive() -> None:
        channel = grpc.aio.insecure_channel("127.0.0.1:1")
        try:
            stream = pb_grpc.WorkerSchedulerStub(channel).Connect()
            await asyncio.sleep(0.1)  # let the dial fail first
            with pytest.raises(asyncio.InvalidStateError):
                await stream.write(pb.WorkerMessage(hello=pb.Hello()))
            err = await _terminal_rpc_error(stream)
            assert isinstance(err, grpc.aio.AioRpcError)
            assert err.code() == grpc.StatusCode.UNAVAILABLE
        finally:
            await channel.close()

    asyncio.run(_drive())
