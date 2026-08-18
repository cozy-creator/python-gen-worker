"""The control/compute process split: the boundary, and the authorization it carries.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import threading
import time
from concurrent import futures
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import grpc
import msgspec
import pytest
from harness.hub_double import (
    is_accept_for,
    is_ready,
    is_result_for,
)
from harness.progress_wait import Cadence, await_count, await_progress
from harness.split import (  # noqa: F401 — the shared pgw#763 rig, fixtures included
    _CRASH_REGISTRY_NAME,
    _INFLIGHT_NAME,
    CHILD_MAIN,
    SplitHarness,
    _payload,
    captured_dials,
    captured_reports,
    isolated_postmortem,
)

from gen_worker.config import load_settings
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.procsplit import actions, frames
from gen_worker.procsplit.parent import DEATH_LABEL

# ============================================================================
# pgw#763 — pgw#763 layer 1: control/compute process split — integration
#   suite.
# ============================================================================

TESTS_DIR = Path(__file__).resolve().parent


SRC_DIR = TESTS_DIR.parent / "src"


FAKE_CHILD = TESTS_DIR / "harness" / "procsplit_fake_child.py"


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
    """pgw#826: a child that repeatedly dies BEFORE Hello has served nothing and never will — after the boot-dea..."""
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


def test_boot_hardware_fatal_is_terminal_reported_and_exits_1(
    tmp_path, captured_dials, captured_reports,
):
    """pgw#826: ONE typed terminal boot verdict (the compute child's CUDA probe refusing) ends the pod — the par..."""
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
    """pgw#763: Measure the boundary's COST — never the runner's speed."""

    sock = str(tmp_path / "bench.sock")
    raw_sock = sock + ".raw"
    results: Dict[str, Any] = {}

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

        async def _framed_trip(ftype: Any, payload: bytes) -> None:
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


def test_hub_drain_exits_zero_and_lets_the_child_finish(split, captured_dials):
    """pgw#763: A hub Drain is a DELIBERATE shutdown end to end."""
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
    """pgw#763: The variant stage 1 left open: a child that dies MID-DRAIN."""
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
    """pgw#763: TimeoutStopSec: a child that ignores SIGTERM is SIGKILLed on a budget."""
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
    """pgw#763: The child's LAST frames must land before attribution runs."""
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
    """pgw#1324: the run-once lifecycle's PARENT half."""
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
    """pgw#763: Unshippable durable results die with the parent — but not silently."""
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
    """pgw#676/pgw#714 parity in split mode."""
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


def test_a_starving_handler_no_longer_starves_the_worker_loop_pgw1373(
    tmp_path, captured_dials,
):
    """pgw#771's hazard is GONE under pgw#1373, so its verdict cannot fire.

    The original case asserted `compute_hang_verdict_held`: an async handler
    that burns CPU without yielding starved the worker's OWN event loop, the
    parent's watchdog ARMED on the resulting silence, and the open activity
    then HELD the verdict instead of killing a live pod.

    The v2 serve path runs author code on a worker thread
    (`asyncio.to_thread` -> `ServeLoop.invoke`), so a starving handler cannot
    reach the loop that answers pings. Nothing arms, so nothing can be held —
    asserting the hold would be asserting a verdict that no longer has a
    trigger. What this now measures is the property that REPLACED it: the same
    9-second non-yielding handler completes, the child is never killed, and no
    hang verdict is reached at all.

    The hold machinery itself is untouched and still covered by the parent-side
    unit that drives the verdict directly; what changed is that the compute
    child no longer produces the silence it defers to.
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
    finally:
        h.close()


def test_parent_originates_the_beat_while_the_child_is_starved(
    tmp_path, captured_dials,
):
    """pgw#771: the beat the hub reaps on must be unstarvable."""
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
    """pgw#763: A fully wedged (SIGSTOPped) child must not take the POD down."""
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
    """pgw#763: The regression the first real-stack soak caught."""
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


class _RefusingScheduler(pb_grpc.WorkerSchedulerServicer):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        self._code, self._details = code, details

    def Connect(self, request_iterator, context):  # noqa: N802
        context.abort(self._code, self._details)


class _AwaitableHelloHandlers:
    """pgw#763: The split parent's shape: `build_hello` is a coroutine, because the Hello is fetched from the co..."""

    def __init__(self, delay_s: float) -> None:
        self._delay = delay_s

    async def build_hello(self) -> pb.Hello:
        await asyncio.sleep(self._delay)
        return pb.Hello(worker_id="split-parent")

    async def on_hello_ack(self, ack: Any) -> None:  # pragma: no cover
        pass

    async def on_message(self, msg: Any) -> None:  # pragma: no cover
        pass

    async def on_disconnect(self) -> None:
        pass


@pytest.mark.parametrize("hello_delay_s", [0.0, 0.05])
def test_awaitable_hello_keeps_a_permanent_refusal_fatal(hello_delay_s: float) -> None:
    """pgw#763: A refusal that cannot heal must still exit, not spin."""
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
    """pgw#763: The belt for the same class: whenever grpc.aio does swallow the status, the finished call is sti..."""
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


# ============================================================================
# pgw#763 — pgw#763 driver 3: the parent/child seam as an AUTHORIZATION
#   boundary.
# ============================================================================

WORKER_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3LXBhcmVudCIsInJlbGVhc2VfaWQiOiJyZWwtNzYzIn0.sig"


def _text(msg: pb.WorkerMessage) -> str:
    """The handler's ProbeOut.response out of an OK JobResult."""
    return msgspec.msgpack.decode(msg.job_result.inline)["response"]


class _HubHTTP(BaseHTTPRequestHandler):
    def _answer(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}") \
            if self.command == "POST" else {}
        self.server.calls.append({  # type: ignore[attr-defined]
            "method": self.command,
            "path": self.path,
            "authorization": self.headers.get("Authorization", ""),
            "body": body,
        })
        payload = json.dumps(
            self.server.reply  # type: ignore[attr-defined]
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _answer
    do_POST = _answer

    def log_message(self, *a: Any) -> None:
        pass


@pytest.fixture()
def hub_http():
    srv = HTTPServer(("127.0.0.1", 0), _HubHTTP)
    srv.calls = []  # type: ignore[attr-defined]
    srv.reply = {"capability_token": "fresh-token", "expires_at_unix": 4102444800}  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def credentialed_split(tmp_path, captured_dials, monkeypatch, hub_http):
    """pgw#763: A split whose PARENT holds a real-shaped worker JWT, delivered the way a pod gets one: in the pr..."""
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        extra_child_env={"PGW763_CHILD_MODULES": "harness.procsplit_endpoints"},
    )
    h.scheduler.file_base_url = f"http://127.0.0.1:{hub_http.server_address[1]}"
    # The harness builds Settings with bootstrap_worker_jwt="" (its own default);
    # give the parent the credential a real pod is launched with.
    h.pc._settings = msgspec.structs.replace(
        h.pc._settings, bootstrap_worker_jwt=WORKER_JWT)
    h.pc.transport._settings = h.pc._settings
    try:
        yield h
    finally:
        h.close()


def test_delta1_tenant_code_finds_no_worker_jwt_in_its_process(credentialed_split):
    """pgw#763: THE ATTACK: an endpoint handler reads the pod's signing identity."""
    conn = credentialed_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    conn.send(run_job=pb.RunJob(
        request_id="r-steal", attempt=1, function_name="steal-credentials",
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-steal"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    leaked = _text(got)
    assert leaked == "", (
        f"tenant code reached the worker JWT via {leaked} — the compute child "
        "must hold no signing identity (pgw#763 delta 1 / th#1311)"
    )

    # ...and the PARENT still has it: the credential moved, it did not vanish.
    assert credentialed_split.pc.transport.current_worker_jwt == WORKER_JWT


def test_delta1_parent_refuses_a_hub_call_the_allowlist_does_not_name(
    credentialed_split, hub_http, captured_dials,
):
    """pgw#763: THE ATTACK: with no credential of its own, the child asks the parent to make the call for it — a..."""
    conn = credentialed_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    conn.send(run_job=pb.RunJob(
        request_id="r-forge", attempt=1, function_name="forge-hub-call",
        input_payload=_payload("/v1/admin/orgs")))
    got = conn.wait_for(is_result_for("r-forge"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    answer = _text(got)
    assert answer.startswith("refused:"), (
        f"the parent performed an un-allowlisted hub call for the child ({answer})"
    )
    assert "not an allowlisted parent-mediated action" in answer
    assert credentialed_split.pc.actions_refused >= 1
    # Nothing reached the hub, so the JWT was never presented.
    assert not [c for c in hub_http.calls if "/v1/admin/" in c["path"]]
    assert any("compute_action_refused" in d for d in captured_dials)


def test_delta1_parent_refuses_capability_renewal_for_a_foreign_request(
    credentialed_split, hub_http,
):
    """pgw#763: THE ATTACK: the path IS allowlisted, so only parent STATE can refuse it — a renewal for a reques..."""
    conn = credentialed_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    conn.send(run_job=pb.RunJob(
        request_id="r-renew-forge", attempt=1,
        function_name="forge-capability-renew",
        input_payload=_payload("victim-request-id")))
    got = conn.wait_for(is_result_for("r-renew-forge"), timeout=60.0)
    answer = _text(got)
    assert answer.startswith("refused:"), (
        f"the parent renewed a capability for a job it never dispatched ({answer})"
    )
    assert "not an in-flight job on this worker" in answer
    assert not [c for c in hub_http.calls if "capability/renew" in c["path"]]


def test_delta1_the_legitimate_mediated_call_still_works(credentialed_split, hub_http):
    """pgw#763: The other half: an ALLOWLISTED action for a job that IS in flight goes through — the parent atta..."""
    pc = credentialed_split.pc
    conn = credentialed_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    # Stand in for a dispatched job (the relay records exactly this).
    pc._slots[0].in_flight[("r-live", 1)] = "echo"
    status, body = _ask(pc, {
        "method": "POST",
        "path": "/v1/worker/capability/renew",
        "json": {"request_id": "r-live", "attempt": 1, "capability_token": "old"},
    })
    assert status == 200, body
    assert json.loads(body)["capability_token"] == "fresh-token"

    call = [c for c in hub_http.calls if "capability/renew" in c["path"]][-1]
    assert call["authorization"] == f"Bearer {WORKER_JWT}", (
        "the parent must present the worker JWT on the child's behalf"
    )
    assert call["body"]["request_id"] == "r-live"


def test_the_childs_timeout_may_only_lower_the_allowlists_own(
    credentialed_split, hub_http, monkeypatch,
):
    """pgw#973 (§4.24): the child supplies `timeout`, so the ALLOWLIST's `timeout_s` is what stops tenant code p..."""
    from gen_worker.procsplit import actions
    from gen_worker.procsplit import parent as parent_mod

    assert max(a.timeout_s for a in actions.ACTIONS.values()) == 60.0, (
        "an action now declares a longer budget than the deleted ceiling "
        "assumed — re-derive it before trusting this clamp"
    )
    seen: List[float] = []
    real = parent_mod._http_call

    def spy(method, url, token, query, body, timeout):
        seen.append(float(timeout))
        return real(method, url, token, query, body, timeout)

    monkeypatch.setattr(parent_mod, "_http_call", spy)

    pc = credentialed_split.pc
    conn = credentialed_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)
    pc._slots[0].in_flight[("r-live", 1)] = "echo"

    status, _body = _ask(pc, {
        "method": "POST",
        "path": "/v1/worker/capability/renew",
        "json": {"request_id": "r-live", "attempt": 1, "capability_token": "old"},
        "timeout": 99999,
    })
    assert status == 200
    action = actions.ACTIONS["capability.renew"]
    assert seen == [action.timeout_s], (
        f"the child's 99999 s reached the socket as {seen}"
    )

    # ... and a SMALLER number from the child is still honoured: the clamp is a
    # ceiling, not an override.
    seen.clear()
    _ask(pc, {
        "method": "POST",
        "path": "/v1/worker/capability/renew",
        "json": {"request_id": "r-live", "attempt": 1, "capability_token": "old"},
        "timeout": 2,
    })
    assert seen == [2.0]


def _ask(pc: Any, req: Dict[str, Any]) -> Tuple[int, str]:
    """Drive one action through the parent's real authorization path."""
    import asyncio

    fut = asyncio.run_coroutine_threadsafe(pc._perform_action(req), pc._loop)
    out = fut.result(60.0)
    return int(out["status"]), str(out["body"])


FAKE_CHILD_pgw763 = Path(__file__).resolve().parent / "harness" / "procsplit_fake_child.py"


@pytest.fixture()
def forging_split(tmp_path, captured_dials, monkeypatch):
    """pgw#763: A hostile compute child: it answers the Hello request with fabricated identity, silicon and cana..."""
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD_pgw763)],
        extra_child_env={"PGW763_FAKE_MODE": "forge_hello"},
    )
    h.pc._settings = msgspec.structs.replace(
        h.pc._settings, bootstrap_worker_jwt=WORKER_JWT,
        worker_image_digest="sha256:real")
    h.pc.transport._settings = h.pc._settings
    try:
        yield h
    finally:
        h.close()


def test_delta2_parent_measurement_replaces_a_forged_hello(forging_split):
    """pgw#763: THE ATTACK: the child reports the hardware and the boot canary."""
    from harness.procsplit_fake_child import (  # type: ignore
        FORGED_GPU_NAME,
        FORGED_MEMCPY_GBPS,
        FORGED_RELEASE_ID,
        FORGED_VRAM_BYTES,
        FORGED_WORKER_ID,
    )

    conn = forging_split.scheduler.wait_connection(0)
    hello = conn.hello
    assert hello is not None

    # Identity (delta 1): asserted by the credential holder, not the child.
    assert hello.worker_id != FORGED_WORKER_ID, (
        "the child named another worker and the hub believed it"
    )
    assert hello.release_id != FORGED_RELEASE_ID

    # Silicon + canary (delta 2): the parent's measurement, or nothing at all.
    res = hello.resources
    assert res.gpu_name != FORGED_GPU_NAME, (
        f"gpu_name={res.gpu_name!r} came from the child — a forged SKU picks "
        "the fleet-wide verdict key (th#1310)"
    )
    assert res.vram_total_bytes != FORGED_VRAM_BYTES
    assert res.gpu_sm != "90"
    assert res.torch_version != "9.9.9"
    assert res.gen_worker_version != "0.0.0-forged"
    assert res.host_canary.memcpy_gbps != FORGED_MEMCPY_GBPS, (
        "a fabricated HostCanary reached the hub; it condemns SKUs"
    )
    assert res.host_canary.d2h_gbps != FORGED_MEMCPY_GBPS
    assert res.host_canary.interconnect != "nvlink"
    # instance_id/image_digest identify the POD, so they come from the process
    # that holds the pod's credential.
    assert res.instance_id != "pod-belonging-to-someone-else"
    assert res.image_digest in ("", "sha256:real")


def test_delta2_the_parent_measures_the_real_host(forging_split):
    """pgw#763: The legitimate half: the numbers on the wire are the ones this box actually has, produced by the..."""
    forging_split.scheduler.wait_connection(0)
    pc = forging_split.pc
    assert pc._measurement is not None, "the parent never measured the host"

    from gen_worker.procsplit.measure import measure

    truth = measure()
    hw = pc._measurement.get("hardware") or {}
    assert hw.get("gpu_name", "") == (truth.get("hardware") or {}).get("gpu_name", "")
    assert hw.get("gpu_count", 0) == (truth.get("hardware") or {}).get("gpu_count", 0)
    assert pc._measurement.get("gen_worker_version") == truth.get("gen_worker_version")
    # The canary is measured on a box with a GPU; on a CPU-only box it is
    # absent, and absent is the honest answer — never a fabricated one.
    assert ("canary" in pc._measurement) == ("canary" in truth)


def test_delta2_measurement_process_imports_no_endpoint_module():
    """pgw#763: The property that makes the measurement trustworthy at all: the process that produces it never n..."""
    import ast

    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "gen_worker" / "procsplit" / "measure.py"
    ).read_text()
    tree = ast.parse(src)

    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
            imported.update(a.name for a in node.names)
    # Endpoint discovery lives behind exactly these two names; either one in
    # this process means tenant code ran before the numbers were taken.
    for banned in ("collect_endpoints", "registry", "worker", "..worker",
                   "..registry", "Worker"):
        assert banned not in imported, (
            f"the pre-import measurement imports {banned!r} — it must reach no "
            "endpoint-discovery code (pgw#763 delta 2)"
        )
    # And nothing dynamic can smuggle one in.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("__import__", "eval", "exec"), (
                "dynamic import in the pre-import measurement"
            )


@pytest.fixture()
def billing_split(tmp_path, captured_dials, monkeypatch):
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD_pgw763)],
        extra_child_env={"PGW763_FAKE_MODE": "forge_metrics"},
    )
    try:
        yield h
    finally:
        h.close()


def test_delta3_forged_billables_are_replaced_by_the_parents_observation(
    billing_split, captured_dials,
):
    """pgw#763: THE ATTACK: the child reports its own billing quantities."""
    from harness.procsplit_fake_child import (  # type: ignore
        FORGED_CONCURRENCY,
        FORGED_RSS_BYTES,
        FORGED_RUNTIME_MS,
    )

    conn = billing_split.scheduler.wait_connection(0)
    conn.send(run_job=pb.RunJob(
        request_id="r-bill", attempt=1, function_name="echo",
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-bill"), timeout=60.0)
    m = got.job_result.metrics

    assert m.runtime_ms < FORGED_RUNTIME_MS, (
        f"runtime_ms={m.runtime_ms} survived: the code being billed set its own "
        "billable wall clock (th#1309)"
    )
    # The clamp must be the parent's OBSERVED wall, not a cap. Anchored to the
    # attack's own magnitude rather than a wall clock: an echo job's
    # parent-observed wall is fractions of a second, so any value within even
    # a thousandth of the forged three hours (10.8 s) is a clamp/cap artifact,
    # not an observation — the old `< 60_000` bound would have PASSED a 59 s
    # cap.
    assert m.runtime_ms < FORGED_RUNTIME_MS // 1000, (
        f"runtime_ms={m.runtime_ms} is a fraction of the forgery, not an "
        "observation — the parent must attest the wall it measured"
    )
    for name in ("queue_ms", "slot_held_ms", "finalize_wall_ms"):
        assert getattr(m, name) < FORGED_RUNTIME_MS, f"{name} survived unattested"
    assert m.concurrency_at_start != FORGED_CONCURRENCY
    assert m.concurrency_at_start == 0, (
        "concurrency_at_start must be the parent's own dispatch-time count"
    )
    assert m.rss_at_end_bytes != FORGED_RSS_BYTES, (
        "rss_at_end_bytes is a /proc reading the parent takes; a process is not "
        "the witness for its own resource use"
    )
    # The divergence is BANKED, not merely clamped away in silence.
    assert billing_split.pc.metric_divergences >= 1
    assert any("compute_billing_attestation" in d for d in captured_dials)
    # `output_media_duration_s=0` is NO LONGER a divergence, and its
    # absence is the assertion. `_scan_output_assets` sums `Asset.duration_s`,
    # which only a TEMPORAL asset has, so a still image legitimately reports
    # 0.0 — the check fired on every successful image job and the parent could
    # never tell "correctly zero for an image" from "wrongly zero for a video".
    # It was not merely noisy: each divergence dials the post-mortem carrier on
    # an unrefreshable boot token, and three rejected dials terminate the pod
    # (pgw#846 attempt sixteen, 35 min into the longest AOT mint on record).
    # th#1309's property moves to the HUB, which knows the settlement model.
    assert not any("output_media_duration_s" in d for d in captured_dials), (
        "the image-job false positive is back — it manufactures a dial per "
        "5 min and three rejected dials terminate the pod (th#1364)"
    )


def test_delta3_an_honest_report_passes_through_unchanged():
    """pgw#763: The other half: attestation is not a rewrite."""
    from gen_worker.procsplit import attest

    metrics = pb.JobMetrics(
        runtime_ms=1200, queue_ms=30, slot_held_ms=1100, finalize_wall_ms=90,
        concurrency_at_start=2, rss_at_end_bytes=4 << 30,
        output_media_duration_s=8.5, output_count=1,
        input_tokens=120, output_tokens=64, lane="fp8-w8a8-dynamic+compiled",
    )
    obs = attest.JobObservation(
        function="generate",
        relayed_at=0.0,
        concurrency_at_relay=2,
    )
    divergences = attest.attest(
        metrics, obs, now=1.6, child_rss_bytes=(4 << 30) + 1000, status_ok=True)

    assert divergences == [], divergences
    assert metrics.runtime_ms == 1200 and metrics.queue_ms == 30
    assert metrics.concurrency_at_start == 2
    # Untouched by design — measuring these parent-side would mean routing the
    # data plane through the parent's interpreter (th#1309 owns the hub bound).
    assert metrics.output_media_duration_s == 8.5
    assert metrics.input_tokens == 120 and metrics.output_tokens == 64
    assert metrics.lane == "fp8-w8a8-dynamic+compiled"


def test_delta5_the_child_signs_through_the_parent_holding_no_credential(
    credentialed_split, hub_http,
):
    """th#1307, finished at the seam."""
    import base64

    hub_http.reply = {"signature_b64": base64.b64encode(b"SIGNATURE").decode()}
    conn = credentialed_split.scheduler.wait_connection(0)
    conn.wait_for(is_ready)

    # The handler passes a base URL of its own choosing. It is IGNORED: the
    # parent aims its own credential, at the host the hub named.
    conn.send(run_job=pb.RunJob(
        request_id="r-sign", attempt=1, function_name="c2pa-sign",
        input_payload=_payload("http://attacker.invalid")))
    got = conn.wait_for(is_result_for("r-sign"), timeout=60.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    assert _text(got) == "signed:SIGNATURE", _text(got)

    call = [c for c in hub_http.calls if c["path"] == "/v1/worker/c2pa/sign"][-1]
    assert call["authorization"] == f"Bearer {WORKER_JWT}", (
        "the signing oracle must be authenticated by the parent's credential"
    )
    # A hash and the algorithm. Nothing else crosses — not the media, not a
    # destination, not a header.
    assert set(call["body"]) == {"alg", "claim_b64"}, call["body"]
    assert base64.b64decode(call["body"]["claim_b64"]) == b"claim-to-be-signed"


def test_delta5_the_sign_action_cannot_be_widened(credentialed_split, hub_http):
    """pgw#763: The narrowness IS the fix: the child asks the parent to sign a hash it was given, and nothing mo..."""
    with pytest.raises(actions.ActionRefused):
        actions.authorize({
            "method": "POST", "path": "/v1/worker/c2pa/sign",
            "json": {"alg": "es256", "claim_b64": "AA==", "key_id": "platform"},
        })


def _cap_token(**claims: Any) -> str:
    """pgw#763: An unsigned-shaped JWT carrying the hub's real capability claims."""
    import base64

    def seg(obj: Dict[str, Any]) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    body: Dict[str, Any] = {
        "cap_kind": "worker_capability",
        "iat": int(__import__("time").time()),
        "exp": int(__import__("time").time()) + 900,
        "grants": [{"resource": "media", "actions": ["write"]}],
    }
    body.update(claims)
    return f"{seg({'alg': 'RS256'})}.{seg(body)}.sig"


def test_delta4_a_grant_for_another_request_is_withheld(split_for_capability):
    """pgw#763: THE ATTACK (or the hub bug that looks like one): a job arrives carrying a capability token minte..."""
    pc, conn = split_for_capability
    conn.send(run_job=pb.RunJob(
        request_id="r-mine", attempt=1, function_name="echo",
        capability_token=_cap_token(
            request_id="r-someone-else", attempt=1, worker_id="split-parent"),
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-mine"), timeout=30.0)
    assert got.job_result.status == pb.JOB_STATUS_FATAL
    assert "CapabilityWithheld" in got.job_result.safe_message
    assert "scoped to request r-someone-else" in got.job_result.safe_message
    assert pc.capability_withheld >= 1
    # Never dispatched: no accounting for a job the parent refused.
    assert ("r-mine", 1) not in pc._all_in_flight()


def test_delta4_an_expired_grant_is_withheld_retryable(split_for_capability):
    """pgw#763: A grant that is already dead cannot upload the job's output."""
    import time as _t

    pc, conn = split_for_capability
    conn.send(run_job=pb.RunJob(
        request_id="r-stale", attempt=1, function_name="echo",
        capability_token=_cap_token(
            request_id="r-stale", attempt=1, worker_id="split-parent",
            iat=int(_t.time()) - 7200, exp=int(_t.time()) - 600),
        input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-stale"), timeout=30.0)
    assert got.job_result.status == pb.JOB_STATUS_RETRYABLE
    assert "expired" in got.job_result.safe_message


def test_delta4_a_correctly_scoped_grant_is_forwarded(split_for_capability):
    """pgw#763: The other half: a grant that names this job on this worker goes through untouched — least author..."""
    pc, conn = split_for_capability
    token = _cap_token(request_id="r-ok", attempt=1, function_name="echo",
                       worker_id="split-parent")
    conn.send(run_job=pb.RunJob(
        request_id="r-ok", attempt=1, function_name="echo",
        capability_token=token, input_payload=_payload()))
    got = conn.wait_for(is_result_for("r-ok"), timeout=30.0)
    assert got.job_result.status == pb.JOB_STATUS_OK
    assert pc.capability_withheld == 0


@pytest.fixture()
def split_for_capability(tmp_path, captured_dials, monkeypatch):
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)
    h = SplitHarness(
        tmp_path,
        child_cmd=[sys.executable, str(FAKE_CHILD_pgw763)],
        extra_child_env={"PGW763_FAKE_MODE": "result_then_exit"},
    )
    try:
        conn = h.scheduler.wait_connection(0)
        yield h.pc, conn
    finally:
        h.close()


@pytest.mark.parametrize(
    "claims,forward",
    [
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-1"}, True),
        ({"request_id": "r-2", "attempt": 1, "worker_id": "w-1"}, False),
        ({"request_id": "r-1", "attempt": 2, "worker_id": "w-1"}, False),
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-other"}, False),
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-1",
          "function_name": "other-fn"}, False),
        ({"request_id": "r-1", "attempt": 1, "worker_id": "w-1",
          "cap_kind": "org_access_token"}, False),
    ],
)
def test_capability_policy_matrix(claims, forward):
    from gen_worker.procsplit import capability

    d = capability.decide(
        _cap_token(**claims),
        request_id="r-1", attempt=1, function_name="generate", worker_id="w-1")
    assert d.forward is forward, d.reason


def test_capability_policy_reports_an_over_long_ttl_without_refusing():
    """pgw#763: Only the hub can shorten a TTL, so refusing legitimate work over one would trade a real outage f..."""
    import time as _t

    from gen_worker.procsplit import capability

    d = capability.decide(
        _cap_token(request_id="r-1", attempt=1, worker_id="w-1",
                   iat=int(_t.time()),
                   exp=int(_t.time()) + capability.MAX_EXPECTED_TTL_S + 3600),
        request_id="r-1", attempt=1, worker_id="w-1")
    assert d.forward is True
    assert "TTL" in d.note


def test_capability_policy_passes_a_job_with_no_grant():
    """pgw#763: Jobs with no file authority legitimately exist; a missing token is not a withheld one."""
    from gen_worker.procsplit import capability

    assert capability.decide("", request_id="r", attempt=1).forward is True


@pytest.mark.parametrize(
    "req,why",
    [
        # pgw#1362: the "unlisted path" row lived here too and is DELETED. Its
        # subject is `authorize`'s path match, and mutating that reddens
        # `test_delta1_parent_refuses_a_hub_call_the_allowlist_does_not_name`,
        # which asserts the same property through a REAL split child. The rows
        # below are NOT variants of it — each was probed and is uniquely
        # guarded (notably "wrong method": breaking method matching alone
        # reddens that row and nothing else).
        ({"method": "POST", "path": "/v1/worker/compiled-graphs/receipt"}, "wrong method"),
        ({"method": "GET", "path": "/api/v1/repos/a/b/../../admin/resolve"},
         "traversal out of the allowlisted prefix"),
        ({"method": "GET", "path": "/v1/worker/compiled-graphs/receipt",
          "query": {"blake3": "x", "owner": "root"}}, "query key not in the action"),
        ({"method": "POST", "path": "/v1/worker/c2pa/sign",
          "json": {"alg": "es256", "claim_b64": "AA", "callback_url": "http://evil"}},
         "body key not in the action"),
        ({"method": "POST", "path": "/v1/worker/capability/renew",
          "json": {"request_id": "r", "attempt": 1,
                   "capability_token": "x" * (300 * 1024)}},
         "oversized body: the seam carries control, not data"),
    ],
)
def test_action_table_refuses(req, why):
    with pytest.raises(actions.ActionRefused):
        actions.authorize(req)


def test_action_table_admits_exactly_the_named_actions():
    for req in (
        {"method": "POST", "path": "/v1/worker/capability/renew",
         "json": {"request_id": "r", "attempt": 1, "capability_token": "t"}},
        {"method": "POST", "path": "/v1/worker/c2pa/sign",
         "json": {"alg": "es256", "claim_b64": "AA=="}},
        {"method": "GET", "path": "/v1/worker/compiled-graphs/receipt",
         "query": {"compiled_graph_key": "k", "artifact_digest": ["sha256:" + "a" * 64]}},
        {"method": "GET", "path": "/v1/worker/compiled-graphs/revocations"},
        {"method": "GET", "path": "/api/v1/repos/root/system-sdxl/checkpoints",
         "query": {"limit": "50"}},
        {"method": "GET", "path": "/api/v1/repos/root/system-sdxl/resolve",
         "query": {"digest": "ck1-abc"}},
    ):
        actions.authorize(req)


def test_a_compute_child_with_no_seam_refuses_to_dial_the_hub_itself(monkeypatch):
    """pgw#763: No silent downgrade."""
    from gen_worker.procsplit import broker

    monkeypatch.setenv("GEN_WORKER_COMPUTE_CHILD", "1")
    broker.install(None)
    with pytest.raises(broker.BrokerError) as exc:
        broker.request("GET", "/v1/worker/compiled-graphs/revocations",
                       base_url="http://hub", bearer="")
    assert "parent-mediated" in str(exc.value)


def test_no_frame_carries_the_worker_jwt():
    """pgw#763: A ratchet over the frame vocabulary itself: T_TOKEN is gone and nothing replaced it."""
    from gen_worker.procsplit import frames

    assert not hasattr(frames, "T_TOKEN")
    names = [n for n in dir(frames) if n.startswith("T_")]
    for name in names:
        assert "TOKEN" not in name and "JWT" not in name, (
            f"frame {name} looks like it carries a credential to the compute child"
        )


def test_th1364_a_still_image_job_is_not_a_billing_divergence():
    """th#1364 ROOT: an image job reports 0.0 media seconds because that is the correct answer, and it must not ..."""
    from gen_worker.procsplit import attest

    metrics = pb.JobMetrics(
        runtime_ms=1200, queue_ms=30, concurrency_at_start=2,
        rss_at_end_bytes=4 << 30,
        output_media_duration_s=0.0,  # a still image: correct, not under-reported
        output_count=1,
    )
    obs = attest.JobObservation(
        function="generate", relayed_at=0.0, concurrency_at_relay=2)
    divergences = attest.attest(
        metrics, obs, now=1.6, child_rss_bytes=(4 << 30) + 1000, status_ok=True)

    assert divergences == [], (
        f"an ordinary image job produced billing divergences {divergences} — "
        "this fires on most of the fleet's work and each one dials the "
        "pod-killing carrier (th#1364)"
    )


# -- the host census: "no GPU" vs "a GPU I failed to see" (pgw#1414) ---------
#
# Measured on a rented 4090 (pod 3ntpe1zwbksuwo, $0.336): the census came back
# `driver="" gpu="" count=0`, the worker registered `class=cpu gpu=0`, and the
# hub declined placement with `compute_class_mismatch` 703+ times in a loop
# with no terminal state while the pod billed. ZERO errors anywhere — because a
# swallowed census and a genuinely cardless box produce byte-identical output.
#
# This census runs in the PARENT before any endpoint import; the child's CUDA
# probe runs later. A driver mount landing between them yields exactly that
# incident: a cpu-class Hello from a pod whose probe then passes.


def _census(
    monkeypatch: pytest.MonkeyPatch, *, devices: bool, readings: List[Any]
) -> Dict[str, Any]:
    """Drive `measure()` with the two facts that decide the arm."""
    from gen_worker.procsplit import measure as measure_mod

    monkeypatch.setattr(measure_mod, "gpu_devices_present", lambda: devices)
    # By TARGET STRING, not `measure_mod.os`: the module imports `os` for its
    # own use and does not re-export it, so reaching through it is a fact about
    # this test's imports rather than about the code under test.
    monkeypatch.setattr("os.path.exists", lambda _p: devices)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = iter(readings)
    monkeypatch.setattr(measure_mod, "probe_hardware", lambda: next(calls))
    return measure_mod.measure()


def _blank_facts() -> Any:
    from gen_worker.hostfacts import HostFacts

    return HostFacts()


def _real_facts() -> Any:
    from gen_worker.hostfacts import HostFacts

    return HostFacts(gpu_count=1, gpu_name="NVIDIA GeForce RTX 4090",
                     driver_version="550.54", vram_total_bytes=24 << 30)


def test_an_empty_census_beside_gpu_device_nodes_is_UNREADABLE_pgw1414(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE INCIDENT. Device nodes exist, so a card was ASSIGNED to this
    container — the census failing to see it is not evidence of a cardless
    box, and registering cpu-class off it is what cost 703 declines."""
    out = _census(monkeypatch, devices=True,
                  readings=[_blank_facts(), _blank_facts(), _blank_facts()])
    assert out.get("census_unreadable"), out
    assert "may be present and not answering" in out["census_unreadable"]
    assert "must NOT be registered as cpu-class" in out["census_unreadable"]


def test_the_retry_wins_when_the_driver_mount_was_merely_LATE_pgw1414(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timing hypothesis, and the reason the retry is not decoration: a
    cold-start driver mount that lands on the second look leaves NO typed
    state, because there is nothing wrong with this host."""
    out = _census(monkeypatch, devices=True,
                  readings=[_blank_facts(), _real_facts()])
    assert "census_unreadable" not in out, out
    assert out["hardware"]["gpu_name"] == "NVIDIA GeForce RTX 4090"


def test_a_genuinely_CARDLESS_host_stays_quiet_pgw1414(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opposite behaviour, and the half that must NOT regress: no device
    nodes means no card was assigned, which is the any-machine ruling's
    warn-and-serve CPU case. One census attempt, no retry, no typed state."""
    out = _census(monkeypatch, devices=False, readings=[_blank_facts()])
    assert "census_unreadable" not in out, out
    assert not out["hardware"].get("gpu_count")
