"""Child faults: a death below Python is NAMED, reaches the hub, and degrades.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any, List

import grpc
import msgspec
import pytest
from harness.split import (  # noqa: F401  (fixtures re-used)
    SplitHarness,
)
from harness.split import (
    captured_dials as captured_dials,
)
from harness.split import (
    captured_reports as captured_reports,
)
from harness.split import (
    isolated_postmortem as isolated_postmortem,
)

from gen_worker import compile_cache as cc
from gen_worker import hot_swap, postmortem
from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.config import load_settings
from gen_worker.executor import Executor
from gen_worker.hostfacts import HostFacts
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.registry import EndpointSpec
from gen_worker.worker_fatal import REASON_CLASS, build_fatal_detail, report_worker_fatal

# ============================================================================
# gw#640 — gw#640: the supervisor must name a death that happens below
#   Python.
# ============================================================================

_SCRIPT = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    from gen_worker.supervisor import supervise

    supervise(Path(sys.argv[2]))
    # only the child gets here
    mode = sys.argv[1]
    if mode == "term":
        # drain semantics: the parent must forward SIGTERM to the child.
        # Block first, then sigwait. A handler plus signal.pause() loses the
        # wakeup whenever the forwarded signal lands in the gap between the
        # READY announcement and pause() — the child consumes it, then waits
        # forever for a signal that already came, and the parent waits forever
        # in waitpid. sigwait has no such gap: a signal that arrives early is
        # pending, and returns immediately.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        print("READY", flush=True)
        signal.sigwait({signal.SIGTERM})
        os._exit(0)
    if mode == "segv":
        os.kill(os.getpid(), signal.SIGSEGV)
    elif mode == "kill":
        os.kill(os.getpid(), signal.SIGKILL)
    elif mode == "code":
        os._exit(7)
    os._exit(0)
    """
)


def _run(mode: str, tmp_path: Path, *, record: Path | None = None) -> Any:
    script = tmp_path / "boot.py"
    script.write_text(_SCRIPT)
    sink = tmp_path / f"postmortem-{mode}.txt"
    record = record or (tmp_path / f"record-{mode}.json")
    env = dict(os.environ)
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(sink)
    env.pop("GEN_WORKER_SUPERVISED", None)
    env.pop("ORCHESTRATOR_PUBLIC_ADDR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, str(script), mode, str(record)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    return proc, sink, record


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
@pytest.mark.parametrize(
    "mode,signal_name,exit_code",
    [("segv", "SIGSEGV", 139), ("kill", "SIGKILL", 137)],
)
def test_signal_death_is_named(tmp_path, mode, signal_name, exit_code):
    proc, sink, record = _run(mode, tmp_path)
    assert proc.returncode == exit_code
    assert sink.exists(), proc.stderr
    detail = sink.read_text()
    assert f"KILLED BY SIGNAL {signal_name}" in detail
    assert "cgroup_oom_kill_delta=" in detail
    assert "memory.max=" in detail and "memory.current=" in detail
    assert "cpu.max=" in detail and "host_cpu_count=" in detail
    # the record is consumed so the next boot does not re-report this death
    assert not record.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_nonzero_exit_is_reported(tmp_path):
    proc, sink, _ = _run("code", tmp_path)
    assert proc.returncode == 7
    assert sink.exists()
    assert "exited normally code=7" in sink.read_text()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_clean_exit_reports_nothing(tmp_path):
    proc, sink, record = _run("ok", tmp_path)
    assert proc.returncode == 0
    assert not sink.exists()
    assert not record.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_previous_container_death_is_reported_on_next_boot(tmp_path):
    """The whole cgroup can go (memory.oom.group) — then the NEXT boot reports."""
    record = tmp_path / "leftover.json"
    record.write_text(json.dumps({"pid": 4242, "boot_unix": 1.0, "oom_kill_at_boot": 0}))
    proc, sink, _ = _run("ok", tmp_path, record=record)
    assert proc.returncode == 0
    assert sink.exists(), proc.stderr
    detail = sink.read_text()
    assert "previous_container_death" in detail
    assert "4242" in detail
    assert not record.exists()


def test_previous_container_death_aggregates_every_group_marker(
    tmp_path, monkeypatch,
):
    """Per-group files must not erase whole-container post-mortem coverage."""
    from gen_worker import postmortem

    record = tmp_path / "boot.json"
    record.write_text(json.dumps({
        "pid": 4242, "boot_unix": 1.0, "oom_kill_at_boot": 0,
    }))
    monkeypatch.setattr(
        postmortem, "CRASH_REGISTRY_PATH", tmp_path / "crash-streaks.json"
    )
    for ordinal, function in enumerate(("image", "video")):
        group_dir = tmp_path / f"g{ordinal}"
        group_dir.mkdir()
        (group_dir / "gen-worker-inflight.json").write_text(json.dumps({
            "active": [{
                "kind": "request", "function": function,
                "request_id": f"r-{ordinal}", "pid": 100 + ordinal,
            }],
        }))
        (group_dir / "gen-worker-fault-dump.txt").write_text(
            f"fault-tail-g{ordinal}"
        )

    detail = postmortem.previous_boot_detail(record)
    assert detail is not None
    assert all(value in detail for value in (
        '"function": "image"', '"function": "video"',
        "fault-tail-g0", "fault-tail-g1",
    ))
    assert not list(tmp_path.glob("g*/gen-worker-inflight.json"))
    streaks = postmortem.native_crash_streaks()
    assert streaks["image"]["count"] == streaks["video"]["count"] == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_sigterm_is_forwarded_to_the_worker(tmp_path):
    """Drain must still work: PID 1 is the supervisor, the worker is the child."""
    import signal as _signal

    script = tmp_path / "boot.py"
    script.write_text(_SCRIPT)
    sink = tmp_path / "postmortem-term.txt"
    env = dict(os.environ)
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(sink)
    env.pop("GEN_WORKER_SUPERVISED", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen(
        [sys.executable, str(script), "term", str(tmp_path / "rec.json")],
        env=env, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(_signal.SIGTERM)
        rc = proc.wait(timeout=60)
    finally:
        # a timing-out shutdown test must not strand the pair it created
        _kill_tree(proc)
    assert rc == 0
    assert not sink.exists()


def test_container_limits_are_readable():
    from gen_worker import postmortem

    limits = postmortem.container_limits()
    assert "memory_max_bytes" in limits
    assert limits["host_cpu_count"] >= 1
    assert postmortem.effective_cpu_count() >= 1
    assert postmortem.effective_cpu_count() <= (os.cpu_count() or 1)


_WEDGED = textwrap.dedent(
    """
    import os, signal, sys, time
    from pathlib import Path
    from gen_worker.supervisor import supervise

    supervise(Path(sys.argv[2]), stop_timeout_s=float(sys.argv[3]))
    # only the child gets here: a worker that cannot answer its drain
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("READY", flush=True)
    if sys.argv[1] == "stderr_stall":
        # the pgw#833 hazard: stderr is a pipe nobody is draining, so the
        # child blocks in write() with the fd full and never runs anything
        os.write(2, b"x" * (8 << 20))
    while True:
        time.sleep(3600)
    """
)


def _drain_after(stream: Any, delay_s: float) -> Any:
    """gw#640: A consumer that stalls for `delay_s`, then drains."""
    import threading

    def run():
        time.sleep(delay_s)
        try:
            stream.read()
        except Exception:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
@pytest.mark.parametrize("shape", ["deaf", "stderr_stall"])
def test_shutdown_is_bounded_when_the_worker_cannot_answer(tmp_path, shape):
    """gw#640: Forwarding is not draining."""
    import signal as _signal

    grace = 3.0
    script = tmp_path / "boot.py"
    script.write_text(_WEDGED)
    env = dict(os.environ)
    sink = tmp_path / f"postmortem-{shape}.txt"
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(sink)
    env.pop("GEN_WORKER_SUPERVISED", None)
    env.pop("ORCHESTRATOR_PUBLIC_ADDR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen(
        [sys.executable, str(script), shape, str(tmp_path / "rec.json"), str(grace)],
        env=env, stdout=subprocess.PIPE, text=True,
        stderr=subprocess.PIPE if shape == "stderr_stall" else None,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        if shape == "stderr_stall":
            _drain_after(proc.stderr, grace * 2)
        proc.send_signal(_signal.SIGTERM)
        started = time.monotonic()
        try:
            rc = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "the supervisor never completed shutdown — a worker that "
                "cannot answer SIGTERM would keep the pod (and its GPU bill) "
                "alive forever"
            )
        elapsed = time.monotonic() - started
    finally:
        _kill_tree(proc)
    # 137 = the child was SIGKILLed, and the post-mortem says so rather than
    # the death being silent.
    assert rc == 137, f"expected the escalation's SIGKILL verdict, got rc={rc}"
    # ...and the post-mortem NAMES it, so the death is never silent. This is the
    # escalation's observable product; `elapsed < 45` was standing in for it
    # (pgw#845 — a constant says nothing about this run, and the repo's own
    # pgw#795 guard is right to refuse it).
    detail = sink.read_text()
    assert "worker_process_exit" in detail, detail
    assert "SIGKILL" in detail, detail
    # The GRACE is what bounds this, so bound it by the grace this test
    # CONFIGURED: an escalation armed on some other clock would still reach
    # rc=137, just far too late to save the pod's GPU bill.
    assert elapsed < grace * 5, (
        f"shutdown took {elapsed:.1f}s for a {grace:.0f}s grace — the "
        f"escalation is not what ended it"
    )


def test_sigterm_forward_survives_an_inherited_blocked_mask(tmp_path):
    """gw#640: The signal mask survives fork AND exec, so the launcher decides whether the drain contract is del..."""
    launcher = tmp_path / "launcher.py"
    launcher.write_text(textwrap.dedent(
        """
        import os, signal, subprocess, sys
        # a hostile launcher: block SIGTERM before exec'ing boot.py
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        # its own process group, so a red run reaps the whole pair
        proc = subprocess.Popen(
            [sys.executable] + sys.argv[1:], stdout=subprocess.PIPE, text=True,
            start_new_session=True)
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signal.SIGTERM)
        try:
            sys.exit(proc.wait(timeout=30))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.kill()
            sys.exit(99)
        """
    ))
    script = tmp_path / "boot.py"
    script.write_text(_SCRIPT)
    sink = tmp_path / "postmortem-mask.txt"
    env = dict(os.environ)
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(sink)
    env.pop("GEN_WORKER_SUPERVISED", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, str(launcher), str(script), "term", str(tmp_path / "rec.json")],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert not sink.exists()


def _kill_tree(proc: subprocess.Popen) -> None:
    """No test of a shutdown path may itself leak a supervisor pair."""
    try:
        kids = subprocess.run(
            ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True
        ).stdout.split()
    except OSError:
        kids = []
    for kid in kids:
        try:
            os.kill(int(kid), 9)
        except (OSError, ValueError):
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


# ============================================================================
# gw#640 — gw#640/th#1077: a worker fatal must reach the HUB, not just pod
#   stdout.
# ============================================================================

class _FatalCatcher(pb_grpc.WorkerSchedulerServicer):
    def __init__(self) -> None:
        self.reports: list = []
        self.got = threading.Event()

    def Connect(self, request_iterator, context):
        for msg in request_iterator:
            if msg.WhichOneof("msg") == "hardware_unsuitable":
                self.reports.append(msg.hardware_unsuitable)
                self.got.set()
        return
        yield  # pragma: no cover - generator marker


def _server():
    catcher = _FatalCatcher()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_WorkerSchedulerServicer_to_server(catcher, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return catcher, server, port


def test_fatal_detail_carries_class_message_and_traceback() -> None:
    try:
        raise ValueError("pipeline exploded")
    except ValueError as exc:
        detail = build_fatal_detail("runtime", exc, exit_code=1)
    assert "phase=runtime" in detail
    assert "exit_code=1" in detail
    assert "ValueError: pipeline exploded" in detail
    assert "Traceback (most recent call last)" in detail
    assert "test_child_faults.py" in detail


def test_fatal_detail_is_clipped_but_keeps_both_ends() -> None:
    exc = RuntimeError("H" * 200 + "M" * 40_000 + "T" * 200)
    detail = build_fatal_detail("runtime", exc, exit_code=1)
    assert len(detail) < 12_000
    assert "HHH" in detail and "TTT" in detail
    assert "[clipped]" in detail


def test_worker_fatal_reaches_the_hub_over_the_wire() -> None:
    catcher, server, port = _server()
    try:
        settings = load_settings(
            orchestrator_public_addr=f"127.0.0.1:{port}",
            worker_id="worker-gw640",
            worker_jwt="",
        )
        try:
            raise RuntimeError("boom in reconcile")
        except RuntimeError as exc:
            delivered = report_worker_fatal(settings, "runtime", exc, exit_code=1)
        assert catcher.got.wait(10), "hub never received a fatal report"
        report = catcher.reports[0]
        assert report.reason_class == REASON_CLASS
        assert "RuntimeError: boom in reconcile" in report.detail
        assert "Traceback" in report.detail
        assert report.worker_id == "worker-gw640"
        assert delivered is True
    finally:
        server.stop(grace=0)


def test_unexplained_loop_exit_is_a_fatal_not_a_silent_zero() -> None:
    """gw#640's signature: transport.run() returning with no Drain and no signal used to be `return 0` — a silen..."""
    import asyncio

    from gen_worker.worker import UnexpectedWorkerExit, Worker

    worker = Worker.__new__(Worker)  # no real wiring needed for this contract
    worker._stop_requested = False

    class _Lifecycle:
        drained = threading.Event()
        draining = False

        def start_drain(self, deadline_ms):
            pass

        async def startup(self):
            await asyncio.sleep(3600)

    class _Transport:
        connected = False

        async def run(self):
            return  # loop ends on its own — the gw#640 shape

    worker.lifecycle = _Lifecycle()  # type: ignore[assignment]
    worker.transport = _Transport()

    with pytest.raises(UnexpectedWorkerExit) as caught:
        asyncio.run(worker.arun())
    assert "without a Drain command" in str(caught.value)


def test_requested_stop_still_exits_zero() -> None:
    import asyncio

    from gen_worker.worker import Worker

    worker = Worker.__new__(Worker)
    worker._stop_requested = True

    class _Lifecycle:
        drained = threading.Event()
        draining = False

        def start_drain(self, deadline_ms):
            pass

        async def startup(self):
            await asyncio.sleep(3600)

    class _Transport:
        connected = False

        async def run(self):
            return

    worker.lifecycle = _Lifecycle()  # type: ignore[assignment]
    worker.transport = _Transport()
    assert asyncio.run(worker.arun()) == 0


# ============================================================================
# pgw#676 — A native crash (SIGSEGV in a CUDA/C extension) must be NAMED and
#   must not crash-loop the pod.
# ============================================================================

_SEGV_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from gen_worker.supervisor import supervise

    supervise(Path(sys.argv[1]))
    # only the child gets here
    from gen_worker import postmortem
    postmortem.enable_fault_dump()
    postmortem.note_inflight("request", "generate", request_id="req-676")
    import ctypes
    ctypes.string_at(0)  # real native fault: NULL deref -> SIGSEGV
    """
)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")
def test_segv_death_is_attributed_and_carries_frames(tmp_path: Path) -> None:
    script = tmp_path / "boot.py"
    script.write_text(_SEGV_SCRIPT)
    record = tmp_path / "record.json"
    sink = tmp_path / "postmortem.txt"
    env = dict(os.environ)
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(sink)
    env["GEN_WORKER_BOOT_RECORD"] = str(record)  # siblings land in tmp_path
    env.pop("GEN_WORKER_SUPERVISED", None)
    env.pop("ORCHESTRATOR_PUBLIC_ADDR", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [sys.executable, str(script), str(record)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 139, proc.stderr
    detail = sink.read_text()
    assert "KILLED BY SIGNAL SIGSEGV" in detail
    # the in-flight marker attributes the death to the executing function
    assert '"function": "generate"' in detail
    assert "req-676" in detail
    # the faulthandler dump gives exit 139 actual Python frames
    assert "fault_dump_tail" in detail
    assert "boot.py" in detail  # the dying frame is visible
    # the crash registry on the "pod" fs recorded the streak
    streaks = json.loads((tmp_path / "gen-worker-crash-streaks.json").read_text())
    assert streaks["generate"]["count"] == 1
    assert streaks["generate"]["last_signal"] == "SIGSEGV"
    # marker consumed: the next death cannot re-attribute this one
    assert not (tmp_path / "gen-worker-inflight.json").exists()


class _In(msgspec.Struct):
    prompt: str = ""


class _Fake:
    def setup(self, pipeline: Any) -> None:  # pragma: no cover
        self.pipeline = pipeline

    def generate(self, ctx: Any, payload: _In) -> dict:  # pragma: no cover
        return {}

    def generate_turbo(self, ctx: Any, payload: _In) -> dict:  # pragma: no cover
        return {}


def _specs() -> List[EndpointSpec]:
    return [
        EndpointSpec(
            name="generate", method=_Fake.generate, kind="inference",
            payload_type=_In, output_mode="single", cls=_Fake,
            models={"pipeline": Hub("acme/sdxl")},
            resources=Resources(gpu=True),
        ),
        EndpointSpec(
            name="generate-turbo", method=_Fake.generate_turbo,
            kind="inference", payload_type=_In, output_mode="single",
            cls=_Fake, models={"pipeline": Hub("acme/sdxl")},
            resources=Resources(gpu=True),
        ),
    ]


def _executor() -> Executor:
    async def _send(msg: pb.WorkerMessage) -> None:  # pragma: no cover
        pass

    return Executor(_specs(), _send)


_GPU = HostFacts(vram_total_bytes=20 * 1024**3, vram_free_bytes=20 * 1024**3,
                 gpu_sm="86")


def test_crash_streak_gates_only_the_crashing_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "streaks.json"
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)

    ex = _executor()
    # One crash = one free retry; nothing is gated yet.
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV")
    ex.gate_functions(_GPU)
    assert "generate" not in ex.unavailable

    # The second signal death trips the refusal — for that function only.
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV")
    ex.gate_functions(_GPU)
    code, detail, axes = ex.unavailable["generate"]
    assert code == "native_crash_streak"
    assert "SIGSEGV" in detail and "siblings keep serving" in detail
    assert axes["streak"] == "2"
    assert "generate-turbo" not in ex.unavailable, (
        "the turbo sibling completed on the same A4500 workers live — it "
        "must keep serving"
    )
    # Idempotent re-gate: the mark is gate-owned, re-derived every pass.
    ex.gate_functions(_GPU)
    assert ex.unavailable["generate"][0] == "native_crash_streak"


def test_inflight_markers_stack_and_clear_by_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "inflight.json"
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", path)
    postmortem.clear_inflight()  # process-global hygiene for the test rig

    t_req = postmortem.note_inflight("request", "generate", request_id="r1")
    t_warm = postmortem.note_inflight("warmup", "generate-turbo")
    active = json.loads(path.read_text())["active"]
    assert {r["function"] for r in active} == {"generate", "generate-turbo"}

    postmortem.clear_inflight(t_warm)
    active = json.loads(path.read_text())["active"]
    assert [r["function"] for r in active] == ["generate"]

    postmortem.clear_inflight(t_req)
    assert not path.exists()

    # take_inflight consumes whatever a dead process left
    postmortem.note_inflight("request", "generate", request_id="r2")
    left = postmortem.take_inflight(path)
    assert [r["request_id"] for r in left] == ["r2"]
    assert not path.exists()
    postmortem.clear_inflight()


def test_streak_counts_distinct_requests_not_attempts_of_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#763 stage 4, measured on a live pod: the hub's blame ladder re-ran ONE deterministically fatal payloa..."""
    registry = tmp_path / "streaks.json"
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)
    ex = _executor()

    # The live shape: attempts 1, 2 and 3 of request r-oom, same pod.
    for _ in range(3):
        postmortem.record_native_crash(
            "generate", kind="request", signal_name="SIGKILL", request_id="r-oom")
    ex.gate_functions(_GPU)
    assert "generate" not in ex.unavailable, (
        "one request's retry ladder refused the function and would condemn the "
        "pod: " + json.dumps(postmortem.native_crash_streaks(registry))
    )
    assert postmortem.native_crash_streaks(registry)["generate"]["count"] == 1

    # The guard rail: a DIFFERENT request crashing the same function is the
    # real signal, and it still trips exactly as before.
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV", request_id="r-other")
    ex.gate_functions(_GPU)
    code, _detail, axes = ex.unavailable["generate"]
    assert code == "native_crash_streak" and axes["streak"] == "2"


def test_compile_deaths_still_count_every_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#676: A background compile carries no request id, so distinct-request counting must not silently disar..."""
    registry = tmp_path / "streaks.json"
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)
    marker = postmortem.compile_marker("unet")
    for expected in (1, 2, 3):
        got = postmortem.record_native_crash(
            marker, kind=postmortem.COMPILE_KIND, signal_name="SIGSEGV")
        assert got == expected


# ============================================================================
# pgw#833 — pgw#833: a pre-Hello compute-child death must carry its OWN
#   crash text.
# ============================================================================

MARKER = "PGW833_THE_CHILD_SAYS_WHY_IT_DIED"


def test_pre_hello_death_dial_carries_the_childs_stderr(
    tmp_path, captured_dials, capfd,
):
    """pgw#833: The load-bearing row (RED without the fix): a child that dies before Hello with a distinctive st..."""
    code = (
        "import sys;"
        f"print('{MARKER}: BootExplosion: the real reason', file=sys.stderr);"
        "sys.exit(1)"
    )
    h = SplitHarness(tmp_path, child_cmd=[sys.executable, "-c", code])
    try:
        exit_code = h.wait_exit(120.0)
        assert exit_code == 1, f"expected the bounded boot-loop exit 1, got {exit_code}"

        exits = [d for d in captured_dials if "phase=compute_process_exit" in d]
        assert exits, "no compute_process_exit dial was made"
        assert all("child_stderr_tail" in d and MARKER in d for d in exits), (
            "a pre-Hello death dial does not carry the child's stderr tail — "
            "the pod is undiagnosable again (pgw#833):\n" + exits[-1]
        )
        giveups = [d for d in captured_dials if "compute_boot_crash_loop" in d]
        assert giveups and any(MARKER in d for d in giveups), (
            "the give-up dial should name the child's last stderr"
        )
    finally:
        h.close()
    # (3) the tee: every byte still reaches the parent's stderr = container log.
    assert MARKER in capfd.readouterr().err


def test_boot_fatal_ack_round_trips_before_the_child_exits(
    tmp_path, captured_dials, captured_reports,
):
    """The pgw#826 follow-on race, closed deterministically: the child spies on the ack wait and exits 7 unless ..."""
    script = tmp_path / "ack_probe_child.py"
    script.write_text(
        "import sys\n"
        "from gen_worker.procsplit import child as c\n"
        "orig = c._wait_boot_fatal_ack\n"
        "seen = {}\n"
        "def spy(sock):\n"
        "    orig(sock)\n"
        "    seen['ok'] = True\n"
        "c._wait_boot_fatal_ack = spy\n"
        "c.send_boot_fatal({'reason_class': 'cuda_unavailable',"
        " 'detail': 'ack race probe'})\n"
        "sys.exit(1 if seen.get('ok') else 7)\n"
    )
    h = SplitHarness(tmp_path, child_cmd=[sys.executable, str(script)])
    try:
        exit_code = h.wait_exit(120.0)
        assert exit_code == 1
        assert h.pc._spawn_count == 1, "a terminal verdict must never respawn"
        assert h.pc.terminal_exit_reason == "boot_fatal:cuda_unavailable"
        # The child exited 1, which per the script means the ack ARRIVED
        # before it exited (exit 7 = frame sent but never acknowledged).
        assert any('"cause": "exit:1"' in d for d in captured_dials), (
            "child exited without seeing the ack: " + repr(captured_dials)
        )
        assert len(captured_reports) == 1
        assert captured_reports[0].reason_class == "cuda_unavailable"
    finally:
        h.close()


def test_the_grpc_fork_abort_names_itself_instead_of_being_rediagnosed():
    """pgw#932 has been diagnosed from first principles at least five times, because ``cause=signal:SIGABRT`` na..."""
    from gen_worker.procsplit.parent import is_grpc_fork_abort

    sighting = (
        "I0803 03:09:46.724278 11992 fork_posix.cc:71] Other threads are "
        "currently calling into gRPC, skipping fork() handlers\n"
        "E0803 03:09:46.728102 12039 ev_epoll1_linux.cc:373] (event_engine) "
        "Epoll1Poller:0x278b97e0 encountered epoll_wait error: Bad file "
        "descriptor\n"
    )
    poller_only = (
        "ev_epoll1_linux.cc:373 (event_engine) Epoll1Poller encountered "
        "epoll_wait error: Bad file descriptor"
    )
    def abort(*, saw_hello: bool = False, oom_delta: int = 0,
              cause: str = "signal:SIGABRT", tail: str = "") -> bool:
        return is_grpc_fork_abort(cause=cause, saw_hello=saw_hello,
                                  oom_delta=oom_delta, stderr_tail=tail)

    assert abort(tail=sighting)
    # The 2026-08-17 master red carried only the poller half of the tail.
    assert abort(tail=poller_only)

    # ...and every neighbouring shape stays UNEXPLAINED, by name:
    assert not abort(tail=sighting, saw_hello=True), (
        "a post-Hello abort is the tenant's process dying, not the launcher's")
    assert not abort(tail=sighting, oom_delta=2), (
        "an OOM kill has an owner and a cost; it must never read as 'rerun it'")
    assert not abort(tail=sighting, cause="signal:SIGSEGV"), (
        "a SIGSEGV is the pgw#676 class, which is a real serving defect")
    assert not abort(tail="free(): invalid pointer")
    assert not abort(tail="")


# ============================================================================
# pgw#714 — pgw#714: background-compile crashes tell the truth and degrade
#   to eager.
# ============================================================================

def _gate(kind: str) -> "contextlib.AbstractContextManager[None]":
    """pgw#714: `Executor._wire_turn_gate`'s factory, as a test double."""
    return contextlib.nullcontext()


def _router(*, fail_closed: bool = False) -> "hot_swap.Router":
    router = hot_swap.Router(fail_closed=fail_closed)
    router.set_turn_gate(_gate)
    return router


@pytest.fixture()
def tmp_postmortem(tmp_path, monkeypatch):
    inflight = tmp_path / "inflight.json"
    registry = tmp_path / "crash_registry.json"
    monkeypatch.setattr(postmortem, "INFLIGHT_PATH", inflight)
    monkeypatch.setattr(postmortem, "CRASH_REGISTRY_PATH", registry)
    postmortem.clear_inflight(path=inflight)
    with postmortem._inflight_lock:
        postmortem._inflight_active.clear()
    yield inflight, registry
    with postmortem._inflight_lock:
        postmortem._inflight_active.clear()


def test_compile_marker_takes_the_blame(tmp_postmortem):
    inflight, registry = tmp_postmortem
    postmortem.note_inflight(
        "request", "generate", request_id="req-1", path=inflight)
    postmortem.note_inflight(
        postmortem.COMPILE_KIND, postmortem.compile_marker("transformer"),
        path=inflight)

    extra = postmortem.attribute_signal_death(
        signal_name="SIGSEGV", inflight_path=inflight,
        registry_path=registry, dump_path=inflight.with_name("no.dump"))

    streaks = extra["native_crash_streaks"]
    assert streaks == {"compile:transformer": 1}
    rows = postmortem.native_crash_streaks(registry)
    assert "generate" not in rows
    assert rows["compile:transformer"]["last_kind"] == "compile"
    # Both executions still appear as evidence.
    fns = {r["function"] for r in extra["inflight"]}
    assert fns == {"generate", "compile:transformer"}


def test_no_compile_marker_keeps_request_attribution(tmp_postmortem):
    inflight, registry = tmp_postmortem
    postmortem.note_inflight(
        "request", "generate", request_id="req-1", path=inflight)
    extra = postmortem.attribute_signal_death(
        signal_name="SIGSEGV", inflight_path=inflight,
        registry_path=registry, dump_path=inflight.with_name("no.dump"))
    assert extra["native_crash_streaks"] == {"generate": 1}


def test_compile_crash_rows_selects_only_compiles(tmp_postmortem):
    _, registry = tmp_postmortem
    postmortem.record_native_crash(
        "generate", kind="request", signal_name="SIGSEGV", path=registry)
    postmortem.record_native_crash(
        postmortem.compile_marker("transformer"),
        kind=postmortem.COMPILE_KIND, signal_name="SIGSEGV", path=registry)
    rows = postmortem.compile_crash_rows(registry)
    assert set(rows) == {"compile:transformer"}


def test_warm_thread_marks_and_clears_the_compile_inflight(tmp_postmortem):
    inflight, registry = tmp_postmortem
    seen: dict = {}

    def compiled() -> None:
        seen["active"] = json.loads(inflight.read_text())["active"]
        raise RuntimeError("boom (contained per-signature)")

    router = _router()
    job = hot_swap._WarmJob(
        router=router, label="unet", sig=("s",), compiled=compiled,
        args=(), kwargs={}, device=None, grad_mode="no_grad",
        autocast_dtype=None, turn=_gate)
    router.pending.add(job.sig)
    hot_swap._run_warm(job)

    assert [r["function"] for r in seen["active"]] == ["compile:unet"]
    assert seen["active"][0]["kind"] == postmortem.COMPILE_KIND
    # Failure contained; marker cleared; signature stays eager.
    assert postmortem.take_inflight(inflight) == []
    assert job.sig in router.bg_failed


@pytest.fixture()
def reset_compile_disable():
    yield
    cc._PROCESS_COMPILES_DISABLED = ""


def test_process_disable_makes_apply_a_noop(reset_compile_disable):
    cc.disable_process_compiles("1 process signal death(s) during compile")
    assert cc._PROCESS_COMPILES_DISABLED

    class Pipe:  # never armed: apply must return before touching torch
        pass

    assert cc.apply(Pipe(), None, cache_ready=True) is False


def test_operator_eager_pin_suppresses_arming(reset_compile_disable):
    class Pipe:
        pass

    pinned = Pipe()
    setattr(pinned, cc.EXECUTION_LANE_ATTR, "bf16-w16a16+eager")
    setattr(pinned, cc.EXECUTION_LANE_PINNED_ATTR, True)
    assert cc.operator_eager_pin(pinned) is True
    assert cc.apply(pinned, None, cache_ready=True) is False

    # The same lane WITHOUT pin provenance is not a kill switch.
    auto = Pipe()
    setattr(auto, cc.EXECUTION_LANE_ATTR, "bf16-w16a16+eager")
    assert cc.operator_eager_pin(auto) is False

    # A pinned COMPILED lane never suppresses.
    comp = Pipe()
    setattr(comp, cc.EXECUTION_LANE_ATTR, "fp8-w8a8-dynamic+compiled")
    setattr(comp, cc.EXECUTION_LANE_PINNED_ATTR, True)
    assert cc.operator_eager_pin(comp) is False


def test_setup_window_carries_pin_provenance(reset_compile_disable):
    class Pipe:
        pass

    execution_lane_tok = cc._SETUP_EXEC_EXECUTION_LANE.set("bf16-w16a16+eager")
    pin_tok = cc._SETUP_EXEC_EXECUTION_LANE_PINNED.set(True)
    try:
        pipe = Pipe()
        assert cc.operator_eager_pin(pipe) is True
        # Stamped through, like the lane itself.
        assert getattr(pipe, cc.EXECUTION_LANE_PINNED_ATTR) is True
    finally:
        cc._SETUP_EXEC_EXECUTION_LANE_PINNED.reset(pin_tok)
        cc._SETUP_EXEC_EXECUTION_LANE.reset(execution_lane_tok)
