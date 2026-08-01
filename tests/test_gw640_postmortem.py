"""gw#640: the supervisor must name a death that happens below Python.

Real forks, real signals, real `waitpid` — the class of death that produced
six silent restarts on the th#1085 cold-boot gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

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


def _run(mode: str, tmp_path: Path, *, record: Path | None = None):
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


def _drain_after(stream, delay_s: float):
    """A consumer that stalls for `delay_s`, then drains. Models a throttled
    container-log collector: the pipe is full while the drain is decided."""
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
    """Forwarding is not draining.

    A child that never answers SIGTERM — deaf, wedged below Python, or
    blocked writing into a stderr pipe with no reader — leaves the supervisor
    in waitpid forever. On a pod that is PID 1 refusing to exit, and a rented
    GPU that keeps billing. The supervisor must therefore bound its own
    shutdown: escalate to SIGKILL when the grace runs out.

    RED without the escalation: both shapes hang until the wait below expires.
    """
    import signal as _signal

    grace = 3.0
    script = tmp_path / "boot.py"
    script.write_text(_WEDGED)
    env = dict(os.environ)
    env["GEN_WORKER_POSTMORTEM_FILE"] = str(tmp_path / f"postmortem-{shape}.txt")
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
    assert elapsed < 45, f"shutdown took {elapsed:.1f}s for a {grace:.0f}s grace"


def test_sigterm_forward_survives_an_inherited_blocked_mask(tmp_path):
    """The signal mask survives fork AND exec, so the launcher decides whether
    the drain contract is deliverable at all — unless supervise() takes it
    back. Red before that: the pair is alive-but-deaf and both sides wait
    forever. (Mechanism found by the 0.90.0 cut lane.)"""
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


def test_describe_exit_decodes_signals():
    from gen_worker import postmortem

    signaled = postmortem.describe_exit(os.WTERMSIG(9) if False else 9)
    assert signaled["signaled"] is True
    assert signaled["signal_name"] == "SIGKILL"
    assert signaled["exit_code"] == 137
    exited = postmortem.describe_exit(3 << 8)
    assert exited["signaled"] is False
    assert exited["exit_code"] == 3
