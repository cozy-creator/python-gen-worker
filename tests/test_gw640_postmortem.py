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
        # drain semantics: the parent must forward SIGTERM to the child
        got = []
        signal.signal(signal.SIGTERM, lambda *_: got.append(1))
        print("READY", flush=True)
        signal.pause()
        os._exit(0 if got else 3)
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
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    proc.send_signal(_signal.SIGTERM)
    assert proc.wait(timeout=60) == 0
    assert not sink.exists()


def test_container_limits_are_readable():
    from gen_worker import postmortem

    limits = postmortem.container_limits()
    assert "memory_max_bytes" in limits
    assert limits["host_cpu_count"] >= 1
    assert postmortem.effective_cpu_count() >= 1
    assert postmortem.effective_cpu_count() <= (os.cpu_count() or 1)


def test_describe_exit_decodes_signals():
    from gen_worker import postmortem

    signaled = postmortem.describe_exit(os.WTERMSIG(9) if False else 9)
    assert signaled["signaled"] is True
    assert signaled["signal_name"] == "SIGKILL"
    assert signaled["exit_code"] == 137
    exited = postmortem.describe_exit(3 << 8)
    assert exited["signaled"] is False
    assert exited["exit_code"] == 3
