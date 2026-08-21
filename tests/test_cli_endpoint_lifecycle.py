"""The `up`/`down`/`run` handle: liveness, staleness, and the readiness wait."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from gen_worker.cli import endpoint_state


def test_an_exited_child_ends_the_readiness_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoint_state, "state_root", lambda: tmp_path)
    handle = endpoint_state.handle_for(tmp_path / "ep")

    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    with pytest.raises(endpoint_state.EndpointStateError) as caught:
        endpoint_state.wait_for_handle(
            handle, still_running=lambda: child.poll() is None, poll_s=0.01
        )
    assert "exited during boot" in str(caught.value)


def test_signalling_a_pid_cannot_answer_liveness_for_your_own_child():
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    while child.returncode is None and not _exited(child.pid):
        pass
    assert endpoint_state.pid_alive(child.pid) is True
    assert child.poll() is not None
    assert endpoint_state.pid_alive(child.pid) is False


def _exited(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def test_a_handle_whose_process_is_gone_reads_as_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoint_state, "state_root", lambda: tmp_path)
    handle = endpoint_state.handle_for(tmp_path / "ep")
    endpoint_state.write_handle(handle, {"state": "ready", "pid": 0x7FFFFFFF})

    assert endpoint_state.read_handle(handle) is None
    assert not handle.handle_path.exists()


def test_a_live_handle_reads_back(tmp_path, monkeypatch):
    monkeypatch.setattr(endpoint_state, "state_root", lambda: tmp_path)
    handle = endpoint_state.handle_for(tmp_path / "ep")
    endpoint_state.write_handle(
        handle, {"state": "ready", "pid": os.getpid(), "functions": ["generate"]}
    )
    document = endpoint_state.read_handle(handle)
    assert document is not None and document["functions"] == ["generate"]


def test_a_handle_from_a_future_version_refuses_rather_than_guessing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(endpoint_state, "state_root", lambda: tmp_path)
    handle = endpoint_state.handle_for(tmp_path / "ep")
    handle.state_dir.mkdir(parents=True, exist_ok=True)
    handle.handle_path.write_text(
        json.dumps({"handle_version": 99, "state": "ready", "pid": os.getpid()}),
        encoding="utf-8",
    )
    with pytest.raises(endpoint_state.EndpointStateError):
        endpoint_state.read_handle(handle)
