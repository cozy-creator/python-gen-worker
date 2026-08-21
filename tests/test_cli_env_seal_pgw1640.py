"""`gen-worker up` serves under the platform's DECLARED env, like the pod does.

pgw#1639 measured the divergence on hardware: the pod entrypoint imposes
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `gen-worker up` imposed
nothing, so at a 0.55 GiB budget the same tree served in ~40 s through one front
door and returned a typed OOM refusal in ~6 s through the other.

Both assertions here are read from the DAEMON PROCESS THAT SERVES — its
`/proc/<pid>/environ` (the environment it was exec'd with) and its own read-back
of `os.environ` published in the handle — never from "the imposition function was
called", which is a guard that cannot go red.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gen_worker.cli import endpoint_state
from gen_worker.settings_authority import DECLARED_ENV

ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"

MAIN = '''
"""A weightless fixture endpoint: no model slot, so `up` boots it on any box."""

from __future__ import annotations

import os

import msgspec

from gen_worker import RequestContext, entrypoint


class Ping(msgspec.Struct):
    prompt: str = ""


class Pong(msgspec.Struct):
    alloc_conf: str


@entrypoint
def report(ctx: RequestContext, payload: Ping) -> Pong:
    return Pong(alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
'''


def _endpoint(root: Path) -> Path:
    package = root / "src" / "envseal_fixture"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "main.py").write_text(MAIN, encoding="utf-8")
    (root / "endpoint.toml").write_text(
        'schema_version = 1\nmain = "envseal_fixture.main"\n', encoding="utf-8"
    )
    return root


def _scrubbed_env(state_root: Path) -> dict[str, str]:
    """The environment a normal shell hands the CLI: no declared settings in it."""
    env: dict[str, str] = dict(os.environ)
    for name in DECLARED_ENV:
        env.pop(name, None)
    env["COZY_ENDPOINT_STATE"] = str(state_root)
    return env


def test_the_cli_imposes_the_declared_env_before_anything_can_import_torch(
    tmp_path: Path,
) -> None:
    """The foreground `up` seal: importing the CLI package is the imposition."""
    probe = (
        "import os, sys\n"
        "assert 'torch' not in sys.modules\n"
        "import gen_worker.cli\n"
        "assert 'torch' not in sys.modules, 'the CLI dragged torch in at import'\n"
        f"print(os.environ.get({ALLOC_CONF!r}, ''))\n"
    )
    said = subprocess.run(
        [sys.executable, "-c", probe],
        env=_scrubbed_env(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert said.stdout.strip() == DECLARED_ENV[ALLOC_CONF], (
        "importing gen_worker.cli must impose the declared env — and must do "
        "it before torch is importable, because the CUDA caching allocator "
        f"reads {ALLOC_CONF} once, at init. stderr: {said.stderr}"
    )


def test_the_detached_daemon_serves_with_the_declared_allocator(
    tmp_path: Path,
) -> None:
    endpoint = _endpoint(tmp_path / "endpoint")
    # NOT tmp_path: the daemon's unix socket lives under the state root and
    # AF_UNIX paths cap at 107 bytes, which pytest's xdist tmp dirs exceed.
    state_root = Path(tempfile.mkdtemp(prefix="gw1640-"))
    env = _scrubbed_env(state_root)

    up = subprocess.run(
        [sys.executable, "-m", "gen_worker.cli", "up", str(endpoint), "-d"],
        cwd=str(endpoint),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    handle = state_root / endpoint_state.endpoint_key(endpoint)
    log = (handle / endpoint_state.LOG_NAME)
    log_text = log.read_text(encoding="utf-8") if log.exists() else ""
    assert up.returncode == 0, (
        f"`gen-worker up -d` failed:\n{up.stdout}\n{up.stderr}\n--- daemon log:\n"
        f"{log_text}"
    )

    document = json.loads((handle / endpoint_state.HANDLE_NAME).read_text())
    pid = int(document["pid"])
    try:
        environ = _proc_environ(pid)

        # 1. The environment the SERVING process was exec'd with.
        assert environ.get(ALLOC_CONF) == DECLARED_ENV[ALLOC_CONF], (
            "the detached daemon was launched WITHOUT the declared allocator "
            "config: a pod and `gen-worker up` would serve the same checkpoint "
            f"on two different allocators (pgw#1639/#1640). /proc/{pid}/environ "
            f"has {environ.get(ALLOC_CONF)!r}"
        )

        # 2. The serving process's own read-back, confessed on the handle.
        assert document["declared_env"][ALLOC_CONF] == DECLARED_ENV[ALLOC_CONF]
        assert document["declared_env"] == dict(DECLARED_ENV), (
            "the daemon must confess EVERY declared name it is serving under, "
            "read from its own os.environ"
        )
        assert "declared env in effect" in log_text
        assert f"{ALLOC_CONF}={DECLARED_ENV[ALLOC_CONF]}" in log_text
    finally:
        subprocess.run(
            [sys.executable, "-m", "gen_worker.cli", "down", str(endpoint)],
            cwd=str(endpoint), env=env, capture_output=True, text=True,
            check=False,
        )
        _await_exit(pid)
        shutil.rmtree(state_root, ignore_errors=True)


def _proc_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    out: dict[str, str] = {}
    for entry in raw.split(b"\0"):
        if b"=" in entry:
            name, _, value = entry.partition(b"=")
            out[name.decode()] = value.decode()
    return out


def _await_exit(pid: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not endpoint_state.pid_alive(pid):
            return
        time.sleep(0.05)
    os.kill(pid, 9)
