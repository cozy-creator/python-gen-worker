"""pgw#858 / th#1380: the compute child runs as an unprivileged uid.

**Why this file hosts its own root.** A privilege drop cannot be measured by a
suite running as uid 1000: every ``/proc`` read the fix is supposed to deny is
*already* denied, so every row would pass green on a tree with the fix reverted.
That is the exact shape of a guard that cannot fail. So when this file is not
root it re-executes itself inside a container — repo and interpreter
bind-mounted, ``--init`` so PID 1 is a root process carrying a ``RUNPOD_API_KEY``
exactly as a RunPod pod does — and asserts the inner run. Nothing here needs a
GPU, a pod, or a paid resource.

**Why every row carries its own red control.** The same probe is run a second
time through a parent whose drop has been removed, and it must SUCCEED. So each
row proves on every run both that the boundary holds and that the measurement
can see it not holding — rather than a revert-and-look-once done by hand.

The attacks are run by REAL endpoint handlers in a REAL compute child, because
that is the threat model: tenant code is imported into that process.

Run: uv run pytest tests/test_pod_privilege_isolation_pgw858.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import msgspec
import pytest

from gen_worker.pb import worker_scheduler_pb2 as pb

TESTS_DIR = Path(__file__).resolve().parent
REPO = TESTS_DIR.parent

# The two credentials th#1380 measured, given sentinel values so a leak is
# unmistakable in an assertion message.
RUNPOD_KEY = "rpa_PGW858_ACCOUNT_AUTHORITY_SENTINEL"
PUBLIC_KEY = "ssh-ed25519 AAAAPGW858OPERATORKEYSENTINEL operator@cozy"
WORKER_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3LTg1OCJ9.pgw858-sentinel"

IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


# ==========================================================================
# The outer half: get root, honestly, without a pod
# ==========================================================================


@pytest.mark.skipif(IS_ROOT, reason="already root — the rows below run directly")
def test_privilege_isolation_rows_under_a_real_root_parent():
    """Run every row below inside a container whose PID 1 is root and carries
    the RunPod key, which is the only faithful stand-in for a pod.

    A failure here is a failure of one of the rows; the inner output is
    relayed verbatim."""
    if shutil.which("docker") is None:
        pytest.skip(
            "docker is required: pgw#858 is a uid boundary and a non-root "
            "runner cannot observe it (every /proc read is already denied)"
        )
    interpreter = Path(sys.executable).resolve()
    # uv keeps the real interpreter outside the venv; mount whichever prefix
    # actually holds it so the container runs THIS environment, not a rebuild.
    py_root = interpreter.parents[1]
    cmd = [
        "docker", "run", "--rm", "--init",
        # This suite kills compute children on purpose. Without this the
        # container's core_pattern drops a multi-GB root-owned core into the
        # bind-mounted worktree, which no non-root developer can then delete.
        "--ulimit", "core=0",
        "-e", f"RUNPOD_API_KEY={RUNPOD_KEY}",
        "-e", f"PUBLIC_KEY={PUBLIC_KEY}",
        "-e", f"WORKER_JWT={WORKER_JWT}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "HOME=/root",
        "-v", f"{REPO}:{REPO}",
        "-v", f"{py_root}:{py_root}:ro",
        "-w", str(REPO),
        "ubuntu:24.04",
        str(REPO / ".venv/bin/python"), "-m", "pytest",
        str(Path("tests") / Path(__file__).name),
        "-q", "-p", "no:cacheprovider", "--no-header",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    # The container failing to START is not the rows failing, and conflating
    # them is how this test read as a pgw#858 regression on GitHub CI while
    # passing on every developer box. Measured there: exit 127 with
    #   [FATAL tini (7)] exec .../.venv/bin/python failed: No such file or directory
    # — `ubuntu:24.04` cannot exec the runner's uv-managed interpreter (the
    # venv's python is a symlink into a prefix whose loader the image does not
    # have). No row ran, so there is nothing to report about the uid boundary.
    # Skip by NAME rather than assert: this is the same distinction the file's
    # own `shutil.which("docker")` guard already makes, one layer deeper.
    if proc.returncode == 127 and "exec" in proc.stderr and "failed" in proc.stderr:
        pytest.skip(
            "the container cannot exec this interpreter, so no row ran: "
            f"{proc.stderr.strip().splitlines()[-1][:200]} — pgw#858 needs an "
            "image that can run the host's python, not a verdict from a "
            "container that never started"
        )
    assert proc.returncode == 0, (
        "the pgw#858 rows failed under a real root parent:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    # The inner run must actually have RUN the rows — a skip that reports 0 is
    # the failure mode this whole file exists to avoid.
    assert " passed" in proc.stdout, (
        f"the inner run executed no rows:\n{proc.stdout}\n{proc.stderr}"
    )


# ==========================================================================
# The rows themselves — root parent, unprivileged compute child.
#
# Skipped rather than absent when not root: the run says out loud that the
# boundary was NOT measured here, instead of a green tick that means nothing.
# ==========================================================================

root_only = pytest.mark.skipif(
    not IS_ROOT,
    reason="pgw#858 is a uid boundary — the containerised row above measures it",
)

from harness.hub_double import is_ready, is_result_for  # noqa: E402
from test_procsplit_pgw763 import (  # noqa: E402,F401 — fixtures come with it
    BOOT_TIMEOUT_S,
    SplitHarness,
    _payload,
    captured_dials,
    isolated_postmortem,
)

from gen_worker.procsplit import privdrop  # noqa: E402


@pytest.fixture()
def pod_shaped_env(monkeypatch):
    """The parent's own environment, as a pod delivers it."""
    monkeypatch.setenv("RUNPOD_API_KEY", RUNPOD_KEY)
    monkeypatch.setenv("PUBLIC_KEY", PUBLIC_KEY)
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)


def _split(tmp_path, *, drop: bool, monkeypatch=None) -> SplitHarness:
    """A real split whose child imports the pgw#858 probe handlers.

    ``drop=False`` is the RED control: the identical harness with the privilege
    drop removed, i.e. the tree as it was before this issue."""
    if not drop:
        monkeypatch.setattr(privdrop, "plan_drop", lambda home: None)
    return SplitHarness(
        tmp_path,
        extra_child_env={"PGW763_CHILD_MODULES": "harness.privdrop_endpoints"},
    )


def _probe(h: SplitHarness, fn: str, text: str = "") -> str:
    conn = h.scheduler.wait_connection(0, timeout=BOOT_TIMEOUT_S)
    conn.wait_for(is_ready, timeout=BOOT_TIMEOUT_S)
    rid = f"r-{fn}-{time.time_ns()}"
    conn.send(run_job=pb.RunJob(
        request_id=rid, attempt=1, function_name=fn, input_payload=_payload(text)))
    got = conn.wait_for(is_result_for(rid), timeout=120.0)
    assert got.job_result.status == pb.JOB_STATUS_OK, got.job_result.safe_message
    return msgspec.msgpack.decode(got.job_result.inline)["response"]


def _json_probe(h: SplitHarness, fn: str, text: str = "") -> Dict[str, Any]:
    return json.loads(_probe(h, fn, text))


@pytest.fixture()
def dropped(tmp_path, captured_dials, pod_shaped_env):
    h = _split(tmp_path, drop=True)
    try:
        yield h
    finally:
        h.close()


@pytest.fixture()
def undropped(tmp_path, captured_dials, monkeypatch, pod_shaped_env):
    """The pre-pgw#858 tree, alive in the same run, so every denial below is
    measured against a control that proves the probe works."""
    h = _split(tmp_path / "red", drop=False, monkeypatch=monkeypatch)
    try:
        yield h
    finally:
        h.close()


# ---- the boundary itself -------------------------------------------------


@root_only
def test_child_execs_as_an_unprivileged_uid_and_cannot_climb_back(dropped, undropped):
    got = _json_probe(dropped, "report-identity")
    assert got["uid"] != 0 and got["euid"] != 0, (
        f"the compute child is still root: {got} — tenant code runs here "
        "(pgw#858)"
    )
    assert got["uid"] == got["euid"], f"a partial drop leaves a way back: {got}"
    assert got["regained_root"] is False, (
        f"the drop is reversible — the child got uid 0 back: {got}"
    )
    assert 0 not in got["groups"], (
        f"the child kept root's group membership: {got['groups']}"
    )

    red = _json_probe(undropped, "report-identity")
    assert red["uid"] == 0, (
        "the RED control did not reproduce the defect — with plan_drop() "
        f"disabled the child must still be root, got {red}"
    )


@root_only
def test_the_child_cannot_climb_back_through_a_setuid_binary(dropped, undropped):
    """#858's "confirm no setuid escalation path in the base images" — answered
    by imposing the property rather than by auditing images we do not build."""
    got = _json_probe(dropped, "escalation-surface")
    assert got["no_new_privs"] == "1", (
        f"NoNewPrivs is not set on the compute child: {got} — a stock base "
        f"image ships {got['setuid_binaries_present']}"
    )
    red = _json_probe(undropped, "escalation-surface")
    assert red["no_new_privs"] == "0", (
        f"the RED control did not reproduce the defect: {red}"
    )


@root_only
def test_child_cannot_read_pid1_environ_where_the_runpod_key_lives(dropped, undropped):
    """th#1380 D1: RunPod injects an account-authority key into every pod and
    the create call cannot suppress it. PID 1's environ is where it sits."""
    # Positive control on the FIXTURE: there is genuinely something to steal.
    pid1 = Path("/proc/1/environ").read_bytes().decode("utf-8", "replace")
    assert RUNPOD_KEY in pid1, (
        "this container's PID 1 does not carry the RunPod key, so the row "
        "below would pass for the wrong reason"
    )

    got = _json_probe(dropped, "steal-pid1-environ")
    assert got["outcome"] == "denied", (
        f"tenant code read PID 1's environment: {got} — the RunPod "
        "account-authority key is exposed (pgw#858 / th#1380 D1+D2)"
    )

    red = _json_probe(undropped, "steal-pid1-environ")
    assert red["outcome"] == "read" and "RUNPOD_API_KEY" in red["keys"], (
        "the RED control did not reproduce the defect — without the drop the "
        f"child must be able to read PID 1's environ, got {red}"
    )


@root_only
def test_child_cannot_read_the_control_parents_environ(dropped, undropped):
    """th#1380 D2: the delta-1 strip deletes WORKER_JWT from the child's env,
    and at a shared uid tenant code read it straight back out of the parent."""
    got = _json_probe(dropped, "steal-parent-environ")
    assert got["outcome"] == "denied", (
        f"tenant code read the control parent's environment: {got} — the "
        "worker JWT is one key away (pgw#858)"
    )

    red = _json_probe(undropped, "steal-parent-environ")
    assert red["outcome"] == "read" and "WORKER_JWT" in red["keys"], (
        "the RED control did not reproduce the defect — without the drop the "
        f"child must be able to read the parent's environ, got {red}"
    )


@root_only
def test_the_childs_own_env_carries_neither_credential(dropped):
    """The strip, now that it is load-bearing rather than polite."""
    keys = set(_json_probe(dropped, "own-environ-keys"))
    for name in ("WORKER_JWT", "RUNPOD_API_KEY", "PUBLIC_KEY"):
        assert name not in keys, f"{name} survived into the compute child"
    # ...and the parent still has all three: they moved, they did not vanish.
    for name in ("WORKER_JWT", "RUNPOD_API_KEY", "PUBLIC_KEY"):
        assert name in os.environ


@root_only
def test_child_cannot_read_root_home(dropped, undropped):
    Path("/root/pgw858-secret").write_text("x", encoding="utf-8")
    Path("/root").chmod(0o700)
    try:
        assert _probe(dropped, "read-root-home") == "PermissionError"
        assert _probe(undropped, "read-root-home").startswith("read:"), (
            "the RED control did not reproduce the defect"
        )
    finally:
        Path("/root/pgw858-secret").unlink(missing_ok=True)


# ---- positive controls: a drop that breaks serving is not a fix ----------


@root_only
def test_the_dropped_child_still_boots_reaches_hello_and_serves(dropped):
    """Reaching Hello and answering a dispatch IS the positive control: the
    child imported torch, wired its executor, connected on a socket it could
    only reach because the parent handed it over, and returned a result."""
    got = _json_probe(dropped, "report-identity")
    assert got["uid"] != 0
    home = _json_probe(dropped, "home-probe")
    assert _probe(dropped, "write-probe", home["tmpdir"]) == "ok"


@root_only
def test_the_dropped_child_can_write_every_path_it_was_granted(dropped, tmp_path):
    """The grant list, checked from inside the child rather than asserted from
    the parent's side of the boundary."""
    home = _json_probe(dropped, "home-probe")
    assert home["home"].startswith("/var/lib/gen-worker"), home
    assert home["user"] and home["user"] != "root", home
    for path in (home["home"], home["tmpdir"], str(tmp_path / "cache")):
        assert _probe(dropped, "write-probe", path) == "ok", (
            f"the compute child cannot write {path} — it is in the grant list "
            "(pgw#858). The answer is another entry there, never root."
        )


@root_only
def test_the_dropped_child_can_still_write_the_config_snapshot(dropped):
    """The one child-side writer that RAISES rather than degrades, and the only
    thing the child writes inside the root-owned image tree. It has no coverage
    in the split suite — a config push is the first thing that would have found
    this, in production."""
    got = _probe(dropped, "config-snapshot-probe")
    assert got.startswith("ok:"), (
        f"the compute child cannot rewrite the th#1087 config snapshot: {got} "
        "— a config-generation push would raise ConfigSnapshotWriteError"
    )


@root_only
def test_the_parent_can_still_signal_and_reap_the_dropped_child(dropped):
    """pgw#845's bounded shutdown across the new uid boundary: root can signal
    any uid, and that direction is the only one that has to work (the child
    never signals the parent — liveness is a pipe)."""
    _probe(dropped, "report-identity")           # a live, serving child
    proc = dropped.pc._proc
    assert proc is not None and proc.returncode is None
    # gw#666/pgw#795: NO `assert time.monotonic() - started < 120.0` here. That
    # asserted the RUNNER'S SPEED, and the two assertions below already state
    # the property it was standing in for — the parent exited and the child was
    # reaped. A slow runner made the old form fail for being slow; a hung
    # `close()` makes the new form fail by never reaping, which is the actual
    # defect. Hang containment belongs to the job timeout, not to a literal in
    # a correctness assertion.
    dropped.close()
    assert not dropped.alive, "the parent did not exit after SIGTERM"
    assert proc.returncode is not None, "the dropped child was never reaped"


# ---- the pieces, exercised directly --------------------------------------


@root_only
def test_a_drop_that_did_not_take_refuses_rather_than_execs_tenant_code():
    """The assertion inside the preexec hook is the last line of defence: if
    setuid silently no-ops on some future kernel/container combination, the
    child must die, not run tenant code as root."""
    lying = privdrop.DropPlan(
        uid=os.getuid() + 1, gid=os.getgid(), groups=(), user="x", home="/tmp")
    with pytest.raises(RuntimeError, match="privilege drop did not take"):
        privdrop._assert_dropped(lying)


@root_only
def test_the_uid_selector_refuses_zero():
    with pytest.raises(ValueError, match="not a way to turn the boundary off"):
        privdrop._resolve_target("0")


@root_only
def test_a_shared_system_directory_is_never_granted(tmp_path):
    """The post-mortem marker dir defaults to /tmp when TENSORHUB_CACHE_DIR is
    unset; a recursive chown of it would hand a shared tree to tenant code."""
    plan = privdrop.plan_drop(str(tmp_path / "home"))
    assert plan is not None
    granted = privdrop.grant_paths(plan, ["/tmp", "/", str(tmp_path / "ok")])
    assert granted == [str(tmp_path / "ok")], granted
    assert os.stat("/tmp").st_uid == 0
