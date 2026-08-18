"""What the compute child may touch on the host: its uid, its OOM rank, its RAM.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict

import msgspec
import pytest
import torch
from harness.hub_double import is_ready, is_result_for  # noqa: E402
from harness.progress_wait import Cadence, await_progress  # noqa: E402
from harness.split import (  # noqa: E402,F401 — fixtures come with it
    SplitHarness,
    _payload,
    captured_dials,
    isolated_postmortem,
)

from gen_worker import (
    host_move_guard,
    local_compiled_graph_store,  # noqa: E402
    postmortem,
)
from gen_worker.api.errors import HostRamMoveRefusedError
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.procsplit import (
    oom_rank,
    privdrop,  # noqa: E402
)

# ============================================================================
# pgw#858 — pgw#858 / th#1380: the compute child runs as an unprivileged
#   uid.
# ============================================================================

TESTS_DIR = Path(__file__).resolve().parent


REPO = TESTS_DIR.parent


RUNPOD_KEY = "rpa_PGW858_ACCOUNT_AUTHORITY_SENTINEL"


PUBLIC_KEY = "ssh-ed25519 AAAAPGW858OPERATORKEYSENTINEL operator@cozy"


WORKER_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ3LTg1OCJ9.pgw858-sentinel"


IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


_TOO_BROAD = {Path("/"), Path("/usr"), Path("/usr/local"), Path("/etc"),
              Path("/lib"), Path("/opt"), Path("/var"), Path("/home")}


def _interpreter_mounts() -> list[Path]:
    """pgw#858: Every prefix the container must see in order to exec THIS interpreter."""
    roots = {Path(sys.prefix), Path(sys.prefix).resolve(),
             Path(sys.base_prefix), Path(sys.base_prefix).resolve()}
    base_exe = getattr(sys, "_base_executable", None)
    if base_exe:
        named = Path(base_exe)
        if named.parent.name == "bin":
            roots |= {named.parent.parent, named.parent.parent.resolve()}
    keep: list[Path] = []
    for root in sorted(roots):
        if root in _TOO_BROAD:
            continue
        if root == REPO or REPO in root.parents:
            continue          # inside the worktree mount already
        if any(parent in roots for parent in root.parents):
            continue          # a broader mount in this same set covers it
        keep.append(root)
    return keep


@pytest.mark.skipif(IS_ROOT, reason="already root — the rows below run directly")
def test_privilege_isolation_rows_under_a_real_root_parent():
    """pgw#858: Run every row below inside a container whose PID 1 is root and carries the RunPod key, which is ..."""
    if shutil.which("docker") is None:
        pytest.skip(
            "docker is required: pgw#858 is a uid boundary and a non-root "
            "runner cannot observe it (every /proc read is already denied)"
        )
    mounts = _interpreter_mounts()
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
        *[arg for root in mounts for arg in ("-v", f"{root}:{root}:ro")],
        "-w", str(REPO),
        "ubuntu:24.04",
        sys.executable, "-m", "pytest",
        str(Path("tests") / Path(__file__).name),
        "-q", "-p", "no:cacheprovider", "--no-header",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    # pgw#966, the third-party half: `ubuntu:24.04` comes from Docker Hub, which
    # has a rate limiter and outages of its own. Exit 125 is the docker CLI/daemon
    # refusing before anything ran — a pull failure, not a uid-boundary failure —
    # and a third party may never turn a required check red. Its own reason, so
    # the census counts it apart from the exec case below, which IS ours to fix.
    if proc.returncode == 125:
        pytest.skip(
            "docker could not start the container, so no row ran (image pull or "
            f"daemon): {proc.stderr.strip().splitlines()[-1][:200] if proc.stderr.strip() else '(no stderr)'}"
        )
    # The container failing to START is not the rows failing, and conflating
    # them is how this test read as a pgw#858 regression on GitHub CI while
    # passing on every developer box. It is also, until pgw#966, how this row
    # read as GREEN COVERAGE on GitHub CI while never once executing: exit 127
    # with `[FATAL tini (7)] exec .../.venv/bin/python failed: No such file or
    # directory`, skipped by name, indistinguishable in a green log from a row
    # that passed. `_interpreter_mounts()` above is the fix for the cause; this
    # branch stays for anything it does not cover, and now carries the facts
    # needed to diagnose it from a log rather than from a developer box.
    if proc.returncode == 127 and "exec" in proc.stderr and "failed" in proc.stderr:
        pytest.skip(
            "the container cannot exec this interpreter, so no row ran: "
            f"{proc.stderr.strip().splitlines()[-1][:200]} | mounted "
            f"{[str(m) for m in mounts]} | executable={sys.executable} "
            f"prefix={sys.prefix} base_prefix={sys.base_prefix} "
            f"base_executable={getattr(sys, '_base_executable', None)} "
            "— pgw#858 needs an image that can run the host's python, not a "
            "verdict from a container that never started"
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


root_only = pytest.mark.skipif(
    not IS_ROOT,
    reason="pgw#858 is a uid boundary — the containerised row above measures it",
)

from harness.hub_double import is_ready, is_result_for  # noqa: E402
from harness.progress_wait import Cadence, await_progress  # noqa: E402
from harness.split import (  # noqa: E402,F401 — fixtures come with it
    SplitHarness,
    _payload,
    captured_dials,
    isolated_postmortem,
)

from gen_worker import local_compiled_graph_store  # noqa: E402
from gen_worker.procsplit import privdrop  # noqa: E402


@pytest.fixture()
def pod_shaped_env(monkeypatch):
    """The parent's own environment, as a pod delivers it."""
    monkeypatch.setenv("RUNPOD_API_KEY", RUNPOD_KEY)
    monkeypatch.setenv("PUBLIC_KEY", PUBLIC_KEY)
    monkeypatch.setenv("WORKER_JWT", WORKER_JWT)


def _split(tmp_path: Path, *, drop: bool,
           monkeypatch: Optional[pytest.MonkeyPatch] = None) -> SplitHarness:
    """A real split whose child imports the pgw#858 probe handlers."""
    if not drop:
        assert monkeypatch is not None
        monkeypatch.setattr(privdrop, "plan_drop", lambda home: None)
    return SplitHarness(
        tmp_path,
        extra_child_env={"PGW763_CHILD_MODULES": "harness.privdrop_endpoints"},
    )


def _probe(h: SplitHarness, fn: str, text: str = "") -> str:
    conn = h.scheduler.wait_connection(0)
    conn.wait_for(is_ready)
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
    """The pre-pgw#858 tree, alive in the same run, so every denial below is measured against a control that pro..."""
    h = _split(tmp_path / "red", drop=False, monkeypatch=monkeypatch)
    try:
        yield h
    finally:
        h.close()


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
    """#858's "confirm no setuid escalation path in the base images" — answered by imposing the property rather ..."""
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
    """th#1380 D1: RunPod injects an account-authority key into every pod and the create call cannot suppress it."""
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
    """th#1380 D2: the delta-1 strip deletes WORKER_JWT from the child's env, and at a shared uid tenant code re..."""
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


@root_only
def test_the_dropped_child_still_boots_reaches_hello_and_serves(dropped):
    """pgw#858: Reaching Hello and answering a dispatch IS the positive control: the child imported torch, wired..."""
    got = _json_probe(dropped, "report-identity")
    assert got["uid"] != 0
    home = _json_probe(dropped, "home-probe")
    assert _probe(dropped, "write-probe", home["tmpdir"]) == "ok"


@root_only
def test_the_dropped_child_can_write_every_path_it_was_granted(dropped, tmp_path):
    """pgw#858: The grant list, checked from inside the child rather than asserted from the parent's side of the..."""
    home = _json_probe(dropped, "home-probe")
    assert home["home"].startswith("/var/lib/gen-worker"), home
    assert home["user"] and home["user"] != "root", home
    for path in (home["home"], home["tmpdir"], str(tmp_path / "cache")):
        assert _probe(dropped, "write-probe", path) == "ok", (
            f"the compute child cannot write {path} — it is in the grant list "
            "(pgw#858). The answer is another entry there, never root."
        )


@root_only
def test_the_dropped_child_can_write_a_RELOCATED_local_compiled_graph_store(dropped):
    """pgw#1349: the mint's own store, when an operator has moved it.

    ``local_compiled_graph_store.store_root()`` defaults under ``~/.cache``, which the
    grant list already covers through the compute uid's home — so the gap was
    invisible for as long as nobody moved it. cozy-local DOES move it
    (``internal/paths/paths.go`` exports ``GEN_WORKER_LOCAL_CELLS_DIR``), and a
    relocated root is created by this root parent at 0755 and never chowned, so
    the child's first memo write died on
    ``PermissionError: .../aot-cells`` — mid-request, taking the stream down
    with it. It surfaced as a NON-DETERMINISTIC red of a neighbouring row in
    this very file, because whether the mint reached its memo write inside a
    given tape was a race. Asserted directly here so the boundary is measured
    instead of stumbled over.

    The suite's own autouse ``_isolated_local_compiled_graph_store`` fixture is what
    relocates it here, which makes this an honest reproduction of the operator
    case rather than a construction: nothing in the test sets the env for the
    test's benefit."""
    root = os.environ.get(local_compiled_graph_store.ENV_STORE_DIR, "")
    assert root, (
        f"{local_compiled_graph_store.ENV_STORE_DIR} is unset, so this row would pass "
        "for the wrong reason — the suite's autouse fixture sets it"
    )
    assert not os.path.exists(os.path.join(root, local_compiled_graph_store.COMPILED_GRAPHS_DIRNAME)), (
        "this row must measure the CHILD creating the compiled graphs root — a "
        f"{local_compiled_graph_store.COMPILED_GRAPHS_DIRNAME} this root parent made would be "
        "root-owned and the probe would report the wrong thing"
    )
    # `write-probe` mkdirs a nested subtree before it writes, which is the
    # operation that actually died: `_write_json_atomic` reaches every sidecar
    # through `parent.mkdir(parents=True)`, and `aot-cells` is the component
    # that did not exist yet.
    assert _probe(dropped, "write-probe", root) == "ok", (
        f"the compute child cannot write its own compiled graph store at {root}. The "
        "mint writes the memo and every per-compiled graph sidecar there, so this is a "
        "dead compute child on any pod whose store has been relocated "
        "(pgw#1349). The answer is another entry in the grant list, never root."
    )


@root_only
def test_the_dropped_child_can_still_write_the_config_snapshot(dropped):
    """pgw#858: The one child-side writer that RAISES rather than degrades, and the only thing the child writes ..."""
    got = _probe(dropped, "config-snapshot-probe")
    assert got.startswith("ok:"), (
        f"the compute child cannot rewrite the th#1087 config snapshot: {got} "
        "— a config-generation push would raise ConfigSnapshotWriteError"
    )


def _reap_state(proc: Any, pid: int, *, parent_alive: bool) -> str:
    """pgw#858: ``running`` -> ``reaped``, with THIS process doing the collecting once nothing else can."""
    if proc.returncode is not None:
        return "reaped"
    if parent_alive:
        try:
            info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return "reaped"      # no longer our child: somebody here reaped it
        return "running" if info is None else "exited"
    try:
        collected, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return "reaped"          # the watcher won the race; equally collected
    return "reaped" if collected == pid else "running"


class _UndeliveredExit:
    """An asyncio Process whose exit was never delivered — pgw#956's drop."""

    returncode = None


def test_a_zombie_no_loop_can_report_is_collected_here_rather_than_waited_on():
    """pgw#1024's mechanism and its fix, without having to lose the coin flip."""
    proc = _UndeliveredExit()
    pid = os.posix_spawn("/bin/sleep", ["sleep", "600"], {})
    try:
        # While a parent could still deliver, the helper LOOKS and does not take.
        assert _reap_state(proc, pid, parent_alive=True) == "running"
        os.kill(pid, 9)
        await_progress(
            lambda: _reap_state(proc, pid, parent_alive=True),
            lambda seen: seen == "exited",
            what="the unwatched child to become a zombie nobody collects",
            cadence=Cadence(),
        )
        # ...and it STAYS there: no watcher, no loop, no later reader recovers it.
        assert _reap_state(proc, pid, parent_alive=True) == "exited"

        # The parent is gone. This process is the only reaper there is.
        assert _reap_state(proc, pid, parent_alive=False) == "reaped"
        assert _reap_state(proc, pid, parent_alive=False) == "reaped"
        assert _reap_state(proc, pid, parent_alive=True) == "reaped"
        pid = 0
    finally:
        if pid:
            os.kill(pid, 9)
            os.waitpid(pid, 0)


@root_only
def test_the_parent_can_still_signal_and_reap_the_dropped_child(dropped):
    """pgw#845's bounded shutdown across the new uid boundary: root can signal any uid, and that direction is th..."""
    _probe(dropped, "report-identity")           # a live, serving child
    proc = dropped.pc._proc
    assert proc is not None and proc.returncode is None
    pid = proc.pid
    # NO `assert time.monotonic() - started < 120.0` here, and
    #  no single read after `close()` either — returning from `close()`
    # is not proof the reap completed. The property is that the parent exited
    # and the child was reaped, so wait on THAT advancing; a hang fails by never
    # reaping, which is the actual defect.
    dropped.close()

    def observe() -> tuple:
        # One read of the parent's liveness for both halves: the answer to
        # "may the loop still deliver?" decides whether we look or collect.
        alive = dropped.alive
        return alive, _reap_state(proc, pid, parent_alive=alive)

    await_progress(
        observe,
        lambda seen: seen == (False, "reaped"),
        what="the parent to exit after SIGTERM and the dropped child to be reaped",
        cadence=Cadence(),
        render=lambda seen: f"parent alive={seen[0]}, child {seen[1]}",
    )


@root_only
def test_a_drop_that_did_not_take_refuses_rather_than_execs_tenant_code():
    """pgw#858: The assertion inside the preexec hook is the last line of defence: if setuid silently no-ops on ..."""
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
    """pgw#858: The post-mortem marker dir defaults to /tmp when TENSORHUB_CACHE_DIR is unset; a recursive chown..."""
    plan = privdrop.plan_drop(str(tmp_path / "home"))
    assert plan is not None
    granted = privdrop.grant_paths(plan, ["/tmp", "/", str(tmp_path / "ok")])
    assert granted == [str(tmp_path / "ok")], granted
    assert os.stat("/tmp").st_uid == 0


# ============================================================================
# pgw#975 — pgw#975: the pgw#763 split's OOM victim order, declared instead
#   of emergent.
# ============================================================================

GIB = 1024 ** 3


linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="oom_score_adj is a Linux /proc interface"
)


def _read_oom_score_adj(pid: int) -> int:
    return int(Path(f"/proc/{pid}/oom_score_adj").read_text().strip())


@pytest.fixture()
def split(tmp_path, captured_dials):  # noqa: F811
    h = SplitHarness(tmp_path)
    try:
        yield h
    finally:
        h.close()


@linux_only
def test_a_real_compute_child_outranks_the_control_parent(split):
    """pgw#975: RED before this issue: the child inherited the parent's value exactly, so the reporter's surviva..."""
    split.scheduler.wait_connection(0).wait_for(is_ready)
    proc = split.pc._proc
    assert proc is not None, "no compute child was spawned"

    child_adj = _read_oom_score_adj(proc.pid)
    parent_adj = _read_oom_score_adj(os.getpid())

    delta = oom_rank.score_adj_delta_for_domain(
        oom_rank.oom_domain_bytes(), oom_rank.parent_ceiling_bytes()
    )
    assert child_adj == min(1000, parent_adj + delta), (
        f"child oom_score_adj={child_adj}, expected parent({parent_adj}) + "
        f"{delta} — read back off /proc/{proc.pid}/oom_score_adj"
    )
    assert child_adj > parent_adj, (
        f"the compute child ({child_adj}) does not outrank the control parent "
        f"({parent_adj}); a kernel OOM can take the reporter"
    )


@linux_only
def test_the_whole_compute_subtree_inherits_it(tmp_path):
    """pgw#975: The mint child and the AOT pool's entry children are spawned BELOW a compute child and get no ca..."""
    child = tmp_path / "child.py"
    child.write_text(
        "import subprocess, sys\n"
        "from gen_worker.procsplit.oom_rank import raise_own_oom_score_adj\n"
        "rank = raise_own_oom_score_adj()\n"
        "assert rank.applied, rank.format()\n"
        "grand = subprocess.run(\n"
        "    [sys.executable, '-c',\n"
        "     'print(open(\"/proc/self/oom_score_adj\").read().strip())'],\n"
        "    capture_output=True, text=True, check=True)\n"
        "print(open('/proc/self/oom_score_adj').read().strip())\n"
        "print(grand.stdout.strip())\n"
    )
    out = subprocess.run(
        [sys.executable, str(child)],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).resolve().parent.parent / "src"),
             os.environ.get("PYTHONPATH", "")])},
    )
    mine, grandchild = (int(v) for v in out.stdout.split())
    delta = oom_rank.score_adj_delta_for_domain(
        oom_rank.oom_domain_bytes(), oom_rank.parent_ceiling_bytes()
    )
    assert mine == min(1000, _read_oom_score_adj(os.getpid()) + delta)
    assert grandchild == mine, (
        "a grandchild did not inherit the rank — the mint child and every "
        "inductor entry child would be unranked"
    )


def test_the_parent_ceiling_still_matches_the_constants_it_was_derived_from():
    """pgw#975: `oom_rank` writes the reship buffer out literally so it can run before grpc exists in the child."""
    from gen_worker.procsplit.seam import CONTROL_FRAME_CEILING_BYTES
    from gen_worker.transport import RESHIP_WINDOW

    assert oom_rank._PARENT_BUFFER_BYTES == (
        RESHIP_WINDOW * CONTROL_FRAME_CEILING_BYTES
    )


@pytest.mark.parametrize(
    "domain_gib, expected, shape",
    [
        (755.07, 1, "RunPod CPU pod, measured live 2026-07-30"),
        (124.91, 2, "RunPod 4090 SECURE, measured live 2026-07-30"),
        (14.9, 15, "tightest cgroup cap observed (0.56.2 ram_total_gb report)"),
        (2.0, 110, "a hypothetical 2 GiB container"),
    ],
)
def test_the_value_is_derived_from_the_domain_not_picked(domain_gib, expected, shape):
    """pgw#975: §4.24: the number has to be re-derivable."""
    got = oom_rank.score_adj_delta_for_domain(
        int(domain_gib * GIB), oom_rank.parent_ceiling_bytes()
    )
    assert got == expected, f"{shape}: expected {expected}, got {got}"


def test_the_margin_always_covers_the_parents_whole_ceiling_twice():
    """pgw#975: The property the table is an instance of: one point is 0.1% of the domain, so `adj` points must ..."""
    ceiling = oom_rank.parent_ceiling_bytes()
    for domain in (1 * GIB, 8 * GIB, 15 * GIB, 64 * GIB, 125 * GIB, 755 * GIB):
        adj = oom_rank.score_adj_delta_for_domain(domain, ceiling)
        assert adj * (domain / 1000) >= 2 * ceiling, (
            f"domain={domain / GIB:.0f}GiB adj={adj} buys "
            f"{adj * domain / 1000 / 1024 ** 2:.0f}MiB against a "
            f"{ceiling / 1024 ** 2:.0f}MiB parent ceiling"
        )


def test_an_unreadable_domain_degrades_toward_protecting_the_reporter():
    """pgw#975: Guessing the roomiest host would silently produce adj=1 on a tight container."""
    tight = oom_rank.score_adj_delta_for_domain(0, oom_rank.parent_ceiling_bytes())
    assert tight == oom_rank.score_adj_delta_for_domain(
        oom_rank._TIGHTEST_OBSERVED_DOMAIN_BYTES, oom_rank.parent_ceiling_bytes()
    )
    assert tight > oom_rank.score_adj_delta_for_domain(755 * GIB,
                                                 oom_rank.parent_ceiling_bytes())


@linux_only
def test_a_failed_set_is_a_named_degradation_never_a_silent_pass(
    monkeypatch, tmp_path, caplog,
):
    """pgw#975: A hardened container with a read-only /proc must not leave us believing the guarantee holds."""
    unwritable = tmp_path / "nonexistent-dir" / "oom_score_adj"
    monkeypatch.setattr(oom_rank, "_SELF_OOM_SCORE_ADJ", unwritable)

    with caplog.at_level(logging.ERROR, logger=oom_rank.__name__):
        rank = oom_rank.raise_own_oom_score_adj()

    assert not rank.applied
    assert rank.unprotected, "the degradation did not name what is unprotected"
    assert "control parent" in rank.unprotected
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert oom_rank.DEGRADE_PHASE in logged, "the failure was not logged typed"
    assert "errno" in rank.reason


@linux_only
def test_the_gap_is_cut_over_whatever_baseline_was_inherited(monkeypatch, tmp_path):
    """pgw#975: The bug a real spawn caught: `oom_score_adj` is INHERITED, so the value the child reads is the p..."""
    fake = tmp_path / "oom_score_adj"
    fake.write_text("200\n")
    monkeypatch.setattr(oom_rank, "_SELF_OOM_SCORE_ADJ", fake)

    rank = oom_rank.raise_own_oom_score_adj()
    delta = oom_rank.score_adj_delta_for_domain(
        oom_rank.oom_domain_bytes(), oom_rank.parent_ceiling_bytes()
    )

    assert rank.applied
    assert int(fake.read_text()) == 200 + delta > 200
    assert rank.previous == 200


@linux_only
def test_a_baseline_already_at_the_kernel_maximum_is_reported_unprotected(
    monkeypatch, tmp_path, caplog,
):
    """pgw#975: At 1000 the parent is already maximally killable and no child can be ranked above it."""
    fake = tmp_path / "oom_score_adj"
    fake.write_text("1000\n")
    monkeypatch.setattr(oom_rank, "_SELF_OOM_SCORE_ADJ", fake)

    with caplog.at_level(logging.ERROR, logger=oom_rank.__name__):
        rank = oom_rank.raise_own_oom_score_adj()

    assert not rank.applied
    assert rank.reason == "baseline_at_kernel_maximum"
    assert rank.unprotected
    assert oom_rank.DEGRADE_PHASE in "\n".join(
        r.getMessage() for r in caplog.records
    )


def test_the_container_facts_now_carry_memory_oom_group():
    """pgw#975: Read off this box's real cgroup chain."""
    facts = postmortem.container_limits()
    assert "memory_oom_group" in facts
    assert facts["memory_oom_group"] in (0, 1, None)


def test_a_group_kill_is_called_out_in_the_death_dial():
    detail = postmortem.format_detail(
        phase="compute_process_exit",
        verdict={"exit_code": None, "signaled": True, "signal": 9,
                 "signal_name": "SIGKILL", "core_dumped": False},
        limits={"memory_oom_group": 1},
    )
    assert "memory.oom.group=1" in detail
    assert "GROUP KILL" in detail

    benign = postmortem.format_detail(
        phase="compute_process_exit",
        verdict={"exit_code": 1, "signaled": False},
        limits={"memory_oom_group": 0},
    )
    assert "memory.oom.group=0" in benign
    assert "GROUP KILL" not in benign


def test_the_entrypoint_ranks_the_child_before_its_heavy_imports():
    """pgw#975: The harness enters one layer below `entrypoint`, so the placement itself is asserted on the sour..."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "gen_worker" / "entrypoint.py"
    ).read_text()
    call = src.index("raise_own_oom_score_adj()")
    assert src.index("is_compute_child") < call
    for heavy in ("from .worker import Worker", "import msgspec"):
        assert call < src.index(heavy), (
            f"the OOM rank is declared after {heavy!r}"
        )


# ============================================================================
# pgw#763 — pgw#763: an oversized ``.to("cpu")`` is refused typed, not
#   cgroup-killed.
# ============================================================================

_GIB = 1 << 30


def _cgroup(tmp_path: Path, *, limit: int, current: int) -> tuple[Path, Path]:
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(limit))
    (root / "memory.current").write_text(str(current))
    (root / "memory.stat").write_text(
        "anon {a}\nfile 0\nactive_file 0\ninactive_file 0\nshmem 0\n"
        "file_dirty 0\nfile_writeback 0\n".format(a=current))
    proc = tmp_path / "self_cgroup"
    proc.write_text("0::/\n")
    return root, proc


@pytest.fixture()
def guard(tmp_path, monkeypatch):
    """Arm the real guard against a synthetic 8GiB cgroup with 6GiB held (te#138's proportions: the resident pip..."""
    monkeypatch.delenv("GEN_WORKER_HOST_MOVE_GUARD", raising=False)
    root, proc = _cgroup(tmp_path, limit=8 * _GIB, current=6 * _GIB)
    monkeypatch.setattr(host_move_guard, "_probe_root", root)
    monkeypatch.setattr(host_move_guard, "_probe_self", proc)
    assert host_move_guard.install()
    return host_move_guard


def _big_meta_module() -> torch.nn.Module:
    # ~3.2GiB of fp32 weights that exist only as metadata: device.type="meta"
    # counts as incoming (not CPU-resident), and no RAM is ever allocated.
    with torch.device("meta"):
        return torch.nn.Linear(20480, 41984, bias=False)


def test_oversized_cpu_move_is_refused_typed(guard: Any) -> None:
    """te#138's ``_free()``: a module bigger than the remaining budget asks for CPU."""
    module = _big_meta_module()
    with pytest.raises(HostRamMoveRefusedError) as exc:
        module.to("cpu")
    msg = str(exc.value)
    assert "host-RAM move refused" in msg
    assert exc.value.incoming_bytes >= 3 * _GIB
    # ~2GiB available (8 limit - 6 held), floored — the numbers are named.
    assert exc.value.available_bytes <= 3 * _GIB

    # .cpu() is the same door.
    with pytest.raises(HostRamMoveRefusedError):
        _big_meta_module().cpu()


def test_small_and_non_cpu_moves_pass_through(guard: Any) -> None:
    """pgw#763: The guard must not tax or break ordinary moves: under-threshold modules skip the probe entirely,..."""
    small = torch.nn.Linear(64, 64)  # CPU-resident already, tiny
    out = small.to("cpu")
    assert out is small
    # dtype-only .to() on a big meta module: not a CPU landing, no refusal.
    _big_meta_module().to(dtype=torch.bfloat16)


def test_kill_switch_disables(guard: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEN_WORKER_HOST_MOVE_GUARD", "0")
    module = _big_meta_module()
    # Guard steps aside; torch itself then refuses to materialize meta
    # tensors via .to() — any non-HostRamMoveRefusedError shape is fine.
    try:
        module.to("cpu")
    except HostRamMoveRefusedError:  # pragma: no cover
        pytest.fail("kill switch did not disable the guard")
    except Exception:
        pass
