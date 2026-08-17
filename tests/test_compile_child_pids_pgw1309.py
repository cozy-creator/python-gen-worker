"""pgw#1309: the compile child's pids have to reach the hub, not a pod log.

The pgw#1232 acceptance legs ask one question the fleet could not answer off a
serve pod (pgw#760): did this compile run in a CHILD, or on the process that
serves tenants? The child pid lived in one ``aot-pool: ... -> pid N`` log line
and the serving pid in ``entrypoint._startup_payload``, both pod-local, and
``worker_activity_events`` — the one channel a leg harness reads — carried no
pid at all.

Both halves are driven for real here. The pool half spawns REAL child
processes through the real ``EntryCompilePool`` (the compile INTERIOR is
``harness.fake_compile_child``, a separate executable: this box forbids local
inductor, and a mocked spawn would produce no second pid to compare against)
and reads the events back off a bound sink as ``ActivityUpdate`` envelopes. The
boot half runs the real ``Executor.ensure_setup``.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import msgspec
import pytest

from gen_worker import Resources, activity, endpoint, process_role
from gen_worker import aot_compile_pool as pool
from gen_worker.child_contract import CompileSpec
from gen_worker.executor import Executor
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs

from harness import fake_compile_child

torch = pytest.importorskip("torch")

_DECLARED = 4


@pytest.fixture(autouse=True)
def _isolated_role():
    """The role is process-global by design; a suite must not leak it."""
    before = process_role.role()
    yield
    process_role.declare(before)
    activity.reset_for_tests()


class _In(msgspec.Struct):
    prompt: str = "x"


class _Out(msgspec.Struct):
    y: str


def _detail(update: pb.ActivityUpdate) -> Dict[str, str]:
    """``k=v`` pairs out of one event's detail — the harness's own read."""
    return dict(re.findall(r"(\w+)=(\S+)", update.detail))


def _run_pool_with_a_bound_sink(tmp_path: Path, workers: int) -> List[Any]:
    """Drive a real K-wide pool with the wire bound, return the updates."""
    sent: List[pb.ActivityUpdate] = []
    loop = asyncio.new_event_loop()

    async def _send(msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") == "activity_update":
            sent.append(msg.activity_update)

    width = pool.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == workers, width.reason
    box = pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path))
    try:
        activity.bind_sink(_send, loop)
        packed = box.compile(pool.EntryJob(
            function="generate",
            cfg=CompileSpec(family="sdxl"),
            modules=("harness.toy_endpoints",),
            out_dir=str(tmp_path / "artifacts")))
        assert len(packed) == _DECLARED, packed
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        activity.reset_for_tests()
        loop.close()
    return sent


def test_a_compile_childs_pids_reach_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every share emits START and FINISH rows naming the REAL child pid, the
    parent that spawned it, and the serving pid it is not.

    The pids are checked against the ones the pool actually spawned — a spy on
    the real ``_spawn``, not a stub of it — because an event carrying a
    plausible-looking number that belongs to no process would satisfy
    ``child_pid != serving_pid`` while proving nothing.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "ok")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    process_role.declare(process_role.ROLE_SERVING)

    spawned: List[int] = []
    real_spawn = pool.EntryCompilePool._spawn

    def _spy(self: Any, job: Any, job_path: Path) -> Any:
        row = real_spawn(self, job, job_path)
        spawned.append(row.proc.pid)
        return row

    monkeypatch.setattr(pool.EntryCompilePool, "_spawn", _spy)

    updates = _run_pool_with_a_bound_sink(tmp_path, workers=2)

    rows = [u for u in updates if u.kind == activity.KIND_COMPILE_CHILD]
    starts = [u for u in rows if u.phase == pool.PHASE_CHILD_START]
    finishes = [u for u in rows if u.phase == pool.PHASE_CHILD_FINISH]
    assert len(spawned) == 2 and len(set(spawned)) == 2, spawned
    assert len(starts) == 2 and len(finishes) == 2, (
        f"the compile children's pids did not reach the wire: "
        f"{[(u.kind, u.phase) for u in updates]}")

    serving = os.getpid()
    for row in starts + finishes:
        facts = _detail(row)
        assert int(facts["serving_pid"]) == serving > 0, facts
        assert facts["role"] == process_role.ROLE_SERVING, facts
        # The point of the whole issue: the compile did not run here.
        assert int(facts["child_pid"]) != int(facts["serving_pid"]), facts
        assert int(facts["child_ppid"]) == serving, facts
        assert row.family == "sdxl", row.family
    assert {int(_detail(u)["child_pid"]) for u in starts} == set(spawned)
    assert {int(_detail(u)["child_pid"]) for u in finishes} == set(spawned)
    # Graph-keyed: a leg can join a pid row to the artifact it produced.
    for row in finishes:
        facts = _detail(row)
        assert facts["status"] == pool.COMPILED, facts
        keys = facts["compiled_graph_keys"].split(",")
        assert keys and all(k.startswith("ek1-") for k in keys), facts
        assert int(facts["classes"]) == len(keys), facts
        assert row.duration_ms > 0, "the child's wall is the span this measures"
    # The union of the pid rows' keys is the cell the mint packed.
    assert sorted(
        k for u in finishes
        for k in _detail(u)["compiled_graph_keys"].split(",")
    ) == sorted(f"ek1-cls-dim={i}" for i in range(_DECLARED))


def test_a_refusing_share_still_leaves_its_pid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FINISH row is emitted before every gate that can raise.

    A share that died is exactly the one whose pid a leg needs — and
    ``_collect`` raises on that path, so an event emitted after the gates would
    exist only for mints that succeeded.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "die")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    process_role.declare(process_role.ROLE_SERVING)

    sent: List[pb.ActivityUpdate] = []
    loop = asyncio.new_event_loop()

    async def _send(msg: pb.WorkerMessage) -> None:
        if msg.WhichOneof("msg") == "activity_update":
            sent.append(msg.activity_update)

    width = pool.entry_workers(
        _DECLARED, limit=2, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    box = pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=fake_compile_child.script(tmp_path))
    try:
        activity.bind_sink(_send, loop)
        with pytest.raises(pool.EntryCompileFailed):
            box.compile(pool.EntryJob(
                function="generate",
                cfg=CompileSpec(family="sdxl"),
                modules=("harness.toy_endpoints",),
                out_dir=str(tmp_path / "artifacts")))
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        activity.reset_for_tests()
        loop.close()

    finishes = [
        u for u in sent
        if u.kind == activity.KIND_COMPILE_CHILD
        and u.phase == pool.PHASE_CHILD_FINISH]
    assert finishes, (
        "the share that FAILED left no pid row — the one case a leg needs")
    facts = _detail(finishes[0])
    assert int(facts["child_pid"]) != os.getpid(), facts
    assert int(facts["serving_pid"]) == os.getpid(), facts


def test_the_serving_pid_and_role_are_stated_at_boot() -> None:
    """One event per sink bind, off the REAL executor setup path.

    Legs B and C assert "no compile ran under the serving PID". Without this
    row they have no PID to assert it about: the number existed only in the
    startup log payload and the postmortem boot record, both pod-local.
    """
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    @endpoint(resources=Resources(gpu=True))
    class Ep:
        def setup(self) -> None:
            pass

        def generate(self, ctx: Any, payload: _In) -> _Out:
            return _Out(y="ok")

    specs = extract_specs(Ep)
    ex = Executor(specs, _send)

    async def _go() -> None:
        await ex.ensure_setup(specs[0])
        for _ in range(10):
            await asyncio.sleep(0)

    process_role.declare(process_role.ROLE_UNKNOWN)
    asyncio.run(_go())

    rows = [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == activity.KIND_PROCESS_ROLE]
    assert rows, "the serving process never stated its pid or its role"
    facts = _detail(rows[0])
    assert int(facts["serving_pid"]) == os.getpid(), facts
    assert facts["role"] == process_role.ROLE_SERVING == rows[0].phase, facts
    assert int(facts["ppid"]) == os.getppid(), facts
    # And the declaration is what the pool's rows then read.
    assert process_role.serving_pid() == os.getpid()


def test_an_undeclared_process_never_passes_itself_off_as_serving() -> None:
    """A compile child that emitted one of these must answer 0, not its own
    pid — otherwise "child_pid != serving_pid" is satisfiable by a compile
    running in the wrong process entirely."""
    process_role.declare(process_role.ROLE_UNKNOWN)
    assert process_role.serving_pid() == 0
    assert "serving_pid=0" in process_role.facts()
