"""pgw#809: the compile pool — the parent half, driven for real.

Scenario-shaped rather than per-behaviour: ONE integration test drives the real
:class:`EntryCompilePool` through a real K-wide run and asserts everything that
must hold when every share lands at once; a second drives a real run-WITH-
FAILURE and asserts everything that must hold when one share dies mid-pool. The
width policy is the one piece that must be exercised across pods the box cannot
be, so it is a table.

What the failure scenario is really testing is the thing that makes a pool
dangerous: K children, one dies, and the others plus every ``cc1plus`` they
spawned have to go with it. A leak there is a serving pod that keeps burning
CPU against a cell nobody will ever adopt — so the assertion is a PROCESS SWEEP
of the real process table, not a mocked call count.

⚠️ **The child's INTERIOR is not exercised here, and since pgw#1215 it cannot
be.** A compile child builds its own weight-free pipeline and traces its own
share, so the four-linear toy program this file used to hand it is not an input
any more; a green end-to-end run needs a real AOTI compile of a real endpoint's
declared graph class, which is pgw#1215 step 3's POD leg. Everything the PARENT
does is still driven for real, against real spawned processes, through
``harness.fake_compile_child`` — a separate executable, not a monkeypatch, so
nothing under test is stubbed. The child's own refusal path is driven against
the real module below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import msgspec
import pytest

from gen_worker import aot_compile_pool as pool
from harness import fake_compile_child
from harness.progress_wait import Cadence, await_progress

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: How many graph classes the fake declaration produces. Not a multiple of the
#: widths used below, so `rows[i::K]` really has to partition unevenly.
_DECLARED = 6


def _template(tmp_path: Path) -> pool.EntryJob:
    return pool.EntryJob(
        function="generate",
        modules=("harness.toy_endpoints",),
        out_dir=str(tmp_path / "artifacts"))


def _child_script(tmp_path: Path, digest: str | None = None) -> str:
    return fake_compile_child.script(tmp_path, digest=digest)


def _pool(tmp_path: Path, *, workers: int, digest: str | None = None,
          ) -> pool.EntryCompilePool:
    # The width is STATED, not derived: a 4-vCPU CI runner honestly derives
    # K=1, and these scenarios would then pass while exercising no pool at all.
    width = pool.entry_workers(
        _DECLARED, limit=workers, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == workers
    return pool.EntryCompilePool(
        tmp_path / "pool", width=width, cache_dir=str(tmp_path / "cache"),
        python=_child_script(tmp_path, digest=digest))


def _survivors(script: str) -> List[int]:
    """Every live process still running THIS test's child script.

    Matched on the command line rather than on the parent's process GROUP: the
    pool spawns with ``start_new_session=True`` precisely so it can kill a
    child's whole group, which means a leaked child is never in the parent's
    group and a pgrp-based sweep can only ever return the empty set. That is
    how this assertion was vacuous — it passed with the teardown deleted.
    """
    out: List[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if script.encode() in cmdline:
            out.append(int(entry.name))
    return out


# ---------------------------------------------------------------------------
# Scenario 1: a run that completes
# ---------------------------------------------------------------------------


def test_the_pool_dispatches_shares_and_assembles_by_class_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six declared classes over K=2 shares, through real child processes.

    Asserts, in one run: every declared class comes back exactly once; the
    packed artifacts exist and are readable by the PARENT; assembly is ordered
    by graph-class NAME and not by completion; the pool really did run K
    children at once (structurally, not by clock); the per-share seconds and
    the child peak RSS are recorded (the memory bound is measured, not
    assumed); and NOTHING was staged — the whole point of the keystone is that
    no ExportedProgram crosses this boundary.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "ok")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path, workers=2)

    out = box.compile(_template(tmp_path))

    assert set(out) == {f"cls/dim={i}" for i in range(_DECLARED)}
    assert list(out) == sorted(out), (
        "the cell must assemble by graph-class NAME; a dict ordered by "
        "completion makes the artifact depend on which child finished first")
    for name, packed in out.items():
        assert packed.key and packed.artifact
        assert Path(packed.artifact).exists(), (
            f"{name}: {packed.artifact} is not visible to the parent — the "
            f"pool's out_dir is how packed graph classes travel")

    assert set(box.entry_seconds) == {"share-000", "share-001"}
    assert set(box.class_spans) == set(out), (
        "the per-CLASS spans are the granularity anybody asks about a "
        "compile; a per-SHARE number answers a question nobody asked")
    # Overlap is asserted STRUCTURALLY — K processes really were alive at
    # once — never as a wall-clock speedup.
    assert box.peak_concurrency == 2, box.peak_concurrency
    assert box.peak_rss_bytes > 0, (
        "per-share peak RSS is what bounds K by memory; an unmeasured peak "
        "makes the width policy a guess")

    staged = list((tmp_path / "pool").rglob("*.pt2"))
    assert not staged, (
        f"something staged a program: {staged}. pgw#1215's whole claim is "
        f"that the ExportedProgram never crosses this boundary")


def test_a_share_that_comes_back_short_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The class set must be WHOLE, and the parent never enumerated it.

    The parent holds no pipeline, so the only evidence that ``rows[i::K]``
    partitioned the declaration is that every child reported the same declared
    count and the union has exactly that many rows. Without this check a share
    that came back empty publishes a SHORT cell that verifies, arms, and is
    missing a class.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "short")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path, workers=2)

    with pytest.raises(pool.EntryCompileFailed) as caught:
        box.compile(_template(tmp_path))
    assert "did not partition the declaration" in str(caught.value)
    assert "short" in str(caught.value)


def test_two_shares_packing_the_same_class_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A class produced twice means the row order is not stable across
    children, so some other class is missing entirely. Last-writer-wins would
    hide exactly that."""
    monkeypatch.setenv("PGW_FAKE_CHILD", "collide")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path, workers=2)

    with pytest.raises(pool.EntryCompileFailed) as caught:
        box.compile(_template(tmp_path))
    assert "was packed by two shares" in str(caught.value)


# ---------------------------------------------------------------------------
# Scenario 2: a run where one share dies
# ---------------------------------------------------------------------------


def test_one_failing_share_fails_the_run_and_takes_its_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One share cannot run. The mint must fail NAMING it, the siblings must
    be torn down group-wide, and nothing may survive the call."""
    monkeypatch.setenv("PGW_FAKE_CHILD", "die")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path, workers=2)
    script = _child_script(tmp_path)

    with pytest.raises(pool.EntryCompileFailed) as caught:
        box.compile(_template(tmp_path))

    exc = caught.value
    assert exc.entry == "share-000"
    assert "share-000" in str(exc), (
        "a pool that fails anonymously is undebuggable — the share name and "
        "its row stride are the whole diagnostic on a pod with no logs")
    assert "rows[0::2]" in str(exc)
    assert "exited" in str(exc)

    # No orphans: the pool kills process GROUPS, so every child's own
    # inductor workers and g++ go with it. Waited on PROGRESS — each pid that
    # disappears is an advance — rather than on a clock, because a clock here
    # asserts the runner's speed and a wedged teardown must still FAIL rather
    # than be outrun (pgw#795).
    await_progress(
        lambda: tuple(sorted(_survivors(script))),
        lambda leaked: not leaked,
        what="the failed run's sibling children to be reaped",
        cadence=Cadence(),
        render=lambda leaked: f"{len(leaked)} orphan(s): {list(leaked)[:8]}",
    )


def test_a_hung_share_is_torn_down_with_its_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling that will NOT exit on its own must be KILLED, not waited for.

    The sharper half of the teardown claim: in the row above every child dies
    by itself, so a pool that reaped nothing would still look clean. Here
    share 1 sleeps for ten minutes and only the group-wide kill ends it — the
    shape a wedged ``cc1plus`` takes on a real pod.
    """
    monkeypatch.setenv("PGW_FAKE_CHILD", "die-and-hang")
    monkeypatch.setenv("PGW_FAKE_DECLARED", str(_DECLARED))
    box = _pool(tmp_path, workers=2)
    script = _child_script(tmp_path)
    with pytest.raises(pool.EntryCompileFailed):
        box.compile(_template(tmp_path))
    await_progress(
        lambda: tuple(sorted(_survivors(script))),
        lambda leaked: not leaked,
        what="the sleeping sibling to be killed with its group",
        cadence=Cadence(),
        render=lambda leaked: f"{len(leaked)} orphan(s): {list(leaked)[:8]}",
    )


# ---------------------------------------------------------------------------
# The real child, on the paths that need no compile
# ---------------------------------------------------------------------------


def test_a_named_child_refusal_is_reported_not_swallowed(tmp_path: Path) -> None:
    """The REAL ``aot_compile_child`` run by hand on a job whose recipe cannot
    be built: non-zero exit, a typed report, and the refusal sentence on disk.

    The boundary is a file exactly so a run that fails on share 2 of 4 is
    reproducible without the parent. This is the child's own code — the
    preflight it inherits from ``boot_trace_child`` — and it refuses before any
    compile, which is why it is affordable here.
    """
    slot = tmp_path / "slot"
    slot.mkdir()
    job = pool.EntryJob(
        share="share-000", share_index=0, share_count=1,
        function="generate",
        modules=("gen_worker_no_such_endpoint_module",),
        out_dir=str(tmp_path / "artifacts"),
        work=str(slot / "work"),
        report=str(slot / pool.ENTRY_REPORT_NAME),
        cache_dir=str(tmp_path / "cache"))
    job_path = slot / "job.json"
    job_path.write_bytes(msgspec.json.encode(job))

    proc = subprocess.run(
        pool.child_argv(job_path), capture_output=True, timeout=600,
        env=pool.child_env(str(tmp_path / "cache")))
    assert proc.returncode == pool.EXIT_REFUSED, proc.stderr[-2000:]
    report = msgspec.json.decode(
        (slot / pool.ENTRY_REPORT_NAME).read_bytes(), type=pool.EntryReport)
    assert report.status == pool.REFUSED
    assert report.entry == "share-000"
    assert "compile target" in report.detail or "not in this image" in report.detail
    assert not report.classes


def test_a_malformed_job_is_a_wiring_defect_not_a_retry(tmp_path: Path) -> None:
    bad = tmp_path / "job.json"
    bad.write_bytes(b"{{")
    proc = subprocess.run(
        pool.child_argv(bad), capture_output=True, timeout=120)
    assert proc.returncode == pool.EXIT_BAD_JOB


# ---------------------------------------------------------------------------
# The width policy — the one surface that must hold for pods this box is not
# ---------------------------------------------------------------------------


#: A pod fat enough that CPU and host RAM are both out of the way, so a case
#: can isolate the ONE bound it is about.
#
# pgw#1175 / §4.33: the third bound is GONE. K used to divide free VRAM by a
# per-entry device ask whose only production source was
# `mint_budget.co_residency().need_bytes` — the mint child's whole
# co-residency estimate, led by the PARENT's resident weights, for a child
# that has held no weights since `fc77b923`. The rows that exercised it are
# deleted with it rather than re-pointed: they asserted the arithmetic of a
# quantity nobody may compute any more.
_ROOMY = dict(vcpus=64, available_bytes=512 * 1024**3, device_lock=True)


@pytest.mark.parametrize(
    "case,over,expect",
    [
        # A 4-vCPU pod is honestly SERIAL: 2 cores after serving headroom, 2
        # cores per entry. pgw#809 predicted this; it must not be papered over.
        ("cpu-bound 4 vCPU", dict(vcpus=4), 1),
        ("cpu-bound 8 vCPU", dict(vcpus=8), 3),
        ("cpu-bound 16 vCPU", dict(vcpus=16), 7),
        # Ceiling, not a bigger number, on a very fat host.
        ("fat host hits the ceiling", {}, pool.MAX_ENTRY_WORKERS),
        # Host RAM: 10 GiB available - 4 GiB reserve = 6 GiB / 3 GiB.
        ("host-RAM-bound", dict(available_bytes=10 * 1024**3), 2),
        # Never wider than there are entries to compile.
        ("3 entries", dict(entries=3), 3),
        # A single-entry cell never pays for a pool.
        ("1 entry", dict(entries=1), 1),
        # pgw#1175: a MEASURED per-entry RSS narrower than the 3 GiB default
        # buys width — the one per-entry footprint that still divides.
        ("measured 1 GiB per entry",
         dict(available_bytes=14 * 1024**3, peak_rss_bytes=1024**3), 8),
    ],
)
def test_width_is_derived_from_the_pod_not_the_host(
    case: str, over: dict, expect: int,
) -> None:
    kwargs = dict(_ROOMY)
    entries = over.pop("entries", 18)
    kwargs.update(over)
    width = pool.entry_workers(entries, **kwargs)   # type: ignore[arg-type]
    assert width.workers == expect, f"{case}: {width.reason}"
    assert width.reason


def test_an_unreadable_host_does_not_license_a_wide_pool() -> None:
    """No memory answer means K=1. A pool that widened on ignorance would
    OOM-kill the serving process it is supposed to be sharing with."""
    width = pool.entry_workers(
        18, vcpus=64, available_bytes=0, device_lock=True)
    assert width.workers == 1
    # pgw#877: the footprint is supplied so this pins the MEMORY bound alone.
    # Without it the width would be 1 for two reasons at once and the test
    # would pass while proving neither.
    assert width.binding == "host-memory", width.reason


def test_without_the_gpu_benchmark_lock_a_gpu_cell_stays_serial() -> None:
    """The pool's safety interlock, as a WIDTH decision.

    An AOTI compile picks kernel configs by timing them on the card. Two
    children timing at once measure each other's contention and bake the loser
    into an artifact whose cell key does not move — a silently slower cell
    under a good cell's identity. If torch cannot serialize those timings, the
    only safe width is 1.
    """
    kwargs = dict(_ROOMY)
    kwargs["device_lock"] = False
    # pgw#1175: PRESENCE is the only thing K still asks the card, and it asks
    # it for exactly this bound. Stated by the test because this box has no
    # usable driver, and because a bound that silently stops firing when the
    # probe changes is the class of defect the width record exists to prevent.
    real_card = pool._has_card
    try:
        pool._has_card = lambda: True
        assert pool.entry_workers(18, **kwargs).workers == 1  # type: ignore[arg-type]
        assert "benchmark" in pool.entry_workers(
            18, **kwargs).reason                              # type: ignore[arg-type]
        # ... and a card-less (CPU) cell is not held back by it: there is no
        # device to benchmark on and nothing to perturb.
        pool._has_card = lambda: False
        assert pool.entry_workers(
            18, **kwargs).workers > 1                         # type: ignore[arg-type]
    finally:
        pool._has_card = real_card


def test_the_cap_narrows_and_never_widens() -> None:
    wide = pool.entry_workers(18, **_ROOMY)               # type: ignore[arg-type]
    assert pool.entry_workers(
        18, limit=2, **_ROOMY).workers == 2               # type: ignore[arg-type]
    assert pool.entry_workers(
        18, limit=99, **_ROOMY).workers == wide.workers   # type: ignore[arg-type]


def test_the_width_and_its_inputs_ride_the_telemetry() -> None:
    """pgw#809 constraint 6: a mint's wall clock is uninterpretable without
    the K it ran at — two mints of the same cell legitimately differ 4x."""
    facts = pool.entry_workers(18, **_ROOMY).facts()      # type: ignore[arg-type]
    assert facts["entry_workers"] >= 1
    for key in ("entries", "vcpus", "cpu_workers", "mem_workers",
                "available_bytes", "per_entry_rss_bytes",
                "device_lock", "width_reason"):
        assert key in facts
    # pgw#1175: and the deleted terms must not creep back as telemetry that
    # somebody later divides by.
    assert not [k for k in facts if "device" in k and k != "device_lock"]


# ---------------------------------------------------------------------------
# The device-benchmark interlock
# ---------------------------------------------------------------------------


def test_the_gpu_benchmark_lock_serializes_real_processes(
    tmp_path: Path,
) -> None:
    """The interlock, exercised as a real cross-process race.

    Two OS processes each take the lock N times around a short critical
    section that appends to a shared file. If the lock works, the file's
    enter/leave markers nest perfectly; if it does not, they interleave.
    ``flock`` (not a multiprocessing primitive) is what makes a SIGKILLed
    child release by dying — the OOM killer is how a compile child is expected
    to go.
    """
    from gen_worker import aot_device_lock

    lock_path = tmp_path / "dev.lock"
    trace = tmp_path / "trace.txt"
    script = tmp_path / "racer.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "from gen_worker.aot_device_lock import DeviceBenchmarkLock\n"
        "lock = DeviceBenchmarkLock(Path(sys.argv[1]))\n"
        "trace = Path(sys.argv[2]); tag = sys.argv[3]\n"
        "for _ in range(12):\n"
        "    with lock.hold():\n"
        "        with trace.open('a') as fh: fh.write(f'+{tag}\\n')\n"
        "        time.sleep(0.01)\n"
        "        with trace.open('a') as fh: fh.write(f'-{tag}\\n')\n")
    procs = [
        subprocess.Popen([sys.executable, str(script), str(lock_path),
                          str(trace), tag])
        for tag in ("a", "b", "c")
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    lines = trace.read_text().split()
    assert len(lines) == 3 * 12 * 2
    held = ""
    for line in lines:
        if line.startswith("+"):
            assert held == "", (
                f"{line} entered while {held} still held it — the pool would "
                f"be benchmarking two graph classes against each other")
            held = line[1:]
        else:
            assert held == line[1:]
            held = ""

    # Reentrant from one thread: inductor's benchmark helpers delegate to one
    # another, and upstream's hook contract requires nesting to be safe.
    lock = aot_device_lock.DeviceBenchmarkLock(tmp_path / "reentrant.lock")
    with lock.hold():
        with lock.hold():
            pass
    assert lock.holds == 1


def test_the_lock_installs_on_torchs_own_hook() -> None:
    """Not a monkeypatch: torch 2.13 ships
    ``set_gpu_benchmark_lock_context`` and decorates every ``benchmark_gpu``
    with ``@gpu_benchmark_lock``. If a future torch drops it, the pool must
    NOTICE rather than silently benchmark against itself — which is why
    ``supported()`` gates the width."""
    from torch._inductor.runtime import benchmarking

    from gen_worker import aot_device_lock

    assert aot_device_lock.supported()
    assert hasattr(benchmarking, "set_gpu_benchmark_lock_context")
    for name in ("benchmark_gpu", "benchmark_gpu_with_cuda_graph"):
        assert hasattr(benchmarking.Benchmarker, name)


def test_the_worker_cannot_override_tcg_compiler_policy() -> None:
    """Compiler options have one owner: TCG. The worker wire and public mint
    seam cannot recreate caller-selected Inductor policy."""
    import inspect

    from gen_worker import aot_mint

    assert "inductor_configs" not in pool.EntryJob.__struct_fields__
    assert "inductor_configs" not in inspect.signature(
        pool.EntryCompilePool).parameters
    assert "inductor_configs" not in inspect.signature(
        aot_mint.mint_graph_classes).parameters


def test_parallelism_only_changes_the_child_environment(tmp_path: Path) -> None:
    """The pool changes when classes compile, never the graph declaration."""
    env = pool.child_env(str(tmp_path / "cache"))
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "cache")
