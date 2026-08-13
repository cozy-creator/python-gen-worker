"""pgw#809: the compiled graph-compile pool — two scenarios, driven for real.

Scenario-shaped rather than per-behaviour: ONE integration test drives the
real :class:`CompiledGraphCompilePool` through a real multi-compiled graph mint-to-completion
(real ``torch.export`` programs, real ``aot_compile`` children, real
``package_aoti`` assembly) and asserts everything that must hold about a
successful pool run at once; a second drives a real mint-WITH-FAILURE and
asserts everything that must hold when one compiled graph dies mid-pool. The width
policy is the one piece that must be exercised across pods the box cannot
be, so it is a table.

What the failure scenario is really testing is the thing that makes a pool
dangerous: 18 children, one dies, and the other 17 plus every ``cc1plus``
they spawned have to go with it. A leak there is a serving pod that keeps
burning CPU against a compiled graph nobody will ever adopt — so the assertion is a
PROCESS SWEEP of the real process table, not a mocked call count.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from gen_worker import aot_compile_pool as pool
from harness.progress_wait import Cadence, await_progress

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

#: Wide enough that the compile is real work (a few seconds of codegen and a
#: real g++ invocation) and small enough that the suite stays cheap.
_HIDDEN = 96


def _program(seed: int) -> Any:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(_HIDDEN, _HIDDEN)
            self.b = torch.nn.Linear(_HIDDEN, _HIDDEN)

        def forward(self, x: Any) -> Any:
            y = torch.relu(self.a(x)) * (1.0 + seed)
            return torch.tanh(self.b(y)) + y

    return torch.export.export(Tiny(), (torch.randn(4, _HIDDEN),))


def _compiled_graphs(n: int) -> List[Tuple[str, Any]]:
    # Deliberately NOT in sorted order: the pool must assemble by compiled graph name,
    # and a list that was already sorted could not tell the difference.
    names = [f"unet/adapter=true/dim={i}" for i in range(n)]
    return [(names[i], _program(i)) for i in reversed(range(n))]


def _descendants(pid: int) -> List[int]:
    """Every live process whose group is this pid's — the orphan sweep."""
    out: List[int] = []
    for compiled_graph in Path("/proc").iterdir():
        if not compiled_graph.name.isdigit():
            continue
        try:
            stat = (compiled_graph / "stat").read_text()
        except OSError:
            continue
        # ... pid (comm) state ppid pgrp ...
        tail = stat.rsplit(")", 1)[-1].split()
        if len(tail) < 3:
            continue
        try:
            if int(tail[2]) == pid:      # pgrp
                out.append(int(compiled_graph.name))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# Scenario 1: a mint that completes
# ---------------------------------------------------------------------------


def test_pool_mints_a_multi_compiled_graph_compiled_graph(tmp_path: Path) -> None:
    """Four compiled graphs, K=2, through the real children and into one ``.pt2``.

    Asserts, in one run: every compiled graph comes back; the loose files exist and
    are readable by the PARENT (the shared cache dir is the whole reason
    they are); assembly is ordered by compiled graph NAME and not by completion; the
    pool really did run K children at once (structurally, not by clock);
    the staged programs are swept; the
    per-compiled graph seconds and the child peak RSS are recorded (the memory bound
    is measured, not assumed); and ``package_aoti`` accepts the result.
    """
    from torch._inductor.package import package_aoti

    compiled_graphs = _compiled_graphs(4)
    # The width is STATED, not derived: a 4-vCPU CI runner honestly derives
    # K=1, and this scenario would then pass while exercising no pool at all.
    width = pool.compiled_graph_workers(
        len(compiled_graphs), limit=2, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    assert width.workers == 2
    box = pool.CompiledGraphCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))

    out = box.compile(compiled_graphs)

    assert set(out) == {name for name, _ in compiled_graphs}
    assert list(out) == sorted(out), (
        "the compiled_graph must assemble by compiled_graph NAME; a dict ordered by completion "
        "makes the artifact depend on which child finished first")
    for name, files in out.items():
        assert files, f"compiled_graph {name!r} came back with no files"
        for path in files:
            assert Path(path).exists(), (
                f"{name}: {path} is not visible to the parent — the pool's "
                f"shared TORCHINDUCTOR_CACHE_DIR is how loose files travel")

    assert set(box.compiled_graph_seconds) == set(out)
    # Overlap is asserted STRUCTURALLY — K processes really were alive at
    # once — never as a wall-clock speedup. A `wall < serial_sum` assertion
    # measures the runner's spare CPU, not this code: on a 4-vCPU CI box it
    # read 85.504 < 85.22 and failed a release. The speedup claim belongs in
    # pgw#809's measured tables, where it is CPU-seconds and byte-identity.
    assert box.peak_concurrency == width.workers, (
        f"pool never reached its own width: saw {box.peak_concurrency} "
        f"concurrent children, K={width.workers}")
    assert box.peak_rss_bytes > 0, (
        "per-compiled-graph peak RSS is what bounds K by memory; an unmeasured peak "
        "makes the width policy a guess")

    staged = list((tmp_path / "pool").rglob("program.pt2"))
    assert not staged, f"staged programs left on disk: {staged}"

    package = package_aoti(str(tmp_path / "compiled_graph.pt2"), dict(out))
    assert Path(package).exists() and Path(package).stat().st_size > 0


# ---------------------------------------------------------------------------
# Scenario 2: a mint where one compiled graph dies
# ---------------------------------------------------------------------------


def test_one_failing_compiled_graph_fails_the_mint_and_takes_its_siblings(
    tmp_path: Path,
) -> None:
    """One compiled graph cannot compile. The mint must fail NAMING it, the siblings
    must be torn down group-wide, and nothing may survive the call.

    The failure is injected the only honest way: a job whose exported program
    is corrupt, so the real child really does exit non-zero on the real code
    path. Everything else in the run is real.
    """
    compiled_graphs = _compiled_graphs(4)
    doomed = compiled_graphs[1][0]
    width = pool.compiled_graph_workers(
        len(compiled_graphs), limit=4, vcpus=16, available_bytes=64 * 1024**3,
        device_lock=True)
    box = pool.CompiledGraphCompilePool(
        tmp_path / "pool", width=width,
        inductor_configs={"compile_threads": 2},
        cache_dir=str(tmp_path / "cache"))

    real_stage = box._stage

    def stage(compiled_graph: str, program: Any, index: int) -> Any:
        job, job_path = real_stage(compiled_graph, program, index)
        if compiled_graph == doomed:
            Path(job.program).write_bytes(b"not an exported program")
        return job, job_path

    box._stage = stage           # type: ignore[method-assign]

    before = set(_descendants(os.getpid()))
    with pytest.raises(pool.CompiledGraphCompileFailed) as caught:
        box.compile(compiled_graphs)

    exc = caught.value
    assert exc.compiled_graph == doomed
    assert doomed in str(exc), (
        "a pool of 18 that fails anonymously is undebuggable — the compiled_graph "
        "name is the whole diagnostic on a pod with no logs")
    assert "exited" in str(exc)

    # No orphans: the pool kills process GROUPS, so every child's own
    # inductor workers and g++ go with it. Waited on PROGRESS — each pid that
    # disappears is an advance — rather than on a clock, because a clock here
    # asserts the runner's speed and a wedged teardown must still FAIL rather
    # than be outrun (pgw#795).
    await_progress(
        lambda: tuple(
            sorted(p for p in _descendants(os.getpid()) if p not in before)),
        lambda leaked: not leaked,
        what="the failed mint's sibling children to be reaped",
        cadence=Cadence(),
        render=lambda leaked: f"{len(leaked)} orphan(s): {list(leaked)[:8]}",
    )

    staged = list((tmp_path / "pool").rglob("program.pt2"))
    assert not staged, f"staged programs left on disk: {staged}"


def test_a_named_child_refusal_is_reported_not_swallowed(tmp_path: Path) -> None:
    """``aot_compile_child`` run by hand on a bad job: non-zero exit, a typed
    report, and the refusal sentence on disk. The boundary is a file exactly
    so a mint that fails on compiled graph 13 of 18 is reproducible without the
    pipeline."""
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / "program.pt2").write_bytes(b"garbage")
    job = pool.CompiledGraphJob(
        compiled_graph="unet/adapter=true/dim=0",
        program=str(slot / "program.pt2"),
        report=str(slot / pool.COMPILED_GRAPH_REPORT_NAME),
        inductor_configs={"compile_threads": 1},
        cache_dir=str(tmp_path / "cache"))
    job_path = slot / "job.json"
    import msgspec

    job_path.write_bytes(msgspec.json.encode(job))

    proc = subprocess.run(
        pool.child_argv(job_path), capture_output=True, timeout=600,
        env=pool.child_env(str(tmp_path / "cache")))
    assert proc.returncode == pool.EXIT_REFUSED, proc.stderr[-2000:]
    report = msgspec.json.decode(
        (slot / pool.COMPILED_GRAPH_REPORT_NAME).read_bytes(), type=pool.CompiledGraphReport)
    assert report.status == pool.REFUSED
    assert report.compiled_graph == job.compiled_graph
    assert "exported program" in report.detail
    assert not report.files


def test_a_malformed_job_is_a_wiring_defect_not_a_retry(tmp_path: Path) -> None:
    bad = tmp_path / "job.json"
    bad.write_bytes(b"{")
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
# per-compiled graph device ask whose only production source was
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
        # cores per compiled graph. pgw#809 predicted this; it must not be papered over.
        ("cpu-bound 4 vCPU", dict(vcpus=4), 1),
        ("cpu-bound 8 vCPU", dict(vcpus=8), 3),
        ("cpu-bound 16 vCPU", dict(vcpus=16), 7),
        # Ceiling, not a bigger number, on a very fat host.
        ("fat host hits the ceiling", {}, pool.MAX_COMPILED_GRAPH_WORKERS),
        # Host RAM: 10 GiB available - 4 GiB reserve = 6 GiB / 3 GiB.
        ("host-RAM-bound", dict(available_bytes=10 * 1024**3), 2),
        # Never wider than there are compiled graphs to compile.
        ("3 compiled_graphs", dict(compiled_graphs=3), 3),
        # A single-compiled graph compiled graph never pays for a pool.
        ("1 compiled_graph", dict(compiled_graphs=1), 1),
        # pgw#1175: a MEASURED per-compiled graph RSS narrower than the 3 GiB default
        # buys width — the one per-compiled graph footprint that still divides.
        ("measured 1 GiB per compiled_graph",
         dict(available_bytes=14 * 1024**3, peak_rss_bytes=1024**3), 8),
    ],
)
def test_width_is_derived_from_the_pod_not_the_host(
    case: str, over: dict, expect: int,
) -> None:
    kwargs = dict(_ROOMY)
    compiled_graphs = over.pop("compiled_graphs", 18)
    kwargs.update(over)
    width = pool.compiled_graph_workers(compiled_graphs, **kwargs)   # type: ignore[arg-type]
    assert width.workers == expect, f"{case}: {width.reason}"
    assert width.reason


def test_an_unreadable_host_does_not_license_a_wide_pool() -> None:
    """No memory answer means K=1. A pool that widened on ignorance would
    OOM-kill the serving process it is supposed to be sharing with."""
    width = pool.compiled_graph_workers(
        18, vcpus=64, available_bytes=0, device_lock=True)
    assert width.workers == 1
    # pgw#877: the footprint is supplied so this pins the MEMORY bound alone.
    # Without it the width would be 1 for two reasons at once and the test
    # would pass while proving neither.
    assert width.binding == "host-memory", width.reason


def test_without_the_gpu_benchmark_lock_a_gpu_compiled_graph_stays_serial() -> None:
    """The pool's safety interlock, as a WIDTH decision.

    An AOTI compile picks kernel configs by timing them on the card. Two
    compiled graphs timing at once measure each other's contention and bake the loser
    into an artifact whose compiled graph key does not move — a silently slower compiled graph
    under a good compiled graph's identity. If torch cannot serialize those timings, the
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
        assert pool.compiled_graph_workers(18, **kwargs).workers == 1  # type: ignore[arg-type]
        assert "benchmark" in pool.compiled_graph_workers(
            18, **kwargs).reason                              # type: ignore[arg-type]
        # ... and a card-less (CPU) compiled graph is not held back by it: there is no
        # device to benchmark on and nothing to perturb.
        pool._has_card = lambda: False
        assert pool.compiled_graph_workers(
            18, **kwargs).workers > 1                         # type: ignore[arg-type]
    finally:
        pool._has_card = real_card


def test_the_cap_narrows_and_never_widens() -> None:
    wide = pool.compiled_graph_workers(18, **_ROOMY)               # type: ignore[arg-type]
    assert pool.compiled_graph_workers(
        18, limit=2, **_ROOMY).workers == 2               # type: ignore[arg-type]
    assert pool.compiled_graph_workers(
        18, limit=99, **_ROOMY).workers == wide.workers   # type: ignore[arg-type]


def test_the_width_and_its_inputs_ride_the_telemetry() -> None:
    """pgw#809 constraint 6: a mint's wall clock is uninterpretable without
    the K it ran at — two mints of the same compiled graph legitimately differ 4x."""
    facts = pool.compiled_graph_workers(18, **_ROOMY).facts()      # type: ignore[arg-type]
    assert facts["compiled_graph_workers"] >= 1
    for key in ("compiled_graphs", "vcpus", "cpu_workers", "mem_workers",
                "available_bytes", "per_compiled_graph_rss_bytes",
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
    child release by dying — the OOM killer is how an compiled graph child is expected
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
                f"be benchmarking two compiled_graphs against each other")
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


def test_the_autotune_posture_the_mint_actually_compiles_under() -> None:
    """The premise the interlock rests on, READ FROM THE PIN rather than
    assumed — because the two branches have very different device footprints.

    ``get_cpp_wrapper_config`` resolves an UNSET
    ``triton.autotune_at_compile_time`` to ``has_triton() and
    V.aot_compilation`` — True for AOTI. So the mint takes the ONE-pass
    autotune-block path, not the two-pass "run the whole model on real
    inputs" path below it. Both benchmark on the card; only the second holds
    a full activation set. If a torch bump flips this default, the per-compiled graph
    VRAM figure in the width policy is wrong and this test says so.
    """
    import torch._inductor.config as inductor_config
    from gen_worker import aot_mint

    assert inductor_config.triton.autotune_at_compile_time is None, (
        "the pin no longer leaves autotune_at_compile_time unset — re-read "
        "compile_fx.get_cpp_wrapper_config before trusting the VRAM policy")
    resolved = aot_mint._compiled_graph_configs(None)
    assert "triton.autotune_at_compile_time" not in resolved, (
        "the mint must not pin this: the resolution is torch's, and pinning "
        "it False switches on the whole-model-on-real-inputs pass")


def test_parallelism_is_not_sealed(tmp_path: Path) -> None:
    """pgw#757 established ``compile_threads`` as outside compiled graph identity; the
    same argument covers K, and the digest check is how it is VERIFIED rather
    than argued. The pool changes WHEN compiled graphs compile, never what."""
    from gen_worker import env_seal

    # pgw#929: the loop that used to sit here set GEN_WORKER_AOT_COMPILED_GRAPH_WORKERS
    # and asserted the digest was unmoved. NOTHING IN src/ HAS EVER READ THAT
    # NAME — the live width constant is `aot_compile_pool.MAX_COMPILED_GRAPH_WORKERS` —
    # so the assertion held for the one reason that proves nothing (C1: a test
    # exercising a knob that does not exist). The real property, that the
    # shared cache DIR is a location and not a recipe, is asserted below
    # against a value the code actually reads.
    base = env_seal.inductor_config_digest()
    env = pool.child_env(str(tmp_path / "cache"))
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "cache")
    assert env_seal.inductor_config_digest() == base, (
        "the shared inductor cache dir is a LOCATION, not a recipe; if it "
        "reached the seal, every pod would key its own compiled_graphs")
