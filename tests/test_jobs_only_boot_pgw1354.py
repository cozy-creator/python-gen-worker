"""pgw#1354 (JOBS program, P0): a JOBS-ONLY IMAGE BOOTS, and a boot that finds
nothing SAYS SO on the wire.

Measured on rented hardware before this landed, twice, on two card families
(e2e#1890 ch.3): the first pod ever booted off a jobs-only image — `conversion`
0.12.4, **27 jobs / 0 functions**, gen-worker 0.122.0 — died at
`entrypoint.py:523` with `deaths_before_hello=2`, `last_cause=exit:1`, no
reason class, `ready_at` NULL. `get_modules_from_manifest` walked
`manifest["functions"]` and nothing else, so a package that declares its
modules in `jobs[].module` yielded ZERO user modules; pgw#1324's `worker.py`
fix (`if not specs and not jobs`) is correct and sits one frame too late,
because `collect_jobs([])` over an empty module list finds nothing either.

Three properties, each with the one-line edit that turns it RED:

1. **THE HEADLINE.** A REAL jobs-only `endpoint.lock` — baked by the REAL
   `python -m gen_worker.discovery` out of a REAL package of `@job`s — yields
   its modules and BOOTS through the REAL `_run_main`, with every job in the
   executor's inventory. RED by restoring the `functions`-only loop in
   `get_modules_from_manifest` (that is master exactly: 0 modules, exit 1).
   Functions-only and mixed locks are the controls: extending the walk must
   not change what a function-shaped release discovers.

2. **THE CUDA PROBE IS NOT BLIND TO JOBS.** `should_probe_cuda` read
   `functions[]` too, so a jobs-only GPU image was never health-probed and the
   gw#529 bad-host guard silently did not apply to the whole jobs program.
   RED by restoring `functions = (manifest or {}).get("functions", [])`.

3. **THE SILENT DEATH DIES TOO.** The no-modules exit was a bare
   `logger.error` + `return 1` — the ONE fatal in `_run_main` that did not
   dial. RunPod exposes no container-logs API, so the hub saw `exit:1` and
   condemned the pod `[hardware-unsuitable]` with empty driver/gpu: a boot bug
   wearing a hardware verdict, and it cost a paid pod to find. It must now dial
   `_log_worker_fatal("no_user_modules", ..., settings=settings)`, the report
   must REACH the worker_fatal wire carrier (pre-Hello reporting is deliberately
   preserved, th#2075), and the detail must NAME the gap — declarations found
   vs modules found. RED by deleting the `_log_worker_fatal` call, or by
   dropping `settings=` from it (which silently makes the dial a no-op).

Nothing here is mocked that matters: the packages are real, the lock is baked
by the real build command in a real subprocess, the registry walk is real, and
`Worker.__init__` is the production constructor. Only the seams that need
hardware or a hub are stood down — the CUDA device, the cache-dir mkdir, and
`Worker.run`'s gRPC loop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from gen_worker import worker_main as entrypoint
from gen_worker.cuda_probe import CudaProbeResult, should_probe_cuda
# The SAME class object `entrypoint` holds (`from .worker import Worker`), so
# patching it here is patching what the boot constructs.
from gen_worker.worker import Worker

pytest.importorskip("torch")


# --------------------------------------------------------------------------
# The packages, and the REAL bake
# --------------------------------------------------------------------------

#: The live shape this issue was measured on: `conversion` 0.12.4 declares 27
#: jobs and zero functions, 10 of them on a card.
JOB_COUNT = 27
GPU_JOB_COUNT = 10
#: Spread across two modules, so the walk is proved to UNION and de-duplicate
#: rather than to return whatever the first row said.
JOB_MODULES = ("main", "extra")

_PREAMBLE = """
import msgspec
from gen_worker import JobContext, RequestContext, Resources, endpoint, job

class In_(msgspec.Struct):
    steps: int = 1

class Out_(msgspec.Struct):
    ok: bool = True
"""


def _job_source(index: int, *, gpu: bool) -> str:
    resources = "Resources(gpu=True)" if gpu else "Resources(vcpus=2)"
    return textwrap.dedent(f"""
        @job(resources={resources})
        def convert_{index:02d}(ctx: JobContext, spec: In_) -> Out_:
            return Out_()
    """)


def _endpoint_source(name: str) -> str:
    return textwrap.dedent(f"""
        @endpoint
        class {name}:
            def generate(self, ctx: RequestContext, p: In_) -> Out_:
                return Out_()
    """)


def _write_package(
    root: Path, package: str, *, jobs: int, gpu_jobs: int, endpoints: bool,
) -> None:
    pkg = root / package
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    bodies: Dict[str, List[str]] = {name: [_PREAMBLE] for name in JOB_MODULES}
    for index in range(jobs):
        module = JOB_MODULES[index % len(JOB_MODULES)]
        bodies[module].append(_job_source(index, gpu=index < gpu_jobs))
    if endpoints:
        bodies["main"].append(_endpoint_source("Gen"))
    for module, parts in bodies.items():
        (pkg / f"{module}.py").write_text("".join(parts))
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{package}"\nversion = "0.0.0"\n'
        f'[tool.gen_worker]\nmain = "{package}.main"\n'
    )


def _bake(root: Path) -> Path:
    """The REAL build step from the generated Dockerfile:
    `python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "gen_worker.discovery"],
        cwd=str(root), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"the real bake refused:\n{proc.stderr}")
    lock = root / ".tensorhub" / "endpoint.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(proc.stdout)
    return lock


def _baked(
    tmp_path_factory: pytest.TempPathFactory,
    package: str,
    *,
    jobs: int,
    gpu_jobs: int,
    endpoints: bool,
) -> Tuple[Path, Path]:
    root = tmp_path_factory.mktemp(package)
    _write_package(
        root, package, jobs=jobs, gpu_jobs=gpu_jobs, endpoints=endpoints)
    return root, _bake(root)


@pytest.fixture(scope="module")
def jobs_only(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    """te#218's shape: 27 jobs, 10 on a card, ZERO `@endpoint`s."""
    return _baked(
        tmp_path_factory, "conv_jobs",
        jobs=JOB_COUNT, gpu_jobs=GPU_JOB_COUNT, endpoints=False)


@pytest.fixture(scope="module")
def functions_only(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    """The control: the shape every walk in this file was written for."""
    return _baked(
        tmp_path_factory, "fn_only", jobs=0, gpu_jobs=0, endpoints=True)


@pytest.fixture(scope="module")
def mixed(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    """The other control: one package carrying both, published once."""
    return _baked(
        tmp_path_factory, "both_kinds", jobs=4, gpu_jobs=2, endpoints=True)


def _manifest(lock: Path) -> Dict[str, Any]:
    manifest = entrypoint.load_manifest(lock)
    assert manifest is not None, f"the baked lock at {lock} did not load"
    return manifest


# --------------------------------------------------------------------------
# 1. the headline — discovery, then the whole boot
# --------------------------------------------------------------------------

def test_the_baked_jobs_only_lock_really_is_jobs_only(
    jobs_only: Tuple[Path, Path],
) -> None:
    """The fixture is the defect's shape, stated by the artifact itself —
    otherwise the arms below could pass against a lock that quietly grew a
    function."""
    manifest = _manifest(jobs_only[1])
    assert manifest.get("functions") in (None, [])
    assert len(manifest["jobs"]) == JOB_COUNT
    assert sum(
        1 for j in manifest["jobs"] if (j.get("resources") or {}).get("gpu")
    ) == GPU_JOB_COUNT


def test_a_jobs_only_manifest_yields_its_modules(
    jobs_only: Tuple[Path, Path],
) -> None:
    """THE headline. RED on master: `[]`."""
    modules = entrypoint.get_modules_from_manifest(_manifest(jobs_only[1]))
    assert modules == [f"conv_jobs.{name}" for name in sorted(JOB_MODULES)], (
        "a jobs-only package declares its modules in jobs[].module")


def test_the_function_walk_is_unchanged_by_the_extension(
    functions_only: Tuple[Path, Path], mixed: Tuple[Path, Path],
) -> None:
    """The controls: one walk over both blocks, not a different answer for the
    shape that already worked."""
    # A function row records the top-level package it is walked from; a job row
    # records its own module. The union carries both spellings, and the registry
    # walk each side runs is the one that reads its own.
    assert entrypoint.get_modules_from_manifest(
        _manifest(functions_only[1])) == ["fn_only"]
    assert entrypoint.get_modules_from_manifest(_manifest(mixed[1])) == [
        "both_kinds", "both_kinds.extra", "both_kinds.main"]


def _boot(
    monkeypatch: pytest.MonkeyPatch, root: Path, lock: Path,
) -> Dict[str, Any]:
    """Run the REAL `_run_main` against a REAL baked lock, standing down only
    the seams that need hardware or a hub."""
    observed: Dict[str, Any] = {"probed": False, "fatals": [], "dialed": []}

    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("ENDPOINT_LOCK_PATH", str(lock))
    monkeypatch.setenv("ORCHESTRATOR_PUBLIC_ADDR", "http://127.0.0.1:1")
    monkeypatch.setattr(entrypoint, "_install_stack_dump_handler", lambda: None)
    monkeypatch.setattr(entrypoint, "_establish_env_seal", lambda: {})
    monkeypatch.setattr(
        entrypoint, "_preflight_cache_dirs",
        lambda: {"model_cache_dir": str(root / "cas"),
                 "local_model_cache_dir": ""})

    def _probe(device_index: int = 0) -> CudaProbeResult:
        observed["probed"] = True
        return CudaProbeResult(ok=True)

    monkeypatch.setattr(entrypoint, "probe_cuda", _probe)

    real_fatal = entrypoint._log_worker_fatal

    def _fatal(phase: str, exc: BaseException, **kw: Any) -> None:
        observed["fatals"].append((phase, str(exc), kw))
        real_fatal(phase, exc, **kw)

    monkeypatch.setattr(entrypoint, "_log_worker_fatal", _fatal)

    # The wire carrier itself — `_log_worker_fatal` reaching it is the whole
    # point of the typed dial (th#2075 preserved pre-Hello reporting).
    from gen_worker import worker_fatal

    def _report(settings: Any, phase: str, exc: Any, *, exit_code: int) -> bool:
        observed["dialed"].append(
            worker_fatal.build_fatal_detail(phase, exc, exit_code=exit_code))
        return True

    monkeypatch.setattr(worker_fatal, "report_worker_fatal", _report)

    workers: List[Any] = []
    real_init = Worker.__init__

    def _init(self: Any, *a: Any, **kw: Any) -> None:
        real_init(self, *a, **kw)
        workers.append(self)

    monkeypatch.setattr(Worker, "__init__", _init)
    monkeypatch.setattr(Worker, "run", lambda self: 0)

    observed["code"] = entrypoint._run_main()
    observed["workers"] = workers
    return observed


def test_a_jobs_only_image_boots_with_every_job_in_its_inventory(
    monkeypatch: pytest.MonkeyPatch, jobs_only: Tuple[Path, Path],
) -> None:
    """The production boot, end to end. RED on master at `code == 1` with no
    Worker ever constructed — which is `deaths_before_hello=2` on a rented
    Blackwell, for $0."""
    root, lock = jobs_only
    observed = _boot(monkeypatch, root, lock)

    assert observed["code"] == 0, "a jobs-only image must reach the worker loop"
    assert observed["fatals"] == [], "a healthy boot dials no fatal"
    (worker,) = observed["workers"]
    assert sorted(worker.executor.job_specs) == sorted(
        f"convert-{i:02d}" for i in range(JOB_COUNT)), (
        "pgw#1324's collect_jobs walk is reachable only once the entrypoint "
        "hands it the jobs' modules")
    assert not worker.executor.specs, "this package declares no @endpoint"


def test_the_function_shaped_controls_still_boot(
    monkeypatch: pytest.MonkeyPatch,
    functions_only: Tuple[Path, Path],
    mixed: Tuple[Path, Path],
) -> None:
    root, lock = functions_only
    observed = _boot(monkeypatch, root, lock)
    assert observed["code"] == 0 and observed["fatals"] == []
    (worker,) = observed["workers"]
    assert worker.executor.specs and not worker.executor.job_specs

    monkeypatch.undo()
    root, lock = mixed
    observed = _boot(monkeypatch, root, lock)
    assert observed["code"] == 0 and observed["fatals"] == []
    (worker,) = observed["workers"]
    assert worker.executor.specs and len(worker.executor.job_specs) == 4


# --------------------------------------------------------------------------
# 2. the CUDA probe
# --------------------------------------------------------------------------

def test_a_jobs_only_gpu_image_is_cuda_probed(
    monkeypatch: pytest.MonkeyPatch, jobs_only: Tuple[Path, Path],
) -> None:
    """gw#529's bad-host guard applies to jobs. RED on master: never probed —
    a GPU job boots on a card nothing health-checked."""
    root, lock = jobs_only
    assert should_probe_cuda(_manifest(lock), cuda_build=True) is True
    assert _boot(monkeypatch, root, lock)["probed"] is True


def test_the_probe_predicate_reads_both_blocks() -> None:
    """The three states, at the predicate. A CPU-only jobs release is not
    probed; an all-GPU one always is; the mixed case defers to the installed
    torch build exactly as it does for functions."""
    cpu_jobs = {"jobs": [{"module": "m", "resources": {"gpu": False}}]}
    gpu_jobs = {"jobs": [{"module": "m", "resources": {"gpu": True}}]}
    mixed_jobs = {"jobs": [
        {"module": "m", "resources": {"gpu": True}},
        {"module": "m", "resources": {"gpu": False}},
    ]}
    assert should_probe_cuda(cpu_jobs, cuda_build=True) is False
    assert should_probe_cuda(gpu_jobs, cuda_build=False) is True
    assert should_probe_cuda(mixed_jobs, cuda_build=True) is True
    assert should_probe_cuda(mixed_jobs, cuda_build=False) is False
    # And the union is across blocks: a GPU job beside a CPU function is the
    # same mixed case, not two independent verdicts.
    assert should_probe_cuda(
        {"functions": [{"module": "m", "resources": {"gpu": False}}], **gpu_jobs},
        cuda_build=False) is False
    assert should_probe_cuda(None, cuda_build=True) is False


# --------------------------------------------------------------------------
# 3. the silent death
# --------------------------------------------------------------------------

def _empty_lock(tmp_path: Path, body: str) -> Tuple[Path, Path]:
    lock = tmp_path / "endpoint.lock"
    lock.write_text(body)
    return tmp_path, lock


def test_declarations_without_modules_dial_a_typed_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The gap this class produces: rows the wheel can count but cannot import.
    The exit stays 1 — and now it says WHY, on the wire, naming both counts."""
    root, lock = _empty_lock(tmp_path, textwrap.dedent("""
        [[jobs]]
        name = "convert-00"
        [[jobs]]
        name = "convert-01"
        [[functions]]
        name = "generate"
    """))
    observed = _boot(monkeypatch, root, lock)

    assert observed["code"] == 1
    (phase, message, kw), = observed["fatals"]
    assert phase == "no_user_modules"
    assert kw.get("settings") is not None, (
        "without settings the dial is a silent no-op — the exact defect")
    # pgw#1395: the summary is DERIVED from DECLARATION_BLOCKS, so a refusal
    # can never again name a subset of the blocks the walk reads.
    assert "1 functions, 0 entrypoints, 2 jobs" in message, (
        "the fatal must name the gap it found, not merely that it exited")
    assert "functions, entrypoints, jobs" in message, (
        "and which blocks this build walks")
    (detail,) = observed["dialed"]
    assert "no_user_modules" in detail and "exit_code=1" in detail


def test_a_missing_manifest_dials_the_same_typed_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The other gap, discriminated: no lock at all is a Dockerfile that never
    ran discovery, and it too must be readable off the hub rather than off pod
    stdout RunPod does not serve."""
    observed = _boot(monkeypatch, tmp_path, tmp_path / "absent.lock")
    assert observed["code"] == 1
    (phase, message, kw), = observed["fatals"]
    assert phase == "no_user_modules"
    assert kw.get("settings") is not None
    assert "no baked manifest" in message
    assert "gen_worker.discovery" in message
    assert observed["dialed"], "the hub must hear about this one too"


# --------------------------------------------------------------------------
# 4. pgw#1395 — the SAME defect, one hardcut later: the v2 `entrypoints[]`
#    block. Everything below is the jobs case re-run against the shape every
#    endpoint has AFTER the pgw#1382 cutover.
# --------------------------------------------------------------------------

_V2_SOURCE = """
import msgspec
from gen_worker import RequestContext, entrypoint

class In_(msgspec.Struct):
    steps: int = 1

class Out_(msgspec.Struct):
    ok: bool = True

@entrypoint
def generate(ctx: RequestContext, payload: In_) -> Out_:
    return Out_()
"""


@pytest.fixture(scope="module")
def entrypoints_only(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    """The pgw#1382 shape: ONE module-level `@entrypoint`, no `@endpoint`
    class, no `@job` — what sdxl and minimax-h3 already are on
    serverless-endpoints master."""
    root = tmp_path_factory.mktemp("v2_only")
    pkg = root / "v2_only"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(_V2_SOURCE)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "v2_only"\nversion = "0.0.0"\n'
        '[tool.gen_worker]\nmain = "v2_only.main"\n'
    )
    return root, _bake(root)


def test_the_baked_v2_lock_really_is_entrypoints_only(
    entrypoints_only: Tuple[Path, Path],
) -> None:
    """The fixture is the defect's shape, stated by the artifact. Note
    `functions = []` is emitted UNCONDITIONALLY beside the real rows — which
    is exactly why a walk that reads `functions` alone scores this release at
    zero declarations instead of refusing."""
    manifest = _manifest(entrypoints_only[1])
    assert manifest.get("functions") in (None, [])
    assert manifest.get("jobs") in (None, [])
    assert len(manifest["entrypoints"]) == 1


def test_a_v2_manifest_yields_its_modules(
    entrypoints_only: Tuple[Path, Path],
) -> None:
    """THE headline, pgw#1395. RED on master: `[]` — a v2-only image imports
    NOTHING, is never CUDA-probed, and dies naming the wrong gap."""
    assert entrypoint.get_modules_from_manifest(
        _manifest(entrypoints_only[1])) == ["v2_only.main"]


def test_the_probe_predicate_reads_the_v2_block_too() -> None:
    """gw#529's bad-host guard reaches a v2 GPU image. A model-bearing v2
    entrypoint emits `resources.gpu = true` (`entrypoints_v2._resources`), so
    the predicate has the same fact to read — it just has to look."""
    gpu_v2 = {"entrypoints": [{"module": "m", "resources": {"gpu": True}}]}
    cpu_v2 = {"entrypoints": [{"module": "m", "resources": {}}]}
    assert should_probe_cuda(gpu_v2, cuda_build=False) is True
    assert should_probe_cuda(cpu_v2, cuda_build=True) is False


def test_a_v2_boot_clears_the_module_wall_and_names_the_next_one(
    monkeypatch: pytest.MonkeyPatch, entrypoints_only: Tuple[Path, Path],
) -> None:
    """What this fix does and does NOT buy, stated rather than assumed.

    The module-load wall is gone: the boot imports the author's module instead
    of dying `no_user_modules` over a manifest declaring an entrypoint. The
    NEXT wall is the v1 serve registry — `registry.extract_specs` reads
    `__gen_worker_endpoint__` and `@entrypoint` stamps `__cozy_entrypoint__`,
    so `Worker.__init__` still finds no serve surface. That wall belongs to the
    pgw#1367/pgw#1373 serving cutover, and this arm exists so it is a NAMED
    refusal in the record rather than a discovery nobody makes twice.
    """
    root, lock = entrypoints_only
    observed = _boot(monkeypatch, root, lock)

    assert observed["code"] == 1
    phases = [phase for phase, _m, _kw in observed["fatals"]]
    assert "no_user_modules" not in phases, (
        "the module-load wall is what pgw#1395 removes")
    (phase, message, _kw), = observed["fatals"]
    assert phase == "runtime"
    assert "no @endpoint classes and no @job functions found" in message
    assert "v2_only.main" in message, (
        "the module WAS imported — the gap is the serve registry, not the walk")
