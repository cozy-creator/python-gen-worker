"""pgw#809: compile a cell's entries K-wide instead of one at a time.

A pgw#758 cell is N independent graph-class entries; an sdxl cell is 18.
``aot_mint`` exports them from the live pipeline (serial by construction —
one pipeline, one card, and the branch arm is toggled once for the whole
branchless group) and then AOTI-compiles each one at ~420 s. The compiles
share nothing: each is ``aot_compile(ep.module(), ...)`` over its own
ExportedProgram, producing its own loose files, combined afterwards by
``package_aoti``. Serially that is ~2 h; K-wide it is ``ceil(N/K)`` x
per-entry.

Why processes, and not threads
------------------------------
MEASURED, not assumed: four concurrent ``aot_compile`` calls in ONE process
(torch 2.13.0+cu130) produced ONE usable result and three distinct internal
failures — ``AssertionError: CURRENT_PATCHER is None in finally block``,
``KeyError: 'custom'`` in ``fx.traceback.annotate``, and a fake-tensor
propagation crash. Inductor's compile path keeps process-global mutable
state (``torch.fx.traceback`` current-meta stack, dynamo's patcher). A
thread pool here is not slower, it is WRONG, and it fails nondeterministically
— the worst shape. So the unit of parallelism is an OS process, which is also
what pgw#784 already established for the mint itself.

The handoff, and its cost
-------------------------
A child cannot inherit the exported program (``fork`` is banned after CUDA
init, pgw#784), so it arrives on disk: ``torch.export.save`` in the parent,
``torch.export.load`` in the child. Two facts make that affordable:

* **It is byte-exact.** A compile after the roundtrip produces a
  ``wrapper.cpp`` byte-identical to the in-process compile, and lands under
  the SAME inductor cache hash — the cache key is the graph, not the process.
* **It is off the critical path.** Only the FIRST save is serial; every later
  save overlaps a child that is already compiling, and a save (~16 s at 2.5 GB)
  is 4 % of a compile (~420 s). The parent is never the bottleneck.

The parent keeps its own in-memory program regardless — every package-side
gate (``program_package_drift``, ``eliminated_constants``, ``input_contract``)
runs against the parent's real program and the child's package, so a child
that diverged is caught by gates that already exist, named by entry.

What this does NOT change
-------------------------
Cell identity. Parallelism is not sealed (pgw#757 established
``compile_threads`` as non-identity by the same argument, and the digest check
is re-run here): the pool changes WHEN entries compile, never what they
compile. Assembly is ordered by ENTRY NAME, not completion, so a cell minted
at K=4 is byte-identical to one minted at K=1.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple)

import msgspec

from . import aot_device_lock
from .postmortem import effective_cpu_count

logger = logging.getLogger(__name__)

ENTRY_CHILD_MODULE = "gen_worker.aot_compile_child"

#: Report file each entry child writes before exiting.
ENTRY_REPORT_NAME = "report.json"

# ---------------------------------------------------------------------------
# Width policy (pgw#809 constraint 1: NEVER os.cpu_count, NEVER a hardcoded K)
# ---------------------------------------------------------------------------

#: Cores the pool must leave to the serving process. A mint runs on a SERVING
#: pod under the pgw#784 contract ("the worker is available the entire time"),
#: so the eager forward, the 10 s heartbeat and the transport all have to keep
#: their share. Two is the measured floor for a worker that is serving: one
#: for the asyncio loop carrying the beat, one for the eager forward's host
#: side. Below that the pool is not allowed to run at all (K falls to 1, which
#: is the in-process serial path and adds nothing).
SERVING_HEADROOM_CPUS = 2

#: Cores one entry child occupies ON AVERAGE over its life. NOT
#: ``compile_threads``: pgw#793 measured that an AOTI compile is dominated by
#: two strictly SINGLE-THREADED phases — inductor's Python codegen (25 %) and
#: the one g++ invocation on the wrapper TU (46 %) — with ``compile_threads``
#: parallelism confined to the Triton kernel compiles in between. So the
#: instantaneous ask is 1 core for ~71 % of the entry and up to
#: ``compile_threads`` for the rest; 2 is that average, rounded up.
#: ``K * compile_threads`` is deliberately ALLOWED to exceed the budget —
#: clamping on the peak would size the pool for a phase that is a quarter of
#: the work and leave the box idle through the other three quarters.
CPUS_PER_ENTRY_WORKER = 2

#: Hard ceiling regardless of how fat the pod is. Past this the shared
#: inductor cache, the page cache and the disk holding N saved programs stop
#: behaving, and the remaining serial terms (export, package, pack) dominate
#: anyway: at 18 entries, K=8 is already ceil(18/8)=3 rounds against K=6's 3.
#:
#: RE-PRICED for regional (pgw#817 / pgw#812 S7). Regional does NOT multiply
#: with K — it changes what K is dividing:
#:
#: * The entry COUNT goes UP, not down: one entry per (plan, block class), so
#:   sdxl's 18 whole-graph entries become 18 x 2 = 36 block entries. #812 S7's
#:   worry ("once a family's cell is 2 entries instead of 18, K > 2 buys
#:   nothing") describes a cell whose SHAPE rows also collapse; on a
#:   static-rows family like sdxl the rows stay and the classes multiply.
#: * Each entry is ~14x cheaper (19.4 s vs 274.7 s measured), so the SERIAL
#:   terms this ceiling was defending against — export, package, pack — are a
#:   much larger fraction of a regional mint. Widening past 8 buys
#:   proportionally less than it did for whole-graph.
#: * The binding resource moves. Whole-graph, K is VRAM-bound because each
#:   child holds the whole model; a block child holds one block, so
#:   ``aot_mint._block_device_fraction`` shrinks the per-entry device ask and
#:   K becomes vCPU-bound again on a fat card.
#:
#: The ceiling therefore stays at 8 deliberately: it is no longer the binding
#: constraint on the shape that matters, and raising it would be sizing for a
#: term that regional already made small.
MAX_ENTRY_WORKERS = 8

#: Host RAM the pool must leave alone: the serving process's own resident set
#: is already counted (we read AVAILABLE, not total), this is the margin on
#: top so that a tenant request arriving mid-mint does not meet an OOM killer.
ENTRY_RSS_RESERVE_BYTES = 4 * 1024**3

#: Per-entry peak RSS assumed before anything has been measured on this pod.
#: Codegen holds the whole generated source plus inductor's IR, and cc1plus on
#: the wrapper TU is the peak — MEASURED at 2.09 GiB on the real sdxl wrapper
#: TU, so host RAM is the LOOSEST of the three bounds, not the binding one.
#: Banked per (family, lane) once measured, exactly like
#: ``mint_budget.record_child_peak`` banks the device peak.
DEFAULT_ENTRY_PEAK_RSS_BYTES = 3 * 1024**3

#: VRAM the pool must leave to the tenant. The mint's whole premise is that
#: the worker keeps serving (pgw#784), so the eager forward's weights AND its
#: activation peak stay untouchable; this is the margin ON TOP of the free
#: figure, because "free right now" is measured between tenant forwards.
DEVICE_RESERVE_BYTES = 2 * 1024**3

#: Per-entry DEVICE footprint assumed before anything has been measured.
#: An AOTI compile is not a pure host job: it benchmarks kernels on the card
#: (and, when ``autotune_at_compile_time`` is explicitly False, runs the whole
#: model on real inputs), so each concurrent entry child holds its own weight
#: copy, an activation set and a CUDA context. This is the bound that actually
#: binds on a 24 GB card, which is why pgw#809's headline K comes from VRAM
#: and not from vCPUs.
DEFAULT_ENTRY_DEVICE_BYTES = 8 * 1024**3

#: Programs staged AHEAD of the running set. The export loop hands the pool
#: every entry at once; staging them all would put ~46 GB of exported programs
#: on disk for an 18-entry sdxl cell. One spare per pool is enough to keep a
#: freed slot from waiting on a multi-GB write.
INFLIGHT_PROGRAM_SLACK = 1

_KILL_GRACE_S = 10.0
_POLL_S = 0.25


@dataclass(frozen=True)
class PoolWidth:
    """The chosen K and every input that chose it — so a mint's telemetry can
    answer "why this width" without re-deriving anything."""

    workers: int
    entries: int
    vcpus: int
    cpu_workers: int
    mem_workers: int
    device_workers: int
    available_bytes: int
    free_device_bytes: int
    per_entry_rss_bytes: int
    per_entry_device_bytes: int
    device_lock: bool
    reason: str

    def facts(self) -> Dict[str, Any]:
        return {
            "entry_workers": int(self.workers),
            "entries": int(self.entries),
            "vcpus": int(self.vcpus),
            "cpu_workers": int(self.cpu_workers),
            "mem_workers": int(self.mem_workers),
            "device_workers": int(self.device_workers),
            "available_bytes": int(self.available_bytes),
            "free_device_bytes": int(self.free_device_bytes),
            "per_entry_rss_bytes": int(self.per_entry_rss_bytes),
            "per_entry_device_bytes": int(self.per_entry_device_bytes),
            "device_lock": bool(self.device_lock),
            "width_reason": self.reason,
        }


def available_memory_bytes() -> int:
    """Host RAM this process may actually take, cgroup-aware.

    ``MemAvailable`` is the host's answer and a container's limit is not; the
    narrower of the two is the only honest one — the same rule
    ``effective_cpu_count`` applies to cores.
    """
    host = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        host = 0
    limits: List[int] = [host] if host > 0 else []
    for path, usage in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
         "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            raw = Path(path).read_text().strip()
            if raw == "max":
                continue
            limit = int(raw)
            used = int(Path(usage).read_text().strip())
        except (OSError, ValueError):
            continue
        # A cgroup v1 "unlimited" is a huge sentinel, not a limit.
        if 0 < limit < (1 << 62):
            limits.append(max(0, limit - used))
    return min(limits) if limits else 0


def free_device_bytes(device: int = -1) -> int:
    """Free VRAM on the mint's card, or 0 when there is no card to read.

    Reads the ALLOCATOR's view of free plus what this process has reserved
    but not allocated, exactly as ``mint_budget`` does — a cached block the
    tenant is not using is free to nobody but this process, and pretending
    otherwise is how a mint OOMs a live request.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        dev = torch.cuda.current_device() if device < 0 else int(device)
        free, _total = torch.cuda.mem_get_info(dev)
        reserved = int(torch.cuda.memory_reserved(dev))
        allocated = int(torch.cuda.memory_allocated(dev))
        return int(free) + max(0, reserved - allocated)
    except Exception:  # noqa: BLE001
        return 0


def entry_workers(
    entries: int,
    *,
    peak_rss_bytes: int = 0,
    device_bytes: int = 0,
    vcpus: int = 0,
    available_bytes: int = -1,
    free_vram_bytes: int = -1,
    limit: int = 0,
    device_lock: Optional[bool] = None,
) -> PoolWidth:
    """How many entries this pod may compile at once.

    Derived, never configured, from THREE bounds:

    * **VRAM — the one that actually binds.** An AOTI compile benchmarks
      kernels on the card, so every concurrent entry child holds its own
      weight copy, activation set and CUDA context. On a 24 GB card with the
      tenant's model resident that is K=2-3 whatever the CPU says.
    * **vCPU**, from :func:`postmortem.effective_cpu_count` (cgroup quota AND
      affinity mask AND host cores, whichever is narrowest) minus
      :data:`SERVING_HEADROOM_CPUS`. ~94 % of an entry compile is ONE core of
      serial host work, so this bound is generous and scales near-perfectly.
    * **Host RAM**, the loosest of the three: the wrapper ``cc1plus`` peaks at
      ~2.1 GiB, so a pod that has VRAM for K has RAM for K several times over.

    ``device_lock=False`` FORCES K=1 on a GPU cell: without torch's
    ``set_gpu_benchmark_lock_context`` hook the pool cannot stop two entries
    benchmarking at once, and a cell whose kernel configs were chosen under
    self-inflicted contention publishes under an unchanged key. Refusing to
    widen is the only safe answer.
    """
    entries = max(0, int(entries))
    locked = aot_device_lock.supported() if device_lock is None \
        else bool(device_lock)
    if entries <= 1:
        return PoolWidth(
            workers=1, entries=entries, vcpus=0, cpu_workers=1, mem_workers=1,
            device_workers=1, available_bytes=0, free_device_bytes=0,
            per_entry_rss_bytes=0, per_entry_device_bytes=0,
            device_lock=locked,
            reason=f"{entries} entr{'y' if entries == 1 else 'ies'}: serial")

    vcpus = int(vcpus) if vcpus > 0 else effective_cpu_count()
    budget = vcpus - SERVING_HEADROOM_CPUS
    cpu_workers = max(1, budget // CPUS_PER_ENTRY_WORKER)

    avail = int(available_bytes) if available_bytes >= 0 \
        else available_memory_bytes()
    per_entry = int(peak_rss_bytes) if peak_rss_bytes > 0 \
        else DEFAULT_ENTRY_PEAK_RSS_BYTES
    if avail <= 0:
        # An unreadable host does not get to license a wide pool.
        mem_workers = 1
    else:
        mem_workers = max(
            1, int(max(0, avail - ENTRY_RSS_RESERVE_BYTES) // per_entry))

    free_vram = int(free_vram_bytes) if free_vram_bytes >= 0 \
        else free_device_bytes()
    per_device = int(device_bytes) if device_bytes > 0 \
        else DEFAULT_ENTRY_DEVICE_BYTES
    if free_vram <= 0:
        # No card, or no reading. A CPU-only cell is not device-bound; a card
        # we cannot measure does not get to license concurrency on it.
        device_workers = MAX_ENTRY_WORKERS
    else:
        device_workers = max(
            1, int(max(0, free_vram - DEVICE_RESERVE_BYTES) // per_device))

    # A caller cap NARROWS. `limit` above MAX_ENTRY_WORKERS is a caller
    # asking for more than the ceiling allows, and the ceiling wins.
    ceiling = min(MAX_ENTRY_WORKERS, int(limit)) if limit > 0 \
        else MAX_ENTRY_WORKERS
    workers = max(
        1, min(cpu_workers, mem_workers, device_workers, ceiling, entries))
    if workers > 1 and free_vram > 0 and not locked:
        workers = 1
        return PoolWidth(
            workers=1, entries=entries, vcpus=vcpus,
            cpu_workers=cpu_workers, mem_workers=mem_workers,
            device_workers=device_workers, available_bytes=avail,
            free_device_bytes=free_vram, per_entry_rss_bytes=per_entry,
            per_entry_device_bytes=per_device, device_lock=False,
            reason=(
                "serial: this torch has no GPU-benchmark lock hook, so a wide "
                "pool would let entries benchmark against each other and bake "
                "contention-chosen kernel configs into a cell whose key would "
                "not move"))
    binding = min(
        (cpu_workers, "cpu"), (mem_workers, "host-memory"),
        (device_workers, "vram"), (ceiling, "ceiling"),
        (entries, "entries"))[1]
    reason = (
        f"K={workers} ({binding}-bound): {vcpus} vCPU -> {cpu_workers}, "
        f"{avail / 1024**3:.1f} GiB RAM -> {mem_workers}, "
        f"{free_vram / 1024**3:.1f} GiB VRAM / "
        f"{per_device / 1024**3:.1f} GiB per entry -> {device_workers}")
    return PoolWidth(
        workers=workers, entries=entries, vcpus=vcpus,
        cpu_workers=cpu_workers, mem_workers=mem_workers,
        device_workers=device_workers, available_bytes=avail,
        free_device_bytes=free_vram, per_entry_rss_bytes=per_entry,
        per_entry_device_bytes=per_device, device_lock=locked, reason=reason)


# ---------------------------------------------------------------------------
# The wire: one job in, one report out (same shape as pgw#784's mint child)
# ---------------------------------------------------------------------------


class EntryJob(msgspec.Struct, frozen=True, kw_only=True):
    """One entry's compile, as a file a human can re-run by hand."""

    entry: str
    program: str
    report: str
    inductor_configs: Dict[str, Any] = {}
    cache_dir: str = ""
    device_lock: str = ""


class EntryReport(msgspec.Struct, frozen=True, kw_only=True):
    entry: str
    status: str = ""
    files: List[str] = []
    detail: str = ""
    elapsed_s: float = 0.0
    peak_rss_bytes: int = 0
    #: Inductor's own phase split (lowering / codegen / host C++ compile+link)
    #: MEASURED IN THE CHILD. pgw#757's instrument-first deliverable is read
    #: from dynamo's in-process counters, which do not move in the parent once
    #: the compile leaves it — so without this the phase table silently goes
    #: dark the moment the pool turns on. Caught by pgw#758's own test.
    phases: Dict[str, float] = {}


COMPILED = "compiled"
REFUSED = "refused"
CRASHED = "crashed"

EXIT_COMPILED = 0
EXIT_REFUSED = 2
EXIT_BAD_JOB = 4


class EntryCompileFailed(RuntimeError):
    """One entry's compile failed. Carries the entry name — a pool of 18 that
    fails anonymously is undebuggable."""

    def __init__(self, entry: str, detail: str) -> None:
        super().__init__(detail)
        self.entry = entry
        self.detail = detail


@dataclass
class _Running:
    entry: str
    proc: subprocess.Popen
    job: EntryJob
    program_path: Path
    started: float
    stderr_path: Path


def child_argv(job_path: Path, *, python: str = "") -> List[str]:
    return [python or sys.executable, "-m", ENTRY_CHILD_MODULE, str(job_path)]


def child_env(
    cache_dir: str, *, base: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """The entry child's environment.

    The parent's env verbatim (the seal must not move between parent and
    child — pgw#784's rule, and here the child produces the very bytes the
    seal describes), plus:

    * ``TORCHINDUCTOR_CACHE_DIR`` pointing every child at ONE worker-local
      directory. This is a LOCATION, not a recipe: nothing in ``env_seal``
      digests it, and the compile's output is byte-identical wherever it
      lands (measured — the same graph hashes to the same cache subdirectory
      from any process). Sharing it is what lets children recover the
      cross-entry kernel dedup a serial loop got from one warm process, and
      it is how the parent reads the loose files the children produced.
    * ``GEN_WORKER_AOT_ENTRY_CHILD=1`` so anything that must not run twice on
      a pod can tell.
    """
    env = dict(os.environ if base is None else base)
    env["GEN_WORKER_AOT_ENTRY_CHILD"] = "1"
    if cache_dir:
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    return env


def _terminate_group(proc: subprocess.Popen, *, grace_s: float = _KILL_GRACE_S) -> None:
    """SIGTERM then SIGKILL the child's process GROUP.

    The GROUP, not the process: an entry child spawns inductor's own compile
    workers and g++ underneath it, and a pool that killed only the direct
    children would leave orphan cc1plus processes burning a serving pod's CPU
    against a cell nobody will adopt. Every child is started with
    ``start_new_session=True`` precisely so this call has a group to aim at.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)


#: ``prctl(2)`` PR_SET_PDEATHSIG.
_PR_SET_PDEATHSIG = 1


def arm_parent_death_signal() -> bool:
    """Ask the kernel to SIGKILL THIS process when its parent dies.

    Called by the entry child on itself, never by the parent through
    ``preexec_fn``. Two reasons, and the second is the expensive one:

    * ``preexec_fn`` runs in the forked child of a process that has threads,
      where only async-signal-safe work is legal — the hazard is real enough
      that CPython documents it.
    * Passing ``preexec_fn`` also FORCES ``fork()`` instead of
      ``posix_spawn()`` for every spawn. The mint child has live gRPC threads,
      and gRPC installs ``pthread_atfork`` handlers; making the pool the only
      thing in the worker that forks a threaded process is a large blast
      radius for a one-line guarantee. Arming from the child costs a
      microscopic race (the parent dying between ``exec`` and this call) and
      buys back the ordinary spawn path.

    Why it exists at all: the pool runs INSIDE pgw#784's mint child, and the
    serving worker reaps that child by process GROUP when a mint is abandoned.
    Entry children hold their OWN session — so the worker's group kill does
    not reach them, and without this an abandoned mint would leave K compiles
    burning a serving pod's CPU with nothing left to notice.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
            return False
    except Exception:  # noqa: BLE001  (non-Linux, or no libc — group kill only)
        return False
    return True


def _read_report(path: Path) -> Optional[EntryReport]:
    try:
        return msgspec.json.decode(path.read_bytes(), type=EntryReport)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError):
        return None


def _peak_rss_bytes(proc: subprocess.Popen) -> int:
    """The child tree's high-water RSS, read from the kernel while it lives."""
    total = 0
    try:
        pids = [proc.pid] + [
            int(p) for p in
            Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
        ]
    except (OSError, ValueError):
        pids = [proc.pid]
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


class EntryCompilePool:
    """Compile N exported programs K-wide, out of process.

    Not a general executor: it exists to hold pgw#809's three invariants —
    entry-named failure, group-wide sibling teardown, and assembly by entry
    NAME rather than completion order.
    """

    def __init__(
        self,
        workdir: Path,
        *,
        width: PoolWidth,
        inductor_configs: Optional[Mapping[str, Any]] = None,
        cache_dir: str = "",
        python: str = "",
    ) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.inductor_configs = dict(inductor_configs or {})
        self.cache_dir = str(cache_dir or (self.workdir / "inductor-cache"))
        # pgw#809: ONE lock file for the whole pool. Every entry child routes
        # its inductor GPU benchmarks through it, so no two entries ever time
        # a kernel on the card at the same moment.
        self.device_lock_path = self.workdir / aot_device_lock.LOCK_NAME
        self.python = python
        self.peak_rss_bytes = 0
        self.peak_concurrency = 0
        self.entry_seconds: Dict[str, float] = {}
        self.entry_phases: Dict[str, Dict[str, float]] = {}

    # -- staging ----------------------------------------------------------

    def _stage(self, entry: str, program: Any, index: int) -> Tuple[EntryJob, Path]:
        import torch

        slot = self.workdir / f"entry-{index:03d}"
        slot.mkdir(parents=True, exist_ok=True)
        program_path = slot / "program.pt2"
        t0 = time.monotonic()
        torch.export.save(program, program_path)
        logger.info(
            "aot-pool: staged %r (%.1f MB) in %.1fs",
            entry, program_path.stat().st_size / 1e6, time.monotonic() - t0)
        job = EntryJob(
            entry=entry,
            program=str(program_path),
            report=str(slot / ENTRY_REPORT_NAME),
            inductor_configs=dict(self.inductor_configs),
            cache_dir=self.cache_dir,
            device_lock=str(self.device_lock_path),
        )
        job_path = slot / "job.json"
        job_path.write_bytes(msgspec.json.encode(job))
        return job, job_path

    def _spawn(self, job: EntryJob, job_path: Path, program_path: Path) -> _Running:
        stderr_path = job_path.parent / "stderr.log"
        handle = stderr_path.open("wb")
        try:
            proc = subprocess.Popen(
                child_argv(job_path, python=self.python),
                stdout=subprocess.DEVNULL,
                stderr=handle,
                env=child_env(self.cache_dir),
                start_new_session=True,   # own group -> group-wide reaping
            )
        finally:
            handle.close()
        logger.info("aot-pool: entry %r -> pid %s", job.entry, proc.pid)
        return _Running(
            entry=job.entry, proc=proc, job=job, program_path=program_path,
            started=time.monotonic(), stderr_path=stderr_path)

    # -- the run ----------------------------------------------------------

    def compile(
        self, entries: Sequence[Tuple[str, Any]],
        *, on_entry: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, List[str]]:
        """``[(entry, ExportedProgram)] -> {entry: [file, ...]}``.

        Raises :class:`EntryCompileFailed` naming the FIRST entry to fail,
        after tearing down every sibling group. Returns a dict ordered by
        entry NAME, never by completion, so the packaged cell cannot depend
        on which child finished first.

        ``on_entry(name, done, total)`` (pgw#824) fires as each entry lands.
        This loop is the longest wire-silent stretch of a mint — an 18-entry
        sdxl cell spends the bulk of its wall clock right here — and until now
        it reported nothing between "compiling" and "packed". Progress
        reporting is best-effort by construction: a raising callback must never
        cost the mint the entries it already has.
        """
        pending: List[Tuple[int, str, Any]] = [
            (i, name, prog) for i, (name, prog) in enumerate(entries)]
        staged: List[Tuple[EntryJob, Path]] = []
        running: List[_Running] = []
        done: Dict[str, List[str]] = {}
        # One program staged AHEAD of the running set, and no more. Staging is
        # a multi-GB write (~16 s at 2.5 GB) and a freed slot that had to wait
        # for one would idle a core through every round; one spare removes
        # that without turning an 18-entry sdxl cell into ~46 GB on disk.
        staged_cap = max(1, self.width.workers + INFLIGHT_PROGRAM_SLACK)
        failure: Optional[EntryCompileFailed] = None
        try:
            while pending or staged or running:
                while (pending and not failure
                       and len(staged) + len(running) < staged_cap):
                    index, name, program = pending.pop(0)
                    staged.append(self._stage(name, program, index))
                while staged and not failure \
                        and len(running) < self.width.workers:
                    job, job_path = staged.pop(0)
                    running.append(
                        self._spawn(job, job_path, Path(job.program)))
                if not running:
                    break
                finished = self._reap(running)
                if finished is None:
                    time.sleep(_POLL_S)
                    continue
                running.remove(finished)
                try:
                    done[finished.entry] = self._collect(finished)
                except EntryCompileFailed as exc:
                    failure = exc
                    break
                if on_entry is not None:
                    try:
                        on_entry(finished.entry, len(done), len(entries))
                    except Exception:  # noqa: BLE001 — telemetry never fails a mint
                        logger.debug(
                            "entry-pool progress callback failed", exc_info=True)
            if failure is not None:
                raise failure
        finally:
            for row in running:
                _terminate_group(row.proc)
            self._sweep()
        return {name: done[name] for name in sorted(done)}

    def _reap(self, running: Sequence[_Running]) -> Optional[_Running]:
        # Observed concurrency, not intended: the ONLY load-independent
        # evidence that the pool actually overlapped rather than looping.
        self.peak_concurrency = max(self.peak_concurrency, len(running))
        for row in running:
            # Sample while it is alive: /proc vanishes at exit, and VmHWM is
            # the only free high-water mark the kernel keeps.
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, _peak_rss_bytes(row.proc))
            if row.proc.poll() is not None:
                return row
        return None

    def _collect(self, row: _Running) -> List[str]:
        elapsed = time.monotonic() - row.started
        self.entry_seconds[row.entry] = round(elapsed, 2)
        code = row.proc.returncode
        report = _read_report(Path(row.job.report))
        # The program is the biggest thing on disk and is dead the moment the
        # child exits; drop it before the next stage runs.
        with_suppress_unlink(row.program_path)
        if code == EXIT_COMPILED and report is not None and report.files:
            if report.peak_rss_bytes:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, int(report.peak_rss_bytes))
            missing = [f for f in report.files if not Path(f).exists()]
            if missing:
                raise EntryCompileFailed(
                    row.entry,
                    f"entry {row.entry!r}: child reported {len(report.files)} "
                    f"compiled file(s) but {len(missing)} do not exist "
                    f"(first: {missing[0]}) — the pool's shared inductor cache "
                    f"dir {self.cache_dir!r} is not visible to this process")
            self.entry_phases[row.entry] = dict(report.phases or {})
            logger.info(
                "aot-pool: entry %r compiled in %.1fs (%d file(s))",
                row.entry, elapsed, len(report.files))
            return list(report.files)
        detail = report.detail if report is not None else ""
        if not detail:
            detail = _stderr_tail(row.stderr_path)
        raise EntryCompileFailed(
            row.entry,
            f"entry {row.entry!r}: compile child exited {code} after "
            f"{elapsed:.0f}s ({_exit_note(code)}): {detail or 'no detail'}")

    def _sweep(self) -> None:
        """Every staged program, gone. The loose compiled files stay: they
        live in the inductor cache dir and are what ``package_aoti`` reads."""
        for slot in sorted(self.workdir.glob("entry-*")):
            with_suppress_unlink(slot / "program.pt2")


def with_suppress_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _stderr_tail(path: Path, limit: int = 2048) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return raw[-limit:].decode("utf-8", "replace").strip()


def _exit_note(code: Optional[int]) -> str:
    if code is None:
        return "still running"
    if code == EXIT_REFUSED:
        return "named refusal"
    if code == EXIT_BAD_JOB:
        return "malformed job — a wiring defect"
    if code < 0:
        name = f"SIG{-code}"
        try:
            name = signal.Signals(-code).name
        except ValueError:
            pass
        if -code in (signal.SIGKILL, signal.SIGSEGV):
            return (
                f"killed by {name} — on a mint this is the OOM killer far "
                f"more often than a compiler bug; the pool's width is "
                f"memory-bounded for exactly this reason")
        return f"killed by {name}"
    return "crashed"


def free_disk_bytes(path: Path) -> int:
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return 0
    return int(usage.free)


__all__ = [
    "COMPILED",
    "CPUS_PER_ENTRY_WORKER",
    "CRASHED",
    "DEFAULT_ENTRY_PEAK_RSS_BYTES",
    "ENTRY_CHILD_MODULE",
    "ENTRY_REPORT_NAME",
    "ENTRY_RSS_RESERVE_BYTES",
    "EXIT_BAD_JOB",
    "EXIT_COMPILED",
    "EXIT_REFUSED",
    "EntryCompileFailed",
    "EntryCompilePool",
    "arm_parent_death_signal",
    "EntryJob",
    "EntryReport",
    "DEFAULT_ENTRY_DEVICE_BYTES",
    "DEVICE_RESERVE_BYTES",
    "MAX_ENTRY_WORKERS",
    "REFUSED",
    "SERVING_HEADROOM_CPUS",
    "PoolWidth",
    "available_memory_bytes",
    "child_argv",
    "child_env",
    "entry_workers",
    "free_device_bytes",
    "free_disk_bytes",
]
