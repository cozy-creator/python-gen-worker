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

from . import aot_compile_spans, aot_device_lock, env_seal
from .postmortem import cpu_quota_cores

logger = logging.getLogger(__name__)

ENTRY_CHILD_MODULE = "gen_worker.aot_compile_child"

#: Report file each entry child writes before exiting.
ENTRY_REPORT_NAME = "report.json"

#: pgw#840: the directory that must be on the child's path for
#: ``-m gen_worker.aot_compile_child`` to mean THIS gen_worker.
PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)

#: The three modules that define the parent/child contract: the child's own
#: entrypoint, this module (the job/report structs) and the span partition.
_CONTRACT_MODULES = (
    "aot_compile_child.py", "aot_compile_pool.py", "aot_compile_spans.py")


def _code_digest() -> str:
    """A digest of the parent/child contract source, taken AT IMPORT.

    pgw#840: the pool spawns ``sys.executable -m gen_worker.aot_compile_child``
    and lets the child's import system decide which ``gen_worker`` that is. It
    can legitimately be a different one — a ``PYTHONPATH`` entry, a ``gen_worker``
    in the cwd, a second checkout, a stale wheel in the interpreter's
    site-packages, or the same tree edited between the parent's import and the
    child's spawn. The child then compiles the very files the cell publishes
    with code the parent never ran, and the parent believes the report.

    Taken at import (not at read time) on purpose: a tree edited mid-run must
    compare the code each process is actually EXECUTING, not what its files say
    afterwards.
    """
    import hashlib

    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _CONTRACT_MODULES:
        try:
            digest.update(hashlib.sha256((here / name).read_bytes()).digest())
        except OSError:  # zipimport / frozen: no source to compare
            return ""
    return digest.hexdigest()[:16]


#: Computed once per process, so parent and child each carry their OWN.
CODE_DIGEST = _code_digest()

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

#: Host RAM a FORGE pod leaves alone (th#1359). Not a tenant reserve — there
#: is no tenant — but the OS, the page cache and the mint child's own
#: supervisor are real on any pod. A quarter of the serving reserve.
FORGE_RSS_RESERVE_BYTES = 1 * 1024**3

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


#: pgw#842: how many times the device bound samples free VRAM before it
#: believes a number, and how far apart. The pool shares the card with a
#: SERVING process by construction (pgw#784), so a single ``mem_get_info``
#: can land inside a tenant forward and read that forward's activation set as
#: "gone". :data:`DEVICE_RESERVE_BYTES` already reserves the tenant's peak;
#: subtracting an in-flight peak on top of it charges the same bytes twice —
#: measured as a 5 -> 3 width swing on identical work. The MAX over a short
#: window is the steady figure the reserve was written against. Three samples
#: over 0.1 s costs nothing against a mint that runs for minutes.
DEVICE_FREE_SAMPLES = 3
DEVICE_FREE_SAMPLE_GAP_S = 0.05


@dataclass(frozen=True)
class CpuFacts:
    """What the pod's cores actually are, and which reading said so.

    pgw#842: a provider ADVERTISES vCPUs (RunPod's ``host_vcpus``) and the
    kernel ENFORCES a quota, and the two are routinely different numbers. The
    pool must size on the enforced one — and must SAY which one it read, or a
    K that came out narrow than the advertisement is indistinguishable from a
    bug in the formula.
    """

    vcpus: int
    basis: str
    os_cpu_count: int
    affinity_cpus: int
    quota_cores: float

    def facts(self) -> Dict[str, Any]:
        return {
            "vcpus": int(self.vcpus),
            "cpu_basis": self.basis,
            "os_cpu_count": int(self.os_cpu_count),
            "affinity_cpus": int(self.affinity_cpus),
            "quota_cores": round(float(self.quota_cores), 2),
        }


@dataclass(frozen=True)
class MemoryFacts:
    """Host RAM the pool may take, and the reading that bounded it."""

    available_bytes: int
    basis: str
    host_available_bytes: int
    cgroup_available_bytes: int
    cgroup_reclaimable_bytes: int

    def facts(self) -> Dict[str, Any]:
        return {
            "available_bytes": int(self.available_bytes),
            "mem_basis": self.basis,
            "host_available_bytes": int(self.host_available_bytes),
            "cgroup_available_bytes": int(self.cgroup_available_bytes),
            "cgroup_reclaimable_bytes": int(self.cgroup_reclaimable_bytes),
        }


@dataclass(frozen=True)
class DeviceFacts:
    """Free VRAM the pool may divide, and every sample behind it."""

    free_bytes: int
    basis: str
    samples: Tuple[int, ...]

    def facts(self) -> Dict[str, Any]:
        return {
            "free_device_bytes": int(self.free_bytes),
            "device_basis": self.basis,
            "free_device_samples": [int(x) for x in self.samples],
        }


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
    #: pgw#842: the constraint that ACTUALLY held K down, by name, plus the
    #: readings each bound was taken from. A width narrower than the pod could
    #: carry is a performance defect, and it has to be legible from one record
    #: rather than inferred by diffing two pods that no longer exist.
    binding: str = ""
    ceiling: int = MAX_ENTRY_WORKERS
    cpu: Optional[CpuFacts] = None
    memory: Optional[MemoryFacts] = None
    device: Optional[DeviceFacts] = None
    #: "measured" when the caller handed a probed per-entry ask, "default"
    #: when the width fell back to a constant. A default is a guess, and a K
    #: chosen off a guess must not read like a K chosen off a measurement.
    per_entry_device_basis: str = "default"
    per_entry_rss_basis: str = "default"
    #: th#1359: whether the tenant reserves were applied. A K of 1 on a
    #: serving pod and a K of 1 on a forge pod are different defects, and the
    #: row has to say which one it is.
    forge: bool = False

    @property
    def underwidth(self) -> int:
        """Workers the pod would have run had the binding constraint not
        bound — 0 when K is already the most this cell can use."""
        return max(0, min(self.entries, self.ceiling) - self.workers)

    def facts(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
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
            "binding": self.binding,
            "ceiling": int(self.ceiling),
            "underwidth": int(self.underwidth),
            "per_entry_device_basis": self.per_entry_device_basis,
            "per_entry_rss_basis": self.per_entry_rss_basis,
            "forge": bool(self.forge),
            "width_reason": self.reason,
        }
        for block in (self.cpu, self.memory, self.device):
            if block is not None:
                out.update(block.facts())
        return out


def _read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # A cgroup v1 "unlimited" is a huge sentinel, not a limit.
    return value if 0 <= value < (1 << 62) else None


def _cgroup_reclaimable_bytes(stat: Path) -> int:
    """File pages this cgroup is charged for that the kernel will hand back
    under pressure — page cache, and reclaimable slab.

    pgw#842: ``memory.current`` counts page cache. A mint reads GBs (weights,
    the toolchain the seal hashes, every staged program) and every one of
    those pages inflates ``current`` until something needs the memory. Sizing
    a pool on ``max - current`` therefore shrinks the pool in proportion to
    how much I/O the pod has already done — a bound that moves with history
    instead of with the box. Subtracting what is reclaimable is the same
    working-set definition every container runtime uses.
    """
    total = 0
    try:
        lines = stat.read_text().splitlines()
    except OSError:
        return 0
    wanted = {
        "inactive_file", "slab_reclaimable",          # cgroup v2
        "total_inactive_file", "total_slab_reclaimable",  # cgroup v1
    }
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0] in wanted:
            try:
                total += int(parts[1])
            except ValueError:
                continue
    return total


def memory_facts(
    *,
    meminfo: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> MemoryFacts:
    """Host RAM this process may actually take, cgroup-aware.

    ``MemAvailable`` is the host's answer and a container's limit is not; the
    narrower of the two is the only honest one — the same rule
    ``effective_cpu_count`` applies to cores. Both readings are kept, so the
    telemetry can say WHICH one bounded the pool.

    The paths are arguments so a test can drive this function against a real
    (synthetic) cgroup tree instead of re-implementing its arithmetic.
    """
    host = 0
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        host = 0
    cgroup = -1
    reclaimable = 0
    for limit_name, usage_name, stat_name in (
        ("memory.max", "memory.current", "memory.stat"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes",
         "memory/memory.stat"),
    ):
        limit = _read_int(cgroup_root / limit_name)
        used = _read_int(cgroup_root / usage_name)
        if limit is None or used is None or limit <= 0:
            continue
        reclaimable = _cgroup_reclaimable_bytes(cgroup_root / stat_name)
        working_set = max(0, used - reclaimable)
        cgroup = max(0, limit - working_set)
        break
    if host > 0 and cgroup >= 0:
        basis = "cgroup" if cgroup < host else "meminfo"
        return MemoryFacts(min(host, cgroup), basis, host, cgroup, reclaimable)
    if cgroup >= 0:
        return MemoryFacts(cgroup, "cgroup", host, cgroup, reclaimable)
    if host > 0:
        return MemoryFacts(host, "meminfo", host, -1, 0)
    return MemoryFacts(0, "unreadable", 0, -1, 0)


def available_memory_bytes() -> int:
    """The single number :func:`memory_facts` bounds the pool with."""
    return memory_facts().available_bytes


def cpu_facts() -> CpuFacts:
    """The pod's honest core count, and which of the three readings it is."""
    os_count = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os_count
    quota = cpu_quota_cores()
    quota_cores = float(quota) if quota is not None else -1.0
    candidates = [(os_count, "cpu_count"), (affinity, "affinity")]
    if quota is not None:
        candidates.append((max(1, int(quota + 0.5)), "quota"))
    vcpus, basis = min(candidates)
    return CpuFacts(max(1, vcpus), basis, os_count, affinity, quota_cores)


def _probe_free_device_bytes(device: int = -1) -> int:
    """One reading of free VRAM on the mint's card, 0 when there is no card.

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


def device_facts(
    device: int = -1,
    *,
    samples: int = DEVICE_FREE_SAMPLES,
    gap_s: float = DEVICE_FREE_SAMPLE_GAP_S,
    probe: Optional[Callable[[int], int]] = None,
) -> DeviceFacts:
    """Free VRAM the pool may divide — the STEADY figure, not an instant.

    pgw#842: the mint shares the card with the serving process, so one
    ``mem_get_info`` taken while a tenant forward holds its activation set
    reads several GiB below the steady free figure — and the pool then
    subtracts :data:`DEVICE_RESERVE_BYTES` for that same tenant peak on top.
    The MAX over a short window is the reading the reserve was written
    against; every sample is kept so the choice is auditable.
    """
    read = probe if probe is not None else _probe_free_device_bytes
    taken: List[int] = []
    for i in range(max(1, int(samples))):
        if i and gap_s > 0:
            time.sleep(gap_s)
        value = int(read(device))
        taken.append(value)
        if value <= 0:
            # No card, or no reading — sampling an absence is not evidence.
            return DeviceFacts(0, "absent", tuple(taken))
    return DeviceFacts(max(taken), "sampled", tuple(taken))


def free_device_bytes(device: int = -1) -> int:
    """The single free-VRAM number :func:`device_facts` sizes the pool with."""
    return device_facts(device).free_bytes


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
    forge: Optional[bool] = None,
) -> PoolWidth:
    """How many entries this pod may compile at once.

    Derived, never configured, from THREE bounds:

    * **VRAM — the one that actually binds.** An AOTI compile benchmarks
      kernels on the card, so every concurrent entry child holds its own
      weight copy, activation set and CUDA context. On a 24 GB card with the
      tenant's model resident that is K=2-3 whatever the CPU says. Read via
      :func:`device_facts` — the STEADY free figure, never one sample.
    * **vCPU**, from :func:`cpu_facts` (cgroup quota AND affinity mask AND
      host cores, whichever is narrowest) minus
      :data:`SERVING_HEADROOM_CPUS`. ~94 % of an entry compile is ONE core of
      serial host work, so this bound is generous and scales near-perfectly.
    * **Host RAM**, the loosest of the three: the wrapper ``cc1plus`` peaks at
      ~2.1 GiB, so a pod that has VRAM for K has RAM for K several times over.
      Read via :func:`memory_facts`, whose cgroup half counts the WORKING SET
      rather than everything the pod has ever paged in.

    ``device_lock=False`` FORCES K=1 on a GPU cell: without torch's
    ``set_gpu_benchmark_lock_context`` hook the pool cannot stop two entries
    benchmarking at once, and a cell whose kernel configs were chosen under
    self-inflicted contention publishes under an unchanged key. Refusing to
    widen is the only safe answer.

    pgw#842: every bound records the READING behind it (:class:`CpuFacts`,
    :class:`MemoryFacts`, :class:`DeviceFacts`) and the returned width names
    the constraint that actually bound. K is the mint's only multiplicative
    lever — two mints of one cell differed 5-vs-3 with nothing recorded to
    say why — so an unexplained K is a defect in itself.
    """
    entries = max(0, int(entries))
    if forge is None:
        from . import worker_mode

        forge = worker_mode.is_forge()
    # th#1359 / pgw#848: THREE of this policy's terms are tenant reserves, and
    # on a forge pod there is no tenant. `SERVING_HEADROOM_CPUS` keeps cores
    # for an eager forward and a heartbeat; `DEVICE_RESERVE_BYTES` keeps VRAM
    # for the tenant's peak; `ENTRY_RSS_RESERVE_BYTES` keeps host RAM so a
    # request arriving mid-mint does not meet the OOM killer. A mint-only pod
    # receives no tenant dispatch, so all three protect nobody — and on
    # pgw#846's attempts fourteen and fifteen the VRAM reserve alone held the
    # pool at K=1 on a host that could have run it 127 CPU-side.
    #
    # A SMALL host-RAM reserve survives: the OS, the page cache and the mint
    # child's own supervisor are real on a forge pod too. It is the TENANT's
    # share that goes, not prudence.
    cpu_headroom = 0 if forge else SERVING_HEADROOM_CPUS
    device_reserve = 0 if forge else DEVICE_RESERVE_BYTES
    rss_reserve = FORGE_RSS_RESERVE_BYTES if forge else ENTRY_RSS_RESERVE_BYTES
    locked = aot_device_lock.supported() if device_lock is None \
        else bool(device_lock)
    if entries <= 1:
        return PoolWidth(
            workers=1, entries=entries, vcpus=0, cpu_workers=1, mem_workers=1,
            device_workers=1, available_bytes=0, free_device_bytes=0,
            per_entry_rss_bytes=0, per_entry_device_bytes=0,
            device_lock=locked, binding="entries", ceiling=1,
            forge=bool(forge or False),
            reason=f"{entries} entr{'y' if entries == 1 else 'ies'}: serial")

    if vcpus > 0:
        cpu = CpuFacts(int(vcpus), "caller", int(vcpus), int(vcpus), -1.0)
    else:
        cpu = cpu_facts()
    vcpus = cpu.vcpus
    budget = vcpus - cpu_headroom
    cpu_workers = max(1, budget // CPUS_PER_ENTRY_WORKER)

    if available_bytes >= 0:
        memory = MemoryFacts(
            int(available_bytes), "caller", int(available_bytes), -1, 0)
    else:
        memory = memory_facts()
    avail = memory.available_bytes
    per_entry = int(peak_rss_bytes) if peak_rss_bytes > 0 \
        else DEFAULT_ENTRY_PEAK_RSS_BYTES
    if avail <= 0:
        # An unreadable host does not get to license a wide pool.
        mem_workers = 1
    else:
        mem_workers = max(
            1, int(max(0, avail - rss_reserve) // per_entry))

    if free_vram_bytes >= 0:
        device = DeviceFacts(
            int(free_vram_bytes),
            "caller" if free_vram_bytes > 0 else "absent",
            (int(free_vram_bytes),))
    else:
        device = device_facts()
    free_vram = device.free_bytes
    per_device = int(device_bytes) if device_bytes > 0 \
        else DEFAULT_ENTRY_DEVICE_BYTES
    if free_vram <= 0:
        # No card, or no reading. A CPU-only cell is not device-bound; a card
        # we cannot measure does not get to license concurrency on it.
        device_workers = MAX_ENTRY_WORKERS
    else:
        device_workers = max(
            1, int(max(0, free_vram - device_reserve) // per_device))

    # A caller cap NARROWS. `limit` above MAX_ENTRY_WORKERS is a caller
    # asking for more than the ceiling allows, and the ceiling wins.
    ceiling = min(MAX_ENTRY_WORKERS, int(limit)) if limit > 0 \
        else MAX_ENTRY_WORKERS
    def _width(
        workers: int, *, binding: str, reason: str, lock: bool,
    ) -> PoolWidth:
        return PoolWidth(
            workers=workers, entries=entries, vcpus=vcpus,
            cpu_workers=cpu_workers, mem_workers=mem_workers,
            device_workers=device_workers, available_bytes=avail,
            free_device_bytes=free_vram, per_entry_rss_bytes=per_entry,
            per_entry_device_bytes=per_device, device_lock=lock,
            reason=reason, binding=binding, ceiling=ceiling, cpu=cpu,
            memory=memory, device=device,
            per_entry_device_basis=(
                "measured" if device_bytes > 0 else "default"),
            per_entry_rss_basis=(
                "measured" if peak_rss_bytes > 0 else "default"),
            forge=bool(forge))

    workers = max(
        1, min(cpu_workers, mem_workers, device_workers, ceiling, entries))
    if workers > 1 and free_vram > 0 and not locked:
        return _width(
            1, binding="device-lock", lock=False,
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
        f"K={workers} ({binding}-bound{', forge' if forge else ''}): "
        f"{vcpus} vCPU ({cpu.basis}) -> "
        f"{cpu_workers}, {avail / 1024**3:.1f} GiB RAM ({memory.basis}) -> "
        f"{mem_workers}, {free_vram / 1024**3:.1f} GiB VRAM ({device.basis}) "
        f"/ {per_device / 1024**3:.1f} GiB per entry "
        f"({'measured' if device_bytes > 0 else 'default'}) -> "
        f"{device_workers}")
    return _width(workers, binding=binding, reason=reason, lock=locked)


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
    #: pgw#830: the child's COMPLETE wall partition — `phases` above measures
    #: only the inside of `aot_compile`, while the recorded `compile_s` is the
    #: parent's Popen-to-reap wall. Everything between those two definitions
    #: (interpreter boot, `import torch`, the seal, the device-lock install,
    #: `torch.export.load` of the staged program) was the dark 44 %.
    spans: Dict[str, float] = {}
    #: Named, and deliberately NOT summed with `spans`: these nest inside
    #: partition members (triton keys inside codegen/host compile; a device
    #: benchmark-lock wait inside `aot_compile`). Adding them to the total was
    #: the second attribution bug.
    overlays: Dict[str, float] = {}
    spans_v: int = 0
    #: Wall-clock stamps that close the spans crossing the process boundary.
    module_import_epoch: float = 0.0
    run_start_epoch: float = 0.0
    report_epoch: float = 0.0
    #: Every `compilation_time_metrics` key that moved. The evidence for
    #: naming whatever `compile_other_s` turns out to contain — a residual
    #: nobody can look inside is only half an attribution.
    metrics_raw: Dict[str, float] = {}
    #: pgw#840: WHICH gen_worker compiled this entry. ``code_digest`` is the
    #: contract-source digest the child computed at ITS import; ``code_dir`` is
    #: where it imported from, which is the actionable half of the message. An
    #: empty digest means a child too old to answer — which is the case that
    #: went dark: it also predates the span table, so the parent absorbed its
    #: whole compile into ``reap_lag_s`` and reported a partition that did not
    #: close (the pgw#830 invariant, red on a tree nobody had changed).
    code_digest: str = ""
    code_dir: str = ""


COMPILED = "compiled"
REFUSED = "refused"
CRASHED = "crashed"

EXIT_COMPILED = 0
EXIT_REFUSED = 2
EXIT_BAD_JOB = 4


class EntryCompileFailed(RuntimeError):
    """One entry's compile failed. Carries the entry name — a pool of 18 that
    fails anonymously is undebuggable — and its CLASSIFICATION.

    pgw#848: ``resource`` is the difference between a mint that is retried at
    a narrower K and one that is never retried at all. Everything this class
    carried used to converge on ``MintRefused`` -> ``EXIT_REFUSED``, which
    ``mint_process`` documents as "typed, deterministic — terminal", so the
    ONE failure class a narrower pool would have fixed was the one class
    routed down the never-retry path. ``basis`` names how the verdict was
    reached, because a verdict from the kernel's own OOM counter and a
    verdict inferred from a signal are not the same evidence.
    """

    def __init__(
        self, entry: str, detail: str, *,
        resource: bool = False, basis: str = "", peak_rss_bytes: int = 0,
    ) -> None:
        super().__init__(detail)
        self.entry = entry
        self.detail = detail
        self.resource = bool(resource)
        self.basis = str(basis)
        self.peak_rss_bytes = int(peak_rss_bytes)


@dataclass
class _Running:
    entry: str
    proc: subprocess.Popen
    job: EntryJob
    program_path: Path
    started: float
    stderr_path: Path
    #: pgw#848: THIS entry's own high-water, sampled while it lives. The
    #: pool-wide max cannot answer "how big was the one that died", and a
    #: child killed by the OOM killer writes no report — so the parent's
    #: live sample is the only measurement that survives it, and it is
    #: exactly the number the next attempt must size K against.
    peak_rss_bytes: int = 0
    #: Wall clock at the moment ``Popen`` returned — the other end of the
    #: child's boot span, which no monotonic clock can close across processes.
    spawn_epoch: float = 0.0


@dataclass
class PoolLedger:
    """pgw#830: where the POOL's seconds went, as against where the entries'
    seconds went. The two are different questions with opposite fixes.

    ``compile_s`` (the sum of entry walls) is SERIAL work: shrinking it means
    compiling less or compiling faster. ``pool_idle_s`` is SCHEDULING loss:
    workers with nothing to run. Attempt nine's 75 % pool efficiency at K=5 is
    the second number, and collapsing the entry count (pgw#829) moves it
    without touching a single compile — so a table that adds the two into one
    "dark 44 %" figure would send that work at the wrong target.

    The idle split is exact, not sampled: the loop charges every free-slot
    second to the state the parent was actually in.
    """

    workers: int = 1
    wall_s: float = 0.0
    #: Sum of the per-entry Popen-to-reap walls (== the mint's ``compile_s``).
    busy_s: float = 0.0
    #: ``workers * wall_s`` — the seconds the pool COULD have compiled for.
    capacity_s: float = 0.0
    #: Free-slot seconds while the parent was inside ``torch.export.save``.
    #: The pool refills a freed slot only after staging the next program, so
    #: this is export-vs-compile SERIALIZATION, measured.
    idle_staging_s: float = 0.0
    #: Free-slot seconds with nothing left to start: the straggler tail.
    idle_drain_s: float = 0.0
    #: Free-slot seconds inside ``Popen`` itself.
    idle_spawn_s: float = 0.0
    #: Free-slot seconds that were neither — the residual of the idle split.
    idle_other_s: float = 0.0
    #: Parent-serial staging cost, summed. Not idle time (it overlaps running
    #: children); the idle FRACTION of it is ``idle_staging_s``.
    stage_total_s: float = 0.0
    spawn_total_s: float = 0.0
    #: pgw#832: parent-serial cost of seeding the seal library memo, paid
    #: BEFORE the pool wall starts (so never part of the capacity identity).
    #: Near-zero in a sealed parent; one full hashing pass in a cold one —
    #: which is the pass every CHILD used to pay.
    seal_seed_s: float = 0.0
    entries: int = 0

    @property
    def idle_s(self) -> float:
        return round(
            self.idle_staging_s + self.idle_drain_s
            + self.idle_spawn_s + self.idle_other_s, 3)

    @property
    def efficiency(self) -> float:
        return round(self.busy_s / self.capacity_s, 4) if self.capacity_s else 0.0

    def facts(self) -> Dict[str, Any]:
        return {
            "pool_wall_s": round(self.wall_s, 3),
            "pool_busy_s": round(self.busy_s, 3),
            "pool_capacity_s": round(self.capacity_s, 3),
            "pool_idle_s": self.idle_s,
            "pool_efficiency": self.efficiency,
            "idle_staging_s": round(self.idle_staging_s, 3),
            "idle_drain_s": round(self.idle_drain_s, 3),
            "idle_spawn_s": round(self.idle_spawn_s, 3),
            "idle_other_s": round(self.idle_other_s, 3),
            "stage_total_s": round(self.stage_total_s, 3),
            "spawn_total_s": round(self.spawn_total_s, 3),
            "seal_seed_s": round(self.seal_seed_s, 3),
            "pool_entries": int(self.entries),
            "pool_workers": int(self.workers),
        }


def child_argv(job_path: Path, *, python: str = "") -> List[str]:
    return [python or sys.executable, "-m", ENTRY_CHILD_MODULE, str(job_path)]


def child_env(
    cache_dir: str, *, base: Optional[Mapping[str, str]] = None,
    seal_memo: str = "",
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
    * ``GEN_WORKER_SEAL_LIB_MEMO`` (pgw#832), when the pool seeded one: the
      parent's toolchain digests, so the child's ``env_seal.establish()``
      stats instead of re-hashing 4 GB. A LOCATION of digests, never a
      recipe: the child verifies every file's (path, mtime_ns, size) itself
      and rehashes on any mismatch, so the seal value cannot move.
    * ``PYTHONPATH`` with the PARENT's own package root in front, and
      ``PYTHONSAFEPATH`` (pgw#840). ``-m gen_worker.aot_compile_child`` used to
      mean "whatever gen_worker this interpreter resolves": the cwd first, then
      any inherited ``PYTHONPATH``, then site-packages. On a box with more than
      one checkout that is a coin flip, and the child that wins compiles the
      files the cell publishes. Pinning the root makes the child the parent's
      OWN code by construction; ``PYTHONSAFEPATH`` removes the cwd, which would
      otherwise still outrank it. The digest check in ``_collect`` is the
      backstop that proves it rather than assuming it.
    """
    env = dict(os.environ if base is None else base)
    env["GEN_WORKER_AOT_ENTRY_CHILD"] = "1"
    if cache_dir:
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    if seal_memo:
        env[env_seal.SEAL_LIB_MEMO_ENV] = str(seal_memo)
    env["PYTHONSAFEPATH"] = "1"
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                if p and p != PACKAGE_ROOT]
    env["PYTHONPATH"] = os.pathsep.join([PACKAGE_ROOT] + existing)
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


def _vmhwm_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _descendants(root: int) -> List[int]:
    """``root`` and every process under it, TRANSITIVELY.

    pgw#848: the previous reading walked ``/proc/<pid>/task/<pid>/children``
    once — direct children of the main thread, one level, and only that
    thread's. MEASURED on a real AOTI compile: the entry child's direct
    children are ``g++`` (a driver that allocates nothing) and inductor's
    ``async_compile`` subprocess workers; **``cc1plus`` — the 2.04 GiB — is at
    depth 2**, and ``as``/``collect2``/``ld`` sit beside it. So the one number
    the pool's memory bound exists to measure was, by construction, the one
    number this function could not see.
    """
    out: List[int] = []
    stack = [int(root)]
    while stack:
        pid = stack.pop()
        out.append(pid)
        try:
            for task in Path(f"/proc/{pid}/task").iterdir():
                stack.extend(
                    int(p) for p in (task / "children").read_text().split())
        except (OSError, ValueError):
            continue
    return out


def cgroup_oom_kills(cgroup_root: Path = Path("/sys/fs/cgroup")) -> int:
    """How many processes this cgroup's memory limit has killed, ever.

    pgw#848: the kernel's own counter, and the only NON-inferential evidence
    that a dead entry child died of memory. A SIGKILL with no report is the
    OOM killer far more often than anything else — the pool has said so in
    ``_exit_note`` since pgw#809 — but "far more often" is not a
    classification, and the retry policy branches on it. ``-1`` = unreadable,
    which is honest and is reported as such rather than folded into 0.
    """
    for name in ("memory.events", "memory/memory.oom_control"):
        path = cgroup_root / name
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in ("oom_kill", "oom_kill_disable"):
                if parts[0] == "oom_kill":
                    try:
                        return int(parts[1])
                    except ValueError:
                        return -1
    return -1


def _peak_rss_bytes(proc: subprocess.Popen) -> int:
    """The child tree's high-water RSS, read from the kernel while it lives.

    Summed across the tree because the members are concurrent: the
    interpreter holds the loaded program while its compiler holds the TU. A
    concurrent pool slot must reserve the pair, not the larger of the two.
    """
    return sum(_vmhwm_bytes(pid) for pid in _descendants(proc.pid))


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
        # pgw#832: the parent's toolchain digests, seeded once per pool so N
        # short-lived children stop re-paying a multi-GB SHA-256 pass each.
        # Emptied if seeding fails — children then rehash in full (safe path).
        self.seal_memo = str(self.workdir / "seal-lib-memo.json")
        self.seal_seed_s = 0.0
        self.python = python
        self.peak_rss_bytes = 0
        self.peak_concurrency = 0
        # pgw#848: the kernel's OOM-kill counter as it stood before this pool
        # ran. A DELTA over the pool's own wall is evidence; the absolute
        # number is a pod's whole history and means nothing here.
        self.oom_kills_at_start = cgroup_oom_kills()
        #: Set when an entry died of memory, so the mint's aborted phase
        #: table carries the actionable half rather than a bare "refused".
        self.oom_entry = ""
        self.oom_basis = ""
        self.entry_seconds: Dict[str, float] = {}
        self.entry_phases: Dict[str, Dict[str, float]] = {}
        # pgw#830: parent-side per-entry spans (staging + spawn) and the
        # pool-level idle split. Kept separate from `entry_phases` because
        # they are NOT inside `compile_s`: staging happens in the parent while
        # other children run, so summing it into the compile total would
        # invent seconds nobody spent compiling.
        self.entry_stage_seconds: Dict[str, float] = {}
        self.entry_spawn_seconds: Dict[str, float] = {}
        self.entry_overlays: Dict[str, Dict[str, float]] = {}
        self.entry_metrics_raw: Dict[str, Dict[str, float]] = {}
        self.ledger = PoolLedger(workers=int(width.workers))

    # -- staging ----------------------------------------------------------

    def _stage(self, entry: str, program: Any, index: int) -> Tuple[EntryJob, Path]:
        import torch

        slot = self.workdir / f"entry-{index:03d}"
        slot.mkdir(parents=True, exist_ok=True)
        program_path = slot / "program.pt2"
        t0 = time.monotonic()
        torch.export.save(program, program_path)
        self.entry_stage_seconds[entry] = round(time.monotonic() - t0, 3)
        self.ledger.stage_total_s = round(
            self.ledger.stage_total_s + self.entry_stage_seconds[entry], 3)
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
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                child_argv(job_path, python=self.python),
                stdout=subprocess.DEVNULL,
                stderr=handle,
                env=child_env(self.cache_dir, seal_memo=self.seal_memo),
                start_new_session=True,   # own group -> group-wide reaping
            )
        finally:
            handle.close()
        started, spawn_epoch = time.monotonic(), time.time()
        self.entry_spawn_seconds[job.entry] = round(started - t0, 3)
        self.ledger.spawn_total_s = round(
            self.ledger.spawn_total_s + (started - t0), 3)
        logger.info("aot-pool: entry %r -> pid %s", job.entry, proc.pid)
        return _Running(
            entry=job.entry, proc=proc, job=job, program_path=program_path,
            started=started, stderr_path=stderr_path, spawn_epoch=spawn_epoch)

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
        # pgw#830: exact idle accounting. Every wall second is multiplied by
        # the number of FREE worker slots at that moment and charged to
        # whatever the parent was doing — so `pool_idle_s` is not a residual
        # anybody has to trust, it is a sum with named causes.
        self.ledger.entries = len(entries)
        # pgw#832: seed BEFORE the pool wall starts, so the cost is its own
        # named line (`seal_seed_s`) and never inside the capacity identity.
        self._seed_seal_memo()
        pool_t0 = mark = time.monotonic()

        def charge(bucket: str, free: int) -> None:
            nonlocal mark
            now = time.monotonic()
            if free > 0:
                setattr(self.ledger, bucket,
                        getattr(self.ledger, bucket) + (now - mark) * free)
            mark = now

        try:
            while pending or staged or running:
                while (pending and not failure
                       and len(staged) + len(running) < staged_cap):
                    index, name, program = pending.pop(0)
                    free = self.width.workers - len(running)
                    staged.append(self._stage(name, program, index))
                    # The freed slot waits for the NEXT program to be written
                    # before it can be refilled: export-vs-compile
                    # serialization, charged where it happens.
                    charge("idle_staging_s", free)
                while staged and not failure \
                        and len(running) < self.width.workers:
                    job, job_path = staged.pop(0)
                    free = self.width.workers - len(running)
                    running.append(
                        self._spawn(job, job_path, Path(job.program)))
                    charge("idle_spawn_s", free)
                if not running:
                    break
                free = self.width.workers - len(running)
                # Nothing left to start: the free slots are the straggler
                # tail, which is a SCHEDULING loss and not a compile cost.
                # pgw#829's entry collapse moves this number; nothing about
                # the compiler does.
                bucket = "idle_drain_s" if not (pending or staged) \
                    else "idle_other_s"
                finished = self._reap(running)
                if finished is None:
                    time.sleep(_POLL_S)
                    charge(bucket, free)
                    continue
                charge(bucket, free)
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
                # Collection (report read, program unlink) and pgw#824's
                # progress callback both run with the slot ALREADY FREE, so
                # they are charged as idle rather than left outside the split
                # — a callback that blocked would otherwise vanish from a
                # ledger whose whole point is that nothing vanishes.
                charge("idle_other_s", self.width.workers - len(running))
            if failure is not None:
                raise failure
        finally:
            self.ledger.wall_s = round(time.monotonic() - pool_t0, 3)
            self.ledger.busy_s = round(sum(self.entry_seconds.values()), 3)
            self.ledger.capacity_s = round(
                self.ledger.wall_s * self.width.workers, 3)
            for row in running:
                _terminate_group(row.proc)
            self._sweep()
            self._emit_ledger()
        return {name: done[name] for name in sorted(done)}

    def _seed_seal_memo(self) -> None:
        """pgw#832: write the parent's toolchain digests where every entry
        child can verify-and-reuse them instead of re-hashing ~4 GB apiece.

        Near-free in a sealed parent (warm cache -> stats only); pays the
        full pass exactly once in a cold one — instead of once per ENTRY.
        Failure is not a mint problem (children fall back to the full rehash,
        which is the safe path), but a systematically unusable snapshot is a
        cost regression somebody must see, so it emits the typed event the
        hub already ingests rather than a bare log."""
        t0 = time.monotonic()
        try:
            count = env_seal.write_library_memo(Path(self.seal_memo))
        except Exception as exc:  # noqa: BLE001 — the fallback path is safe
            detail = f"{type(exc).__name__}: {exc}"
            self.seal_memo = ""
            logger.warning(
                "aot-pool: pgw#832 seal memo seeding failed (%s) — every "
                "entry child will re-pay the full toolchain hash", detail)
            try:
                from . import activity as activity_mod

                activity_mod.emit_event(
                    activity_mod.KIND_AOT_MINT,
                    "seal library memo seeding failed (pgw#832): "
                    f"{detail} — entry children fall back to a full "
                    "per-child toolchain rehash (correct, but re-pays a "
                    "multi-GB SHA-256 pass once per entry)",
                    phase="pool",
                )
            except Exception:  # pragma: no cover — telemetry never fails a mint
                logger.debug("aot-pool: seed-failure event failed", exc_info=True)
            return
        self.seal_seed_s = round(time.monotonic() - t0, 3)
        self.ledger.seal_seed_s = self.seal_seed_s
        logger.info(
            "aot-pool: pgw#832 seal memo seeded (%d lib(s), %.2fs) at %s",
            count, self.seal_seed_s, self.seal_memo)

    def _emit_ledger(self) -> None:
        """The pool's own typed event. Separate from the mint's roll-up on
        purpose: pool idle is not a mint phase, and a reader who groups them
        together would price a scheduling loss as compile work."""
        try:
            from . import activity as activity_mod

            facts = self.ledger.facts()
            activity_mod.emit_event(
                activity_mod.KIND_AOT_MINT,
                f"pool ledger (pgw#830) {facts}",
                phase="pool",
                duration_ms=int(round(self.ledger.wall_s * 1000)),
            )
            logger.info("aot-pool: pgw#830 ledger %s", facts)
        except Exception:  # pragma: no cover — telemetry never fails a mint
            logger.debug("aot-pool: ledger emission failed", exc_info=True)

    def _reap(self, running: Sequence[_Running]) -> Optional[_Running]:
        # Observed concurrency, not intended: the ONLY load-independent
        # evidence that the pool actually overlapped rather than looping.
        self.peak_concurrency = max(self.peak_concurrency, len(running))
        for row in running:
            # Sample while it is alive: /proc vanishes at exit, and VmHWM is
            # the only free high-water mark the kernel keeps. Per ROW as well
            # as pool-wide (pgw#848) — a child the OOM killer takes writes no
            # report, so this sample is the only measurement of it that ever
            # exists, and it is what sizes the next attempt's K.
            row.peak_rss_bytes = max(
                row.peak_rss_bytes, _peak_rss_bytes(row.proc))
            self.peak_rss_bytes = max(self.peak_rss_bytes, row.peak_rss_bytes)
            if row.proc.poll() is not None:
                return row
        return None

    def _collect(self, row: _Running) -> List[str]:
        elapsed = time.monotonic() - row.started
        reap_epoch = time.time()
        self.entry_seconds[row.entry] = round(elapsed, 2)
        code = row.proc.returncode
        report = _read_report(Path(row.job.report))
        # The program is the biggest thing on disk and is dead the moment the
        # child exits; drop it before the next stage runs.
        with_suppress_unlink(row.program_path)
        if report is not None:
            self._verify_child_code(row, report)
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
            self.entry_phases[row.entry] = self._close_entry_partition(
                row, report, elapsed=elapsed, reap_epoch=reap_epoch)
            self.entry_overlays[row.entry] = dict(report.overlays or {})
            self.entry_metrics_raw[row.entry] = dict(report.metrics_raw or {})
            logger.info(
                "aot-pool: entry %r compiled in %.1fs (%d file(s)) spans=%s",
                row.entry, elapsed, len(report.files),
                self.entry_phases[row.entry])
            return list(report.files)
        detail = report.detail if report is not None else ""
        if not detail:
            detail = _stderr_tail(row.stderr_path)
        resource, basis = self._memory_verdict(code, report)
        if resource:
            self.oom_entry, self.oom_basis = row.entry, basis
        raise EntryCompileFailed(
            row.entry,
            f"entry {row.entry!r}: compile child exited {code} after "
            f"{elapsed:.0f}s ({_exit_note(code)}): {detail or 'no detail'}"
            + (
                f" [pgw#848 classification: MEMORY SHORTFALL, basis={basis}; "
                f"this entry's measured high-water was "
                f"{row.peak_rss_bytes / 1024**3:.2f} GiB and the pool ran "
                f"K={self.width.workers} against a "
                f"{self.width.per_entry_rss_bytes / 1024**3:.2f} GiB/entry "
                f"({self.width.per_entry_rss_basis}) ask — this is retryable "
                f"at a narrower K, NOT a deterministic refusal]"
                if resource else ""),
            resource=resource, basis=basis,
            peak_rss_bytes=row.peak_rss_bytes)

    def _memory_verdict(
        self, code: Optional[int], report: Optional[EntryReport],
    ) -> Tuple[bool, str]:
        """Did this entry die of MEMORY, and on what evidence?

        pgw#848. Two bases, and they are not equivalent:

        * ``cgroup`` — the kernel's own ``memory.events`` ``oom_kill`` counter
          moved while this pool ran. That is a fact, not an inference.
        * ``sigkill`` — the child was SIGKILLed and wrote no report, with no
          usable counter to corroborate it. The pool has documented this
          shape as "the OOM killer far more often than a compiler bug" since
          pgw#809; one retry at a narrower K is the right response to a
          "far more often", and a wrong guess costs one retry rather than a
          permanently unmintable cell.

        A child that wrote a report classified ITSELF and is believed: a
        named refusal is deterministic no matter how it exited.
        """
        if report is not None:
            return False, ""
        if code is None or code >= 0 or -int(code) != int(signal.SIGKILL):
            return False, ""
        now = cgroup_oom_kills()
        if now >= 0 and self.oom_kills_at_start >= 0 \
                and now > self.oom_kills_at_start:
            return True, "cgroup"
        return True, "sigkill"

    def _verify_child_code(self, row: _Running, report: EntryReport) -> None:
        """pgw#840: the child that compiled this entry must BE the parent.

        Not a telemetry check. The child produces the loose files
        ``package_aoti`` packs and the cell publishes, while every gate runs in
        the parent against the parent's program — an assignment that is only
        sound while both are the same code. A skewed child was invisible: it
        compiled successfully, returned files that exist, and differed only in
        what it reported. MEASURED on this box: of 236 preserved entry reports,
        150 were written by a child that predates pgw#830's span table, several
        of them under a parent that had it (the pool workdir holds the pgw#832
        seal memo only a post-pgw#832 parent writes). That is the whole of
        pgw#840: the invariant went red on a tree nobody had changed, because
        the child was not from that tree.

        Refused, not warned: an artifact compiled by unknown code must not be
        packed into a cell whose identity claims the parent's.
        """
        if not CODE_DIGEST:
            return  # no source to compare (zipimport) — cannot prove either way
        if report.code_digest == CODE_DIGEST:
            return
        raise EntryCompileFailed(
            row.entry,
            f"entry {row.entry!r}: the compile child ran a DIFFERENT "
            f"gen_worker than this parent — child code "
            f"{report.code_digest or '<too old to report one>'} from "
            f"{report.code_dir or '<unknown>'}, parent code {CODE_DIGEST} from "
            f"{PACKAGE_ROOT}. `python -m {ENTRY_CHILD_MODULE}` resolves "
            f"whatever the interpreter's path yields (a second checkout, an "
            f"inherited PYTHONPATH, a stale wheel, or this tree edited between "
            f"the parent's import and this spawn), and that child compiled the "
            f"files this cell would publish while every gate ran against the "
            f"parent's program")

    def _close_entry_partition(
        self, row: _Running, report: EntryReport, *,
        elapsed: float, reap_epoch: float,
    ) -> Dict[str, float]:
        """pgw#830: close the outermost partition — the one that spans the
        process boundary and therefore could never be closed inside the child.

        ``compile_s = child_boot_s + child_wall_s + reap_lag_s``, where
        ``child_boot_s`` is interpreter startup plus this package's import
        (paid once per ENTRY, because the pool's unit of parallelism is a
        process that exits) and ``reap_lag_s`` is the child's exit plus the
        parent's poll granularity.

        A child that predates this instrumentation reports no epochs; rather
        than invent a split, the whole wall lands in the residual and the
        invariant check says so out loud.
        """
        spans = dict(report.spans or {})
        spans["compile_s"] = round(elapsed, 3)
        if report.run_start_epoch and row.spawn_epoch:
            spans["child_boot_s"] = round(
                report.run_start_epoch - row.spawn_epoch, 3)
        if report.module_import_epoch and row.spawn_epoch:
            # Split of `child_boot_s`, reported as an overlay-style detail:
            # interpreter exec + gen_worker import, before any of the child's
            # own code runs. It is what a persistent worker would delete.
            spans["child_interp_s"] = round(
                report.module_import_epoch - row.spawn_epoch, 3)
        if report.report_epoch:
            spans["reap_lag_s"] = round(reap_epoch - report.report_epoch, 3)
        if "child_boot_s" not in spans or "reap_lag_s" not in spans:
            spans["child_boot_s"] = spans.get("child_boot_s", 0.0)
            spans["reap_lag_s"] = round(
                spans["compile_s"] - spans["child_boot_s"]
                - float(spans.get("child_wall_s", 0.0)), 3)
        violations = aot_compile_spans.check(spans)
        if violations:
            # Named, loud, and non-fatal: an attribution defect must never
            # fail a mint, and must never be silent either (pgw#824's class).
            logger.warning(
                "aot-pool: pgw#830 attribution defect on entry %r: %s",
                row.entry, "; ".join(violations))
        spans["child_interp_s"] = spans.get("child_interp_s", 0.0)
        # Parent-side work for THIS entry. Prefixed, and listed in
        # `aot_compile_spans.SUBSPANS`, because it is not inside `compile_s`:
        # staging overlaps other children, so summing it into the compile
        # total would invent seconds nobody spent compiling. Its idle FRACTION
        # is `ledger.idle_staging_s`, which is a pool number, not an entry one.
        spans["parent_stage_s"] = self.entry_stage_seconds.get(row.entry, 0.0)
        spans["parent_spawn_s"] = self.entry_spawn_seconds.get(row.entry, 0.0)
        return spans

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
    "CODE_DIGEST",
    "COMPILED",
    "CPUS_PER_ENTRY_WORKER",
    "CRASHED",
    "DEFAULT_ENTRY_PEAK_RSS_BYTES",
    "ENTRY_CHILD_MODULE",
    "ENTRY_REPORT_NAME",
    "ENTRY_RSS_RESERVE_BYTES",
    "FORGE_RSS_RESERVE_BYTES",
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
    "PACKAGE_ROOT",
    "REFUSED",
    "SERVING_HEADROOM_CPUS",
    "CpuFacts",
    "DeviceFacts",
    "MemoryFacts",
    "PoolLedger",
    "PoolWidth",
    "available_memory_bytes",
    "child_argv",
    "child_env",
    "cpu_facts",
    "device_facts",
    "entry_workers",
    "free_device_bytes",
    "free_disk_bytes",
    "memory_facts",
]
